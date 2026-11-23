"""Run four unit-controller ablations against the saved contact_response scenarios."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

from compare_protoss_parallel import ROOT, digest, validate_manifest, write_json
from contact_factorial_report import LABELS, MODE, summarize
from contact_simple_units import apply_simple
from run_contact_zerg_ab import micro_args, micro_env

REFERENCE = ROOT / 'runs/contact_response_micro_20261121_153552_909880'


def prepare(batch, env, muster=False, local_packs=False, all_attack=False):
    if all_attack:
        from contact_all_micro_report import MODE_MAP
        labels = tuple(MODE_MAP)
    else:
        labels = ('baseline', 'packs', 'packs_reserve') if local_packs else ('baseline', 'muster') if muster else LABELS
    design = json.loads((REFERENCE / 'design.json').read_text())
    original = design['snapshot_files']
    if digest(original) != design['snapshot_sha256']:
        raise ValueError('Reference fingerprint mismatch')
    (batch / 'manifests').mkdir()
    jobs, files = [], {}
    for shard, spec in sorted(design['shards'].items()):
        path = batch / 'manifests' / (shard + '.json')
        shutil.copyfile(REFERENCE / 'manifests' / path.name, path)
        manifest = json.loads(path.read_text())
        validate_manifest(manifest)
        if manifest['sha256'] != spec['manifest_sha256'] or len(manifest['episodes']) != 5:
            raise ValueError(f'Invalid reference manifest {path}')
        jobs.extend((label, path) for label in labels)
    if len(jobs) != 30 * len(labels):
        raise ValueError('Expected 30 five-game shards per policy')
    for label in labels:
        work = batch / label
        shutil.copytree(REFERENCE / 'source_snapshot', work, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        def fingerprint():
            return {p.relative_to(work).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in work.rglob('*') if p.is_file()}
        if fingerprint() != original:
            raise ValueError(f'{label}: copy differs from reference')
        if all_attack:
            from contact_all_attackers import apply_variant as install_grants
            from contact_local_packs_variant import apply_variant as install_packs
            # Every arm has the exact same engine and grant implementation. Only mode
            # changes for the first contrast; subsequent arms vary the macro policy.
            for name in ('runner.py', 'micro_trigger.py'):
                rel = Path('sc2_skill_adapter/bridge') / name
                shutil.copyfile(ROOT / rel, work / rel)
            skill = work / 'skills/tactical_v1'
            install_packs(skill, label in ('marginal_far','marginal_near'), marginal=label.startswith('marginal_'),
                          reserve_horizon=2 if label == 'marginal_near' else 4)
            install_grants(skill)
        elif local_packs:
            if label != 'baseline':
                from contact_local_packs_variant import apply_variant
                apply_variant(work / 'skills/tactical_v1', dispatch=(label == 'packs_reserve'))
        elif label == 'muster':
            from contact_muster_variant import apply_variant
            apply_variant(work / 'skills/tactical_v1')
        elif not muster:
            apply_simple(work / 'skills/tactical_v1', label in ('simple_ling', 'simple_both'),
                         label in ('simple_hydra', 'simple_both'))
        files[label] = fingerprint()
        subprocess.run([sys.executable, '-B', '-c', 'from bridge.skill_runtime import load_manager; '
                        'm=load_manager("tactical_v1")(); assert hasattr(m,"last_micro_intents")'],
                       cwd=work / 'sc2_skill_adapter', env=micro_env(env, work), check=True)
    if all_attack and files['packs_hydra'] != files['packs_all']:
        raise ValueError('Micro comparison requires byte-identical policies and runtime')
    write_json(batch / 'design.json', dict(seed=design['seed'], reference=str(REFERENCE), labels=labels,
               **({'mode_map': MODE_MAP} if all_attack else {'execution_mode': MODE}), shards=design['shards'], policy_files=files))
    return jobs


def play(batch, job, env, mode=MODE):
    label, path = job
    work = batch / label
    out = work / path.stem
    with out.with_suffix('.log').open('w') as log:
        subprocess.run([sys.executable, '-B', '-u', 'sc2_skill_adapter/run.py', *micro_args(mode, path, out)],
                       cwd=work, env=micro_env(env, work), stdout=log, stderr=subprocess.STDOUT, check=True)


def retry_local(batch, all_attack=False):
    design = json.loads((batch / 'design.json').read_text())
    original = json.loads((REFERENCE / 'design.json').read_text())
    if all_attack:
        from contact_all_micro_report import LABELS as packs_labels, MODE_MAP, _games
        valid = (design.get('mode_map') == MODE_MAP and design['policy_files']['packs_hydra'] == design['policy_files']['packs_all'])
    else:
        from contact_local_packs_report import LABELS as packs_labels, _games
        valid = design.get('execution_mode') == MODE and design['policy_files'].get('baseline') == original['snapshot_files']
    if tuple(design['labels']) != packs_labels or design['shards'] != original['shards'] or not valid:
        raise ValueError('Expected a frozen local-pack comparison')
    jobs = []
    for label in packs_labels:
        work = batch / label
        files = design['policy_files'][label]
        for name, checksum in files.items():
            path = (work / name).resolve()
            if not path.is_relative_to(work) or hashlib.sha256(path.read_bytes()).hexdigest() != checksum:
                raise ValueError(f'{label}: invalid policy snapshot')
        for stem, spec in design['shards'].items():
            path = batch / 'manifests' / (stem + '.json')
            manifest = json.loads(path.read_text())
            validate_manifest(manifest)
            if manifest['sha256'] != spec['manifest_sha256']:
                raise ValueError(f'Changed manifest: {path}')
            try:
                _games(work / stem, manifest, spec['composition'], label, files, *([MODE_MAP[label]] if all_attack else []))
            except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
                print(f'Retry {label}/{stem}: {exc}', flush=True)
                jobs.append((label, path))
    archive = batch / 'failed' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    for label, path in jobs:
        out = batch / label / path.stem
        for artifact in (out, out.with_suffix('.log')):
            if artifact.exists():
                (archive / label).mkdir(parents=True, exist_ok=True)
                artifact.rename(archive / label / artifact.name)
    return jobs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workers', type=int, default=60)
    p.add_argument('--output-root', type=Path, default=ROOT / 'runs')
    p.add_argument('--dry-run', action='store_true')
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--opening-muster', action='store_true', help='test opening regrouping versus frozen baseline on 150 pairs')
    mode.add_argument('--local-packs', action='store_true', help='test baseline, local engagements, and engagements with reserve dispatch')
    mode.add_argument('--all-attack-near', action='store_true', help='test all combat-unit micro triggers, marginal local allocation, and near-only reserves')
    p.add_argument('--retry', type=Path, metavar='BATCH', help='with --local-packs or --all-attack-near: retry failed shards in a finished batch')
    args = p.parse_args()
    if not 1 <= args.workers <= 60:
        p.error('--workers must be in 1..60')
    if args.retry and (not (args.local_packs or args.all_attack_near) or args.dry_run):
        p.error('--retry requires a local pack experiment and cannot be combined with --dry-run')
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    if args.dry_run:
        with TemporaryDirectory() as tmp:
            jobs = prepare(Path(tmp), env, args.opening_muster, args.local_packs, args.all_attack_near)
        print(f'Validated 150 scenarios x {len(jobs)//30} policies, max {args.workers} concurrent. SC2 not launched.')
        return
    prefix = 'contact_all_attack_' if args.all_attack_near else 'contact_local_packs_' if args.local_packs else 'contact_opening_muster_' if args.opening_muster else 'contact_factorial_'
    batch = args.retry.resolve() if args.retry else args.output_root.resolve() / (prefix + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    if not args.retry:
        batch.mkdir(parents=True)
    print(f'Output: {batch}', flush=True)
    jobs = retry_local(batch, args.all_attack_near) if args.retry else prepare(batch, env, args.opening_muster, args.local_packs, args.all_attack_near)
    modes = json.loads((batch/'design.json').read_text()).get('mode_map', {})
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        tasks = {pool.submit(play, batch, job, env, modes.get(job[0], MODE)): job for job in jobs}
        for i, f in enumerate(as_completed(tasks), 1):
            label, path = tasks[f]
            try:
                f.result()
                print(f'[{i}/{len(jobs)}] {label}/{path.stem} complete', flush=True)
            except Exception as exc:
                failures.append(str(exc))
                print(f'FAILED: {batch / label / (path.stem + ".log")}: {exc}', flush=True)
    if failures:
        raise SystemExit(f'{len(failures)} failed shards; no complete comparison')
    if args.all_attack_near:
        from contact_all_micro_report import summarize as all_report
        report = all_report(batch)
    elif args.local_packs:
        from contact_local_packs_report import summarize as local_report
        report = local_report(batch)
    elif args.opening_muster:
        from contact_muster_report import summarize as muster_report
        report = muster_report(batch)
    else:
        report = summarize(batch)
    for label, stats in report['total']['versions'].items():
        print(f"{label}: {stats['outcomes']['win']}/{stats['games']} wins")
    print(f'Results: {batch / "comparison.json"}')


if __name__ == '__main__':
    main()
