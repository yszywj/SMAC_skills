"""Four-group paired test of the DEFAULT-OFF hydra switches on the frozen
contact_response snapshot (micro8-hydra execution for every group).

Groups (identical code; only hydra_switches.json flags differ):
  baseline        A off, B off
  escape_only     A on,  B off
  shot_guard_only A off, B on
  both            A on,  B on

150 paired scenarios from the frozen manifests, 30 shards x 4 labels = 120
jobs, at most 60 concurrent. No seeds are resampled and the frozen source,
historical runs and workspace default policy are never modified.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

ADAPTER = Path(__file__).resolve().parents[1]
ROOT = ADAPTER.parent
sys.path.insert(0, str(ADAPTER))
from bridge.scenarios import validate_manifest  # noqa: E402

from compare_protoss_parallel import write_json  # noqa: E402
from hydra_control_switches import apply_variant  # noqa: E402

PREVIOUS = ROOT / 'runs/contact_response_micro_20261121_153552_909880'
SOURCE = PREVIOUS / 'source_snapshot'
MANIFESTS = PREVIOUS / 'manifests'

GROUPS = (
    ('baseline', False, False),
    ('escape_only', True, False),
    ('shot_guard_only', False, True),
    ('both', True, True),
)
MICRO_MODE = 'micro8-hydra'


def base_env():
    env = dict(os.environ)
    env.pop('PYTHONPATH', None)
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        env[name] = '1'
    return env


def shards():
    manifests = sorted(MANIFESTS.glob('zerg_*.json'))
    if len(manifests) != 30:
        raise ValueError(f'Expected 30 frozen manifest shards, found {len(manifests)}')
    return manifests


def prepare(batch, env):
    """Copy the frozen snapshot once per group, inject the switches, and
    verify offline (imports + flag readback) before any SC2 launch."""
    jobs = []
    reference = json.loads((PREVIOUS / 'design.json').read_text())
    for name, checksum in reference['snapshot_files'].items():
        if hashlib.sha256((SOURCE / name).read_bytes()).hexdigest() != checksum:
            raise ValueError(f'Changed reference source: {name}')
    manifest_dir = batch / 'manifests'
    manifest_dir.mkdir()
    manifests, hashes, policy_hashes = [], {}, {}
    for original in shards():
        path = manifest_dir / original.name
        shutil.copyfile(original, path)
        data = json.loads(path.read_text())
        validate_manifest(data)
        if len(data['episodes']) != 5 or data['sha256'] != reference['shards'][path.stem]['manifest_sha256']:
            raise ValueError(f'Changed reference manifest: {path}')
        manifests.append(path)
        hashes[path.stem] = data['sha256']
    for label, escape, shot_guard in GROUPS:
        work = batch / label
        for item in ('sc2_skill_adapter', 'skills', 'smacv2'):
            shutil.copytree(SOURCE / item, work / item,
                            ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        apply_variant(work, escape=escape, shot_guard=shot_guard)
        policy_hashes[label] = {p.relative_to(work).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in work.rglob('*') if p.is_file()}
        if label != 'baseline' and {k for k in policy_hashes[label].keys() | policy_hashes['baseline'].keys()
                                   if policy_hashes[label].get(k) != policy_hashes['baseline'].get(k)} != {'skills/tactical_v1/hydra_switches.json'}:
            raise ValueError(f'{label}: unexpected source differences')
        subprocess.run([sys.executable, '-B', '-c',
                        'import json, sys; '
                        'from pathlib import Path; '
                        'from bridge.environment import ReplayableEnv; '
                        'from bridge.micro_trigger import MicroAttackWindow; '
                        'from bridge.runner import parser; '
                        'from bridge.skill_runtime import load_manager; '
                        'manager = load_manager("tactical_v1")(tactical_config=None); '
                        'assert hasattr(manager, "last_micro_intents"); '
                        'flags = json.loads(Path("../skills/tactical_v1/hydra_switches.json").read_text()); '
                        'sys.exit(0 if (flags["isolated_escape"], flags["protect_next_shot"]) '
                        '== (%d, %d) else 3)' % (escape, shot_guard)],
                       cwd=work / 'sc2_skill_adapter', env=env, check=True)
    for path in manifests:
        for label, _, _ in GROUPS:
            jobs.append((label, path.stem, path))
    if len(jobs) != 120:
        raise ValueError(f'Expected 120 jobs, built {len(jobs)}')
    write_json(batch / 'design.json', dict(
        source=str(SOURCE), manifests='manifests', labels=[g[0] for g in GROUPS], policy_hashes=policy_hashes,
        switch_flags={label: {'isolated_escape': escape, 'protect_next_shot': shot}
                      for label, escape, shot in GROUPS},
        execution_mode=MICRO_MODE, shards=hashes))
    return jobs


def shard_complete(work, stem, path):
    """A shard counts only when every episode ran and matched the manifest."""
    out = work / stem
    try:
        records = json.loads((out / 'episodes.json').read_text(encoding='utf-8'))
        summary = json.loads((out / 'summary.json').read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None, 'missing outputs'
    expected = json.loads(path.read_text(encoding='utf-8'))['episodes']
    if summary.get('status') != 'complete':
        return None, f"summary status {summary.get('status')!r}"
    if len(records) != len(expected) or \
            [r['scenario_sha256'] for r in records] != [r['sha256'] for r in expected]:
        return None, 'scenario mismatch'
    if summary.get('execution_mode') != MICRO_MODE or summary.get('manifest_sha256') != json.loads(path.read_text())['sha256']:
        return None, 'wrong execution mode or manifest'
    for idx, record in enumerate(records, 1):
        if (record.get('micro_trigger') or {}).get('mode') != MICRO_MODE:
            return None, 'missing micro_trigger accounting'
        if not (out/'damage_by_type'/f'episode-{idx:04d}.json').is_file() or not (out/'replays'/f'episode-{idx:04d}.SC2Replay').is_file():
            return None, 'missing engine damage or replay'
    return records, None


def play(batch, job, env):
    label, stem, path = job
    work = batch / label
    out = work / stem
    with out.with_suffix('.log').open('w') as log:
        subprocess.run([sys.executable, '-B', '-u', 'sc2_skill_adapter/run.py', 'run',
                        '--method', 'adaptive-skill', '--skill-version', 'tactical_v1',
                        '--execution-mode', MICRO_MODE,
                        '--manifest', str(path), '--output-dir', str(out)],
                       cwd=work, env=dict(env, SMAC_DAMAGE_MAP_DIR=str(work/'sc2_skill_adapter/damage_maps')), stdout=log, stderr=subprocess.STDOUT,
                       check=True)
    records, error = shard_complete(work, stem, path)
    if error is not None:
        raise RuntimeError(f'{label}/{stem}: {error}')
    return label, stem, records


def collect(batch, jobs):
    """Re-read every shard; unfinished shards are reported, never losses."""
    results, failures = {}, {}
    for label, stem, path in jobs:
        records, error = shard_complete(batch / label, stem, path)
        if error is None:
            results[(label, stem)] = records
        else:
            failures[f'{label}/{stem}'] = error
    return results, failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=60)
    parser.add_argument('--dry-run', action='store_true',
                        help='copy, patch and import-check all four groups; no SC2')
    parser.add_argument('--retry', type=Path, default=None,
                        help='existing batch directory: rerun only failed shards')
    parser.add_argument('--report-only', action='store_true',
                        help='rebuild comparison.json from an existing batch')
    args = parser.parse_args()
    if not 1 <= args.workers <= 60:
        parser.error('--workers must be in 1..60')
    if args.report_only and not args.retry or args.retry and args.dry_run:
        parser.error('--report-only requires --retry; --dry-run cannot be used with --retry')
    env = base_env()

    if args.retry is not None:
        batch = args.retry.resolve()
        design = json.loads((batch / 'design.json').read_text(encoding='utf-8'))
        if design['labels'] != [g[0] for g in GROUPS] or design.get('manifests') != 'manifests' or not design.get('policy_hashes'):
            raise ValueError('Not a verified switch batch made by this runner')
        jobs = []
        for label in design['labels']:
            for name, checksum in design['policy_hashes'][label].items():
                if hashlib.sha256((batch/label/name).read_bytes()).hexdigest() != checksum:
                    raise ValueError(f'Batch snapshot changed: {label}/{name}')
            for stem, checksum in design['shards'].items():
                path = batch/'manifests'/f'{stem}.json'
                data = json.loads(path.read_text())
                validate_manifest(data)
                if data['sha256'] != checksum:
                    raise ValueError(f'Batch manifest changed: {stem}')
                jobs.append((label, stem, path))
        pending = []
        for label, stem, path in jobs:
            _, error = shard_complete(batch / label, stem, path)
            if error is not None:
                pending.append((label, stem, path))
        print(f'Retry: {len(pending)} of {len(jobs)} shards need rerunning', flush=True)
        broken = batch / 'failed'
        for label, stem, path in ([] if args.report_only else pending):
            stale = batch / label / stem
            keep = broken / label
            keep.mkdir(parents=True, exist_ok=True)
            marker = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            for artifact in (stale, stale.with_suffix('.log')):
                if artifact.exists():
                    shutil.move(str(artifact), str(keep / f'{artifact.name}_{marker}'))
        failures = {}
        if pending and not args.report_only:
            errors = []
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                tasks = {pool.submit(play, batch, job, env): job for job in pending}
                for i, future in enumerate(as_completed(tasks), 1):
                    label, stem, _ = tasks[future]
                    try:
                        future.result()
                        print(f'[{i}/{len(pending)}] {label}/{stem} repaired', flush=True)
                    except Exception as exc:
                        errors.append(str(exc))
                        print(f'FAILED: {batch / label / (stem + ".log")}: {exc}',
                              flush=True)
            if errors:
                write_json(batch / 'failures.json', dict(
                    updated=datetime.now().isoformat(), errors=errors))
                raise SystemExit(f'{len(errors)} shards still failing; '
                                 'rerun --retry to try again')
        results, missing = collect(batch, jobs)
        if missing:
            write_json(batch / 'failures.json', dict(
                updated=datetime.now().isoformat(), missing_shards=missing))
        else:
            (batch / 'failures.json').unlink(missing_ok=True)
        from hydra_switch_report import build_report
        report = build_report(batch, design, results, missing)
        print(f'Report: {batch / "comparison.json"}; '
              f'{len(missing)} shards still missing', flush=True)
        return

    if args.dry_run:
        with TemporaryDirectory() as tmp:
            jobs = prepare(Path(tmp), env)
            print(f'Dry run passed: {len(jobs)} jobs (30 shards x 4 groups), '
                  'switch code imported and flags verified in all four copies; '
                  'no SC2 launched.')
        return

    prefix = 'hydra_switches_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    batch = (ROOT / 'runs' / prefix).resolve()
    batch.mkdir(parents=True)
    print(f'Output: {batch}', flush=True)
    jobs = prepare(batch, env)
    errors = []
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        tasks = {pool.submit(play, batch, job, env): job for job in jobs}
        for future in as_completed(tasks):
            label, stem, _ = tasks[future]
            try:
                future.result()
                done += 1
                print(f'[{done}/{len(jobs)}] {label}/{stem} complete', flush=True)
            except Exception as exc:
                errors.append(f'{label}/{stem}: {exc}')
                print(f'FAILED: {batch / label / (stem + ".log")}: {exc}', flush=True)
    if errors:
        write_json(batch / 'failures.json', dict(
            updated=datetime.now().isoformat(), errors=errors))
    results, missing = collect(batch, jobs)
    from hydra_switch_report import build_report
    build_report(batch, json.loads((batch / 'design.json').read_text(
        encoding='utf-8')), results, missing)
    if errors:
        raise SystemExit(f'{len(errors)} failed shards; rerun with '
                         f'--retry {batch} to rerun only those')
    if missing:
        write_json(batch / 'failures.json', dict(
            updated=datetime.now().isoformat(), missing_shards=missing))
    print(f'Report: {batch / "comparison.json"}', flush=True)


if __name__ == '__main__':
    main()
