"""Two independent contact_response ablations; 60 historical + 300 new runs."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

from compare_protoss_parallel import ADAPTER, ROOT, SMAC_ROOT, digest, generate_manifest, play, validate_manifest, write_json
from contact_ab_report import summarize
from contact_micro_report import MICRO_LABELS, micro_report
from contact_formation_variant import apply_variant as formation
from contact_target_variant import apply_variant as target

PREVIOUS = ROOT / 'runs/contact_540_20261120_225957_302289'
SOURCE = PREVIOUS / 'contact_response'


def prepare_early(batch, seed):
    from contact_early_variant import apply_variant
    jobs = []
    (batch / 'manifests').mkdir()
    for group, lings in enumerate((3, 6)):
        name = f'zerg_{lings}zergling_0baneling_{10-lings}hydralisk'
        template = json.loads((PREVIOUS / 'manifests' / f'{name}_0.json').read_text())
        validate_manifest(template)
        team = template['episodes'][0]['episode_config']['team_gen']['ally_team']
        m = generate_manifest(template['env_args'], 75, seed + group*75)
        for row in m['episodes']:
            row['episode_config']['team_gen'].update(ally_team=team.copy(), enemy_team=team.copy())
            row['sha256'] = digest(row['episode_config'])
        for offset in range(0, 75, 5):
            part = {k: v for k, v in m.items() if k != 'sha256'}
            part['episodes'] = m['episodes'][offset:offset+5]
            part['sha256'] = digest(part)
            validate_manifest(part)
            path = batch / 'manifests' / f'{name}_{offset:02d}.json'
            write_json(path, part)
            jobs.extend((label, name, path) for label in ('baseline', 'early'))
    return jobs, [('baseline', None), ('early', apply_variant)]


def prepare_front(batch, seed, live=False):
    from contact_front_report import QUOTA, _layout
    if live:
        from contact_live_support_variant import apply_variant
        if live == 'screen':
            from functools import partial
            apply_variant = partial(apply_variant, screen_fire=True)
    else:
        from contact_front_variant import apply_variant
    change = 'support' if live else 'route'
    jobs = []
    (batch / 'manifests').mkdir()
    for group, lings in enumerate((3, 6)):
        name = f'zerg_{lings}zergling_0baneling_{10-lings}hydralisk'
        template = json.loads((PREVIOUS / 'manifests' / f'{name}_0.json').read_text())
        validate_manifest(template)
        team = template['episodes'][0]['episode_config']['team_gen']['ally_team']
        remaining = dict(QUOTA)
        for attempt in range(500):
            m = generate_manifest(template['env_args'], 1, seed + 2*attempt + group)
            row, = m['episodes']
            cfg = row['episode_config']
            layout = _layout(cfg, name)
            if remaining[layout] == 0:
                continue
            cfg['team_gen'].update(ally_team=team.copy(), enemy_team=team.copy())
            row['sha256'] = digest(cfg)
            del m['sha256']
            m['sha256'] = digest(m)
            validate_manifest(m)
            path = batch / 'manifests' / f'{name}_{layout}_{row["sc2_seed"]}.json'
            write_json(path, m)
            jobs.extend((label, name, path) for label in ('baseline', change))
            remaining[layout] -= 1
            if not any(remaining.values()):
                break
        else:
            raise ValueError(f'Could not fill spawn quotas for {name}: {remaining}')
    return jobs, [('baseline', None), (change, apply_variant)]


def prepare(batch, seed, env, early=False, front=False, live=False):
    if front or live:
        jobs, versions = prepare_front(batch, seed, live)
    elif early:
        jobs, versions = prepare_early(batch, seed)
    else:
        jobs = prepare_ab(batch, seed)
        versions = [('baseline', None), ('target', target), ('formation', formation)]
    for label, patch in versions:
        work = batch / label
        for folder in ('skills', 'sc2_skill_adapter'):
            shutil.copytree(SOURCE / folder, work / folder, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        (work / 'smacv2').symlink_to(SMAC_ROOT.resolve(), target_is_directory=True)
        if patch:
            patch(work / 'skills/tactical_v1')
        subprocess.run([sys.executable, '-B', '-c', 'from bridge.environment import ReplayableEnv; '
                        'from bridge.skill_runtime import load_manager; load_manager("tactical_v1")(tactical_config=None)'],
                       cwd=work / 'sc2_skill_adapter', env=env, check=True)
    if len(jobs) != (60 if early or front or live else 150):
        raise ValueError('Incorrect number of jobs')
    write_json(batch / 'design.json', dict(seed=seed, source=str(SOURCE), groups=sorted({name for _,name,_ in jobs}), new_games=60 if front or live else 300))
    return jobs


def prepare_ab(batch, seed):
    jobs, seen = [], set()
    (batch / 'manifests').mkdir()
    def split(kind, source, name, m):
        if len(m['episodes']) != 10:
            raise ValueError(source)
        for offset in range(0, 10, 2):
            part = {k: v for k, v in m.items() if k != 'sha256'}
            part['episodes'] = m['episodes'][offset:offset+2]
            part['sha256'] = digest(part)
            validate_manifest(part)
            path = batch / 'manifests' / f'{kind}__{source}__{offset}.json'
            write_json(path, part)
            labels = ('target', 'formation') if kind == 'old' else ('baseline', 'target', 'formation')
            jobs.extend((label, name, path) for label in labels)

    old = sorted((PREVIOUS / 'manifests').glob('zerg_*.json'))
    if len(old) != 6:
        raise ValueError('Expected six historical ten-game shards')
    for path in old:
        m = json.loads(path.read_text())
        validate_manifest(m)
        if m != json.loads((SOURCE / path.stem / 'manifest.json').read_text()):
            raise ValueError(f'Baseline manifest mismatch: {path}')
        name = path.stem.rsplit('_', 1)[0]
        seen.add(name)
        split('old', path.stem, name, m)

    config = m['env_args']
    rng = random.Random(seed)
    dist = config['capability_config']['team_gen']
    for i in range(6):
        while True:
            team = rng.choices(dist['unit_types'], weights=dist['weights'], k=10)
            name = 'zerg_' + '_'.join(f'{team.count(u)}{u}' for u in dist['unit_types'])
            if name not in seen and not all(u in dist.get('exception_unit_types', ()) for u in team):
                break
        seen.add(name)
        m = generate_manifest(config, 10, seed + 10*i)
        for row in m['episodes']:
            row['episode_config']['team_gen'].update(ally_team=team.copy(), enemy_team=team.copy())
            row['sha256'] = digest(row['episode_config'])
        split('new', name, name, m)

    return jobs


def micro_args(label, path, out):
    return ['run', '--method', 'adaptive-skill', '--skill-version', 'tactical_v1',
            '--execution-mode', label, '--manifest', str(path), '--output-dir', str(out)]


def micro_env(env, work):
    child = dict(env, SMAC_DAMAGE_MAP_DIR=str(work / 'sc2_skill_adapter/damage_maps'))
    child.pop('PYTHONPATH', None)  # never resolve a policy from another worktree
    return child


def prepare_micro(batch, seed, env, contact_response=False):
    """Freeze one policy and run the two execution modes on 150 matched pairs."""
    jobs, shards = [], {}
    (batch / 'manifests').mkdir()
    for group, lings in enumerate((3, 6)):
        name = f'zerg_{lings}zergling_0baneling_{10-lings}hydralisk'
        template = json.loads((PREVIOUS / 'manifests' / f'{name}_0.json').read_text())
        validate_manifest(template)
        team = template['episodes'][0]['episode_config']['team_gen']['ally_team']
        config = template['env_args']
        if (team.count('zergling') != lings or team.count('hydralisk') != 10 - lings
                or len(team) != 10 or config['step_mul'] != 8 or config['map_name'] != '10gen_zerg'
                or config['unit_balance'] != 'native'):
            raise ValueError(f'Invalid micro scenario template: {name}')
        m = generate_manifest(template['env_args'], 75, seed + group*75)
        for row in m['episodes']:
            row['episode_config']['team_gen'].update(ally_team=team.copy(), enemy_team=team.copy())
            row['sha256'] = digest(row['episode_config'])
        for offset in range(0, 75, 5):
            part = {k: v for k, v in m.items() if k != 'sha256'}
            part['episodes'] = m['episodes'][offset:offset+5]
            part['sha256'] = digest(part)
            validate_manifest(part)
            path = batch / 'manifests' / f'{name}_{offset:02d}.json'
            write_json(path, part)
            shards[path.stem] = {'composition': name, 'manifest_sha256': part['sha256']}
            jobs.extend((label, name, path) for label in MICRO_LABELS)
    snapshot = batch / 'source_snapshot'
    ignore = shutil.ignore_patterns('__pycache__', 'tests', '*.pyc')
    source = SOURCE if contact_response else ROOT
    for rel in ('skills/current', 'skills/tactical_v1'):
        shutil.copytree(source / rel, snapshot / rel, ignore=ignore)
    if contact_response:
        from contact_response_micro_variant import apply_variant
        apply_variant(snapshot / 'skills/tactical_v1')
    for rel in ('sc2_skill_adapter/bridge', 'sc2_skill_adapter/display_guard'):
        shutil.copytree(ROOT / rel, snapshot / rel, ignore=ignore)
    shutil.copytree(SMAC_ROOT / 'smacv2', snapshot / 'smacv2/smacv2', ignore=ignore)
    for rel in ('run.py', 'scripts/make_damage_map.py', 'damage_maps/10gen_zerg.SC2Map'):
        dest = snapshot / 'sc2_skill_adapter' / rel
        dest.parent.mkdir(exist_ok=True, parents=True)
        shutil.copy2(ADAPTER / rel, dest)
    files = {p.relative_to(snapshot).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(snapshot.rglob('*')) if p.is_file()}
    for label in MICRO_LABELS:
        work = batch / label
        shutil.copytree(snapshot, work)
        adapter = work / 'sc2_skill_adapter'
        subprocess.run([sys.executable, '-B', '-c',
                        'import sys; '
                        'from bridge.environment import ReplayableEnv; '
                        'from bridge.micro_trigger import MicroAttackWindow; '
                        'from bridge.runner import parser; '
                        'from bridge.skill_runtime import load_manager, manager_configuration; '
                        'args = parser().parse_args(sys.argv[1:]); '
                        'manager = load_manager(args.skill_version)(**manager_configuration(args)["manager_kwargs"]); '
                        'assert hasattr(manager, "last_micro_intents")',
                        *micro_args(label, jobs[0][2], work / jobs[0][2].stem)],
                       cwd=adapter, env=micro_env(env, work), check=True)
    if len(jobs) != 60:
        raise ValueError('Incorrect number of jobs')
    write_json(batch / 'design.json', dict(
        seed=seed, source='source_snapshot', policy_base=str(source / 'skills/tactical_v1'), snapshot_files=files, snapshot_sha256=digest(files), shards=shards,
        labels=list(MICRO_LABELS), groups=sorted({name for _, name, _ in jobs}),
        paired_scenarios=150, child_execution_flags={label: ['--execution-mode', label]
                                                     for label in MICRO_LABELS}))
    return jobs


def play_micro(batch, job, env):
    """One shard of one group; the label IS the child's --execution-mode."""
    label, name, path = job
    work = batch / label
    out = work / path.stem
    with out.with_suffix('.log').open('w') as log:
        subprocess.run([sys.executable, '-B', '-u', 'sc2_skill_adapter/run.py',
                        *micro_args(label, path, out)], cwd=work,
                       env=micro_env(env, work), stdout=log, stderr=subprocess.STDOUT, check=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed', type=int, default=20261120)
    p.add_argument('--workers', type=int, default=60)
    p.add_argument('--output-root', type=Path, default=ROOT / 'runs')
    p.add_argument('--dry-run', action='store_true')
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--early-contact', action='store_true', help='run 150 matched pure ling/hydra pairs')
    mode.add_argument('--front-route', action='store_true', help='test opening ling routes on 20 scattered and 10 center pairs')
    mode.add_argument('--live-support', action='store_true', help='test hydras joining observed ling engagements on 30 pairs')
    mode.add_argument('--screen-fire', action='store_const', dest='live_support', const='screen', help='test ready hydras focusing dogs close to the back line')
    mode.add_argument('--micro-8', '--micro-trigger', action='store_true', dest='micro_trigger',
                      help='paired micro8-off vs micro8-hydra execution modes on 150 matched pairs (60 shards)')
    mode.add_argument('--contact-response-micro', dest='micro_trigger', action='store_const', const='contact_response',
                      help='same micro-trigger comparison with the frozen contact_response policy')
    args = p.parse_args()
    if not 1 <= args.workers <= 60:
        p.error('--workers must be in 1..60')
    env = dict(os.environ, SMAC_DAMAGE_MAP_DIR=str(ADAPTER / 'damage_maps'), OPENBLAS_NUM_THREADS='1',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    if args.dry_run:
        with TemporaryDirectory() as tmp:
            if args.micro_trigger:
                jobs = prepare_micro(Path(tmp), args.seed, env, args.micro_trigger == 'contact_response')
                print(f'Validated {len(jobs)} shards, 150 paired scenarios x 2 execution modes, '
                      f'max {args.workers} concurrent. No SC2 launched.')
            else:
                jobs = prepare(Path(tmp), args.seed, env, args.early_contact, args.front_route, args.live_support)
                print(f'Validated {len(jobs)} shards, {60 if args.front_route or args.live_support else 300} new games, max {args.workers} concurrent. No SC2 launched.')
        return
    prefix = 'zerg_early_contact_' if args.early_contact else 'zerg_contact_ab_'
    if args.front_route:
        prefix = 'zerg_front_route_'
    if args.live_support:
        prefix = 'zerg_screen_fire_' if args.live_support == 'screen' else 'zerg_live_support_'
    if args.micro_trigger:
        prefix = 'contact_response_micro_' if args.micro_trigger == 'contact_response' else 'zerg_micro_trigger_'
    batch = args.output_root.resolve() / (prefix + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    batch.mkdir(parents=True)
    print(f'Output: {batch}; seed: {args.seed}', flush=True)
    if args.micro_trigger:
        jobs = prepare_micro(batch, args.seed, env, args.micro_trigger == 'contact_response')
        errors = []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            tasks = {pool.submit(play_micro, batch, job, env): job for job in jobs}
            for i, f in enumerate(as_completed(tasks), 1):
                label, _, path = tasks[f]
                try:
                    f.result()
                    print(f'[{i}/{len(jobs)}] {label}/{path.stem} complete', flush=True)
                except Exception as exc:
                    errors.append(str(exc))
                    print(f'FAILED: {batch / label / (path.stem + ".log")}: {exc}', flush=True)
        if errors:
            raise SystemExit(f'{len(errors)} failed jobs; no complete comparison')
        report = micro_report(batch)
        for label in MICRO_LABELS:
            stats = report['total']['versions'][label]
            print(f"{label}: {stats['outcomes'].get('win', 0)}/{stats['games']} wins; "
                  f"submitted {stats['micro_trigger']['submitted']} of {stats['micro_trigger']['authorizations']} authorizations "
                  f"({stats['micro_trigger']['states']})")
        print(f'Results: {batch / "comparison.json"}')
        return
    jobs = prepare(batch, args.seed, env, args.early_contact, args.front_route, args.live_support)
    errors = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        tasks = {pool.submit(play, batch, job, env): job for job in jobs}
        for i, f in enumerate(as_completed(tasks), 1):
            label, _, path = tasks[f]
            try:
                f.result()
                print(f'[{i}/{len(jobs)}] {label}/{path.stem} complete', flush=True)
            except Exception as exc:
                errors.append(str(exc))
                print(f'FAILED: {batch / label / (path.stem + ".log")}: {exc}', flush=True)
    if errors:
        raise SystemExit(f'{len(errors)} failed jobs; no complete comparison')
    if args.live_support:
        from contact_live_support_report import summarize as support_report
        support_report(batch, jobs)
        print(f'Results: {batch / "comparison.json"}')
        return
    if args.front_route:
        from contact_front_report import summarize as front_report
        front_report(batch, jobs)
        print(f'Results: {batch / "comparison.json"}')
        return
    if args.early_contact:
        from contact_early_report import summarize as early_report
        early_report(batch, jobs)
        print(f'Results: {batch / "comparison.json"}')
        return
    report = summarize(batch, PREVIOUS, jobs)
    for label, stats in report['total']['versions'].items():
        print(f"{label}: {stats['outcomes']['win']}/{stats['games']} wins")


if __name__ == '__main__':
    main()
