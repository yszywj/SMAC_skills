"""Compare two tactical_v1 snapshots on 3 x 20 paired Protoss scenarios."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

ADAPTER = Path(__file__).resolve().parents[1]
MAX_WORKERS = 30
ROOT = ADAPTER.parent
OLD = ROOT / 'runs/tactical_20x2x3_20261117_185934/source_snapshot'
sys.path.insert(0, str(ADAPTER))
from bridge.config import SMAC_ROOT, digest, load_config
from bridge.scenarios import generate_manifest, validate_manifest


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def prepare(batch, seed, env, replay=None, matrix=False, random_compositions=False, contact=False):
    # Use the same adapter and SMAC installation, but isolate both skill trees
    # (tactical_v1 also imports helpers from skills/current).
    ignore = shutil.ignore_patterns('__pycache__', 'tests', '*.pyc')
    sources = ([('nov16', ROOT / 'runs/tactical_20x3x3_20261116_142630/source_snapshot')]
               if replay or matrix or contact else [('old', OLD), ('latest', ROOT)])
    if matrix or contact:
        third = ('contact_response', ROOT / 'runs/tactical_z_6l4h_contact_response_20/source_snapshot') if contact else ('latest', ROOT)
        sources += [('nov17', OLD), third]
    for label, source in sources:
        work = batch / label
        bridge = work / 'sc2_skill_adapter/bridge'
        shutil.copytree(ADAPTER / 'bridge', bridge, ignore=ignore)
        shutil.copy2(ADAPTER / 'run.py', bridge.parent / 'run.py')
        if env.get('SMAC_DAMAGE_MAP_DIR'):
            (bridge.parent / 'scripts').mkdir()
            shutil.copy2(ADAPTER / 'scripts/make_damage_map.py', bridge.parent / 'scripts')
        for pkg in ['current', 'tactical_v1']:
            shutil.copytree(source / 'skills' / pkg, work / 'skills' / pkg, ignore=ignore)
        (work / 'smacv2').symlink_to(SMAC_ROOT.resolve(), target_is_directory=True)
        subprocess.run([sys.executable, '-B', '-c',
                        'from bridge.environment import ReplayableEnv; '
                        'from bridge.runner import parser; from bridge.skill_runtime import load_manager; '
                        'parser(); load_manager("tactical_v1")(tactical_config=None)'],
                       cwd=bridge.parent, env=env, check=True)

    if matrix or contact:
        jobs = []
        (batch / 'manifests').mkdir()
        rng = random.Random(seed)
        for race_index, race in enumerate(('protoss', 'terran', 'zerg')):
            config = load_config(10, race)
            config['unit_balance'] = 'native'
            dist = config['capability_config']['team_gen']
            seen = set()
            groups, episodes = (3, 20) if contact else (6, 10)
            for group in range(groups):
                while True:
                    team = rng.choices(dist['unit_types'], weights=dist['weights'], k=10)
                    key = tuple(sorted(team))
                    if key in seen or all(u in dist.get('exception_unit_types', ()) for u in team):
                        continue
                    if not random_compositions and race == 'protoss' and team.count('colossus') != group // (groups // 3):
                        continue
                    seen.add(key)
                    break
                name = race + '_' + '_'.join(f'{team.count(u)}{u}' for u in dist['unit_types'])
                manifest = generate_manifest(config, episodes, seed + (groups*race_index + group)*episodes)
                for row in manifest['episodes']:
                    episode = row['episode_config']
                    episode['team_gen'].update(ally_team=team.copy(), enemy_team=team.copy())
                    row['sha256'] = digest(episode)
                del manifest['sha256']
                for start in range(0, episodes, 10):
                    part = dict(manifest, episodes=manifest['episodes'][start:start+10])
                    part['sha256'] = digest(part)
                    validate_manifest(part)
                    suffix = f'_{start // 10}' if contact else ''
                    path = batch / 'manifests' / f'{name}{suffix}.json'
                    write_json(path, part)
                    jobs.extend((label, name, path) for label, _ in sources)
        return jobs

    if replay:
        jobs = []
        (batch / 'manifests').mkdir()
        for name in ('c0_z8_s2', 'c1_z5_s4', 'c2_z6_s2'):
            for i in range(5):
                src = replay / 'manifests' / f'{name}_{i}.json'
                manifest = json.loads(src.read_text())
                validate_manifest(manifest)
                rows = manifest['episodes']
                if len(rows) != 4:
                    raise ValueError(f'Expected four saved episodes: {src}')
                for j in range(2):
                    part = {k: v for k, v in manifest.items() if k != 'sha256'}
                    part['episodes'] = rows[j*2:j*2+2]
                    part['sha256'] = digest(part)
                    validate_manifest(part)
                    path = batch / 'manifests' / f'{name}_{i}_{j}.json'
                    write_json(path, part)
                    jobs.append(('nov16', name, path))
        return jobs

    config = load_config(10, 'protoss')
    config['unit_balance'] = 'native'
    dist = config['capability_config']['team_gen']
    weights = dict(zip(dist['unit_types'], dist['weights']))
    rng = random.Random(seed)
    jobs = []
    (batch / 'manifests').mkdir()
    for c in range(3):
        team = ['colossus'] * c + rng.choices(['zealot', 'stalker'],
                weights=[weights['zealot'], weights['stalker']], k=10-c)
        rng.shuffle(team)
        name = f"c{c}_z{team.count('zealot')}_s{team.count('stalker')}"
        print(f'{name}: 20 matched pairs', flush=True)
        manifest = generate_manifest(config, 20, seed + c * 20)
        for row in manifest['episodes']:
            episode = row['episode_config']
            episode['team_gen'].update(ally_team=team.copy(), enemy_team=team.copy())
            row['sha256'] = digest(episode)
        for shard in range(5):
            part = {k: v for k, v in manifest.items() if k != 'sha256'}
            part['episodes'] = manifest['episodes'][4*shard:4*shard+4]
            part['sha256'] = digest(part)
            validate_manifest(part)
            path = batch / 'manifests' / f'{name}_{shard}.json'
            write_json(path, part)
            jobs.extend((label, name, path) for label in ('old', 'latest'))
    return jobs


def play(batch, job, env):
    label, name, path = job
    work = batch / label
    out = work / path.stem
    with out.with_suffix('.log').open('w') as log:
        subprocess.run([sys.executable, '-B', '-u', 'sc2_skill_adapter/run.py', 'run',
                        '--method', 'adaptive-skill', '--skill-version', 'tactical_v1',
                        '--manifest', str(path), '--output-dir', str(out)], cwd=work,
                       env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    records = json.loads((out / 'episodes.json').read_text())
    expected = json.loads(path.read_text())['episodes']
    if [r['scenario_sha256'] for r in records] != [r['sha256'] for r in expected]:
        raise ValueError(f'Incomplete or mismatched scenarios: {out}')
    return label, name, path.stem, records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=20261120)
    parser.add_argument('--output-root', type=Path, default=ROOT / 'runs')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--matrix-540', action='store_true', help='3 races x 6 compositions x 10 games x 3 policies')
    mode.add_argument('--contact-540', action='store_true', help='nov16/nov17/contact_response: 3 races x 3 compositions x 20 games')
    parser.add_argument('--workers', type=int, default=MAX_WORKERS)
    parser.add_argument('--damage-by-type', action='store_true', help='enable instrumented flat maps and record damage by attacking side and unit type')
    parser.add_argument('--random-compositions', action='store_true', help='sample without stratifying by colossus count')
    mode.add_argument('--replay-nov16', type=Path, metavar='PAIRED_RUN',
                        help='run only the Nov 16 snapshot against the saved paired run')
    parser.add_argument('--dry-run', action='store_true', help='validate both policies and manifests without SC2')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    if args.damage_by_type:
        maps = ADAPTER / 'damage_maps'
        for race in ('protoss', 'terran', 'zerg'):
            if not (maps / f'10gen_{race}.SC2Map').is_file():
                parser.error(f'Missing instrumented {race} map in {maps}')
        env['SMAC_DAMAGE_MAP_DIR'] = str(maps)
    if args.dry_run:
        with TemporaryDirectory() as tmp:
            jobs = prepare(Path(tmp), args.seed, env, args.replay_nov16, args.matrix_540, args.random_compositions, args.contact_540)
            print(f'Offline validation passed: {len(jobs)} tasks x {10 if args.matrix_540 or args.contact_540 else 2 if args.replay_nov16 else 4} games; max {args.workers} concurrent; SC2 not launched.')
        return
    prefix = 'protoss_nov16_replay_' if args.replay_nov16 else 'protoss_paired_'
    if args.matrix_540:
        prefix = 'matrix_540_'
    elif args.contact_540:
        prefix = 'contact_540_'
    batch = args.output_root.resolve() / (prefix + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    batch.mkdir(parents=True)
    print(f'Output: {batch}', flush=True)
    jobs = prepare(batch, args.seed, env, args.replay_nov16, args.matrix_540, args.random_compositions, args.contact_540)
    counts, states = {}, {}
    failed = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(play, batch, job, env): job for job in jobs}
        for i, future in enumerate(as_completed(futures), 1):
            label, _, path = futures[future]
            try:
                label, name, shard, rows = future.result()
                counts.setdefault((label, name), Counter()).update(r['outcome'] for r in rows)
                states[label, shard] = [r['initial_state_sha256'] for r in rows]
                print(f'[{i}/{len(jobs)}] {label}/{shard} complete', flush=True)
            except Exception as error:
                failed.append(str(error))
                print(f'[{i}/{len(jobs)}] FAILED: see {batch / label / (path.stem + ".log")}: {error}', flush=True)
    if failed:
        raise SystemExit(f'{len(failed)} failed workers; no final comparison.')
    for _, _, path in jobs:
        if args.replay_nov16:
            original, half = path.stem.rsplit('_', 1)
            ref = json.loads((args.replay_nov16 / 'latest' / original / 'episodes.json').read_text())
            start = int(half)*2
            matched = states['nov16', path.stem] == [r['initial_state_sha256'] for r in ref[start:start+2]]
        elif args.matrix_540 or args.contact_540:
            third = 'contact_response' if args.contact_540 else 'latest'
            matched = states['nov16', path.stem] == states['nov17', path.stem] == states[third, path.stem]
        else:
            matched = states['old', path.stem] == states['latest', path.stem]
        if not matched:
            raise SystemExit(f'Initial states differ for {path.stem}; no paired comparison.')
    report = {}
    for label in dict.fromkeys(label for label, _, _ in jobs):
        groups = {name: n for (version, name), n in counts.items() if version == label}
        totals = [('total', sum(groups.values(), Counter()))]
        if args.matrix_540 or args.contact_540:
            totals = [(r, sum((n for k, n in groups.items() if k.startswith(r + '_')), Counter()))
                      for r in ('protoss', 'terran', 'zerg')] + totals
        for name, n in [*sorted(groups.items()), *totals]:
            total = sum(n.values())
            report.setdefault(label, {})[name] = dict(games=total, wins=n['win'], win_rate=n['win']/total)
            print(f"{label:6} {name:12} {n['win']}/{total} = {n['win']/total:.1%}")
    write_json(batch / 'comparison.json', report)
    if args.damage_by_type:
        totals = {}
        for label, name, path in jobs:
            race = json.loads(path.read_text())['env_args']['map_name'].removeprefix('10gen_')
            group = totals.setdefault(label, {}).setdefault(race, {'games': 0, 'ally': Counter(), 'enemy': Counter()})
            for i, _ in enumerate(states[label, path.stem], 1):
                entry = json.loads((batch / label / path.stem / 'damage_by_type' / f'episode-{i:04d}.json').read_text())
                group['games'] += 1
                for player, side in [('1', 'ally'), ('2', 'enemy')]:
                    group[side].update(entry['players'][player]['damage_dealt_by_source_type'])
        write_json(batch / 'damage_by_type_summary.json', totals)


if __name__ == '__main__':
    main()
