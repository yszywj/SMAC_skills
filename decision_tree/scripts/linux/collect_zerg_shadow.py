"""Collect one marginal_near Zerg Shadow episode per runner process.

Use the RSI Python (see /home/amax/cz/SC2_RSI/env.sh). --dry-run writes the
same validated manifests and design.json as a real batch, without launching SC2.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from importlib.metadata import PackageNotFoundError, version
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from tempfile import mkdtemp

ROOT = Path(__file__).resolve().parents[3]
ADAPTER = ROOT / 'sc2_skill_adapter'
REFERENCE = ROOT / 'runs/contact_all_attack_20261122_230107_662982/marginal_near'
TEMPLATE = REFERENCE / 'zerg_3zergling_0baneling_7hydralisk_00'
MAP_DIR = REFERENCE / 'sc2_skill_adapter/damage_maps'
RSI_PYTHON = '/home/amax/cz/SMAC_skills/sc2_skill_adapter/.venv/bin/python'
sys.path.insert(0, str(ADAPTER))
from bridge.config import digest


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def historical_config():
    run = read_json(TEMPLATE / 'run.json')
    try:
        installed = {pkg: version(pkg) for pkg in run['versions']}
    except PackageNotFoundError as exc:
        raise ValueError(f'{exc}; use {RSI_PYTHON}') from exc
    if installed != run['versions']:
        raise ValueError(f'Historical dependencies differ: {installed}; use {RSI_PYTHON}')
    from bridge.scenarios import validate_manifest
    template = read_json(TEMPLATE / 'manifest.json')
    validate_manifest(template)
    config = template['env_args']
    protocol = ('adaptive-skill', 'tactical_v1', 'micro8-all', 8, 1, True)
    if (tuple(run[k] for k in ('method', 'skill_version', 'execution_mode', 'macro_loops',
                               'engine_step_loops', 'damage_by_type_enabled')) != protocol
            or config != run['env_args'] or template['sha256'] != run['manifest_sha256']
            or (config['map_name'], config['step_mul'], config['unit_balance'],
                config['capability_config']['n_units'], config['capability_config']['n_enemies'])
            != ('10gen_zerg', 8, 'native', 10, 10)):
        raise ValueError(f'Unexpected historical protocol at {TEMPLATE}')
    map_hash = hashlib.sha256((MAP_DIR / '10gen_zerg.SC2Map').read_bytes()).hexdigest()
    if map_hash != run['source_hashes']['sc2_skill_adapter/damage_maps/10gen_zerg.SC2Map']:
        raise ValueError('Historical damage map checksum mismatch')
    return config, installed


def prepare(batch, config, seed, roster_count, episodes, banelings=None):
    from bridge.scenarios import generate_manifest, validate_manifest
    rng = random.Random(seed)
    dist = config['capability_config']['team_gen']
    units = dist['unit_types']
    rosters, jobs, layouts = {}, [], set()
    while len(rosters) < roster_count:
        if banelings is None:
            team = rng.choices(units, weights=dist['weights'], k=10)
        else:
            n = banelings[len(rosters)]
            # Sample a remaining legal composition, not a new team per episode.
            choices = []
            for z in range(11 - n):
                option = ['baneling'] * n + ['zergling'] * z + ['hydralisk'] * (10 - n - z)
                if (not all(u in dist['exception_unit_types'] for u in option)
                        and not any(t.count('baneling') == n and t.count('zergling') == z
                                    for t in rosters.values())):
                    choices.append(option)
            if not choices:
                raise ValueError(f'No distinct legal roster left with {n} banelings')
            team = rng.choice(choices)
            rng.shuffle(team)
        name = 'zerg_' + '_'.join(f'{team.count(u)}{u}' for u in units)
        if name not in rosters and not all(u in dist['exception_unit_types'] for u in team):
            rosters[name] = team
    for group, (name, team) in enumerate(rosters.items()):
        for index in range(episodes):
            # Keep smoke subsets identical to the prefix of the full design.
            sc2_seed = seed + group * 40 + index
            manifest = generate_manifest(config, 1, sc2_seed)
            row, = manifest['episodes']
            cfg = row['episode_config']
            layout = digest([cfg[f'{side}_start_positions']['item'] for side in ('ally', 'enemy')])
            if layout in layouts:
                raise ValueError(f'Duplicate layout for seed {sc2_seed}')
            layouts.add(layout)
            cfg['team_gen'].update(ally_team=team.copy(), enemy_team=team.copy())
            row['sha256'] = digest(cfg)
            del manifest['sha256']
            manifest['sha256'] = digest(manifest)
            validate_manifest(manifest)
            work = Path('workers') / name / f'episode-{index+1:03d}-seed-{sc2_seed}'
            (batch / work).mkdir(parents=True)
            write_json(batch / work / 'manifest.json', manifest)
            jobs.append(dict(worker=str(work), roster=name, sc2_seed=sc2_seed, episode=index+1,
                             layout_sha256=layout, scenario_sha256=row['sha256'],
                             manifest_sha256=manifest['sha256']))
    return rosters, jobs


def play(batch, job, env):
    work = batch / job['worker']
    out, shadow = work / 'runner', work / 'shadow'
    cmd = [sys.executable, '-B', '-u', str(ADAPTER / 'run.py'), 'run', '--method',
           'adaptive-skill', '--skill-version', 'tactical_v1', '--execution-mode', 'micro8-all',
           '--manifest', str(work / 'manifest.json'), '--output-dir', str(out)]
    print(f"seed={job['sc2_seed']} runner={out} shadow={shadow} log={work/'worker.log'}", flush=True)
    with (work / 'worker.log').open('x') as log:
        subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True,
                       env=dict(env, ZERG_SHADOW_ROSTER=job['roster'], ZERG_SHADOW_DIR=str(shadow)))
    record, = read_json(out / 'episodes.json')
    if (record['scenario_sha256'] != job['scenario_sha256'] or record['steps'] <= 0
            or record.get('zerg_shadow_decisions') != record['steps']
            or not (shadow / 'shadow.jsonl').stat().st_size):
        raise ValueError(f'Incomplete episode or Shadow output: {work}')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed', type=int, default=20261123)
    p.add_argument('--workers', type=int, default=60)
    p.add_argument('--rosters', type=int, default=6, help='1..6; reduce for smoke tests')
    p.add_argument('--banelings', type=int, nargs='+', metavar='N',
                   help='fixed banelings per side for each roster, in order')
    p.add_argument('--episodes-per-roster', type=int, default=40, help='1..40')
    p.add_argument('--output-root', type=Path, default=ROOT / 'runs')
    p.add_argument('--dry-run', action='store_true', help='write manifests and design only')
    args = p.parse_args(argv)
    if not 1 <= args.workers <= 60 or not 1 <= args.rosters <= 6 or not 1 <= args.episodes_per_roster <= 40:
        p.error('require workers in 1..60, rosters in 1..6, episodes-per-roster in 1..40')
    if not 0 <= args.seed <= 2**31 - 240:
        p.error('seed must fit in a nonnegative signed 32-bit integer with room for 240 episodes')
    if args.banelings is not None and (len(args.banelings) != args.rosters
                                      or any(not 0 <= n <= 10 for n in args.banelings)):
        p.error('--banelings requires one count in 0..10 per roster')
    try:
        config, versions = historical_config()
    except ValueError as exc:
        p.error(str(exc))
    root = args.output_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    batch = Path(mkdtemp(prefix='zerg_shadow_dry_' if args.dry_run else 'zerg_shadow_', dir=root))
    print(f'Batch: {batch}', flush=True)
    rosters, jobs = prepare(batch, config, args.seed, args.rosters, args.episodes_per_roster,
                            args.banelings)
    env = dict(os.environ, SC2PATH=os.environ.get('SC2PATH') or '/home/amax/cz/SC2_RSI/StarCraftII',
               SMAC_DAMAGE_MAP_DIR=str(MAP_DIR), OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
               MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    env.pop('PYTHONPATH', None)
    for key in ('NO_PROXY', 'no_proxy'):
        env[key] = ','.join(filter(None, (env.get(key), '127.0.0.1,localhost,::1')))
    write_json(batch / 'design.json', dict(seed=args.seed, rosters=rosters, jobs=jobs, workers=args.workers,
               dry_run=args.dry_run, python=sys.executable, versions=versions, reference=str(TEMPLATE),
               execution_mode='micro8-all', damage_map_dir=str(MAP_DIR), sc2_path=env['SC2PATH']))
    print(f'{len(rosters)} rosters x {args.episodes_per_roster} episodes = {len(jobs)}; '
          f'max concurrency {args.workers}; design: {batch / "design.json"}', flush=True)
    if args.dry_run:
        print('Dry-run: SC2 not launched.')
        return
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(play, batch, job, env): job for job in jobs}
        for future in as_completed(pending):
            try:
                future.result()
            except Exception as exc:
                failures.append(pending[future]['worker'])
                print(f'FAILED {batch / failures[-1]}: {exc}', file=sys.stderr, flush=True)
    if failures:
        raise SystemExit(f'{len(failures)} of {len(jobs)} episodes failed; see worker.log in each failed path')
    print(f'Completed {len(jobs)} episodes: {batch}', flush=True)


if __name__ == '__main__':
    main()
