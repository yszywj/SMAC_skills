"""Exploratory whole-game A/C on fresh scenes, including non-triggering games.

Run with the historical RSI Python. For an independent repeat, keep the same
checkout, frozen selector/rule and per-roster count, specify a new --out, and
omit --seed. Keep runs/.zerg_whole_ac_history: its reservations include dry
runs and failed launches, even when their output directories are elsewhere.
This experiment is not a deployment decision or an exporter of gain labels.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import hashlib
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from decision_tree.scripts.linux.collect_zerg_shadow import (
    ADAPTER, MAP_DIR, RSI_PYTHON, historical_config, read_json, write_json)
from decision_tree.zerg.rules_v1 import RULE_ID, rule_hash
from bridge.config import digest

KIND = 'zerg_whole_ac_v1'
ARMS = ('A', 'C')
HISTORY = ROOT / 'runs/.zerg_whole_ac_history'
SHADOW = {
    'zerg_shadow_az17j4ru': '2d6ecc1b60fad9be481bf675f5a80befdfff1e9ba1db85d9d11a5a97ffca2fb7',
    'zerg_shadow_ptpy1yzu': '3d73195af600bd3449eb417a0690d0943cdd013cd72c20fbe39aff1635658843',
}
PROTOCOL = dict(method='adaptive-skill', skill_version='tactical_v1', execution_mode='micro8-all')
SC2_PATH = '/home/amax/cz/SC2_RSI/StarCraftII'
KEYS = ('seed', 'scenario_sha256', 'layout_sha256')


def frozen_identity():
    from bridge.zerg_whole_ac import SELECTOR
    card = ROOT / 'decision_tree/zerg/configs/dual_lane_trial_v1.json'
    if (hashlib.sha256(card.read_bytes()).hexdigest() != '4165ad93339e77fae3e543bb6ad190d19475b22872a85dbf1730dfff622be355'
            or rule_hash() != 'ea21f021ceeb49612499a24b74276ae6dfc33e2ecea60b739e4361030a009adc'):
        raise ValueError('frozen trial card or rule changed')
    data = read_json(card)
    return dict(rule_id=RULE_ID, rule_hash=rule_hash(), source_hash=data['parent_source'],
                collection_context=data['parent_context'], selector=SELECTOR,
                selector_hash=hashlib.sha256((ADAPTER/'bridge/zerg_whole_ac.py').read_bytes()).hexdigest())


def originals():
    rosters, seen = {}, {k: set() for k in KEYS}
    for batch, sha in SHADOW.items():
        raw = (ROOT / 'runs' / batch / 'design.json').read_bytes()
        if hashlib.sha256(raw).hexdigest() != sha:
            raise ValueError(f'{batch}: original Shadow design checksum mismatch')
        design = json.loads(raw)
        rosters.update(design['rosters'])
        for job in design['jobs']:
            for key in KEYS:
                seen[key].add(job['sc2_seed' if key == 'seed' else key])
    return rosters, seen


def generate(config, rosters, per_roster, seed, seen):
    from bridge.scenarios import generate_manifest, validate_manifest
    pairs, manifests = [], []
    for roster, team in rosters.items():
        for _ in range(per_roster):
            n = seed + len(pairs)
            if n in seen['seed']:
                raise ValueError(f'previously used seed: {n}')
            manifest = generate_manifest(config, 1, n)
            row, = manifest['episodes']
            cfg = row['episode_config']
            cfg['team_gen'].update(ally_team=team.copy(), enemy_team=team.copy())
            layout = digest([cfg[f'{side}_start_positions']['item'] for side in ('ally', 'enemy')])
            row['sha256'] = digest(cfg)
            del manifest['sha256']
            manifest['sha256'] = digest(manifest)
            validate_manifest(manifest)
            pid = f'pair-{len(pairs)+1:03d}'
            pair = dict(pair_id=pid, roster=roster, seed=n, manifest=f'{pid}/manifest.json',
                        manifest_sha256=manifest['sha256'], scenario_sha256=row['sha256'], layout_sha256=layout)
            for key in KEYS:
                if pair[key] in seen[key]:
                    raise ValueError(f'{pid}: previously used {key}: {pair[key]}')
                seen[key].add(pair[key])
            pairs.append(pair)
            manifests.append(manifest)
    return pairs, manifests


def prepare(out, per_roster, workers, seed, dry_run):
    config, versions = historical_config()
    rosters, seen = originals()
    identity = frozen_identity()
    limit = 2**31 - len(rosters)*per_roster
    if limit < 0 or (seed is not None and not 0 <= seed <= limit):
        raise ValueError('seed range must fit signed nonnegative 32-bit SC2 seeds')
    HISTORY.mkdir(parents=True, exist_ok=True)
    with (HISTORY/'lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for path in sorted(HISTORY.glob('*.json')):
            prior = read_json(path)
            if path.stem != digest(prior) or any(prior[k] != v for k, v in identity.items()):
                raise ValueError(f'{path}: invalid reservation or changed frozen selector/rule')
            for pair in prior['pairs']:
                for k in KEYS:
                    seen[k].add(pair[k])
        if seed is None:
            seed = secrets.randbelow(limit+1)
        pairs, manifests = generate(config, rosters, per_roster, seed, seen)
        design = dict(kind=KIND, arms=list(ARMS), exploratory_only=True, dry_run=dry_run, workers=workers,
                      seed=seed, rosters=rosters, per_roster=per_roster, pairs=pairs, python=RSI_PYTHON,
                      sc2_path=SC2_PATH, damage_map_dir=str(MAP_DIR), shadow_design_hashes=SHADOW,
                      versions=versions, **PROTOCOL, **identity)
        out.mkdir(parents=True, exist_ok=False)
        write_json(HISTORY / f'{digest(design)}.json', design)
    for pair, manifest in zip(pairs, manifests):
        work = out / pair['pair_id']
        for arm in ARMS:
            (work/arm).mkdir(parents=True)
        write_json(out / pair['manifest'], manifest)
    write_json(out/'design.json', design)
    return design


def play(out, pair, arm):
    work = out / pair['pair_id'] / arm
    env = {k: v for k, v in os.environ.items() if not k.startswith(('ZERG_', 'SMAC_', 'PYTHON'))}
    env.update(SC2PATH=SC2_PATH, SMAC_DAMAGE_MAP_DIR=str(MAP_DIR), OPENBLAS_NUM_THREADS='1',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1',
               ZERG_WHOLE_AC_ARM=arm, ZERG_SHADOW_ROSTER=pair['roster'], ZERG_SHADOW_DIR=str(work/'shadow'))
    for key in ('NO_PROXY', 'no_proxy'):
        env[key] = ','.join(filter(None, (env.get(key), '127.0.0.1,localhost,::1')))
    cmd = [RSI_PYTHON, '-B', '-u', str(ADAPTER/'run.py'), 'run', '--method', PROTOCOL['method'],
           '--skill-version', PROTOCOL['skill_version'], '--execution-mode', PROTOCOL['execution_mode'],
           '--manifest', str(out/pair['manifest']), '--output-dir', str(work/'runner')]
    with (work/'worker.log').open('x') as log:
        subprocess.run(cmd, cwd=ROOT, env=env, check=True, stdout=log, stderr=subprocess.STDOUT)


def report(run):
    from decision_tree.scripts.linux.compare_zerg_whole_ac import compare
    design = read_json(run/'design.json')
    result = compare(run)
    print(json.dumps(result, indent=2, allow_nan=False))
    if (design['dry_run'] is not False or result['excluded'] != [] or type(result['accepted_pairs']) is not int
            or result['accepted_pairs'] != len(design['pairs']) or not design['pairs']):
        raise SystemExit(1)


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    mode = cli.add_mutually_exclusive_group(required=True)
    mode.add_argument('--out', type=Path, help='new run directory')
    mode.add_argument('--compare', type=Path, metavar='RUN', help='compare an existing run only')
    cli.add_argument('--per-roster', type=int, help='pairs per roster (default 10)')
    cli.add_argument('--workers', type=int, help='maximum concurrent games (default 48)')
    cli.add_argument('--seed', type=int, help='first scenario seed (default fresh random)')
    cli.add_argument('--dry-run', action='store_true', help='reserve scenes and write manifests; no games')
    args = cli.parse_args(argv)
    try:
        if args.compare is not None:
            if args.dry_run or any(v is not None for v in (args.seed, args.per_roster, args.workers)):
                cli.error('--compare does not take launch options')
            return report(args.compare.expanduser().resolve())
        workers = 48 if args.workers is None else args.workers
        per_roster = 10 if args.per_roster is None else args.per_roster
        out = args.out.expanduser().resolve()
        if workers < 1 or per_roster < 1 or out.exists():
            cli.error('require positive counts and a new --out')
        design = prepare(out, per_roster, workers, args.seed, args.dry_run)
        print(f"{len(design['pairs'])} matched pairs, seed {design['seed']}: {out}", flush=True)
        if args.dry_run:
            return
        failures = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            pending = {pool.submit(play, out, p, a): (p['pair_id'], a) for p in design['pairs'] for a in ARMS}
            for future in as_completed(pending):
                try:
                    future.result()
                except Exception as exc:
                    failures += 1
                    print(f'FAILED {pending[future]}: {exc}', file=sys.stderr, flush=True)
        report(out)
        if failures:
            raise SystemExit(f'{failures} worker processes failed')
    except (OSError, ValueError, KeyError, TypeError, ImportError) as exc:
        cli.error(str(exc))


if __name__ == '__main__':
    main()
