"""240 paired games per tree. Replay one pinned old panel OR draw fresh native mirrored games."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import json
from pathlib import Path
import secrets
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from decision_tree.scripts.linux import eval_zerg_policy as ev
from bridge.scenarios import validate_manifest

ARMS = ('v1', 'v2')
SELECTORS = dict(v1='decision_tree/zerg/policy_distill.py', v2='decision_tree/zerg/policy_repair_v2.py')
OLD = {
    'zerg_policy_eval_20261124_162949': '91d9fa1f1ffa07c985be66bd5d247a88bd43859dd352ff441cd57357a6db8c88',
    'zerg_policy_eval_6x40x2_20261124_164422': '310cea82f812ec05a5bcdf35aad9125b4c3e855e2a60109dd058a63c2b5df955',
    'zerg_policy_eval_random_20261124_165149': '154b95c25fd67bab37fc6d76763ae54e9fcb14caf0256bf03ac330c52af6f8b4',
}


def validate_pair(p, m, i, config):
    validate_manifest(m)
    ep, = m['episodes']
    cfg = ep['episode_config']
    ally = cfg['team_gen']['ally_team']
    roster = 'zerg_' + '_'.join(f'{ally.count(t)}{t}' for t in ('zergling', 'baneling', 'hydralisk'))
    pid = f'pair-{i:03d}'
    ev.require(m['env_args'] == config and m['seed'] == ep['sc2_seed'] == p['seed'] and
               m['sha256'] == p['manifest_sha256'] and ep['sha256'] == p['scenario_sha256'] and
               p['pair_id'] == pid and p['manifest'] == f'{pid}/manifest.json' and
               ally == cfg['team_gen']['enemy_team'] and roster == p['roster'] and
               p['layout_sha256'] == ev.digest([cfg[f'{s}_start_positions']['item'] for s in ('ally', 'enemy')]),
               f'{pid}: invalid mirrored manifest')


def old_pairs(batch):
    path = ev.RUNS / batch / 'design.json'
    ev.require(ev.sha(path) == OLD[batch], 'archived design changed')
    return ev.read_json(path)['pairs']


def historical_keys():
    _, seen = ev.originals()
    for history in (ev.AC_HISTORY, ev.HISTORY):
        for path in history.glob('*.json'):
            d = ev.read_json(path)
            ev.require(path.stem == ev.digest(d), f'corrupt reservation: {path}')
            for p in d['pairs']:
                for k in ev.KEYS:
                    seen[k].add(p[k])
    for batch in OLD:
        for p in old_pairs(batch):
            for k in ev.KEYS:
                seen[k].add(p[k])
    # Older, pre-reservation runs also contain SC2 seeds, sometimes in multigame manifests.
    for path in ev.RUNS.rglob('*manifest*.json'):
        manifest = ev.read_json(path)
        if isinstance(manifest, dict):
            for episode in manifest.get('episodes', []) if isinstance(manifest.get('episodes'), list) else ():
                if isinstance(episode, dict) and 'sc2_seed' in episode:
                    seen['seed'].add(episode['sc2_seed'])
    return seen


def prepare(out, workers=60, seed=None, dry=False, batch=None, pilot=None, ids=None, opening=False):
    ev.require(sys.executable == ev.RSI_PYTHON, f'use historical interpreter {ev.RSI_PYTHON}')
    ev.require(1 <= workers <= 60 and (pilot is None or batch and 1 <= pilot < 240), 'invalid workers/pilot')
    ev.require(ids is None or batch and pilot is None and 0 < len(set(ids)) == len(ids) < 240 and
               all(type(i) is int and 1 <= i <= 240 for i in ids), 'invalid pilot ids')
    ev.require(batch is None or batch in OLD and seed is None, 'replay cannot override original seed')
    ev.require(not out.exists(), f'output already exists: {out}')
    original = ev.reference()
    config, versions = ev.historical_config()
    sources = ev.frozen_sources(original)
    ev.require(sources[SELECTORS['v1']] == '46674db85a3a29dafba02e83bb33c1c82863087b02079490c286433c27c4ddfd',
               'v1 scorer differs from archived panels')
    # Fail at design time rather than creating jobs with an unavailable v2 scorer.
    from decision_tree.zerg.policy_repair_v2 import v2_choose
    ev.require(callable(v2_choose), 'missing v2 selector')
    common = ev.model_contract(ev.MODEL, original)
    policies = {a: dict(common, kind=f'zerg_distill_{a}', policy_version=a, selector_sha256=sources[s])
                 for a, s in SELECTORS.items()}
    if opening:
        from decision_tree.zerg.opening_support_v2 import make_entry
        ev.require(callable(make_entry),'missing native opening generator')
        policies['v2']['opening_sha256']=sources['decision_tree/zerg/opening_support_v2.py']
    return freeze_panel(out, workers, seed, dry, batch, pilot, ids, policies, sources, config,
                        versions, OLD, 'zerg_two_trees_v1')


def freeze_panel(out, workers, seed, dry, batch, pilot, ids, policies, sources, config, versions,
                 archives, kind, panel_size=240, **identity):
    ev.HISTORY.mkdir(exist_ok=True, parents=True)
    with (ev.HISTORY/'lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if batch:
            path = ev.RUNS / batch / 'design.json'
            ev.require(ev.sha(path) == archives[batch], 'archived design changed')
            pairs = ev.read_json(path)['pairs'][:pilot]
            if ids:
                pairs = [pairs[i-1] for i in ids]
            manifests = [ev.read_json(ev.RUNS/batch/p['manifest']) for p in pairs]
            seed = pairs[0]['seed']
        else:
            seen = historical_keys()
            if seed is None:
                while True:
                    seed = secrets.randbelow(2**31-panel_size+1)
                    if seen['seed'].isdisjoint(range(seed, seed+panel_size)):
                        break
            ev.require(0 <= seed <= 2**31-panel_size, 'invalid first seed')
            pairs, manifests = [], []
            for i in range(panel_size):
                name, team = ev.native_team(config, seed+i)
                (p,), (m,) = ev.generate(config, {name: team}, 1, seed+i, seen)
                pid = f'pair-{i+1:03d}'
                pairs.append(dict(p, pair_id=pid, manifest=f'{pid}/manifest.json'))
                manifests.append(m)
        ev.require(len(pairs) == (len(ids) if ids else pilot or panel_size), 'incomplete panel')
        for i, (p, m) in enumerate(zip(pairs, manifests), 1):
            validate_pair(p, m, ids[i-1] if ids else i, config)
        d = dict(kind=kind, dry_run=dry, arms=list(policies), pairs=pairs, policies=policies,
                 replay_batch=batch, replay_design_sha256=archives.get(batch), pilot_pairs=len(ids) if ids else pilot,
                 pilot_ids=ids,
                 native_random_mirror=batch is None, seed=seed, workers=workers, sources=sources,
                 versions=versions, python=ev.RSI_PYTHON, sc2_path=ev.SC2_PATH, damage_map_dir=str(ev.MAP_DIR),
                 historical_run_sha256=ev.REFERENCE_HASH, **ev.PROTOCOL, **identity)
        out.mkdir(parents=True, exist_ok=False)
        (out/'model.json').write_bytes(ev.MODEL.read_bytes())
        ev.require(ev.sha(out/'model.json') == ev.MODEL_HASH, 'model changed while freezing')
        for p, m in zip(pairs, manifests):
            for a in policies:
                (out/p['pair_id']/a).mkdir(parents=True)
            ev.write_json(out/p['manifest'], m)
        ev.write_json(out/'design.json', d)
        reservation = ev.HISTORY/f'{ev.digest(d)}.json'
        if not reservation.exists():
            ev.write_json(reservation, d)
    return d


def aggregate(rows):
    def severe(x):
        return x['outcome'] == 'loss' and x['enemy_elimination_fraction'] <= .5
    stats = {}
    for a in ARMS:
        arm = [r[a] for r in rows]
        stats[a] = dict(games=len(arm), outcomes=dict(Counter(x['outcome'] for x in arm)),
                        mean_return=sum(x['reward'] for x in arm)/len(arm), severe_losses=sum(map(severe, arm)))
    transitions = Counter((r['v1']['outcome'], r['v2']['outcome']) for r in rows)
    return dict(pairs=len(rows), arms=stats, win_to_loss=transitions['win', 'loss'], loss_to_win=transitions['loss','win'],
                transitions={f'{a}_to_{b}': n for (a,b), n in transitions.items()},
                both_severe_losses=sum(all(severe(r[a]) for a in ARMS) for r in rows),
                severe_definition='loss and enemy_elimination_fraction <= 0.5',
                mean_return_delta=stats['v2']['mean_return']-stats['v1']['mean_return'])


def compare(out):
    d, original = ev.read_json(out/'design.json'), ev.reference()
    ev.require(d['kind'] == 'zerg_two_trees_v1' and d['dry_run'] is False and d['arms'] == list(ARMS)
               and len(d['pairs']) == (d['pilot_pairs'] or 240) and ev.sha(out/'model.json') == ev.MODEL_HASH and
               ev.read_json(ev.HISTORY/f'{ev.digest(d)}.json') == d, 'not a complete frozen two-tree design')
    batch = d['replay_batch']
    ids = d['pilot_ids']
    if batch:
        old = old_pairs(batch)
        ev.require(d['replay_design_sha256'] == OLD[batch] and d['pairs'] ==
                   ([old[i-1] for i in ids] if ids else old[:d['pilot_pairs']]),
                   'replay panel differs from archive')
    rows = []
    for i, p in enumerate(d['pairs'], 1):
        try:
            m = ev.read_json(out/p['manifest'])
            validate_pair(p, m, ids[i-1] if ids else i, original['env_args'])
            arms = {a: ev.audit_arm(out, p, a, m, original, d, d['policies'][a]) for a in ARMS}
            ev.require(arms['v1']['initial'] == arms['v2']['initial'], 'observed initial states differ')
            if batch:
                for old_arm in ('tree', 'baseline'):
                    ep, = ev.read_json(ev.RUNS/batch/p['pair_id']/old_arm/'runner/episodes.json')
                    ev.require(ep['initial_state_sha256'] == arms['v1']['initial'], 'archived initial state differs')
            rows.append(dict(p, **arms))
        except (KeyError, TypeError, ValueError, OSError) as exc:
            raise ValueError(f"{p['pair_id']}: {exc}") from exc
    return dict(**aggregate(rows), replay_batch=batch, replay_design_sha256=d['replay_design_sha256'],
                pilot_pairs=d['pilot_pairs'], pilot_ids=ids, policies=d['policies'], paired_outcomes=rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--out', type=Path, help='new directory; refuses existing output')
    mode.add_argument('--compare', type=Path, help='read-only, audit every requested pair')
    parser.add_argument('--replay-batch', choices=OLD, help='reuse exact seeds and manifests, not independent new evidence')
    pilot = parser.add_mutually_exclusive_group()
    pilot.add_argument('--pilot-pairs', type=int, help='exploratory prefix of an old panel only')
    pilot.add_argument('--pilot-ids', nargs='+', type=int, help='targeted regression only: 1-based archived pair IDs')
    parser.add_argument('--seed', type=int, help='first fresh SC2 seed; any collision is refused')
    parser.add_argument('--workers', type=int, help='1..60 (default 60)')
    parser.add_argument('--dry-run', action='store_true', help='freeze and reserve, never launch games')
    parser.add_argument('--opening-support',action='store_true',help='also enable native formation generation in v2; omit for reranking-only ablation')
    args = parser.parse_args(argv)
    try:
        if args.compare:
            ev.require(not args.dry_run and not args.opening_support and all(v is None for v in (args.seed, args.workers, args.replay_batch,
                       args.pilot_pairs, args.pilot_ids)), 'compare takes no launch options')
            print(json.dumps(compare(args.compare.resolve()), indent=2, allow_nan=False))
            return
        out = args.out.resolve()
        d = prepare(out, 60 if args.workers is None else args.workers, args.seed, args.dry_run,
                     args.replay_batch, args.pilot_pairs, args.pilot_ids, args.opening_support)
        print(f"{out}: {len(d['pairs'])} pairs / {2*len(d['pairs'])} games; {d['workers']} workers", flush=True)
        if args.dry_run:
            print('Dry-run: no games launched.')
            return
        with ThreadPoolExecutor(max_workers=d['workers']) as pool:
            jobs = {pool.submit(ev.play, out, p, a, d['policies'][a]): (p['pair_id'], a) for p in d['pairs'] for a in ARMS}
            for f in as_completed(jobs):
                try:
                    f.result()
                except Exception as exc:
                    for job in jobs:
                        job.cancel()
                    raise RuntimeError(f'{jobs[f]} failed; see worker.log') from exc
        report = compare(out)
        ev.write_json(out/'report.json', report)
        print(json.dumps({k:v for k,v in report.items() if k != 'paired_outcomes'}, indent=2))
    except (OSError, ImportError, KeyError, ValueError, TypeError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
