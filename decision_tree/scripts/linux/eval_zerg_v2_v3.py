"""Frozen full v2 vs v3: fresh 240 pairs by default; archives are regression evidence only."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import secrets
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from decision_tree.scripts.linux import eval_zerg_two_trees as trees

ev = trees.ev
ARMS = ('v2', 'v3')
KIND = 'zerg_full_v2_vs_v3'
FRESH = 'zerg_v1_v2_fresh_20261124_190338'
REGRESSION = 'zerg_v1_v2_regression_20261124_185519'
ARCHIVES = {FRESH: '662c156d18240802a1cde73d1ccd78057c365c30752cf8ca6af11235d7c7b510',
            REGRESSION: '7a36b89c1c07102c1ddee08cf1416fd77244a6f973e794944cf5f433b9052cd8'}
ALIASES = {'fresh': FRESH, 'regression': REGRESSION}
V3 = 'decision_tree/zerg/policy_repair_v3.py'
OPENING = 'decision_tree/zerg/opening_support_v2.py'


def archive(name):
    path = ev.RUNS / name / 'design.json'
    ev.require(ev.sha(path) == ARCHIVES[name], 'archive changed')
    return ev.read_json(path)


def policies(sources, common, trial):
    old = archive(FRESH)
    control = old['policies']['v2']
    for path, expected in old['sources'].items():
        if path not in ('sc2_skill_adapter/bridge/runner.py', 'sc2_skill_adapter/bridge/zerg_distill.py'):
            ev.require(sources.get(path) == expected, f'frozen policy dependency changed: {path}')
    ev.require(all(control[k] == v for k, v in common.items()) and
               control['opening_sha256'] == sources[OPENING] and
               control['selector_sha256'] == sources[trees.SELECTORS['v2']], 'wrong frozen v2')
    return dict(v2=dict(control, trial_id=trial), v3=dict(control, trial_id=trial,
                kind='zerg_distill_v3', policy_version='v3', selector_sha256=sources[V3]))


def prepare(out, workers=60, batch=None, ids=None, dry=False):
    ev.require(sys.executable == ev.RSI_PYTHON, f'use historical interpreter {ev.RSI_PYTHON}')
    batch = ALIASES.get(batch, batch)
    ev.require(not out.exists() and 1 <= workers <= 60 and (batch is None or batch in ARCHIVES),
               'invalid output, workers or replay batch')
    ev.require(ids is None or batch and 0 < len(set(ids)) == len(ids) < 240 and
               all(type(i) is int and 1 <= i <= 240 for i in ids), 'pilot needs unique archived pair IDs in 1..240')
    from decision_tree.zerg.policy_repair_v3 import v3_choose
    ev.require(callable(v3_choose), 'missing v3 selector')
    original = ev.reference()
    sources = ev.frozen_sources(original)
    trial = secrets.token_hex(16)
    p = policies(sources, ev.model_contract(ev.MODEL, original), trial)
    return trees.freeze_panel(out, workers, None, dry, batch, None, ids, p, sources,
                              *ev.historical_config(), ARCHIVES, KIND, trial_id=trial,
                              frozen_v2_design_sha256=ARCHIVES[FRESH])


def check_trial_decision(row, card_map, arm, policy):
    ev.require(row['trial_id'] == policy['trial_id'] and row['opening_sha256'] == policy['opening_sha256'] and
               row['applied'] is row['offered'] and row['committed'] is True and row['macro_receipt'] == 'accepted',
               'wrong trial or native receipt')
    choice = card_map[row['selected_candidate']]
    ev.require(choice['legal'] is True and choice['proposed_actions'] == row['final_actions'], 'wrong native choice')
    opening = row['selection_reason'] == 'generated_opening_support'
    ev.require(opening == (choice.get('origin') == 'distill_opening_support') and
               (not opening or choice['rule_hash'] == policy['opening_sha256']), 'wrong opening identity')
    if arm == 'v2':
        ev.require('v3_triggered' not in row and 'v3_actions_changed' not in row, 'v3 receipt on control')
        return False, False
    audit = row['selector_audit']
    vec = audit['v2_actions']
    old_actions = {f'u{i}': a for i, a in enumerate(vec)}
    ev.require(any(card_map[c]['proposed_actions'] == old_actions for c in row['scores']) and
               (not opening or vec == audit['before_opening_actions']), 'invalid v2 reference candidate')
    reference = row['final_actions'] if opening else old_actions
    triggered, changed = row['v3_triggered'], row['v3_actions_changed']
    ev.require(reference == row['v2_reference_actions'] and type(triggered) is bool and
               triggered is audit['v3_triggered'] and changed is (reference != row['final_actions']) and
               (triggered or not changed), 'incorrect v3 intervention receipt')
    return triggered, changed


def trial_counts(work, arm, policy):
    decisions = ev.read_json(work/'runner/distill/episode-0001.json')['decisions']
    triggered = changed = 0
    with (work/'shadow/shadow.jsonl').open() as stream:
        for row, line in zip(decisions, stream):  # Coverage and ordering already checked by audit_arm.
            cards = json.loads(line)['candidates']
            t, c = check_trial_decision(row, {x['candidate_id']: x for x in cards}, arm, policy)
            triggered += t
            changed += c
    return dict(triggered_steps=triggered, changed_steps=changed)


def aggregate(rows):
    def severe(r):
        return r['outcome'] == 'loss' and r['enemy_elimination_fraction'] <= .5

    n = len(rows)
    stats = {}
    for a in ARMS:
        games = [r[a] for r in rows]
        stats[a] = dict(games=n, outcomes=dict(Counter(r['outcome'] for r in games)),
                        wins=sum(r['outcome'] == 'win' for r in games), severe_losses=sum(map(severe, games)),
                        mean_return=sum(r['reward'] for r in games)/n)
    trans = Counter((r['v2']['outcome'], r['v3']['outcome']) for r in rows)
    gained = sum(c for (a,b),c in trans.items() if a != 'win' and b == 'win')
    lost = sum(c for (a,b),c in trans.items() if a == 'win' and b != 'win')
    beats, triggers, changes = [sum(r['v3'][k] for r in rows) for k in ('steps','triggered_steps','changed_steps')]
    return dict(pairs=n, arms=stats, win_flips=dict(gained=gained,lost=lost,net=gained-lost),
                win_to_loss=trans['win','loss'], loss_to_win=trans['loss','win'],
                transitions={f'{a}_to_{b}':v for (a,b),v in trans.items()},
                both_severe_losses=sum(all(severe(r[a]) for a in ARMS) for r in rows),
                severe_definition='literal loss and enemy_elimination_fraction <= 0.5',
                mean_return_delta=stats['v3']['mean_return']-stats['v2']['mean_return'],
                v3_intervention=dict(steps=beats, triggered=triggers, changed=changes, trigger_rate=triggers/beats,
                                     action_change_rate=changes/beats, games_changed=sum(r['v3']['changed_steps']>0 for r in rows),
                                     reference='full v2 on the same v3 observation and original native menu'))


def compare(out):
    d, original = ev.read_json(out/'design.json'), ev.reference()
    ev.require(d['kind'] == KIND and d['dry_run'] is False and d['arms'] == list(ARMS) and
               d['frozen_v2_design_sha256'] == ARCHIVES[FRESH] and
               d['policies'] == policies(d['sources'], ev.model_contract(out/'model.json', original), d['trial_id']) and
               ev.read_json(ev.HISTORY/f'{ev.digest(d)}.json') == d, 'not a frozen full v2-v3 trial')
    batch, ids = d['replay_batch'], d['pilot_ids']
    ev.require(len(d['pairs']) == (len(ids) if ids else 240), 'incomplete panel')
    if batch:
        old = archive(batch)['pairs']
        ev.require(d['replay_design_sha256'] == ARCHIVES[batch] and d['pairs'] ==
                   ([old[i-1] for i in ids] if ids else old), 'wrong archived panel')
    rows = []
    for i,p in enumerate(d['pairs'], 1):
        try:
            m = ev.read_json(out/p['manifest'])
            trees.validate_pair(p, m, ids[i-1] if ids else i, original['env_args'])
            arms = {}
            for a in ARMS:
                policy = d['policies'][a]
                arms[a] = ev.audit_arm(out, p, a, m, original, d, policy)
                counts = trial_counts(out/p['pair_id']/a, a, policy)
                if a == 'v3':
                    arms[a].update(counts)
            ev.require(arms['v2']['initial'] == arms['v3']['initial'], 'initial states differ')
            if batch:
                ep, = ev.read_json(ev.RUNS/batch/p['pair_id']/'v2/runner/episodes.json')
                ev.require(arms['v2']['initial'] == ep['initial_state_sha256'], 'archived initial state differs')
            rows.append(dict(p, **arms))
        except (OSError, TypeError, KeyError, ValueError) as exc:
            raise ValueError(f"{p['pair_id']}: {exc}") from exc
    return dict(**aggregate(rows), kind=KIND, trial_id=d['trial_id'], policies=d['policies'],
                replay_batch=batch, replay_design_sha256=d['replay_design_sha256'], pilot_ids=ids, paired_outcomes=rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--out', type=Path)
    mode.add_argument('--compare', type=Path)
    parser.add_argument('--workers', type=int, help='1..60 (default 60)')
    parser.add_argument('--replay-batch', choices=(*ALIASES, *ARCHIVES))
    parser.add_argument('--pilot-ids', nargs='+', type=int, help='1-based historical pair IDs; replay only')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    try:
        if args.compare:
            ev.require(not args.dry_run and args.workers is args.replay_batch is args.pilot_ids is None,
                       'compare takes no launch options')
            print(json.dumps(compare(args.compare.resolve()), indent=2, allow_nan=False))
            return
        out = args.out.resolve()
        d = prepare(out, 60 if args.workers is None else args.workers, args.replay_batch, args.pilot_ids, args.dry_run)
        print(f"{out}: {2*len(d['pairs'])} games; {d['workers']} workers; dry_run={args.dry_run}", flush=True)
        if args.dry_run:
            return
        with ThreadPoolExecutor(max_workers=d['workers']) as pool:
            jobs = {pool.submit(ev.play, out, p, a, d['policies'][a]): (p['pair_id'],a) for p in d['pairs'] for a in ARMS}
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
    except (OSError, ValueError, ImportError, KeyError, TypeError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
