"""Compare accepted one-step Zerg fork pairs. No games or training are started."""
import argparse
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from decision_tree.scripts.linux.run_zerg_forks import validate_arm


def read(path):
    return json.loads(path.read_text())


def arm(run, index, name, pair, plan):
    path = run / f'pair-{index:03d}' / name / 'runner'
    receipt = validate_arm(path.parent, name, pair, plan)
    if receipt['engine_receipt'] != 'accepted':
        raise ValueError(f'{name}: engine receipt not accepted: {receipt["engine_receipt"]}')
    ep, = read(path / 'episodes.json')
    anchor = pair['game_loop']
    snapshots = {}
    with (path / 'events.jsonl').open() as f:
        for line in f:
            e = json.loads(line)
            if e.get('event') == 'skill_observation':
                state = e['payload']['state']
                elapsed = state['game_loop'] - anchor
                if elapsed in (8, 24, 40):
                    snapshots[elapsed // 8] = state['allies']
    return ep, snapshots


def compare(run):
    plan, design = read(run/'plan.json'), read(run/'design.json')
    if 'abc_protocol' in plan or any('C' in p for p in plan['pairs']):
        raise ValueError('A/B/C evidence requires run_zerg_abc.py --compare')
    pairs, excluded = [], []
    for i in design['pair_indices']:
        p = plan['pairs'][i-1]
        try:
            a, after_a = arm(run, i, 'A', p, plan)
            b, after_b = arm(run, i, 'B', p, plan)
            if any(a[k] != b[k] for k in ('scenario_sha256', 'initial_state_sha256')):
                raise ValueError('different initial scenario or state')
            followup = {}
            for n in (1, 3, 5):
                if n not in after_a or n not in after_b:
                    followup[n] = None  # a game already ended; not an observed zero
                    continue
                x, y = after_a[n], after_b[n]
                if x.keys() != y.keys() or any(x[u]['tag'] != y[u]['tag'] for u in x):
                    raise ValueError(f'ally identities differ after {n} macro steps')
                common = [u for u in x if x[u]['alive'] and y[u]['alive']]
                followup[n] = dict(ally_alive_delta=sum(v['alive'] for v in y.values()) - sum(v['alive'] for v in x.values()),
                    ally_hp_delta=sum(v['health'] for v in y.values()) - sum(v['health'] for v in x.values()),
                    max_ally_position_distance=max((math.dist(x[u]['position'], y[u]['position']) for u in common), default=None))
            pairs.append(dict(pair=i, seed=p['seed'], roster=p['roster'], changed=p['changed'],
                outcome_A=a['outcome'], outcome_B=b['outcome'], return_A=a['reward'], return_B=b['reward'],
                return_delta=b['reward']-a['reward'], win_delta=int(b['outcome']=='win')-int(a['outcome']=='win'),
                after_macro_steps=followup))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            excluded.append(dict(pair=i, reason=str(exc)))
    return dict(planned_pairs=len(design['pair_indices']), accepted_pairs=len(pairs), excluded=excluded,
                more_wins=sum(p['win_delta']>0 for p in pairs), fewer_wins=sum(p['win_delta']<0 for p in pairs),
                mean_return_delta=sum(p['return_delta'] for p in pairs)/len(pairs) if pairs else None, pairs=pairs)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--run', type=Path, required=True)
    report = compare(cli.parse_args().run)
    print(json.dumps(report, indent=2, allow_nan=False))
    raise SystemExit(1 if report['excluded'] else 0)
