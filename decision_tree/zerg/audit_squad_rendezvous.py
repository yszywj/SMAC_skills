"""Replay v2 opening prefixes, not counterfactual games. Invoke with -m and archive path."""
import argparse
import json
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from .squad_rendezvous import SquadRendezvous


def replay(job):
    path, row = job
    policies = {m: SquadRendezvous(m) for m in ('fixed', 'adaptive')}
    stats = {m: Counter() for m in policies}
    with (path / row['pair_id'] / 'v2/runner/events.jsonl').open() as stream:
        payload = None
        for line in stream:
            if line.startswith('{"event": "skill_observation"'):
                assert payload is None
                obs = json.loads(line)
                payload = obs['payload']
            elif line.startswith('{"event": "step"'):
                step = json.loads(line)
                assert payload and step['step'] == obs['step'] + 1
                assert step['game_loop'] == payload['state']['game_loop']
                base = step['actions']
                masks = [[str(a) in payload['questions'][f'u{i}']['criteria'] for a in range(16)]
                         for i in range(len(base))]
                for mode, policy in policies.items():
                    if policy.exit_reason:
                        continue
                    result = policy.propose(payload, masks, base)
                    acts = result['actions']
                    assert len(acts) == len(base) and all(0 <= a < 16 and masks[i][a] for i, a in enumerate(acts))
                    assert all(a == b for a, b in zip(acts, base) if b >= 6)
                    assert not result['handoff'] or acts == base
                    n = sum(a != b for a, b in zip(acts, base))
                    stats[mode].update(checked=len(base), changed=n, covered=int(n > 0 and not stats[mode]['covered']))
                payload = None
                if all(p.exit_reason for p in policies.values()):
                    break
    assert all(p.exit_reason for p in policies.values())
    arms = [row[a] for a in ('v2', 'swarm')]
    group = '/'.join(a['outcome'] for a in arms)
    groups = ['all', group]
    if group == 'loss/loss' and max(a['enemy_elimination_fraction'] for a in arms) <= .5:
        groups.append('both_severe')
    return groups, stats


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    rows = json.loads((args.archive / 'report.json').read_text())['paired_outcomes']
    totals = defaultdict(lambda: defaultdict(Counter))
    with ProcessPoolExecutor(args.workers) as pool:
        for groups, stats in pool.map(replay, [(args.archive, r) for r in rows], chunksize=4):
            for group in groups:
                for mode, counts in stats.items():
                    totals[group][mode].update(counts)
                    totals[group][mode]['pairs'] += 1
    print(json.dumps(totals, indent=2, sort_keys=True))
