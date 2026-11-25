"""Prefix-only coverage audit: no SC2, outcome labels, or post-divergence simulation."""
import argparse
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

from .combat_relay_r8 import CombatRelayR8


def replay(pair, lock_registered=False):
    run = pair / 'R6/runner'
    doc = json.loads((run / 'distill/episode-0001.json').read_bytes())
    decisions = {r['game_loop']: r for r in doc['decisions']}
    with (pair / 'R6/shadow/shadow.jsonl').open() as f:
        cards = {r['game_loop']: r['candidates'] for r in map(json.loads, f)}
    assert doc['seed'] == json.loads((pair / 'manifest.json').read_bytes())['seed']
    skills = {m: CombatRelayR8(m) for m in ('menu', 'full')}
    hits = {}
    with (run / 'events.jsonl').open('rb') as f:
        for line in f:
            if line.startswith(b'{"event": "skill_observation"'):
                obs = json.loads(line)
                claims = set()
            elif line.startswith(b'{"event": "squad_rendezvous_decision"'):
                parent = json.loads(line)
            elif line.startswith(b'{"event": "micro_authorization_registered"'):
                claims.add(json.loads(line)['actor'])
            elif line.startswith(b'{"event": "step"'):
                step = json.loads(line)
                loop = parent['game_loop']
                base = parent['actions']
                assert loop == obs['payload']['state']['game_loop'] == step['game_loop']
                assert parent['step'] == obs['step'] == step['step'] - 1 and base == step['actions']
                menu = {}
                if parent['handoff']:
                    d = decisions[loop]
                    for c in cards[loop]:
                        k = c['candidate_id']
                        if k in d['scores']:
                            assert c['legal'] and c['candidate_family'] == 'joint_plan'
                            menu[k] = [c['proposed_actions'][f'u{i}'] for i in range(len(base))]
                    assert menu.keys() == d['scores'].keys() and menu[d['selected_candidate']] == base
                for mode in skills.keys() - hits.keys():
                    payload = obs['payload']
                    if lock_registered:
                        # Historical registration is the parent macro's claim,
                        # not a later firing/submission or outcome signal.
                        payload = dict(payload, relay_protected_actors=sorted(claims))
                    r = skills[mode].propose(payload, parent['avail_actions'], base, menu)
                    if r['actions'] != base:
                        novel = r['actions'] not in menu.values()
                        assert mode != 'menu' or not novel
                        changed = {f'u{i}' for i,(x,y) in enumerate(zip(base,r['actions'])) if x != y}
                        if lock_registered:
                            assert not changed.intersection(claims)
                        hits[mode] = dict(step=obs['step'], loop=loop, novel=novel, base=base,
                                          actions=r['actions'], changed_registered=sorted(changed & claims),
                                          audit=r['audit'])
                if len(hits) == len(skills):
                    break
    return dict(pair=pair.name, seed=doc['seed'], hits=hits)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path)
    p.add_argument('--count', type=int, default=100)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--lock-registered', action='store_true', help='conservatively lock all R6 registered actors')
    args = p.parse_args()
    pairs = [args.root / f'pair-{i:03d}' for i in range(1, args.count + 1)]
    with ProcessPoolExecutor(args.workers) as pool:
        rows = list(pool.map(partial(replay, lock_registered=args.lock_registered), pairs))
    print(json.dumps(dict(sample=len(rows), lock_registered=args.lock_registered,
                          changed=Counter(m for r in rows for m in r['hits']),
                          novel_full=sum(r['hits'].get('full', {}).get('novel', False) for r in rows),
                          examples=[r for r in rows if r['hits']][:5]), indent=2))
