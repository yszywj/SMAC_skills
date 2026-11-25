"""Frozen-v2 reference replay. Run from repo root: python -m decision_tree.zerg.audit_route_alt --out PATH"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

from .analyze_engagement_v3 import PANELS, model
from .policy_distill import vector
from .policy_route_alt import choose


def examine(path):
    root = path / 'v2/runner'
    terminal, = json.loads((root / 'episodes.json').read_text())
    outcome = terminal['outcome']
    kills = terminal['info']['dead_enemies']
    cohort = ('severe' if kills <= 5 else 'ordinary_loss') if outcome == 'loss' else outcome
    receipt = json.loads((root / 'distill/episode-0001.json').read_text())
    decisions = receipt['decisions']
    changes = []; first_contact = None; metrics = Counter()
    with (root / 'events.jsonl').open() as events, (path / 'v2/shadow/shadow.jsonl').open() as shadows:
        observations = (json.loads(x) for x in events if '"skill_observation"' in x[:150])
        for event, shadow, d in zip(observations, shadows, decisions, strict=True):
            s = event['payload']['state']; sh = json.loads(shadow); step = event['step']
            assert step == d['step'] and s['game_loop'] == d['game_loop'] == sh['game_loop']
            assert int(sh['snapshot_id']) == d['snapshot_id'] and d['committed'] and d['macro_receipt'] == 'accepted'
            if step == 0:
                hydras = sum(u['type'].lower() == 'hydralisk' for u in s['allies'].values())
            if first_contact is None and any(r['smac_attack_available'] for rel in s['relations'].values() for r in rel.values()):
                first_contact = step
            cards = {c['candidate_id']: c for c in sh['candidates']}; ids = list(d['scores'])
            assert all(cards[i]['legal'] for i in ids)
            menu = [vector(s, cards[i]['proposed_actions']) for i in ids]
            actual = vector(s, d['final_actions']); assert actual == menu[ids.index(d['selected_candidate'])]
            generated = d['selection_reason'] == 'generated_opening_support'
            if generated:
                assert cards[ids[-1]]['origin'] == 'distill_opening_support'
                menu.pop(); metrics['opening_beats'] += 1
            result = choose(model(path.parent), s, menu); audit = result['audit']
            before = tuple(audit['v2_actions']); after = tuple(result['actions'])
            assert after in menu and len(menu) == len(result['scores']) and result['score'] == max(result['scores'])
            assert before == (tuple(d['selector_audit']['before_opening_actions']) if generated else actual)
            assert audit['v2_reason'] == ('no_existing_formation_fix' if generated else d['selection_reason'])
            if generated:
                assert result['reason'] == 'no_existing_formation_fix' and after == before
            phase = audit['route_phase']
            if phase != 'inactive':
                metrics['eligible_beats'] += 1
                metrics['movement_menu_beats'] += any(v != before and all(a == b or max(a,b) < 6 for a,b in zip(before,v)) for v in menu)
            if after != before:
                assert not generated and audit['v3_triggered'] and all(a == b or max(a,b) < 6 for a,b in zip(before,after))
                changes.append(dict(step=step, loop=s['game_loop'], phase=phase, before=before, after=after,
                                    actors=sum(a != b for a,b in zip(before,after)), before_cost=audit['before_cost'], after_cost=audit['after_cost']))
    return dict(panel=path.parent.name, pair=path.name, cohort=cohort, kills=kills, hydras=hydras, steps=len(decisions),
                metrics=metrics, changes=changes, first_contact=first_contact)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True); parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args(); pairs = []
    for panel in PANELS:
        paths = sorted((Path('runs') / ('zerg_v1_v2_' + panel)).glob('pair-*')); assert len(paths) == 240
        pairs.extend(paths)
    with ProcessPoolExecutor(args.workers) as pool:
        rows = list(pool.map(examine, pairs, chunksize=2))
    groups = defaultdict(Counter)
    for r in rows:
        changes = r['changes']; c = r['first_contact']; first = changes[0]['step'] if changes else None
        for key in ('all', r['cohort'], r['panel']+'/'+r['cohort'], ('H6+' if r['hydras']>=6 else 'H0-5')+'/'+r['cohort']):
            g = groups[key]; g.update(r['metrics']); g.update(games=1, beats=r['steps'], changed_games=int(bool(changes)), changed_beats=len(changes),
                first_before_contact=int(first is not None and (c is None or first<c)), first_contact_window=int(first is not None and c is not None and c<=first<c+4),
                eligible_games=int(bool(r['metrics']['eligible_beats'])), movement_menu_games=int(bool(r['metrics']['movement_menu_beats'])))
            for b in changes:
                g[b['phase']+'_beats'] += 1; g['changed_'+str(b['actors'])+'_actors'] += 1
    output = dict(scope='Frozen v2 trajectories and certified menus; only the first divergence has a shared prefix. Movement endpoints and static enemies are proxies, not damage or win evidence.',
                  severe='v2 loss with at most 5 dead enemies', source_sha256={s:hashlib.sha256(Path(__file__).with_name(s).read_bytes()).hexdigest() for s in ('policy_route_alt.py','policy_repair_v2.py','policy_distill.py')}, groups=groups, rows=rows)
    args.out.write_text(json.dumps(output,separators=(',',':'))+'\n')
    for k in ('all','severe','ordinary_loss','win','draw'): print(k, groups[k])


if __name__ == '__main__':
    main()
