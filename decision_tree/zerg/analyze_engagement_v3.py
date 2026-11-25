"""Replay scored native menus on recorded v2 states. From root: python -m decision_tree.zerg.analyze_engagement_v3"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import json
from hashlib import sha256
from pathlib import Path

from .policy_distill import vector
from .policy_repair_v3 import volley, profile, v3_choose

PANELS = ('regression_20261124_185519', 'fresh_20261124_190338')


@lru_cache(None)
def model(panel):
    return json.loads((panel / 'model.json').read_text())


def examine(path):
    root = path/'v2/runner'
    terminals = [json.loads((path/arm/'runner/episodes.json').read_text())[0] for arm in ('v1','v2')]
    assert len({(t['initial_state_sha256'], t['scenario_sha256']) for t in terminals}) == 1
    outcomes = [t['outcome'] for t in terminals]
    kills = [t['info']['dead_enemies'] for t in terminals]
    receipt = json.loads((root/'distill/episode-0001.json').read_text())
    row = dict(pair=path.name,panel=path.parent.name,outcomes=outcomes,kills=kills,roster=receipt['roster'],
               window=[],changes=[], steps=len(receipt['decisions']))
    cfg=json.loads((path/'manifest.json').read_text())['episodes'][0]['episode_config']
    allies,enemies=[cfg[k+'_start_positions']['item'] for k in ('ally','enemy')]
    row['layout']='reflected' if all(abs(x+u-32)<1e-6 and abs(y-v)<1e-6 for (x,y),(u,v) in zip(allies,enemies,strict=True)) else 'other_layout'
    counts=Counter(); first=None
    with (root/'events.jsonl').open() as events, (path/'v2/shadow/shadow.jsonl').open() as shadows:
        observations=(json.loads(line) for line in events if '"skill_observation"' in line[:150])
        for ev,shline,d in zip(observations,shadows,receipt['decisions'],strict=True):
            sh=json.loads(shline); s=ev['payload']['state']; step=ev['step']
            assert step==d['step'] and s['game_loop']==sh['game_loop']==d['game_loop'] and int(sh['snapshot_id'])==d['snapshot_id'] and d['committed']
            cards={c['candidate_id']:c for c in sh['candidates']}; ids=list(d['scores']); assert all(cards[i]['legal'] for i in ids)
            menu=[vector(s,cards[i]['proposed_actions']) for i in ids]; actual=vector(s,d['final_actions']); assert actual==menu[ids.index(d['selected_candidate'])]
            if step==0: row['hydras']=sum(u['type']=='Hydralisk' for u in s['allies'].values())
            generated=d['selection_reason']=='generated_opening_support'
            if generated:
                assert cards[ids[-1]]['origin']=='distill_opening_support'; menu=menu[:-1]
            result=v3_choose(model(path.parent),s,menu)
            before=tuple(result['audit']['v2_actions'])
            assert before==(tuple(d['selector_audit']['before_opening_actions']) if generated else actual)
            if generated: assert result['reason']=='no_existing_formation_fix'
            else: assert result['audit']['v2_reason']==d['selection_reason']
            changed=not generated and tuple(result['actions'])!=actual
            counts['multimenu']+=len(menu)>1; counts['generated_opening']+=generated; counts['protected']+=d['selection_reason']=='protect_hydra_support_gap'
            if changed:
                assert d['selection_reason'] not in ('protect_hydra_support_gap','no_existing_formation_fix')
                row['changes'].append(dict(step=step,loop=s['game_loop'],before=actual,after=result['actions'],audit=result['audit']['engagement']))
            table=volley(s); p=profile(s,table,actual)
            contact=any(r['smac_attack_available'] for rel in s['relations'].values() for r in rel.values())
            if contact and first is None: first=step
            if first is None or step-first>=4: continue
            # Incoming range is geometry only; enemy cooldown and future orders are unknown.
            incoming={e for rel in s['relations'].values() for e,r in rel.items() if r['in_enemy_weapon_range'] is True}
            reachable=sum(bool(targets) for targets in table.values()); fired=len(p['actors'])
            own_range=sum(any(r['in_weapon_range'] is True for r in rel.values()) for rel in s['relations'].values())
            row['window'].append(dict(step=step,loop=s['game_loop'],ready_reachable=reachable,fired=fired,own_range=own_range,
                incoming=len(incoming),missed=reachable>fired,split=len(p['loads'])>=2,
                overkill=p['excess']>0,shortfall=len(incoming)>reachable,range_deficit=len(incoming)>own_range,changed=changed,menu=len(menu),
                max_fired=max(len(profile(s,table,v)['actors']) for v in menu),focus=max(Counter(actual[i] for i in p['actors']).values(),default=0)))
    row.update(counts=counts,first=first); return row


def summarize(rows):
    groups=defaultdict(Counter)
    for r in rows:
        w=r['window']; ll=r['outcomes']==['loss','loss']; outcome=r['outcomes'][1]
        cohorts=['all','v2_'+outcome]+(['LL'] if ll else [])+(['both_severe'] if ll and max(r['kills'])<=5 else [])+(['v2_severe'] if outcome=='loss' and r['kills'][1]<=5 else [])
        for cohort in cohorts:
            band='H6+' if r['hydras']>=6 else 'H0-5'
            for split in ('all',r['panel'],f"H{r['hydras']}",band,r['layout'],r['layout']+'/'+band,r['roster']):
                c=groups[split+'/'+cohort]; c['n']+=1;c['contact']+=bool(w);c['changed_any']+=bool(r['changes']);c['changed_beats']+=len(r['changes']);c['steps']+=r['steps'];c['multi_beats']+=r['counts']['multimenu']
                for key in ('shortfall','range_deficit','missed','split','overkill','changed'):
                    c[key]+=any(b[key] for b in w); c['first_'+key]+=bool(w and w[0][key]);c['early_'+key+'_beats']+=sum(b[key] for b in w)
                c['early_beats']+=len(w);c['more_shots_available']+=any(b['max_fired']>b['fired'] for b in w)
    return groups


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--runs',type=Path,default=Path('runs'));ap.add_argument('--workers',type=int,default=8);ap.add_argument('--out',type=Path,default=Path('runs/zerg_engagement_v3_analysis.json'));a=ap.parse_args()
    pairs=[]
    for panel in PANELS:
        ps=sorted((a.runs/('zerg_v1_v2_'+panel)).glob('pair-*'));assert len(ps)==240;pairs+=ps
    with ProcessPoolExecutor(a.workers) as pool: rows=list(pool.map(examine,pairs,chunksize=2))
    groups=summarize(rows)
    report=dict(window='first any SMAC attack available through next three recorded macros',
        limitation='Same-state native action coverage, not trajectories under v3. Known direct weapons only, not realized damage, splash, micro orders or causal win effects.',
        counts='n is games; first_ is first contact only; early_*_beats sums four macro decisions per game; other signals are games with at least one occurrence in the window. changed_any/changed_beats cover the entire game.',
        source_sha256={name:sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in ('policy_repair_v2.py','policy_repair_v3.py','policy_distill.py')},groups=groups,rows=rows)
    a.out.write_text(json.dumps(report,separators=(',',':'))+'\n');print({k:(v['n'],v['first_shortfall'],v['changed'],v['changed_any']) for k,v in groups.items() if k.startswith('all/')})


if __name__=='__main__': main()
