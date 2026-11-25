"""Offline audit of paired archived observations, actual commands and receipts."""
import hashlib
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from itertools import zip_longest
from pathlib import Path
from math import dist

RUNS = Path(__file__).resolve().parents[3] / 'runs'
BATCHES = ['zerg_v1_v2_regression_20261124_185519', 'zerg_v1_v2_fresh_20261124_190338']
OUT = RUNS / 'zerg_v1_v2_trace_audit'


def load(p):
    return json.loads(p.read_bytes())


def beats(path):
    with path.open('rb') as f:
        for line in f:
            if line.startswith(b'{"event": "skill_observation",'):
                pre = line
            elif line.startswith(b'{"event": "step",'):
                yield pre, json.loads(line)
                pre = None


def snapshot(raw, e):
    state = json.loads(raw)['payload']['state']; allies = state['allies']
    hydras = [u['position'] for u in allies.values() if u['alive'] and u['type']=='Hydralisk']
    return dict(alive=sum(u['alive'] for u in allies.values()), hp=round(sum(u['health'] for u in allies.values()),2),
                hydra_gap=round(max(sorted(dist(p,q) for j,q in enumerate(hydras) if i!=j)[1] for i,p in enumerate(hydras)),2) if len(hydras)>=3 else None,
                visible=len(state['visible_enemies']), attacks=sum(a>=6 for a in e['actions']),
                micro=len(e['feedback']['micro_submissions']),reward=e['reward'])


def audit(task):
    batch,p = task; pair=p['pair_id']; arms=[RUNS/batch/pair/a/'runner' for a in ('v1','v2')]
    ep=[load(d/'episodes.json')[0] for d in arms]; rec=[load(d/'distill/episode-0001.json') for d in arms]
    for d,x,r in zip(arms,ep,rec):
        assert load(d/'manifest.json')['sha256']==p['manifest_sha256'] and r['seed']==p['seed']
        assert r['scenario_sha256']==x['scenario_sha256']==p['scenario_sha256']
        assert r['initial_state_sha256']==x['initial_state_sha256']==ep[0]['initial_state_sha256']
        assert len(r['decisions'])==x['steps']
    severe=[x['outcome']=='loss' and x['enemy_elimination_fraction']<=.5 for x in ep]
    outcome=[x['outcome'] for x in ep]; first=None; timeline=[]; triggers=[]; milestones={}
    lengths=[0,0]; unconfirmed=[0,0]
    for i, ab in enumerate(zip_longest(*(beats(d/'events.jsonl') for d in arms))):
        for j,item in enumerate(ab):
            if item is None: continue
            raw,e=item; dec=rec[j]['decisions'][i]; a=e['actions']; lengths[j]+=1
            assert dec['step']==i==e['step']-1 and dec['game_loop']==e['game_loop']
            assert a==[dec['final_actions'][f'u{k}'] for k in range(len(a))] and dec['committed']
            ok=dec['macro_receipt']=='accepted' and dec['micro_receipt'] in ('accepted','not_submitted'); unconfirmed[j]+=not ok
            if j and dec.get('selection_reason') in ('generated_opening_support','protect_hydra_support_gap'):
                triggers.append(dict(step=i,loop=e['game_loop'],reason=dec['selection_reason'],confirmed=ok,audit=dec['selector_audit']))
        if None in ab:
            if first is None: first=dict(kind='length',step=i)
            continue
        (u,e),(v,f)=ab; a,b=e['actions'],f['actions']; loop=e['game_loop']; assert loop==f['game_loop']
        same=u==v or json.loads(u)['payload']==json.loads(v)['payload']; micro=[z['feedback']['micro_submissions'] for z in (e,f)]
        for name,different in [('pre',not same),('macro',a!=b),('micro',micro[0]!=micro[1]),('reward',e['reward']!=f['reward'])]:
            if different and name not in milestones: milestones[name]=dict(step=i,loop=loop)
        if first is None:
            kind=next((k for k,t in [('pre',not same),('macro',a!=b),('micro',micro[0]!=micro[1]),('feedback',e['feedback']!=f['feedback']),('reward',e['reward']!=f['reward'])] if t),None)
            if kind:
                first=dict(kind=kind,step=i,loop=loop,same_pre=same,actions=[a,b],micro=micro,reason=rec[1]['decisions'][i].get('selection_reason'))
                if kind=='macro':
                    state=json.loads(u)['payload']['state']; first['changed']=[(f'u{k}',state['allies'][f'u{k}']['type'],x,y) for k,(x,y) in enumerate(zip(a,b)) if x!=y]
                    first['accepted']=all(z['feedback']['actors'][k]['submission']=='accepted' for k,*_ in first['changed'] for z in (e,f)); first['pre_sha256']=hashlib.sha256(u).hexdigest()
        if first and (triggers or severe[0]!=severe[1] or outcome[0]!=outcome[1]):
            timeline.append(dict(step=i,loop=loop,v1=snapshot(u,e),v2=snapshot(v,f)))
    assert lengths==[x['steps'] for x in ep]
    return dict(batch=batch,pair=pair,seed=p['seed'],roster=p['roster'],outcome=outcome,severe=severe,elim=[x['enemy_elimination_fraction'] for x in ep],first=first,triggers=triggers,milestones=milestones,unconfirmed=unconfirmed,timeline=timeline)


if __name__=='__main__':
    OUT.mkdir(exist_ok=True); summary={}
    with ProcessPoolExecutor(8) as pool, (OUT/'pairs.jsonl').open('w') as out:
        for batch in BATCHES:
            design=load(RUNS/batch/'design.json')['pairs']; assert len({p['pair_id'] for p in design})==len(design)==240
            rows=list(pool.map(audit,((batch,p) for p in design)))
            for r in rows: print(json.dumps(r,separators=(',',':')),file=out)
            summary[batch]=dict(first=Counter((r['first'] or {}).get('kind','identical') for r in rows),reasons=Counter(t['reason'] for r in rows for t in r['triggers']),unconfirmed=[sum(r['unconfirmed'][j] for r in rows) for j in range(2)],cases=[{k:r[k] for k in ('pair','outcome','severe','elim','milestones')} for r in rows if r['triggers'] or r['outcome'][0]!=r['outcome'][1] or r['severe'][0]!=r['severe'][1]])
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
