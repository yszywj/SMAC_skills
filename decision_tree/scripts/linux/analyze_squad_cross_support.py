"""Compare observed cross-squad attack availability (not actual hits); labels only for analysis."""
from collections import defaultdict,Counter
from concurrent.futures import ProcessPoolExecutor
from statistics import mean
import json
from pathlib import Path
from decision_tree.zerg.squad_rendezvous import _squads

ROOT=Path(__file__).resolve().parents[3]/'runs'
OLD=ROOT/'zerg_router_old900_live_20261125_105505'
NEW=ROOT/'zerg_squad_adaptive_900_20261125_141633'
def trace(path):
    groups=None;first=None;m={};changed=False
    with path.open() as f:
        for line in f:
            if line.startswith('{"event": "squad_rendezvous_decision"'):
                d=json.loads(line); changed |= d['actions']!=d['v2_actions']
                if d['handoff'] and 'exit' not in m:m['exit']=d['reason'];m['exit_cross']=cross
            if not line.startswith('{"event": "skill_observation"'):continue
            s=json.loads(line)['payload']['state'];loop=s['game_loop'];ally=s['allies']
            if groups is None:groups={f'u{i}':j for j,g in enumerate(_squads({i:ally[f'u{i}']['position'] for i in range(10)})) for i in g}
            tg=defaultdict(set); active=set()
            for u,rel in s['relations'].items():
                for t,r in rel.items():
                    if r['smac_attack_available']:
                        tg[t].add(groups[u]);active.add(groups[u])
            cross=any(len(x)>=2 for x in tg.values());first=loop if first is None and active else first
            if first is None:continue
            k=(loop-first)//8
            if cross and 'cross' not in m:m['cross']=k
            if len(active)>=2 and 'second' not in m:m['second']=k
            if k==4:m.update(different_fronts4=len(active)>=2 and not cross,alive4=sum(u['alive'] for u in ally.values()))
    m.update(changed=changed);return m

def one(r):
    pid=r['pair_id'];a,b=[trace(x/pid/name/'runner/events.jsonl') for x,name in [(OLD,'online_router'),(NEW,'squad_rendezvous')]]
    return dict(id=pid,old=a,new=b,router=r['online_outcome'],new_out=r['outcome'],v2=r['original_v2'],strata=r['strata'])

if __name__=='__main__':
    rows=json.loads((NEW/'report.json').read_text())['paired_outcomes']
    with ProcessPoolExecutor(10) as pool:rows=list(pool.map(one,rows,chunksize=6))
    cuts={'all':rows,'gain':[r for r in rows if r['router']!='win' and r['new_out']=='win'],
          'lost':[r for r in rows if r['router']=='win' and r['new_out']!='win'],
          'both_severe':[r for r in rows if 'both_severe' in r['strata']],
          'both_win':[r for r in rows if 'both_win' in r['strata']],
          'rescued_both':[r for r in rows if 'both_loss' in r['strata'] and r['new_out']=='win']}
    for k,rs in list(cuts.items()):
        if k in ('gain','lost'): cuts[k+'_changed_vs_v2']=[r for r in rs if r['new']['changed'] and r['v2']==r['router']]
    out={}
    for k,rs in cuts.items():
        out[k]=dict(n=len(rs),changed=sum(r['new']['changed'] for r in rs),outcome_matches_v2=sum(r['new_out']==r['v2'] for r in rs),
          exits=Counter(r['new'].get('exit') for r in rs),same_squad_exit=sum(r['new'].get('exit')=='shared_target' and not r['new']['exit_cross'] for r in rs))
        for arm in ('old','new'):
            out[k][arm]=dict(second_by4=sum(r[arm].get('second',999)<=4 for r in rs),cross_by4=sum(r[arm].get('cross',999)<=4 for r in rs),
                             split_front4=sum(r[arm].get('different_fronts4',False) for r in rs),n4=sum('alive4' in r[arm] for r in rs))
    print(json.dumps(out,indent=1));(NEW/'cross_support_analysis.json').write_text(json.dumps({'summary':out,'pairs':rows},indent=1)+'\n')
