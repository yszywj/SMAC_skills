"""Describe archived paired contact trajectories. Conditional outcome groups are not causal labels."""
import json
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from math import dist
from pathlib import Path
from statistics import fmean

ROOT=Path(__file__).resolve().parents[3]/'runs'
P=ROOT/'zerg_v2_vs_swarm_random900_20261124_223951'
Q=ROOT/'zerg_router_old900_live_20261125_105505'


def one(job):
    row,route=job
    a,b=row['v2'],row['swarm'];g=(a['outcome'],b['outcome']);ll=g==('loss','loss')
    groups=['/'.join(g)]
    if ll and max(a['enemy_elimination_fraction'],b['enemy_elimination_fraction'])<=.5:groups.append('both_severe')
    if route['choice']=='swarm' and g in [('win','loss'),('loss','win')]:groups.append('routed_'+'/'.join(g))
    res={'id':row['pair_id'],'groups':groups}
    for arm,evt in [('v2','skill_observation'),('swarm','policy_observation')]:
        metrics={}; contact=None
        with (P/row['pair_id']/arm/'runner/events.jsonl').open() as f:
            for line in f:
                if not line.startswith('{"event": "'+evt+'"'):continue
                e=json.loads(line);s=e['payload']['state'];units=[u for u in s['allies'].values() if u['alive']]; loop=s['game_loop']
                def width():return max((dist(x['position'],y['position']) for x in units for y in units), default=0)
                if not metrics:metrics['start_span']=width()
                if s['visible_enemies'] and 'seen_span' not in metrics:metrics['seen_span']=width()
                by_target=Counter(en for u,rr in s['relations'].items() for en,r in rr.items() if r['smac_attack_available']); shooters=sum(any(r['smac_attack_available'] for r in rr.values()) for rr in s['relations'].values())
                if contact is None and shooters:contact=loop;metrics.update(first_loop=loop,first_shooters=shooters)
                if contact is not None and loop==contact+32:
                    metrics.update(common4=max(by_target.values(),default=0),alive4=len(units),shooters4=shooters);break
        res[arm]=metrics
    return res


if __name__=='__main__':
    read=lambda p:json.loads((p/'report.json').read_text())['paired_outcomes']
    routes={r['scenario_sha256']:r for r in read(Q)}; rows=read(P)
    with ProcessPoolExecutor(8) as pool: obs=list(pool.map(one,[(r,routes[r['scenario_sha256']]) for r in rows],chunksize=8))
    out={}
    for g in sorted({g for x in obs for g in x['groups']}):
        rs=[x for x in obs if g in x['groups']];out[g]={'n':len(rs)}
        for k in ['start_span','seen_span','first_shooters','common4','alive4','shooters4']:
            both=[r for r in rs if all(k in r[a] for a in ('v2','swarm'))];out[g][k]=[len(both)]+[round(fmean(r[a][k] for r in both),2) for a in ('v2','swarm')] if both else []
    print(json.dumps(out,indent=1));(Q/'contact_groups.json').write_text(json.dumps(out,indent=2)+'\n')
