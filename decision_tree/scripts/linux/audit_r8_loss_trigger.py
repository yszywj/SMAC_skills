"""Audit whether a first friendly loss is an actionable trigger; never infer an explosion from death."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

RUN=Path(__file__).resolve().parents[3]/'runs/zerg_R1_R6_R7_random1000_20261125_155323'


def check(r):
    previous={};first=None;contact=None;anchor=None;end=0
    with (RUN/r['pair_id']/'R6/runner/events.jsonl').open() as f:
        for line in f:
            if not line.startswith('{"event": "skill_observation",'):continue
            e=json.loads(line);step=e['step'];s=e['payload']['state'];end=step
            live={u:v['type'].lower() for u,v in s['allies'].items() if v['alive']}
            vanished=set(previous)-set(live)
            if contact is None and any(v['smac_attack_available'] for rel in s['relations'].values() for v in rel.values()):contact=step
            if vanished:
                types=[previous[u] for u in vanished]
                if first is None:first=(step,sorted(types))
                if anchor is None and any(t!='baneling' for t in types):anchor=step
            previous=live
    return dict(pair=r['pair_id'],v2=r['R6']['outcome'],r7=r['R7']['outcome'],first=first,nonbane=anchor,contact=contact,last=end,
                group='severe' if all(r[k]['outcome']=='loss' and r[k]['enemy_elimination_fraction']<=.5 for k in ('R6','R7')) else
                'near' if all(r[k]['outcome']=='loss' and r[k]['enemy_elimination_fraction']>=.8 for k in ('R6','R7')) else 'other')


def summarize(rs):
    types=Counter('+'.join(sorted(set(r['first'][1]))) if r['first'] else 'unobserved' for r in rs)
    return dict(n=len(rs),first_loss_types=types,nonbane_with_4_more_observations=sum(r['nonbane'] is not None and r['last']-r['nonbane']>=4 for r in rs),
                nonbane_no_later_than_first_legal_attack=sum(r['nonbane'] is not None and (r['contact'] is None or r['nonbane']<=r['contact']) for r in rs))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    with ProcessPoolExecutor(8) as pool: rows=list(pool.map(check,json.loads((RUN/'report.json').read_text())['paired_outcomes'],chunksize=10))
    tables={k:summarize([r for r in rows if r['group']==k]) for k in ('severe','near')}
    tables['R6_won']=summarize([r for r in rows if r['v2']=='win'])
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('x') as f:json.dump(dict(tables=tables,rows=rows),f,indent=1)
    print(json.dumps(tables,indent=1))
