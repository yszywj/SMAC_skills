"""Confirm contact geometry symmetrically, independent of the V3 volley damage model."""
import argparse
from collections import Counter,defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
PANELS=('zerg_v1_v2_regression_20261124_185519','zerg_v1_v2_fresh_20261124_190338')


def first(x):
    run, p = x
    with (run/p['pair_id']/'v2/runner/events.jsonl').open() as f:
        for line in f:
            if not line.startswith('{"event": "skill_observation",'): continue
            ev=json.loads(line);s=ev['payload']['state']; rels=s['relations']
            if not any(r['smac_attack_available'] for rel in rels.values() for r in rel.values()): continue
            incoming={e for rel in rels.values() for e,r in rel.items() if r['in_enemy_weapon_range'] is True}
            outgoing={u for u,rel in rels.items() if s['allies'][u]['alive'] and any(r['in_weapon_range'] is True for r in rel.values())}
            ready={u for u in outgoing if s['allies'][u]['weapon_ready']}
            # Same observational basis for outgoing/incoming geometry; enemy readiness is unknown.
            return dict(group=run.name,pair=p['pair_id'],roster=p['roster'],loop=s['game_loop'],
                v2out=p['v2']['outcome'],severe_both=all(p[k]['outcome']=='loss' and p[k]['enemy_elimination_fraction']<=.5 for k in ('v1','v2')),
                ally_in_range=len(outgoing),ally_ready_in_range=len(ready),visible_enemy_in_range=len(incoming),
                ally_wpn_unknown=sum(r['in_weapon_range'] is None for rel in rels.values() for r in rel.values()),
                enemy_wpn_unknown=sum(r['in_enemy_weapon_range'] is None for rel in rels.values() for r in rel.values()))
    return dict(group=run.name,pair=p['pair_id'],roster=p['roster'],loop=None,v2out=p['v2']['outcome'],
                severe_both=all(p[k]['outcome']=='loss' and p[k]['enemy_elimination_fraction']<=.5 for k in ('v1','v2')))


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    jobs=[]
    for n in PANELS:
        run=ROOT/'runs'/n;report=json.loads((run/'report.json').read_text());jobs.extend((run,p) for p in report['paired_outcomes'])
    with ProcessPoolExecutor(6) as pool: rows=list(pool.map(first,jobs,chunksize=8))
    tables=defaultdict(Counter)
    for r in rows:
        for group in ['all',r['group'],r['roster']]:
            for kind in ['total',r['v2out']] + (['both_severe'] if r['severe_both'] else []):
                c=tables[group+'/'+kind];c['n']+=1
                if r['loop'] is None:continue
                c['observable']+=1
                c['enemy_count_exceeds_ally_by_geometry']+=r['visible_enemy_in_range']>r['ally_in_range']
                c['enemy_count_exceeds_ready_allies']+=r['visible_enemy_in_range']>r['ally_ready_in_range']
                c['has_unknown_enemy_ranges']+=r['enemy_wpn_unknown']>0
    out=dict(scope='first legal attack option, not necessarily first damage; visible weapons only; enemy readiness unknown',tables=tables,pairs=rows)
    args.out.parent.mkdir(exist_ok=True,parents=True)
    with args.out.open('x') as f:json.dump(out,f,indent=2)
    print(json.dumps({k:v for k,v in tables.items() if k.startswith('all/')},indent=2))


if __name__=='__main__':main()
