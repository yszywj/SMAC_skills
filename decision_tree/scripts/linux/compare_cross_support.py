"""Old-panel ablation. Archived labels are used only here, after both policies have played."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
from statistics import fmean

from decision_tree.scripts.linux.analyze_squad_cross_support import trace
from decision_tree.scripts.linux.summarize_squad_ablation import effect

ROOT=Path(__file__).resolve().parents[3]
PARENT=ROOT/'runs/zerg_squad_adaptive_900_20261125_141633'
PARENT_REPORT='32d20ec8f073f7a63949122b84a74d67fad9caf45ec5662a8270c03dcd929f11'


def cohorts(r):
    return ['all', *r['strata'], *(['previous_gain'] if r['online_outcome']!='win' and r['outcome']=='win' else []),
            *(['previous_lost'] if r['online_outcome']=='win' and r['outcome']!='win' else [])]


def summarize(rows):
    out={}
    for t in ('parent','guard','front'):
        contacts=[r[t+'_trace'] for r in rows]
        out[t]=dict(wins=sum(r[t]=='win' for r in rows),severe=sum(r[t+'_severe'] for r in rows),
            cross_target_within_four=sum(m.get('cross',999)<=4 for m in contacts),
            two_squads_engaged_within_four=sum(m.get('second',999)<=4 for m in contacts),
            separate_fronts_at_four=sum(m.get('different_fronts4',False) for m in contacts),observed_at_four=sum('alive4' in m for m in contacts))
    for a,b in [('parent','guard'),('parent','front'),('guard','front')]:
        out[a+'_to_'+b]=dict(**effect(rows,a,b),mean_return_change=fmean(r[b+'_reward']-r[a+'_reward'] for r in rows))
    return dict(pairs=len(rows),**out)


def read(path):return json.loads(path.read_text())


def traces(args):
    pid,roots=args
    return [trace(root/pid/'squad_rendezvous/runner/events.jsonl') for root in roots]


def compare(guard,front):
    roots=(PARENT,guard,front);ds=[read(d/'design.json') for d in roots];reports=[read(d/'report.json') for d in roots]
    if sha256((PARENT/'report.json').read_bytes()).hexdigest()!=PARENT_REPORT:raise ValueError('changed archived adaptive parent')
    if [rep['mode'] for rep in reports]!=['adaptive','guard','front']:raise ValueError('expected archived adaptive + guard + front')
    for d,rep,root in zip(ds,reports,roots):
        if sha256((root/'design.json').read_bytes()).hexdigest()!=rep['design_sha256']:raise ValueError('design digest differs')
    if ds[1]['sources']!=ds[2]['sources'] or ds[1]['pairs']!=ds[2]['pairs']:raise ValueError('not the same worker build / scenarios')
    ids=[x['pair_id'] for x in ds[1]['pairs']];tables=[{r['pair_id']:r for r in rep['paired_outcomes']} for rep in reports]
    if (not ids or len(set(ids))!=len(ids) or any(set(ids)!=t.keys() for t in tables[1:])
            or any(len(t)!=rep['pairs'] for t,rep in zip(tables,reports))):raise ValueError('incomplete panel')
    groups=defaultdict(list)
    with ProcessPoolExecutor(6) as pool:
        for pid,ts in zip(ids,pool.map(traces,[(pid,roots) for pid in ids],chunksize=6)):
            p=tables[0][pid];row={}
            for name,t,m in zip(('parent','guard','front'),tables,ts):
                r=t[pid]
                if any(r[key]!=p[key] for key in ('initial','manifest_sha256','scenario_sha256','seed','strata','online_outcome')):
                    raise ValueError(f'{pid}: inconsistent endpoints')
                row.update({name:r['outcome'],name+'_trace':m,name+'_reward':r['reward'],
                            name+'_severe':r['outcome']=='loss' and r['enemy_elimination_fraction']<=.5})
            for group in cohorts(p):groups[group].append(row)
    return {g:summarize(rows) for g,rows in groups.items()}


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--guard',type=Path,required=True);ap.add_argument('--front',type=Path,required=True)
    opts=ap.parse_args();print(json.dumps(compare(opts.guard,opts.front),indent=2,allow_nan=False))
