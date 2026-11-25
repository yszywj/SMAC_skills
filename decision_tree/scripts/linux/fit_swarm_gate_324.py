"""Audited scenario-split search for a sticky V2/RadicalSwarm controller switch."""
import argparse
import itertools
import json
from math import dist
from pathlib import Path

from decision_tree.scripts.linux.fit_zerg_route_gate import fold, normalize, safe, summarize
from decision_tree.scripts.linux.replay_swarm_gate import load
from decision_tree.zerg.policy_route_gate import CUTS

CANDIDATES = dict(CUTS, spread=(6,10,14,18), escort_gap=(3,5,7), seen=(0,3,6), banes=(0,1,2), n_lings=(2,5,7))


def samples(folder, *, excluded_micro=None):
    hashes,rows=load(folder)
    index={p['scenario_sha256']:p['pair_id'] for p in json.loads((folder/'design.json').read_text())['pairs']}
    out=[]
    for row in rows:
        if excluded_micro is not None and row['exclusion_reason']=='v2_micro_only' and not row['prefix_verified']:
            excluded_micro.append(row['scenario_sha256']);continue
        if not row['prefix_verified'] or not row['micro_verified']:raise ValueError('unverified prefix')
        r=normalize(row); pid=index[r['scenario']];step=row['anchor']['step']
        with (folder/pid/'v2/runner/events.jsonl').open() as f:
            st=next(e['payload']['state'] for line in f if line.startswith('{"event": "skill_observation",') and (e:=json.loads(line))['step']==step)
        units=[u for u in st['allies'].values() if u['alive']]; guns=[u for u in units if u['type']=='hydralisk']; escorts=[u for u in units if u['type']!='hydralisk']
        r['features'].update(spread=max((dist(a['position'],b['position']) for a,b in itertools.combinations(units,2)),default=0),
            escort_gap=sum(min(dist(a['position'],b['position']) for b in escorts) for a in guns)/len(guns) if escorts and guns else -1,
            seen=len(st['visible_enemies']),banes=sum(u['type']=='baneling' for u in units),n_lings=sum(u['type']=='zergling' for u in units))
        r.update(seed=row['seed'],panel=folder.name,pair_id=pid);out.append(r)
    return hashes,out


def select(rows,rule):
    return [] if not rule else [r for r in rows if all((r['features'][k]>v)==gt for k,v,gt in rule)]


def learn(rows):
    atoms=[(k,v,gt) for k,vs in CANDIDATES.items() for v in vs for gt in (False,True)]
    rules=[(a,) for a in atoms]+[(a,b) for a,b in itertools.combinations(atoms,2) if a[0]!=b[0]]
    best=None; best_score=None
    for rule in rules:
        s=summarize(select(rows,rule))
        if not safe(s,8,4):continue
        score=(s['rescued'],s['severe_to_win'],s['relieved_severe'],-len(rule),-s['n'])
        if best_score is None or score>best_score:best,best_score=rule,score
    return best


def fit(rows):
    for key in ('scenario','seed'):
        if len({r[key] for r in rows})!=len(rows):raise ValueError('duplicate '+key)
    bins=[[r for r in rows if fold(r['scenario'])==i] for i in range(5)]
    train=sum(bins[1:],[]); rule=learn(train); votes=[];folds=[]
    for i in range(1,5):
        cand=learn(sum((bins[j] for j in range(1,5) if j!=i),[])); v=select(bins[i],cand);votes.extend(v)
        folds.append(dict(fold=i,n=len(bins[i]),rule=cand,selected=summarize(v)))
    cv=summarize(votes); hold=summarize(select(bins[0],rule))
    passed=rule is not None and safe(cv,8,4) and sum(f['selected']['rescued']>0 for f in folds)>=2 and safe(hold,4,2)
    return dict(accepted=passed,tree=rule if passed else None,training_candidate=rule,training=summarize(select(train,rule)),
                train_pairs=len(train),cv=cv,folds=folds,holdout=hold,holdout_pairs=len(bins[0]),all_pairs=summarize(rows))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--panels',type=Path,nargs='+',required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();rows=[];hashes=None
    if args.out.exists():p.error('output directory must be new')
    for path in args.panels:
        h,rs=samples(path)
        if hashes and hashes!=h:raise ValueError('policy mismatch')
        rows.extend(rs);hashes=h
    result=fit(rows);args.out.mkdir(parents=True,exist_ok=False)
    for name,obj in [('report',result),('gate',dict(schema='swarm_sticky_conjunction_v1',default='v2',switch_to='swarm',
                          tree=result['tree'],comparison='(value > cut) == boolean',policy_sha256=hashes)),('cut_grid',CANDIDATES),('rows',rows)]:
        (args.out/(name+'.json')).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2))
