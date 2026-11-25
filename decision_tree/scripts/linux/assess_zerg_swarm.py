"""Assess an independent controller as a whole policy and a source of complementary wins."""
import argparse
from collections import Counter, defaultdict
import json
from math import comb, isclose
from pathlib import Path
from random import Random
from statistics import fmean

ROOT = Path(__file__).resolve().parents[3]


def read(path):
    return json.loads(path.read_text())


def severe(arm):
    return arm['outcome']=='loss' and arm['enemy_elimination_fraction']<=.5


def metrics(rows):
    n=len(rows)
    if not n: return {'pairs':0}
    counts=Counter((p['v2']['outcome'],p['swarm']['outcome']) for p in rows)
    plus=sum(v for (a,b),v in counts.items() if a!='win' and b=='win')
    minus=sum(v for (a,b),v in counts.items() if a=='win' and b!='win')
    d=plus+minus; rng=Random(20261124)
    delta=[p['swarm']['reward']-p['v2']['reward'] for p in rows]
    boot=sorted(fmean(rng.choices(delta,k=n)) for _ in range(2000))
    return dict(pairs=n,v2_wins=sum(p['v2']['outcome']=='win' for p in rows),swarm_wins=sum(p['swarm']['outcome']=='win' for p in rows),
                gained=plus,lost=minus,loss_to_win=counts['loss','win'],win_to_loss=counts['win','loss'],
                posthoc_oracle_wins=sum(p['v2']['outcome']=='win' or p['swarm']['outcome']=='win' for p in rows),
                v2_severe=sum(severe(p['v2']) for p in rows),swarm_severe=sum(severe(p['swarm']) for p in rows),
                severe_to_win=sum(severe(p['v2']) and p['swarm']['outcome']=='win' for p in rows),
                win_to_severe=sum(p['v2']['outcome']=='win' and severe(p['swarm']) for p in rows),
                paired_win_exact_p=min(1.,2*sum(comb(d,j) for j in range(min(plus,minus)+1))/2**d),
                return_delta=fmean(delta),return_delta_ci95=[boot[49],boot[1949]],
                transitions={f'{a}_to_{b}':v for (a,b),v in sorted(counts.items())})


def assess(runs):
    old={}
    for label in ('fresh','regression'):
        for r in read(ROOT/'runs/zerg_route_training_r2'/label/'report.json')['paired_outcomes']:
            old[r['scenario_sha256']]=r
    allrows=[]; panels=[]; overlap=[];changed=[];seen=set();variants=set();modes=set()
    for run in runs:
        report=read(run/'report.json');design=read(run/'design.json');rows=report['paired_outcomes']
        if len(rows)!=len(design['pairs']):raise ValueError('incomplete report')
        variants.add((design['policies']['v2']['model_sha256'],design['sources']['decision_tree/zerg/radical_swarm.py']))
        is_replay=design['replay_batch'] is not None;modes.add(is_replay)
        for p in rows:
            k=p['scenario_sha256']
            if k in seen:raise ValueError('duplicate scenario')
            seen.add(k)
            if p['v2']['initial']!=p['swarm']['initial']:raise ValueError('pair initial mismatch')
            if k in old:
                q=old[k];v=q['v2'];u=p['v2']
                if v['initial']!=u['initial']:raise ValueError('historical initial state mismatch')
                if severe(v) and severe(q['alternative']):
                    if (u['outcome'],u['enemy_elimination_fraction'])==(v['outcome'],v['enemy_elimination_fraction']) and isclose(u['reward'],v['reward'],abs_tol=1e-8):overlap.append(p)
                    else:changed.append(dict(pair=p['pair_id'],scenario=k))
        allrows.extend(rows);panels.append(str(run))
    if len(variants)!=1 or len(modes)!=1:raise ValueError('do not pool different policies or mix replay and confirmation')
    strata=defaultdict(list)
    for r in allrows:
        h=int(r['roster'].rsplit('_',1)[1].split('hydralisk')[0]);strata['hydras_6plus' if h>=6 else 'hydras_0to5'].append(r)
    result=dict(panels=panels,scope='development_replay' if modes=={True} else 'new_random_pairs',all=metrics(allrows),
                by_hydra_count={k:metrics(v) for k,v in strata.items()})
    if overlap or changed:result.update(previous_v2_and_alt_both_severe_with_reproduced_v2=metrics(overlap),previous_severe_v2_endpoint_changed=changed)
    return result


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__);cli.add_argument('--run',type=Path,nargs='+',required=True);cli.add_argument('--out',type=Path)
    args=cli.parse_args();text=json.dumps(assess(args.run),indent=2,allow_nan=False)+'\n'
    if args.out:
        with args.out.open('x') as f:f.write(text)
    print(text)
