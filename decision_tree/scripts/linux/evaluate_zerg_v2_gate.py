"""A pre-result reporting gate, not a training or hyperparameter-selection script."""
import argparse
import json
import math
import random
from pathlib import Path
from statistics import fmean


def assess(rows, *, fresh, seed=2472, resamples=10000):
    if not rows: raise ValueError('cannot assess an empty comparison')
    deltas=[]; gains=losses=severe1=severe2=0
    for r in rows:
        a,b=r['v1'],r['v2']
        win1,win2=a['outcome']=='win',b['outcome']=='win'
        gains+=win2 and not win1; losses+=win1 and not win2
        severe1+=a['outcome']=='loss' and a['enemy_elimination_fraction']<=.5
        severe2+=b['outcome']=='loss' and b['enemy_elimination_fraction']<=.5
        deltas.append(b['reward']-a['reward'])
    n=len(rows); discord=gains+losses
    p=min(1., 2*sum(math.comb(discord,j) for j in range(min(gains,losses)+1))/2**discord)
    rng=random.Random(seed);boot=sorted(fmean(rng.choices(deltas,k=n)) for _ in range(resamples))
    ci=[boot[int(q*(len(boot)-1))] for q in (.025,.975)]
    prompts=[]
    if not fresh:prompts.append('diagnostic / regression panel, not confirmation')
    if n!=240:prompts.append('not a full 240-pair confirmation panel')
    if gains<=losses or p>=.05:prompts.append('no significant positive paired win difference')
    if ci[0]<=0:prompts.append('paired mean return not convincingly positive')
    if severe2>severe1:prompts.append('more severe losses')
    return dict(n=n,gained=gains,lost=losses,net=gains-losses,win_rate_change_percentage_points=100*(gains-losses)/n,
                paired_exact_two_sided_win_p=p,paired_mean_return_change=fmean(deltas),
                episode_paired_bootstrap_95pct_mean_return=ci,severe_v1=severe1,severe_v2=severe2,
                decision='research improvement; independently replicate before promotion' if not prompts else 'not established',
                reasons=prompts)


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__);cli.add_argument('--run',type=Path,required=True)
    args=cli.parse_args();report=json.loads((args.run/'report.json').read_text())
    fresh=report['replay_batch'] is None
    print(json.dumps(assess(report['paired_outcomes'],fresh=fresh),indent=2,allow_nan=False))
