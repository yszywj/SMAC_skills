"""Paired win-count exploratory router, with a sealed scenario-sorted holdout."""
import argparse
import hashlib
import json
from itertools import combinations, product
from pathlib import Path

from decision_tree.scripts.linux.fit_swarm_gate_324 import CANDIDATES, samples


def learn(rows):
    full=(1<<len(rows))-1
    gain=sum(1<<i for i,r in enumerate(rows) if r['transition'] in ('loss_to_win','draw_to_win'))
    loss=sum(1<<i for i,r in enumerate(rows) if r['dropped'])
    options={}
    for k, cuts in CANDIDATES.items():
        uniq={}
        for t in cuts:
            gt=sum(1<<i for i,r in enumerate(rows) if r['features'][k]>t)
            for op,m in [('>',gt),('<=',full^gt)]:
                if m!=full and m.bit_count()>=8:uniq.setdefault(m,(k,op,t))
        if uniq:options[k]=list(uniq.items())
    patterns={}
    for size in range(1,5):
        for keys in combinations(options,size):
            for terms in product(*(options[k] for k in keys)):
                m=full
                for mask,_ in terms:m &= mask
                if m.bit_count()>=8 and m&gain and m not in patterns:patterns[m]=[c for _,c in terms]
    remaining=full;rules=[]
    for _ in range(2):
        best=None;score=(0,)
        for mask,rule in patterns.items():
            m=mask&remaining; wins=(m&gain).bit_count(); losses=(m&loss).bit_count()
            key=(wins-losses,-len(rule),-losses,-m.bit_count())
            if m.bit_count()>=8 and wins>losses and key>score:score=key;best=mask
        if best is None:break
        rules.append(patterns[best]);remaining &= ~best
    return rules


def use(rulelist,r):
    return any(all((r['features'][k]>t)==(op=='>') for k,op,t in rule) for rule in rulelist)


def metrics(picks):
    parent=sum(r['transition'].startswith('win_') for r,c in picks)
    chosen=[r for r,c in picks if c];gained=sum(r['transition'] in ('loss_to_win','draw_to_win') for r in chosen);lost=sum(r['dropped'] for r in chosen)
    return dict(pairs=len(picks),selected=len(chosen),baseline_wins=parent,router_wins=parent+gained-lost,gained=gained,lost=lost,
                relieved_severe=sum(r['relieved_severe'] for r in chosen),new_severe=sum(r['new_severe'] for r in chosen))


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--panel',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    if args.out.exists():ap.error('out must not exist')
    hashes,rows=samples(args.panel)
    if len(rows)!=900 or any(len({r[k] for r in rows})!=900 for k in ('scenario','seed')):raise ValueError('expected 900 unique audited pairs')
    rows.sort(key=lambda r:hashlib.sha256(('swarm-explore-v1:'+r['scenario']).encode()).digest())
    dev,test=rows[180:],rows[:180];rules=learn(dev)
    output=args.out;output.mkdir(parents=True)
    def save(name,obj):(output/name).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
    save('gate.json',dict(schema='swarm_first_macro_exploration_v1',default='v2',when_any_conjunction_matches='RadicalSwarm',rules=rules,policy_hashes=hashes))
    save('split.json',dict(development=[r['scenario'] for r in dev],holdout=[r['scenario'] for r in test]))
    cv=[]; folds=[]
    for i in range(5):
        rr=learn([r for j,r in enumerate(dev) if j%5!=i]);p=[(r,use(rr,r)) for r in dev[i::5]];cv+=p;folds.append(metrics(p))
    held=[(r,use(rules,r)) for r in test]
    report=dict(rule=rules,development=metrics([(r,use(rules,r)) for r in dev]),cv=metrics(cv),folds=folds,holdout=metrics(held))
    save('holdout_predictions.json',[dict(scenario=r['scenario'],choice='swarm' if c else 'v2',paired_outcome=r['transition']) for r,c in held]);save('report.json',report);print(json.dumps(report,indent=2))


if __name__=='__main__':main()
