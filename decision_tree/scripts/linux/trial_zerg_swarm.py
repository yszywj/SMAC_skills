"""Frozen comparison: old enriched scenarios and a distinct fresh mirror panel; do not pool."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from decision_tree.scripts.linux.assess_zerg_swarm import assess


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True,type=Path)
    p.add_argument('--workers',type=int,default=32);p.add_argument('--stage',choices=('history','fresh','both'),default='both')
    args=p.parse_args();out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    old=json.loads((ROOT/'runs/zerg_route_training_r2/sampling.json').read_text())['panels']
    for sample, jobs in [('history', [(label,['--replay-batch',label,'--pilot-ids',*spec['ids']]) for label,spec in old.items()]),
                         ('fresh',[('confirmation',[])])]:
        if args.stage not in (sample,'both'):continue
        completed=[]
        for label,params in jobs:
            dest=out/label
            if not dest.exists():
                subprocess.run([sys.executable,'-B','-m','decision_tree.scripts.linux.eval_zerg_swarm','--out',str(dest),
                                '--workers',str(args.workers),*map(str,params)],cwd=ROOT,check=True)
            if not (dest/'report.json').is_file():raise ValueError(f'incomplete panel; not silently repeating {dest}')
            completed.append(dest)
        report=assess(completed); target=out/(sample+'_summary.json')
        if target.exists():
            if json.loads(target.read_text())!=report:raise ValueError('changed results')
        else:
            with target.open('x') as f:json.dump(report,f,indent=2,allow_nan=False)
        print(sample, report['all'],flush=True)


if __name__=='__main__':main()
