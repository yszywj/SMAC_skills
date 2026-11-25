"""Run outcome-blind paired development panels, then audit and fit a sticky controller gate."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[3]
PANELS={'regression':'zerg_v1_v2_regression_20261124_185519','fresh':'zerg_v1_v2_fresh_20261124_190338'}


def freeze(screen):
    panels={}
    for alias, name in PANELS.items():
        rows=[r for r in screen['rows'] if r['panel']==name]
        if len(rows)!=240 or len({r['pair'] for r in rows})!=240:raise ValueError(f'invalid original panel {name}')
        changed=[r['pair'] for r in rows if r['changes']]
        # These are checks for shared execution when the controllers make the same choices.
        # Do not read terminal scores, health, or labels to select them.
        controls=sorted((r['pair'] for r in rows if not r['changes']),
                        key=lambda p:hashlib.sha256(f'route-dev-A:{name}:{p}'.encode()).digest())[:8]
        panels[alias]=dict(source=name,expected_divergence=sorted(changed),control=controls,
                           ids=sorted(int(k.split('-')[1]) for k in changed+controls))
    return dict(scope='all reachable historical divergences plus eight outcome-blind controls per 240-task panel; not a population win-rate estimate',
                analysis_source=screen['source_sha256'],panels=panels)


def main():
    cli=argparse.ArgumentParser(description=__doc__);cli.add_argument('--out',type=Path,required=True)
    cli.add_argument('--workers',type=int,default=32);cli.add_argument('--plan-only',action='store_true')
    args=cli.parse_args();out=args.out.resolve();out.mkdir(parents=True,exist_ok=True)
    evidence=ROOT/'runs/zerg_route_alt_480.json'; design=freeze(json.loads(evidence.read_text()))
    path=out/'sampling.json'
    if path.exists():
        if json.loads(path.read_text())!=design:raise ValueError('selection inputs changed; choose a new output')
    else:
        with path.open('x') as f:json.dump(design,f,indent=2)
    if args.plan_only:return
    def call(module,*argv):
        subprocess.run([sys.executable,'-B','-m','decision_tree.scripts.linux.'+module,*map(str,argv)],cwd=ROOT,check=True)
    dirs=[]
    for alias,p in design['panels'].items():
        target=out/alias;dirs+=['--panel',target]
        if not target.exists():
            call('eval_zerg_route','--out',target,'--replay-batch',alias,'--pilot-ids',*p['ids'],'--workers',args.workers)
        if not (target/'report.json').exists():raise ValueError(f'{target}: incomplete run; refusing to silently replace games')
    if not (out/'gate.json').exists():
        call('fit_zerg_route_gate',*dirs,'--out',out/'gate.json','--report',out/'fit_report.json')
    print(f'Gate: {out}/gate.json; held-out metrics: {out}/fit_report.json',flush=True)


if __name__=='__main__':main()
