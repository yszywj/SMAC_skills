"""Prepare a frozen, independent C feasibility/screen plan; optionally launch its three-arm games."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--out', required=True, type=Path, help='dedicated screen directory, reused on restart')
    cli.add_argument('--workers', type=int, default=48)
    cli.add_argument('--play', action='store_true', help='actually launch the frozen games; default is offline')
    cli.add_argument('--discovery-smoke', action='store_true', help='known trigger, exactly three games, not independent evidence')
    args = cli.parse_args()
    if args.workers < 1: cli.error('--workers must be positive')
    out = args.out.resolve(); out.mkdir(parents=True,exist_ok=True)
    plan = out/'plan.json'; games = out/'games'
    def run(script,*argv,**kw):
        return subprocess.run([sys.executable,'-B',str(HERE/(script+'.py')),*map(str,argv)],cwd=ROOT,check=True,**kw)
    if not plan.is_file(): run('plan_zerg_abc','--out',plan,*(['--discovery-smoke'] if args.discovery_smoke else []))
    frozen = json.loads(plan.read_text())
    if (frozen.get('abc_protocol') != 'zerg_abc_v1' or not frozen.get('pairs') or
            (frozen.get('stage')=='discovery_anchor_engine_smoke_only') != args.discovery_smoke):
        raise ValueError('not a frozen one-step ABC plan')
    if not args.play:
        dry = out/'preflight'
        if not dry.exists(): run('run_zerg_abc','--plan',plan,'--out',dry,'--dry-run')
        elif json.loads((dry/'plan.json').read_text()) != frozen: raise ValueError('preflight does not match plan')
        print(f"Offline only: {len(frozen['pairs'])} anchors, {3*len(frozen['pairs'])} planned games.")
        return
    if not games.exists(): run('run_zerg_abc','--plan',plan,'--out',games,'--workers',args.workers)
    if json.loads((games/'plan.json').read_text()) != frozen: raise ValueError('game plan differs; refusing to summarize')
    # Preserve an incomplete triplet report too, without treating it as success.
    result = subprocess.run([sys.executable,'-B',str(HERE/'run_zerg_abc.py'),'--compare',str(games)],
                            cwd=ROOT,text=True,stdout=subprocess.PIPE)
    report = json.loads(result.stdout)
    target = out/'comparison.json'; tmp = target.with_suffix('.tmp')
    tmp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n'); tmp.replace(target)
    print(f'ABC report: {target}')
    result.check_returncode()


if __name__ == '__main__':
    main()
