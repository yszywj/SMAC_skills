"""Three live arms on one fresh, reserved native mirror panel."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import itertools
import json
from pathlib import Path
import secrets
import subprocess
import tempfile
import time

from decision_tree.scripts.linux import eval_zerg_squad_rendezvous as sq

ev=sq.ev
ARMS={'R1':'v2','R6':'guard','R7':'front'}
KIND='zerg_fresh_r1_r6_r7_v1'


def startup_disconnect(work):
    log=(work/'worker.log').read_text(errors='replace')
    if 'env.reset_record(scenario)' not in log or not any(s in log for s in (
            'pysc2.lib.remote_controller.ConnectError:', 'Address already in use',
            'WebSocketConnectionClosedException:', 'pysc2.lib.protocol.ConnectionError:')):
        return False
    events=work/'runner/events.jsonl'
    if events.exists():
        with events.open() as f:
            if any(json.loads(line).get('event') in ('episode_started','skill_observation','step','episode_completed') for line in f if line.strip()):return False
    return not sq.sw.completed(work)


def run_arm(out,work,cmd,env,fd,retries):
    for attempt in range(retries+1):
        if any(work.iterdir()):
            backup=Path(tempfile.mkdtemp(prefix='old-attempt-',dir=out));work.rename(backup/(work.parent.name+'-'+work.name));work.mkdir()
        try:
            with (work/'worker.log').open('x') as f:
                subprocess.run(cmd,env=env,cwd=ev.ROOT,stdout=f,stderr=subprocess.STDOUT,check=True,pass_fds=(fd,))
            return True
        except subprocess.CalledProcessError:
            if not startup_disconnect(work):raise
            if attempt==retries:
                print(f'{work}: startup failed after {attempt+1} attempts; continuing other arms',flush=True);return False
            delay=min(30,2**(attempt+1))+secrets.randbelow(2000)/1000
            print(f'{work}: SC2 startup connection failed; retry {attempt+1}/{retries} in {delay:.1f}s',flush=True)
            time.sleep(delay)


def main():
    ap=argparse.ArgumentParser(description=__doc__);g=ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--out',type=Path);g.add_argument('--resume',type=Path)
    ap.add_argument('--pairs',type=int);ap.add_argument('--workers',type=int,default=24);ap.add_argument('--dry-run',action='store_true')
    ap.add_argument('--startup-retries',type=int,default=3,choices=range(11),help='extra attempts per arm for pre-game SC2 connection failures')
    args=ap.parse_args();out=(args.out or args.resume).resolve();n=args.pairs if args.pairs is not None else 1000
    if not 1<=args.workers<=60 or not 1<=n<=10000 or (args.resume and args.pairs is not None):ap.error('invalid size, workers or resume options')
    parent,_,_=sq.baseline();original,cfg,versions,src=sq.runtime(parent)
    if args.out:
        policies={a:dict(parent['policies'][sq.router.ARM],trial_id=secrets.token_hex(16)) for a in ARMS}
        for name,mode in [('R6','guard'),('R7','front')]:
            ref=ev.read_json(ev.RUNS/f'zerg_cross_{mode}_900_20261125_150515'/'report.json')['controller']
            ev.require(ref['controller_sha256']==src[sq.CROSS_SOURCE] and ref['parent_controller_sha256']==src[sq.SRC], 'changed R6/R7')
        sq.sw.trees.freeze_panel(out,args.workers,None,args.dry_run,None,None,None,policies,src,cfg,versions,{},KIND,panel_size=n,expected_pairs=n)
    with sq.sw.run_lock(out) as fd:
        d=ev.read_json(out/'design.json');ev.require(d['kind']==KIND and d['arms']==list(ARMS) and len(d['pairs'])==d['expected_pairs'] and d['sources']==src and d['versions']==versions and ev.sha(out/'model.json')==ev.MODEL_HASH and ev.read_json(ev.HISTORY/f'{ev.digest(d)}.json')==d,'changed panel or code')
        ev.require(args.dry_run or not d['dry_run'],'dry-run panels cannot be played')
        rows={};pending=[]
        def audit(p,a):
            policy=d['policies'][a];manifest=ev.read_json(out/p['manifest']);mode=ARMS[a];w=out/p['pair_id']/a/'runner'
            if a=='R1':
                r=ev.audit_arm(out,p,'v2',manifest,original,d,policy,work_override=w.parent)
                ep,=ev.read_json(w/'episodes.json');ev.require('squad_rendezvous' not in ep,'R1 ran squad');r['enemy_elimination_fraction']=ep['enemy_elimination_fraction']
            else:
                sub=dict(d,mode=mode,policies={sq.ARM:policy});r=sq.audit(out,p,sub,original,None,None,a)
            rows[p['pair_id'],a]=r
        for i,p in enumerate(d['pairs'],1):
            sq.sw.trees.validate_pair(p,ev.read_json(out/p['manifest']),i,cfg)
            for a in ARMS:
                work=out/p['pair_id']/a
                if sq.sw.completed(work):audit(p,a)
                else: pending.append((p,a))
        print(f'{len(d["pairs"])} matched scenarios; {len(rows)} audited games, {len(pending)} pending; {out}',flush=True)
        if args.dry_run:return
        ev.require(not pending or not (out/'report.json').exists(),'partial run has a final report')
        def play(p,a):
            policy=d['policies'][a];cmd,env=ev.command(out,p,'v2',policy);work=out/p['pair_id']/a
            cmd[-1]=str(work/'runner');env['ZERG_SHADOW_DIR']=str(work/'shadow')
            if a!='R1':env.update(ZERG_SQUAD_MODE=ARMS[a],ZERG_SQUAD_SHA256=src[sq.SRC],ZERG_SQUAD_CROSS_SHA256=src[sq.CROSS_SOURCE])
            if not run_arm(out,work,cmd,env,fd,args.startup_retries):return False
            audit(p,a)
            return True
        with ThreadPoolExecutor(args.workers) as pool:
            jobs={pool.submit(play,p,a):(p['pair_id'],a) for p,a in pending}
            try:
                failed=[jobs[j] for j in as_completed(jobs) if not j.result()]
            except BaseException:
                for j in jobs:j.cancel()
                raise
        if failed:
            print('Unfinished arms: '+', '.join('/'.join(key) for key in failed),flush=True)
            raise SystemExit(f'{len(failed)} arms still need --resume; no partial win-rate report written')
        pairs=[dict(p,**{a:rows[p['pair_id'],a] for a in ARMS}) for p in d['pairs']]
        ev.require(all(len({p[a]['initial'] for a in ARMS})==1 for p in pairs),'different engine initial states')
        stats={a:dict(wins=sum(p[a]['outcome']=='win' for p in pairs),win_rate=sum(p[a]['outcome']=='win' for p in pairs)/len(pairs),severe_losses=sum(p[a]['outcome']=='loss' and p[a]['enemy_elimination_fraction']<=.5 for p in pairs)) for a in ARMS}
        flips={}
        for a,b in itertools.combinations(ARMS,2):
            t=Counter((p[a]['outcome'],p[b]['outcome']) for p in pairs);g=sum(v for (x,y),v in t.items() if x!='win' and y=='win');l=sum(v for (x,y),v in t.items() if x=='win' and y!='win')
            from math import comb
            discord=g+l
            flips[a+'_to_'+b]=dict(gained=g,lost=l,net=g-l,paired_win_exact_p=min(1.,2*sum(comb(discord,j) for j in range(min(g,l)+1))/2**discord),transitions={x+'_to_'+y:v for (x,y),v in t.items()})
        rep=dict(scenarios=len(pairs),games=len(rows),arms=stats,comparisons=flips,paired_outcomes=pairs)
        target=out/'report.json'
        if target.exists():ev.require(ev.read_json(target)==rep,'report differs')
        else:ev.write_json(target,rep)
        print(json.dumps({k:v for k,v in rep.items() if k!='paired_outcomes'}))


if __name__=='__main__':main()
