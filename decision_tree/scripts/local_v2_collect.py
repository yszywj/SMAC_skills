"""Resumable collection of every registered non-BASE effect of a local-v2 plan."""
import argparse
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import closing, contextmanager
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
from tempfile import TemporaryDirectory
import time

from decision_tree.local import study
from decision_tree.local.runtime import digest
from decision_tree.scripts.local_campaign_recover import atomic_json, terminate_owned
from decision_tree.scripts.local_campaign_validate import single_manifest, validation_lock
from skills.local_v2 import explore

MAX_SCENE_STEPS = 4
MAX_STEP_EFFECTS = 8
MAX_FAMILY_EFFECTS = 2
MAX_SCENE_BRANCHES = MAX_SCENE_STEPS * MAX_STEP_EFFECTS

RETRYABLE = ('ConnectError', 'ConnectionRefusedError', 'WebSocketConnectionClosed',
             'SC2 process', 'ConnectionResetError', 'transport connection', 'connection lost',
             'connection to sc2 lost', 'unable to restart sc2', 'watchdog timeout')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def checked_directory(path):
    path = Path(path).resolve()
    root = (study.ROOT/'decision_tree/artifacts').resolve()
    require(path.is_relative_to(root) and path != root, 'Collection output must be inside project artifacts')
    return path


def open_db(out, readonly=False):
    path = out/'coverage.sqlite3'
    db = sqlite3.connect(path.as_uri()+'?mode=ro', uri=True) if readonly else sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    db.execute('PRAGMA busy_timeout=10000')
    return db


def load_plan(directory):
    directory = directory.resolve()
    summary = study.read(directory/'summary.json')
    require(summary['kind']=='local_v2_exploration_plan' and summary['scope']==explore.SCOPE and
            summary['source_hashes']==explore.source_hashes(), 'Changed or invalid plan')
    require(explore.checksum(directory/'menus.jsonl')==summary['menus_sha256'], 'Changed registered menu')
    base = Path(summary['base'])
    require(all(explore.checksum(base/name)==summary['base_hashes'][name] for name in explore.FILES),
            'BASE evidence changed')
    entries, ranks, jobs = {}, Counter(), []
    with (directory/'menus.jsonl').open(encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            scene, step = row['scene_hash'], row['step']
            require((scene,step) not in entries and 'BASE' in row['effects'], 'Invalid step/effect menu')
            entries[scene,step] = {k:row[k] for k in ('seed','anchor_sha256','effects')}
            for effect, binding in sorted(row['effects'].items()):
                if effect == 'BASE':
                    continue
                require(binding['id'] in binding['aliases'], 'Invalid effect representative')
                jobs.append((ranks[scene],row['episode'],scene,row['seed'],step,effect,binding['id']))
                ranks[scene] += 1
    require(len(entries)==summary['counts']['decision_steps'] and len(jobs)==summary['counts']['nonbase_effects'],
            'Denominator differs from complete plan')
    return summary, entries, sorted(jobs)


def check_scene_budget(entries):
    """Reject the entire run before scheduling; all registered effects count on resume."""
    step_counts = Counter()
    branch_counts = Counter()
    for (scene, step), entry in entries.items():
        effects = [effect for key, effect in entry['effects'].items() if key != 'BASE']
        if not effects:
            continue
        label = f"Scene {scene}, step {step}"
        step_counts[scene] += 1
        branch_counts[scene] += len(effects)
        require(step_counts[scene] <= MAX_SCENE_STEPS,
                f"{label}: exceeds {MAX_SCENE_STEPS} intervention steps per scene")
        require(branch_counts[scene] <= MAX_SCENE_BRANCHES,
                f"{label}: exceeds {MAX_SCENE_BRANCHES} non-BASE branches per scene")
        require(len(effects) <= MAX_STEP_EFFECTS,
                f"{label}: {len(effects)} non-BASE effects exceed the limit of {MAX_STEP_EFFECTS}")
        families = Counter()
        for effect in effects:
            # One shared effect costs one game and counts for each family it represents.
            families.update({alias.split(':', 1)[0] for alias in effect['aliases']})
        for family, count in families.items():
            require(count <= MAX_FAMILY_EFFECTS,
                    f"{label}: {family} has {count} distinct effects; limit is {MAX_FAMILY_EFFECTS}")


def initialize(out, plan, summary, jobs, db):
    protocol={'kind':'local_v2_full_collection','plan':str(plan.resolve()),'base':summary['base'],
              'base_hashes':summary['base_hashes'],'source_hashes':summary['source_hashes'],
              'menus_sha256':summary['menus_sha256'],'scope':summary['scope'],'total':len(jobs),
              'collector_sha256':explore.checksum(Path(__file__)),
              'auditor_sha256':explore.checksum(Path(explore.__file__)),
              'ordering':'interleave scenes, then increasing decision step and effect hash'}
    study.save_once(out/'protocol.json',protocol)
    manifest = study.read(Path(summary['base'])/'manifest.json')
    (out/'scenes').mkdir(exist_ok=True)
    for s in manifest['episodes']:
        study.save_once(out/'scenes'/f"{s['sha256']}.json",single_manifest(manifest,s))
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('PRAGMA synchronous=FULL')
    with db:
        db.execute('''CREATE TABLE IF NOT EXISTS tasks(
            id INTEGER PRIMARY KEY, episode INTEGER, scene_hash TEXT, seed INTEGER, step INTEGER,
            effect TEXT, binding TEXT, state TEXT DEFAULT 'pending', attempts INTEGER DEFAULT 0,
            accepted_path TEXT, result_json TEXT, last_error TEXT, UNIQUE(scene_hash,step,effect))''')
        db.execute('CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT)')
        db.execute('CREATE INDEX IF NOT EXISTS pending_jobs ON tasks(state,id)')
        stamp = digest(protocol)
        old = db.execute("SELECT value FROM meta WHERE key='protocol'").fetchone()
        if old is None:
            db.executemany('INSERT INTO tasks(episode,scene_hash,seed,step,effect,binding) VALUES(?,?,?,?,?,?)',
                           (job[1:] for job in jobs))
            db.execute("INSERT INTO meta VALUES('protocol',?)",(stamp,))
        else:
            require(old['value']==stamp,'Ledger protocol differs from plan')
    validate_job_keys(db,jobs)


def validate_job_keys(db,jobs):
    recorded=db.execute('SELECT episode,scene_hash,seed,step,effect,binding FROM tasks ORDER BY id')
    require([tuple(r) for r in recorded]==[item[1:] for item in jobs],
            'Ledger tasks differ from complete frozen menu')


def status(db):
    counts=dict(db.execute('SELECT state,count(*) FROM tasks GROUP BY state'))
    return {'total':sum(counts.values()),'accepted':counts.get('accepted',0),'counts':counts,
            'attempts':db.execute('SELECT coalesce(sum(attempts),0) FROM tasks').fetchone()[0]}


def attempt_path(out,task):
    return out/'attempts'/f"{task['id']//1000:03d}"/f"{task['id']:06d}"/f"{task['attempts']:02d}"


def compressed_digest(path):
    h=hashlib.sha256()
    with gzip.open(path,'rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def archive_events(run, result, release_raw=True):
    """Publish only after validating both uncompressed and gzip-restored byte hashes."""
    raw = run/'events.jsonl'
    archive = run/'events.jsonl.gz'
    expected = result['run_hashes']['events.jsonl']
    if not archive.exists():
        require(explore.checksum(raw)==expected,'Raw event stream changed after audit')
        temp = run/'events.unpublished.gz'
        with raw.open('rb') as src, gzip.open(temp,'wb',compresslevel=1) as dst:
            shutil.copyfileobj(src,dst,1024*1024)
        require(compressed_digest(temp)==expected,'Invalid compressed evidence')
        temp.replace(archive)
    else:
        require(compressed_digest(archive)==expected,'Damaged existing archive')
    proof={'audit':result,'compressed_events_sha256':explore.checksum(archive)}
    atomic_json(run/'proof.json',proof)
    if release_raw and raw.exists():
        require(explore.checksum(raw)==expected,'Refusing to release changed raw events')
        raw.unlink()
    return proof


@contextmanager
def raw_view(path):
    if (path/'events.jsonl').exists():
        yield path
        return
    with TemporaryDirectory(prefix='local-v2-proof-') as temp:
        root=Path(temp)
        for name in explore.FILES[:-1]:
            shutil.copyfile(path/name,root/name)
        with gzip.open(path/'events.jsonl.gz','rb') as src,(root/'events.jsonl').open('xb') as dst:
            shutil.copyfileobj(src,dst,1024*1024)
        yield root


def audit_attempt(out,task,summary,entries,release_raw=True):
    run=attempt_path(out,task)/'run'
    require(study.read(run/'run.json')['local_v2']['source_hashes']==summary['source_hashes'],
            'Forced process used a different version than the frozen plan')
    with raw_view(run) as view:
        result=explore.audit_branch(view,summary,entries)
    require((result['scene_hash'],result['seed'],result['step'],result['effect'],result['binding'])==
            (task['scene_hash'],task['seed'],task['step'],task['effect'],task['binding']),
            'Audit result does not identify the scheduled job')
    archive_events(run,result,release_raw)
    return str(run.relative_to(out)),result


def launch(out,task,timeout,min_free):
    if shutil.disk_usage(out).free < min_free:
        raise RuntimeError('Collection disk reserve reached')
    attempt=attempt_path(out,task)
    attempt.mkdir(parents=True,exist_ok=False)
    run=attempt/'run'
    cmd=[sys.executable,'-B','-m','skills.local_v2.run','--manifest',
         str(out/'scenes'/f"{task['scene_hash']}.json"),'--output-dir',str(run),
         '--force-step',str(task['step']),'--force-binding',task['binding']]
    study.save_once(attempt/'command.json',cmd)
    kwargs={'creationflags':subprocess.CREATE_NO_WINDOW|subprocess.CREATE_NEW_PROCESS_GROUP} if os.name=='nt' else {'start_new_session':True}
    with (attempt/'console.log').open('xb') as log:
        process=subprocess.Popen(cmd,cwd=study.ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                                 env={**os.environ,'PYTHONUNBUFFERED':'1'},**kwargs)
        record={'pid':process.pid,'command_sha256':digest(cmd),'started_unix':time.time()}
        try:
            atomic_json(attempt/'process.json',record)
            try:
                code=process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                terminate_owned(process,cmd,record)
                raise RuntimeError(f'watchdog timeout after {timeout} seconds') from None
            if code:
                with (attempt/'console.log').open('rb') as f:
                    f.seek(max(0, f.seek(0,os.SEEK_END)-2500))
                    excerpt=f.read().decode('utf-8','replace')
                raise RuntimeError(f'Native runner exited {code}: '+excerpt)
        finally:
            if process.poll() is None:
                terminate_owned(process,cmd,record)
            atomic_json(attempt/'exited.json',{'returncode':process.returncode})


def still_running(pid):
    if os.name=='nt':
        import ctypes
        from ctypes import wintypes
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.OpenProcess.restype=wintypes.HANDLE
        handle=kernel.OpenProcess(0x1000,False,pid)
        if not handle:
            return ctypes.get_last_error()!=87  # ERROR_INVALID_PARAMETER for an absent PID.
        try:
            code=wintypes.DWORD()
            return not kernel.GetExitCodeProcess(wintypes.HANDLE(handle),ctypes.byref(code)) or code.value==259
        finally:
            kernel.CloseHandle(wintypes.HANDLE(handle))
    try:
        os.kill(pid,0)
    except ProcessLookupError:
        return False
    return True


def accept(db,task,path,result):
    with db:
        db.execute("UPDATE tasks SET state='accepted',accepted_path=?,result_json=?,last_error=NULL WHERE id=? AND state='running'",
                   (path,json.dumps(result,sort_keys=True),task['id']))


def recover(out,db,summary,entries):
    """Recover after interruption without ever killing a discovered/reused PID."""
    for row in list(db.execute("SELECT * FROM tasks WHERE state='running'")):
        task=dict(row);path=attempt_path(out,task)
        complete=(path/'run/summary.json').exists() and study.read(path/'run/summary.json').get('status')=='complete'
        unknown_process=False
        if (path/'command.json').exists() and not (path/'exited.json').exists():
            record=path/'process.json'
            unknown_process=not record.exists() or still_running(study.read(record)['pid'])
            require(not unknown_process or complete,
                    'Prior worker may still be alive; do not duplicate: '+str(path))
        if complete:
            dst,result=audit_attempt(out,task,summary,entries,release_raw=not unknown_process)
            accept(db,task,dst,result)
        else:
            with db:
                db.execute("UPDATE tasks SET state='failed',last_error='Interrupted before completed audit' WHERE id=?",(task['id'],))


def collect(plan,out,workers,timeout,max_games,reserve,retry_failed):
    out=checked_directory(out)
    require(1<=workers<=2 and timeout>=30 and reserve>=1 and (max_games is None or max_games>=0),'Invalid collection limits')
    require(not plan.resolve().is_relative_to(out) and out!=plan.resolve(),'Collection output overlaps the plan')
    markers = [root/'STOPPED_BY_USER.json' for root in {out.parent, plan.resolve().parent}]
    require(not any(path.exists() for path in markers), 'Collection stopped by user; choose a new bounded plan')
    summary,entries,jobs=load_plan(plan)
    check_scene_budget(entries)
    out.mkdir(parents=True,exist_ok=True)
    with validation_lock(out),closing(open_db(out)) as db:
        initialize(out,plan,summary,jobs,db)
        recover(out,db,summary,entries)
        if retry_failed:
            with db:
                db.execute("UPDATE tasks SET state='pending' WHERE state='failed'")
        else:
            require(not db.execute("SELECT 1 FROM tasks WHERE state='failed' LIMIT 1").fetchone(),
                    'Preserved failures require diagnosis before explicit --retry-failed')
        registered, active, failures, stop = 0, {}, 0, None
        min_free=int(reserve*1024**3)
        def work(task):
            require(explore.source_hashes()==summary['source_hashes'],'Execution sources changed mid-collection')
            launch(out,task,timeout,min_free)
            return audit_attempt(out,task,summary,entries)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            while True:
                while not stop and len(active)<workers and (max_games is None or registered<max_games):
                    if shutil.disk_usage(out).free<min_free:
                        stop='Disk reserve reached';break
                    row=db.execute("SELECT * FROM tasks WHERE state='pending' ORDER BY id LIMIT 1").fetchone()
                    if row is None:break
                    task=dict(row);task['attempts']+=1
                    with db:
                        db.execute("UPDATE tasks SET state='running',attempts=? WHERE id=?",(task['attempts'], task['id']))
                    active[pool.submit(work,task)]=task
                    registered+=1
                if not active:break
                done,_=wait(active,return_when=FIRST_COMPLETED)
                for future in done:
                    task=active.pop(future)
                    try:
                        path,result=future.result()
                        accept(db,task,path,result)
                        failures=0
                    except Exception as exc:
                        msg=str(exc)
                        retry=any(t in msg for t in RETRYABLE) and task['attempts']<3
                        with db:
                            db.execute("UPDATE tasks SET state=?,last_error=? WHERE id=?",
                                       ('pending' if retry else 'failed',msg,task['id']))
                        failures+=1
                        if not retry or failures>=3:
                            stop=msg
                    print(json.dumps({**status(db),'last_task':task['id'],'stop':stop}),flush=True)
        report={**status(db),'stopped_reason':stop,'complete':status(db)['accepted']==len(jobs)}
        if stop:
            raise RuntimeError('Collection stopped with evidence preserved: '+stop)
        return report


def verify(plan,out,task_id):
    with closing(open_db(out,True)) as db:
        summary,entries,jobs=load_plan(plan)
        validate_job_keys(db,jobs)
        protocol=study.read(out/'protocol.json')
        require(protocol['menus_sha256']==summary['menus_sha256'] and
                db.execute("SELECT value FROM meta WHERE key='protocol'").fetchone()[0]==digest(protocol),
                'Ledger/plan mismatch')
        sql="SELECT * FROM tasks WHERE state='accepted'"
        rows=db.execute(sql if task_id is None else sql+' AND id=?',() if task_id is None else (task_id,))
        count=0
        for row in rows:
            run=(out/row['accepted_path']).resolve()
            require(run==attempt_path(out,row)/'run' and checksum_proof(run,row,summary,entries),'Archived audit changed')
            count+=1
        require(task_id is None or count==1,'Requested task has no accepted measurement')
        return {**status(db),'reaudited':count}


def checksum_proof(run,row,summary,entries):
    proof=study.read(run/'proof.json')
    if (explore.checksum(run/'events.jsonl.gz')!=proof['compressed_events_sha256'] or
            study.read(run/'run.json')['local_v2']['source_hashes']!=summary['source_hashes']):
        return False
    with raw_view(run) as view:
        actual=explore.audit_branch(view,summary,entries)
        return (actual==proof['audit']==json.loads(row['result_json']) and
                (actual['scene_hash'],actual['step'],actual['effect'])==(row['scene_hash'],row['step'],row['effect']))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='cmd',required=True)
    run=sub.add_parser('run')
    run.add_argument('--workers',type=int,default=2)
    run.add_argument('--timeout',type=int,default=150)
    run.add_argument('--max-games',type=int)
    run.add_argument('--min-free-gb',type=float,default=12)
    run.add_argument('--retry-failed',action='store_true',help='retry preserved failures only after diagnosis')
    check=sub.add_parser('verify')
    check.add_argument('--task',type=int)
    monitor=sub.add_parser('status')
    for p in (run,check):
        p.add_argument('--plan',type=Path,required=True)
    for p in (run,check,monitor):
        p.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    out=checked_directory(args.output)
    if args.cmd=='run':
        result=collect(args.plan,out,args.workers,args.timeout,args.max_games,args.min_free_gb,args.retry_failed)
    elif args.cmd=='verify':
        result=verify(args.plan,out,args.task)
    else:
        with closing(open_db(out,True)) as db:
            result=status(db)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
