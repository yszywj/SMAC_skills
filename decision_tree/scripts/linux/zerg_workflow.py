"""One-time Zerg baseline setup and repeatable, cumulative one-step fork rounds."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from decision_tree.zerg.schema import digest


def read(path):
    return json.loads(Path(path).read_text())


def freeze(path, value):
    if path.exists():
        if read(path) != value:
            raise ValueError(f'conflicting frozen inputs: {path}')
        return
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def call(script, *args, capture=False):
    print(f'[{script}]', *args, flush=True)
    return subprocess.run([sys.executable, '-B', str(HERE / (script+'.py')), *map(str,args)],
                          cwd=ROOT, text=True, stdout=subprocess.PIPE if capture else None, check=True)


def baseline(work, fit):
    fit = fit.resolve()
    model, report, split = (read(fit/f) for f in ('model.json','report.json','split.json'))
    if (model['target'] != 'imitation' or model['scorer'] != 'joint_scorer' or
            report['model_sha256'] != digest(model) or not any(ep['partition']=='train' for ep in split.values())):
        raise ValueError(f'invalid frozen baseline: {fit}')
    freeze(work/'baseline.json', dict(fit=str(fit), model_sha256=digest(model), split_sha256=digest(split)))
    return fit


def setup(work, args):
    if args.fit:
        return baseline(work, args.fit)
    batches = [b.resolve() for b in args.batches] if args.batches else []
    if (work/'baseline.json').is_file() and not (work/'init_inputs.json').is_file():
        raise ValueError('baseline already adopted; use round N or start initialization in a different --work directory')
    freeze(work/'init_inputs.json', dict(collect=bool(args.collect), batches=list(map(str,batches))))
    if args.collect:
        for i, seed in enumerate((20261123, 20262123)):
            root = work/f'collection-{i+1}'
            if not root.exists():
                counts = ['--banelings',3,0,0,2,2,1] if i else []
                call('collect_zerg_shadow','--seed',seed,'--workers',args.workers,'--output-root',root,*counts)
            found = list(root.glob('zerg_shadow_*/design.json'))
            if len(found) != 1:
                raise ValueError(f'expected exactly one prior collection under {root}; refusing to repeat games')
            batches.append(found[0].parent)
    fit = work/'imitation'
    if not fit.exists():
        call('fit_zerg_shadow','--batches',*batches,'--out',fit)
    return baseline(work,fit)


def round_dir(work, n):
    return work/f'round-{n:03d}'


def round_inputs(work, n):
    if n < 1:
        raise ValueError('round number must be positive')
    previous = []
    for k in range(1,n):
        marker = round_dir(work,k)/'complete.json'
        if not marker.is_file():
            raise ValueError(f'missing completed round {k}; rounds must be consecutive')
        info = read(marker)
        if digest(read(info['plan'])) != info['plan_hash'] or digest(read(Path(info['data'])/'envelope.json')) != info['envelope_hash']:
            raise ValueError(f'inputs changed since completing round {k}')
        previous.append(info)
    return previous


def finish(folder, run, data, previous, existing_fit=None):
    plan = run/'plan.json'
    comparison = json.loads(call('compare_zerg_forks','--run',run,capture=True).stdout)
    if not comparison['planned_pairs'] or comparison['accepted_pairs'] != comparison['planned_pairs']:
        raise ValueError('not all frozen pairs have accepted engine receipts')
    freeze(folder/'comparison.json',comparison)
    if not data.exists():
        call('export_zerg_gains','--run',run,'--out',data)
    env = read(data/'envelope.json')
    if env['experiment_id'] != run.name or env['pairs_trainable'] != comparison['accepted_pairs']:
        raise ValueError('gain dataset does not belong to this completed run')
    result = folder/'review'
    if not result.exists():
        reuse = ['--review-existing',existing_fit] if existing_fit else []
        call('fit_zerg_gains','--data',*[v['data'] for v in previous],data,'--out',result,*reuse)
    for file in ('model.json','report.json','pair_facts.json'):
        if not (result/file).is_file():
            raise ValueError(f'incomplete offline review: {result/file}; no games were resubmitted')
    facts = read(result/'pair_facts.json')
    if len(facts['pairs']) != comparison['accepted_pairs'] + sum(v['accepted'] for v in previous):
        raise ValueError('review facts do not cover the cumulative accepted pairs')
    freeze(result/'llm_train_facts.json', dict(scope='training_pairs_only', model_id=facts['model_id'],
           pairs=[r for r in facts['pairs'] if r['partition']=='train']))
    info = dict(run=str(run), plan=str(plan), plan_hash=digest(read(plan)), data=str(data),
                envelope_hash=digest(env), accepted=comparison['accepted_pairs'], review=str(result))
    freeze(folder/'complete.json',info)
    print(f'Review: {result} (pair_facts.json, report.json, model.json)',flush=True)


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--work',type=Path,default=ROOT/'runs/zerg_workflow',help='persistent workflow directory')
    sub = cli.add_subparsers(dest='action',required=True)
    init = sub.add_parser('init',help='one-time baseline only; --collect explicitly starts 480 games')
    src = init.add_mutually_exclusive_group(required=True)
    src.add_argument('--collect',action='store_true'); src.add_argument('--batches',nargs='+',type=Path); src.add_argument('--fit',type=Path)
    init.add_argument('--workers',type=int,default=60)
    init.add_argument('--first-round',action='store_true',help='also execute the first 24 A/B pairs after setup')
    adopt = sub.add_parser('adopt',help='import completed pilot as round 1; never runs games or refits')
    adopt.add_argument('--run',type=Path,required=True); adopt.add_argument('--gain-fit',type=Path,required=True)
    turn = sub.add_parser('round',help='one new 24-pair round; reuse completed stages when repeating the same number')
    turn.add_argument('number',type=int); turn.add_argument('--workers',type=int,default=48)
    turn.add_argument('--seed',type=int); turn.add_argument('--plan-only',action='store_true',help='stop before launching SC2')
    sub.add_parser('status',help='show completed rounds without starting games')
    args = cli.parse_args(argv)
    work = args.work.resolve()
    try:
        if args.action == 'status':
            print(json.dumps([read(p) for p in sorted(work.glob('round-*/complete.json'))],indent=2)); return
        work.mkdir(parents=True,exist_ok=True)
        with (work/'workflow.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            if args.action == 'init':
                print('Frozen baseline:',setup(work,args))
                if not args.first_round:
                    return
                args.action, args.number, args.seed, args.plan_only = 'round', 1, None, False
                args.workers = min(args.workers,48)
            if args.action == 'adopt':
                run, gain_fit = args.run.resolve(), args.gain_fit.resolve()
                fit = baseline(work, ROOT / read(run/'plan.json')['fit'])
                n, data = 1, run/'gain_dataset_v1'
                inputs = dict(mode='adopt',run=str(run),gain_fit=str(gain_fit),fit=str(fit))
                if not data.is_dir() or not (gain_fit/'model.json').is_file():
                    raise ValueError('adopt requires a completed export and gain fit')
            else:
                if args.workers <= 0:
                    raise ValueError('--workers must be positive')
                fit = baseline(work,Path(read(work/'baseline.json')['fit']))
                n, gain_fit = args.number, None
                run = round_dir(work,n)/f'forks_round_{n:03d}'
                data = run/'gain_dataset_v1'
                inputs = dict(mode='round',fit=str(fit),seed=args.seed if args.seed is not None else 42+n)
            previous = round_inputs(work,n)
            folder = round_dir(work,n); folder.mkdir(exist_ok=True)
            if args.action == 'round' and (folder/'complete.json').exists():
                if args.seed is not None and args.seed != read(folder/'inputs.json').get('seed'):
                    raise ValueError('round already complete with different selection inputs')
                print('Already complete; no games run:',read(folder/'complete.json')['review']); return
            freeze(folder/'inputs.json',inputs)
            if (folder/'complete.json').exists():
                print('Already complete; no games run:',read(folder/'complete.json')['review']); return
            if args.action == 'round':
                plan = folder/'plan.json'
                if not plan.exists():
                    exclusions = [arg for item in previous for arg in ('--exclude-plan',item['plan'])]
                    call('plan_zerg_forks','--fit',fit,'--out',plan,'--seed',inputs['seed'],'--verified-round',*exclusions)
                if args.plan_only:
                    print('Frozen plan; no games started:',plan); return
                if not run.exists():
                    call('run_zerg_forks','--plan',plan,'--out',run,'--workers',args.workers)
                if read(plan) != read(run/'plan.json'):
                    raise ValueError('existing fork run does not match frozen round plan')
            finish(folder,run,data,previous,gain_fit)
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as e:
        cli.exit(1,f'Workflow stopped: {e}\n' + ((e.stdout or '') if isinstance(e,subprocess.CalledProcessError) else ''))


if __name__ == '__main__':
    main()
