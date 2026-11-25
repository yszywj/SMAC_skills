"""Live paired full v2 vs independent macro-only RadicalSwarm, from reset to terminal.

With historical RSI Python: -m decision_tree.scripts.linux.eval_zerg_swarm
--out runs/NAME --replay-batch regression --pilot-ids 1 11 --workers 2.
Omit replay options for 240 new pairs. --dry-run freezes without starting SC2.
--compare DIR re-audits played runs. --repeat-design DIR repeats a fresh draw.
--resume DIR [--workers N] continues a stopped run; add --resume-check to audit only.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import fcntl
import json
import math
import os
from pathlib import Path
import secrets
import subprocess
import sys
import tempfile

from decision_tree.scripts.linux import eval_zerg_v2_v3 as old

ev, trees = old.ev, old.trees
SRC = 'decision_tree/zerg/radical_swarm.py'
KIND = 'full_v2_vs_radical_swarm_v1'
ARMS = ['v2', 'swarm']
PROTOCOLS = {'v2': ev.PROTOCOL, 'swarm': dict(method='radical-swarm', skill_version=None,
                                              execution_mode='micro8-off')}
MICRO = 'macro-only; one decision per 8 loops; no interior orders'


def prepare(out, workers=2, batch=None, ids=None, dry=False, repeat=None, pairs=None):
    ev.require(sys.executable == ev.RSI_PYTHON, f'use historical interpreter {ev.RSI_PYTHON}')
    archives = dict(old.ARCHIVES)
    batch = old.ALIASES.get(batch, batch)
    ev.require(not out.exists() and 1 <= workers <= 60 and (not batch or batch in archives),
               'invalid output (must be new), workers (1..60) or batch')
    ev.require(pairs is None or (not batch and repeat is None and 1 <= pairs <= 10000), '--pairs is for fresh panels (1..10000)')
    panel_size = 240 if pairs is None else pairs
    if repeat is not None:
        batch_dir = Path(repeat).resolve()
        prev = ev.read_json(batch_dir/'design.json')
        ev.require(not batch and prev['kind'] == KIND and prev['native_random_mirror'] is True
                   and len(prev['pairs']) == prev.get('expected_panel_size',240) and ev.read_json(ev.HISTORY/f'{ev.digest(prev)}.json') == prev,
                   '--repeat-design requires an original, frozen fresh draw')
        panel_size = len(prev['pairs'])
        batch = str(batch_dir)
        archives[batch] = ev.sha(batch_dir/'design.json')
    ev.require(ids is None or batch and 0 < len(set(ids)) == len(ids) <= panel_size
               and all(type(i) is int and 1 <= i <= panel_size for i in ids), 'invalid replay pair IDs')
    from decision_tree.zerg.radical_swarm import RadicalSwarm
    ev.require(callable(RadicalSwarm().choose), 'missing independent controller')
    original = ev.reference()
    sources = ev.frozen_sources(original)
    trial = secrets.token_hex(16)
    control = old.policies(sources, ev.model_contract(ev.MODEL, original), trial)['v2']
    policies = dict(v2=control, swarm=dict(controller_sha256=sources[SRC], trial_id=trial, protocol=MICRO))
    d = trees.freeze_panel(out, workers, None, dry, batch, None, ids, policies, sources,
                          *ev.historical_config(), archives, KIND, trial_id=trial, arm_protocols=PROTOCOLS,
                          panel_size=panel_size, expected_panel_size=panel_size, frozen_v2_design_sha256=old.ARCHIVES[old.FRESH])
    (out/'radical_swarm.py').write_bytes((ev.ROOT/SRC).read_bytes())
    ev.require(ev.sha(out/'radical_swarm.py') == sources[SRC], 'swarm changed during freeze')
    return d


def command(out, p, arm, policy):
    if arm == 'v2':
        return ev.command(out, p, arm, policy)
    ev.require(arm == 'swarm', 'unknown arm')
    env = ev.worker_environment()
    env.update(ZERG_SWARM_SHA256=policy['controller_sha256'], ZERG_SWARM_TRIAL_ID=policy['trial_id'])
    return ([ev.RSI_PYTHON, '-B', '-u', str(ev.ADAPTER/'run.py'), 'run', '--method', 'radical-swarm',
             '--execution-mode', 'micro8-off', '--manifest', str(out/p['manifest']),
             '--output-dir', str(out/p['pair_id']/arm/'runner')], env)


def play(out, p, arm, policy, lock_fd=None):
    cmd, env = command(out, p, arm, policy)
    work = out/p['pair_id']/arm
    with (work/'worker.log').open('x') as log:
        subprocess.run(cmd, cwd=ev.ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True,
                       pass_fds=() if lock_fd is None else (lock_fd,))


def check_macro(actions, masks, feedback, loop):
    ev.require(len(actions) == len(masks) == 10 and feedback['from_game_loop'] == loop
               and feedback['to_game_loop'] == loop + 8 and feedback['response_mapping'] == 'exact'
               and feedback['micro_submissions'] == [] and set(feedback['actors']) == {f'u{i}' for i in range(10)},
               'wrong macro interval, micro submission or actor coverage')
    indices = []
    for i, (a, mask) in enumerate(zip(actions, masks)):
        actor = feedback['actors'][f'u{i}']
        ev.require(len(mask) == 16 and all(type(v) is int and v in (0,1) for v in mask) and
                   type(a) is int and 0 <= a < 16 and mask[a] == 1 and actor['requested_action_id'] == str(a),
                   'illegal or different executed action')
        if a == 0:
            ev.require(mask == [1]+[0]*15 and actor['submission'] == 'dead_noop' and
                       actor['command'] is actor['response'] is None, 'invalid dead noop acknowledgement')
        else:
            ev.require(actor['submission'] == 'accepted' and actor['command'] and actor['response']['code'] == 1,
                       'missing/rejected macro command')
            indices.append(actor['command_index'])
    ev.require(sorted(indices) == list(range(len(indices))) and feedback['raw_response_results'] == [1]*len(indices),
               'raw engine acknowledgements do not cover commands one-to-one')
    return len(indices)


def swarm_arm(work, p, m, d, original):
    runner = work/'runner'
    meta, summary = [ev.read_json(runner/f'{name}.json') for name in ('run','summary')]
    ep, = ev.read_json(runner/'episodes.json')
    score = ev.read_json(runner/'score_details/episode-0001.json')
    ev.require(ev.read_json(runner/'manifest.json') == m and all(meta[k] == original[k] for k in (
        'macro_loops','engine_step_loops','native_teacher_control','observation_protocol','map_file',
        'damage_by_type_enabled','env_args','versions')) and meta['manifest_sha256'] == m['sha256'] and
        all(meta[k] == v for k,v in PROTOCOLS['swarm'].items()) and meta['swarm'] == d['policies']['swarm'] and
        meta['skills'] == 'off' and meta['rule_controller_executed'] is False and not meta['skill_source_hashes'] and
        not any(k in meta for k in ('zerg_shadow','zerg_distill')), 'incorrect swarm protocol')
    required = (set(original['source_hashes']) - set(original['skill_source_hashes'])) | {SRC} | {
        k for k in d['sources'] if k.startswith('sc2_skill_adapter/bridge/')}
    ev.require(meta['source_hashes'] == {k:d['sources'][k] for k in required}, 'swarm source mismatch')
    steps = ep['steps']
    ev.require(summary['status'] == 'complete' and summary['completed_episodes'] == 1 and
               ep['episode'] == 1 and type(steps) is int and steps > 0 and math.isfinite(ep['reward']) and
               ep['scenario_sha256'] == score['scenario_sha256'] == p['scenario_sha256'] and
               score['manifest_sha256'] == m['sha256'] and score['sc2_seed'] == p['seed'] and
               score['game_version'] == '4.10.0.75689' and score['steps'] == steps and
               score['outcome'] == ep['outcome'] in ('win','loss','draw','timeout') and
               0 <= ep['info']['dead_enemies'] <= 10 and ep['enemy_elimination_fraction'] == ep['info']['dead_enemies']/10,
               'incomplete swarm terminal or wrong scenario/build')
    empty_windows = dict(mode='micro8-off',macro_loops=8,engine_step_loops=1,windows=steps,authorizations=0,states={})
    ev.require(ep['micro_trigger'] == summary['micro_trigger_totals'] == empty_windows, 'unexpected swarm micro authorizations')
    initial_loop = score['initial']['game_loop']
    trace, n_commands, total = [], 0, 0.0
    started = completed = 0
    obs = decision = window = None
    with (runner/'events.jsonl').open() as stream:
        for line in stream:
            e = json.loads(line)
            k = e['event']
            i, loop = len(trace), initial_loop+8*len(trace)
            if k == 'episode_started':
                started += 1
                ev.require(e['initial_state_sha256'] == ep['initial_state_sha256'] and e['game_loop'] == initial_loop
                           and e['scenario_sha256'] == p['scenario_sha256'] and e['game_version'] == score['game_version'], 'incorrect reset')
            elif k == 'policy_observation':
                ev.require(started == 1 and obs is None and e['episode'] == 1 and e['step'] == i and
                           e['payload']['state']['game_loop'] == loop, 'missing or duplicate observation')
                obs = ev.digest(e['payload'])
            elif k == 'swarm_decision':
                ev.require(obs is not None and decision is None and e['episode'] == 1 and e['step'] == i
                           and e['game_loop'] == loop, 'missing or duplicate swarm decision')
                decision = e
            elif k == 'micro_window_closed':
                ev.require(decision is not None and window is None and e['macro_step'] == i and
                           e['issued_loop'] == loop and e['closed_loop'] == loop+8 and e['interior_checks'] == 7 and e['authorizations'] == 0
                           and e['records'] == [], 'unexpected interior action')
                window = e
            elif k == 'step':
                ev.require(window is not None and e['episode'] == 1 and e['step'] == i+1 and e['game_loop'] == loop
                           and e['actions'] == decision['decision']['actions'] and e['terminated'] is (i+1 == steps),
                           'step differs from independent decision or terminal')
                n_commands += check_macro(e['actions'], decision['avail_actions'], e['feedback'], loop)
                trace.append((obs, e['actions']))
                total += e['reward']
                obs = decision = window = None
            elif k == 'episode_completed':
                completed += 1
                ev.require(len(trace) == steps and all(e[x] == v for x,v in ep.items()), 'missing terminal evidence')
            else:
                ev.require(k == 'unit_balance_verified', f'unexpected swarm event {k}')
    ev.require(started == completed == 1 and len(trace) == steps and obs is None and total == ep['reward'] and
               score['final']['game_loop'] == initial_loop+8*steps, 'incomplete swarm action coverage')
    return dict(outcome=ep['outcome'],reward=total,steps=steps,initial=ep['initial_state_sha256'],
                enemy_elimination_fraction=ep['enemy_elimination_fraction'],controller_steps=len(trace),
                accepted_actor_commands=n_commands), trace


def prefix(v2_runner, swarm):
    """Compare actual orders only until observations or submitted commands diverge."""
    i = 0
    obs = None
    with (v2_runner/'events.jsonl').open() as stream:
        for line in stream:
            if line.startswith('{"event": "skill_observation",'):
                obs = ev.digest(json.loads(line)['payload'])
            elif line.startswith('{"event": "step",'):
                if i >= len(swarm) or obs != swarm[i][0]:
                    return dict(status='unmatched_observation', step=i)
                step = json.loads(line)
                if step['actions'] != swarm[i][1]:
                    return dict(status='macro', step=i, v2=step['actions'], swarm=swarm[i][1])
                if step['feedback']['micro_submissions']:
                    return dict(status='v2_micro_only', step=i)
                i += 1
                obs = None
    return dict(status='identical' if i == len(swarm) else 'unmatched_observation', step=i)


def aggregate(rows):
    n = len(rows)
    def severe(r):
        return r['outcome'] == 'loss' and r['enemy_elimination_fraction'] <= .5
    stats = {a:dict(outcomes={k:sum(r[a]['outcome'] == k for r in rows) for k in ('win','loss','draw','timeout')},
                    win_rate=sum(r[a]['outcome'] == 'win' for r in rows)/n,
                    severe_losses=sum(severe(r[a]) for r in rows),
                    mean_return=sum(r[a]['reward'] for r in rows)/n) for a in ARMS}
    transitions = Counter(f"{r['v2']['outcome']}_to_{r['swarm']['outcome']}" for r in rows)
    diffs = Counter(r['first_order_difference']['status'] for r in rows)
    checked = sum(r['swarm']['controller_steps'] for r in rows)
    steps = sum(r['swarm']['steps'] for r in rows)
    return dict(pairs=n, arms=ARMS, arm_summary=stats, transitions=dict(transitions), win_to_loss=transitions['win_to_loss'],
                loss_to_win=transitions['loss_to_win'], both_severe_losses=sum(all(severe(r[a]) for a in ARMS) for r in rows),
                severe_definition='literal loss and enemy elimination <= 0.5',
                intervention=dict(controller_steps=checked, swarm_steps=steps, controller_coverage=checked/steps,
                                  first_order_difference=dict(diffs), matched_prefix_macro_divergence_rate=diffs['macro']/n,
                                  matched_prefix_any_order_divergence_rate=(diffs['macro']+diffs['v2_micro_only'])/n,
                                  denominator='all pairs; never compare actions after trajectories diverge'))


def frozen_panel(out):
    d, original = ev.read_json(out/'design.json'), ev.reference()
    ev.require(d['kind'] == KIND and d['arms'] == ARMS and d['dry_run'] is False and d['arm_protocols'] == PROTOCOLS
               and ev.read_json(ev.HISTORY/f'{ev.digest(d)}.json') == d and ev.sha(out/'radical_swarm.py') == d['sources'][SRC]
               and d['policies']['swarm'] == dict(controller_sha256=d['sources'][SRC], trial_id=d['trial_id'], protocol=MICRO)
               and d['frozen_v2_design_sha256'] == old.ARCHIVES[old.FRESH] and d['policies']['v2'] ==
               old.policies(d['sources'],ev.model_contract(out/'model.json',original),d['trial_id'])['v2'], 'invalid frozen panel')
    batch, ids = d['replay_batch'], d['pilot_ids']
    ev.require(len(d['pairs']) == (len(ids) if ids else d.get('expected_panel_size',240)), 'incomplete panel')
    if batch:
        archive = ev.RUNS/batch/'design.json'
        ev.require(ev.sha(archive) == d['replay_design_sha256'], 'changed replay design')
        all_pairs = ev.read_json(archive)['pairs']
        ev.require(d['pairs'] == ([all_pairs[i-1] for i in ids] if ids else all_pairs), 'different replay scenarios')
    return d, original


def audited_arm(out, p, a, m, original, d):
    work = out/p['pair_id']/a
    if a == 'v2':
        result = ev.audit_arm(out,p,a,m,original,d,d['policies'][a])
        old.trial_counts(work,a,d['policies'][a])
        trace = None
    else:
        result, trace = swarm_arm(work,p,m,d,original)
    runner = work/'runner'
    dmg = ev.read_json(runner/'damage_by_type/episode-0001.json')
    score = ev.read_json(runner/'score_details/episode-0001.json')
    ev.require(dmg['method'] == PROTOCOLS[a]['method'] and dmg['map_file'] == original['map_file'] and
               dmg['map_sha256'] == d['sources'][str(Path(dmg['map_file']).relative_to(ev.ROOT))] and
               all(dmg[k] == score[k] for k in ('scenario_sha256','manifest_sha256','game_version')) and
               dmg['begin_request_loop'] == score['initial']['game_loop'] and
               dmg['terminal_loop'] == score['final']['game_loop'] and dmg['ticks_per_damage'] == 4096,
               'wrong damage map/measurement window')
    if d['replay_batch'] in old.ARCHIVES:
        ep, = ev.read_json(ev.RUNS/d['replay_batch']/p['pair_id']/'v2/runner/episodes.json')
        ev.require(ep['initial_state_sha256'] == result['initial'], 'different archived initial state')
    return result, trace


def compare(out):
    d, original = frozen_panel(out)
    batch, ids = d['replay_batch'], d['pilot_ids']
    rows = []
    for index,p in enumerate(d['pairs'],1):
        work = out/p['pair_id']
        try:
            m = ev.read_json(out/p['manifest'])
            trees.validate_pair(p,m,ids[index-1] if ids else index,original['env_args'])
            v2, _ = audited_arm(out,p,'v2',m,original,d)
            swarm, trace = audited_arm(out,p,'swarm',m,original,d)
            ev.require(v2['initial'] == swarm['initial'], 'different paired initial state')
            rows.append(dict(p, v2=v2, swarm=swarm, first_order_difference=prefix(work/'v2/runner',trace)))
        except (KeyError, ValueError, TypeError, OSError) as exc:
            raise ValueError(f"{p['pair_id']}: {exc}") from exc
    return dict(**aggregate(rows), kind=KIND, trial_id=d['trial_id'], protocols=PROTOCOLS, policies=d['policies'],
                replay_batch=batch, pilot_ids=ids, paired_outcomes=rows)


@contextmanager
def run_lock(out):
    # Lock the existing directory inode: no stale lockfile or dry-run writes.
    fd = os.open(out, os.O_RDONLY | os.O_DIRECTORY)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(f'run already locked: {out}') from exc
        yield fd
    finally:
        os.close(fd)


def completed(work):
    runner = work/'runner'
    summary = runner/'summary.json'
    if summary.exists():
        s = ev.read_json(summary)
        if s['status'] == 'complete':
            ev.require(s['completed_episodes'] == 1, 'incorrect completed episode count')
            return True
        ev.require(s['status'] == 'failed' and s['completed_episodes'] == 0, 'ambiguous worker summary')
    # Missing/changed summary is not permission to replay a terminal episode.
    for marker in ('episodes.json', 'score_details/episode-0001.json', 'damage_by_type/episode-0001.json'):
        ev.require(not (runner/marker).exists(), f'terminal evidence without complete summary: {marker}')
    for path, marker in ((runner/'events.jsonl', b'"episode_completed"'), (work/'worker.log', b': episode 1/1:')):
        if path.exists():
            with path.open('rb') as stream:
                ev.require(not any(marker in line for line in stream), f'terminal evidence in {path.name}')
    return False


def resume(out, workers, check):
    with run_lock(out) as fd:
        d, original = frozen_panel(out)
        workers = d['workers'] if workers is None else workers
        ev.require(1 <= workers <= 60, 'workers must be in 1..60')
        config, versions = ev.historical_config()
        ev.require(sys.executable == d['python'] == ev.RSI_PYTHON and versions == d['versions'] == original['versions']
                   and config == original['env_args'] and d['sc2_path'] == ev.SC2_PATH
                   and d['damage_map_dir'] == str(ev.MAP_DIR) and d['historical_run_sha256'] == ev.REFERENCE_HASH
                   and all(d[k] == v for k,v in ev.PROTOCOL.items()), 'changed worker runtime or protocol')
        current = ev.frozen_sources(original)
        for k in sorted(set(current) | set(d['sources'])):
            ev.require(current.get(k) == d['sources'].get(k), f'changed frozen source: {k}')
        pending, attempts = [], []
        for i,p in enumerate(d['pairs'], 1):
            m = ev.read_json(out/p['manifest'])
            trees.validate_pair(p,m,d['pilot_ids'][i-1] if d['pilot_ids'] else i,config)
            initials = set()
            for a in ARMS:
                work = out/p['pair_id']/a
                try:
                    ev.require(work.resolve() == work, 'symlinked arm path')
                    if completed(work):
                        result, _ = audited_arm(out,p,a,m,original,d)
                        initials.add(result['initial'])
                    else:
                        pending.append((p,a))
                        if work.exists() and any(work.iterdir()):
                            attempts.append(work)
                except (OSError, KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f'{work}: {exc}') from exc
            ev.require(len(initials) <= 1, f"{p['pair_id']}: different paired initial state")
        report = out/'report.json'
        ev.require(not pending or not report.exists(), 'report exists for incomplete panel')
        print(json.dumps(dict(valid=len(ARMS)*len(d['pairs'])-len(pending), pending=len(pending), workers=workers,
                              archive=[str(w.relative_to(out)) for w in attempts], check_only=check)), flush=True)
        if check:
            return
        if attempts:
            archive = Path(tempfile.mkdtemp(prefix='resume-attempts-', dir=out))
            for work in attempts:
                dest = archive/work.relative_to(out)
                dest.parent.mkdir(exist_ok=True)
                work.rename(dest)
        for p,a in pending:
            (out/p['pair_id']/a).mkdir(exist_ok=True)
        run_jobs(out,d,pending,workers,fd)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument('--out',type=Path)
    mode.add_argument('--compare',type=Path)
    mode.add_argument('--resume',type=Path)
    ap.add_argument('--resume-check',action='store_true',help='audit --resume without writing or launching')
    replay = ap.add_mutually_exclusive_group()
    replay.add_argument('--replay-batch',choices=old.ALIASES,help='historical v2: two full 240-pair archives')
    replay.add_argument('--repeat-design',type=Path,help='reuse manifests from a prior fresh swarm panel')
    ap.add_argument('--pilot-ids',nargs='+',type=int,help='1-based replay pair IDs')
    ap.add_argument('--workers',type=int,help='1..60 (default 2)')
    ap.add_argument('--pairs',type=int,help='number of new random matched scenarios (default 240)')
    ap.add_argument('--dry-run',action='store_true')
    args = ap.parse_args(argv)
    if args.resume:
        if any(v is not None for v in (args.replay_batch,args.repeat_design,args.pilot_ids,args.pairs)):
            ap.error('--resume cannot change the frozen design')
        return resume(args.resume.resolve(),args.workers,args.resume_check or args.dry_run)
    if args.resume_check:
        ap.error('--resume-check requires --resume DIR')
    if args.compare:
        if args.dry_run or any(v is not None for v in (args.workers,args.replay_batch,args.repeat_design,args.pilot_ids,args.pairs)):
            ap.error('--compare takes no launch options')
        print(json.dumps(compare(args.compare.resolve()),indent=2,allow_nan=False))
        return
    out = args.out.resolve()
    d = prepare(out,2 if args.workers is None else args.workers,args.replay_batch,args.pilot_ids,args.dry_run,args.repeat_design,args.pairs)
    print(f"{out}: {len(d['pairs'])} pairs (v2 micro8-all vs swarm macro-only); dry_run={args.dry_run}",flush=True)
    if args.dry_run:
        return
    with run_lock(out) as fd:
        run_jobs(out,d,((p,a) for p in d['pairs'] for a in ARMS),d['workers'],fd)


def run_jobs(out,d,pending,workers,fd):
    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = {pool.submit(play,out,p,a,d['policies'][a],fd):out/p['pair_id']/a for p,a in pending}
        for f in as_completed(jobs):
            try:
                f.result()
            except Exception as exc:
                for job in jobs:
                    job.cancel()
                raise RuntimeError(f'worker failed: {jobs[f]}/worker.log') from exc
    report = compare(out)
    if (out/'report.json').exists():
        ev.require(ev.read_json(out/'report.json') == report, 'existing report differs from full comparison')
    else:
        ev.write_json(out/'report.json',report)
    print(json.dumps({k:v for k,v in report.items() if k != 'paired_outcomes'},indent=2,allow_nan=False))


if __name__ == '__main__':
    main()
