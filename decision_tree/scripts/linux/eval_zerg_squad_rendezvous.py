"""Replay a squad handoff on the old 900; comparator: completed LIVE router, never inferred v2/swarm.

Use --out DIR --mode adaptive|fixed|guard|front [--pilot-ids 36 347] [--workers 60] [--dry-run].
For guard/front the LIVE router is a long-term reference, not the direct adaptive parent.
Use --resume DIR to continue, --resume DIR --dry-run to audit, --compare DIR to report without play.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import itertools
import json
from pathlib import Path
import secrets
import subprocess
import sys
import tempfile

from decision_tree.scripts.linux import eval_zerg_swarm_router as router
from sc2_skill_adapter.bridge.zerg_squad import (ADAPTIVE_SHA256, CROSS_MODES, CROSS_SOURCE,
    MODES, SOURCE as SRC, controller_class, squad_contract)

ev, sw, require = router.ev, router.sw, router.require
ONLINE = ev.RUNS/'zerg_router_old900_live_20261125_105505'
DESIGN = 'd4baa874e75f79efcc1038136b3c54a5c78b25f25bcb69fd6d7fc8f0f4e4ab05'
REPORT = 'aca4aeb5cf684938d9d4962d874f218418c181eb004641db74fe911503bf26f8'
ARM = 'squad_rendezvous'
BRIDGE = 'sc2_skill_adapter/bridge/zerg_squad.py'
KIND = 'squad_rendezvous_old900_v1'


def baseline():
    require(ev.sha(ONLINE/'design.json') == DESIGN and ev.sha(ONLINE/'report.json') == REPORT
            and ev.sha(router.ARCHIVE/'report.json') == router.REPORT_HASH, 'changed archived comparator')
    design = ev.read_json(ONLINE/'design.json')
    online = ev.read_json(ONLINE/'report.json')['paired_outcomes']
    old = ev.read_json(router.ARCHIVE/'report.json')['paired_outcomes']
    require(len(design['pairs']) == len(online) == len(old) == 900 and all(
        all(a[k] == b[k] == v for k, v in p.items()) and a['initial'] == b['v2']['initial'] == b['swarm']['initial']
        for p, a, b in zip(design['pairs'], online, old)), 'misaligned archived panels')
    return design, online, old


def runtime(parent):
    require(sys.executable == ev.RSI_PYTHON, f'use {ev.RSI_PYTHON}')
    original = ev.reference()
    cfg, versions = ev.historical_config()
    require(versions == parent['versions'] and cfg == original['env_args'], 'changed historical runtime')
    sources = ev.frozen_sources(original)
    require(SRC in sources and BRIDGE in sources and all(sources.get(k) == v for k,v in parent['sources'].items()
            if k.startswith('decision_tree/zerg/')), 'missing squad or changed historical policy source')
    return original, cfg, versions, sources


def prepare(out, mode, ids, workers, dry):
    parent, _, _ = baseline()
    _, cfg, versions, sources = runtime(parent)
    require(mode in MODES and (mode not in CROSS_MODES or
            CROSS_SOURCE in sources and sources[SRC] == ADAPTIVE_SHA256), 'unknown mode or missing cross-support/parent source')
    require(not ids or len(set(ids)) == len(ids) and all(1 <= i <= 900 for i in ids), 'invalid pilot IDs')
    policy = dict(parent['policies'][router.ARM], trial_id=secrets.token_hex(16))
    return sw.trees.freeze_panel(out, workers, None, dry, ONLINE.name, None, ids, {ARM:policy}, sources,
                                 cfg, versions, {ONLINE.name:DESIGN}, KIND, panel_size=900, mode=mode,
                                 online_report_sha256=REPORT, original_report_sha256=router.REPORT_HASH)


def command(out, p, d):
    cmd, env = ev.command(out, p, 'v2', d['policies'][ARM])
    work = out/p['pair_id']/ARM
    cmd[-1] = str(work/'runner')
    env.update(ZERG_SHADOW_DIR=str(work/'shadow'), ZERG_SQUAD_MODE=d['mode'], ZERG_SQUAD_SHA256=d['sources'][SRC])
    if d['mode'] in CROSS_MODES:
        env['ZERG_SQUAD_CROSS_SHA256'] = d['sources'][CROSS_SOURCE]
    return cmd, env


def play(out, p, d, fd):
    work = out/p['pair_id']/ARM
    work.mkdir(exist_ok=True)
    cmd, env = command(out, p, d)
    with (work/'worker.log').open('x') as log:
        subprocess.run(cmd, env=env, cwd=ev.ROOT, check=True, stdout=log, stderr=subprocess.STDOUT, pass_fds=(fd,))


def orders(obs, step):
    micro = sorted((x['submitted_loop'], x['actor'], x['submitted_action_id'], x['target'])
                   for x in step['feedback']['micro_submissions'])
    return ev.digest(obs), step['actions'], [list(x) for x in micro]


def first_difference(work, trace, ref):
    # The archive report pins results/initial state; orders must come from its actual live trace.
    ep, = ev.read_json(work/'episodes.json')
    require(ep['outcome'] == ref['online_outcome'] and ep['initial_state_sha256'] == ref['initial'] and
            ep['steps'] == ref['steps'] and ep['scenario_sha256'] == ref['scenario_sha256'] and
            'swarm_route' in ep, 'wrong online comparator episode')
    obs = None
    i = 0
    with (work/'events.jsonl').open() as stream:
        for line in stream:
            if line.startswith('{"event": "skill_observation",'):
                event = json.loads(line)
                require(event['step'] == i and obs is None, 'invalid comparator observation')
                obs = event['payload']
            elif line.startswith('{"event": "step",'):
                event = json.loads(line)
                require(obs is not None and event['step'] == i+1, 'invalid comparator step')
                left = orders(obs, event)
                if i >= len(trace) or left[0] != trace[i][0]:
                    return dict(status='unmatched_observation', step=i)
                for j, status in ((1,'macro'), (2,'micro')):
                    if left[j] != trace[i][j]:
                        return dict(status=status, step=i, online=left[j], squad=trace[i][j])
                obs = None
                i += 1
    require(i == ref['steps'] and obs is None, 'truncated online trace')
    return dict(status='identical' if i == len(trace) else 'unmatched_observation', step=i)


def groups(old):
    left, right = old['v2'], old['swarm']
    result = []
    if left['outcome'] == right['outcome'] == 'loss':
        result.append('both_loss')
        if max(left['enemy_elimination_fraction'], right['enemy_elimination_fraction']) <= .5:
            result.append('both_severe')
    if left['outcome'] == right['outcome'] == 'win':
        result.append('both_win')
    return result


def audit(out, p, d, original, online, old, arm_dir=ARM):
    work = out/p['pair_id']/arm_dir/'runner'
    meta, summary, episodes, score, dmg = (ev.read_json(work/k) for k in (
        'run.json','summary.json','episodes.json','score_details/episode-0001.json','damage_by_type/episode-0001.json'))
    ep, = episodes
    policy = d['policies'][ARM]
    expected = squad_contract(d['mode'], d['sources'], policy['trial_id'])
    require(meta['squad_rendezvous'] == expected and all(meta[k] == original[k] for k in ev.RUN_KEYS) and
            'swarm_route' not in meta and all(meta['zerg_distill'][k] == v for k,v in policy.items()), 'wrong deployed policy')
    required = set(original['source_hashes']) | {k for k in d['sources'] if k.startswith('sc2_skill_adapter/bridge/')} | {
        SRC, 'decision_tree/zerg/policy_distill.py','decision_tree/zerg/policy_repair_v2.py','decision_tree/zerg/opening_support_v2.py'}
    if d['mode'] in CROSS_MODES:
        required.add(CROSS_SOURCE)
    require(required <= meta['source_hashes'].keys() and all(d['sources'].get(k) == v for k,v in meta['source_hashes'].items()),
            'worker source mismatch')
    n, start = ep['steps'], score['initial']['game_loop']
    initial = ep['initial_state_sha256'] if online is None else online['initial']
    require(initial and ev.read_json(work/'manifest.json') == ev.read_json(out/p['manifest']) and
            p['manifest_sha256'] == meta['manifest_sha256'] == score['manifest_sha256'] == dmg['manifest_sha256'] and
            p['scenario_sha256'] == ep['scenario_sha256'] == score['scenario_sha256'] == dmg['scenario_sha256'] and
            score['sc2_seed'] == p['seed'] and ep['initial_state_sha256'] == initial, 'wrong game identity')
    require(summary['status'] == 'complete' and summary['completed_episodes'] == ep['episode'] == 1 and
            n == score['steps'] > 0 and ep['outcome'] == score['outcome'] in ('win','loss','draw','timeout') and
            score['game_version'] == dmg['game_version'] == '4.10.0.75689' and dmg['begin_request_loop'] == start and
            dmg['terminal_loop'] == score['final']['game_loop'] == start+8*n and dmg['map_file'] == meta['map_file'] and
            dmg['map_sha256'] == d['sources'][str(Path(meta['map_file']).relative_to(ev.ROOT))] and
            (work/'replays/episode-0001.SC2Replay').stat().st_size > 0, 'missing terminal evidence')
    controller = controller_class(d['mode'])(mode=d['mode'])
    trace, suffix, records = [], [], []
    decision = native = obs = window = None
    starts = ends = macro = micro = 0
    reward = 0.
    with (work/'events.jsonl').open() as stream:
        for line in stream:
            e = json.loads(line); kind = e['event']; i = len(trace); loop = start+8*i
            if kind == 'episode_started':
                starts += 1
                require(e['initial_state_sha256'] == initial and e['scenario_sha256'] == p['scenario_sha256']
                        and e['game_loop'] == start, 'incorrect reset')
            elif kind == 'skill_observation':
                require(starts == 1 and ends == 0 and obs is None and e['step'] == i and e['payload']['state']['game_loop'] == loop, 'bad observation')
                obs = e['payload']
            elif kind in ('skill_step','squad_v2_proposal'):
                require(decision is not None and native is None and e['step'] == i and e['game_loop'] == loop and
                        decision['handoff'] is (kind == 'skill_step') and
                        e['actions'] == {f'u{j}':a for j,a in enumerate(decision['v2_actions'])}, 'bad v2 proposal')
                native = e['actions']
            elif kind == 'squad_rendezvous_decision':
                base = e['v2_actions']
                require(obs is not None and decision is None and
                        e['step'] == i and e['game_loop'] == loop, 'incorrect v2 action join')
                proposed = controller.propose(obs, e['avail_actions'], base)
                if d['mode'] in CROSS_MODES:
                    require(e.get('skill_audit') == proposed, 'incorrect cross-support audit details')
                require(all(e[k] == proposed[k] for k in ('actions','phase','handoff','reason')) and
                        (not suffix or e['handoff']) and (not e['handoff'] or base == e['actions']), 'incorrect handoff')
                decision = e
            elif kind == 'micro_window_closed':
                require(native is not None and window is None and e['macro_step'] == i and e['issued_loop'] == loop and
                        e['closed_loop'] == loop+8 and e['interior_checks'] == 7, 'incorrect micro window')
                window = e
            elif kind == 'step':
                require(window is not None and e['step'] == i+1 and e['game_loop'] == loop and e['actions'] == decision['actions']
                        and e['terminated'] is (i+1 == n), 'wrong executed actions')
                fb = e['feedback']; subs = fb['micro_submissions']; rec = window['records']
                require(len(rec) == window['authorizations'] and (decision['handoff'] or not rec) and
                        sorted((r for r in rec if r['status'] == 'submitted'), key=lambda r:r['actor']) == sorted(subs,key=lambda r:r['actor']) and
                        all(r['sc2_response'] == [1] and loop < r['submitted_loop'] < loop+8 and
                            r['macro_action_id'] == e['actions'][int(r['actor'][1:])] for r in subs), 'wrong micro receipt')
                macro += sw.check_macro(e['actions'], decision['avail_actions'], dict(fb,micro_submissions=[]), loop)
                records.extend(rec); micro += len(subs); reward += e['reward']
                if decision['handoff']: suffix.append(e)
                trace.append(orders(obs, e))
                decision = native = obs = window = None
            elif kind == 'episode_completed':
                ends += 1
                require(i == n and all(e[k] == v for k,v in ep.items()), 'incorrect terminal event')
    counts = dict(squad_steps=n-len(suffix), v2_steps=len(suffix))
    require(starts == ends == 1 and len(trace) == n and reward == ep['reward'] and counts == ep[ARM] and
            ep['zerg_shadow_decisions'] == len(suffix) and ep['micro_trigger'] == summary['micro_trigger_totals'] ==
            dict(mode='micro8-all',macro_loops=8,engine_step_loops=1,windows=n,authorizations=len(records),
                 states=dict(Counter(r['status'] for r in records))) and records ==
            ev.read_json(work/'micro_triggers/episode-0001.json')['authorizations'], 'incomplete execution evidence')
    audit_suffix(work, ep, policy, suffix)
    if online is None:
        return dict(initial=initial,outcome=ep['outcome'],reward=reward,enemy_elimination_fraction=ep['enemy_elimination_fraction'],**counts)
    return dict(p, **counts, initial=online['initial'], online_outcome=online['online_outcome'], outcome=ep['outcome'],
                steps=n, reward=reward, enemy_elimination_fraction=ep['enemy_elimination_fraction'], macro_commands=macro,
                micro_submissions=micro, original_v2=old['v2']['outcome'], original_swarm=old['swarm']['outcome'],
                strata=groups(old), first_order_difference=first_difference(ONLINE/p['pair_id']/router.ARM/'runner',trace,online))


def audit_suffix(work, ep, policy, suffix):
    receipt = ev.read_json(work/'distill/episode-0001.json'); rows = receipt['decisions']
    require(receipt['status'] == 'complete' and receipt['scope'] == 'executed_v2_suffix_only' and
            receipt['steps'] == len(rows) and all(receipt[k] == v for k,v in policy.items()) and
            all(receipt[k] == ep[k] for k in ('scenario_sha256','initial_state_sha256')) and len(rows) == len(suffix), 'wrong v2 receipt')
    shadow = work.parent/'shadow/shadow.jsonl'
    if rows or shadow.exists():
        with shadow.open() as stream:
            for r, line, e in itertools.zip_longest(rows,stream,suffix):
                require(line is not None and r is not None and e is not None, 'incomplete suffix shadow')
                s = json.loads(line)
                require(r['step'] == e['step']-1 and r['selector_sha256'] == policy['selector_sha256'] and
                        r['trial_id'] == policy['trial_id'] and all(s[k] == r[k] for k in
                        ('snapshot_id','game_loop','selected_candidate','final_actions')), 'wrong suffix receipt')
    if rows:
        # audit_decisions expects a zero-based contiguous segment; absolute steps checked above.
        normalized = dict(receipt,decisions=[dict(r,step=i) for i,r in enumerate(rows)])
        counts, _ = ev.audit_decisions(normalized, suffix, ep['zerg_distill'])
        require(counts['engine_unconfirmed'] == 0, 'unconfirmed v2 receipt')


def summarize(rows):
    return dict(pairs=len(rows), online_wins=sum(r['online_outcome']=='win' for r in rows),
                wins=sum(r['outcome']=='win' for r in rows),
                transitions=dict(Counter(f"{r['online_outcome']}_to_{r['outcome']}" for r in rows)),
                first_order_changes=dict(Counter(r['first_order_difference']['status'] for r in rows)))


def execute(out, workers, mode=None, check=False, compare=False):
    with sw.run_lock(out) as fd:
        parent, online, old = baseline(); original, cfg, versions, sources = runtime(parent)
        d = ev.read_json(out/'design.json')
        require(d['kind'] == KIND and d['arms'] == [ARM] and mode in (None,d['mode']) and
                ev.read_json(ev.HISTORY/f'{ev.digest(d)}.json') == d and ev.sha(out/'model.json') == ev.MODEL_HASH and
                d['sources'] == sources and d['versions'] == versions, 'changed frozen panel/mode/runtime')
        require(not d['dry_run'] or check, 'dry-run design cannot be executed or compared')
        ids = d['pilot_ids'] or range(1,901)
        require(d['pairs'] == [parent['pairs'][i-1] for i in ids], 'different replay panel')
        rows, pending, partial = {}, [], []

        def verify(i, p):
            try:
                rows[i] = audit(out,p,d,original,online[i-1],old[i-1])
            except (ValueError, KeyError, TypeError, OSError) as exc:
                raise ValueError(f"{p['pair_id']}: {exc}") from exc

        for i,p in zip(ids,d['pairs']):
            work = out/p['pair_id']/ARM
            require(work == work.resolve(), 'symlinked work directory')
            sw.trees.validate_pair(p,ev.read_json(out/p['manifest']),i,cfg)
            if sw.completed(work): verify(i,p)
            else:
                pending.append((i,p))
                if work.exists() and any(work.iterdir()): partial.append(work)
        reportfile = out/'report.json'
        require(not pending or not (compare or reportfile.exists()), 'incomplete comparison panel')
        print(f"{out}: {len(rows)} audited, {len(pending)} pending {d['mode']} games",flush=True)
        if check and pending: return
        if partial:
            archive = Path(tempfile.mkdtemp(prefix='resume-attempts-',dir=out))
            for work in partial:
                dest = archive/work.relative_to(out); dest.parent.mkdir(); work.rename(dest)
        if pending:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                jobs = {pool.submit(play,out,p,d,fd):(i,p) for i,p in pending}
                try:
                    for job in as_completed(jobs):
                        job.result(); verify(*jobs[job])
                except BaseException:
                    for job in jobs: job.cancel()
                    raise
        results = [rows[i] for i in sorted(rows)]
        report = dict(**summarize(results), kind=KIND, mode=d['mode'], design_sha256=ev.sha(out/'design.json'),
                      comparator=ONLINE.name, online_report_sha256=REPORT, original_report_sha256=router.REPORT_HASH,
                      strata={g:summarize([r for r in results if g in r['strata']]) for g in ('both_loss','both_severe','both_win')},
                      paired_outcomes=results)
        if d['mode'] in CROSS_MODES:
            report.update(controller=squad_contract(d['mode'], sources, d['policies'][ARM]['trial_id']),
                          direct_parent_mode='adaptive', comparator_role='long_term_reference_not_direct_parent')
        if reportfile.exists(): require(report == ev.read_json(reportfile), 'existing report differs')
        elif not check: ev.write_json(reportfile,report)
        print(json.dumps({k:v for k,v in report.items() if k!='paired_outcomes'}))
        return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument('--out',type=Path); target.add_argument('--resume',type=Path)
    target.add_argument('--compare','--comparison',type=Path)
    parser.add_argument('--mode',choices=MODES); parser.add_argument('--workers',type=int,default=60)
    parser.add_argument('--pilot-ids',type=int,nargs='+'); parser.add_argument('--dry-run',action='store_true')
    a = parser.parse_args(argv)
    if not 1 <= a.workers <= 60 or (not a.out and a.pilot_ids) or (a.out and not a.mode) or (a.compare and a.dry_run):
        parser.error('--out needs --mode; pilot IDs only with --out; workers 1..60; --compare does not launch')
    out = (a.out or a.resume or a.compare).resolve()
    if a.out:
        d = prepare(out,a.mode,a.pilot_ids,a.workers,a.dry_run)
        print(f"{out}: frozen {len(d['pairs'])} {a.mode} games vs {ONLINE.name}", flush=True)
        if a.dry_run: return
    return execute(out,a.workers,a.mode,check=a.dry_run,compare=bool(a.compare))


if __name__ == '__main__': main()
