"""Run ONLY the live sticky router on the old 900 manifests. Control: archived, audited v2 (not rerun).

Historical RSI Python: -m decision_tree.scripts.linux.eval_zerg_swarm_router
--out runs/NEW [--workers 60] [--pilot-ids 36 347 810] [--dry-run].
--resume DIR re-audits terminal games before continuing. --resume DIR --dry-run audits only.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import itertools
import json
import math
from pathlib import Path
import secrets
import subprocess
import sys
import tempfile

from decision_tree.scripts.linux import eval_zerg_swarm as sw
from decision_tree.scripts.linux.fit_swarm_explorer import use
from decision_tree.zerg.policy_route_gate import anchor_features
from decision_tree.zerg.radical_swarm import RadicalSwarm

ev = sw.ev
require = ev.require
ARCHIVE = ev.RUNS/'zerg_v2_vs_swarm_random900_20261124_223951'
GATE = ev.RUNS/'zerg_swarm_explorer_900_v1/gate.json'
DESIGN_HASH = '3f69deb5470255f52b4ddd2890a13aaada4b4cca97825a9c4ba157967e6f0041'
REPORT_HASH = '6880aff7dfdd32389b37912918f4e90315d94ac04383547faba77bb29874331e'
GATE_HASH = '6d3a2ad32e0f9543c9d1ed9734de681592b887c874c3cb4dc02d5dbe0007fb2f'
ARM = 'online_router'
EXTRA = ('decision_tree/scripts/linux/fit_swarm_explorer.py', 'decision_tree/scripts/linux/fit_swarm_gate_324.py')


def sources(original):
    return dict(ev.frozen_sources(original), **{k:ev.sha(ev.ROOT/k) for k in EXTRA})


def prepare(out, ids, workers, dry):
    require(sys.executable == ev.RSI_PYTHON, f'use {ev.RSI_PYTHON}')
    require(ev.sha(ARCHIVE/'design.json') == DESIGN_HASH and ev.sha(ARCHIVE/'report.json') == REPORT_HASH
            and ev.sha(GATE) == GATE_HASH, 'fixed panel, audited report or gate changed')
    require(not ids or len(set(ids)) == len(ids) and all(1 <= i <= 900 for i in ids), 'invalid pilot IDs')
    original = ev.reference()
    src = sources(original)
    cfg, versions = ev.historical_config()
    hist = ev.read_json(ARCHIVE/'design.json')
    require(versions == hist['versions'] and cfg == original['env_args'], 'different historical runtime')
    policy = dict(hist['policies']['v2'], trial_id=secrets.token_hex(16))
    h = ev.read_json(GATE)['policy_hashes']
    require(all(policy[k] == v for k,v in h.items() if k != 'radical_swarm') and
            src[sw.SRC] == h['radical_swarm'] and src['decision_tree/zerg/policy_repair_v2.py'] == h['selector_sha256']
            and src['decision_tree/zerg/opening_support_v2.py'] == h['opening_sha256'], 'changed frozen policy')
    d = sw.trees.freeze_panel(out, workers, None, dry, ARCHIVE.name, None, ids, {ARM:policy}, src, cfg, versions,
                              {ARCHIVE.name:DESIGN_HASH}, 'online_swarm_router_old900_v1', panel_size=900,
                              comparator='archived audited v2; not rerun', archived_report_sha256=REPORT_HASH,
                              gate_sha256=GATE_HASH)
    (out/'gate.json').write_bytes(GATE.read_bytes())
    require(ev.sha(out/'gate.json') == GATE_HASH, 'gate changed during freeze')
    return d


def command(out, p, d):
    cmd, env = ev.command(out, p, 'v2', d['policies'][ARM])
    work = out/p['pair_id']/ARM
    cmd[-1] = str(work/'runner')
    env.update(ZERG_SHADOW_DIR=str(work/'shadow'), ZERG_SWARM_GATE=str(out/'gate.json'),
               ZERG_SWARM_GATE_SHA256=GATE_HASH, ZERG_SWARM_SHA256=d['sources'][sw.SRC])
    return cmd, env


def play(out, p, d, fd):
    cmd, env = command(out, p, d)
    with (out/p['pair_id']/ARM/'worker.log').open('x') as log:
        subprocess.run(cmd, env=env, cwd=ev.ROOT, stdout=log, stderr=subprocess.STDOUT, check=True, pass_fds=(fd,))


def features(obs, left, right):
    f = anchor_features(obs, {'actions':left}, {'actions':right})
    st = obs['state']; units = [u for u in st['allies'].values() if u['alive']]
    guns = [u for u in units if u['type'] == 'hydralisk']; escorts = [u for u in units if u not in guns]
    f.update(seen=len(st['visible_enemies']), spread=max((math.dist(a['position'],b['position']) for a,b in
             itertools.combinations(units,2)), default=0), banes=sum(u['type']=='baneling' for u in units),
             n_lings=sum(u['type']=='zergling' for u in units), escort_gap=sum(min(math.dist(a['position'],b['position'])
             for b in escorts) for a in guns)/len(guns) if guns and escorts else -1)
    return f


def pick(gate, f):
    return 'swarm' if use(gate['rules'], {'features':f}) else 'v2'


def offline_choice(p, archived, gate):
    diff = archived['first_order_difference']
    require(diff['status'] == 'macro', 'archive has no verified macro anchor')
    with (ARCHIVE/p['pair_id']/'v2/runner/events.jsonl').open() as stream:
        for line in stream:
            if line.startswith('{"event": "skill_observation",'):
                obs = json.loads(line)
                if obs['step'] == diff['step']:
                    return pick(gate, features(obs['payload'], diff['v2'], diff['swarm']))
    raise ValueError('missing archived anchor')


def audit(out, p, d, original, archived, gate):
    work = out/p['pair_id']/ARM/'runner'
    meta, summary, episodes, score, dmg = [ev.read_json(work/k) for k in (
        'run.json','summary.json','episodes.json','score_details/episode-0001.json','damage_by_type/episode-0001.json')]
    ep, = episodes
    policy = d['policies'][ARM]
    route = dict(kind='swarm_first_macro_online_v1', gate_sha256=GATE_HASH,
                 controller_sha256=d['sources'][sw.SRC], trial_id=policy['trial_id'])
    require(all(meta[k] == original[k] for k in ev.RUN_KEYS) and meta['swarm_route'] == route
            and all(meta['zerg_distill'][k] == v for k,v in policy.items()), 'wrong deployed router/protocol')
    required = {k for k in d['sources'] if k.startswith('sc2_skill_adapter/bridge/')} | set(original['source_hashes']) | set(EXTRA) | {sw.SRC, 'decision_tree/zerg/policy_route_gate.py',
            'decision_tree/zerg/policy_distill.py','decision_tree/zerg/opening_support_v2.py','decision_tree/zerg/policy_repair_v2.py'}
    require(required <= meta['source_hashes'].keys() and all(d['sources'].get(k) == v for k,v in meta['source_hashes'].items()),
            'worker source mismatch')
    require(ev.read_json(work/'manifest.json') == ev.read_json(out/p['manifest']) and
            score['manifest_sha256'] == dmg['manifest_sha256'] == meta['manifest_sha256'] == p['manifest_sha256'] and
            score['scenario_sha256'] == dmg['scenario_sha256'] == ep['scenario_sha256'] == p['scenario_sha256'] and
            all(archived[k] == v for k,v in p.items()) and ep['initial_state_sha256'] == archived['v2']['initial']
            == archived['swarm']['initial'] and score['sc2_seed'] == p['seed'], 'wrong manifest, seed or initial state')
    n = ep['steps']; start = score['initial']['game_loop']
    require(summary['status'] == 'complete' and summary['completed_episodes'] == ep['episode'] == 1 and
            ep['outcome'] == score['outcome'] in ('win','loss','timeout','draw') and score['steps'] == n > 0 and
            score['game_version'] == dmg['game_version'] == '4.10.0.75689' and score['final']['game_loop'] ==
            dmg['terminal_loop'] == start+8*n and dmg['begin_request_loop'] == start and dmg['map_file'] == meta['map_file']
            and dmg['map_sha256'] == d['sources'][str(Path(meta['map_file']).relative_to(ev.ROOT))], 'incomplete terminal/damage output')
    # Per-game, ordered joins; never infer a live choice from the archive.
    obs = decision = window = choice = anchor = None
    i = starts = ends = micro = macros = 0
    reward = 0.
    v2_events = []
    records = []; expected_lock = None; swarm = RadicalSwarm(); native = None
    with (work/'events.jsonl').open() as stream:
        for line in stream:
            e = json.loads(line); kind = e['event']; loop = start+8*i
            if expected_lock is not None:
                require(kind == 'swarm_route_micro_lock' and e['step'] == i-1 and e['choice'] == 'v2' and
                        e['submissions'] == expected_lock, 'missing prefix micro lock')
                expected_lock = None
            elif kind == 'swarm_route_micro_lock':
                raise ValueError('spurious prefix micro lock')
            if kind == 'episode_started':
                starts += 1
                require(e['initial_state_sha256'] == ep['initial_state_sha256'] and e['game_loop'] == start and
                        e['scenario_sha256'] == p['scenario_sha256'], 'bad reset evidence')
            elif kind == 'skill_observation':
                require(starts == 1 and obs is None and e['step'] == i and e['payload']['state']['game_loop'] == loop, 'bad observation')
                obs = e['payload']
            elif kind == 'skill_step':
                require(native is None and obs is not None and e['step'] == i and e['game_loop'] == loop, 'incorrect v2 proposal step')
                native = e['actions']
            elif kind == 'swarm_route_decision':
                require(obs is not None and decision is None and e['step'] == i and e['game_loop'] == loop, 'bad route step')
                left, right = e['v2_actions'], e['swarm_actions']
                require((native is left is None) if choice == 'swarm' else
                        (left is not None and native == {f'u{j}':a for j,a in enumerate(left)}), 'incorrect native v2 proposal')
                if choice != 'v2':
                    replay = swarm.choose(obs,e['avail_actions'])
                    require(right == replay['actions'] and right == e['swarm_decision']['actions'], 'incorrect Swarm proposal')
                else:
                    require(right is None, 'unexpected recomputed Swarm proposal')
                if choice is None:
                    require(left is not None and right is not None, 'missing online alternatives')
                    if left != right:
                        f = features(obs,left,right)
                        require(e['features'] == f and e['reason'] == 'first_macro_difference', 'bad first-divergence features')
                        choice, anchor = pick(gate,f), i
                    else:
                        require(e['reason'] == 'common_macro', 'not common macro')
                else:
                    require(e['reason'] == 'locked', 'route was reconsidered')
                require(e['choice'] == choice and e['actions'] == (right if choice == 'swarm' else left), 'incorrect sticky branch/action')
                decision = e
            elif kind == 'micro_window_closed':
                require(window is None and e['macro_step'] == i and e['issued_loop'] == loop and
                        e['closed_loop'] == loop+8 and e['interior_checks'] == 7, 'incorrect micro interval')
                window = e
            elif kind == 'step':
                require(decision is not None and window is not None and e['step'] == i+1 and e['game_loop'] == loop
                        and e['actions'] == decision['actions'] and e['terminated'] is (i+1 == n), 'bad executed decision')
                fb = e['feedback']; subs = fb['micro_submissions']; rec = window['records']
                records.extend(rec)
                require(window['authorizations'] == len(rec) and (not rec or choice != 'swarm') and
                        sorted((r for r in rec if r['status']=='submitted'),key=lambda r:r['actor']) == sorted(subs,key=lambda r:r['actor'])
                        and all(r['sc2_response'] == [1] and loop < r['submitted_loop'] < loop+8 and
                                r['macro_action_id'] == e['actions'][int(r['actor'][1:])] for r in subs), 'wrong branch micro execution')
                macros += sw.check_macro(e['actions'], decision['avail_actions'], dict(fb,micro_submissions=[]),loop)
                if choice != 'swarm':
                    v2_events.append(e)
                if subs and choice is None:
                    choice = 'v2'; expected_lock = subs
                micro += len(subs); reward += e['reward']; i += 1
                obs = decision = window = native = None
            elif kind == 'episode_completed':
                ends += 1
                require(i == n and all(e[k] == v for k,v in ep.items()), 'incorrect terminal event')
    require(starts == ends == 1 and i == n and obs is decision is window is None and reward == ep['reward'], 'incomplete episode')
    require(ep['swarm_route'] == dict(choice=choice,v2_steps=len(v2_events),swarm_steps=n-len(v2_events)) and
            ep['zerg_shadow_decisions'] == len(v2_events) and ep['micro_trigger'] == summary['micro_trigger_totals'] ==
            dict(mode='micro8-all',macro_loops=8,engine_step_loops=1,windows=n,authorizations=len(records),
                 states=dict(Counter(r['status'] for r in records))) and
            ev.read_json(work/'micro_triggers/episode-0001.json')['authorizations'] == records and
            (work/'replays/episode-0001.SC2Replay').stat().st_size > 0, 'missing replay, branch or micro evidence')
    receipt = ev.read_json(work/'distill/episode-0001.json')
    require(receipt['status'] == 'complete' and receipt['scenario_sha256'] == p['scenario_sha256'] and
            receipt['initial_state_sha256'] == ep['initial_state_sha256'] and len(receipt['decisions']) == len(v2_events)
            and all(receipt[k] == v for k,v in policy.items()), 'wrong v2 prefix receipt')
    if v2_events:
        counts,_ = ev.audit_decisions(receipt,v2_events,ep['zerg_distill'])
        require(counts['engine_unconfirmed'] == 0, 'unconfirmed v2 command')
        with (work.parent/'shadow/shadow.jsonl').open() as shadow:
            for entry, line in itertools.zip_longest(receipt['decisions'],shadow):
                require(entry is not None and line is not None,'incomplete prefix shadow')
                row = json.loads(line)
                require(all(row[k] == entry[k] for k in ('snapshot_id','game_loop','selected_candidate','final_actions'))
                        and entry['selector_sha256'] == policy['selector_sha256'] and entry['trial_id'] == policy['trial_id'],
                        'shadow/native action mismatch')
    expected = offline_choice(p, archived, gate)
    return dict(p, choice=choice or 'v2', anchor_step=anchor, offline_choice=expected,
                choice_matches_offline=(choice or 'v2') == expected, archived_v2=archived['v2']['outcome'],
                offline_selected_outcome=archived[expected]['outcome'], online_outcome=ep['outcome'], reward=reward,
                macro_commands=macros, micro_submissions=micro, steps=n, initial=ep['initial_state_sha256'])


def execute(out, workers, check):
    with sw.run_lock(out) as fd:
        d = ev.read_json(out/'design.json'); original = ev.reference()
        require(d['kind'] == 'online_swarm_router_old900_v1' and ev.read_json(ev.HISTORY/f'{ev.digest(d)}.json') == d
                and ev.sha(ARCHIVE/'design.json') == DESIGN_HASH and ev.sha(ARCHIVE/'report.json') == REPORT_HASH
                and ev.sha(out/'gate.json') == GATE_HASH and ev.sha(out/'model.json') == ev.MODEL_HASH
                and sys.executable == ev.RSI_PYTHON and sources(original) == d['sources'], 'changed frozen panel/runtime')
        require(check or not d['dry_run'], 'dry-run design cannot be executed; use a new --out')
        gate = ev.read_json(out/'gate.json'); archived = ev.read_json(ARCHIVE/'report.json')['paired_outcomes']
        cfg, versions = ev.historical_config()
        require(versions == d['versions'] and cfg == original['env_args'], 'changed runtime config')
        rows, pending, attempts = {}, [], []
        for p in d['pairs']:
            i = int(p['pair_id'].split('-')[1]); work = out/p['pair_id']/ARM
            sw.trees.validate_pair(p,ev.read_json(out/p['manifest']),i,cfg)
            require(work == work.resolve(), 'symlinked work directory')
            if sw.completed(work):
                rows[i] = audit(out,p,d,original,archived[i-1],gate)
            else:
                pending.append((i,p))
                if work.exists() and any(work.iterdir()): attempts.append(work)
        print(f'{len(rows)} audited, {len(pending)} pending online games; archived v2 is not rerun.',flush=True)
        require(not pending or not (out/'report.json').exists(), 'report exists for partial panel')
        if check: return
        if attempts:
            archive = Path(tempfile.mkdtemp(prefix='resume-attempts-',dir=out))
            for work in attempts:
                target = archive/work.relative_to(out); target.parent.mkdir(); work.rename(target)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            jobs = {}
            for i,p in pending:
                (out/p['pair_id']/ARM).mkdir(exist_ok=True)
                jobs[pool.submit(play,out,p,d,fd)] = (i,p)
            try:
                for job in as_completed(jobs):
                    i,p = jobs[job]; job.result(); rows[i] = audit(out,p,d,original,archived[i-1],gate)
            except BaseException:
                for job in jobs: job.cancel()
                raise
        results = [rows[k] for k in sorted(rows)]
        report = dict(mode='live router only vs archived audited v2; replay, not a new holdout',
                      pairs=len(results), archived_report_sha256=REPORT_HASH, gate_sha256=GATE_HASH,
                      online_wins=sum(r['online_outcome']=='win' for r in results),
                      archived_v2_wins=sum(r['archived_v2']=='win' for r in results),
                      offline_selected_wins=sum(r['offline_selected_outcome']=='win' for r in results),
                      offline_outcome_mismatches=sum(r['online_outcome']!=r['offline_selected_outcome'] for r in results),
                      transitions=dict(Counter(f"{r['archived_v2']}_to_{r['online_outcome']}" for r in results)),
                      offline_choice_mismatches=sum(not r['choice_matches_offline'] for r in results),
                      choices=dict(Counter(r['choice'] for r in results)), paired_outcomes=results)
        if (out/'report.json').exists(): require(report == ev.read_json(out/'report.json'), 'existing report differs')
        else: ev.write_json(out/'report.json',report)
        print(json.dumps({k:v for k,v in report.items() if k!='paired_outcomes'}))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument('--out',type=Path); mode.add_argument('--resume',type=Path)
    ap.add_argument('--pilot-ids',type=int,nargs='+'); ap.add_argument('--workers',type=int,default=60)
    ap.add_argument('--dry-run',action='store_true'); a = ap.parse_args(argv)
    if not 1 <= a.workers <= 60 or a.resume and a.pilot_ids: ap.error('workers must be 1..60; --resume takes no pilot IDs')
    out = (a.out or a.resume).resolve()
    if a.out:
        d = prepare(out,a.pilot_ids,a.workers,a.dry_run)
        print(f"{out}: {len(d['pairs'])} online_router games, {a.workers} workers; comparator = archived audited v2, not rerun",flush=True)
        if a.dry_run: return
    execute(out,a.workers,a.dry_run)


if __name__ == '__main__': main()
