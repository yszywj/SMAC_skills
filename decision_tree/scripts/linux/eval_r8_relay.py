"""Replay menu/full siblings on the frozen R6 1k panel. --dry-run never launches SC2."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations
import json
import hashlib
from math import comb
from pathlib import Path
import secrets
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from decision_tree.scripts.linux import eval_r1_r6_r7 as old

sq, ev = old.sq, old.ev
require = ev.require
ARCHIVE = ev.RUNS/'zerg_R1_R6_R7_random1000_20261125_155323'
DESIGN = '09b4d8dbc2f5f8cf2e34341c0a56e8e19211eca774c270a7ef946690a4cbed4b'
REPORT = 'd684f436687797c50e94334b6207947ba5faf07418085593ec1905460a2a9774'
KIND = 'r8_relay_archived_r6_1k_v1'
RELAY = 'decision_tree/zerg/combat_relay_r8.py'
RELAY_SHA = '08d0b2a4ef19c87f31926ba0fa2c562a2f026b961ad21e6b229c956fc7db8878'
BRIDGE_HASHES = {'sc2_skill_adapter/bridge/runner.py':'e4798a3da3b596a1c2a347e87f1d586f96c502deedc6bfc1181b5e114c418ffd',
                 'sc2_skill_adapter/bridge/zerg_relay.py':'16a6e50cdec24e094124acefe5fe21d3e6cbcbecf8e6ba2a5a58ffdb7144283b'}
ARMS = ('menu', 'full')
COHORTS = dict(common_loss=701, severe=81, close=224, original_R6_wins=283, rescued_vs_R1=30)


def cohorts(row):
    a, b = row['R6'], row['R7']
    names = []
    if a['outcome'] == b['outcome'] == 'loss':
        names.append('common_loss')
        fractions = [x['enemy_elimination_fraction'] for x in (a, b)]
        if max(fractions) <= .5:
            names.append('severe')
        if min(fractions) >= .8:
            names.append('close')
    if a['outcome'] == 'win':
        names.append('original_R6_wins')
        if row['R1']['outcome'] != 'win':
            names.append('rescued_vs_R1')
    return names


def archive():
    require(ev.sha(ARCHIVE/'design.json') == DESIGN and ev.sha(ARCHIVE/'report.json') == REPORT,
            'changed archive design/report')
    parent = ev.read_json(ARCHIVE/'design.json')
    rows = ev.read_json(ARCHIVE/'report.json')['paired_outcomes']
    require(ev.read_json(ev.HISTORY/f'{ev.digest(parent)}.json') == parent and
            len(rows) == len(parent['pairs']) == 1000 and all(len({r[k] for r in rows}) == 1000
            for k in ('pair_id', 'seed', 'scenario_sha256', 'layout_sha256')), 'invalid archive reservation')
    for p, r in zip(parent['pairs'], rows):
        require(all(r[k] == v for k, v in p.items()) and
                r['R6']['initial'] == r['R7']['initial'] == r['R1']['initial'], 'misaligned archive')
    require(Counter(c for r in rows for c in cohorts(r)) == COHORTS, 'wrong archive cohorts')
    return parent, rows


def runtime(parent):
    original, cfg, versions, src = sq.runtime(parent)
    for k, v in parent['sources'].items():
        if k == 'sc2_skill_adapter/bridge/zerg_distill.py':
            data = (ev.ROOT/k).read_bytes()
            hook = b'''        # Explicit R8 only: expose the certified, deduplicated native menu,
        # after resolving the frozen selector (including its opening branch).
        if hasattr(self, 'relay_select'):
            selected = self.relay_select(payload, masks, menu, v)
            if selected != v:
                generated = False
            v = selected
'''
            require(data.count(hook) == 1 and hashlib.sha256(data.replace(hook,b'')).hexdigest() == v,
                    'distill differs outside the opt-in relay hook')
        elif k != 'sc2_skill_adapter/bridge/runner.py':
            require(src.get(k) == v, f'changed frozen R6 dependency: {k}')
    require(src.get(RELAY) == RELAY_SHA and ev.sha(ARCHIVE/'model.json') == ev.MODEL_HASH, 'changed relay or model')
    require(all(src.get(k) == v for k,v in BRIDGE_HASHES.items()), 'changed relay bridge or runner')
    return original, cfg, versions, src


def command(out, p, d, arm):
    cmd, env = sq.command(out, p, dict(d, mode='guard', policies={sq.ARM:d['policies'][arm]}))
    work = out/p['pair_id']/arm
    cmd[-1] = str(work/'runner')
    env.update(ZERG_SHADOW_DIR=str(work/'shadow'), ZERG_RELAY_MODE=arm, ZERG_RELAY_SHA256=d['sources'][RELAY])
    return cmd, env


def retryable(work):
    # Terminal artifacts or an ambiguous/started episode can never be discarded, even on resume.
    if not (work/'worker.log').is_file() or sq.sw.completed(work) or not old.startup_disconnect(work):
        return False
    allowed = {'worker.log','runner/run.json','runner/manifest.json','runner/events.jsonl','runner/summary.json'}
    if any(p.is_file() and str(p.relative_to(work)) not in allowed for p in work.rglob('*')):
        return False
    events = work/'runner/events.jsonl'
    if events.exists():
        with events.open() as f:
            return all(e['event'] in ('skill_configuration','unit_value_configuration') or
                       e['event'] == 'run_failed' and e['completed_episodes'] == 0 for e in map(json.loads,f))
    return True


def attempts(work):
    root = work.parent/'old-attempts'/work.name
    prior = sorted(root.iterdir()) if root.exists() else []
    for i, path in enumerate(prior, 1):
        require(path.name == str(i) and path == path.resolve() and retryable(path), f'ambiguous prior attempt: {path}')
    require(len(prior) <= 3, 'retry budget exceeded')
    return root, len(prior)


def play(out, p, d, arm, fd):
    work = out/p['pair_id']/arm
    root, number = attempts(work)
    cmd, env = command(out, p, d, arm)
    while True:
        if any(work.iterdir()):
            require(retryable(work), f'not a verified pre-reset failure: {work}')
            require(number < 3, f'three retries exhausted: {work}')
            number += 1
            root.mkdir(exist_ok=True, parents=True)
            work.rename(root/str(number))
            work.mkdir()
            time.sleep(2**number + secrets.randbelow(1000)/1000)
        with (work/'worker.log').open('x') as log:
            proc = subprocess.run(cmd, env=env, cwd=ev.ROOT, stdout=log, stderr=subprocess.STDOUT, pass_fds=(fd,))
        if proc.returncode == 0:
            return


def check_decision(guard, relay, payload, g, r, prior_handoff, mode):
    masks = g['avail_actions']
    for i,m in enumerate(masks):
        criteria = payload['questions'].get(f'u{i}',{}).get('criteria',{'0':None})
        require({str(a) for a,v in enumerate(m) if v} == set(criteria), 'mask differs from observed criteria')
    parent = guard.propose(payload, masks, g['v2_actions'])
    require(parent == g['skill_audit'] and all(g[k] == parent[k] for k in ('actions','phase','reason','handoff')),
            'R6 guard replay differs')
    base, menu, handoff = parent['actions'], r['candidates'], parent['handoff']
    require(r['mode'] == mode and r['avail_actions'] == masks and r['r6_actions'] == base and
            (handoff or not prior_handoff) and r['opening_handoff'] is (handoff and not prior_handoff) and
            (menu.get(r['r6_candidate_id']) == base if handoff else menu == {}), 'wrong relay inputs')
    for vector in menu.values():
        require(len(vector) == len(masks) and all(type(a) is int and 0 <= a < len(m) and m[a]
                for a,m in zip(vector,masks)), 'incomplete or illegal menu')
    pred = relay.propose(payload, masks, base, menu)
    require(all(r[k] == v for k,v in pred.items()), 'relay replay differs')
    actions = pred['actions']; changed = [f'u{i}' for i,(x,y) in enumerate(zip(actions,base)) if x != y]
    owner = 'opening' if not handoff else 'r6' if not changed else 'menu' if actions in menu.values() else 'custom'
    require(r['owner'] == owner and r['changed_actors'] == changed and (not changed or handoff) and
            (owner != 'custom' or mode == 'full'), 'wrong relay owner')
    return owner


def audit(out,p,d,ref,arm,original):
    from decision_tree.zerg.combat_relay_r8 import CombatRelayR8
    work = out/p['pair_id']/arm/'runner'
    meta, summary, episodes, score, damage, distill = (ev.read_json(work/k) for k in (
        'run.json','summary.json','episodes.json','score_details/episode-0001.json',
        'damage_by_type/episode-0001.json','distill/episode-0001.json'))
    ep, = episodes
    src, policy = d['sources'], d['policies'][arm]
    guard_id = sq.squad_contract('guard',src,policy['trial_id'])
    identity = dict(kind='combat_relay_r8', mode=arm, controller_sha256=src[RELAY], parent_mode='guard',
                    parent_controller_sha256=src[sq.CROSS_SOURCE], trial_id=policy['trial_id'])
    require(meta['combat_relay'] == identity and meta['squad_rendezvous'] == guard_id and
            all(meta[k] == original[k] for k in ev.RUN_KEYS) and 'swarm_route' not in meta and
            all(meta['zerg_distill'][k] == summary['zerg_distill'][k] == v for k,v in policy.items()), 'wrong deployed policy')
    required = set(original['source_hashes']) | {k for k in src if k.startswith('sc2_skill_adapter/bridge/')} | {
        RELAY,sq.SRC,sq.CROSS_SOURCE,'decision_tree/zerg/policy_distill.py','decision_tree/zerg/policy_repair_v2.py',
        'decision_tree/zerg/opening_support_v2.py'}
    require(required <= meta['source_hashes'].keys() and all(src.get(k) == v for k,v in meta['source_hashes'].items()),
            'worker source mismatch')
    sh_meta = meta['zerg_shadow']
    require(sh_meta['source_fingerprint'] == policy['source_hash'] and sh_meta['execution_context_hash'] == policy['execution_context_hash']
            and sh_meta['scene_config_id'] == p['roster'], 'wrong parent/shadow identity')
    n, start, initial = ep['steps'], score['initial']['game_loop'], ref['R6']['initial']
    require(ev.read_json(work/'manifest.json') == ev.read_json(out/p['manifest']) and
            p['manifest_sha256'] == meta['manifest_sha256'] == score['manifest_sha256'] == damage['manifest_sha256'] and
            p['scenario_sha256'] == ep['scenario_sha256'] == score['scenario_sha256'] == damage['scenario_sha256'] and
            score['sc2_seed'] == p['seed'] and ep['initial_state_sha256'] == initial, 'different manifest, seed or initial state')
    require(summary['status'] == 'complete' and summary['completed_episodes'] == ep['episode'] == 1 and
            ep['outcome'] == score['outcome'] in ('win','loss','draw','timeout') and n == score['steps'] > 0 and
            score['game_version'] == damage['game_version'] == '4.10.0.75689' and damage['begin_request_loop'] == start and
            damage['terminal_loop'] == score['final']['game_loop'] == start+8*n and damage['map_file'] == meta['map_file'] and
            damage['map_sha256'] == src[str(Path(meta['map_file']).relative_to(ev.ROOT))] and
            (work/'replays/episode-0001.SC2Replay').stat().st_size > 0, 'incomplete terminal evidence')
    shadow_path = work.parent/'shadow/shadow.jsonl'
    shadow = [json.loads(line) for line in shadow_path.read_text().splitlines()] if shadow_path.exists() else []
    native_rows = distill['decisions']
    require(distill['status'] == 'complete' and distill['scope'] == 'executed_native_relay_steps_only' and
            all(distill[k] == v for k,v in policy.items()) and distill['scenario_sha256'] == p['scenario_sha256'] and
            distill['initial_state_sha256'] == initial and distill['seed'] == p['seed'] and
            len(native_rows) == distill['steps'] == ep['zerg_shadow_decisions'] == len(shadow), 'invalid native receipt')
    guard = sq.controller_class('guard')(mode='guard'); relay = CombatRelayR8(mode=arm)
    counts = dict.fromkeys(('opening_steps','r6_steps','menu_steps','custom_steps','handoffs_to_r6'),0)
    i = j = starts = ends = macro = micro = 0
    reward = 0.; prev = None; handoff = False; frame = {}; records = []
    with (work/'events.jsonl').open() as stream:
        for line in stream:
            e = json.loads(line); k = e['event']; loop = start+i*8
            if k == 'episode_started':
                starts += 1
                require(e['initial_state_sha256'] == initial and e['scenario_sha256'] == p['scenario_sha256'] and
                        e['game_loop'] == start and i == 0, 'wrong reset receipt')
            elif k == 'episode_completed':
                ends += 1
                require(not frame and i == n and all(e[k] == v for k,v in ep.items()), 'wrong terminal receipt')
            elif k in ('skill_observation','squad_rendezvous_decision','combat_relay_decision','skill_step',
                       'relay_r6_proposal','squad_v2_proposal','micro_window_closed','step','combat_relay_receipt'):
                require(starts == 1 and not ends and e['episode'] == 1 and k not in frame, 'duplicate/misplaced relay event')
                if k == 'micro_window_closed':
                    require(e['macro_step'] == i and e['issued_loop'] == loop and e['closed_loop'] == loop+8 and
                            e['interior_checks'] == 7, 'wrong micro window')
                else:
                    require(e['step'] == i+(k == 'step') and (e['payload']['state']['game_loop'] if k == 'skill_observation'
                            else e['game_loop']) == loop, 'wrong relay clock')
                frame[k] = e
                if k != 'combat_relay_receipt':
                    continue
                g,r,w,s = (frame[k] for k in ('squad_rendezvous_decision','combat_relay_decision','micro_window_closed','step'))
                owner = check_decision(guard,relay,frame['skill_observation']['payload'],g,r,handoff,arm)
                handoff = g['handoff']; native = owner in ('menu','r6'); acts = r['actions']
                telemetry = 'skill_step' if native else 'squad_v2_proposal' if owner == 'opening' else 'relay_r6_proposal'
                require(list(frame) == ['skill_observation','squad_rendezvous_decision','combat_relay_decision',telemetry,
                        'micro_window_closed','step','combat_relay_receipt'], 'incorrect event sequence')
                prepared = acts if native else g['v2_actions']
                require(frame[telemetry]['actions'] == {f'u{i}':a for i,a in enumerate(prepared)} and
                        e['actions'] == s['actions'] == acts and s['terminated'] is (i+1 == n) and e['owner'] == owner,
                        'wrong owner, preparation or execution')
                handback = prev in ('menu','custom') and owner == 'r6'
                require(e['handoff_to_r6'] is handback, 'wrong handback receipt')
                counts['handoffs_to_r6'] += handback; counts[owner+'_steps'] += 1; prev = owner
                fb = s['feedback']; subs = fb['micro_submissions']; rec = w['records']
                grants = {a['actor']:a for a in e['micro_grants']}
                require(len(rec) == w['authorizations'] and e['micro_submissions'] == subs == [x for x in rec if x['status']=='submitted'] and
                        all(x['sc2_response'] == [1] and loop < x['submitted_loop'] < loop+8 and
                            x['macro_action_id'] == acts[int(x['actor'][1:])] for x in subs) and
                        all(x['actor'] in grants and all(x[k] == grants[x['actor']][k] for k in
                            ('actor_tag','target_tag','target','macro_action_id','issued_loop')) for x in rec) and
                        (owner != 'opening' or not (rec or e['micro_grants'])) and (native or not e['plan_ids']) and
                        (owner != 'custom' or not set(r['changed_actors']).intersection(a['actor'] for a in e['micro_grants'])),
                        'incorrect plan/micro receipt')
                macro += sq.sw.check_macro(acts,g['avail_actions'],dict(fb,micro_submissions=[]),loop)
                micro += len(subs); records.extend(rec); reward += s['reward']
                if native:
                    row, sh = native_rows[j], shadow[j]; j += 1
                    amap = {f'u{i}':a for i,a in enumerate(acts)}
                    cards = {x['candidate_id']:x for x in sh['candidates']}
                    require(row['step'] == i and row['game_loop'] == loop and row['committed'] is True and
                            row['execution_owner'] == owner and row['trial_id'] == policy['trial_id'] and
                            row['r6_candidate_id'] == r['r6_candidate_id'] and row['macro_receipt'] == 'accepted' and
                            row['micro_receipt'] == ('accepted' if subs else 'not_submitted') and row['plan_ids'] == e['plan_ids'] and
                            row['final_actions'] == sh['final_actions'] == amap and row['snapshot_id'] == sh['snapshot_id'] and
                            row['selected_candidate'] == sh['selected_candidate'] and sh['game_loop'] == loop and
                            sh['run_id'] == sh_meta['run_id'] and sh['source_hash'] == policy['source_hash'] and
                            sh['execution_context_hash'] == policy['execution_context_hash'] and sh['seed'] == p['seed'] and
                            sh['scene_config_id'] == p['roster'] and sh['outcome'] == ep['outcome'] and sh['return'] == ep['reward'] and
                            set(row['scores']) == set(r['candidates']) and all(cards[c]['legal'] is True and
                            cards[c]['proposed_actions'] == {f'u{i}':a for i,a in enumerate(vec)} for c,vec in r['candidates'].items()) and
                            r['candidates'][row['selected_candidate']] == acts, 'invalid native menu/commit evidence')
                i += 1; frame = {}
    require(starts == ends == 1 and not frame and i == n and reward == ep['reward'] and j == len(native_rows) and
            counts == ep['combat_relay'] == distill['combat_relay'] and records ==
            ev.read_json(work/'micro_triggers/episode-0001.json')['authorizations'] and ep['micro_trigger'] == summary['micro_trigger_totals'] ==
            dict(mode='micro8-all',macro_loops=8,engine_step_loops=1,windows=n,authorizations=len(records),
                 states=dict(Counter(r['status'] for r in records))), 'incomplete relay evidence')
    return dict(initial=initial,outcome=ep['outcome'],steps=n,reward=reward,enemy_elimination_fraction=ep['enemy_elimination_fraction'],
                macro_commands=macro,micro_submissions=micro,changed_steps=counts['menu_steps']+counts['custom_steps'],**counts)


def summarize(rows):
    arms = ('R6', *ARMS)
    stats, comparisons = {}, {}
    for a in arms:
        games = [r[a] for r in rows]
        wins = sum(g['outcome'] == 'win' for g in games)
        stats[a] = dict(wins=wins, win_rate=wins/len(rows) if rows else None,
                        outcomes=dict(Counter(g['outcome'] for g in games)),
                        severe_losses=sum(g['outcome'] == 'loss' and g['enemy_elimination_fraction'] <= .5 for g in games))
        if a in ARMS:
            stats[a]['changed_steps'] = sum(g['changed_steps'] for g in games)
            stats[a]['changed_games'] = sum(g['changed_steps'] > 0 for g in games)
    for a,b in combinations(arms, 2):
        t = Counter((r[a]['outcome'], r[b]['outcome']) for r in rows)
        gain = sum(n for (x,y),n in t.items() if x != 'win' and y == 'win')
        loss = sum(n for (x,y),n in t.items() if x == 'win' and y != 'win')
        n = gain+loss
        comparisons[f'{a}_to_{b}'] = dict(gained=gain, lost=loss, net=gain-loss,
            paired_win_exact_p=min(1., 2*sum(comb(n,i) for i in range(min(gain,loss)+1))/2**n) if rows else None,
            transitions={f'{x}_to_{y}':count for (x,y),count in t.items()})
    return dict(pairs=len(rows), arms=stats, comparisons=comparisons)


def report(out, d, selected, results):
    rows = []
    for p,r in zip(d['pairs'], selected):
        arms = {a:results[p['pair_id'], a] for a in ARMS}
        require(all(r[k] == v for k,v in p.items()), 'misaligned comparison row')
        require(all(g['initial'] == r['R6']['initial'] for g in arms.values()), 'different initial states')
        rows.append(dict(p, R6=r['R6'], **arms, strata=cohorts(r), original_R1=r['R1']['outcome'], original_R7=r['R7']['outcome']))
    require(len(rows) == len(d['pairs']) == len({r['scenario_sha256'] for r in rows}) and
            len(results) == 2*len(rows), 'duplicate or incomplete panel')
    return dict(**summarize(rows), kind=KIND, direct_parent='archived_R6', archive=ARCHIVE.name,
                archive_report_sha256=REPORT, archive_design_sha256=DESIGN, design_sha256=ev.sha(out/'design.json'),
                new_games=2*len(rows), archived_games=len(rows), pilot_ids=d['pilot_ids'],
                strata={c:summarize([r for r in rows if c in r['strata']]) for c in COHORTS},
                paired_outcomes=rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group()
    target.add_argument('--out', type=Path); target.add_argument('--resume', type=Path)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--pilot-ids', nargs='+', help='1-based archived IDs, space- or comma-separated')
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 60 or args.resume and args.pilot_ids:
        parser.error('workers must be 1..60; resume uses frozen pilot IDs')
    if not (args.dry_run or args.out or args.resume):
        parser.error('choose --out DIR or --resume DIR')
    ids = [int(i) for group in args.pilot_ids for i in group.split(',')] if args.pilot_ids else None
    require(not ids or len(ids) == len(set(ids)) and all(1 <= i <= 1000 for i in ids), 'invalid pilot IDs')
    parent, rows = archive()
    if not (args.out or args.resume):
        print(f'Archived R6: {len(rows)} verified pairs; {(len(ids) if ids else 1000)*2} planned sibling games. No workers launched.')
        return
    original, cfg, versions, src = runtime(parent)
    out = (args.out or args.resume).resolve()
    if args.out:
        policies = {a:dict(parent['policies']['R6'], trial_id=secrets.token_hex(16)) for a in ARMS}
        sq.sw.trees.freeze_panel(out,args.workers,None,args.dry_run,ARCHIVE.name,None,ids,policies,src,cfg,versions,
                                 {ARCHIVE.name:DESIGN},KIND,panel_size=1000,archive_report_sha256=REPORT)
    with sq.sw.run_lock(out) as fd:
        d = ev.read_json(out/'design.json'); ids = d['pilot_ids'] or range(1,1001)
        require(d['kind'] == KIND and d['arms'] == list(ARMS) and d['sources'] == src and d['versions'] == versions and
                d['replay_batch'] == ARCHIVE.name and d['replay_design_sha256'] == DESIGN and d['archive_report_sha256'] == REPORT and
                ev.read_json(ev.HISTORY/f'{ev.digest(d)}.json') == d and ev.sha(out/'model.json') == ev.MODEL_HASH and
                d['pairs'] == [parent['pairs'][i-1] for i in ids], 'changed design, source or panel')
        require(args.dry_run or not d['dry_run'], 'cannot play dry-run reservation')
        selected = [rows[i-1] for i in ids]; pending = []; results = {}
        for i,p,r in zip(ids,d['pairs'],selected):
            sq.sw.trees.validate_pair(p,ev.read_json(out/p['manifest']),i,cfg)
            for a in ARMS:
                work = out/p['pair_id']/a
                require(work.is_dir() and work.resolve() == work, 'invalid arm directory')
                _, n = attempts(work)
                if sq.sw.completed(work):
                    results[p['pair_id'],a] = audit(out,p,d,r,a,original)
                else:
                    require(not any(work.iterdir()) or retryable(work) and n < 3, f'ineligible incomplete attempt: {work}')
                    pending.append((p,r,a))
        print(f'{out}: {len(results)} audited; {len(pending)} pending', flush=True)
        target = out/'report.json'
        require(not pending or not target.exists(), 'final report on incomplete panel')
        if args.dry_run and pending:
            return
        def worker(p,r,a):
            play(out,p,d,a,fd)
            return audit(out,p,d,r,a,original)
        with ThreadPoolExecutor(args.workers) as pool:
            jobs = {pool.submit(worker,p,r,a):(p['pair_id'],a) for p,r,a in pending}
            try:
                for job in as_completed(jobs):
                    results[jobs[job]] = job.result()
            except BaseException:
                for job in jobs:
                    job.cancel()
                raise
        rep = report(out,d,selected,results)
        if target.exists():
            require(ev.read_json(target) == rep, 'existing report differs')
        elif not args.dry_run:
            tmp = out/'report.json.tmp'; ev.write_json(tmp,rep); tmp.replace(target)
        print(json.dumps({k:v for k,v in rep.items() if k != 'paired_outcomes'}))
        return rep


if __name__ == '__main__':
    main()
