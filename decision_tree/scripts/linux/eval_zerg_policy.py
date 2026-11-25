"""Evaluate a frozen distilled policy against its original rule on 240 matched games per arm."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import hashlib
import json
import math
import os
import random
from pathlib import Path
import re
import secrets
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from decision_tree.scripts.linux.run_zerg_whole_ac import (
    ADAPTER, HISTORY as AC_HISTORY, KEYS, MAP_DIR, PROTOCOL, RSI_PYTHON, SC2_PATH,
    SHADOW, digest, generate, historical_config, originals, read_json, write_json)

RUNS = ROOT / 'runs'
HISTORY = RUNS / '.zerg_policy_eval_history'
MODEL = RUNS / 'zerg_policy_distill_v1/model.json'
MODEL_HASH = '1243710f83f14a171799855befb804122591acd150b0477dfd759b07641eb578'
ORIGINAL = RUNS / ('zerg_shadow_az17j4ru/workers/zerg_4zergling_0baneling_6hydralisk/'
                   'episode-001-seed-20261123/runner/run.json')
REFERENCE_HASH = '4f5e94f5fc308e0f71613a708a61d180009cbadd6fe3f0321124eaf92d3f2dd2'
SPLIT_HASH = '92a8b695f48702f30f314588962fac0df79f188bd64dfafcb66ca035568093b7'
ARMS = ('baseline', 'tree')
MODES = ('model_base', 'model_switch', 'fallback')
RUN_KEYS = (*PROTOCOL, 'macro_loops', 'engine_step_loops', 'skill_source_hashes', 'skill_parameters',
            'env_args', 'observation_protocol', 'native_teacher_control', 'damage_by_type_enabled',
            'versions', 'map_file')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reference():
    require(sha(ORIGINAL) == REFERENCE_HASH, 'historical run.json changed')
    return read_json(ORIGINAL)


def model_contract(path, original):
    from decision_tree.zerg.policy_distill import load_model
    require(sha(path) == MODEL_HASH, 'unexpected distilled model SHA256')
    m = load_model(path)
    s = original['zerg_shadow']
    require((m['source_hash'], m['ctx_hash'], m['split_hash']) ==
            (s['source_fingerprint'], s['execution_context_hash'], SPLIT_HASH),
            'model does not match original policy source / execution context / training split')
    return dict(source_hash=m['source_hash'], execution_context_hash=m['ctx_hash'],
                projection_version=m['schema_hash'], model_sha256=MODEL_HASH, split_hash=SPLIT_HASH)


def random_team(seed, banelings):
    rng = random.Random(f'zerg-conditional-v1:{seed}:{banelings}')
    z = rng.randrange(11 - banelings)
    h = 10 - banelings - z
    team = ['zergling'] * z + ['baneling'] * banelings + ['hydralisk'] * h
    rng.shuffle(team)
    return f'zerg_{z}zergling_{banelings}baneling_{h}hydralisk', team


def six_rosters(seed, batch):
    path = RUNS / batch / 'design.json'
    require(batch in SHADOW and sha(path) == SHADOW[batch], 'invalid original roster batch')
    historical = read_json(path)['rosters']
    require(len(historical) == 6, 'expected six historical rosters')
    rng = random.Random(f'six-rosters-v1:{batch}:{seed}')
    rosters = {}
    for old in historical.values():
        b = old.count('baneling')
        choices = [(f'zerg_{z}zergling_{b}baneling_{10-b-z}hydralisk', z) for z in range(11-b)]
        name, z = rng.choice([item for item in choices if item[0] not in rosters])
        team = ['zergling']*z + ['baneling']*b + ['hydralisk']*(10-b-z)
        rng.shuffle(team)
        rosters[name] = team
    return rosters


def native_team(config, seed):
    from bridge.scenarios import generate_manifest
    cfg = generate_manifest(config, 1, seed)['episodes'][0]['episode_config']
    team = cfg['team_gen']['ally_team']
    name = 'zerg_' + '_'.join(f'{team.count(t)}{t}' for t in ('zergling','baneling','hydralisk'))
    return name, team


def frozen_sources(original):
    # Pin the inference implementation as well as the model and resolved parent.
    paths = [*ADAPTER.joinpath('bridge').glob('*.py'), *ROOT.joinpath('decision_tree/zerg').glob('*.py')]
    sources = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    for key, expected in original['source_hashes'].items():
        # The runner now has the opt-in distill hook; all other historical files stay byte-identical.
        if key != 'sc2_skill_adapter/bridge/runner.py':
            require(sha(ROOT / key) == expected, f'historical source mismatch: {key}')
            sources[key] = expected
    return sources


def prepare(out, model, workers, seed, dry_run, banelings=None, roster_batch=None, all_random=False):
    require(sys.executable == RSI_PYTHON, f'use historical interpreter {RSI_PYTHON}')
    require(1 <= workers <= 60, 'workers must be in 1..60')
    config, versions = historical_config()
    original = reference()
    contract = model_contract(model, original)
    sources = frozen_sources(original)
    rosters, seen = originals()
    require(len(rosters) == 12, 'expected 12 original rosters')
    limit = 2**31 - 240
    seed = secrets.randbelow(limit + 1) if seed is None else seed
    require(0 <= seed <= limit, 'seed range must fit signed nonnegative int32')
    HISTORY.mkdir(parents=True, exist_ok=True)
    with (HISTORY/'lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for history in (AC_HISTORY, HISTORY):
            for path in history.glob('*.json'):
                prior = read_json(path)
                require(path.stem == digest(prior), f'corrupt reservation: {path}')
                for p in prior['pairs']:
                    for k in KEYS:
                        seen[k].add(p[k])
        if roster_batch is not None:
            require(banelings is None, 'choose one roster mode')
            rosters = six_rosters(seed, roster_batch)
            pairs, manifests = generate(config, rosters, 40, seed, seen)
        elif banelings is None and not all_random:
            pairs, manifests = generate(config, rosters, 20, seed, seen)
        else:
            require(all_random or 0 <= banelings < 10, 'fixed banelings must be in 0..9 (at least one non-baneling)')
            pairs, manifests, rosters = [], [], {}
            for i in range(240):
                name, team = native_team(config,seed+i) if all_random else random_team(seed+i, banelings)
                rosters[name] = sorted(team)
                (pair,), (manifest,) = generate(config, {name:team}, 1, seed+i, seen)
                pid = f'pair-{i+1:03d}'
                pair.update(pair_id=pid, manifest=f'{pid}/manifest.json')
                pairs.append(pair); manifests.append(manifest)
        design = dict(kind='zerg_policy_eval_v1', dry_run=dry_run, arms=list(ARMS), pairs=pairs,
                      seed=seed, workers=workers, rosters=rosters, python=RSI_PYTHON, versions=versions,
                      sc2_path=SC2_PATH, damage_map_dir=str(MAP_DIR), shadow_design_hashes=SHADOW,
                       historical_run_sha256=REFERENCE_HASH, model_contract=contract, sources=sources, **PROTOCOL)
        if banelings is not None:
            design['fixed_banelings'] = banelings
        if roster_batch is not None:
            design['roster_batch'] = roster_batch
        if all_random:
            design['native_random_mirror'] = True
        out.mkdir(parents=True, exist_ok=False)
        (out/'model.json').write_bytes(model.read_bytes())
        require(sha(out/'model.json') == contract['model_sha256'], 'model changed during freeze')
        write_json(HISTORY/f'{digest(design)}.json', design)
    for pair, manifest in zip(pairs, manifests):
        for arm in ARMS:
            (out/pair['pair_id']/arm).mkdir(parents=True)
        write_json(out/pair['manifest'], manifest)
    write_json(out/'design.json', design)
    return design


def worker_environment():
    env = {k: v for k, v in os.environ.items() if not k.startswith(('ZERG_', 'SMAC_', 'PYTHON'))}
    env.update(SC2PATH=SC2_PATH, SMAC_DAMAGE_MAP_DIR=str(MAP_DIR), OPENBLAS_NUM_THREADS='1',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1')
    for k in ('NO_PROXY', 'no_proxy'):
        env[k] = ','.join(filter(None, (env.get(k), '127.0.0.1,localhost,::1')))
    return env


def command(out, pair, arm, policy=None):
    require(arm in ARMS if policy is None else arm in ('v1', 'v2', 'v3','alternative','route') and policy['policy_version'] == arm, 'unknown arm')
    work = out/pair['pair_id']/arm
    env = worker_environment()
    env.update(ZERG_SHADOW_ROSTER=pair['roster'], ZERG_SHADOW_DIR=str(work/'shadow'))
    if arm == 'tree' or policy is not None:
        env['ZERG_DISTILL_MODEL'] = str(out/'model.json')
        env['ZERG_DISTILL_MODEL_SHA256'] = MODEL_HASH
        if policy is not None:
            env.update(ZERG_DISTILL_VERSION=arm, ZERG_DISTILL_SELECTOR_SHA256=policy['selector_sha256'])
            if 'opening_sha256' in policy:
                require(arm != 'v1','opening proposal on v1 arm')
                env['ZERG_DISTILL_OPENING_SHA256']=policy['opening_sha256']
            if 'trial_id' in policy:
                env['ZERG_DISTILL_TRIAL_ID'] = policy['trial_id']
            if 'policy_sha256' in policy:
                env['ZERG_ROUTE_IDENTITY'] = json.dumps(policy['policy_sha256'])
            if arm == 'route':
                env.update(ZERG_ROUTE_MODEL=str(out/'gate.json'), ZERG_ROUTE_MODEL_SHA256=policy['gate_sha256'])
    cmd = [RSI_PYTHON, '-B', '-u', str(ADAPTER/'run.py'), 'run', '--method', PROTOCOL['method'],
           '--skill-version', PROTOCOL['skill_version'], '--execution-mode', PROTOCOL['execution_mode'],
           '--manifest', str(out/pair['manifest']), '--output-dir', str(work/'runner')]
    return cmd, env


def play(out, pair, arm, policy=None):
    cmd, env = command(out, pair, arm, policy)
    with (out/pair['pair_id']/arm/'worker.log').open('x') as log:
        subprocess.run(cmd, cwd=ROOT, env=env, check=True, stdout=log, stderr=subprocess.STDOUT)


def audit_decisions(receipt, events, stats):
    counts = Counter(dict.fromkeys((*MODES, 'steps', 'engine_unconfirmed', 'filtered_candidates'), 0))
    reasons = Counter()
    rows = receipt['decisions']
    require(len(rows) == len(events) > 0, 'missing decision receipt')
    for i, (r, e) in enumerate(zip(rows, events)):
        actions = {f'u{j}': a for j, a in enumerate(e['actions'])}
        mode = r['mode']
        require(r['step'] == i and r['game_loop'] == e['game_loop'] and mode in MODES
                and r['committed'] is True and r['final_actions'] == actions, f'wrong decision/engine join: step {i}')
        require((r['parent_actions'] != actions) == (mode == 'model_switch'), 'invalid switch count')
        reason = r['fallback_reason']
        require((isinstance(reason, str) and bool(reason)) if mode == 'fallback' else reason is None,
                'invalid fallback reason')
        if reason:
            reasons[reason] += 1
        macro, micro = r['macro_receipt'], r['micro_receipt']
        require(macro in ('accepted', 'unknown') and micro in ('accepted', 'not_submitted', 'unknown'),
                'unknown receipt status')
        counts[mode] += 1
        counts['steps'] += 1
        counts['engine_unconfirmed'] += 'unknown' in (macro, micro)
        counts['filtered_candidates'] += len(r['filtered'])
    require(stats == receipt['stats'] and all(type(stats[k]) is int and stats[k] == v
            for k, v in counts.items()), 'receipt and episode counters differ')
    return dict(counts), dict(reasons)


def audit_arm(out, pair, arm, manifest, original, design, policy=None, work_override=None):
    is_tree = arm == 'tree' or policy is not None
    if policy is not None:
        require(arm in ('v1', 'v2', 'v3','alternative','route') and arm == policy['policy_version'], 'incorrect policy arm')
    work = work_override if work_override is not None else out/pair['pair_id']/arm
    runner = work/'runner'
    meta, summary = [read_json(runner/f'{name}.json') for name in ('run', 'summary')]
    ep, = read_json(runner/'episodes.json')
    require(read_json(runner/'manifest.json') == manifest and meta['manifest_sha256'] == manifest['sha256']
            and ep['scenario_sha256'] == pair['scenario_sha256'], 'different worker manifest/scenario')
    require(all(meta[k] == original[k] for k in RUN_KEYS), 'historical protocol or skill source mismatch')
    required = set(original['source_hashes']) | {k for k in design['sources'] if k.startswith('sc2_skill_adapter/bridge/')}
    if is_tree:
        if policy is None:
            require(meta['zerg_distill'].get('kind') == 'zerg_distill_v1', 'incorrect historical tree version')
        required.add('decision_tree/zerg/policy_distill.py')
    if policy and 'policy_sha256' in policy:
        required.update(f'decision_tree/zerg/{name}.py' for name in ('policy_route_alt','policy_route_gate'))
    if arm in ('v2', 'v3','alternative','route'):
        required.add('decision_tree/zerg/policy_repair_v2.py')
        if 'opening_sha256' in policy: required.add('decision_tree/zerg/opening_support_v2.py')
        if arm == 'v3':
            required.add('decision_tree/zerg/policy_repair_v3.py')
        else:
            require('decision_tree/zerg/policy_repair_v3.py' not in meta['source_hashes'], 'v3 on v2 arm')
    if arm == 'v1':
        require('decision_tree/zerg/policy_repair_v2.py' not in meta['source_hashes'], 'v2 source on v1 arm')
    require(all(meta['source_hashes'].get(k) == design['sources'][k] for k in required)
            and all(h == design['sources'].get(k) for k, h in meta['source_hashes'].items()), 'worker source mismatch')
    contract = design['model_contract'] if policy is None else policy
    shadow = meta['zerg_shadow']
    steps = ep['steps']
    require(summary['status'] == 'complete' and ep['episode'] == 1 and type(steps) is int and steps > 0
            and ep['outcome'] in ('win', 'loss', 'timeout', 'draw') and math.isfinite(ep['reward'])
            and re.fullmatch('[a-f0-9]{64}', ep['initial_state_sha256'])
            and ep['zerg_shadow_decisions'] == steps and summary['zerg_shadow']['stats']['model_errors'] == 0,
            'incomplete or invalid game')
    require(shadow['source_fingerprint'] == contract['source_hash'] and
            shadow['execution_context_hash'] == contract['execution_context_hash'] and
            shadow['scene_config_id'] == pair['roster'], 'wrong parent provenance')
    score = read_json(runner/'score_details/episode-0001.json')
    require(score['sc2_seed'] == pair['seed'] and score['game_version'] == '4.10.0.75689'
            and score['manifest_sha256'] == manifest['sha256'] and score['outcome'] == ep['outcome']
            and score['steps'] == steps, 'engine seed/version/result mismatch')
    events, starts = [], []
    with (runner/'events.jsonl').open() as stream:
        for line in stream:
            if line.startswith('{"event": "step",'):
                events.append(json.loads(line))
            elif line.startswith('{"event": "episode_started",'):
                starts.append(json.loads(line))
    start, = starts
    require(all(start[k] == ep[k] for k in ('episode', 'initial_state_sha256', 'scenario_sha256'))
            and start['game_version'] == score['game_version'], 'wrong initial-state evidence')
    require(len(events) == steps and all(e['episode'] == 1 and e['step'] == i+1
            and e['game_loop'] == score['initial']['game_loop'] + 8*i for i, e in enumerate(events)),
            'incomplete micro8-all macro-step coverage')
    i = 0
    shadow_join = []
    with (work/'shadow/shadow.jsonl').open() as stream:
        for i, line in enumerate(stream, 1):
            r = json.loads(line)
            require(i <= steps and r['game_loop'] == events[i-1]['game_loop'] and r['run_id'] == shadow['run_id']
                    and r['seed'] == pair['seed'] and r['source_hash'] == contract['source_hash']
                    and r['execution_context_hash'] == contract['execution_context_hash']
                    and r['return'] == ep['reward'] and r['outcome'] == ep['outcome']
                    and r['scene_config_id'] == pair['roster'] and r['final_actions'] ==
                    {f'u{j}': a for j, a in enumerate(events[i-1]['actions'])}, 'wrong shadow decision')
            if arm == 'baseline':
                require(r['selected_candidate'] == r['teacher_choice'], 'baseline differs from teacher')
            if policy is not None:
                shadow_join.append((r['snapshot_id'], r['selected_candidate'], r['teacher_choice']))
        require(i == steps, 'missing shadow decisions')
    result = dict(outcome=ep['outcome'], reward=ep['reward'], steps=steps, initial=ep['initial_state_sha256'])
    for obj in (meta, summary, ep):
        require(('zerg_distill' in obj) == is_tree, 'incorrect policy arm')
    if is_tree:
        require(all(obj['zerg_distill'][k] == v for obj in (meta, summary) for k, v in contract.items()),
                'wrong deployed model identity')
        receipt = read_json(runner/'distill/episode-0001.json')
        expected = dict(contract, status='complete', seed=pair['seed'], steps=steps, roster=pair['roster'],
                        scenario_sha256=pair['scenario_sha256'], initial_state_sha256=ep['initial_state_sha256'])
        require(all(receipt[k] == v for k, v in expected.items()), 'stale tree receipt')
        counts, reasons = audit_decisions(receipt, events, ep['zerg_distill'])
        require(all(summary['zerg_distill']['stats'][k] == v for k, v in counts.items()), 'summary counts differ')
        result.update(decisions=counts, fallback_reasons=reasons)
        if policy is not None:
            require(counts['engine_unconfirmed'] == 0, 'missing engine acknowledgement')
            selected = Counter()
            for row, join, event in zip(receipt['decisions'], shadow_join, events):
                require((row['snapshot_id'], row['selected_candidate'], row['teacher_choice']) == join and
                        row['policy_version'] == arm and row['selector_sha256'] == policy['selector_sha256'] and
                        isinstance(row['selection_reason'], str) and row['selection_reason'].strip() and
                        row['selected_candidate'] in row['scores'] and
                        abs(row['scores'][row['selected_candidate']] - max(row['scores'].values())) <= 1e-12,
                        'invalid selector receipt')
                feedback = event['feedback']
                require(feedback['from_game_loop'] == row['game_loop'] and feedback['response_mapping'] == 'exact'
                        and feedback['actors'].keys() == row['final_actions'].keys() and
                        all(code == 1 for code in feedback['raw_response_results']), 'incomplete engine feedback')
                for u, a in row['final_actions'].items():
                    actor = feedback['actors'][u]
                    code = (actor.get('response') or {}).get('code')
                    require(str(actor['requested_action_id']) == str(a) and
                            ((actor['submission'] == 'accepted' and actor['command'] and code == 1) or
                             (actor['submission'] == 'dead_noop' and a == 0 and not actor['command'] and code is None)),
                            'incomplete macro acknowledgement')
                micro = feedback.get('micro_submissions', [])
                require(row['micro_receipt'] == ('accepted' if micro else 'not_submitted') and
                         all(sub['sc2_response'] == [1] for sub in micro), 'incomplete micro acknowledgement')
                if 'trial_id' in policy:
                    indices = [actor['command_index'] for actor in feedback['actors'].values()
                               if actor['submission'] == 'accepted']
                    require(sorted(indices) == list(range(len(feedback['raw_response_results']))),
                            'raw command acknowledgements are not one-to-one')
                    require(len({sub['actor'] for sub in micro}) == len(micro) and all(
                            sub['episode'] == 1 and sub['status'] == 'submitted' and sub['macro_step'] == row['step'] and
                            sub['issued_loop'] == row['game_loop'] < sub['submitted_loop'] <= sub['expires_loop'] == row['game_loop'] + 7 and
                            sub['target'] == f"e{sub['submitted_action_id'] - 6}" and
                            sub['macro_action_id'] == row['final_actions'][sub['actor']] for sub in micro),
                            'micro receipts differ from selected native macro')
                selected[row['selection_reason']] += 1
            require(ep['enemy_elimination_fraction'] == ep['info']['dead_enemies'] / 10 and
                    0 <= ep['info']['dead_enemies'] <= 10, 'invalid elimination fraction')
            result.update(selection_reasons=dict(selected), enemy_elimination_fraction=ep['enemy_elimination_fraction'])
    return result


def aggregate(rows):
    n = len(rows)
    bw = sum(r['baseline']['outcome'] == 'win' for r in rows)
    tw = sum(r['tree']['outcome'] == 'win' for r in rows)
    gains = sum(r['baseline']['outcome'] != 'win' and r['tree']['outcome'] == 'win' for r in rows)
    delta = sum(r['tree']['reward'] - r['baseline']['reward'] for r in rows)
    counts, reasons = Counter(), Counter()
    for r in rows:
        counts.update(r['tree']['decisions'])
        reasons.update(r['tree']['fallback_reasons'])
    return dict(pairs=n, baseline_wins=bw, tree_wins=tw, baseline_win_rate=bw/n, tree_win_rate=tw/n,
                win_flips=dict(gained=gains, lost=gains+bw-tw, net=tw-bw), mean_return_delta=delta/n,
                total_return_delta=delta, tree_decisions=dict(counts), fallback_reasons=dict(reasons))


def compare(out):
    from bridge.scenarios import validate_manifest
    d, original = read_json(out/'design.json'), reference()
    rosters, seen = originals()
    fixed = d.get('fixed_banelings')
    native = d.get('native_random_mirror', False)
    if native:
        require(native is True and fixed is None and 'roster_batch' not in d, 'conflicting random modes')
        config, _ = historical_config()
        teams = [native_team(config,d['seed']+i) for i in range(240)]
        counts = Counter(name for name,_ in teams)
        rosters = {name:sorted(team) for name,team in teams}
    elif 'roster_batch' in d:
        require(fixed is None, 'conflicting roster modes')
        rosters = six_rosters(d['seed'], d['roster_batch'])
        counts = Counter(dict.fromkeys(rosters,40))
    elif fixed is None:
        counts = Counter(dict.fromkeys(rosters,20))
    else:
        require(type(fixed) is int and 0 <= fixed < 10, 'invalid fixed banelings')
        teams = [random_team(d['seed']+i, fixed) for i in range(240)]
        counts = Counter(name for name,_ in teams)
        rosters = {name:sorted(team) for name,team in teams}
    require(d['dry_run'] is False and d['rosters'] == rosters and d['historical_run_sha256'] == REFERENCE_HASH
            and Counter(p['roster'] for p in d['pairs']) == counts
            and d['model_contract'] == model_contract(out/'model.json', original)
            and (HISTORY/f'{digest(d)}.json').is_file(), 'not a complete frozen evaluation design')
    rows = []
    for i, p in enumerate(d['pairs'], 1):
        try:
            pid = f'pair-{i:03d}'
            require(p['pair_id'] == pid and p['manifest'] == f'{pid}/manifest.json', 'bad pair id')
            m = read_json(out/p['manifest'])
            validate_manifest(m)
            ep, = m['episodes']
            cfg = ep['episode_config']
            team = rosters[p['roster']]
            if fixed is not None or native:
                name, team = teams[i-1]
                require(name == p['roster'] and p['seed'] == d['seed']+i-1, 'random-team assignment changed')
            require(m['env_args'] == original['env_args'] and m['seed'] == ep['sc2_seed'] == p['seed']
                    and m['sha256'] == p['manifest_sha256'] and ep['sha256'] == p['scenario_sha256']
                    and p['layout_sha256'] == digest([cfg[f'{s}_start_positions']['item'] for s in ('ally','enemy')])
                    and all(cfg['team_gen'][k] == team for k in ('ally_team', 'enemy_team')),
                    'pair design/manifest mismatch')
            for k in KEYS:
                require(p[k] not in seen[k], f'duplicate/training {k}')
                seen[k].add(p[k])
            arms = {a: audit_arm(out, p, a, m, original, d) for a in ARMS}
            require(arms['baseline']['initial'] == arms['tree']['initial'], 'observed initial states differ')
            rows.append(dict(pair_id=pid, roster=p['roster'], seed=p['seed'], **arms))
        except (OSError, ValueError, TypeError, KeyError) as exc:
            raise ValueError(f'pair {i}: {exc}') from exc
    return dict(**aggregate(rows), model=d['model_contract'], by_roster={r: aggregate(
        [row for row in rows if row['roster'] == r]) for r in rosters}, paired_outcomes=rows)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--out', type=Path, help='new run directory')
    mode.add_argument('--compare', type=Path, help='re-audit completed run and print report JSON')
    p.add_argument('--model', type=Path, default=MODEL)
    p.add_argument('--workers', default=60, type=int)
    roster_mode = p.add_mutually_exclusive_group()
    roster_mode.add_argument('--all-random', action='store_true', help='draw a fresh native Zerg composition and layout per pair, with mirrored teams')
    roster_mode.add_argument('--banelings', type=int, help='fix this count per side; randomize the remaining composition and order per pair')
    roster_mode.add_argument('--roster-batch', choices=list(SHADOW), help='reuse six baneling counts; sample six distinct teams once, 40 paired scenes each')
    p.add_argument('--seed', type=int, help='first fresh seed (default random)')
    p.add_argument('--dry-run', action='store_true', help='freeze model and reserve 240 scenes; no SC2')
    args = p.parse_args(argv)
    try:
        if args.compare:
            require(not args.dry_run, '--dry-run is for a new design')
            print(json.dumps(compare(args.compare.resolve()), indent=2, allow_nan=False))
            return
        out = args.out.resolve()
        extra = [] if args.banelings is None else [args.banelings]
        if args.roster_batch is not None:
            extra = [None,args.roster_batch]
        if args.all_random:
            extra = [None,None,True]
        d = prepare(out, args.model.resolve(), args.workers, args.seed, args.dry_run, *extra)
        print(f"{out}: 240 pairs, 480 games, seed={d['seed']}, workers={args.workers}", flush=True)
        if args.dry_run:
            print('Dry-run: SC2 not launched.')
            return
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            jobs = {pool.submit(play, out, pair, a): (pair['pair_id'], a) for pair in d['pairs'] for a in ARMS}
            for future in as_completed(jobs):
                try:
                    future.result()
                except Exception as exc:
                    for f in jobs:
                        f.cancel()
                    raise RuntimeError(f'{jobs[future]} failed; see worker.log') from exc
        result = compare(out)
        write_json(out/'report.json', result)
        print(json.dumps({k: v for k, v in result.items() if k not in ('by_roster', 'paired_outcomes')}, indent=2))
    except (ImportError, KeyError, OSError, ValueError, TypeError) as exc:
        p.error(str(exc))


if __name__ == '__main__':
    main()
