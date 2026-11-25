"""Launch a frozen one-step A/B/C plan, or independently compare its complete triplets.

--dry-run freezes the same plan, design and specs as a live launch; it never starts SC2.
--compare RUN writes descriptive paired differences to stdout, not a gain dataset.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from decision_tree.scripts.linux import run_zerg_forks as ab
from decision_tree.zerg.schema import digest

ABC = 'zerg_abc_v1'
CARD_PATH = ab.ROOT / 'decision_tree/zerg/configs/dual_lane_trial_v1.json'
ARMS = ('A', 'B', 'C')
CONTRASTS = (('B', 'A'), ('C', 'A'), ('C', 'B'))
INPUT_ERRORS = (OSError, KeyError, ValueError, TypeError)
PREFLIGHT_ERRORS = INPUT_ERRORS + (ImportError,)


def rule_identity():
    from decision_tree.zerg.rules_v1 import RULE_ID, rule_hash
    return dict(rule_id=RULE_ID, rule_hash=rule_hash())


def validate_frozen_plan(plan):
    rule = rule_identity()
    ab.require_fields(plan, dict(kind='zerg_fork_review_plan_v1', abc_protocol=ABC,
                                  horizon_steps=1, **rule), 'ABC plan')
    card = ab.read_json(CARD_PATH)
    ab.require_fields(card, dict(rule_id=rule['rule_id'], parent_source=plan['source_hash'],
                                 parent_context=plan['collection_context']), CARD_PATH)
    card_hash = digest(card)
    if 'rule_card' in plan or 'rule_card_hash' in plan:
        ab.require_fields(plan, dict(rule_card=card, rule_card_hash=card_hash), 'trial card')
    if 'screening' in plan:
        ab.require_fields(plan['screening'], dict(min_complete=24, mean_delta_C_A_must_exceed=0,
            mean_delta_C_B_must_exceed=0, net_win_flips_C_A_minimum=0, max_C_A_drops_at_least_5=1), 'screening')
    if not isinstance(plan['pairs'], list) or not plan['pairs']:
        raise ValueError('expected nonempty pairs')
    # Guard every original path using split metadata before opening any episodes
    # or passing even a smoke subset of this plan to the historical validator.
    split = ab.read_json(ab.ROOT / plan['fit'] / 'split.json')
    ab.require_fields(plan, dict(split_hash=digest(split)), 'split')
    seen = set()
    for i, p in enumerate(plan['pairs'], 1):
        group = p['episode_group']
        if group != f"{p['run_id']}:{p['episode_id']}" or group in seen:
            raise ValueError(f'anchor {i}: invalid or repeated episode group')
        seen.add(group)
        ep = split[group]
        ab.require_fields(ep, dict(partition='train', seed=p['seed'], roster=p['roster']), group)
        ab.require_fields(p, dict(manifest=str(ab.ROOT / ep['batch'] / ep['worker'] / 'manifest.json')), group)
        a, b, c = (p[arm]['actions'] for arm in ARMS)
        ab.require_fields(p['C'], dict(candidate_id=rule['rule_id'], source_plan_id=rule['rule_id']), group)
        if (not all(isinstance(actions, dict) and actions for actions in (a, b, c))
                or c.keys() != a.keys() or c.keys() != b.keys()
                or any(type(v) is not int or v < 0 for v in c.values()) or c in (a, b)
                or rule['rule_id'] in (p['A']['candidate_id'], p['B']['candidate_id'])):
            raise ValueError(f'anchor {i}: C is not a distinct complete action proposal')
    return card_hash


def indices_for(plan, index):
    n = len(plan['pairs'])
    if index is None:
        return list(range(1, n + 1))
    if type(index) is not int or not 1 <= index <= n:
        raise ValueError(f'anchor index must be in 1..{n}')
    return [index]


def spec_for(arm, pair, plan):
    spec = dict(arm=arm, pair=pair, source_hash=plan['source_hash'], collection_context=plan['collection_context'])
    if arm == 'C':
        spec['rule_hash'] = plan['rule_hash']
    return spec


def arm_facts(work, arm, p, plan):
    if digest(ab.read_json(work / 'fork_spec.json')) != digest(spec_for(arm, p, plan)):
        raise ValueError('spec differs from frozen plan')
    receipt = ab.validate_arm(work, arm, p, plan)
    ab.require_fields(receipt, dict(engine_receipt='accepted'), work)
    if arm == 'C':
        ab.require_fields(receipt, dict(rule_hash=plan['rule_hash']), work)
        ids = receipt['validated_plan_ids']
        if not isinstance(ids, list) or not ids or not all(isinstance(k, str) and k for k in ids):
            raise ValueError('missing native C plan evidence')
    original = Path(p['manifest']).parent
    context = dict(scene_config_id=p['roster'], source_fingerprint=plan['source_hash'],
                   execution_context_hash=plan['collection_context'])
    ab.require_fields(ab.read_json(original / 'runner/run.json')['zerg_shadow'],
                      dict(run_id=p['run_id'], **context), original)
    run = ab.read_json(work / 'runner/run.json')['zerg_shadow']
    run_id = run['run_id']
    if not isinstance(run_id, str) or not run_id or run_id == p['run_id']:
        raise ValueError('invalid replay run id')
    ab.require_fields(run, context, work)
    ab.require_fields(ab.read_json(work / 'runner/summary.json')['zerg_shadow'], dict(run_id=run_id), work)
    with (work / 'shadow/shadow.jsonl').open(encoding='utf-8') as stream:
        # validate_arm already checked that this anchor occurs exactly once.
        anchor = next((r for r in map(json.loads, stream) if
                       (r['snapshot_id'], r['game_loop']) == (p['snapshot_id'], p['game_loop'])), {})
    ab.require_fields(anchor, dict(teacher_choice=p['A']['candidate_id'], selected_candidate=p[arm]['candidate_id'],
                                   final_actions=p[arm]['actions']), work)
    if arm == 'C':
        c, = (c for c in anchor['candidates'] if c['candidate_id'] == p['C']['candidate_id'])
        ab.require_fields(c, dict(candidate_family='joint_plan', legal=True, source_plan_id=plan['rule_id'],
                                  proposed_actions=p['C']['actions'], origin='one_shot_fork_C',
                                  rule_hash=plan['rule_hash']), work)
    ep, = ab.read_json(work / 'runner/episodes.json')
    reward, outcome = ep['reward'], ep['outcome']
    if type(reward) not in (int, float) or not math.isfinite(reward) or outcome not in ('win', 'loss', 'draw', 'timeout'):
        raise ValueError('invalid terminal return or outcome')
    return dict(run_id=run_id, receipt_hash=digest(receipt), candidate_id=receipt['candidate_id'],
                source_plan_id=receipt['source_plan_id'], actions=receipt['final_actions'], outcome=outcome,
                scenario_sha256=receipt['scenario_sha256'], initial_state_sha256=receipt['initial_state_sha256'],
                **{'return': reward})


def play(work, arm, p, plan):
    ab.run_worker(work, p, work / 'fork_spec.json')
    arm_facts(work, arm, p, plan)


def compare(run):
    run = Path(run)
    plan, design = (ab.read_json(run / name) for name in ('plan.json', 'design.json'))
    indices = indices_for(plan, design['anchor_index'])
    card_hash = validate_frozen_plan(plan)
    ab.require_fields(design, dict(abc_protocol=ABC, plan_hash=digest(plan), rule_card_hash=card_hash,
                                   pair_indices=indices, arms=list(ARMS), **ab.PROTOCOL), 'design')
    accepted, excluded = [], []
    for i in indices:
        p = plan['pairs'][i-1]
        facts, problems = {}, {}
        for arm in ARMS:
            try:
                if design['dry_run'] is not False:
                    raise ValueError('dry-run: no live evidence')
                facts[arm] = arm_facts(run / f'pair-{i:03d}' / arm, arm, p, plan)
            except INPUT_ERRORS as exc:
                problems[arm] = str(exc)
        if not problems:
            if len({f['run_id'] for f in facts.values()}) != 3:
                problems['triplet'] = 'reused replay run id'
            if len({(f['scenario_sha256'], f['initial_state_sha256']) for f in facts.values()}) != 1:
                problems['triplet'] = 'initial state or scenario differs'
        identity = dict(anchor_index=i, **{k: p[k] for k in ('episode_group', 'seed', 'roster', 'snapshot_id', 'game_loop')})
        if problems:
            excluded.append(dict(**identity, missing_or_invalid=problems))
            continue
        deltas = {f'{x}-{y}': dict(return_delta=facts[x]['return']-facts[y]['return'],
                                    win_delta=int(facts[x]['outcome']=='win')-int(facts[y]['outcome']=='win'))
                  for x, y in CONTRASTS}
        accepted.append(dict(**identity, arms=facts, deltas=deltas))
    totals = {}
    for x, y in CONTRASTS:
        key = f'{x}-{y}'
        ds = [r['deltas'][key] for r in accepted]
        totals[key] = dict(mean_return_delta=sum(d['return_delta'] for d in ds)/len(ds) if ds else None,
                           more_wins=sum(d['win_delta'] > 0 for d in ds), fewer_wins=sum(d['win_delta'] < 0 for d in ds))
    return dict(abc_protocol=ABC, plan_hash=design['plan_hash'], rule_card_hash=card_hash,
                rule_id=plan['rule_id'], rule_hash=plan['rule_hash'],
                source_hash=plan['source_hash'], collection_context=plan['collection_context'],
                planned_triplets=len(indices), accepted_triplets=len(accepted), missing_triplets=len(excluded),
                contrasts=totals, triplets=accepted, excluded=excluded)


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    mode = cli.add_mutually_exclusive_group(required=True)
    mode.add_argument('--compare', type=Path, metavar='RUN')
    mode.add_argument('--plan', type=Path)
    cli.add_argument('--out', type=Path, help='new directory, required with --plan')
    cli.add_argument('--workers', type=int, help='maximum simultaneous arms (default: 48)')
    cli.add_argument('--dry-run', action='store_true')
    cli.add_argument('--anchor-index', type=int, help='only launch this original 1-based triplet')
    args = cli.parse_args(argv)
    if args.compare is not None:
        if args.out is not None or args.workers is not None or args.anchor_index is not None or args.dry_run:
            cli.error('--compare is independent of launch options')
        try:
            result = compare(args.compare)
        except PREFLIGHT_ERRORS as exc:
            cli.error(f'cannot compare run: {exc}')
        print(json.dumps(result, indent=2, allow_nan=False))
        if result['excluded']:
            raise SystemExit(1)
        return

    workers = 48 if args.workers is None else args.workers
    if args.out is None or args.out.exists() or workers < 1:
        cli.error('require a new --out and positive --workers')
    out = args.out.expanduser().resolve()
    try:
        raw = args.plan.read_bytes()
        plan = json.loads(raw)
        indices = indices_for(plan, args.anchor_index)
        card_hash = validate_frozen_plan(plan)
        config, versions = ab.historical_config()
        ab.validate_plan(plan, config, versions)
    except PREFLIGHT_ERRORS as exc:
        cli.error(f'preflight failed: {exc}')
    out.mkdir(parents=True, exist_ok=False)
    with (out / 'plan.json').open('xb') as stream:
        stream.write(raw)
    ab.write_json(out / 'design.json', dict(abc_protocol=ABC, rule_card_hash=card_hash, arms=list(ARMS), pair_indices=indices,
        anchor_index=args.anchor_index, dry_run=args.dry_run, workers=workers, plan_hash=digest(plan),
        python=ab.RSI_PYTHON, versions=versions, sc2_path=ab.SC2_PATH, damage_map_dir=str(ab.MAP_DIR), **ab.PROTOCOL))
    jobs = []
    for i in indices:
        p = plan['pairs'][i-1]
        for arm in ARMS:
            work = out / f'pair-{i:03d}' / arm
            work.mkdir(parents=True)
            ab.write_json(work / 'fork_spec.json', spec_for(arm, p, plan))
            jobs.append((work, arm, p, plan))
    print(f'{len(indices)} triplets, {len(jobs)} independent arms: {out}', flush=True)
    if args.dry_run:
        print('Dry-run: no games started.')
        return
    failures = 0
    with ThreadPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
        pending = {pool.submit(play, *job): job[0] for job in jobs}
        for future in as_completed(pending):
            try:
                future.result()
            except Exception as exc:
                failures += 1
                print(f'FAILED {pending[future]}: {exc}', file=sys.stderr, flush=True)
    print(f'Compare separately with --compare {out}')
    if failures:
        raise SystemExit(f'{failures}/{len(jobs)} arms failed')


if __name__ == '__main__':
    main()
