"""Select outcome-blind single-step BASE/alternative pairs; does not run SC2."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from decision_tree.scripts.linux.fit_zerg_shadow import change_count
from decision_tree.zerg.dataset import episode_group
from decision_tree.zerg.schema import build_model_features, digest
from decision_tree.zerg.scorer import ScorerModel


def arm(c, score):
    return dict(candidate_id=c['candidate_id'], source_plan_id=c['source_plan_id'],
                actions=c['proposed_actions'], imitation_score=score)


def build_plan(fit, seed=42, *, verified=False, exclude_plans=(), repair_from=None, replace_pairs=()):
    """Build but do not publish a plan. Historical exclusions imply strict archive verification."""
    fit = Path(fit)
    verified = verified or bool(exclude_plans)
    if bool(repair_from) != bool(replace_pairs):
        raise ValueError('supply both --repair-from and --replace-pairs')
    if verified and repair_from:
        raise ValueError('new round selection cannot be combined with repair')
    split = json.loads((fit / 'split.json').read_text())
    model = json.loads((fit / 'model.json').read_text())
    report = json.loads((fit / 'report.json').read_text())
    ctx = report['offline_feature_context']
    if (ctx['projection'] != 'joint_team_skill_int_action_changes_v1'
            or model['execution_context_hash'] != digest(ctx) or model['model_id'] != report['model_id']
            or model['target'] != 'imitation' or model['scorer'] != 'joint_scorer'):
        raise ValueError('not a compatible joint imitation fit')
    held = {g for g, ep in split.items() if ep['partition'] == 'validation'}
    if set(model['training_configuration']['validation_episodes']) != held:
        raise ValueError('fit and split disagree')
    scorer = ScorerModel(model)
    by_roster = defaultdict(list)
    plan = dict(kind='zerg_fork_review_plan_v1', fit=str(fit), seed=seed,
                model_id=model['model_id'], split_hash=digest(split), source_hash=model['source_fingerprint'],
                collection_context=ctx['collection_context'], horizon_steps=1)
    old = json.loads(Path(repair_from).read_text()) if repair_from else None
    avoid = {r['episode_group'] for r in old['pairs']} if old else set()
    if verified:
        from decision_tree.scripts.linux.zerg_fork_preflight import excluded_groups, verify_archived_b
        if digest(model) != report['model_sha256']:
            raise ValueError('fit model checksum differs')
        avoid, hashes = excluded_groups(exclude_plans, plan, split)
        plan['archive_preflight'] = dict(method='strict_fork_b_commit_v1', quota_per_roster=2,
                                         excluded_plan_hashes=hashes, excluded_groups=len(avoid), rejected=0)
        # Include a roster even when history exhausted *all* of its train groups.
        by_roster.update({ep['roster']: [] for ep in split.values()})
    for group, ep in split.items():
        if ep['partition'] != 'train' or group in avoid:
            continue
        pool = by_roster[ep['roster']]
        worker = ROOT / ep['batch'] / ep['worker']
        with (worker/'shadow/shadow.jsonl').open() as f:
            for line in f:
                r = json.loads(line)
                if (episode_group(r) != group or r['scene_config_id'] != ep['roster']
                        or r['seed'] != ep['seed'] or r['source_hash'] != model['source_fingerprint']
                        or r['execution_context_hash'] != ctx['collection_context']):
                    raise ValueError(f'unexpected provenance: {worker}')
                legal = [c for c in r['candidates'] if c['legal'] and c['candidate_family'] == 'joint_plan']
                base = next((c for c in legal if c['candidate_id'] == r['teacher_choice']), None)
                if not base or base['proposed_actions'] != r['final_actions'] or len(legal) < 2:
                    continue
                def score(c):
                    flat = build_model_features(r['global_features'], {'candidate_action_change_count': change_count(c)})
                    return scorer.estimate(flat)[0]['value']
                a = base['proposed_actions']
                base_score = score(base)
                for c in legal:
                    b = c['proposed_actions']
                    if a.keys() != b.keys() or a == b:
                        continue
                    delta = {u: {'unit': r['global_features']['unit_types'].get(u),
                                 'A': a[u], 'B': b[u]} for u in a if a[u] != b[u]}
                    alt_score = score(c)
                    diff = alt_score-base_score
                    row = dict(episode_group=group, seed=ep['seed'], roster=ep['roster'],
                               manifest=str(worker/'manifest.json'), run_id=r['run_id'], episode_id=r['episode_id'],
                               snapshot_id=r['snapshot_id'], game_loop=r['game_loop'],
                               A=arm(base, base_score), B=arm(c, alt_score), changed=delta)
                    key = digest([seed, ep['seed'], r['snapshot_id'], c['candidate_id']])
                    pool.append((diff, any(d['unit']=='baneling' for d in delta.values()), key, row))
    pairs = []
    for roster, pool in sorted(by_roster.items()):
        rejected = set()
        for tied in (False, True):
            if not pool:
                raise ValueError(f'not enough distinct eligible episodes in {roster}')
            def rank(item):
                diff, bane, key, _ = item
                preferred = abs(diff) <= scorer.tie_tolerance if tied else diff > scorer.tie_tolerance
                return not preferred, not bane, key
            for _, _, key, row in sorted(pool, key=rank):
                if key in rejected:
                    continue
                if verified:
                    try:
                        verify_archived_b(row, plan)
                    except ValueError as exc:
                        # Only replay-contract rejections are eligibility signals;
                        # do not hide corrupt JSON, I/O failures or setup errors.
                        if not str(exc).startswith('Zerg fork: '):
                            raise
                        rejected.add(key)
                        plan['archive_preflight']['rejected'] += 1
                        continue
                break
            else:
                raise ValueError(f'not enough replayable distinct episodes in {roster} ({len(rejected)} rejected candidates)')
            pairs.append(row)
            pool = [item for item in pool if item[3]['episode_group'] != row['episode_group']]
    if not pairs:
        raise ValueError('no eligible pairs')
    plan['pairs'] = pairs
    if repair_from:
        if any(old[k] != plan[k] for k in ('kind','model_id','split_hash','source_hash','collection_context','horizon_steps')):
            raise ValueError('repair source belongs to a different fit')
        idx = set(replace_pairs)
        if not all(1 <= i <= len(pairs) for i in idx) or len(old['pairs']) != len(pairs):
            raise ValueError('invalid replacement indices')
        frozen_groups = {r['episode_group'] for r in old['pairs']}
        for i, original in enumerate(old['pairs'], 1):
            if i not in idx:
                pairs[i-1] = original
            elif pairs[i-1]['roster'] != original['roster'] or pairs[i-1]['episode_group'] in frozen_groups:
                raise ValueError('replacement overlaps original episodes or changes roster; use another seed')
        plan.update(parent_plan_sha256=digest(old), replaced_pairs=sorted(idx))
    return plan


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--fit', required=True, type=Path)
    cli.add_argument('--out', required=True, type=Path, help='new JSON file')
    cli.add_argument('--seed', type=int, default=42)
    cli.add_argument('--repair-from', type=Path, help='keep all but explicitly replaced pairs from this frozen plan')
    cli.add_argument('--replace-pairs', type=int, nargs='+', help='original 1-based pair indices')
    cli.add_argument('--verified-round', '--require-replayable', action='store_true',
                     help='require two distinct, archive-replayed B commits per fit roster or fail')
    cli.add_argument('--exclude-plan', type=Path, action='append', default=[], metavar='PLAN',
                     help='exclude every episode group in this prior plan (repeatable; implies --verified-round)')
    args = cli.parse_args(argv)
    if args.out.exists():
        cli.error('--out already exists; use a new filename')
    try:
        plan = build_plan(args.fit, args.seed, verified=args.verified_round, exclude_plans=args.exclude_plan,
                          repair_from=args.repair_from, replace_pairs=args.replace_pairs)
    except ValueError as exc:
        cli.error(str(exc))
    pairs = plan['pairs']
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as f:
        json.dump(plan, f, indent=2, allow_nan=False)
        f.write('\n')
    print(f'{len({p["roster"] for p in pairs})} rosters, {len(pairs)} decision points, {len(pairs)} alternate arms; '
          f'{2*len(pairs)} planned games, 0 executed: {args.out}')
    first = pairs[0]
    print('First example:', first['roster'], 'seed', first['seed'], 'loop', first['game_loop'], first['changed'])


if __name__ == '__main__':
    main()
