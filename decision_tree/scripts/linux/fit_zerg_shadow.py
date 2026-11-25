"""Fit an offline joint-plan imitation baseline on completed Shadow batches."""
import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from decision_tree.zerg.dataset import candidate_rows, episode_group, read_records, record_key
from decision_tree.zerg.scorer import ScorerModel
from decision_tree.zerg.training import ZergTrainingBackend
from decision_tree.zerg.schema import digest


def change_count(candidate):
    # The historical snapshot records previous action IDs as strings and
    # proposed action IDs as integers. An absent previous action counts as new.
    previous = candidate['current_actions']
    return sum(actor not in previous or int(previous[actor]) != int(action)
               for actor, action in candidate['proposed_actions'].items())


def load(batches):
    records, episodes, seeds, keys, contexts, scenarios = [], {}, set(), set(), set(), set()
    for batch in batches:
        design = json.loads((batch / 'design.json').read_text())
        if design['dry_run'] or not design['jobs']:
            raise ValueError(f'not a real collection: {batch}')
        for job in design['jobs']:
            work = batch / job['worker']
            summary = json.loads((work / 'runner/summary.json').read_text())
            episode, = json.loads((work / 'runner/episodes.json').read_text())
            meta = summary['zerg_shadow']
            rows = read_records(str(work / 'shadow/shadow.jsonl'))
            seed, roster = job['sc2_seed'], job['roster']
            if (summary['status'] != 'complete' or meta['stats']['model_errors'] or not rows
                    or len(rows) != episode['steps'] or len(rows) != episode['zerg_shadow_decisions']
                    or episode['scenario_sha256'] != job['scenario_sha256']
                    or seed in seeds or job['scenario_sha256'] in scenarios):
                raise ValueError(f'incomplete or repeated episode: {work}')
            group = episode_group(rows[0])
            if group in episodes:
                raise ValueError(f'duplicate episode: {group}')
            episodes[group] = dict(batch=str(batch), worker=job['worker'], roster=roster, seed=seed)
            seeds.add(seed)
            scenarios.add(job['scenario_sha256'])
            context = (meta['source_fingerprint'], meta['execution_context_hash'])
            contexts.add(context)
            for r in rows:
                k = record_key(r)
                if (k in keys or episode_group(r) != group or r['run_id'] != meta['run_id']
                        or r['scene_config_id'] != roster or r['seed'] != seed
                        or (r['source_hash'], r['execution_context_hash']) != context
                        or r['outcome'] != episode['outcome'] or r['return'] != episode['reward']
                        or not any(c['legal'] and c['candidate_id'] == r['teacher_choice'] for c in r['candidates'])):
                    raise ValueError(f'inconsistent decision {k}: {work}')
                keys.add(k)
                # Keep only inputs used by the joint baseline. The summary logs a richer
                # skill/actor description for the selected plan than for its alternatives.
                # Using those asymmetric fields can reveal the label even on held-out games.
                rec = {name: r[name] for name in ('run_id','episode_id','snapshot_id','game_loop',
                                                   'scene_config_id','teacher_choice')}
                rec['global_features'] = {name: r['global_features'][name] for name in ('team','skill')}
                rec['candidates'] = [{**{name: c[name] for name in ('candidate_id','candidate_family',
                    'candidate_skill','legal')}, 'features': {'candidate_action_change_count':
                    change_count(c)}} for c in r['candidates']]
                records.append(rec)
    if len(contexts) != 1:
        raise ValueError('expected exactly one source and execution context')
    return records, episodes, contexts.pop()


def split(episodes, n, split_seed):
    strata = defaultdict(list)
    for group, ep in episodes.items():
        strata[ep['roster']].append(group)
    held = set()
    for roster, groups in strata.items():
        if not 0 < n < len(groups):
            raise ValueError(f'cannot hold out {n} of {len(groups)} games for {roster}')
        ordered = sorted(groups, key=lambda g: sha256(
            f'{split_seed}:{roster}:{episodes[g]["seed"]}'.encode()).digest())
        held.update(ordered[:n])
    return held


def evaluate(model, records):
    scorer = ScorerModel(model)
    totals = defaultdict(lambda: dict(points=0, tied=0, credit=0., random_credit=0.))
    for r in records:
        rows = candidate_rows([r])
        if len(rows) < 2:
            continue
        scores = [scorer.estimate(row['features'])[0]['value'] for row in rows]
        best = max(scores)
        top = [row for row, score in zip(rows, scores) if best - score <= scorer.tie_tolerance]
        credit = sum(row['teacher'] for row in top) / len(top)
        for key in ('overall', r['scene_config_id']):
            v = totals[key]
            v['points'] += 1
            v['tied'] += len(top) > 1
            v['credit'] += credit
            v['random_credit'] += 1 / len(rows)
    return {k: dict(multi_candidate_points=v['points'], tie_rate=v['tied']/v['points'],
                    tie_neutral_top1=v['credit']/v['points'], uniform_random_top1=v['random_credit']/v['points'])
            for k, v in totals.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batches', type=Path, nargs='+', required=True)
    p.add_argument('--out', type=Path)
    p.add_argument('--check-only', action='store_true')
    p.add_argument('--holdout-per-roster', type=int, default=10)
    p.add_argument('--split-seed', type=int, default=42)
    args = p.parse_args()
    if not args.check_only and (args.out is None or args.out.exists()):
        p.error('fitting requires --out pointing to a new directory')
    records, episodes, (source, ctx) = load(args.batches)
    held = split(episodes, args.holdout_per_roster, args.split_seed)
    print(f'{len(records)} decisions, {len(episodes)} games: {len(episodes)-len(held)} train, '
          f'{len(held)} validation, {len({ep["roster"] for ep in episodes.values()})} rosters', flush=True)
    if args.check_only:
        return
    args.out.mkdir(parents=True, exist_ok=False)
    def save(name, value):
        (args.out/name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    save('split.json', {g: {**e, 'partition': 'validation' if g in held else 'train'}
                        for g, e in episodes.items()})
    # Bind this offline transformation separately. The unmodified runtime
    # must not silently load a model that expects canonical action counts.
    fit_context = {'collection_context': ctx, 'projection': 'joint_team_skill_int_action_changes_v1'}
    model, report = ZergTrainingBackend().train('imitation', records, scorer='joint_scorer',
        source_fingerprint=source, execution_context_hash=digest(fit_context), split_seed=args.split_seed,
        validation_episodes=sorted(held), capacity={'max_depth': 2, 'minimum_leaf_groups': 6,
                                                    'minimum_leaf_support': 8})
    report['offline_feature_context'] = fit_context
    report['heldout_multi_candidate'] = evaluate(model, (r for r in records if episode_group(r) in held))
    save('report.json', report)
    save('model.json', model)
    print(args.out, report['heldout_multi_candidate'].get('overall', {}))


if __name__ == '__main__':
    main()
