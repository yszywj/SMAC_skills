"""Fit a shallow joint gain tree offline. Keep the pilot first and append independent fork exports."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from decision_tree.scripts.linux.zerg_gain_exports import cumulative, load, require, verify_export
from decision_tree.zerg.dataset import candidate_rows, episode_group, split_records
from decision_tree.zerg.objectives import gain_rows
from decision_tree.zerg.schema import digest
from decision_tree.zerg.scorer import ScorerModel
from decision_tree.zerg.training import ZergTrainingBackend

SEED = 42


def partition(exports, records):
    # Freeze the native backend's outcome-blind rank boundary at the first trainable
    # prefix of exports. Appending a round cannot promote a former holdout to training.
    pilot = []
    for batch in exports:
        pilot.extend(batch.records)
        if len(pilot) >= 6:  # ceil(.25*n) held, minimum 4 training groups
            break
    _, first_held = split_records(pilot, validation_fraction=.25, split_seed=SEED)

    def rank(r):
        group = episode_group(r)
        return hashlib.sha256(f'{SEED}\0{group}'.encode()).hexdigest(), group

    cutoff = max(map(rank, first_held))
    wanted = {episode_group(r) for r in records if rank(r) <= cutoff}
    fraction = .25
    if math.ceil(len(records) * fraction) != len(wanted):
        # Backend takes ceil(n*f), not an explicit group list for gain. Use a
        # midpoint rather than an exact integer boundary subject to float rounding.
        fraction = (len(wanted) - .5) / len(records)
    train, held = split_records(records, validation_fraction=fraction, split_seed=SEED)
    require(wanted == {episode_group(r) for r in held}, 'backend split differs from frozen cutoff')
    return dict(group_by='episode', split_seed=SEED, rank_cutoff=cutoff, calibration_pairs=len(pilot),
                backend_validation_fraction=fraction, train_groups=sorted(map(episode_group, train)),
                validation_groups=sorted(wanted))


def review_model(path, env, split):
    model, report = load(path / 'model.json'), load(path / 'report.json')
    cfg = model['training_configuration']
    expected = {k: env[k] for k in ('anchor_manifest_sha256', 'records_hash', 'dataset_hash',
                                     'pairs_trainable', 'pairs_rejected')}
    require(report['model_sha256'] == digest(model) and report['model_id'] == model['model_id'] and
            model['target'] == cfg['task'] == 'gain' and model['scorer'] == 'joint_scorer' and
            model['gain_audit'] == expected and model['dataset_hash'] == env['dataset_hash'] and
            model['source_fingerprint'] == env['source_fingerprint'] and
            model['execution_context_hash'] == env['execution_context_hash'] and
            cfg['split_seed'] == SEED and cfg['validation_fraction'] == split['backend_validation_fraction'] and
            model['validation_report'] == report['validation'], 'existing model/report do not match audited fit')
    return model, report


def fit(data, out, review_existing=None):
    if out.exists():
        raise FileExistsError(f'--out must be a new directory: {out}')
    exports = [verify_export(d) for d in data]
    records, env = cumulative(exports)
    split = partition(exports, records)
    if review_existing:
        model, report = review_model(review_existing, env, split)
    else:
        model, report = ZergTrainingBackend().train('gain', records, scorer='joint_scorer',
            execution_context_hash=env['execution_context_hash'], source_fingerprint=env['source_fingerprint'],
            dataset_hash=env['dataset_hash'], audit_envelope=env, split_seed=SEED,
            validation_fraction=split['backend_validation_fraction'],
            capacity={'max_depth': 1, 'number_of_trees': 1, 'minimum_leaf_support': 4, 'minimum_leaf_groups': 4})
    for part, groups in [('training', split['train_groups']), ('validation', split['validation_groups'])]:
        observed = report[part]['per_scenario']
        require(sorted(s['group_id'] for s in observed) == groups and all(s['n_rows'] == 1 for s in observed),
                f'{part}: fitted groups differ from audited split')

    scorer = ScorerModel(model)
    rows, problems = gain_rows(candidate_rows(records), strict=True)
    by_key = {row['decision_key']: row for row in rows}
    require(not problems and len(rows) == len(by_key) == len(records), 'expected one measured B per episode')
    held = set(split['validation_groups'])
    facts, decisions = [], []
    for batch in exports:
        for fact in batch.facts:
            row = by_key[fact['decision_key']]
            require(row['gain'] == fact['gain'] and row['candidate_id'] == fact['alternate']['candidate_id'],
                    f'{fact["decision_key"]}: merged target mismatch')
            estimate, = scorer.estimate(row['features'])
            switch = estimate['supported'] and estimate['value'] > scorer.min_gain + scorer.tie_tolerance
            is_held = row['group_id'] in held
            pred = dict(prediction=estimate['value'], switch_to_B=switch, policy_gain=row['gain'] if switch else 0.0)
            facts.append(dict(fact, partition='validation' if is_held else 'train', supported=estimate['supported'], **pred))
            if is_held:
                decisions.append(dict(group=row['group_id'], candidate=row['candidate_id'], measured_gain=row['gain'], **pred))

    n = len(decisions)
    report['holdout_policy'] = dict(train_pairs=len(records)-n, heldout_pairs=n, switched=sum(r['switch_to_B'] for r in decisions),
        mean_gain_vs_always_A=sum(r['policy_gain'] for r in decisions)/n,
        model_mse=sum((r['prediction']-r['measured_gain'])**2 for r in decisions)/n,
        constant_zero_mse=sum(r['measured_gain']**2 for r in decisions)/n, decisions=decisions)
    report['gain_provenance'] = dict(inputs=[dict(path=str(b.path), envelope=b.envelope,
        pairs_sha256=digest(b.pairs), receipts_sha256=digest(b.receipts)) for b in exports],
        training_envelope=env, derived_namespace=len(exports) > 1,
        review_existing=str(review_existing.resolve()) if review_existing else None)
    table = dict(kind='zerg_gain_pair_facts_v1', model_id=model['model_id'], dataset_hash=env['dataset_hash'],
                 split=split, pairs=facts)
    # Only this projection should be used for hypothesis generation/tuning. The
    # full retrospective table (and the model validation report) expose outcomes.
    train_table = dict(kind='zerg_gain_training_pair_facts_v1', model_id=model['model_id'],
                       pairs=[f for f in facts if f['partition'] == 'train'])
    report.update(pair_facts_sha256=digest(table), train_pair_facts_sha256=digest(train_table), split=split)
    out.mkdir(parents=True, exist_ok=False)
    for name, body in [('model', model), ('report', report), ('pair_facts', table), ('train_pair_facts', train_table)]:
        (out / (name + '.json')).write_text(json.dumps(body, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return {k: v for k, v in report['holdout_policy'].items() if k != 'decisions'}


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--data', nargs='+', type=Path, required=True,
                     help='v1 export directories, in append order; retain original fork run for receipt verification')
    cli.add_argument('--out', type=Path, required=True, help='new model/report/pair_facts directory')
    cli.add_argument('--review-existing', type=Path, help='verify and score this exact audited fit without retraining')
    args = cli.parse_args(argv)
    try:
        metrics = fit(args.data, args.out, args.review_existing)
    except (ValueError, KeyError, OSError, TypeError) as exc:
        cli.error(str(exc))
    print(json.dumps(metrics, indent=2))
    print(f'Pair facts: {args.out / "pair_facts.json"}')


if __name__ == '__main__':
    main()
