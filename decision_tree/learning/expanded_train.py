"""Fit a shared scalar return tree on sparse state/candidate comparisons.

Validation scores rank only candidates actually tested at each anchor. They do
not estimate a policy allowed to choose untested calls or change multiple calls.
"""
from copy import deepcopy
import argparse
import hashlib
import json
import math
from pathlib import Path

from decision_tree.descriptors import MODEL_FEATURE_NAMES, candidate_features
from decision_tree.experiment import digest
from decision_tree.learning.expanded_dataset import EVIDENCE_KIND
from decision_tree.learning.train import finite, group_split, group_weights, positive_int


def validate_rows(rows):
    if not isinstance(rows, list) or not rows:
        raise ValueError('No accepted isolated-call comparisons')
    seen = set()
    for row in rows:
        if row.get('evidence_kind') != EVIDENCE_KIND:
            raise ValueError('single-call audited evidence required; free exploration is not a causal label')
        if not isinstance(row.get('group_id'), str) or not row['group_id'] or type(row.get('sc2_seed')) is not int:
            raise ValueError('scenario identity and seed required')
        positive_int(row.get('anchor_call_number'), 'anchor_call_number')
        if not isinstance(row.get('candidate_id'), str) or not row['candidate_id']:
            raise ValueError('candidate identity required')
        key = (row['group_id'], row['sc2_seed'], row['anchor_call_number'], row['candidate_id'])
        if key in seen:
            raise ValueError('duplicate scenario/seed/anchor/candidate comparison')
        seen.add(key)
        if set(row.get('features', {})) != set(MODEL_FEATURE_NAMES):
            raise ValueError('feature allowlist mismatch')
        for name in MODEL_FEATURE_NAMES:
            finite(row['features'][name], name)
        expected = candidate_features(row['state_features'], row['candidate_call'], row['base_call'])
        if row['features'] != expected:
            raise ValueError('candidate feature derivation mismatch')
        base = finite(row.get('base_return'), 'base_return')
        alternative = finite(row.get('candidate_return'), 'candidate_return')
        gain = finite(row.get('gain'), 'gain')
        if not math.isclose(alternative - base, gain, rel_tol=1e-10, abs_tol=1e-10):
            raise ValueError('gain must equal actual candidate return minus BASE')
        gate = row.get('gate', {})
        if (gate.get('invocation_applied') is not True or gate.get('configuration_changed') is not True
                or not isinstance(gate.get('prefix_sha256'), str) or len(gate['prefix_sha256']) != 64):
            raise ValueError('audited invocation and prefix evidence required')
        if gate.get('candidate_id') != row['candidate_id'] or gate.get('anchor_call_number') != row['anchor_call_number']:
            raise ValueError('row does not match audited invocation')
    return rows


def load_dataset(path):
    from decision_tree.schema import load_policy
    path = Path(path)
    raw = path.read_bytes()
    data = json.loads(raw.decode('utf-8-sig'))
    if data.get('schema_version') != 2 or data.get('kind') != 'skill_candidate_call_pairs':
        raise ValueError('expanded isolated-call dataset required')
    if data.get('split') not in (None, 'development', 'train'):
        raise ValueError('sealed validation/test data cannot train this model')
    if set(data.get('feature_names', [])) != set(MODEL_FEATURE_NAMES) or len(data['feature_names']) != len(MODEL_FEATURE_NAMES):
        raise ValueError('feature schema mismatch')
    if not data.get('context') or digest(data['context']) != data.get('context_hash'):
        raise ValueError('context fingerprint mismatch')
    if not isinstance(data.get('game_version'), str) or not data['game_version']:
        raise ValueError('actual game version required')
    contract = deepcopy(data['policy_contract'])
    if 'selection' in contract:
        raise ValueError('policy_contract must omit arm selection')
    contract['selection'] = {'mode': 'base', 'max_interventions_per_episode': 1}
    policy = load_policy(contract)
    if not policy.get('exploration', {}).get('enabled'):
        raise ValueError('expanded exploration contract required')
    if data['context'].get('policy_contract') != data['policy_contract']:
        raise ValueError('policy context binding mismatch')
    if data['candidate_ids'] != [row['id'] for row in policy['candidates']]:
        raise ValueError('candidate contract mismatch')
    validate_rows(data['rows'])
    if any(row['candidate_id'] not in data['candidate_ids'] for row in data['rows']):
        raise ValueError('unknown training candidate')
    source = {'path': str(path.resolve()), 'sha256': hashlib.sha256(raw).hexdigest(),
              'accepted_rows': len(data['rows']), 'rejected_pairs': len(data.get('rejected', []))}
    return data, policy, source


def fit_scalar_tree(rows, *, context_hash, max_depth=2, min_samples_leaf=4, min_leaf_groups=4):
    validate_rows(rows)
    if type(max_depth) is not int or not 0 <= max_depth <= 3:
        raise ValueError('max_depth must be 0..3')
    positive_int(min_samples_leaf, 'min_samples_leaf')
    positive_int(min_leaf_groups, 'min_leaf_groups')
    weights = group_weights(rows)

    def supported(indices):
        return len(indices) >= min_samples_leaf and len({rows[i]['group_id'] for i in indices}) >= min_leaf_groups

    def mean(indices):
        return math.fsum(weights[i] * rows[i]['gain'] for i in indices) / math.fsum(weights[i] for i in indices)

    def error(indices):
        average = mean(indices)
        return math.fsum(weights[i] * (rows[i]['gain'] - average) ** 2 for i in indices)

    def build(indices, depth):
        score, best = error(indices), None
        tolerance = 1e-12 * max(1., score)
        if depth < max_depth:
            for feature in MODEL_FEATURE_NAMES:
                values = sorted({float(rows[i]['features'][feature]) for i in indices})
                for low, high in zip(values, values[1:]):
                    threshold = low / 2 + high / 2
                    if threshold >= high:
                        threshold = low
                    left = [i for i in indices if rows[i]['features'][feature] <= threshold]
                    right = [i for i in indices if rows[i]['features'][feature] > threshold]
                    if not supported(left) or not supported(right):
                        continue
                    new_score = error(left) + error(right)
                    if new_score < score - tolerance:
                        score, best = new_score, (feature, threshold, left, right)
        if best is None:
            groups = sorted({rows[i]['group_id'] for i in indices})
            return {'gain': mean(indices), 'n_samples': len(indices), 'n_groups': len(groups),
                    'group_ids': groups, 'weight_sum': math.fsum(weights[i] for i in indices)}
        feature, threshold, left, right = best
        return {'feature': feature, 'threshold': threshold, 'left': build(left, depth + 1), 'right': build(right, depth + 1)}

    indices = list(range(len(rows)))
    if not supported(indices):
        raise ValueError('insufficient independent scenario support for leaf')
    return {'schema_version': 2, 'kind': 'skill_candidate_gain_tree',
            'feature_names': list(MODEL_FEATURE_NAMES), 'context_hash': context_hash, 'root': build(indices, 0),
            'training': {'algorithm': 'weighted_scalar_squared_error_greedy_cart',
                         'weighting': 'each_scenario_total_weight_one', 'max_depth': max_depth,
                         'min_samples_leaf': min_samples_leaf, 'min_leaf_groups': min_leaf_groups,
                         'target': 'candidate_episode_return_minus_base_episode_return', 'evidence_kind': EVIDENCE_KIND}}


def predict(model, features):
    node = model['root']
    while 'feature' in node:
        node = node['left' if features[node['feature']] <= node['threshold'] else 'right']
    return node['gain']


def tree_counts(node):
    if 'gain' in node:
        return {'nodes': 1, 'predicates': 0, 'leaves': 1, 'depth': 0}
    left, right = tree_counts(node['left']), tree_counts(node['right'])
    return {'nodes': 1 + left['nodes'] + right['nodes'], 'predicates': 1 + left['predicates'] + right['predicates'],
            'leaves': left['leaves'] + right['leaves'], 'depth': 1 + max(left['depth'], right['depth'])}


def evaluate_tested_candidates(model, rows, min_gain=0.):
    """Offline choice per anchor among BASE and tested candidates only."""
    candidate_ids = model.get('candidate_ids')
    if (not isinstance(candidate_ids, list) or not candidate_ids
            or any(not isinstance(cid, str) or not cid for cid in candidate_ids)
            or len(set(candidate_ids)) != len(candidate_ids)):
        raise ValueError('model must record unique candidate_ids in policy declaration order')
    candidate_order = {cid: index for index, cid in enumerate(candidate_ids)}
    anchors = {}
    for row in rows:
        if row['candidate_id'] not in candidate_order:
            raise ValueError('unknown candidate outside model declaration order')
        key = (row['group_id'], row['sc2_seed'], row['anchor_call_number'])
        anchors.setdefault(key, []).append(row)
    details = []
    for key, candidates in sorted(anchors.items()):
        best, highest = None, max(0., min_gain)
        predictions = {}
        for row in sorted(candidates, key=lambda row: candidate_order[row['candidate_id']]):
            prediction = predict(model, row['features'])
            predictions[row['candidate_id']] = prediction
            if prediction > highest:
                best, highest = row, prediction
        details.append({'group_id': key[0], 'sc2_seed': key[1], 'anchor_call_number': key[2],
                        'tested_candidates': [r['candidate_id'] for r in candidates], 'predictions': predictions,
                        'actual_gains': {r['candidate_id']: r['gain'] for r in candidates},
                        'selected': 'BASE' if best is None else best['candidate_id'],
                        'selected_empirical_gain': 0. if best is None else best['gain'],
                        'tested_subset_oracle_gain': max(0., *(r['gain'] for r in candidates))})
    weights = group_weights(details)
    row_weights = group_weights(rows)
    total = math.fsum(weights)
    selected = {}
    for item in details:
        selected[item['selected']] = selected.get(item['selected'], 0) + 1
    return {'scope': 'offline_tested_candidate_subset_per_anchor_not_new_policy_games',
            'tie_break': 'policy_candidate_declaration_order_with_BASE_on_nonpositive_gain',
            'n_rows': len(rows), 'n_anchors': len(details), 'n_groups': len({r['group_id'] for r in rows}),
            'weighting': 'each_scenario_total_weight_one_across_its_anchors',
            'selector_mean_gain': math.fsum(w * r['selected_empirical_gain'] for w, r in zip(weights, details)) / total,
            'tested_subset_oracle_mean_gain': math.fsum(w * r['tested_subset_oracle_gain'] for w, r in zip(weights, details)) / total,
            'prediction_mse': math.fsum(w * (r['gain'] - predict(model, r['features'])) ** 2
                                        for w, r in zip(row_weights, rows)) / math.fsum(row_weights),
            'selection_counts': selected,
            'target_sign_counts_tolerance_1e-9': {'positive': sum(r['gain'] > 1e-9 for r in rows),
                                               'negative': sum(r['gain'] < -1e-9 for r in rows),
                                               'near_zero': sum(abs(r['gain']) <= 1e-9 for r in rows)},
            'action_effect_count': sum(r['gate'].get('action_changed_during_comparison_window', False) for r in rows),
            'rows': details}


def train_expanded(dataset_path, output_dir=None, *, max_depth=2, min_samples_leaf=4,
                   min_leaf_groups=4, validation_fraction=.25, split_seed=0):
    from decision_tree.schema import load_policy, validate_model
    data, policy, source = load_dataset(dataset_path)
    train, validation = group_split(data['rows'], validation_fraction=validation_fraction,
                                    split_seed=split_seed, min_train_groups=min_leaf_groups)
    model = fit_scalar_tree(train, context_hash=data['context_hash'], max_depth=max_depth,
                            min_samples_leaf=min_samples_leaf, min_leaf_groups=min_leaf_groups)
    split = {'method': 'sha256_original_scenario_group_split', 'seed': split_seed,
             'validation_fraction': validation_fraction, 'train_groups': sorted({r['group_id'] for r in train}),
             'validation_groups': sorted({r['group_id'] for r in validation})}
    model.update(context=deepcopy(data['context']), game_version=data['game_version'], candidate_ids=data['candidate_ids'])
    model['training'].update(split=split, inputs=[source], rows_sha256=digest(train),
                             train_groups=split['train_groups'], validation_groups=split['validation_groups'])
    validate_model(model)
    policy['selection'] = {'mode': 'model', 'max_interventions_per_episode': 1, 'min_gain': 0., 'model': model}
    policy = load_policy(policy)
    report = {'schema_version': 2, 'status': 'trained_not_game_validated', 'evidence_kind': EVIDENCE_KIND,
              'context_hash': data['context_hash'], 'game_version': data['game_version'],
              'dataset_inputs': [source], 'split': split, 'tree': tree_counts(model['root']),
              'model_sha256': digest(model), 'policy_sha256': digest(policy),
              'training': evaluate_tested_candidates(model, train),
              'validation': evaluate_tested_candidates(model, validation),
              'limitations': ['An observed prefix is matched, not an internal SC2 snapshot.',
                              'The same BASE rollout is reused: rows are correlated within original scenario.',
                              'Offline choice is limited to tested candidates at each anchor; untested candidate choices are unevaluated.',
                              'Anchor returns are separate single-intervention counterfactuals and must never be summed as a combined policy return.',
                              'Multiple calls per episode and free exploration require separate sequence or isolated replay evidence.',
                              'These development scenarios are reused; final improvement requires frozen-policy disjoint held-out games.',
                              'Executed action-equivalent configurations retain measured return difference, including run noise.']}
    if output_dir is not None:
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=False)
        for name, content in (('policy.json', policy), ('model.json', model), ('report.json', report)):
            (directory / name).write_text(json.dumps(content, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    return policy, model, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--max-depth', type=int, choices=(0, 1, 2, 3), default=2)
    parser.add_argument('--min-samples-leaf', type=int, default=4)
    parser.add_argument('--min-leaf-groups', type=int, default=4)
    parser.add_argument('--validation-fraction', type=float, default=.25)
    parser.add_argument('--split-seed', type=int, default=0)
    args = parser.parse_args()
    _, _, report = train_expanded(args.dataset, args.output_dir, max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf, min_leaf_groups=args.min_leaf_groups,
        validation_fraction=args.validation_fraction, split_seed=args.split_seed)
    print(json.dumps({'status': report['status'], 'tree': report['tree'],
                      'training_rows': report['training']['n_rows'], 'validation_rows': report['validation']['n_rows']}))


if __name__ == '__main__':
    main()
