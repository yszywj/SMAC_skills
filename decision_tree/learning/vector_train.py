"""State tree with separate, evidence-supported returns for each candidate.

Missing candidate outcomes remain missing. A leaf never transfers one candidate's
return to another. Validation ranks only tested alternatives, not unseen choices
or a multi-intervention rollout policy.
"""
from copy import deepcopy
import argparse
import hashlib
import json
import math
from pathlib import Path

from decision_tree.experiment import digest
from decision_tree.features import FEATURE_NAMES
from decision_tree.descriptors import MODEL_FEATURE_NAMES
from decision_tree.learning.expanded_dataset import EVIDENCE_KIND
from decision_tree.learning.expanded_train import validate_rows as validate_paired_rows
from decision_tree.learning.train import finite, group_split, group_weights, names, positive_int


def allowed_feature_schemas():
    schemas = [list(FEATURE_NAMES)]
    try:
        from decision_tree.choice_features import CHOICE_FEATURE_NAMES
    except ModuleNotFoundError as exc:
        if exc.name != 'decision_tree.choice_features':
            raise
    else:
        schemas.append(list(CHOICE_FEATURE_NAMES))
    return schemas


def state_values(row):
    return row.get('vector_features', row['state_features'])


def candidate_balanced_weights(rows):
    """Each scenario has mass one, shared equally by its observed candidates.

    A candidate's scenario share is divided across its anchors/seeds. Weights
    are computed once for the training set and remain fixed across tree leaves,
    so leaf estimates and the splitting objective use the same weighted SSE.
    """
    counts = {}
    for row in rows:
        by_candidate = counts.setdefault(row['group_id'], {})
        cid = row['candidate_id']
        by_candidate[cid] = by_candidate.get(cid, 0) + 1
    return [1. / (len(counts[row['group_id']]) * counts[row['group_id']][row['candidate_id']])
            for row in rows]


def validate_rows(rows, candidate_ids, feature_names):
    names(candidate_ids, 'candidate_ids')
    if feature_names not in allowed_feature_schemas():
        raise ValueError('vector tree requires an explicit public state feature schema')
    validate_paired_rows(rows)
    anchors = {}
    for row in rows:
        if row['candidate_id'] not in candidate_ids:
            raise ValueError('unknown candidate in training rows')
        values = state_values(row)
        if set(values) != set(feature_names):
            raise ValueError('vector state feature allowlist mismatch')
        for name in feature_names:
            finite(values[name], name)
        if values != {name: row['state_features'][name] for name in feature_names}:
            raise ValueError('vector features must be derived from shared pre-decision state')
        key = (row['group_id'], row['sc2_seed'], row['anchor_call_number'])
        identity = (values, row['base_return'], row['base_call'])
        if key in anchors and anchors[key] != identity:
            raise ValueError('candidates at one anchor must share state and BASE')
        anchors[key] = identity
    return rows


def load_dataset(path):
    from decision_tree.schema import load_policy
    path = Path(path)
    raw = path.read_bytes()
    data = json.loads(raw.decode('utf-8-sig'))
    if (type(data.get('schema_version')) is not int or data['schema_version'] not in (2, 3) or
            data.get('kind') != 'skill_candidate_call_pairs'):
        raise ValueError('audited sparse candidate dataset required')
    if data.get('feature_names') != list(MODEL_FEATURE_NAMES):
        raise ValueError('paired invocation feature schema mismatch')
    if data.get('split') not in (None, 'development', 'train'):
        raise ValueError('sealed validation/test data cannot train this model')
    if not data.get('context') or digest(data['context']) != data.get('context_hash'):
        raise ValueError('context fingerprint mismatch')
    if not isinstance(data.get('game_version'), str) or not data['game_version']:
        raise ValueError('actual game version required')
    contract = deepcopy(data['policy_contract'])
    if 'selection' in contract or data['context'].get('policy_contract') != contract:
        raise ValueError('policy context binding mismatch')
    contract['selection'] = {'mode': 'base', 'max_interventions_per_episode': 1}
    policy = load_policy(contract)
    if not policy.get('exploration', {}).get('enabled'):
        raise ValueError('expanded exploration contract required')
    if data['candidate_ids'] != [row['id'] for row in policy['candidates']]:
        raise ValueError('candidate contract mismatch')
    feature_names = (data.get('vector_feature_names') if data['schema_version'] == 3
                     else list(FEATURE_NAMES))
    if feature_names not in allowed_feature_schemas():
        raise ValueError('vector tree requires an explicit public state feature schema')
    if feature_names != list(FEATURE_NAMES) and policy['exploration'].get('feature_schema') != 3:
        raise ValueError('choice state feature schema must be declared in policy context')
    validate_rows(data['rows'], data['candidate_ids'], feature_names)
    source = {'path': str(path.resolve()), 'sha256': hashlib.sha256(raw).hexdigest(),
              'accepted_rows': len(data['rows']), 'rejected_pairs': len(data.get('rejected', []))}
    return data, policy, source, feature_names


def fit_vector_tree(rows, candidate_ids, *, context_hash, feature_names=None,
                    max_depth=2, min_samples_leaf=4, min_leaf_groups=4,
                    min_candidate_groups=4, tie_tolerance=1e-9):
    feature_names = list(FEATURE_NAMES) if feature_names is None else list(feature_names)
    validate_rows(rows, candidate_ids, feature_names)
    if type(max_depth) is not int or not 0 <= max_depth <= 3:
        raise ValueError('max_depth must be 0..3')
    for name, value in (('min_samples_leaf', min_samples_leaf), ('min_leaf_groups', min_leaf_groups),
                        ('min_candidate_groups', min_candidate_groups)):
        positive_int(value, name)
    if finite(tie_tolerance, 'tie_tolerance') < 0:
        raise ValueError('tie_tolerance must be nonnegative')
    weights = candidate_balanced_weights(rows)

    def supported(indices):
        return len(indices) >= min_samples_leaf and len({rows[i]['group_id'] for i in indices}) >= min_leaf_groups

    def estimates(indices):
        observations = {cid: {} for cid in candidate_ids}
        counts = {cid: 0 for cid in candidate_ids}
        for i in indices:
            row = rows[i]
            observations[row['candidate_id']].setdefault(row['group_id'], []).append(i)
            counts[row['candidate_id']] += 1
        result = {}
        for cid, groups in observations.items():
            observed = [i for indices in groups.values() for i in indices]
            gain = (math.fsum(weights[i]*rows[i]['gain'] for i in observed) /
                    math.fsum(weights[i] for i in observed)) if observed else None
            result[cid] = {'gain': gain,
                           'n_samples': counts[cid], 'n_groups': len(groups), 'group_ids': sorted(groups),
                           'supported': len(groups) >= min_candidate_groups}
        return result

    def error(indices, outputs):
        # Each estimate is the exact minimizer of this candidate's weighted SSE.
        # No other candidate's gain contributes to the estimate.
        return math.fsum(weights[i] * (rows[i]['gain'] - outputs[rows[i]['candidate_id']]['gain'])**2
                         for i in indices)

    def build(indices, depth):
        outputs = estimates(indices)
        current, best = error(indices, outputs), None
        tolerance = 1e-12 * max(1., current)
        if depth < max_depth:
            for feature in feature_names:
                values = sorted({float(state_values(rows[i])[feature]) for i in indices})
                for low, high in zip(values, values[1:]):
                    threshold = low/2 + high/2
                    if threshold >= high:
                        threshold = low
                    left = [i for i in indices if state_values(rows[i])[feature] <= threshold]
                    right = [i for i in indices if state_values(rows[i])[feature] > threshold]
                    if not supported(left) or not supported(right):
                        continue
                    le, re = estimates(left), estimates(right)
                    if not any(e['supported'] for e in le.values()) or not any(e['supported'] for e in re.values()):
                        continue
                    loss = error(left, le) + error(right, re)
                    if loss < current - tolerance:
                        current, best = loss, (feature, threshold, left, right)
        if best is None:
            groups = sorted({rows[i]['group_id'] for i in indices})
            return {'estimates': outputs, 'n_samples': len(indices), 'n_groups': len(groups),
                    'group_ids': groups, 'weight_sum': math.fsum(weights[i] for i in indices)}
        feature, threshold, left, right = best
        return {'feature': feature, 'threshold': threshold,
                'left': build(left, depth+1), 'right': build(right, depth+1)}

    indices = list(range(len(rows)))
    if not supported(indices):
        raise ValueError('insufficient independent scenario support for leaf')
    return {'schema_version': 3, 'kind': 'skill_vector_gain_tree', 'candidate_ids': list(candidate_ids),
            'feature_names': feature_names, 'context_hash': context_hash,
            'min_candidate_groups': min_candidate_groups, 'tie_tolerance': tie_tolerance,
            'root': build(indices, 0),
            'training': {'algorithm': 'masked_candidate_specific_squared_error_greedy_state_tree',
                         'split_weighting': 'each_scenario_one_equal_share_per_observed_candidate_then_rows',
                         'estimate_weighting': 'candidate_specific_fixed_weight_SSE_minimizing_mean',
                         'max_depth': max_depth, 'min_samples_leaf': min_samples_leaf,
                         'min_leaf_groups': min_leaf_groups, 'min_candidate_groups': min_candidate_groups,
                         'target': 'candidate_episode_return_minus_base_episode_return',
                         'missing_labels': 'absent_never_zero', 'evidence_kind': EVIDENCE_KIND}}


def predict_estimates(model, state_features):
    node = model['root']
    while 'feature' in node:
        value = finite(state_features.get(node['feature']), node['feature'])
        node = node['left' if value <= node['threshold'] else 'right']
    return deepcopy(node['estimates'])


def select_vector(model, state_features, available_ids, min_gain=0.):
    if finite(min_gain, 'min_gain') < 0:
        raise ValueError('min_gain must be nonnegative')
    if len(available_ids) != len(set(available_ids)) or set(available_ids)-set(model['candidate_ids']):
        raise ValueError('unknown or duplicate available candidates')
    estimates = predict_estimates(model, state_features)
    supported = {cid: estimates[cid]['gain'] for cid in available_ids
                 if estimates[cid]['supported'] and estimates[cid]['gain'] is not None}
    eligible = {cid: gain for cid, gain in supported.items() if gain > min_gain + model['tie_tolerance']}
    if not eligible:
        return 'BASE', estimates, 'no_positive_supported_gain'
    best = max(eligible.values())
    tied = [cid for cid, gain in eligible.items() if best-gain <= model['tie_tolerance']]
    if len(tied) != 1:
        return 'BASE', estimates, 'ambiguous_positive_gain'
    return tied[0], estimates, 'supported_candidate_gain'


def tree_counts(node):
    if 'estimates' in node:
        return {'nodes': 1, 'predicates': 0, 'leaves': 1, 'depth': 0}
    a, b = tree_counts(node['left']), tree_counts(node['right'])
    return {'nodes': 1+a['nodes']+b['nodes'], 'predicates': 1+a['predicates']+b['predicates'],
            'leaves': a['leaves']+b['leaves'], 'depth': 1+max(a['depth'], b['depth'])}


def evaluate_tested_candidates(model, rows, min_gain=0.):
    validate_rows(rows, model['candidate_ids'], model['feature_names'])
    anchors = {}
    for row in rows:
        key = (row['group_id'], row['sc2_seed'], row['anchor_call_number'])
        anchors.setdefault(key, []).append(row)
    details = []
    for key, tested in sorted(anchors.items()):
        actual = {'BASE': 0., **{r['candidate_id']: r['gain'] for r in tested}}
        selected, estimates, reason = select_vector(model, state_values(tested[0]),
                                                    list(actual)[1:], min_gain)
        predicted = {'BASE': 0., **{cid: estimates[cid]['gain'] for cid in actual if cid != 'BASE'
                                   and estimates[cid]['supported']}}
        comparable = correct = tied = 0
        ids = list(predicted)
        for i, a in enumerate(ids):
            for b in ids[i+1:]:
                truth, guess = actual[a]-actual[b], predicted[a]-predicted[b]
                if abs(truth) <= model['tie_tolerance']:
                    continue
                comparable += 1
                tied += abs(guess) <= model['tie_tolerance']
                correct += abs(guess) > model['tie_tolerance'] and truth*guess > 0
        best = max(actual.values())
        details.append({'group_id': key[0], 'sc2_seed': key[1], 'anchor_call_number': key[2],
                        'tested_candidates': list(actual)[1:], 'estimates': estimates,
                        'actual_gains': actual, 'selected': selected, 'reason': reason,
                        'selected_empirical_gain': actual[selected], 'tested_subset_oracle_gain': best,
                        'regret_to_tested_best': best-actual[selected],
                        'harmful_selection': actual[selected] < -model['tie_tolerance'],
                        'unsupported_candidates': [cid for cid in actual if cid != 'BASE' and not estimates[cid]['supported']],
                        'rank_pairs': comparable, 'rank_correct': correct, 'rank_prediction_ties': tied})
    weights, row_weights = group_weights(details), candidate_balanced_weights(rows)
    total = math.fsum(weights)
    def mean(field):
        return math.fsum(w*r[field] for w,r in zip(weights, details))/total
    rank_groups = {}
    for detail in details:
        if detail['rank_pairs']:
            rank_groups.setdefault(detail['group_id'], []).append(detail['rank_correct']/detail['rank_pairs'])
    rank_group_means = [math.fsum(values)/len(values) for values in rank_groups.values()]
    intervention_weight = math.fsum(w for w,d in zip(weights,details) if d['selected'] != 'BASE')
    selection_counts = {}
    for d in details:
        selection_counts[d['selected']] = selection_counts.get(d['selected'], 0)+1
    supported_rows = [(r, w, predict_estimates(model, state_values(r))[r['candidate_id']])
                      for r, w in zip(rows, row_weights)]
    supported_rows = [(r,w,e) for r,w,e in supported_rows if e['supported']]
    supported_weight = math.fsum(w for _,w,_ in supported_rows)
    return {'scope': 'offline_tested_candidate_subset_per_anchor_not_new_policy_games',
            'tie_break': 'BASE_on_ambiguous_positive_gain_or_nonpositive_supported_gain',
            'n_rows': len(rows), 'n_anchors': len(details), 'n_groups': len({r['group_id'] for r in rows}),
            'weighting': 'each_scenario_total_weight_one', 'selector_mean_gain': mean('selected_empirical_gain'),
            'tested_subset_oracle_mean_gain': mean('tested_subset_oracle_gain'),
            'mean_regret_to_tested_best': mean('regret_to_tested_best'),
            'harmful_selection_rate_per_anchor': mean('harmful_selection'),
            'harmful_selection_rate_when_modified': (math.fsum(w for w,d in zip(weights,details)
                                                               if d['harmful_selection'])/intervention_weight
                                                      if intervention_weight else None),
            'baseline_fallback_rate': math.fsum(w for w,d in zip(weights,details) if d['selected']=='BASE')/total,
            'supported_prediction_coverage': supported_weight/math.fsum(row_weights),
            'unsupported_prediction_fraction': 1-supported_weight/math.fsum(row_weights),
            'supported_prediction_mse': (math.fsum(w*(r['gain']-e['gain'])**2 for r,w,e in supported_rows)/supported_weight
                                         if supported_weight else None),
            'rank_agreement_supported_pairs': (math.fsum(rank_group_means)/len(rank_group_means)
                                                if rank_group_means else None),
            'ranking_weighting': 'equal_scenario_mean_of_comparable_anchor_rates',
            'ranking_scenarios': len(rank_group_means),
            'rank_comparisons': sum(d['rank_pairs'] for d in details),
            'rank_prediction_ties': sum(d['rank_prediction_ties'] for d in details),
            'selection_counts': selection_counts,
            'candidate_observed_groups': {cid: len({r['group_id'] for r in rows if r['candidate_id']==cid})
                                          for cid in model['candidate_ids']},
            'rows': details}


def train_vector(dataset_path, output_dir=None, *, max_depth=2, min_samples_leaf=4,
                 min_leaf_groups=4, min_candidate_groups=4, validation_fraction=.25,
                 split_seed=0, tie_tolerance=1e-9):
    from decision_tree.schema import load_policy, validate_model
    data, policy, source, feature_names = load_dataset(dataset_path)
    train, validation = group_split(data['rows'], validation_fraction=validation_fraction,
                                    split_seed=split_seed, min_train_groups=min_leaf_groups)
    model = fit_vector_tree(train, data['candidate_ids'], context_hash=data['context_hash'],
                            feature_names=feature_names, max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                            min_leaf_groups=min_leaf_groups, min_candidate_groups=min_candidate_groups,
                            tie_tolerance=tie_tolerance)
    split = {'method': 'sha256_original_scenario_group_split', 'seed': split_seed,
             'validation_fraction': validation_fraction, 'train_groups': sorted({r['group_id'] for r in train}),
             'validation_groups': sorted({r['group_id'] for r in validation})}
    model.update(context=deepcopy(data['context']), game_version=data['game_version'])
    model['training'].update(split=split, inputs=[source], rows_sha256=digest(train),
                             train_groups=split['train_groups'], validation_groups=split['validation_groups'])
    validate_model(model)
    policy['selection'] = {'mode': 'model', 'max_interventions_per_episode': 1, 'min_gain': 0., 'model': model}
    policy = load_policy(policy)
    report = {'schema_version': 3, 'status': 'trained_not_game_validated', 'evidence_kind': EVIDENCE_KIND,
              'context_hash': data['context_hash'], 'game_version': data['game_version'],
              'dataset_inputs': [source], 'split': split, 'tree': tree_counts(model['root']),
              'model_sha256': digest(model), 'policy_sha256': digest(policy),
              'training': evaluate_tested_candidates(model, train),
              'validation': evaluate_tested_candidates(model, validation),
              'limitations': ['Each candidate has its own estimate; missing/unsupported estimates are never eligible.',
                              'Support counts are independent original scenarios, not trajectory rows.',
                              'Reported candidate choices are limited to tested subsets at isolated anchors.',
                              'Anchor gains cannot be added into a multi-intervention policy gain.',
                              'Repeated use of development holdouts is diagnostic, not a fresh final test.',
                              'Models collected under old runtime contexts are not automatically deployable after code changes.',
                              'Frozen-policy disjoint real games are required before claiming improvement.']}
    if output_dir is not None:
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=False)
        for name, content in (('policy.json', policy), ('model.json', model), ('report.json', report)):
            (directory/name).write_text(json.dumps(content,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    return policy, model, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--max-depth', type=int, choices=(0,1,2,3), default=2)
    parser.add_argument('--min-samples-leaf', type=int, default=4)
    parser.add_argument('--min-leaf-groups', type=int, default=4)
    parser.add_argument('--min-candidate-groups', type=int, default=4)
    parser.add_argument('--validation-fraction', type=float, default=.25)
    parser.add_argument('--split-seed', type=int, default=0)
    args = parser.parse_args()
    _, _, report = train_vector(args.dataset,args.output_dir,max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,min_leaf_groups=args.min_leaf_groups,
        min_candidate_groups=args.min_candidate_groups,validation_fraction=args.validation_fraction,split_seed=args.split_seed)
    print(json.dumps({'status':report['status'],'tree':report['tree'],
                      'training_rows':report['training']['n_rows'],'validation_rows':report['validation']['n_rows']}))


if __name__ == '__main__':
    main()
