"""Small family-specific return trees over pre-decision local patch features.

No score is transferred between skill families. Support counts are independent
scenarios, not frames or alternative actions from the same scenario.
"""
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
import re

FAMILIES = ('retarget', 'retreat_veto', 'retreat_step', 'regroup_step')
KINDS = ('local_gain_model', 'local_veto_model')
FORBIDDEN = {'gain', 'reward', 'return', 'returns', 'outcome', 'seed', 'group_id',
             'actor_id', 'target_id', 'candidate_id', 'map_id', 'future_outcome',
             'base_return', 'candidate_return', 'final_score'}
TOLERANCE = 1e-9


def finite(value, name='value'):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f'{name} must be a finite number')
    return float(value)


def _integer(value, name, low=0, high=None):
    if type(value) is not int or value < low or (high is not None and value > high):
        raise ValueError(f'invalid {name}')
    return value


def _names(names):
    if (not isinstance(names, (list, tuple)) or not names or
            any(not isinstance(x, str) or not re.fullmatch('[A-Za-z][A-Za-z0-9_]*', x)
                or x.lower() in FORBIDDEN for x in names) or len(set(names)) != len(names)):
        raise ValueError('invalid pre-decision feature allowlist')
    return list(names)


def _features(features, names):
    if not isinstance(features, dict) or set(features) != set(names):
        raise ValueError('feature allowlist mismatch')
    return {name: finite(features[name], name) for name in names}


def model_hash(model):
    validate_model(model)
    return hashlib.sha256(json.dumps(model, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode('utf-8')).hexdigest()


def validate_model(model):
    if not isinstance(model, dict):
        raise ValueError('model must be an object')
    required = {'schema_version', 'kind', 'feature_names', 'context_hash', 'families',
                'max_depth', 'min_groups', 'training_group_ids', 'weighting'}
    optional = {'old_model_hash', 'old_min_gain', 'game_version', 'validation_group_ids'}
    if not required <= set(model) or set(model) - required - optional:
        raise ValueError('invalid model fields')
    if type(model['schema_version']) is not int or model['schema_version'] != 1 or model['kind'] not in KINDS:
        raise ValueError('unsupported local model schema')
    names = _names(model['feature_names'])
    if not isinstance(model['context_hash'], str) or not model['context_hash']:
        raise ValueError('execution context hash required')
    if 'game_version' in model and (not isinstance(model['game_version'], str) or not model['game_version']):
        raise ValueError('actual game version must be a nonempty string')
    depth = _integer(model['max_depth'], 'max_depth', 0, 8)
    _integer(model['min_groups'], 'min_groups', 1)
    groups = model['training_group_ids']
    if (not isinstance(groups, list) or not groups or
            any(not isinstance(x, str) or not x for x in groups) or len(set(groups)) != len(groups)):
        raise ValueError('unique training scene groups required')
    validation = model.get('validation_group_ids', [])
    if (not isinstance(validation, list) or any(not isinstance(x, str) or not x for x in validation)
            or len(set(validation)) != len(validation) or set(validation) & set(groups)):
        raise ValueError('validation scene groups must be unique and separate from training')
    if model['weighting'] not in ('equal_scenario_mass_within_each_family',
                                  'equal_scenario_mass_within_each_leaf'):
        raise ValueError('unsupported weighting')
    if model['kind'] == 'local_veto_model':
        if not re.fullmatch('[0-9a-f]{64}', str(model.get('old_model_hash', ''))):
            raise ValueError('veto model requires frozen gain model hash')
        if finite(model.get('old_min_gain'), 'old_min_gain') < 0:
            raise ValueError('old_min_gain cannot be negative')
    elif 'old_model_hash' in model or 'old_min_gain' in model:
        raise ValueError('gain model cannot bind another model')
    families = model['families']
    if not isinstance(families, dict) or not families or set(families) - set(FAMILIES):
        raise ValueError('invalid model skill families')

    def visit(node, level):
        if not isinstance(node, dict):
            raise ValueError('tree node must be an object')
        if set(node) == {'value', 'groups', 'samples'}:
            finite(node['value'], 'leaf value')
            _integer(node['groups'], 'leaf groups', 1, len(groups))
            _integer(node['samples'], 'leaf samples', node['groups'])
            return
        if set(node) != {'feature', 'threshold', 'left', 'right'} or level >= depth:
            raise ValueError('invalid tree node or tree depth')
        if node['feature'] not in names:
            raise ValueError('tree uses a forbidden feature')
        finite(node['threshold'], 'threshold')
        visit(node['left'], level + 1)
        visit(node['right'], level + 1)
    for tree in families.values():
        visit(tree, 0)
    return model


def predict(model, family, features, min_groups=None):
    validate_model(model)
    values = _features(features, model['feature_names'])
    if family not in FAMILIES:
        raise ValueError('unknown skill family')
    required = model['min_groups']
    if min_groups is not None:
        required = max(required, _integer(min_groups, 'min_groups', 1))
    node = model['families'].get(family)
    if node is None:
        return {'value': None, 'supported': False, 'groups': 0}
    while 'feature' in node:
        node = node['left'] if values[node['feature']] <= node['threshold'] else node['right']
    return {'value': node['value'], 'supported': node['groups'] >= required,
            'groups': node['groups']}


def select_candidate(model, candidates, min_gain=0):
    """One local patch or BASE; positive ties use a stable canonical ID.

    Shallow trees intentionally share leaves across bindings. Treating every
    such positive tie as BASE would suppress most supported alternatives.
    IDs only settle ties; they are never input features or additional evidence.
    """
    validate_model(model)
    finite(min_gain, 'min_gain')
    if min_gain < 0:
        raise ValueError('min_gain cannot be negative')
    predictions = {}
    for candidate in candidates:
        cid = candidate.get('id')
        if not isinstance(cid, str) or not cid or cid == 'BASE' or cid in predictions:
            raise ValueError('unique non-BASE candidate ids required')
        predictions[cid] = predict(model, candidate['family'], candidate['features'])
    eligible = [(cid, p['value']) for cid, p in predictions.items()
                if p['supported'] and p['value'] > min_gain + TOLERANCE]
    result = {'candidate_id': 'BASE', 'reason': 'no_supported_positive_gain', 'predictions': predictions}
    if not eligible:
        return result
    best = max(value for _, value in eligible)
    winners = [cid for cid, value in eligible if best - value <= TOLERANCE]
    return {**result, 'candidate_id': min(winners),
            'reason': 'canonical_positive_tie' if len(winners) > 1 else 'supported_positive_gain'}


def fit_gain(rows, feature_names, context_hash, max_depth=2, min_groups=4):
    names = _names(feature_names)
    _integer(max_depth, 'max_depth', 0, 8)
    _integer(min_groups, 'min_groups', 1)
    if not isinstance(rows, list) or not rows:
        raise ValueError('nonempty training rows required')
    records = []
    seen_ids = set()
    for row in rows:
        group, family = row.get('group_id'), row.get('candidate_family')
        if not isinstance(group, str) or not group or family not in FAMILIES:
            raise ValueError('each row requires a scene group and known skill family')
        record_id = row.get('record_id')
        if record_id is not None:
            if not isinstance(record_id, str) or not record_id or record_id in seen_ids:
                raise ValueError('duplicate or invalid training record id')
            seen_ids.add(record_id)
        records.append({'group_id': group, 'family': family,
                        'features': _features(row['features'], names),
                        'gain': finite(row['gain'], 'gain')})
    trees = {}
    for family in FAMILIES:
        part = [r for r in records if r['family'] == family]
        if not part:
            continue
        counts = Counter(r['group_id'] for r in part)
        for r in part:
            r['weight'] = 1.0 / counts[r['group_id']]

        def stats(subset):
            mass = sum(r['weight'] for r in subset)
            mean = sum(r['weight'] * r['gain'] for r in subset) / mass
            error = sum(r['weight'] * (r['gain'] - mean) ** 2 for r in subset)
            return mean, error

        def build(subset, depth):
            mean, error = stats(subset)
            leaf = {'value': mean, 'groups': len({r['group_id'] for r in subset}), 'samples': len(subset)}
            if depth >= max_depth or error <= TOLERANCE:
                return leaf
            best, best_error = None, error
            for name in names:
                values = sorted({r['features'][name] for r in subset})
                for a, b in zip(values, values[1:]):
                    threshold = a / 2.0 + b / 2.0
                    # Adjacent floats may round to the upper endpoint.
                    if threshold >= b:
                        threshold = a
                    left = [r for r in subset if r['features'][name] <= threshold]
                    right = [r for r in subset if r['features'][name] > threshold]
                    if any(len({r['group_id'] for r in side}) < min_groups for side in (left, right)):
                        continue
                    loss = stats(left)[1] + stats(right)[1]
                    if loss < best_error - TOLERANCE:
                        best, best_error = (name, threshold, left, right), loss
            if best is None:
                return leaf
            name, threshold, left, right = best
            return {'feature': name, 'threshold': threshold,
                    'left': build(left, depth + 1), 'right': build(right, depth + 1)}
        trees[family] = build(part, 0)
    result = {'schema_version': 1, 'kind': 'local_gain_model', 'feature_names': names,
              'context_hash': context_hash, 'families': trees, 'max_depth': max_depth,
              'min_groups': min_groups,
              'training_group_ids': sorted({r['group_id'] for r in records}),
              'weighting': 'equal_scenario_mass_within_each_family'}
    validate_model(result)
    return result
