"""Validate and populate LLM-authored trees without searching for splits.

The assistant supplies every predicate and topology in structure.json. This
module only routes audited training rows, computes leaf statistics, and scores
the frozen structure on the separately declared scene holdout. It never calls
fit_gain and never chooses a split or threshold.
"""
import argparse
from collections import defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from .models import FAMILIES, finite, model_hash, validate_model
from .runtime import digest, load_policy
from .training import (correction_rows, evaluate, group_split, load_dataset,
                       validate_dataset)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def load_datasets(paths):
    """Re-audit every input; combining batches never relabels an old context."""
    batches = [load_dataset(path) for path in paths]
    require(bool(batches), 'at least one audited dataset is required')
    data = deepcopy(batches[0])
    data['authored_group_ids'] = sorted(set().union(*(_source_groups(batch) for batch in batches)))
    data['authored_reference_policies'] = [deepcopy(batch.get('protocol', {}).get('plan', {}).get('reference_policy'))
                                           for batch in batches]
    for batch in batches[1:]:
        for key in ('context_hash', 'feature_names', 'game_version'):
            require(batch[key] == data[key], 'combined dataset mismatch: ' + key)
        require(batch['policy']['candidates'] == data['policy']['candidates'],
                'combined candidate configuration mismatch')
        data['rows'].extend(deepcopy(batch['rows']))
        data['rejected'].extend(deepcopy(batch.get('rejected', [])))
    data['authored_sources'] = [
        {'path': str(Path(path).resolve()),
         'sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest()}
        for path in paths]
    return validate_dataset(data)


def _source_groups(data):
    groups = {r['group_id'] for r in data['rows']}
    # No-opportunity episodes remain part of the declared cohort. They have no
    # invented training rows or zero labels but must not disappear from splits.
    for scene in data.get('protocol', {}).get('plan', {}).get('scenes', []):
        group = scene.get('group_id')
        require(isinstance(group, str) and bool(group), 'invalid planned scene group')
        groups.add(group)
    for group in data.get('authored_group_ids', []):
        require(isinstance(group, str) and bool(group), 'invalid combined scene group')
        groups.add(group)
    return groups


def validate_split(split, data):
    require(isinstance(split, dict) and set(split) ==
            {'schema_version', 'kind', 'training_group_ids', 'validation_group_ids'},
            'expected explicit local_scene_split fields')
    require(type(split['schema_version']) is int and split['schema_version'] == 1
            and split['kind'] == 'local_scene_split', 'unsupported scene split')
    for key in ('training_group_ids', 'validation_group_ids'):
        values = split[key]
        require(isinstance(values, list) and bool(values)
                and all(isinstance(v, str) and v for v in values)
                and len(values) == len(set(values)), 'nonempty unique scene IDs required: ' + key)
    training, validation = set(split['training_group_ids']), set(split['validation_group_ids'])
    require(not training & validation, 'training and validation scenes overlap')
    require(training | validation == _source_groups(data),
            'fixed split must partition every source scene exactly')
    return training, validation


def make_split(data, validation_fraction=0.25, seed=0):
    validate_dataset(data)
    # Split the full cohort BEFORE eligibility filtering for correction. This
    # cannot resample a more convenient holdout after seeing the old choices.
    training, validation = group_split([{'group_id': g} for g in _source_groups(data)],
                                       validation_fraction, seed)
    return {'schema_version': 1, 'kind': 'local_scene_split',
            'training_group_ids': sorted({r['group_id'] for r in training}),
            'validation_group_ids': sorted({r['group_id'] for r in validation})}


def _eligible(data, kind, old_model, on_policy_only=False):
    require(kind in ('gain', 'veto'), 'kind must be gain or veto')
    if kind == 'veto':
        require(old_model is not None, 'veto requires the frozen old gain model')
        previous = set(old_model['training_group_ids']) | set(old_model.get('validation_group_ids', []))
        require(not previous & _source_groups(data), 'correction cohort must be unseen by the old model')
        if on_policy_only:
            references = data.get('authored_reference_policies',
                                  [data.get('protocol', {}).get('plan', {}).get('reference_policy')])
            require(bool(references), 'on-policy correction needs audited reference policies')
            for reference in references:
                require(isinstance(reference, dict) and isinstance(reference.get('model'), dict),
                        'on-policy correction needs the recorded reference policy')
                require(model_hash(reference['model']) == model_hash(old_model),
                        'on-policy reference uses a different frozen gain model')
                require(reference.get('selection', {}).get('min_gain') == 0,
                        'on-policy correction requires reference min_gain=0')
                require(reference['model']['context_hash'] == data['context_hash'],
                        'on-policy reference execution context mismatch')
            data = deepcopy(data)
            data['rows'] = [row for row in data['rows'] if row.get('reference_first_proposal') is True]
            require(bool(data['rows']), 'no audited first-reference-proposal rows available for on-policy correction')
        return correction_rows(data, old_model)
    require(old_model is None, 'gain does not accept an old model')
    require(not on_policy_only, 'on-policy-only is for veto evidence')
    return deepcopy(data['rows']), []


def _structure(structure, feature_names, max_depth):
    require(isinstance(structure, dict), 'authored structure must be an object')
    require({'schema_version', 'kind', 'families'} <= set(structure)
            and not set(structure) - {'schema_version', 'kind', 'families', 'rationale'},
            'invalid authored structure fields')
    require(type(structure['schema_version']) is int and structure['schema_version'] == 1
            and structure['kind'] == 'local_authored_structure', 'unsupported authored structure')
    if 'rationale' in structure:
        require(isinstance(structure['rationale'], str), 'rationale must be text')
    families = structure['families']
    require(isinstance(families, dict) and bool(families)
            and not set(families) - set(FAMILIES), 'invalid authored families')

    def visit(node, depth):
        require(isinstance(node, dict), 'authored node must be an object')
        if set(node) == {'leaf'}:
            require(node['leaf'] is True, 'leaf placeholder must be true')
            return
        require(set(node) == {'feature', 'threshold', 'left', 'right'},
                'only a predicate or leaf placeholder is allowed; leaf values are recomputed')
        require(depth < max_depth, 'authored tree exceeds maximum depth')
        require(node['feature'] in feature_names, 'authored predicate uses an unlisted feature')
        finite(node['threshold'], 'threshold')
        visit(node['left'], depth + 1)
        visit(node['right'], depth + 1)
    for node in families.values():
        visit(node, 0)
    return families


def training_summary(rows, feature_names):
    """Descriptive training-only statistics; no threshold ranking or search."""
    result = {}
    for family in FAMILIES:
        part = [r for r in rows if r['candidate_family'] == family]
        if not part:
            continue
        groups = defaultdict(list)
        for row in part:
            groups[row['group_id']].append(row['gain'])
        feature_ranges = {}
        for feature in feature_names:
            xs = sorted(r['features'][feature] for r in part)
            feature_ranges[feature] = {'min': xs[0], 'median': xs[len(xs)//2], 'max': xs[-1]}
        result[family] = {'rows': len(part), 'groups': len(groups),
                          'scene_equal_mean_gain': sum(sum(v)/len(v) for v in groups.values())/len(groups),
                          'positive_rows': sum(r['gain'] > 1e-9 for r in part),
                          'negative_rows': sum(r['gain'] < -1e-9 for r in part),
                          'zero_rows': sum(abs(r['gain']) <= 1e-9 for r in part),
                          'feature_ranges': feature_ranges}
    return result


def prepare(data, *, kind='gain', old_model=None, split=None,
            validation_fraction=0.25, seed=0, on_policy_only=False):
    validate_dataset(data)
    split = deepcopy(split) if split is not None else make_split(data, validation_fraction, seed)
    training_ids, validation_ids = validate_split(split, data)
    rows, rejected = _eligible(data, kind, old_model, on_policy_only)
    training = [r for r in rows if r['group_id'] in training_ids]
    validation = [r for r in rows if r['group_id'] in validation_ids]
    require(bool(training), 'no eligible training rows under the fixed split')
    return {'split': split, 'training_rows': training, 'validation_rows': validation,
            'correction_rejected': rejected,
            'training_summary': training_summary(training, data['feature_names'])}


def build(data, structure, split, *, kind='gain', old_model=None,
          max_depth=2, min_groups=4, on_policy_only=False):
    """Populate only the supplied topology; holdout labels cannot affect leaves."""
    require(type(max_depth) is int and 0 <= max_depth <= 8, 'invalid maximum depth')
    require(type(min_groups) is int and min_groups >= 1, 'invalid minimum scene support')
    prepared = prepare(data, kind=kind, old_model=old_model, split=split, on_policy_only=on_policy_only)
    families = _structure(structure, data['feature_names'], max_depth)
    leaf_audit = []

    def populate(node, rows, family, route):
        if 'leaf' in node:
            require(bool(rows), 'authored leaf has no training evidence: ' + family + '/' + route)
            per_scene = defaultdict(list)
            for row in rows:
                per_scene[row['group_id']].append(finite(row['gain'], 'gain'))
            means = {g: sum(v)/len(v) for g, v in per_scene.items()}
            leaf = {'value': sum(means.values())/len(means),
                    'groups': len(means), 'samples': len(rows)}
            leaf_audit.append({'family': family, 'route': route, **leaf,
                               'supported': leaf['groups'] >= min_groups,
                               'training_group_ids': sorted(means),
                               'scene_means': means,
                               'records': [{'group_id': r['group_id'], 'anchor_id': r['anchor_id'],
                                            'candidate_id': r['candidate_id']} for r in rows]})
            return leaf
        name, threshold = node['feature'], float(node['threshold'])
        left = [r for r in rows if r['features'][name] <= threshold]
        right = [r for r in rows if r['features'][name] > threshold]
        return {'feature': name, 'threshold': threshold,
                'left': populate(node['left'], left, family, route + 'L'),
                'right': populate(node['right'], right, family, route + 'R')}

    trees = {family: populate(node, [r for r in prepared['training_rows']
                                     if r['candidate_family'] == family], family, '')
             for family, node in families.items()}
    model = {'schema_version': 1, 'kind': 'local_gain_model' if kind == 'gain' else 'local_veto_model',
             'feature_names': list(data['feature_names']), 'context_hash': data['context_hash'],
             'game_version': data['game_version'], 'families': trees,
             'max_depth': max_depth, 'min_groups': min_groups,
             'training_group_ids': sorted(split['training_group_ids']),
             'validation_group_ids': sorted(split['validation_group_ids']),
             'weighting': 'equal_scenario_mass_within_each_leaf'}
    if kind == 'veto':
        model.update(old_model_hash=model_hash(old_model), old_min_gain=0.0)
    validate_model(model)
    policy = {'schema_version': 1, 'kind': 'local_patch_policy',
              'selection': {'mode': 'model', 'max_interventions': 1, 'min_gain': 0.0},
              'candidates': deepcopy(data.get('candidate_config', data.get('policy', {}).get('candidates', {}))),
              'model': deepcopy(model if kind == 'gain' else old_model)}
    if kind == 'veto':
        policy['veto_model'] = deepcopy(model)
    report = {'schema_version': 1, 'kind': kind, 'construction': 'LLM_supplied_predicates_and_topology',
              'structure_sha256': digest(structure), 'structure': deepcopy(structure),
              'model_hash': model_hash(model), 'context_hash': data['context_hash'],
              'dataset_sources': deepcopy(data.get('authored_sources', [])),
              'on_policy_only': on_policy_only,
              'split': {'train_groups': sorted(split['training_group_ids']),
                        'validation_groups': sorted(split['validation_group_ids'])},
              'leaf_statistics': leaf_audit,
              'omitted_families': sorted({r['candidate_family'] for r in prepared['training_rows']} - set(families)),
              'training': evaluate(model, prepared['training_rows'], old_model if kind == 'veto' else None),
              'validation': evaluate(model, prepared['validation_rows'], old_model if kind == 'veto' else None),
              'correction_rejected': prepared['correction_rejected'],
              'limitations': ['Structure supplied by LLM; no automatic split or threshold search.',
                              'Leaf values use training rows only, with equal scene mass within each leaf.',
                              'Unsupported leaves abstain at runtime; omitted families have no predicted gain.',
                              'Validation diagnostics over tested anchors are not complete-policy rollout results.',
                              'Correction evidence uses old choices on BASE prefixes, not proof of deployed arrival.']}
    return model, policy, report


def _save_directory(directory, artifacts):
    directory = Path(directory)
    require(not directory.exists(), 'output directory already exists; refusing to overwrite evidence')
    directory.mkdir(parents=True, exist_ok=False)
    for name, value in artifacts.items():
        (directory/name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for command in ('prepare', 'gain', 'veto'):
        p = sub.add_parser(command)
        p.add_argument('--dataset', type=Path, action='append', required=True)
        p.add_argument('--output', type=Path, required=True)
        p.add_argument('--old-model', type=Path)
        p.add_argument('--on-policy-only', action='store_true',
                       help='veto only: use audited first-reference-proposal anchors')
        p.add_argument('--split', type=Path, required=command != 'prepare')
        if command == 'prepare':
            p.add_argument('--kind', choices=('gain', 'veto'), default='gain')
            p.add_argument('--validation-fraction', type=float, default=0.25)
            p.add_argument('--seed', type=int, default=0)
        else:
            p.add_argument('--structure', type=Path, required=True)
            p.add_argument('--max-depth', type=int, default=2)
            p.add_argument('--min-groups', type=int, default=4)
    args = parser.parse_args(argv)
    require(not args.output.exists(), 'output directory already exists; refusing to overwrite evidence')
    data = load_datasets(args.dataset)
    old = read(args.old_model) if args.old_model else None
    split = read(args.split) if args.split else None
    if args.command == 'prepare':
        prepared = prepare(data, kind=args.kind, old_model=old, split=split,
                           validation_fraction=args.validation_fraction, seed=args.seed,
                           on_policy_only=args.on_policy_only)
        _save_directory(args.output, {'split.json': prepared['split'],
                                     'training_rows.json': prepared['training_rows'],
                                     'training_summary.json': prepared['training_summary'],
                                     'preparation.json': {'kind': args.kind, 'dataset_sources': data['authored_sources'],
                                                         'on_policy_only': args.on_policy_only,
                                                         'correction_rejected': prepared['correction_rejected'],
                                                         'note': 'No tree constructed. Read training inputs before authoring; holdout remains in source data.'}})
        print(json.dumps({'output': str(args.output), 'training_rows': len(prepared['training_rows']),
                          'validation_rows': len(prepared['validation_rows']), 'model_created': False}))
        return
    structure = read(args.structure)
    model, policy, report = build(data, structure, split, kind=args.command, old_model=old,
                                 max_depth=args.max_depth, min_groups=args.min_groups,
                                 on_policy_only=args.on_policy_only)
    load_policy(policy)
    _save_directory(args.output, {'structure.json': structure, 'split.json': split,
                                 'model.json': model, 'policy.json': policy, 'report.json': report})
    print(json.dumps({'output': str(args.output), 'model_hash': model_hash(model),
                      'validation_mean_gain': report['validation']['mean_gain'],
                      'construction': report['construction']}))


if __name__ == '__main__':
    main()
