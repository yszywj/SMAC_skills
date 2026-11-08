"""Fit local gain or frozen-model veto trees from audited paired reruns.

Reports are offline selection diagnostics over tested candidates, not new
policy rollouts. Correction evidence must come from new scene groups.
"""
from collections import defaultdict
from copy import deepcopy
import argparse
import hashlib
import json
from pathlib import Path

from .models import (fit_gain, finite, model_hash, predict, select_candidate,
                     validate_model, _features, _names)


def anchor_key(row):
    return row['group_id'], str(row['anchor_id'])


def validate_dataset(data, expected_feature_names=None):
    if (not isinstance(data, dict) or type(data.get('schema_version')) is not int
            or data.get('schema_version') != 1 or data.get('kind') != 'local_patch_dataset'
            or data.get('audited') is not True):
        raise ValueError('audited local_patch_dataset schema 1 required')
    if data.get('split') not in (None, 'development', 'train', 'correction'):
        raise ValueError('sealed test/validation data cannot be used for training')
    names = _names(data.get('feature_names'))
    if expected_feature_names is not None and names != list(expected_feature_names):
        raise ValueError('dataset feature schema differs from local skill runtime')
    if not isinstance(data.get('context_hash'), str) or not data['context_hash']:
        raise ValueError('frozen execution context required')
    if not isinstance(data.get('game_version'), str) or not data['game_version']:
        raise ValueError('actual game version required')
    rows = data.get('rows')
    if not isinstance(rows, list) or not rows:
        raise ValueError('nonempty paired rows required')
    seen, anchors = set(), {}
    for row in rows:
        for name in ('group_id', 'anchor_id', 'candidate_id', 'candidate_family'):
            if not isinstance(row.get(name), str) or not row[name]:
                raise ValueError(f'row requires {name}')
        if row['candidate_id'] == 'BASE':
            raise ValueError('BASE is not a training alternative')
        key = (*anchor_key(row), row['candidate_id'])
        if key in seen:
            raise ValueError('duplicate candidate row at one scene anchor')
        seen.add(key)
        _features(row['features'], names)
        gain = finite(row['gain'], 'gain')
        base, candidate = finite(row['base_return'], 'base_return'), finite(row['candidate_return'], 'candidate_return')
        if abs(gain - (candidate - base)) > 1e-8:
            raise ValueError('gain must equal measured candidate return minus BASE return')
        ids = row.get('available_candidate_ids')
        if (not isinstance(ids, list) or not ids or len(ids) != len(set(ids))
                or any(not isinstance(cid, str) or not cid or cid == 'BASE' for cid in ids)
                or row['candidate_id'] not in ids):
            raise ValueError('complete canonical available candidate list required')
        if type(row.get('anchor_complete')) is not bool:
            raise ValueError('anchor completeness flag required')
        identity = (base, tuple(sorted(ids)), row['anchor_complete'])
        ak = anchor_key(row)
        if ak in anchors and anchors[ak] != identity:
            raise ValueError('candidate rows disagree about their shared BASE/anchor')
        anchors[ak] = identity
    for ak, (_, ids, complete) in anchors.items():
        actual = {r['candidate_id'] for r in rows if anchor_key(r) == ak}
        if complete != (actual == set(ids)):
            raise ValueError('anchor completeness inconsistent with measured rows')
    return data


def load_dataset(path):
    from skills.local_v1 import FEATURE_NAMES
    data = validate_dataset(json.loads(Path(path).read_text(encoding='utf-8-sig')), FEATURE_NAMES)
    # The audit flag is a format marker, not an authority. Re-read original
    # complete runs so edited labels/features or stale logs cannot train.
    from .dataset import build_dataset
    provenance = data.get('provenance', {})
    try:
        base = provenance['BASE']['path']
        entries = [arm['entry'] for arm in provenance['arms']]
    except (KeyError, TypeError) as exc:
        raise ValueError('training requires reproducible original-run provenance') from exc
    rebuilt = build_dataset(base, entries, data.get('protocol', {}).get('plan'))
    if rebuilt['rejected']:
        raise ValueError('original-run re-audit rejected paired evidence')
    for key in ('rows', 'context_hash', 'feature_names', 'game_version', 'policy'):
        if rebuilt.get(key) != data.get(key):
            raise ValueError(f'paired evidence changed during training re-audit: {key}')
    if rebuilt['provenance'] != provenance:
        raise ValueError('source run contents changed since original paired audit')
    return data


def group_split(rows, validation_fraction=0.25, seed=0):
    fraction = finite(validation_fraction, 'validation_fraction')
    if not 0 < fraction < 1:
        raise ValueError('validation_fraction must be between 0 and 1')
    groups = sorted({r['group_id'] for r in rows},
                    key=lambda g: hashlib.sha256(f'{seed}:{g}'.encode()).hexdigest())
    if len(groups) < 2:
        raise ValueError('at least two independent scenes needed for train/validation split')
    n_test = max(1, min(len(groups) - 1, int(len(groups) * fraction + 0.5)))
    heldout = set(groups[:n_test])
    return ([r for r in rows if r['group_id'] not in heldout],
            [r for r in rows if r['group_id'] in heldout])


def _anchor_rows(rows):
    anchors = defaultdict(list)
    for row in rows:
        anchors[anchor_key(row)].append(row)
    return anchors


def _candidates(rows):
    return [{'id': r['candidate_id'], 'family': r['candidate_family'], 'features': r['features']}
            for r in rows]


def correction_rows(data, old_model):
    """Select only the exact frozen gain model's winner, then label its veto.

    All alternatives at the anchor must have measured outcomes. This avoids
    selecting a convenient winner from a partial tested set.
    """
    validate_dataset(data)
    validate_model(old_model)
    if old_model['kind'] != 'local_gain_model':
        raise ValueError('correction requires an unmodified frozen gain model')
    if old_model['context_hash'] != data['context_hash'] or old_model['feature_names'] != data['feature_names']:
        raise ValueError('correction evidence execution context/schema mismatch')
    if old_model.get('game_version') is not None and old_model['game_version'] != data['game_version']:
        raise ValueError('correction evidence game version mismatch')
    previous_groups = set(old_model['training_group_ids']) | set(old_model.get('validation_group_ids', []))
    if previous_groups & {r['group_id'] for r in data['rows']}:
        raise ValueError('correction evidence must use scenes unseen during old model fitting')
    eligible, rejected = [], []
    old_hash = model_hash(old_model)
    for key, rows in _anchor_rows(data['rows']).items():
        if not all(r['anchor_complete'] for r in rows):
            rejected.append({'anchor': list(key), 'reason': 'incomplete_candidate_comparison'})
            continue
        choice = select_candidate(old_model, _candidates(rows))
        if choice['candidate_id'] == 'BASE':
            rejected.append({'anchor': list(key), 'reason': choice['reason']})
            continue
        row = next(r for r in rows if r['candidate_id'] == choice['candidate_id'])
        eligible.append({**deepcopy(row), 'gain': -row['gain'],
                         'original_gain': row['gain'], 'old_model_hash': old_hash,
                         'label_definition': 'BASE_return_minus_frozen_old_choice_return'})
    if not eligible:
        raise ValueError('no complete, non-BASE frozen-model choices available for correction')
    return eligible, rejected


def evaluate(model, rows, old_model=None):
    """Equal-scene offline evaluation, keeping candidate gaps explicit."""
    by_group = defaultdict(list)
    decisions = []
    skipped = 0
    for key, part in _anchor_rows(rows).items():
        if old_model is None:
            if not all(r['anchor_complete'] for r in part):
                skipped += 1
                continue
            choice = select_candidate(model, _candidates(part))
            selected = choice['candidate_id']
            gain = 0.0 if selected == 'BASE' else next(r['gain'] for r in part if r['candidate_id'] == selected)
        else:
            # correction_rows has already recorded exactly one eligible old choice.
            row = part[0]
            p = predict(model, row['candidate_family'], row['features'])
            veto = p['supported'] and p['value'] > 1e-9
            selected = 'BASE' if veto else row['candidate_id']
            gain = row['gain'] if veto else 0.0  # improvement relative to old model
        by_group[key[0]].append(gain)
        decisions.append({'group_id': key[0], 'anchor_id': key[1], 'selected_id': selected, 'gain': gain})
    means = [sum(values) / len(values) for values in by_group.values()]
    return {'scope': 'offline_tested_anchors_not_new_policy_rollouts',
            'reference': 'frozen_old_model' if old_model is not None else 'BASE',
            'groups': len(by_group), 'anchors': len(decisions), 'rows': len(rows),
            'incomplete_anchors_excluded': skipped,
            'mean_gain': sum(means) / len(means) if means else None,
            'negative_selections': sum(row['gain'] < -1e-9 for row in decisions),
            'decisions': decisions}


def train(data, *, kind='gain', old_model=None, max_depth=2, min_groups=4,
          validation_fraction=0.25, seed=0):
    validate_dataset(data)
    if kind not in ('gain', 'veto'):
        raise ValueError('kind must be gain or veto')
    rows, rejected = data['rows'], []
    if kind == 'veto':
        if old_model is None:
            raise ValueError('veto fitting requires --old-model')
        rows, rejected = correction_rows(data, old_model)
    training, validation = group_split(rows, validation_fraction, seed)
    if len({r['group_id'] for r in training}) < min_groups:
        raise ValueError('insufficient independent training scenes for requested support threshold')
    model = fit_gain(training, data['feature_names'], data['context_hash'], max_depth, min_groups)
    model['validation_group_ids'] = sorted({r['group_id'] for r in validation})
    if data.get('game_version'):
        model['game_version'] = data['game_version']
    if kind == 'veto':
        model.update(kind='local_veto_model', old_model_hash=model_hash(old_model), old_min_gain=0.0)
    validate_model(model)
    policy = {'schema_version': 1, 'kind': 'local_patch_policy',
              'selection': {'mode': 'model', 'max_interventions': 1, 'min_gain': 0},
              'candidates': deepcopy(data.get('candidate_config', data.get('policy', {}).get('candidates', {}))),
              'model': deepcopy(model if kind == 'gain' else old_model)}
    if kind == 'veto':
        policy['veto_model'] = deepcopy(model)
    report = {'schema_version': 1, 'kind': kind, 'model_hash': model_hash(model),
              'context_hash': data['context_hash'],
              'split': {'seed': seed, 'validation_fraction': validation_fraction,
                        'train_groups': sorted({r['group_id'] for r in training}),
                        'validation_groups': sorted({r['group_id'] for r in validation})},
              'training': evaluate(model, training, old_model if kind == 'veto' else None),
              'validation': evaluate(model, validation, old_model if kind == 'veto' else None),
              'correction_rejected': rejected,
              'limitations': ['Offline diagnostics only; no new deployment performance measured.',
                              'Support counts are scene groups; small support is not a reliability guarantee.',
                              'Validation scenes are not included in the exported fitted model.',
                              'Veto evidence evaluates a frozen model recommendation at a BASE-prefix anchor; '
                              'it does not prove that a full deployed model rollout reaches that anchor.']}
    return model, policy, report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='kind', required=True)
    for kind in ('gain', 'veto'):
        cmd = sub.add_parser(kind)
        cmd.add_argument('--dataset', required=True, type=Path)
        cmd.add_argument('--output', required=True, type=Path, help='new output directory')
        cmd.add_argument('--max-depth', type=int, default=2)
        cmd.add_argument('--min-groups', type=int, default=4)
        cmd.add_argument('--validation-fraction', type=float, default=0.25)
        cmd.add_argument('--seed', type=int, default=0)
        if kind == 'veto':
            cmd.add_argument('--old-model', required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error('output directory already exists; refusing to overwrite evidence')
    data = load_dataset(args.dataset)
    old = (json.loads(args.old_model.read_text(encoding='utf-8-sig')) if args.kind == 'veto' else None)
    model, policy, report = train(data, kind=args.kind, old_model=old, max_depth=args.max_depth,
                                  min_groups=args.min_groups,
                                  validation_fraction=args.validation_fraction, seed=args.seed)
    from .runtime import load_policy
    load_policy(policy)
    report['dataset_sha256'] = hashlib.sha256(args.dataset.read_bytes()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=False)
    for name, value in (('model.json', model), ('policy.json', policy), ('report.json', report)):
        (args.output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'output': str(args.output), 'model_hash': model_hash(model),
                      'validation_mean_gain': report['validation']['mean_gain']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
