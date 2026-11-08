"""Freeze training evidence for a supplied tree, then verify its exported leaves."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

from decision_tree.local.authored import _structure, require
from decision_tree.local.models import model_hash
from decision_tree.local.runtime import digest, load_policy
from decision_tree.scripts.local_campaign_analysis import file_hash, load_training, read
from skills.local_v1 import FEATURE_NAMES


def proposal(prepared, structure, *, min_groups=4, max_depth=2):
    require(type(min_groups) is int and min_groups > 0, 'min_groups must be positive')
    require(type(max_depth) is int and 0 <= max_depth <= 8, 'invalid maximum depth')
    prepared = Path(prepared).resolve()
    evidence = load_training(prepared)
    families = _structure(structure, list(FEATURE_NAMES), max_depth)
    rows_by_leaf = defaultdict(list)
    for row in evidence['rows']:
        family = row['candidate_family']
        if family not in families:
            continue
        node, route = families[family], ''
        while 'feature' in node:
            left = row['features'][node['feature']] <= node['threshold']
            node = node['left'] if left else node['right']
            route += 'L' if left else 'R'
        rows_by_leaf[family, route].append(row)

    statistics = []

    def populate(family, route, node):
        if 'feature' in node:
            return {'feature': node['feature'], 'threshold': float(node['threshold']),
                    'left': populate(family, route+'L', node['left']),
                    'right': populate(family, route+'R', node['right'])}
        rows = rows_by_leaf[family, route]
        require(bool(rows), f'no training evidence at {family}/{route}')
        by_scene = defaultdict(list)
        for row in rows:
            by_scene[row['group_id']].append(row['gain'])
        means = {group: sum(values)/len(values) for group, values in by_scene.items()}
        leaf = {'value': sum(means.values())/len(means), 'groups': len(means), 'samples': len(rows)}
        statistics.append({**leaf, 'family': family, 'route': route, 'supported': len(means) >= min_groups,
                           'scene_means': means, 'training_group_ids': sorted(means),
                           'records': [{k: r[k] for k in ('group_id', 'anchor_id', 'candidate_id')} for r in rows],
                           'sign_counts': {'positive': sum(r['gain'] > 1e-9 for r in rows),
                                           'zero': sum(abs(r['gain']) <= 1e-9 for r in rows),
                                           'negative': sum(r['gain'] < -1e-9 for r in rows)}})
        return leaf

    trees = {family: populate(family, '', node) for family, node in families.items()}
    kind = evidence['preparation']['kind']
    require(kind in ('gain', 'veto'), 'unknown prepared model kind')
    old_hash = None
    if kind == 'veto':
        old_hashes = {r['old_model_hash'] for r in evidence['rows']}
        require(len(old_hashes) == 1, 'veto evidence must bind one frozen model')
        old_hash = old_hashes.pop()
    content = {'schema_version': 1, 'kind': 'local_tree_proposal', 'model_kind': kind,
               'label_definition': ('candidate_return_minus_BASE' if kind == 'gain'
                                    else 'BASE_return_minus_frozen_old_choice_return'),
               'aggregation': 'leaf mean of per-scene means of audited training gains',
               'structure': structure, 'structure_sha256': digest(structure), 'families': trees,
               'leaf_statistics': statistics, 'min_groups': min_groups, 'max_depth': max_depth,
               'prepared_directory': str(prepared), 'split': read(prepared/'split.json'),
               'input_sha256': {name: file_hash(prepared/name) for name in
                                ('training_rows.json', 'split.json', 'preparation.json', 'training_summary.json')},
               'dataset_sources': evidence['preparation']['dataset_sources'],
               'on_policy_only': evidence['preparation'].get('on_policy_only', False),
               'old_model_hash': old_hash}
    return {**content, 'proposal_sha256': digest(content)}


def verify(frozen, exported):
    exported = Path(exported)
    recomputed = proposal(frozen['prepared_directory'], frozen['structure'],
                          min_groups=frozen['min_groups'], max_depth=frozen['max_depth'])
    require(frozen == recomputed, 'proposal or its frozen training inputs changed')
    model, report = read(exported/'model.json'), read(exported/'report.json')
    structure, split = frozen['structure'], frozen['split']
    require(read(exported/'structure.json') == structure == report['structure']
            and report['structure_sha256'] == frozen['structure_sha256'], 'exported structure differs')
    require(report['construction'] == 'LLM_supplied_predicates_and_topology'
            and report['dataset_sources'] == frozen['dataset_sources']
            and report['on_policy_only'] == frozen['on_policy_only'], 'exported evidence binding differs')
    require(read(exported/'split.json') == split and sorted(split['training_group_ids']) == model['training_group_ids']
            and sorted(split['validation_group_ids']) == model['validation_group_ids'], 'exported split differs')
    require(report['split'] == {'train_groups': model['training_group_ids'],
                                'validation_groups': model['validation_group_ids']}, 'reported split differs')
    require(model['kind'] == 'local_'+frozen['model_kind']+'_model'
            and report['kind'] == frozen['model_kind'] and report['context_hash'] == model['context_hash']
            and model['feature_names'] == list(FEATURE_NAMES)
            and model.get('old_model_hash') == frozen['old_model_hash'], 'model/old-model kind or binding differs')
    require(model['weighting'] == 'equal_scenario_mass_within_each_leaf'
            and model['max_depth'] == frozen['max_depth'] and model['min_groups'] == frozen['min_groups']
            and model['families'] == frozen['families'], 'exported leaf values, topology or counts differ')
    expected = [{k: v for k, v in leaf.items() if k != 'sign_counts'} for leaf in frozen['leaf_statistics']]
    require(report['leaf_statistics'] == expected, 'exported leaf evidence or support differs')
    policy = load_policy(exported/'policy.json')
    embedded = policy['veto_model'] if frozen['model_kind'] == 'veto' else policy['model']
    require(model_hash(model) == report['model_hash'] and embedded == model, 'deployment policy differs')
    return {'verified': True, 'proposal_sha256': frozen['proposal_sha256'],
            'model_hash': report['model_hash'], 'leaves': len(expected)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    create = sub.add_parser('propose', help='populate and record the submitted topology')
    for flag in ('prepared', 'structure', 'output'):
        create.add_argument('--'+flag, type=Path, required=True)
    create.add_argument('--min-groups', type=int, default=4)
    create.add_argument('--max-depth', type=int, default=2)
    check = sub.add_parser('verify', help='check frozen proposal against exported model, report and policy')
    check.add_argument('--proposal', required=True, type=Path)
    check.add_argument('--export', required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == 'verify':
        result = verify(read(args.proposal), args.export)
    else:
        content = proposal(args.prepared, read(args.structure), min_groups=args.min_groups, max_depth=args.max_depth)
        with args.output.open('x', encoding='utf-8') as f:
            json.dump(content, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write('\n')
        result = {'proposal': str(args.output), 'proposal_sha256': content['proposal_sha256'],
                  'leaves': [{k: leaf[k] for k in ('family', 'route', 'value', 'groups', 'samples',
                                                   'supported', 'sign_counts')} for leaf in content['leaf_statistics']]}
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
