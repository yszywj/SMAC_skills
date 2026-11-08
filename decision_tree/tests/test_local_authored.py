"""LLM topology stays fixed; statistics and exports are program-verified.

All data here is synthetic contract evidence, not a performance result.
"""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from decision_tree.local.authored import (build, load_datasets, main, make_split,
                                         prepare, validate_split)
from decision_tree.local.models import model_hash, predict, validate_model
from decision_tree.local.runtime import load_policy
from decision_tree.tests.test_local_models import row, dataset
from decision_tree.tests.test_local_pipeline import paired_batch


def structure(node=None, family='retarget'):
    return {'schema_version': 1, 'kind': 'local_authored_structure',
            'rationale': 'Synthetic test structure, explicitly supplied by the test author.',
            'families': {family: node or {'leaf': True}}}


def split(training, validation):
    return {'schema_version': 1, 'kind': 'local_scene_split',
            'training_group_ids': training, 'validation_group_ids': validation}


class AuthoredTests(unittest.TestCase):
    def test_exact_authored_split_and_equal_scene_mass_inside_leaf(self):
        rows = [row('A', gain=100, x=0, anchor=f'a{i}') for i in range(10)]
        rows += [row('A', gain=-100, x=1, anchor='a10'), row('B', gain=0, x=0),
                 row('C', gain=-50, x=1), row('V', gain=9999, x=0)]
        authored = structure({'feature': 'x', 'threshold': 0.25,
                              'left': {'leaf': True}, 'right': {'leaf': True}})
        untouched = deepcopy(authored)
        with patch('decision_tree.local.models.fit_gain', side_effect=AssertionError('must never search')):
            model, _, report = build(dataset(rows), authored, split(['A', 'B', 'C'], ['V']), min_groups=2)
        self.assertEqual(authored, untouched)
        tree = model['families']['retarget']
        self.assertEqual(tree['threshold'], 0.25)
        self.assertEqual(tree['left']['value'], 50)
        self.assertEqual(tree['right']['value'], -75)
        self.assertEqual(tree['left']['samples'], 11)
        self.assertEqual(tree['left']['groups'], 2)
        self.assertEqual(report['construction'], 'LLM_supplied_predicates_and_topology')
        self.assertEqual(model['weighting'], 'equal_scenario_mass_within_each_leaf')

    def test_validation_returns_cannot_change_authored_leaf_statistics(self):
        data = dataset([row('T', gain=2), row('V', gain=-100)])
        model, _, _ = build(data, structure(), split(['T'], ['V']), min_groups=1)
        changed = deepcopy(data)
        changed['rows'][1].update(gain=1000, candidate_return=1010)
        other, _, _ = build(changed, structure(), split(['T'], ['V']), min_groups=1)
        self.assertEqual(model, other)

    def test_sparse_leaves_abstain_and_absent_families_are_not_imputed(self):
        data = dataset([row('T', gain=2), row('V', gain=2)])
        model, _, report = build(data, structure(), split(['T'], ['V']))
        self.assertFalse(predict(model, 'retarget', {'x': 0})['supported'])
        self.assertFalse(predict(model, 'retreat_step', {'x': 0})['supported'])
        self.assertFalse(report['leaf_statistics'][0]['supported'])

    def test_empty_authored_leaf_rejected_without_repairing_the_tree(self):
        data = dataset([row('T'), row('V')])
        authored = structure({'feature': 'x', 'threshold': 100,
                              'left': {'leaf': True}, 'right': {'leaf': True}})
        with self.assertRaisesRegex(ValueError, 'no training evidence'):
            build(data, authored, split(['T'], ['V']))

    def test_future_features_invented_leaf_values_and_excess_depth_rejected(self):
        data = dataset([row('T'), row('V')])
        node = {'feature': 'gain', 'threshold': 0, 'left': {'leaf': True}, 'right': {'leaf': True}}
        with self.assertRaisesRegex(ValueError, 'unlisted'):
            build(data, structure(node), split(['T'], ['V']))
        with self.assertRaisesRegex(ValueError, 'recomputed'):
            build(data, structure({'value': 999, 'groups': 100, 'samples': 100}), split(['T'], ['V']))
        node['feature'] = 'x'
        with self.assertRaisesRegex(ValueError, 'maximum depth'):
            build(data, structure(node), split(['T'], ['V']), max_depth=0)

    def test_split_retains_no_opportunity_scenes_without_inventing_rows(self):
        data = dataset([row('T'), row('V')])
        data['protocol'] = {'plan': {'scenes': [{'group_id': g} for g in ('T', 'V', 'EMPTY')]}}
        s = make_split(data, seed=5)
        self.assertEqual(set(s['training_group_ids'] + s['validation_group_ids']), {'T', 'V', 'EMPTY'})
        with self.assertRaisesRegex(ValueError, 'every source scene'):
            validate_split(split(['T'], ['V']), data)
        prepared = prepare(data, split=split(['T', 'EMPTY'], ['V']))
        self.assertEqual(len(prepared['training_rows']), 1)
        model, _, _ = build(data, structure(), prepared['split'], min_groups=1)
        self.assertEqual(model['training_group_ids'], ['EMPTY', 'T'])
        self.assertEqual(model['families']['retarget']['groups'], 1)

    def test_fixed_split_overlap_and_unknown_scene_rejected(self):
        data = dataset([row('T'), row('V')])
        for bad in (split(['T'], ['T', 'V']), split(['T'], ['V', 'UNKNOWN'])):
            with self.assertRaises(ValueError):
                validate_split(bad, data)

    def test_authored_veto_keeps_old_model_and_correct_label_sign(self):
        old, old_policy, _ = build(dataset([row('oldT', gain=2), row('oldV', gain=2)]),
                           structure(), split(['oldT'], ['oldV']), min_groups=1)
        data = dataset([row('newT', gain=-3), row('newV', gain=-3)])
        model, policy, report = build(data, structure(), split(['newT'], ['newV']),
                                      kind='veto', old_model=old, min_groups=1)
        self.assertEqual(model['families']['retarget']['value'], 3)
        self.assertEqual(model['old_model_hash'], model_hash(old))
        self.assertEqual(policy['model'], old)
        self.assertEqual(report['validation']['mean_gain'], 3)
        data['protocol'] = {'plan': {'scenes': [{'group_id': 'oldV'}]}}
        with self.assertRaisesRegex(ValueError, 'unseen'):
            prepare(data, kind='veto', old_model=old,
                    split=split(['newT', 'oldV'], ['newV']))

    def test_new_weighting_is_validated_but_unknown_weighting_rejected(self):
        model, _, _ = build(dataset([row('T'), row('V')]), structure(), split(['T'], ['V']))
        validate_model(model)
        model['weighting'] = 'untracked_average'
        with self.assertRaisesRegex(ValueError, 'weighting'):
            validate_model(model)

    def test_on_policy_veto_only_uses_audited_first_proposal_anchors(self):
        old, old_policy, _ = build(dataset([row('oldT', gain=2), row('oldV', gain=2)]),
                           structure(), split(['oldT'], ['oldV']), min_groups=1)
        rows = [row(group, gain=gain, anchor=anchor)
                for group in ('newT', 'newV') for anchor, gain in (('first', -3), ('later', 100))]
        for r in rows:
            r['reference_first_proposal'] = r['anchor_id'] == 'first'
        data = dataset(rows)
        data['protocol'] = {'plan': {'reference_policy': old_policy}}
        model, _, report = build(data, structure(), split(['newT'], ['newV']), kind='veto',
                                  old_model=old, min_groups=1, on_policy_only=True)
        self.assertEqual(model['families']['retarget']['value'], 3)
        self.assertTrue(report['on_policy_only'])
        self.assertEqual(report['validation']['mean_gain'], 3)
        bad_reference = deepcopy(data)
        bad_reference['protocol']['plan']['reference_policy']['selection']['min_gain'] = 1
        with self.assertRaisesRegex(ValueError, 'min_gain=0'):
            prepare(bad_reference, kind='veto', old_model=old, on_policy_only=True)
        bad_reference['protocol']['plan']['reference_policy']['selection']['min_gain'] = 0
        bad_reference['protocol']['plan']['reference_policy']['model']['families']['retarget']['value'] = 100
        with self.assertRaisesRegex(ValueError, 'different frozen gain model'):
            prepare(bad_reference, kind='veto', old_model=old, on_policy_only=True)
        for r in data['rows']:
            r['reference_first_proposal'] = False
        with self.assertRaisesRegex(ValueError, 'no audited first-reference-proposal'):
            prepare(data, kind='veto', old_model=old, on_policy_only=True)

    def test_audited_cli_prepares_and_exports_gain_then_veto_without_search(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path, _, _ = paired_batch(root/'gain-data', 'authored-gain', 2)
            authored_path = root/'structure.json'
            authored_path.write_text(json.dumps(structure()), encoding='utf-8')
            with patch('decision_tree.local.models.fit_gain', side_effect=AssertionError('must not search')):
                with redirect_stdout(io.StringIO()):
                    main(['prepare', '--dataset', str(path), '--output', str(root/'prepared')])
                    main(['gain', '--dataset', str(path), '--structure', str(authored_path),
                          '--split', str(root/'prepared/split.json'), '--output', str(root/'gain')])
            self.assertFalse((root/'prepared/validation_rows.json').exists())
            self.assertFalse((root/'prepared/model.json').exists())
            policy = load_policy(root/'gain/policy.json')
            self.assertEqual(policy['model']['weighting'], 'equal_scenario_mass_within_each_leaf')
            correction, _, _ = paired_batch(root/'veto-data', 'authored-veto', -2)
            with redirect_stdout(io.StringIO()):
                main(['prepare', '--kind', 'veto', '--dataset', str(correction),
                      '--old-model', str(root/'gain/model.json'), '--output', str(root/'veto-prepared')])
                main(['veto', '--dataset', str(correction), '--structure', str(authored_path),
                      '--split', str(root/'veto-prepared/split.json'), '--old-model', str(root/'gain/model.json'),
                      '--output', str(root/'veto')])
            veto = load_policy(root/'veto/policy.json')
            self.assertEqual(veto['model'], policy['model'])
            self.assertEqual(veto['veto_model']['families']['retarget']['value'], 2)
            with self.assertRaisesRegex(ValueError, 'duplicate candidate'):
                load_datasets([path, path])


if __name__ == '__main__':
    unittest.main()
