"""Local family support, scene leakage, and correction sign contract."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from decision_tree.local.models import (fit_gain, predict, select_candidate,
                                       validate_model, model_hash)
from decision_tree.local.training import (validate_dataset, group_split,
                                         correction_rows, train, evaluate, load_dataset)


def row(group, family='retarget', gain=2, x=0, cid='target', anchor='a1'):
    return {'group_id': group, 'anchor_id': anchor, 'candidate_id': cid,
            'candidate_family': family, 'features': {'x': x}, 'gain': gain,
            'base_return': 10, 'candidate_return': 10 + gain,
            'available_candidate_ids': [cid], 'anchor_complete': True}


def dataset(rows):
    return {'schema_version': 1, 'kind': 'local_patch_dataset', 'audited': True,
            'context_hash': 'test-context', 'feature_names': ['x'], 'rows': rows,
            'game_version': 'test-version', 'candidate_config': {'max_candidates': 24, 'max_per_family': 8}}


def model(rows=None, **kw):
    rows = rows or [row(f'train{i}') for i in range(4)]
    return fit_gain(rows, ['x'], 'test-context', **kw)


class LocalModelTests(unittest.TestCase):
    def test_different_families_never_share_leaf_value(self):
        m = model([row(f'g{i}', gain=3) for i in range(4)] +
                  [row(f'g{i}', 'retreat_step', -2, cid='escape') for i in range(4)])
        self.assertEqual(predict(m, 'retarget', {'x': 0})['value'], 3)
        self.assertEqual(predict(m, 'retreat_step', {'x': 0})['value'], -2)
        self.assertIsNone(predict(m, 'regroup_step', {'x': 0})['value'])

    def test_one_scene_cannot_become_four_support_groups(self):
        m = model([row('only', anchor=f'a{i}') for i in range(20)])
        p = predict(m, 'retarget', {'x': 0}, min_groups=1)
        self.assertEqual(p['groups'], 1)
        self.assertFalse(p['supported'])  # support threshold cannot be weakened

    def test_scene_weights_do_not_count_repeated_rows_as_more_evidence(self):
        m = model([row('one', gain=100, anchor=f'a{i}') for i in range(30)] +
                  [row('two', gain=0)], max_depth=0, min_groups=1)
        self.assertAlmostEqual(predict(m, 'retarget', {'x': 0})['value'], 50)

    def test_splits_obey_per_leaf_scene_support(self):
        rows = [row(f'g{i}', gain=-2 if i < 4 else 3, x=i) for i in range(8)]
        m = model(rows)
        self.assertIn('feature', m['families']['retarget'])
        self.assertEqual(predict(m, 'retarget', {'x': 0})['value'], -2)
        self.assertEqual(predict(m, 'retarget', {'x': 7})['groups'], 4)

    def test_positive_ties_are_stable_but_nonpositive_values_keep_base(self):
        m = model()
        candidates = [{'id': 'a', 'family': 'retarget', 'features': {'x': 0}},
                      {'id': 'b', 'family': 'retarget', 'features': {'x': 1}}]
        self.assertEqual(select_candidate(m, candidates)['candidate_id'], 'a')
        self.assertEqual(select_candidate(m, list(reversed(candidates)))['candidate_id'], 'a')
        self.assertEqual(select_candidate(m, candidates[:1])['candidate_id'], 'a')
        self.assertEqual(select_candidate(m, candidates[:1], min_gain=3)['candidate_id'], 'BASE')
        zero = model([row(f'g{i}', gain=0) for i in range(4)])
        self.assertEqual(select_candidate(zero, candidates)['candidate_id'], 'BASE')

    def test_bad_numeric_fields_and_unlisted_features_fail_closed(self):
        m = model()
        for features in ({'x': float('nan')}, {'x': True}, {'x': 0, 'reward': 10}, {}):
            with self.assertRaises(ValueError):
                predict(m, 'retarget', features)
        bad = deepcopy(m)
        bad['families']['retarget'] = {'feature': 'reward', 'threshold': 0,
                                     'left': m['families']['retarget'], 'right': m['families']['retarget']}
        with self.assertRaises(ValueError):
            validate_model(bad)

    def test_hash_binds_context_and_values(self):
        m = model()
        copied = deepcopy(m)
        self.assertEqual(model_hash(m), model_hash(copied))
        copied['context_hash'] = 'another'
        self.assertNotEqual(model_hash(m), model_hash(copied))


class LocalTrainingTests(unittest.TestCase):
    def test_dataset_rejects_unmeasured_fabricated_and_duplicate_labels(self):
        good = dataset([row('g1')])
        validate_dataset(good)
        bad = deepcopy(good); bad['audited'] = False
        with self.assertRaises(ValueError): validate_dataset(bad)
        bad = deepcopy(good); bad['rows'][0]['gain'] = 99
        with self.assertRaises(ValueError): validate_dataset(bad)
        bad = deepcopy(good); bad['rows'].append(deepcopy(bad['rows'][0]))
        with self.assertRaises(ValueError): validate_dataset(bad)
        bad = deepcopy(good); bad['rows'][0]['available_candidate_ids'].append('missing')
        with self.assertRaises(ValueError): validate_dataset(bad)

    def test_group_split_never_separates_candidates_from_same_scene(self):
        rows = [row(f'g{i}', anchor=f'a{j}') for i in range(8) for j in range(3)]
        a, b = group_split(rows)
        self.assertFalse({r['group_id'] for r in a} & {r['group_id'] for r in b})
        self.assertEqual(len(a) + len(b), len(rows))

    def test_correction_requires_unseen_complete_frozen_model_choices(self):
        old = model()
        data = dataset([row(f'new{i}', gain=-3) for i in range(6)])
        corrected, rejected = correction_rows(data, old)
        self.assertFalse(rejected)
        self.assertEqual(corrected[0]['gain'], 3)
        self.assertEqual(corrected[0]['old_model_hash'], model_hash(old))
        overlap = dataset([row('train0', gain=-3)])
        with self.assertRaisesRegex(ValueError, 'unseen'):
            correction_rows(overlap, old)
        partial = deepcopy(data)
        for r in partial['rows']:
            r['available_candidate_ids'].append('missing'); r['anchor_complete'] = False
        with self.assertRaisesRegex(ValueError, 'no complete'):
            correction_rows(partial, old)

    def test_correction_does_not_train_nonselected_candidate(self):
        old = model([row(f'g{i}', gain=2) for i in range(4)] +
                    [row(f'g{i}', 'retreat_step', gain=-2, cid='escape') for i in range(4)])
        rows = [row('new', gain=-1), row('new', 'retreat_step', gain=-100, cid='escape')]
        for r in rows: r['available_candidate_ids'] = ['target', 'escape']
        fixed, _ = correction_rows(dataset(rows), old)
        self.assertEqual(len(fixed), 1)
        self.assertEqual(fixed[0]['candidate_id'], 'target')
        self.assertEqual(fixed[0]['gain'], 1)

    def test_training_exports_split_model_and_veto_policy(self):
        data = dataset([row(f'new{i}', gain=-3) for i in range(8)])
        veto, policy, report = train(data, kind='veto', old_model=model())
        self.assertEqual(veto['kind'], 'local_veto_model')
        self.assertEqual(veto['game_version'], 'test-version')
        self.assertEqual(policy['veto_model']['old_model_hash'], model_hash(policy['model']))
        self.assertFalse(set(veto['training_group_ids']) & set(report['split']['validation_groups']))
        self.assertEqual(veto['validation_group_ids'], report['split']['validation_groups'])
        self.assertEqual(report['validation']['mean_gain'], 3)

    def test_incomplete_anchors_not_scored_as_complete_selection(self):
        r = row('new'); r['available_candidate_ids'].append('missing'); r['anchor_complete'] = False
        report = evaluate(model(), [r])
        self.assertIsNone(report['mean_gain'])
        self.assertEqual(report['incomplete_anchors_excluded'], 1)

    def test_cli_loader_reaudits_sources_instead_of_trusting_audit_flag(self):
        from decision_tree.local.dataset import build_dataset
        from decision_tree.tests.test_local_dataset import fixture, P1
        with tempfile.TemporaryDirectory() as td:
            folder = Path(td)
            base, arm = folder/'base', folder/'arm'
            fixture(base, candidates=1)
            fixture(arm, P1, 3.0, candidates=1)
            data = build_dataset(base, [{'path': str(arm), 'base_episode': 1,
                                        'anchor_step': 1, 'candidate_id': P1}])
            path = folder/'paired.json'
            path.write_text(json.dumps(data), encoding='utf-8')
            self.assertEqual(load_dataset(path)['rows'][0]['gain'], 2)
            data['rows'][0]['features']['actor_health_ratio'] = 99
            path.write_text(json.dumps(data), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'changed during training re-audit'):
                load_dataset(path)


if __name__ == '__main__':
    unittest.main()
