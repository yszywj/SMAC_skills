"""Synthetic contracts for sparse anchored labels; fixtures are not game evidence."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from decision_tree.descriptors import candidate_features
from decision_tree.experiment import digest, policy_contract
from decision_tree.learning.dataset import episode
from decision_tree.learning.expanded_dataset import audit_expanded_pair, build_expanded_dataset
from decision_tree.learning.expanded_train import (evaluate_tested_candidates, fit_scalar_tree,
                                                  load_dataset, train_expanded, validate_rows)
from decision_tree.learning.train import group_split, group_weights
from decision_tree.tests.test_dataset import synthetic_run, event, IDS


def expanded_run(arm='BASE', gain=0., call_end=5):
    run = synthetic_run(arm, gain, call_end=call_end)
    policy = run['run.json']['skill_tree']['policy']
    policy['exploration'] = {'enabled': True, 'seed': 11}
    policy['gate']['effect_check'] = 'call_config'
    if arm != 'BASE':
        policy['selection']['anchor_call_number'] = 1
    run['run.json']['skill_tree']['policy_contract'] = policy_contract(policy)
    gate = event(run, 'skill_tree_decision', 1)
    base_config = {'skill': 'adaptive_focus_fire', 'priority_target': None, 'max_steps': 4}
    configs = {IDS[0]: {**base_config, 'priority_target': 'e1'}, IDS[1]: {**base_config, 'posture': 'defensive'}}
    gate.update(base_effective_config=base_config, candidate_effective_configs=configs,
                selected_effective_config=base_config if arm == 'BASE' else configs[arm], invocation_applied=True)
    return run


def data_fixture():
    runs = {'BASE': expanded_run(), 'candidate': expanded_run(IDS[0], 2.)}
    with patch('decision_tree.learning.expanded_dataset.read_run', side_effect=lambda path: deepcopy(runs[path])):
        return build_expanded_dataset('BASE', [{'path': 'candidate', 'base_episode': 1, 'anchor_id': 'a1',
            'anchor_call_number': 1, 'candidate_id': IDS[0], 'candidate_rank': 9}])


def training_fixture():
    data = data_fixture()
    template = data['rows'][0]
    data['rows'] = []
    for i in range(12):
        for cid in IDS if i % 3 == 0 else (IDS[0],):
            row = deepcopy(template)
            row['group_id'] = f'scenario_{i}'
            row['sc2_seed'] = i
            row['candidate_id'] = cid
            row['gate']['candidate_id'] = cid
            row['gain'] = (-2. if i < 6 else 2.) * (1. if cid == IDS[0] else -1.)
            row['candidate_return'] = row['base_return'] + row['gain']
            row['state_features']['game_loop'] = i
            if cid == IDS[1]:
                row['candidate_call'] = {**row['base_call'], 'posture': 'defensive'}
            row['features'] = candidate_features(row['state_features'], row['candidate_call'], row['base_call'])
            data['rows'].append(row)
    return data


class ExpandedAuditTests(unittest.TestCase):
    def pair(self, candidate=None, base=None):
        return audit_expanded_pair(episode(base or expanded_run(), 1),
                                   episode(candidate or expanded_run(IDS[0], 2.), 1), 1, IDS[0])

    def test_candidate_plan_rank_is_not_runtime_rank_when_id_is_explicit(self):
        data = data_fixture()
        self.assertEqual(data['rejected'], [])
        self.assertEqual(len(data['rows']), 1)
        self.assertEqual(data['rows'][0]['gain'], 2.)

    def test_applied_action_equivalent_call_keeps_actual_measured_label(self):
        base, candidate = expanded_run(), expanded_run(IDS[0], .01)
        for run in (base, candidate):
            event(run, 'skill_tree_decision', 1)['candidate_actions'][IDS[0]] = [6, 6]
        event(candidate, 'step', 2)['actions'] = [6, 6]
        audit, _, _ = self.pair(candidate, base)
        self.assertFalse(audit['first_action_changed'])
        self.assertFalse(audit['action_changed_during_comparison_window'])

    def test_only_administrative_episode_ordinal_is_normalized(self):
        base, candidate = expanded_run(), expanded_run(IDS[0])
        for run, number in ((base, 29), (candidate, 1)):
            for e in run['events.jsonl']:
                if e['event'] == 'skill_observation':
                    e['payload']['state']['episode'] = number
        self.pair(candidate, base)
        event(candidate, 'skill_observation', 1)['payload']['state']['visible'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'observed_prefix'):
            self.pair(candidate, base)

    def test_multiple_baseline_opportunities_do_not_block_label(self):
        base, candidate = expanded_run(), expanded_run(IDS[0])
        event(base, 'skill_tree_decision', 5)['opportunity'] = True
        event(candidate, 'skill_tree_decision', 5)['opportunity'] = True
        self.pair(candidate, base)

    def test_extra_intervention_is_rejected_even_if_not_opportunity(self):
        candidate = expanded_run(IDS[0])
        event(candidate, 'skill_tree_decision', 5)['selected_id'] = IDS[1]
        with self.assertRaisesRegex(ValueError, 'isolated_intervention'):
            self.pair(candidate)

    def test_not_applied_or_config_mismatch_is_rejected(self):
        for field, value in (('invocation_applied', False), ('selected_effective_config', {'bad': 1})):
            candidate = expanded_run(IDS[0])
            event(candidate, 'skill_tree_decision', 1)[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.pair(candidate)

    def test_full_longer_base_call_still_checks_engine_feedback(self):
        base, candidate = expanded_run(), expanded_run(IDS[0], call_end=3)
        event(base, 'step', 4)['feedback']['actors']['u0']['submission'] = 'rejected'
        with self.assertRaisesRegex(ValueError, 'command_rejected'):
            self.pair(candidate, base)

    def test_bad_candidate_does_not_erase_other_independent_label(self):
        runs = {'BASE': expanded_run(), 'good': expanded_run(IDS[0]), 'bad': expanded_run(IDS[1])}
        event(runs['bad'], 'skill_observation', 1)['payload']['state']['visible'] = 'different'
        arms = [{'path': path, 'base_episode': 1, 'anchor_call_number': 1, 'candidate_id': cid}
                for path, cid in [('good', IDS[0]), ('bad', IDS[1])]]
        with patch('decision_tree.learning.expanded_dataset.read_run', side_effect=lambda path: deepcopy(runs[path])):
            data = build_expanded_dataset('BASE', arms)
        self.assertEqual(len(data['rows']), 1)
        self.assertEqual(len(data['rejected']), 1)
        self.assertEqual(len(data['comparisons']), 2)


class ExpandedTrainingTests(unittest.TestCase):
    def test_sparse_rows_valid_without_unexecuted_zero_label(self):
        data = training_fixture()
        validate_rows(data['rows'])
        self.assertNotEqual(len(data['rows']), 12 * len(IDS))

    def test_group_weight_and_split_keep_all_anchors_candidates_together(self):
        rows = training_fixture()['rows']
        for row, weight in zip(rows, group_weights(rows)):
            self.assertAlmostEqual(weight, 1. / sum(r['group_id'] == row['group_id'] for r in rows))
        train, val = group_split(rows)
        self.assertFalse({r['group_id'] for r in train} & {r['group_id'] for r in val})

    def test_duplicate_and_forged_gain_rejected(self):
        rows = training_fixture()['rows']
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            validate_rows(rows + [deepcopy(rows[0])])
        rows[0]['gain'] = 999
        with self.assertRaisesRegex(ValueError, 'gain must'):
            validate_rows(rows)

    def test_descriptor_cannot_be_overwritten_with_outcome(self):
        rows = training_fixture()['rows']
        rows[0]['features']['game_loop'] = 999
        with self.assertRaisesRegex(ValueError, 'derivation'):
            validate_rows(rows)

    def test_leaf_requires_independent_groups_not_just_rows(self):
        rows = training_fixture()['rows'][:2]
        with self.assertRaisesRegex(ValueError, 'independent'):
            fit_scalar_tree(rows, context_hash='x', min_leaf_groups=2, min_samples_leaf=1)

    def test_scalar_tree_and_offline_tested_subset_report(self):
        rows = training_fixture()['rows']
        model = fit_scalar_tree(rows, context_hash='x', min_leaf_groups=2, min_samples_leaf=2)
        model['candidate_ids'] = list(IDS)
        report = evaluate_tested_candidates(model, rows)
        self.assertEqual(report['n_groups'], 12)
        self.assertEqual(report['n_anchors'], 12)
        self.assertLessEqual(report['selector_mean_gain'], report['tested_subset_oracle_mean_gain'])
        for row in report['rows']:
            self.assertIn(row['selected'], ['BASE', *row['tested_candidates']])

    def test_offline_positive_ties_match_runtime_declaration_order(self):
        from decision_tree.descriptors import MODEL_FEATURE_NAMES
        from decision_tree.policy import choose
        data = training_fixture()
        rows = deepcopy(data['rows'][:2])
        policy = deepcopy(data['policy_contract'])
        # Declaration order is deliberately opposite alphabetical order.
        policy['candidates'].reverse()
        order = [candidate['id'] for candidate in policy['candidates']]
        model = {'schema_version': 2, 'kind': 'skill_candidate_gain_tree',
                 'feature_names': list(MODEL_FEATURE_NAMES), 'candidate_ids': order, 'root': {'gain': 1.}}
        rows[0]['gain'], rows[1]['gain'] = -2., 2.
        available = [{'id': row['candidate_id'], 'call': row['candidate_call']} for row in rows]
        for gain, expected in ((1., IDS[1]), (0., 'BASE')):
            model['root']['gain'] = gain
            policy['selection'] = {'mode': 'model', 'model': model, 'max_interventions_per_episode': 1, 'min_gain': 0.}
            actual = evaluate_tested_candidates(model, rows)['rows'][0]
            runtime = choose(policy, rows[0]['state_features'], available, rows[0]['base_call'])
            with self.subTest(predicted_gain=gain):
                self.assertEqual(actual['selected'], expected)
                self.assertEqual(actual['selected'], runtime['candidate_id'])
                self.assertEqual(actual['selected_empirical_gain'], 2. if gain else 0.)

    def test_offline_evaluation_rejects_missing_order_and_unknown_candidates(self):
        rows = training_fixture()['rows'][:2]
        for ids in (None, [IDS[0]], [IDS[0], IDS[0]]):
            model = {'candidate_ids': ids, 'root': {'gain': 1.}}
            with self.subTest(candidate_ids=ids), self.assertRaises(ValueError):
                evaluate_tested_candidates(model, rows)

    def test_full_training_exports_loadable_model_and_disjoint_split(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / 'pairs.json'
            dataset.write_text(json.dumps(training_fixture()), encoding='utf-8')
            policy, model, report = train_expanded(dataset, Path(directory) / 'model', min_leaf_groups=2)
            self.assertEqual(policy['selection']['mode'], 'model')
            self.assertEqual(model['kind'], 'skill_candidate_gain_tree')
            self.assertEqual(report['status'], 'trained_not_game_validated')
            self.assertFalse(set(report['split']['train_groups']) & set(report['split']['validation_groups']))
            self.assertTrue((Path(directory) / 'model' / 'report.json').exists())

    def test_sealed_test_cannot_be_used_for_training(self):
        data = training_fixture()
        data['split'] = 'test'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'data.json'
            path.write_text(json.dumps(data), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'sealed'):
                load_dataset(path)


if __name__ == '__main__':
    unittest.main()
