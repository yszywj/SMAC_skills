"""Independent candidate estimates, public triggers and target semantics."""
from copy import deepcopy
import unittest

from decision_tree.features import FEATURE_NAMES, extract_features
from decision_tree.policy import choose, bind_call, candidate_calls, predict_vector_estimates
from decision_tree.schema import load_policy, validate_model
from decision_tree.targeting import choose_target, target_metrics, TARGET_METRIC_NAMES
from decision_tree.tests.test_invocation import combat_scene
from decision_tree.tests.test_integration import observed
from decision_tree.integration import SkillTreeController
from sc2_skill_adapter.bridge.policy import nearest_actions
from sc2_skill_adapter.bridge.skill_runtime import load_manager


def policy():
    return {'schema_version': 1, 'kind': 'skill_return_policy',
        'supported_races': ['protoss', 'terran', 'zerg'], 'exploration': {'enabled': True, 'seed': 1},
        'gate': {'phases': ['engage'], 'effect_check': 'call_config'},
        'candidates': [{'id': 'budget', 'skill': 'focus_fire_budget'},
                       {'id': 'protect', 'skill': 'adaptive_focus_fire', 'posture': 'defensive'},
                       {'id': 'short', 'skill': 'adaptive_focus_fire', 'max_steps': 2}],
        'selection': {'mode': 'base'}}


def model(gains=(1.5, 0., None), counts=(4, 4, 0)):
    estimates = {cid: {'gain': gain, 'n_samples': count, 'n_groups': count,
                      'group_ids': [f'g{i}' for i in range(count)], 'supported': count >= 4}
                 for cid, gain, count in zip(('budget', 'protect', 'short'), gains, counts)}
    return {'schema_version': 3, 'kind': 'skill_vector_gain_tree',
            'candidate_ids': ['budget', 'protect', 'short'], 'feature_names': list(FEATURE_NAMES),
            'min_candidate_groups': 4, 'tie_tolerance': 1e-9,
            'root': {'estimates': estimates, 'n_samples': sum(counts), 'n_groups': max(counts)}}


def scored(gains=(1.5, 0., None), counts=(4, 4, 0)):
    p = policy()
    p['selection'] = {'mode': 'model', 'model': model(gains, counts)}
    return p


class VectorRuntimeTests(unittest.TestCase):
    def selection(self, p):
        state = combat_scene()
        base = {'skill': 'adaptive_focus_fire'}
        features = extract_features(state)
        return choose(p, features, candidate_calls(p, state, base, features), base)

    def test_budget_evidence_never_raises_protect_or_missing_candidate(self):
        result = self.selection(scored())
        self.assertEqual(result['candidate_id'], 'budget')
        self.assertEqual(result['predicted_gains'], {'budget': 1.5, 'protect': 0., 'short': None})
        self.assertEqual(result['candidate_support']['short']['n_groups'], 0)
        self.assertFalse(result['candidate_support']['short']['supported'])

    def test_observed_high_gain_with_insufficient_groups_is_unsupported(self):
        result = self.selection(scored((-1., 9., None), (4, 3, 0)))
        self.assertEqual(result['candidate_id'], 'BASE')
        self.assertIsNone(result['predicted_gains']['protect'])

    def test_near_tied_positive_candidates_and_near_zero_keep_base(self):
        for gains in ((1., 1.+5e-10, None), (5e-10, 0., None), (0., -1., None)):
            self.assertEqual(self.selection(scored(gains))['candidate_id'], 'BASE')
        self.assertEqual(self.selection(scored((1., 1.+2e-9, None)))['candidate_id'], 'protect')

    def test_each_candidate_can_win_in_same_state(self):
        self.assertEqual(self.selection(scored((.5, 1., None)))['candidate_id'], 'protect')
        self.assertEqual(self.selection(scored((1., .5, None)))['candidate_id'], 'budget')

    def test_schema_rejects_zero_filled_missing_and_false_support(self):
        bad = model()
        bad['root']['estimates']['short']['gain'] = 0.
        with self.assertRaises(ValueError):
            validate_model(bad)
        bad = model((1., 1., None), (4, 3, 0))
        bad['root']['estimates']['protect']['supported'] = True
        with self.assertRaises(ValueError):
            validate_model(bad)
        bad = model()
        bad['root']['feature'] = 'ally_count'
        bad['root']['threshold'] = 1
        bad['root']['left'] = deepcopy(model()['root'])
        bad['root']['right'] = deepcopy(model()['root'])
        with self.assertRaises(ValueError):
            validate_model(bad)

    def test_schema_binds_candidates_and_state_feature_schema(self):
        with self.assertRaises(ValueError):
            validate_model(model(), ['protect', 'budget', 'short'])
        bad = model()
        bad['feature_names'].append('final_reward')
        with self.assertRaises(ValueError):
            validate_model(bad)

    def test_public_conditions_gate_each_candidate_independently(self):
        p, state = policy(), combat_scene()
        p['candidates'][0]['when'] = [{'feature': 'ally_recent_loss_ratio', 'op': 'ge', 'threshold': .2}]
        p['candidates'][1]['phases'] = ['search']
        base = {'skill': 'adaptive_focus_fire'}
        features = extract_features(state)
        self.assertEqual([c['id'] for c in candidate_calls(p, state, base, features)], ['short'])
        features['ally_recent_loss_ratio'] = .3
        self.assertEqual([c['id'] for c in candidate_calls(p, state, base, features)], ['budget', 'short'])
        p['candidates'][0]['when'][0]['feature'] = 'final_reward'
        with self.assertRaises(ValueError):
            load_policy(p)

    def test_target_rules_use_distinct_visible_legal_semantics(self):
        state = combat_scene()
        state['visible_enemies']['e1']['unit_type_id'] = 999
        state['unit_type_profiles'].update({
            '4': {'known': True, 'weapons': [{'target_type': 'ground', 'base_damage': 5., 'hits_per_attack': 1}]},
            '999': {'known': True, 'weapons': [{'target_type': 'ground', 'base_damage': 20., 'hits_per_attack': 2}]}})
        self.assertEqual(choose_target('lowest_effective_health', state), 'e0')
        self.assertEqual(choose_target('highest_threat', state), 'e1')
        state['allies']['u1'] = deepcopy(state['allies']['u0'])
        state['relations']['u1'] = {'e0': {'smac_attack_available': False, 'distance': 8.},
                                    'e1': {'smac_attack_available': True, 'distance': 2.}}
        self.assertEqual(choose_target('easiest_focus', state), 'e1')
        call = bind_call({'skill': 'adaptive_focus_fire', 'target_rule': 'highest_threat'}, state)
        self.assertEqual(call['priority_target'], 'e1')
        state['visible_enemies']['e1']['alive'] = False
        self.assertEqual(choose_target('highest_threat', state), 'e0')

    def test_target_metrics_do_not_invent_dps_or_hidden_enemies(self):
        state = combat_scene()
        unknown = target_metrics(state, 'e1')
        self.assertEqual(set(unknown), set(TARGET_METRIC_NAMES))
        self.assertEqual(unknown['threat_known'], 0.)
        self.assertIsNone(choose_target('highest_threat', state))
        self.assertEqual(target_metrics(state, 'hidden'), dict.fromkeys(TARGET_METRIC_NAMES, 0.))
        self.assertNotIn('threat_dps', unknown)

    def test_extended_features_and_support_reach_runtime_events(self):
        from decision_tree.choice_features import CHOICE_FEATURE_NAMES
        p = scored()
        p['exploration']['feature_schema'] = 3
        p['selection']['model']['feature_names'] = list(CHOICE_FEATURE_NAMES)
        controller = SkillTreeController(load_manager()(), p, race='protoss')
        _, _, events = controller.step(observed(combat_scene()), 1, nearest_actions)
        self.assertEqual(set(events[0]['features']), set(CHOICE_FEATURE_NAMES))
        self.assertEqual(events[0]['selected_id'], 'budget')
        self.assertEqual(events[0]['candidate_support']['budget']['n_groups'], 4)


if __name__ == '__main__':
    unittest.main()
