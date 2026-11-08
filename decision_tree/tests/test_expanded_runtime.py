"""Expanded collection retains causal anchors without an action-change gate."""
import copy
import unittest

from decision_tree.descriptors import MODEL_FEATURE_NAMES, candidate_features
from decision_tree.integration import SkillTreeController
from decision_tree.policy import predict_candidate_gain
from decision_tree.schema import load_policy, validate_model
from decision_tree.tests.test_integration import policy, observed, legacy_step
from decision_tree.tests.test_invocation import combat_scene
from sc2_skill_adapter.bridge.policy import nearest_actions
from sc2_skill_adapter.bridge.skill_runtime import load_manager


def expanded(mode='base', **selection):
    value = policy()
    value['exploration'] = {'enabled': True, 'seed': 49}
    value['candidates'] = [
        {'id': 'short', 'skill': 'adaptive_focus_fire', 'max_steps': 2},
        {'id': 'long', 'skill': 'adaptive_focus_fire', 'max_steps': 8},
        {'id': 'same', 'skill': 'adaptive_focus_fire'},
        {'id': 'unavailable', 'skill': 'heal_assign'},
    ]
    value['selection'] = {'mode': mode, **selection}
    return value


class ExpandedRuntimeTests(unittest.TestCase):
    def manager(self):
        return load_manager()()

    def decisions(self, controller, steps=18):
        state = combat_scene()
        events = []
        for _ in range(steps):
            _, _, current = controller.step(observed(state), 1, nearest_actions)
            events.extend(current)
            state['game_loop'] += 8
        return events

    def test_base_trajectory_unchanged_and_all_boundaries_exposed(self):
        old, new = self.manager(), self.manager()
        controller = SkillTreeController(new, expanded(), race='protoss')
        state, events = combat_scene(), []
        for _ in range(14):
            request = observed(state)
            expected, _ = legacy_step(old, copy.deepcopy(request), 1)
            actual, _, current = controller.step(request, 1, nearest_actions)
            self.assertEqual(actual, expected)
            self.assertEqual(new.unit_states, old.unit_states)
            events.extend(current)
            state['game_loop'] += 8
        self.assertGreater(len(events), 2)
        for event in events:
            self.assertTrue(event['opportunity'])
            self.assertEqual({c['id'] for c in event['candidate_calls']}, {'short', 'long'})
            self.assertEqual(event['candidate_rejected']['same'], 'identical_effective_config')
            self.assertEqual(event['selected_effective_config'], event['base_effective_config'])
        self.assertEqual(controller.interventions, 0)

    def test_exact_later_anchor_executes_even_with_equal_first_action(self):
        controller = SkillTreeController(self.manager(), expanded(
            'candidate', candidate_id='short', anchor_call_number=2), race='protoss')
        events = self.decisions(controller)
        chosen = [event for event in events if event['selected_id'] != 'BASE']
        self.assertEqual(len(chosen), 1)
        event = chosen[0]
        self.assertEqual(event['call_number'], 2)
        self.assertEqual(event['selected_id'], 'short')
        self.assertFalse(event['first_action_changed'])
        self.assertTrue(event['invocation_applied'])
        self.assertEqual(event['selected_effective_config'], event['candidate_effective_configs']['short'])
        self.assertNotEqual(event['selected_effective_config'], event['base_effective_config'])
        self.assertEqual(controller.manager.max_steps, 4)

    def test_unavailable_anchor_never_moves_to_later_opportunity(self):
        p = expanded('candidate', candidate_id='short', anchor_call_number=1)
        controller = SkillTreeController(self.manager(), p, race='protoss')
        state = combat_scene()
        state['visible_enemies'] = {}
        state['relations'] = {}
        _, _, first = controller.step(observed(state), 1, nearest_actions)
        self.assertTrue(first[0]['reason'].startswith('selected_anchor_unavailable:'))
        later = self.decisions(controller)
        self.assertTrue(later)
        self.assertTrue(all(e['selected_id'] == 'BASE' for e in later))

    def test_free_mode_budget_and_seed_are_reproducible(self):
        p = expanded('free', max_interventions_per_episode=3)
        first = self.decisions(SkillTreeController(self.manager(), p, race='protoss'), 40)
        second = self.decisions(SkillTreeController(self.manager(), p, race='protoss'), 40)
        self.assertEqual(first, second)
        selected = [event for event in first if event['selected_id'] != 'BASE']
        self.assertEqual(len(selected), 3)
        self.assertTrue(all(event['sampling_probability'] == .5 for event in selected))

    def test_scalar_model_scores_available_calls_and_obeys_intervention_budget(self):
        p = expanded('model')
        p['selection']['model'] = {
            'schema_version': 2, 'kind': 'skill_candidate_gain_tree',
            'feature_names': list(MODEL_FEATURE_NAMES),
            'root': {'feature': 'call_max_steps', 'threshold': 4,
                     'left': {'gain': 1.5}, 'right': {'gain': -.5}}}
        events = self.decisions(SkillTreeController(self.manager(), p, race='protoss'))
        selected = [e for e in events if e['selected_id'] != 'BASE']
        self.assertEqual([e['selected_id'] for e in selected], ['short'])
        self.assertEqual(selected[0]['predicted_gains'], {'short': 1.5, 'long': -.5})
        self.assertEqual(predict_candidate_gain(p['selection']['model'], selected[0]['features'],
                         selected[0]['selected_call'], selected[0]['base_call']), 1.5)

    def test_descriptors_exclude_identity_and_preserve_inheritance(self):
        from decision_tree.features import extract_features
        features = extract_features(combat_scene())
        base = {'skill': 'adaptive_focus_fire', 'max_steps': 4, 'posture': 'balanced',
                'adaptive_params': {'max_recover_steps': 0}}
        first = candidate_features(features, {'skill': 'adaptive_focus_fire', 'priority_target': 'e0'}, base)
        second = candidate_features(features, {'skill': 'adaptive_focus_fire', 'priority_target': 'e999'}, base)
        self.assertEqual(first, second)
        self.assertEqual(set(first), set(MODEL_FEATURE_NAMES))
        self.assertEqual(first['call_max_steps'], 4)
        self.assertEqual(first['call_has_max_recover_steps'], 1)
        self.assertEqual(first['call_max_recover_steps'], 0)

    def test_expanded_contract_rejects_ambiguous_rank_or_nonpublic_model(self):
        for p in (expanded('candidate', candidate_id='short', candidate_rank=0),
                  expanded('candidate', candidate_id='short', anchor_call_number=0),
                  expanded('free', max_interventions_per_episode=65)):
            with self.assertRaises(ValueError):
                load_policy(p)
        with self.assertRaises(ValueError):
            validate_model({'schema_version': 2, 'kind': 'skill_candidate_gain_tree',
                            'feature_names': list(MODEL_FEATURE_NAMES),
                            'root': {'feature': 'episode_return', 'threshold': 1,
                                     'left': {'gain': 1}, 'right': {'gain': -1}}})


if __name__ == '__main__':
    unittest.main()
