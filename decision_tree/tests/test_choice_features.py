"""Public candidate context and evidence derivation regression checks."""
from copy import deepcopy
import math
import unittest
from unittest.mock import patch

from decision_tree.choice_features import CHOICE_FEATURE_NAMES, extract_choice_features
from decision_tree.learning.choice_dataset import promote_choice_dataset
from decision_tree.tests.test_invocation import combat_scene
from decision_tree.tests.test_integration import observed
from decision_tree.integration import SkillTreeController
from decision_tree.schema import load_policy
from sc2_skill_adapter.bridge.policy import nearest_actions
from sc2_skill_adapter.bridge.skill_runtime import load_manager
from pathlib import Path


class ChoiceFeatureTests(unittest.TestCase):
    def test_context_is_finite_pure_and_does_not_consume_outcome_fields(self):
        state = combat_scene()
        before = deepcopy(state)
        a = extract_choice_features(state)
        self.assertEqual(state, before)
        self.assertEqual(set(a), set(CHOICE_FEATURE_NAMES))
        self.assertTrue(all(math.isfinite(v) for v in a.values()))
        state.update(reward=999, outcome='win', hidden_enemies={'unseen': {'health': 999}})
        state['episode'] = 99
        self.assertEqual(a, extract_choice_features(state))

    def test_no_target_is_missing_and_not_fictional_low_health_target(self):
        state = combat_scene()
        state['visible_enemies'] = {}
        a = extract_choice_features(state)
        self.assertEqual(a['target_weakest_available'], 0)
        self.assertEqual(a['target_threat_available'], 0)
        self.assertEqual(a['nearest_distance_known_fraction'], 0)

    def test_relative_target_health_and_attackability_change_context(self):
        state = combat_scene()
        a = extract_choice_features(state)
        self.assertEqual(a['target_weakest_available'], 1)
        state['visible_enemies']['e0']['health'] = 30
        b = extract_choice_features(state)
        self.assertNotEqual(a['target_weakest_effective_health'], b['target_weakest_effective_health'])
        for relation in state['relations']['u0'].values():
            relation['smac_attack_available'] = False
        self.assertEqual(extract_choice_features(state)['target_weakest_available'], 0)

    def test_retreat_context_uses_legal_public_moves_only(self):
        state = combat_scene()
        for relation in state['relations']['u0'].values():
            relation['in_enemy_weapon_range'] = True
        state['movement_candidates'] = {'u0': {
            '2': {'distances_to_visible_enemies': {'e0': 4, 'e1': 5}},
            '3': {'distances_to_visible_enemies': {'e0': 1, 'e1': 2}},
        }}
        features = extract_choice_features(state)
        self.assertEqual(features['ally_threatened_fraction'], 1)
        self.assertEqual(features['retreat_increasing_distance_fraction'], .5)

    def test_dataset_recomputes_and_rejects_tampered_vector(self):
        state = combat_scene()
        features = extract_choice_features(state)
        base = {'hashes': {'events': 'verified'},
                'run.json': {'skill_tree': {'policy': {'exploration': {'feature_schema': 3}}}}}
        episode = {'observations': {0: {'payload': {'state': state}}}}
        data = {'rows': [{'base_episode': 1, 'gate': {'step': 0}, 'state_features': features}],
                'provenance': {'BASE': {'hashes': base['hashes']}}, 'protocol': {}}
        with patch('decision_tree.learning.choice_dataset.read_run', return_value=base), \
                patch('decision_tree.learning.choice_dataset.episode', return_value=episode):
            result = promote_choice_dataset(data, 'fixture')
            self.assertEqual(result['schema_version'], 3)
            self.assertEqual(result['rows'][0]['vector_features'], features)
            self.assertNotIn('vector_features', data['rows'][0])
            data['rows'][0]['state_features']['target_weakest_health_ratio'] = 123
            with self.assertRaisesRegex(ValueError, 'derivation_mismatch'):
                promote_choice_dataset(data, 'fixture')

    def test_extended_previews_and_enabled_executors_do_not_change_base_actions(self):
        policy = load_policy(Path(__file__).resolve().parents[1]/'configs/candidate_v3.json')
        legacy = load_manager()()
        manager = load_manager()(spatial_params={'enable_regroup': True, 'enable_cohesive_search': True})
        controller = SkillTreeController(manager, policy, race='protoss')
        state = combat_scene()
        for i in range(12):
            state['game_loop'] = 8+i*8
            payload = observed(state)
            if not legacy.active:
                legacy.select('adaptive_focus_fire', posture='balanced')
            expected, _ = legacy.step(deepcopy(payload), 1)
            actual, _, events = controller.step(payload, 1, nearest_actions)
            self.assertEqual(actual, expected)
            self.assertEqual(manager.unit_states, legacy.unit_states)
            for event in events:
                self.assertEqual(event['selected_id'], 'BASE')
                self.assertEqual(set(event['features']), set(CHOICE_FEATURE_NAMES))


if __name__ == '__main__':
    unittest.main()
