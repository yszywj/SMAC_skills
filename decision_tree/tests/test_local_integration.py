"""Exercise the real allocator/manager commit boundary, not a mock executor."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

from decision_tree.local.runtime import LocalController, load_policy, validate_options, execution_context
from decision_tree.tests.test_integration import observed, legacy_step
from decision_tree.tests.test_invocation import combat_scene
from sc2_skill_adapter.bridge.skill_runtime import load_manager
from skills.local_v1 import FEATURE_NAMES


def policy():
    return load_policy(Path(__file__).parents[1] / 'local/base.json')


class LocalIntegrationTests(unittest.TestCase):
    def test_base_preserves_actual_controller_state_across_windows(self):
        old, new = load_manager()(), load_manager()()
        controller = LocalController(new, policy())
        state = combat_scene()
        for _ in range(9):
            p = observed(state)
            before = copy.deepcopy(p)
            expected, log = legacy_step(old, copy.deepcopy(p), 1)
            actual, actual_log, events = controller.step(p, 1)
            self.assertEqual(p, before)
            self.assertEqual(actual, expected)
            self.assertEqual(actual_log, log)
            for key in ('unit_states', 'target_assignments', '_last_move', 'all_in',
                        'stats', 'elapsed_steps', '_last_seen_enemies'):
                self.assertEqual(getattr(new, key), getattr(old, key), key)
            self.assertEqual(events[0]['selected_id'], 'BASE')
            state['game_loop'] += 8

    def test_forced_patch_commits_final_target_and_only_one_step(self):
        p = observed(combat_scene())
        probe = LocalController(load_manager()(), policy())
        _, _, events = probe.step(p, 1)
        candidate = next(c for c in events[0]['candidates'] if c['family'] == 'retarget')
        cfg = policy()
        cfg['selection'].update(mode='forced', anchor_step=0, candidate_id=candidate['id'])
        manager = load_manager()()
        controller = LocalController(manager, cfg)
        actions, telemetry, events = controller.step(p, 1)
        self.assertEqual(actions[0], candidate['actions']['u0'])
        self.assertEqual(manager.target_assignments['u0'], 'e' + str(actions[0] - 6))
        self.assertEqual(manager.unit_states['u0']['role'], candidate['roles']['u0'])
        self.assertEqual(telemetry['attack_assignments'], manager.target_assignments)
        # The next decision uses ordinary baseline policy on its resulting history.
        p['state']['game_loop'] += 8
        expected_manager = copy.deepcopy(manager)
        expected_manager.local_decision_hook = None
        expected = expected_manager.step(copy.deepcopy(p), 1)[0]
        actual, _, later = controller.step(p, 1)
        self.assertEqual(actual, expected)
        self.assertEqual(later[0]['selected_id'], 'BASE')
        self.assertEqual(controller.interventions, 1)

    def test_same_snapshot_is_idempotent_and_reset_clears_budget(self):
        manager = load_manager()()
        controller = LocalController(manager, policy())
        p = observed(combat_scene())
        a, log, events = controller.step(p, 1)
        count = manager.stats['steps']
        b, _, again = controller.step(p, 1)
        self.assertEqual(a, b)
        self.assertFalse(again)
        self.assertEqual(manager.stats['steps'], count)
        changed = copy.deepcopy(p)
        changed['state']['allies']['u0']['health'] -= 1
        with self.assertRaisesRegex(ValueError, 'same clock'):
            controller.step(changed, 1)
        manager.reset(); controller.reset()
        self.assertEqual(controller.step(p, 1)[2][0]['step'], 0)

    def test_missing_forced_candidate_fails_instead_of_silent_base(self):
        cfg = policy()
        cfg['selection'].update(mode='forced', anchor_step=0, candidate_id='missing')
        controller = LocalController(load_manager()(), cfg)
        with self.assertRaisesRegex(ValueError, 'missing'):
            controller.step(observed(combat_scene()), 1)

    def test_rejects_conflicting_owners_and_shared_memory_switches(self):
        for changes in ({'skill_tree': 'old.json'}, {'enable_regroup': True},
                        {'native_teacher_control': 'all'}, {'race_profile': 'terran'}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_options(SimpleNamespace(**changes), 'adaptive-skill', {'map_name': '10gen_protoss'})
        validate_options(SimpleNamespace(), 'adaptive-skill', {'map_name': '10gen_protoss'})

    def test_context_ignores_arm_but_binds_baseline_and_candidate_semantics(self):
        metadata = {'env_args': {'map_name': '10gen_protoss'}, 'source_hashes': {'a': 'hash'},
                    'skill_parameters': {'posture': 'balanced'}}
        p = policy(); q = policy()
        q['selection'].update(mode='forced', candidate_id='x', anchor_step=2)
        self.assertEqual(execution_context(metadata, p), execution_context(metadata, q))
        q['candidates']['max_candidates'] += 1
        self.assertNotEqual(execution_context(metadata, p), execution_context(metadata, q))

    def test_supported_gain_can_be_vetoed_before_commit(self):
        from decision_tree.local.models import fit_gain, model_hash
        p = observed(combat_scene())
        _, _, events = LocalController(load_manager()(), policy()).step(p, 1)
        c = next(c for c in events[0]['candidates'] if c['family'] == 'retarget')
        rows = [{'group_id': str(i), 'candidate_family': c['family'],
                 'features': c['features'], 'gain': 2.0} for i in range(4)]
        model = fit_gain(rows, list(FEATURE_NAMES), 'context')
        veto = copy.deepcopy(model)
        veto.update(kind='local_veto_model', old_model_hash=model_hash(model), old_min_gain=0)
        cfg = policy(); cfg['selection']['mode'] = 'model'
        cfg.update(model=model, veto_model=veto)
        manager = load_manager()()
        controller = LocalController(manager, cfg, context_hash='context')
        actions, _, events = controller.step(p, 1)
        self.assertEqual(actions, events[0]['baseline_actions'])
        self.assertTrue(events[0]['vetoed'])
        self.assertEqual(controller.interventions, 0)
        self.assertEqual(controller.proposals_used, 1)
        p['state']['game_loop'] += 8
        later = controller.step(p, 1)[2][0]
        self.assertEqual(later['proposed_id'], 'BASE')
        self.assertEqual(later['reason'], 'intervention_budget_exhausted')


if __name__ == '__main__':
    unittest.main()
