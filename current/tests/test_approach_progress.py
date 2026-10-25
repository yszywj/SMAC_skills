"""Personal approach progress boundaries; entirely offline geometric fixtures."""
import copy
import unittest

from skills.current.skills.approach_progress import guard_approach, finalize_progress
from skills.current.skills.coordinated_approach import coordinate_approaches


def scene(loop=80, position=(0., 0.)):
    origin = list(position)
    unit = {'tag': 'ally-0', 'alive': True, 'type': 'Zealot', 'unit_type_id': 73,
            'position': origin, 'radius': .5, 'is_flying': False,
            'health': 100., 'shield': 50., 'max_health': 100., 'max_shield': 50.,
            'weapon_cooldown_loops': 0.}
    enemies = {f'e{i}': {'position': [6. + i * .1, 0.], 'radius': .5, 'alive': True,
                         'unit_type_id': 4, 'is_flying': False} for i in range(3)}
    destinations = {'2': [origin[0], origin[1] + 1.125],
                    '3': [origin[0], origin[1] - 1.125],
                    '4': [origin[0] + 1.125, origin[1]],
                    '5': [origin[0] - 1.125, origin[1]]}
    state = {'allies': {'u0': unit}, 'visible_enemies': enemies, 'game_loop': loop, 'step_mul': 8,
             'relations': {'u0': {enemy: {'native_weapon_range': .1, 'smac_attack_available': False,
                                        'smac_attack_distance_limit': 6., 'in_weapon_range': False}
                                  for enemy in enemies}},
             'movement_candidates': {'u0': {a: {'destination': p} for a, p in destinations.items()}},
             'unit_type_profiles': {'73': {'movement_speed_native': 2.25},
                                    '4': {'weapons': [{'target_type': 'ground', 'range': 7}]}}}
    return state, destinations


def cycle_history():
    return {'approach_progress': {'samples': [
        {'loop': 64, 'position': [0., 0.], 'tag': 'ally-0', 'action': 2,
         'target': 'e0', 'target_position': [8., 0.], 'cooldown': 0., 'expected_travel': 1.125},
        {'loop': 72, 'position': [0., 1.125], 'tag': 'ally-0', 'action': 3,
         'target': 'e1', 'target_position': [7., 0.], 'cooldown': 0., 'expected_travel': 1.125}]}}


def choose(state=None, history=None, original=2, moves=None):
    if state is None:
        state, candidates = scene()
    else:
        candidates = {a: row['destination'] for a, row in state['movement_candidates']['u0'].items()}
    if moves is not None:
        candidates = {a: p for a, p in candidates.items() if a in moves}
    features = {a: {'base_score': {'2': 0., '3': 2., '4': 1., '5': 4.}[a],
                    'estimated_contact_windows': 4., 'collision_cost': 0.,
                    'visible_danger_count': 0, 'blocked_move_penalty': 0.}
                for a in candidates}
    return guard_approach(state, 'u0', history or cycle_history(), 'e2', candidates, features,
                          original, list(candidates))


class ApproachProgressTests(unittest.TestCase):
    def test_own_aba_survives_enemy_approach_and_target_switch(self):
        state, _ = scene()
        history = cycle_history()
        pristine = copy.deepcopy((state, history))
        action, row, memory = choose(state, history)
        self.assertEqual(action, 4)
        self.assertTrue(row['cycle_detected'])
        self.assertTrue(row['target_changed'])
        self.assertEqual(row['own_net_displacement'], 0)
        self.assertEqual(row['own_target_progress'], 0)
        self.assertEqual(row['target_motion_progress'], 2)
        self.assertEqual(row['target_distance_progress'], 2)
        self.assertEqual(row['blocked_actions'], [2])
        self.assertEqual(memory['avoid_until_loop'], 96)
        self.assertEqual((state, history), pristine)

    def test_cycle_window_does_not_extend_on_repeated_or_continuing_detection(self):
        action, _, memory = choose()
        duplicate, row, duplicate_memory = choose(history={'approach_progress': memory})
        self.assertEqual(duplicate, action)
        self.assertTrue(row['repeated_observation'])
        self.assertEqual(duplicate_memory, memory)
        state, _ = scene(loop=88, position=(0., 1.125))
        _, row, following = choose(state, {'approach_progress': memory}, original=3)
        self.assertTrue(row['cycle_detected'])
        self.assertEqual(following['avoid_until_loop'], 96)

    def test_single_route_is_never_forbidden(self):
        action, row, _ = choose(moves=['2'])
        self.assertTrue(row['cycle_detected'])
        self.assertTrue(row['single_route_release'])
        self.assertEqual(action, 2)
        self.assertFalse(row['applied'])

    def test_intentional_frontline_wait_is_not_overridden_by_cycle_memory(self):
        action, row, memory = choose(original=1)
        self.assertEqual(action, 1)
        self.assertFalse(row['applied'])
        self.assertEqual(row['reset_reason'], 'intentional_nonmoving_approach')
        self.assertEqual(memory, {})

    def test_gap_rollback_replacement_and_wait_break_the_chain(self):
        for change, expected in (({'game_loop': 96}, 'nonconsecutive_observation'),
                                 ({'game_loop': 64}, 'nonconsecutive_observation'),
                                 ({'tag': 'replacement'}, 'unit_replaced'),
                                 ({'action': 1}, 'previous_approach_did_not_move')):
            with self.subTest(change=change):
                state, _ = scene()
                history = cycle_history()
                if 'game_loop' in change:
                    state.update(change)
                elif 'tag' in change:
                    state['allies']['u0'].update(change)
                else:
                    history['approach_progress']['samples'][-1].update(change)
                action, row, memory = choose(state, history)
                self.assertEqual(action, 2)
                self.assertFalse(row['cycle_detected'])
                self.assertEqual(row['reset_reason'], expected)
                self.assertEqual(len(memory['samples']), 1)
                self.assertEqual(memory['avoid_until_loop'], 0)

    def test_observed_shot_releases_but_unknown_cooldown_does_not_invent_shot(self):
        state, _ = scene()
        state['allies']['u0']['weapon_cooldown_loops'] = 12.
        _, row, memory = choose(state)
        self.assertTrue(row['shot_proxy'])
        self.assertFalse(row['cycle_detected'])
        self.assertEqual(memory['avoid_until_loop'], 0)
        state['allies']['u0']['weapon_cooldown_loops'] = None
        _, row, _ = choose(state)
        self.assertFalse(row['shot_proxy'])
        self.assertTrue(row['cycle_detected'])

    def test_two_blocked_movements_trigger_without_enemy_progress(self):
        history = cycle_history()
        for sample in history['approach_progress']['samples']:
            sample.update(position=[0., 0.], action=4)
        _, row, memory = choose(history=history, original=4)
        self.assertFalse(row['cycle_detected'])
        self.assertTrue(row['blocked_stall'])
        self.assertIn(4, row['blocked_actions'])
        self.assertNotEqual(memory['samples'][-1]['action'], 4)

    def test_guard_expiry_and_real_self_progress_release_avoidance(self):
        _, _, memory = choose()
        state, _ = scene(loop=88, position=(1.125, 0.))
        _, row, progress = choose(state, {'approach_progress': memory}, original=4)
        self.assertFalse(row['active'])
        self.assertEqual(row['reset_reason'], 'observed_self_progress')
        self.assertEqual(progress['avoid_until_loop'], 0)
        # A stationary latest sample at the expiry time is not an ABA renewal.
        memory['samples'] = [memory['samples'][-1]]
        memory['samples'][-1]['loop'] = 88
        state, _ = scene(loop=96, position=(0., 0.))
        _, row, expired = choose(state, {'approach_progress': memory})
        self.assertFalse(row['active'])
        self.assertEqual(expired['avoid_until_loop'], 0)

    def test_final_filter_action_is_saved_and_reversed_proposal_not_counted(self):
        state, candidates = scene()
        _, row, memory = choose(state)
        updates = {'u0': {'approach_progress': memory}}
        diagnostics = {'u0': {'progress_guard': row, 'predicted_destinations': candidates}}
        original = copy.deepcopy(state)
        # Guard proposed east, but a later filter selected the original north.
        finalize_progress(state, [2], {'u0': 'reposition'}, {'u0': cycle_history()}, updates, diagnostics)
        self.assertEqual(updates['u0']['approach_progress']['samples'][-1]['action'], 2)
        self.assertEqual(row['final_action'], 2)
        self.assertFalse(row['applied'])
        self.assertTrue(row['postfilter_changed'])
        self.assertEqual(state, original)

    def test_exiting_approach_does_not_change_attacks_retreat_or_recovery(self):
        for role, action, dead, legal in (('attack', 6, False, True),
                                          ('retreat', 3, False, False),
                                          ('recover', 5, False, False),
                                          ('reposition', 4, False, True),
                                          ('reposition', 0, True, False)):
            with self.subTest(role=role, dead=dead, legal=legal):
                state, _ = scene()
                state['allies']['u0']['alive'] = not dead
                state['relations']['u0']['e0']['smac_attack_available'] = legal
                actions, updates, diagnostics = [action], {}, {}
                finalize_progress(state, actions, {'u0': role}, {'u0': cycle_history()}, updates, diagnostics)
                self.assertEqual(actions, [action])
                self.assertEqual(updates['u0']['approach_progress'], {})

    def test_coordinate_hook_is_opt_in_and_never_overrides_legal_attack(self):
        state, _ = scene()
        history = {'u0': cycle_history()}
        original = copy.deepcopy((state, history))
        plain = coordinate_approaches(state, ['u0'], history)
        off = coordinate_approaches(state, ['u0'], history, parameters={'approach_progress_guard': False})
        self.assertEqual(plain, off)
        _, diagnostics, updates = coordinate_approaches(state, ['u0'], history,
                                                       parameters={'approach_progress_guard': True})
        self.assertTrue(diagnostics['u0']['progress_guard']['cycle_detected'])
        self.assertIn('approach_progress', updates['u0'])
        self.assertEqual((state, history), original)
        state['relations']['u0']['e0']['smac_attack_available'] = True
        actions, _, _ = coordinate_approaches(state, ['u0'], history,
                                              parameters={'approach_progress_guard': True})
        self.assertNotIn('u0', actions)


if __name__ == '__main__':
    unittest.main()
