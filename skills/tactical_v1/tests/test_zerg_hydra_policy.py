"""Zerg focus-fire behavior through the native tactical step and commit path."""
from .fixtures import TacticalTestCase, enemy, scene


class ZergHydraPolicyTest(TacticalTestCase):
    def run_scene(self, state):
        manager = self.manager()
        actions, trace = self.step(manager, state)
        manager.commit_final_actions(actions)
        self.assertEqual(manager.memory.last_committed_actions['u0'], actions[0])
        self.assertEqual(trace['units']['u0']['skill'], 'focus_fire')
        return actions, trace

    def test_enemy_hydra_overrides_one_shot_ling(self):
        state = scene([((0., 0.), 'hydralisk')], {
            'e0': enemy((2., 0.), 'zergling', health=1.),
            'e1': enemy((4., 0.), 'hydralisk')}, map_name='10gen_zerg')
        self.assertEqual(self.run_scene(state)[0], [7])

    def test_fire_at_in_range_ling_then_press_during_cooldown(self):
        state = scene([((0., 0.), 'hydralisk')], {
            'e0': enemy((2., 0.), 'zergling'),
            'e1': enemy((6., 0.), 'hydralisk')}, map_name='10gen_zerg')
        manager = self.manager()
        for expected, cooldown in ((6, 0.), (4, 1.)):
            state['allies']['u0'].update(weapon_ready=cooldown == 0., weapon_cooldown_loops=cooldown)
            actions, trace = self.step(manager, state)
            self.assertEqual(actions, [expected])
            self.assertEqual(trace['units']['u0']['skill'], 'focus_fire')
            manager.commit_final_actions(actions)
            self.assertEqual(manager.memory.last_committed_actions['u0'], expected)
            state['game_loop'] += state['step_mul']

    def test_approach_boundary_without_lings(self):
        # The fixture's center-distance command cap is 5.5 (physical reach is 6).
        for x, expected in ((7.15, 4), (7.16, 1)):
            with self.subTest(distance=x):
                state = scene([((0., 0.), 'hydralisk')], {
                    'e0': enemy((x, 0.), 'hydralisk')}, map_name='10gen_zerg')
                self.assertEqual(self.run_scene(state)[0], [expected])

    def test_missing_pair_range_uses_catalog(self):
        state = scene([((0., 0.), 'hydralisk')], {
            'e0': enemy((6., 0.), 'hydralisk')}, map_name='10gen_zerg')
        state['relations']['u0']['e0'].update(native_weapon_range=None, weapon_range_status='unknown',
                                               in_weapon_range=None)
        self.assertEqual(self.run_scene(state)[0], [4])

    def test_cooldown_tracks_nearest_lings_actual_target(self):
        state = scene([((0., 0.), 'hydralisk'), ((0., 1.), 'zergling'),
                       ((4., 0.), 'zergling')], {
            'e0': enemy((-1., 1.), 'zergling'), 'e1': enemy((0., 3.), 'zergling')},
            map_name='10gen_zerg')
        state['allies']['u0'].update(weapon_ready=False, weapon_cooldown_loops=1.)
        # Only the northern target is an available order for the nearest ling.
        # The nearer enemy e0 would give the opposite horizontal axis.
        state['legal_actions'] = {'u1': [7], 'u2': [1]}
        actions, trace = self.run_scene(state)
        self.assertEqual(actions[:2], [3, 7])
        self.assertEqual(trace['units']['u0']['goal'], 'u1')

    def test_no_enemy_or_friendly_ling_holds(self):
        state = scene([((0., 0.), 'hydralisk')], {}, map_name='10gen_zerg')
        self.assertEqual(self.run_scene(state)[0], [1])

    def test_follow_lings_search_direction_without_visible_enemy(self):
        state = scene([((0., 0.), 'hydralisk'), ((-1., 0.), 'zergling')],
                      {}, map_name='10gen_zerg')
        state['legal_actions'] = {'u1': [4]}
        actions, trace = self.run_scene(state)
        self.assertEqual(actions, [5, 4])
        self.assertEqual(trace['units']['u0']['goal'], 'u1')

    def test_volley_reservations_distribute_finishing_shots(self):
        state = scene([((0., 0.), 'hydralisk'), ((0., 1.), 'hydralisk')], {
            'e0': enemy((2., 0.), 'hydralisk', health=1.),
            'e1': enemy((3., 0.), 'hydralisk')}, map_name='10gen_zerg')
        actions, trace = self.run_scene(state)
        self.assertEqual(actions, [6, 7])
        self.assertEqual(trace['units']['u1']['skill'], 'focus_fire')
