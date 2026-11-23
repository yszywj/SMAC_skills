import math
import unittest

from skills.tactical_v1.context import Context, build_snapshot
from skills.tactical_v1.hydra_support import zergling_guard_plans
from .fixtures import enemy, scene, with_questions


class ZerglingScreenTests(unittest.TestCase):
    def _scene(self, ling, *, cooling=False, bomber=None):
        allies = [(ling, 'zergling'), ((6., 10.), 'hydralisk')]
        if bomber:
            allies.append((bomber, 'baneling'))
        state = scene(allies, {'e0': enemy((10., 10.), 'hydralisk')}, attack_range=5.)
        state['relations']['u0']['e0'].update(in_weapon_range=False,
            smac_attack_available=False, weapon_range_status='out_of_range',
            native_weapon_range=.1, smac_attack_distance_limit=2.)
        state['relations']['u1']['e0'].update(in_enemy_weapon_range=True,
                                               enemy_weapon_range_status='in_range')
        if cooling:
            state['allies']['u1']['weapon_cooldown_loops'] = 24.
        return state

    def _plans(self, state):
        context = Context(build_snapshot(with_questions(state), 1))
        return context, zergling_guard_plans(context, None, 'test')

    def test_flank_reaches_melee_band_and_carries_covering_shot(self):
        state = self._scene((8., 11.))
        context, plans = self._plans(state)
        flank = next(p for p in plans if p.goal_data['role'] == 'harass')
        objective = flank.goal_data['objective']['position']
        self.assertTrue(flank.atomic)
        self.assertEqual(context.decode('u1', flank.current_actions['u1']).target, 'e0')
        self.assertLessEqual(math.dist(objective, (10., 10.)), context.effective_attack_reach('u0', 'e0'))
        self.assertLess(objective[0], 10.)
        self.assertGreater(objective[1], 10.)
        state['relations']['u0']['e0']['native_weapon_range'] = None
        self.assertFalse(any(p.goal_data['role'] == 'harass' for p in self._plans(state)[1]))

    def test_only_free_ling_can_screen_until_slot_occupied_by_bomber(self):
        _, plans = self._plans(self._scene((6., 7.), cooling=True))
        self.assertEqual([(p.goal_data['role'], p.current_actions) for p in plans],
                         [('rear_guard', {'u0': 2})])
        _, plans = self._plans(self._scene((6., 7.), cooling=True, bomber=(7., 10.)))
        self.assertEqual(plans, [])
