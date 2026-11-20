import unittest

from ..context import Context, build_snapshot
from ..ranged_targets import support_attack
from .fixtures import enemy, scene, with_questions


class RangedTargetTests(unittest.TestCase):
    def _state(self, old_type='zergling'):
        state = scene([((1., 1.), 'hydralisk'), ((4., 1.), 'hydralisk'),
                       ((6., 2.), 'hydralisk')],
                      {'e0': enemy((3., 3.), 'hydralisk'),
                       'e1': enemy((0., 4.), 'hydralisk'),
                       'e2': enemy((7., 1.), old_type)})
        for relations in state['relations'].values():
            relations['e2'].update(in_enemy_weapon_range=False,
                                   enemy_weapon_range_status='out_of_range')
        return state

    def _choice(self, state):
        ctx = Context(build_snapshot(with_questions(state), 1))
        return support_attack(ctx, 'u1', 'e0', 'u0', {'u0': 1, 'u1': 8, 'u2': 8})

    def test_offer_return_fire_only_while_threat_concentrates(self):
        state = self._state()
        self.assertEqual(self._choice(state), 6)
        state['relations']['u0']['e1']['in_enemy_weapon_range'] = False
        self.assertIsNone(self._choice(state))

    def test_keep_interception_and_finishing_volleys(self):
        self.assertIsNone(self._choice(self._state('baneling')))
        state = self._state()
        # Both 12-HP shots are needed to finish the existing 20-HP target.
        state['visible_enemies']['e2']['health'] = 20.
        self.assertIsNone(self._choice(state))

    def test_keep_cover_of_another_wounded_ally(self):
        state = self._state()
        state['allies']['u2']['health_lost_since_previous_decision'] = 12.
        state['relations']['u2']['e2'].update(in_enemy_weapon_range=True,
                                              enemy_weapon_range_status='in_range')
        self.assertIsNone(self._choice(state))

    def test_closed_mask_or_long_cooldown_cannot_issue_return_fire(self):
        state = self._state()
        state['relations']['u1']['e0']['smac_attack_available'] = False
        self.assertIsNone(self._choice(state))
        state = self._state()
        state['allies']['u1']['weapon_cooldown_loops'] = 9.
        self.assertIsNone(self._choice(state))
