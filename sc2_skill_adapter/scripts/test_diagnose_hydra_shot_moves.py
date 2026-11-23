"""Offline regression tests for the reverse screen (no SC2 dependencies)."""
import unittest
from copy import deepcopy

if __package__:
    from .diagnose_hydra_shot_moves import screen, source
else:
    from diagnose_hydra_shot_moves import screen, source


class ReverseScreenTests(unittest.TestCase):
    def setUp(self):
        unit = dict(tag='10', type='Hydralisk', alive=True, weapon_cooldown_loops=8, weapon_ready=False,
                    orders=[dict(ability_id=3674, target_tag='20')])
        state = dict(game_loop=10, allies={'u0': unit}, visible_enemies={'e0': dict(tag='20', alive=True)},
                     relations={'u0': {'e0': dict(in_weapon_range=True, smac_attack_available=True)}})
        self.rec = dict(requested_action_id='2', submission='accepted', response={'code': 1}, command={'ability_id': 16})
        self.frame = dict(state=state, micro=dict(closed_loop=18, interior_checks=7, records=[], authorizations=0),
                          step=dict(feedback=dict(from_game_loop=10, to_game_loop=18, actors={'u0': self.rec})))
        self.after = deepcopy(state)
        self.after['game_loop'] = 18
        self.after['allies']['u0'].update(weapon_cooldown_loops=0, weapon_ready=True)
        self.row = dict(original_kind='move', actor_tag='10', original_action=2, final_action=2,
                        current_target='e0', target_tag='20', original_skill='ranged_fire_formation',
                        original_phase='exclusive_control', formation_move_role='alignment',
                        result='out_of_scope:provenance_not_retreat', retreat_geometry=True)

    def check(self):
        return screen(self.frame, self.after, 'u0', 2, self.row)

    def test_actual_move_outside_b_without_range_loss(self):
        status, case = self.check()
        self.assertEqual(status, 'move_only_candidate')
        self.assertEqual((case['source'], case['target_evidence']), ('formation_alignment', 'preserved_legal'))
        self.assertFalse(case['lost_native_range'])
        self.rec['submission'] = 'rejected'
        self.assertEqual(self.check()[0], 'unconfirmed_move')

    def test_target_tiers_and_range_vs_mask(self):
        self.row['target_tag'] = 'invalid'
        self.after['relations']['u0']['e0']['smac_attack_available'] = False
        case = self.check()[1]
        self.assertEqual(case['target_evidence'], 'native_order_legal')
        self.assertFalse(case['lost_native_range'])
        self.frame['state']['relations']['u0']['e0']['smac_attack_available'] = False
        self.assertEqual(self.check()[1]['target_evidence'], 'native_order_only_not_reissuable')
        self.frame['state']['allies']['u0']['orders'] = []
        self.assertEqual(self.check()[0], 'no_observed_target')
        self.frame['state']['relations']['u0']['e0']['smac_attack_available'] = True
        self.assertEqual(self.check()[1]['target_evidence'], 'any_legal_only')

    def test_no_future_target_substitution(self):
        self.after['visible_enemies']['e0']['tag'] = '99'
        self.assertEqual(self.check()[0], 'target_missing_dead_or_range_unknown')
        self.after['visible_enemies']['e3'] = dict(tag='20', alive=True)
        self.after['relations']['u0']['e3'] = dict(in_weapon_range=False, smac_attack_available=False)
        self.assertTrue(self.check()[1]['lost_native_range'])

    def test_no_restart_death_or_missing_confusion(self):
        unit = self.after['allies']['u0']
        unit.update(weapon_ready=False, weapon_cooldown_loops=2)  # lower than 8, higher than expected 0
        self.assertEqual(self.check()[0], 'restart_observed')
        unit['alive'] = False
        self.assertEqual(self.check()[0], 'actor_dead')
        unit['tag'] = '99'
        self.assertEqual(self.check()[0], 'actor_missing_or_replaced')
        self.after = None
        self.assertEqual(self.check()[0], 'no_next_macro_observation')

    def test_not_due_or_missing_cd(self):
        for cd, status in [(8.01, 'not_due'), (None, 'missing_cooldown')]:
            self.frame['state']['allies']['u0']['weapon_cooldown_loops'] = cd
            self.assertEqual(self.check()[0], status)

    def test_interior_command_and_incomplete_window(self):
        self.frame['micro'].update(authorizations=1, records=[dict(actor='u0', actor_tag='10', macro_action_id=2,
                                      status='submitted', submitted_loop=17, target_tag='20', sc2_response=[1])])
        self.assertEqual(self.check()[0], 'mixed_attack_window')
        self.frame['micro']['records'] = []
        self.assertEqual(self.check()[0], 'micro_window_missing_or_incomplete')

    def test_provenance_not_inferred_from_geometry_or_replaced_action(self):
        self.row.pop('formation_move_role')
        self.assertEqual(source(self.row), 'formation_unclassified')
        self.row['result'] = 'exempt:urgent_rescue'
        self.assertEqual(source(self.row), 'rescue')
        self.row['original_action'] = 3
        self.assertEqual(self.check()[1]['source'], 'unclassified')
        self.assertIn('no_matching_provenance', self.check()[1]['risk_flags'])


if __name__ == '__main__':
    unittest.main()
