"""Safety regressions for known weapons and bounded critical-heal decisions."""
import copy
import unittest

from .test_terran_combat import scene, unit
from ..skills.movement_scoring import threat_count, visible_volley_pressure
from ..skills.race_movement_guard import evaluate_retreat_override, retreat_override_allowed


FIX = {'known_compatible_threats': True}


class WeaponThreatTests(unittest.TestCase):
    def test_medivac_does_not_manufacture_damage_but_legacy_is_reproducible(self):
        state = scene([unit('Marine', (0, 0))], {'e0': unit('Medivac', (0, 1))})
        original = copy.deepcopy(state)
        self.assertEqual((threat_count(state, 'u0', [0, 0]), visible_volley_pressure(state, 'u0', [0, 0])), (1, 12))
        self.assertEqual((threat_count(state, 'u0', [0, 0], FIX), visible_volley_pressure(state, 'u0', [0, 0], FIX)), (0, 0))
        self.assertEqual(state, original)

    def test_ground_only_enemy_does_not_threaten_air_but_still_threatens_ground(self):
        state = scene([unit('Medivac', (0, 0)), unit('Marine', (0, 0))], {'e0': unit('Marauder', (1, 0))})
        self.assertEqual(threat_count(state, 'u0', [0, 0], FIX), 0)
        self.assertEqual(visible_volley_pressure(state, 'u0', [0, 0], FIX), 0)
        self.assertEqual(threat_count(state, 'u1', [0, 0], FIX), 1)
        self.assertEqual(visible_volley_pressure(state, 'u1', [0, 0], FIX), 10)

    def test_unknown_or_omitted_effect_weapon_is_not_declared_harmless(self):
        state = scene([unit('Marine', (0, 0))], {'e0': unit('Marine', (1, 0))})
        state['relations']['u0']['e0']['enemy_weapon_range_status'] = 'unknown'
        for profile in ({'known': False}, {'known': True, 'weapons': []}):
            state['unit_type_profiles']['48'] = profile
            self.assertEqual(threat_count(state, 'u0', [0, 0], FIX), 1)
            self.assertEqual(visible_volley_pressure(state, 'u0', [0, 0], FIX), 12)

    def test_catalog_compatibility_can_exclude_without_relation(self):
        state = scene([unit('Medivac', (0, 0))], {'e0': unit('Marauder', (1, 0))})
        state['relations']['u0'] = {}
        self.assertEqual(threat_count(state, 'u0', [0, 0], FIX), 0)


class CriticalCoordinationGuardTests(unittest.TestCase):
    def make_case(self, enemy_x=6):
        return scene([unit('Marine', (0, 0), 10), unit('Medivac', (0, 1))],
                     {'e0': unit('Marine', (enemy_x, 0))})

    def context(self, **kwargs):
        return {'healer': 'u1', 'primary_target': 'e0', 'allow_reentry_delay': True,
                'allow_blocked_reversal': False, 'reversal_used': False, **kwargs}

    def test_critical_healing_may_delay_primary_shot_one_window(self):
        state = self.make_case();original = copy.deepcopy(state)
        self.assertFalse(retreat_override_allowed(state, 'u0', 4, 5, FIX))
        result = evaluate_retreat_override(state, 'u0', 4, 5, FIX, coordination=self.context())
        self.assertTrue(result['allowed'])
        self.assertIn('critical_heal_one_window_reentry', result['exceptions'])
        self.assertEqual(result['reentry']['e0']['proposal']['estimated_shot_loops'], 16)
        self.assertEqual(state, original)

    def test_healthy_unpaired_and_unspecified_target_cannot_claim_exception(self):
        for variant in ('healthy', 'unpaired', 'no_target', 'zero_energy'):
            state = self.make_case();context = self.context()
            if variant == 'healthy':state['allies']['u0']['health'] = 30
            elif variant == 'unpaired':state['heal_candidates']['u1'] = {}
            elif variant == 'no_target':context.pop('primary_target')
            else:state['allies']['u1']['energy'] = 0
            with self.subTest(variant=variant):
                self.assertFalse(retreat_override_allowed(state, 'u0', 4, 5, FIX, coordination=context))

    def test_more_than_one_window_of_delay_remains_rejected(self):
        state = self.make_case(8)
        state['movement_candidates']['u0']['4']['destination'] = [2, 0]
        state['movement_candidates']['u0']['5']['destination'] = [-2, 0]
        result = evaluate_retreat_override(state, 'u0', 4, 5, FIX, coordination=self.context())
        self.assertFalse(result['allowed'])
        self.assertIn('reentry_delayed:e0', result['reasons'])

    def test_healing_never_buys_extra_real_weapon_exposure(self):
        state = self.make_case(6.5)
        result = evaluate_retreat_override(state, 'u0', 5, 4, FIX, coordination=self.context())
        self.assertFalse(result['allowed'])
        self.assertIn('threat_count_increased', result['reasons'])
        self.assertIn('base_volley_pressure_increased', result['reasons'])

    def test_blocked_reversal_needs_progress_evidence_and_unused_allowance(self):
        state = scene([unit('Marine', (0, 0), 10), unit('Medivac', (-1, 0))])
        history = dict(tag=state['allies']['u0']['tag'], loop=0, action=4, position=[0, 0])
        context = self.context(allow_blocked_reversal=True)
        result = evaluate_retreat_override(state, 'u0', 4, 5, FIX, history, coordination=context)
        self.assertTrue(result['allowed'])
        self.assertIn('critical_heal_blocked_reversal', result['exceptions'])
        for variant in ('already_used', 'made_progress', 'stale', 'wrong_reference'):
            h = copy.deepcopy(history);c = dict(context)
            if variant == 'already_used':c['reversal_used'] = True
            elif variant == 'made_progress':h['position'] = [-.5, 0]
            elif variant == 'stale':h['loop'] = -8
            else:c['healer'] = 'u0'
            with self.subTest(variant=variant):
                # Stale history is ignored entirely and therefore does not
                # authorize a reversal exception (nor falsely forbid a turn).
                row = evaluate_retreat_override(state, 'u0', 4, 5, FIX, h, coordination=c)
                self.assertNotIn('critical_heal_blocked_reversal', row['exceptions'])
                if variant != 'stale':self.assertFalse(row['allowed'])


if __name__ == '__main__':
    unittest.main()
