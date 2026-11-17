"""Joint burst accounting, independent of pending native effect calibration."""
import copy
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from skills.tactical_v1.context import Context, build_snapshot
from skills.tactical_v1.effects import EffectsReport
from skills.tactical_v1.evaluator import Evaluator
from .fixtures import scene, enemy, with_questions


def context(state):
    return Context(build_snapshot(with_questions(state), state['game_loop']))


class BurstJointScoringTests(unittest.TestCase):
    def setUp(self):
        self.context = context(scene([((0., 0.), 'marine'), ((0., 2.), 'baneling'),
                                      ((0., 4.), 'baneling')],
            enemies={'e0': enemy((4., 0.), health=100., max_health=100.),
                     'e1': enemy((4., 3.), health=50., max_health=50.)}))
        self.evaluator = Evaluator()

    def score(self, report, risk=None, unsupported=None):
        original = copy.deepcopy(report)
        pressure = {'pressure_relief': 0., 'unsupported_actors': unsupported or {},
                    'unknown_enemy_pairs': 0}
        with patch.object(self.evaluator, 'effects', return_value=report), \
             patch.object(self.evaluator, '_forecast_costs', return_value=(risk or {}, 0)), \
             patch.object(self.evaluator, 'task_quality', return_value=(0., {})), \
             patch.object(self.evaluator, '_coordination_costs', return_value={}), \
             patch.object(self.evaluator, '_future_attack_value', return_value=0.), \
             patch.object(self.evaluator, '_sustained_contact_value', return_value=0.), \
             patch('skills.tactical_v1.battle_pressure.joint_pressure', return_value=pressure), \
             patch('skills.tactical_v1.targeting.allocation_progress', return_value=0.):
            result = self.evaluator.score_joint(self.context, [], {})[1]
        self.assertEqual(report, original, 'score queries must not consume or rewrite the report')
        return result

    def test_ordinary_and_burst_share_each_targets_remaining_ehp(self):
        row = self.score(EffectsReport(damage_known={'e0': 60.},
                                      burst_damage_potential={'e0': 80.}))
        self.assertAlmostEqual(row['useful_damage'], 1.8)
        self.assertAlmostEqual(row['burst_opportunity'], 1.2)
        self.assertEqual(row['expected_kills'], 0., 'a potential burst is not a known volley')
        self.assertAlmostEqual(row['useful_damage'] + row['burst_opportunity'], 3.)

    def test_ordinary_kill_leaves_no_reserve_for_burst_and_no_extra_kill(self):
        row = self.score(EffectsReport(damage_known={'e0': 130.},
                                      burst_damage_potential={'e0': 200.}))
        self.assertEqual(row['useful_damage'], 3.)
        self.assertEqual(row['expected_kills'], 2.)
        self.assertEqual(row['burst_opportunity'], 0.)

    def test_multiple_bursts_share_one_budget_per_target_without_cross_target_loss(self):
        one = self.score(EffectsReport(damage_known={'e0': 80.},
            burst_damage_potential={'e0': 30.}, burst_outcomes={'u1': SimpleNamespace(opportunity=1.)}))
        two = self.score(EffectsReport(damage_known={'e0': 80.},
            burst_damage_potential={'e0': 60.}, burst_outcomes={
                'u1': SimpleNamespace(opportunity=1.), 'u2': SimpleNamespace(opportunity=1.)}))
        self.assertAlmostEqual(one['burst_opportunity'], .6)
        self.assertEqual(one['burst_opportunity'], two['burst_opportunity'])
        self.assertEqual(two['burst_sources'], 2.)
        separate = self.score(EffectsReport(damage_known={'e0': 80.},
            burst_damage_potential={'e0': 60., 'e1': 25.}))
        self.assertAlmostEqual(separate['burst_opportunity'], .6 + 1.5)

    def test_unknown_burst_preserves_known_partial_credit_without_inventing_damage_or_kill(self):
        report = EffectsReport(damage_known={'e0': 10.},
            burst_damage_potential={'e0': 20.}, burst_unknown_sources={'e0': ['u2']},
            burst_outcomes={'u1': SimpleNamespace(opportunity=1.), 'u2': SimpleNamespace(opportunity=1.)})
        mixed = self.score(report)
        self.assertAlmostEqual(mixed['useful_damage'], .3)
        self.assertAlmostEqual(mixed['burst_opportunity'], .6)
        self.assertEqual(mixed['expected_kills'], 0.)
        self.assertEqual(mixed['burst_unknown_sources'], 1.)
        report.burst_damage_potential.clear()
        unknown = self.score(report)
        self.assertEqual(unknown['burst_opportunity'], 0.)
        self.assertEqual(unknown['useful_damage'], mixed['useful_damage'])
        self.assertEqual(unknown['expected_kills'], 0.)

    def test_planned_consumption_and_forecast_risk_use_max_per_actor(self):
        report = EffectsReport(planned_loss={'u1': 1.})
        for risk, expected in ((.4, 1.5), (2., 2.)):
            row = self.score(report, {'u1': risk})
            self.assertAlmostEqual(-row['risk'] - row['planned_loss'], expected)
            self.assertEqual(row['planned_loss_raw'], -1.5)
        separate = self.score(report, {'u0': 2.})
        self.assertEqual(-separate['risk'] - separate['planned_loss'], 3.5,
                         'another actor cannot cancel this source\'s self-consumption')
        two = self.score(EffectsReport(planned_loss={'u1': 1., 'u2': .5}),
                         {'u1': .4, 'u2': 1.})
        self.assertEqual(-two['risk'] - two['planned_loss'], 2.5)

    def test_death_trigger_potential_is_diagnostic_not_a_second_risk_charge(self):
        base = self.score(EffectsReport(), {'u0': 2.})
        row = self.score(EffectsReport(enemy_death_burst={'u0': 35.}), {'u0': 2.})
        self.assertEqual(row['enemy_death_burst_potential'], 35.)
        self.assertEqual(row['total'], base['total'])

    def test_planned_consumption_also_deduplicates_same_actor_unsupported_exposure(self):
        report = EffectsReport(planned_loss={'u1': 1.})
        for risk in (.1, 0.):
            row = self.score(report, {'u1': risk}, {'u1': 1.})
            self.assertAlmostEqual(-row['risk'] - row['planned_loss']
                                   - row['unsupported_frontline'], 1.5)
        different = self.score(report, {'u1': .1}, {'u0': 1.})
        self.assertAlmostEqual(-different['risk'] - different['planned_loss']
                               - different['unsupported_frontline'], 2.)
        self.assertEqual(different['unsupported_frontline'], -.5,
                         'consuming one source cannot pay for another actor\'s exposure')


class FrozenS1ScoringCompatibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[3]
        frozen = root / 'runs/tactical_all_races_10v10_9x20_mechanisms_20261115/before_source_snapshot/skills/tactical_v1'
        if not (frozen / 'evaluator.py').exists():
            raise unittest.SkipTest('optional immutable pre-change snapshot unavailable')
        def module(name, path):
            spec = importlib.util.spec_from_file_location('skills.tactical_v1.' + name, path)
            loaded = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = loaded
            spec.loader.exec_module(loaded)
            return loaded
        cls.old_effects = module('_test_before_burst_effects', frozen / 'effects.py')
        cls.old_evaluator = module('_test_before_burst_evaluator', frozen / 'evaluator.py')

    def test_pure_s1_actions_keep_frozen_damage_coverage_risk_and_total(self):
        for cooldown, actions in ((0., {'u0': 6, 'u1': 6}),
                                  (12., {'u0': 4, 'u1': 6}),
                                  (0., {'u0': 1, 'u1': 7})):
            with self.subTest(cooldown=cooldown, actions=actions):
                state = scene([((0., 0.), 'colossus'), ((0., 2.), 'colossus')],
                    enemies={'e0': enemy((4., 0.), 'zealot', health=90., max_health=100.),
                             'e1': enemy((4., 1.), 'stalker', health=150., max_health=150.),
                             'e2': enemy((5., 2.), 'zealot', health=100., max_health=100.)},
                    attack_range=7.)
                state['allies']['u0']['weapon_cooldown_loops'] = cooldown
                old, new = self.old_evaluator.Evaluator(), Evaluator(
                    common_combat={'joint_assessment_enabled': False})
                c_old, c_new = context(copy.deepcopy(state)), context(copy.deepcopy(state))
                # Exercise the frozen S1 effects as well as its frozen evaluator;
                # unrelated context APIs stay shared for identical pure input.
                old.effects = lambda c, a: self.old_effects.physical_effects(
                    c, a, old.mechanisms, **old.effect_params)
                _, before = old.score_joint(c_old, [], actions)
                _, after = new.score_joint(c_new, [], actions)
                for term, value in before.items():
                    self.assertAlmostEqual(after[term], value, places=10, msg=term)
                self.assertEqual(after['burst_opportunity'], 0.)
                self.assertEqual(after['burst_sources'], 0.)
                self.assertEqual(after['burst_unknown_sources'], 0.)


if __name__ == '__main__':
    unittest.main()
