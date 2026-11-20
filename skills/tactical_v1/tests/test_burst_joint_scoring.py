"""Joint burst accounting, independent of pending native effect calibration."""
import copy
import json
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from skills.tactical_v1.context import Context, build_snapshot
from skills.tactical_v1.effects import EffectsReport, physical_effects
from skills.tactical_v1.effects import EffectsReport, physical_effects
from skills.tactical_v1.evaluator import Evaluator
from skills.tactical_v1.targeting import DAMAGE_SCALE
from .fixtures import scene, enemy, with_questions


def context(state):
    return Context(build_snapshot(with_questions(state), state['game_loop']))


class BurstJointScoringTests(unittest.TestCase):
    """Mixed gun/burst settlement on the shared per-target pool.

    Reports come from REAL physical_effects on a calibrated scene: a marine
    volley and adjacent baneling contacts against the same zealot. Values
    are stated through the fixed DAMAGE_SCALE so this test documents the
    scale itself instead of pinning the old per-target health pools.
    """

    def setUp(self):
        contract = json.loads((Path(__file__).resolve().parents[3]
                               / 'skills/tactical_v1/burst_native_profile.json')
                              .read_text(encoding='utf8'))['catalog_contract']

        def build(e0_health=100., calibrate=True):
            state = scene([((0., 0.), 'marine'), ((3.3, 0.), 'baneling'),
                           ((3.3, 2.5), 'baneling')],
                enemies={'e0': enemy((4., 0.), health=e0_health,
                                     max_health=100.),
                         'e1': enemy((4., 3.), health=50., max_health=50.)})
            if calibrate:
                state['unit_type_profiles']['9'] = dict(contract)
            return state

        self.build = build
        self.state = build()
        self.context = context(self.state)
        self.evaluator = Evaluator()

    def score(self, report, risk=None, unsupported=None):
        """Score a hand-built report with the physical pipeline patched out."""
        original = copy.deepcopy(report)
        pressure = {'pressure_relief': 0., 'unsupported_actors': unsupported or {},
                    'unknown_enemy_pairs': 0}
        with patch.object(self.evaluator, 'effects', return_value=report),              patch.object(self.evaluator, '_forecast_costs', return_value=(risk or {}, 0)),              patch.object(self.evaluator, 'task_quality', return_value=(0., {})),              patch.object(self.evaluator, '_coordination_costs', return_value={}),              patch.object(self.evaluator, '_future_attack_value', return_value=0.),              patch.object(self.evaluator, '_sustained_contact_value', return_value=0.),              patch('skills.tactical_v1.battle_pressure.joint_pressure', return_value=pressure),              patch('skills.tactical_v1.targeting.allocation_progress', return_value=0.):
            result = self.evaluator.score_joint(self.context, [], {})[1]
        self.assertEqual(report, original,
                         'score queries must not consume or rewrite the report')
        return result

    def score_real(self, state, actions, risk=None, unsupported=None):
        report = physical_effects(
            Context(build_snapshot(with_questions(copy.deepcopy(state)),
                                   state['game_loop'])), actions)
        original = copy.deepcopy(report)
        pressure = {'pressure_relief': 0., 'unsupported_actors': unsupported or {},
                    'unknown_enemy_pairs': 0}
        with patch.object(self.evaluator, 'effects', return_value=report),              patch.object(self.evaluator, '_forecast_costs', return_value=(risk or {}, 0)),              patch.object(self.evaluator, 'task_quality', return_value=(0., {})),              patch.object(self.evaluator, '_coordination_costs', return_value={}),              patch.object(self.evaluator, '_future_attack_value', return_value=0.),              patch.object(self.evaluator, '_sustained_contact_value', return_value=0.),              patch('skills.tactical_v1.battle_pressure.joint_pressure', return_value=pressure),              patch('skills.tactical_v1.targeting.allocation_progress', return_value=0.):
            result = self.evaluator.score_joint(
                Context(build_snapshot(with_questions(copy.deepcopy(state)),
                                       state['game_loop'])), [], actions)[1]
        self.assertEqual(report, original,
                         'score queries must not consume or rewrite the report')
        self.last_report = report
        return result


    def last_report_bounded_pool(self, target, e0_health=100.):
        pools = {'e0': e0_health, 'e1': 50.}
        return pools.get(target, 100.)

    def test_ordinary_and_burst_share_each_targets_remaining_ehp(self):
        # 6 gun + 35 burst land in the same window on a 100 HP zealot:
        # both are valued on the fixed scale, proportional to their share
        # of the pooled 41 damage.
        row = self.score_real(self.build(),
                              {'u0': 6, 'u1': 6})
        self.assertAlmostEqual(row['useful_damage'], 3. * 6. / DAMAGE_SCALE, places=6)
        self.assertAlmostEqual(row['burst_opportunity'], 3. * 35. / DAMAGE_SCALE, places=6)
        self.assertAlmostEqual(row['useful_damage'] + row['burst_opportunity'],
                               3. * 41. / DAMAGE_SCALE, places=6)
        self.assertEqual(row['expected_kills'], 0.,
                         'a potential burst is not a known volley')

    def test_ordinary_kill_leaves_no_reserve_for_burst_and_no_extra_kill(self):
        # A 5 HP target: the marine volley alone is lethal, the whole pool
        # is capped at 5 and the unused baneling earns nothing extra.
        row = self.score_real(self.build(e0_health=5.), {'u0': 6})
        self.assertAlmostEqual(row['useful_damage'], 3. * 5. / DAMAGE_SCALE, places=6)
        self.assertEqual(row['expected_kills'], 2.)
        self.assertEqual(row['burst_opportunity'], 0.)

    def test_multiple_bursts_share_one_budget_per_target_without_cross_target_loss(self):
        # Expectations are derived from the REAL report: each target's pool
        # caps its own bucket on the fixed scale; one saturated target
        # cannot absorb the second burst in full, while a distinct target
        # keeps its independent budget.
        actions = {'u1': 6, 'u2': 7}
        saturated = self.score_real(self.build(e0_health=50.), actions)
        saturated_expected = 3. * sum(
            min(dmg, self.last_report_bounded_pool(target, e0_health=50.))
            for target, dmg in self.last_report.burst_damage_potential.items()
        ) / DAMAGE_SCALE
        self.assertAlmostEqual(saturated['burst_opportunity'], saturated_expected,
                               places=6)
        distinct = self.score_real(self.build(), actions)
        distinct_expected = 3. * sum(
            min(dmg, self.last_report_bounded_pool(target))
            for target, dmg in self.last_report.burst_damage_potential.items()
        ) / DAMAGE_SCALE
        self.assertAlmostEqual(distinct['burst_opportunity'], distinct_expected,
                               places=6)
        self.assertGreater(distinct_expected, saturated_expected)

    def test_unknown_burst_preserves_known_partial_credit_without_inventing_damage_or_kill(self):
        # Calibrated geometry with an upgraded baneling: the explosion is
        # UNKNOWN (unsupported shield/upgrade combination) — the known
        # marine volley keeps its credit, no burst value or kill appears.
        state = self.build()
        state['allies']['u1']['attack_upgrade_level'] = 1
        row = self.score_real(state, {'u0': 6, 'u1': 6})
        self.assertAlmostEqual(row['useful_damage'], 3. * 6. / DAMAGE_SCALE, places=6)
        self.assertEqual(row['burst_opportunity'], 0.)
        self.assertEqual(row['expected_kills'], 0.)
        self.assertGreaterEqual(row['burst_unknown_sources'], 1.)

    def test_planned_consumption_and_forecast_risk_use_max_per_actor(self):
        report = EffectsReport(planned_loss={'u1': 1.})
        for risk, expected in ((.4, 1.5), (2., 2.)):
            row = self.score(report, {'u1': risk})
            self.assertAlmostEqual(-row['risk'] - row['planned_loss'], expected)
            self.assertEqual(row['planned_loss_raw'], -1.5)
        separate = self.score(report, {'u0': 2.})
        self.assertEqual(-separate['risk'] - separate['planned_loss'], 3.5,
                         "another actor cannot cancel this source's self-consumption")
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
                         "consuming one source cannot pay for another actor's exposure")


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
                # The unified damage scale is an intentional all-race
                # change: everything EXCEPT the damage-pool-scaled terms
                # must stay frozen against the pre-change snapshot.
                scale_terms = {'useful_damage', 'burst_opportunity', 'total'}
                for term, value in before.items():
                    if term in scale_terms:
                        continue
                    self.assertAlmostEqual(after[term], value, places=10, msg=term)
                self.assertEqual(after['burst_opportunity'], 0.)
                self.assertEqual(after['burst_sources'], 0.)
                self.assertEqual(after['burst_unknown_sources'], 0.)


if __name__ == '__main__':
    unittest.main()
