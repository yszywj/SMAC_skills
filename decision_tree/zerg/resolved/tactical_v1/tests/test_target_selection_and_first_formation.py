"""Targeted regression: normal target selection and hydralisk first formation.

Scenario ① — a hydralisk that can shoot BOTH a full-health zergling and an
actively threatening enemy hydralisk must allocate the volley to the
threat; an adjacent one-shot-finishable wounded zergling is still finished.
Scenario ② — the front hydra with a ready shot keeps firing while the rear
hydra actually moves into the shared firing position (end-to-end).
Scenario ③ — the Zerg profile advances on a nearby enemy hydralisk when
there is no ready legal shot, even across the legacy formation's hold line.
"""
import copy
import math
import unittest

from skills.tactical_v1.context import Context, build_snapshot
from skills.tactical_v1.memory import TacticalMemory
from skills.tactical_v1.targeting import (DAMAGE_SCALE, avoided_output,
                                          shared_target_plan)
from skills.tactical_v1.tests.fixtures import (TacticalTestCase, ally, enemy,
                                               scene, with_questions)


def period_profiles():
    """Fixture catalogs WITH attack periods, so the timing term is active."""
    return {
        "105": {"known": True, "movement_speed_native": 2.25, "armor": 0.,
                 "attributes": [1], "weapons": [
                     {"target_type": "ground", "range": 2., "base_damage": 5.,
                      "hits_per_attack": 1, "damage_bonus": [],
                      "attack_period_loops": 8}],
                 "attack_upgrade_level": 0., "armor_upgrade_level": 0.},
        "107": {"known": True, "movement_speed_native": 2.25, "armor": 0.,
                 "attributes": [1], "weapons": [
                     {"target_type": "ground", "range": 5., "base_damage": 12.,
                      "hits_per_attack": 1, "damage_bonus": [],
                      "attack_period_loops": 8}],
                 "attack_upgrade_level": 0., "armor_upgrade_level": 0.},
    }


def scored_scene(zergling_health=35.):
    """Hydra u0 can shoot BOTH targets: a zergling at 4.0 and an enemy
    hydralisk at 3.5 whose fire covers the wounded friendly hydra u1."""
    state = scene([((0., 0.), "hydralisk"), ((2., 0.), "hydralisk")],
                  enemies={"e0": enemy((4., 1.5), "zergling",
                                       health=zergling_health, max_health=35.),
                           "e1": enemy((3., -3.), "hydralisk")})
    state["unit_type_profiles"].update(period_profiles())
    # the enemy hydralisk covers u1 (threat evidence for pressure)
    state["relations"]["u1"]["e1"]["in_enemy_weapon_range"] = True
    state["relations"]["u1"]["e1"]["enemy_weapon_range_status"] = "in_range"
    state["allies"]["u1"].update(health=30., max_health=90.)
    return state


class TargetSelectionTest(TacticalTestCase):

    def _assign(self, state):
        context = Context(build_snapshot(with_questions(copy.deepcopy(state)), 1))
        context.unit_value_parameters = {"enabled": False}
        plan = shared_target_plan(context, TacticalMemory())
        return plan.assignments["u0"].target, context

    def test_1a_threat_hydra_beats_full_health_zergling(self):
        target, context = self._assign(scored_scene())
        self.assertEqual(context.contact_status("u0", "e0"), "in_range")
        self.assertEqual(context.contact_status("u0", "e1"), "in_range")
        self.assertEqual(target, "e1",
                         "a full-health zergling must not outrank an actively "
                         "threatening enemy hydralisk")

    def test_1b_adjacent_killable_zergling_is_still_finished(self):
        # One 12-point volley kills the 12 HP zergling: finish + kill credit
        # beat the multi-window timing value on the healthy hydralisk.
        target, _ = self._assign(scored_scene(zergling_health=12.))
        self.assertEqual(target, "e0",
                         "a real one-shot finish must still be taken")

    def test_1c_damage_scale_is_pool_independent(self):
        self.assertAlmostEqual(3. * 12. / DAMAGE_SCALE, 1.2, places=6)

    def test_expected_output_uses_both_weapon_cycles(self):
        state = scored_scene()
        state["visible_enemies"]["e1"]["health"] = 36.
        # Separate catalog rows so changing the opponent's attack period
        # does not also change the allied shooters' rate of fire.
        state["unit_type_profiles"]["5107"] = copy.deepcopy(state["unit_type_profiles"]["107"])
        for unit in state["allies"].values():
            unit["unit_type_id"] = 5107
        ours = state["unit_type_profiles"]["5107"]["weapons"][0]
        theirs = state["unit_type_profiles"]["107"]["weapons"][0]
        ours["attack_period_loops"] = 13.2
        def relief(shooters):
            context = Context(build_snapshot(with_questions(copy.deepcopy(state)), 1))
            return avoided_output(context, "e1", shooters)
        fast_enemy = relief(["u0"])
        self.assertGreater(fast_enemy, 0.)
        theirs["attack_period_loops"] *= 2
        self.assertAlmostEqual(relief(["u0"]) * 2, fast_enemy)
        ours["attack_period_loops"] = 20.
        self.assertEqual(relief(["u0"]), 0.)  # one ready shooter cannot land three volleys
        self.assertGreater(relief(["u0", "u1"]), 0.)


class FirstFormationTest(TacticalTestCase):

    def _front_state(self, rear_position=(3., 3.), crossfire=False):
        """Front u0 inside e0's line; e0 at 4.5. With ``crossfire`` the east
        step is additionally covered by two flankers that do NOT cover the
        origin, and the rear sits outside any next-window counter."""
        flankers = {"e1": enemy((10.2, 5.), "hydralisk"),
                    "e2": enemy((10.2, -5.), "hydralisk")} if crossfire else {}
        state = scene([((6., 0.), "hydralisk"), ((2., 0.), "hydralisk"),
                       (tuple(rear_position), "hydralisk")],
                      enemies={"e0": enemy((10.5, 0.), "hydralisk"), **flankers})
        for actor in ("u0", "u1", "u2"):
            state["allies"][actor].update(health=90., max_health=90.,
                                          weapon_cooldown_loops=0.)
            for eid in ("e0", "e1", "e2"):
                if eid in state["relations"][actor]:
                    state["relations"][actor][eid].update(
                        smac_attack_available=False, smac_attack_distance_limit=5,
                        in_weapon_range=False, weapon_range_status="out_of_range")
        state["relations"]["u0"]["e0"].update(
            in_enemy_weapon_range=True, enemy_weapon_range_status="in_range")
        return state

    def test_2_front_keeps_firing_rear_fills_end_to_end(self):
        state = self._front_state(rear_position=(7., 3.))
        # front has a READY, LEGAL shot on the threat e0 (in range + menu)
        state["relations"]["u0"]["e0"].update(
            in_weapon_range=True, weapon_range_status="in_range",
            smac_attack_available=True, smac_attack_distance_limit=5)
        manager = self.manager()
        actions, telemetry = self.step(manager, state)
        context = self._context(state)
        self.assertEqual(context.decode("u0", actions[0]).target, "e0",
                         "front ready shot was abandoned")
        self.assertEqual(context.decode("u2", actions[2]).kind, "move",
                         "rear did not reposition")
        point = context.predict_position("u2", actions[2]).position
        self.assertLessEqual(math.hypot(point[0] - 10.5, point[1]), 5.6,
                             "rear move did not reach the firing band")

    def test_3_zerg_advances_on_nearby_hydra_despite_crossfire(self):
        state = self._front_state(rear_position=(3., 3.), crossfire=True)
        manager = self.manager()
        actions, telemetry = self.step(manager, state)
        context = self._context(state)
        point = context.predict_position("u0", actions[0]).position
        self.assertGreater(point[0], state["allies"]["u0"]["position"][0])
        self.assertEqual(telemetry["units"]["u0"]["skill"], "focus_fire")
        self.assertEqual(telemetry["units"]["u0"]["phase"], "approach_hydralisk")

    def test_formation_hold_relief_matches_crossfire(self):
        from skills.tactical_v1.baseline import baseline_actions as compute_baseline
        from skills.tactical_v1.skills.ranged_fire_formation import (
            formation_hold_relief)
        manager = self.manager()
        state = self._front_state(rear_position=(3., 3.), crossfire=True)
        context = self._context(state)
        context.deployment_memory = manager.memory
        context.unit_value_parameters = {"enabled": False}
        context.intercept_config = {}
        context.common_config = {}
        context.burst_config = {}
        context.baseline_actions = compute_baseline(context, manager.memory,
                                                    manager.skills)
        self.assertTrue(formation_hold_relief(context, "u0"),
                        "crossfire-condemned front advance did not earn relief")
        # without the second flanker the single line keeps the normal rule
        plain = self._front_state(rear_position=(3., 3.), crossfire=False)
        plain_ctx = self._context(plain)
        plain_ctx.deployment_memory = manager.memory
        plain_ctx.unit_value_parameters = {"enabled": False}
        plain_ctx.intercept_config = {}
        plain_ctx.common_config = {}
        plain_ctx.burst_config = {}
        plain_ctx.baseline_actions = compute_baseline(
            plain_ctx, manager.memory, manager.skills)
        from skills.tactical_v1.skills.ranged_fire_formation import _forecast
        model = _forecast(plain_ctx)
        fresh = model.covering("u0", plain_ctx.baseline_actions["u0"])[0] \
            - model.current_threats["u0"]
        self.assertLessEqual(len(fresh), 1)

    def _context(self, state):
        return Context(build_snapshot(with_questions(copy.deepcopy(state)), 1))
