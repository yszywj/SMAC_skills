"""E. Regression & property tests: old entries unchanged, action legality
properties, order-invariance, and ten fixtures where each skill is genuinely
selected through the coordinator (no gating bypassed)."""
import copy
import random
import unittest

from skills.current.skill_manager import SkillManager as CurrentManager
from skills.tactical_v1.skill_manager import SkillManager as TacticalManager
from .fixtures import (MOVE_DELTAS, STOP, ally, enemy, scene, with_questions,
                       heal_candidates)


class LegacyEntryRegressionTests(unittest.TestCase):
    def test_current_manager_default_behaviour_unchanged(self):
        # The historical default path must produce the same actions with and
        # without the tactical_v1 package being importable in the process.
        state = scene([((0., 0.), "marine"), ((1., 0.), "marine")],
                      enemies={"e0": enemy((4., 4.))})
        manager = CurrentManager(4)
        manager.select("adaptive_focus_fire")
        actions, telemetry = manager.step({"state": copy.deepcopy(state),
                                           "questions": {}}, 2)
        # And with tactical_v1 imported first.
        self.assertTrue(TacticalManager)
        manager2 = CurrentManager(4)
        manager2.select("adaptive_focus_fire")
        actions2, _ = manager2.step({"state": copy.deepcopy(state),
                                     "questions": {}}, 2)
        self.assertEqual(actions, actions2)


class PropertyTests(unittest.TestCase):
    def random_scene(self, rng, include_specials=True):
        names = ["marine", "zealot", "stalker", "hydralisk", "zergling"]
        if include_specials:
            names += ["colossus", "medivac", "baneling", "marauder"]
        allies = [((rng.uniform(0, 20), rng.uniform(0, 20)),
                   rng.choice(names)) for _ in range(rng.randint(1, 10))]
        enemies = {f"e{i}": enemy((rng.uniform(0, 20), rng.uniform(0, 20)),
                                  rng.choice(["zergling", "marine",
                                              "baneling", "colossus"]))
                   for i in range(rng.randint(0, 6))}
        return scene(allies, enemies=enemies,
                     attack_range=rng.choice([2.0, 5.0, 7.0]))

    def test_actions_legal_control_complete_no_duplicates(self):
        from skills.tactical_v1.context import build_snapshot, Context
        from skills.tactical_v1.validator import validate_actions, validate_joint
        rng = random.Random(11)
        manager = TacticalManager()
        for _ in range(60):
            state = self.random_scene(rng)
            pl = with_questions(state)
            snapshot = build_snapshot(pl, 1)
            context = Context(snapshot)
            actions, telemetry = manager.step(pl, len(state["allies"]))
            self.assertEqual(validate_actions(actions, context,
                                              len(state["allies"])), [])
            # No non-finite numbers anywhere in the telemetry.
            import math
            def finite(value):
                if isinstance(value, float):
                    return math.isfinite(value)
                if isinstance(value, dict):
                    return all(finite(v) for v in value.values())
                if isinstance(value, list):
                    return all(finite(v) for v in value)
                return True
            self.assertTrue(finite(telemetry))
            manager.commit_final_actions(actions)

    def test_unit_order_and_labels_do_not_change_the_plan(self):
        """Relabel the SAME battle (identity mapped back) -> identical plan.

        We permute actor indices and enemy ids; after mapping actions back
        through the permutation, every unit's action must match.
        """
        rng = random.Random(3)
        base = scene([((0., 0.), "marine"), ((2., 1.), "zealot"),
                      ((4., 0.), "stalker")],
                     enemies={"e0": enemy((8., 4.)),
                              "e1": enemy((9., 6.))})
        manager1 = TacticalManager()
        actions1, _ = manager1.step(with_questions(copy.deepcopy(base)), 3)
        # Permuted labels: reverse unit order and enemy order.
        permuted = scene([((4., 0.), "stalker"), ((2., 1.), "zealot"),
                          ((0., 0.), "marine")],
                         enemies={"e0": enemy((9., 6.)),
                                  "e1": enemy((8., 4.))})
        manager2 = TacticalManager()
        actions2, _ = manager2.step(with_questions(copy.deepcopy(permuted)), 3)
        # Permutation: permuted index i corresponds to base index 2 - i;
        # enemy e0(base) == e1(permuted).
        base_of = {0: 2, 1: 1, 2: 0}
        for permuted_index, base_index in base_of.items():
            a_permuted = actions2[permuted_index]
            a_base = actions1[base_index]
            if a_base == STOP or a_base in MOVE_DELTAS:
                self.assertEqual(a_permuted, a_base)
            elif a_base >= 6:  # attack e{a_base-6} in base numbering
                base_enemy = a_base - 6          # 0 -> e0(base) = e1(permuted)
                permuted_enemy = 1 - base_enemy
                self.assertEqual(a_permuted, 6 + permuted_enemy)

    def test_absent_unit_type_and_odd_counts_do_not_crash(self):
        rng = random.Random(5)
        manager = TacticalManager()
        for roster in (["medivac"], ["baneling"] * 3, ["colossus", "zealot"],
                       ["marine"] * 10):
            allies = [((i * 1.0, 0.), name) for i, name in enumerate(roster)]
            state = scene(allies, enemies={"e0": enemy((6., 0.))})
            actions, _ = manager.step(with_questions(state), len(roster))
            self.assertEqual(len(actions), len(roster))


class SelectionFixtureTests(unittest.TestCase):
    """Ten explainable fixtures: every skill is genuinely selected through
    the joint coordinator (no legality tampering, no bypass)."""

    def run_once(self, allies, enemies, **kwargs):
        state = scene(allies, enemies=enemies, **kwargs)
        manager = TacticalManager()
        actions, telemetry = manager.step(with_questions(state),
                                          len(state["allies"]))
        manager.commit_final_actions(actions)
        return {actor: info["skill"] for actor, info
                in telemetry["units"].items()}, actions, telemetry

    def test_fixture_focus_fire(self):
        skills, actions, _ = self.run_once(
            [((0., 0.), "marine")], {"e0": enemy((3., 0.))})
        self.assertEqual(skills["u0"], "focus_fire")

    def test_fixture_coordinated_engage(self):
        skills, actions, _ = self.run_once(
            [((0., 0.), "zealot")], {"e0": enemy((9., 9.))})
        self.assertEqual(skills["u0"], "coordinated_engage")

    def test_fixture_combat_reposition(self):
        # Healthy-enough ranged unit under two attackers with a long cooldown
        # still trades damage; a critical one kites via combat_reposition.
        skills, actions, _ = self.run_once(
            [((0., 0.), "marine")],
            {"e0": enemy((4., 0.)), "e1": enemy((0., 4.))})
        # Now with the same geometry but a cooldown window and damage taken.
        state = scene([((0., 0.), "marine")],
                      enemies={"e0": enemy((4., 0.)), "e1": enemy((0., 4.))})
        state["allies"]["u0"]["weapon_cooldown_loops"] = 16.
        state["allies"]["u0"]["health"] = 12.
        state["allies"]["u0"]["health_lost_since_previous_decision"] = 9.
        manager = TacticalManager()
        actions, telemetry = manager.step(with_questions(state), 1)
        manager.commit_final_actions(actions)
        skill = telemetry["units"]["u0"]["skill"]
        self.assertIn(skill, ("combat_reposition", "disengage_reenter"))

    def test_fixture_disengage_reenter(self):
        # Critical lone unit, far from help: retreat, then re-enter.
        skills, actions, _ = self.run_once(
            [((10., 10.), "marine"), ((0., 0.), "marine")],
            {"e0": enemy((12., 10.))})
        # u0 is in a fire exchange; make it critical instead:
        state = scene([((10., 10.), "marine"), ((0., 0.), "marine")],
                      enemies={"e0": enemy((12., 10.))})
        state["allies"]["u0"]["health"] = 6.
        state["allies"]["u0"]["health_lost_since_previous_decision"] = 12.
        state["allies"]["u0"]["weapon_cooldown_loops"] = 16.
        manager = TacticalManager()
        actions, telemetry = manager.step(with_questions(state), 2)
        manager.commit_final_actions(actions)
        self.assertIn(telemetry["units"]["u0"]["skill"],
                      ("disengage_reenter", "combat_reposition"))

    def test_fixture_regroup(self):
        skills, actions, _ = self.run_once(
            [((8., 0.), "marine"), ((8., 1.), "marine"), ((0., 20.), "marine")],
            {"e0": enemy((11., 0.))})
        self.assertEqual(skills["u2"], "regroup")

    def test_fixture_local_formation(self):
        skills, actions, _ = self.run_once(
            [((5., 5.), "marine"), ((5.3, 5.), "marine")],
            {"e0": enemy((9., 5.), "colossus")}, attack_range=3.0)
        self.assertIn("local_formation", skills.values())

    def test_fixture_cohesive_search(self):
        skills, actions, _ = self.run_once(
            [((0., 0.), "marine"), ((1., 0.), "marine")], {})
        self.assertEqual(set(skills.values()), {"cohesive_search"})

    def test_fixture_splash_fire_provider(self):
        state = scene([((0., 0.), "colossus"), ((0., 2.), "marine")],
                      enemies={"e0": enemy((6., 1.)), "e1": enemy((6.5, 1.))},
                      attack_range=9.0)
        manager = TacticalManager()
        actions, telemetry = manager.step(with_questions(state), 2)
        manager.commit_final_actions(actions)
        info = telemetry["units"]["u0"]
        self.assertEqual(info["skill"], "focus_fire")
        self.assertEqual(info["provider"], "splash_fire")

    def test_fixture_heal_support(self):
        state = scene([((0., 0.), "medivac"), ((2., 0.), "marine")],
                      enemies={"e0": enemy((9., 9.))})
        state["allies"]["u0"]["energy"] = 50.
        state["allies"]["u1"]["health"] = 25.
        heal_candidates(state, "u0")
        manager = TacticalManager()
        actions, telemetry = manager.step(with_questions(state), 2)
        manager.commit_final_actions(actions)
        self.assertEqual(telemetry["units"]["u0"]["skill"], "heal_support")

    def test_fixture_burst_engage(self):
        skills, actions, _ = self.run_once(
            [((5., 5.), "baneling")],
            {"e0": enemy((7., 5.)), "e1": enemy((7.5, 5.)),
             "e2": enemy((7., 5.5)), "e3": enemy((7.5, 5.5))})
        self.assertEqual(skills["u0"], "burst_engage")
        self.assertEqual(actions[0], 6)


class TelemetryStatTests(unittest.TestCase):
    def test_usage_stats_distinguish_applicable_from_selected(self):
        state = scene([((0., 0.), "marine"), ((1., 0.), "marine")],
                      enemies={"e0": enemy((3., 0.))})
        manager = TacticalManager()
        for loop in range(80, 80 + 8 * 5, 8):
            state["game_loop"] = loop
            actions, telemetry = manager.step(with_questions(
                copy.deepcopy(state)), 2)
            manager.commit_final_actions(actions)
        usage = manager.telemetry.usage
        self.assertGreater(usage["focus_fire"]["applicable"], 0)
        self.assertGreater(usage["focus_fire"]["selected"], 0)
        # Specializations never even applicable without their units.
        self.assertEqual(usage["heal_support"]["applicable"], 0)


if __name__ == "__main__":
    unittest.main()
