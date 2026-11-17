"""Behavioral checks for shared next-shot allocation and conservative splash."""
import copy
import unittest

from skills.tactical_v1.context import Context, build_snapshot
from skills.tactical_v1.effects import physical_effects
from skills.tactical_v1.memory import TacticalMemory
from skills.tactical_v1.skills.combat_reposition import CombatRepositionSkill
from skills.tactical_v1.skills.focus_fire import FocusFireSkill
from skills.tactical_v1.skills.splash_fire import SplashFireSkill
from skills.tactical_v1.targeting import (allocation_progress, next_shot_target,
                                         shared_target_plan)
from skills.tactical_v1.telemetry import Telemetry
from .fixtures import enemy, scene, with_questions


def context(state):
    return Context(build_snapshot(with_questions(state), state["game_loop"]))


class ShortTargetPlanTests(unittest.TestCase):
    def pressure_scene(self):
        return scene([((0., 0.), "stalker")], {
            "e0": enemy((1., 0.), "zealot", health=150., max_health=150.),
            "e9": enemy((5., 0.), "colossus", health=350., max_health=350.)},
            attack_range=6.)

    def test_high_health_high_pressure_target_is_valuable_before_lethal(self):
        c, memory = context(self.pressure_scene()), TacticalMemory()
        assignment = shared_target_plan(c, memory).assignments["u0"]
        self.assertEqual(assignment.target, "e9")
        rear = physical_effects(c, {"u0": 15})
        front = physical_effects(c, {"u0": 6})
        self.assertLess(rear.damage_known["e9"], c.snapshot.enemies["e9"].effective_health)
        self.assertGreater(allocation_progress(c, {"u0": 15}, rear, memory), .5)
        self.assertEqual(allocation_progress(c, {"u0": 6}, front, memory), 0.)

    def test_type_name_does_not_override_actual_weapon_pressure(self):
        state = self.pressure_scene()
        state["unit_type_profiles"]["73"]["weapons"][0]["base_damage"] = 60.
        state["unit_type_profiles"]["4"]["weapons"][0]["base_damage"] = 1.
        self.assertEqual(shared_target_plan(context(state), TacticalMemory())
                         .assignments["u0"].target, "e0")

    def test_immediate_kill_remains_a_candidate_and_has_no_duplicate_pressure_reward(self):
        state = self.pressure_scene()
        state["visible_enemies"]["e0"]["health"] = 1.
        c, memory = context(state), TacticalMemory()
        self.assertEqual(shared_target_plan(c, memory).assignments["u0"].target, "e0")
        plans = FocusFireSkill({"search": {"max_candidates_per_unit": 1}}).propose(
            c, memory, Telemetry())
        self.assertEqual(plans[0].goal, "e0")
        report = physical_effects(c, {"u0": 6})
        self.assertEqual(allocation_progress(c, {"u0": 6}, report, memory), 0.)

    def test_unique_sources_do_not_reserve_overkill_on_the_same_small_target(self):
        state = scene([((0., 0.), "stalker"), ((0., 1.), "stalker")], {
            "e0": enemy((3., 0.), health=10.),
            "e1": enemy((3., 1.), health=10.)}, attack_range=6.)
        plan = shared_target_plan(context(state), TacticalMemory())
        self.assertEqual({p.target for p in plan.assignments.values()}, {"e0", "e1"})

    def test_queries_are_cached_order_independent_and_do_not_commit_memory(self):
        c, memory = context(self.pressure_scene()), TacticalMemory()
        before = copy.deepcopy(memory.__dict__)
        first = shared_target_plan(c, memory)
        FocusFireSkill({}).propose(c, memory, Telemetry())
        CombatRepositionSkill({}).propose(c, memory, Telemetry())
        self.assertIs(first, shared_target_plan(c, memory))
        self.assertEqual(before, memory.__dict__)
        self.assertFalse(shared_target_plan(c, memory, {
            "focus_fire": {"target_planning_enabled": False}}).assignments)

    def test_cooling_or_unknown_shots_do_not_earn_progress(self):
        state = self.pressure_scene()
        state["allies"]["u0"]["weapon_cooldown_loops"] = 12.
        c, memory = context(state), TacticalMemory()
        self.assertFalse(shared_target_plan(c, memory).assignments["u0"].immediate)
        report = physical_effects(c, {"u0": 15})
        self.assertEqual(allocation_progress(c, {"u0": 15}, report, memory), 0.)
        state["allies"]["u0"]["weapon_cooldown_loops"] = 0.
        del state["unit_type_profiles"]["74"]
        c = context(state)
        report = physical_effects(c, {"u0": 15})
        self.assertEqual(allocation_progress(c, {"u0": 15}, report, memory), 0.)

    def test_distant_target_does_not_create_a_chase_allocation(self):
        state = scene([((0., 0.), "stalker")], {
            "e0": enemy((3., 0.), "zealot", health=150.),
            "e9": enemy((30., 0.), "colossus", health=350.)}, attack_range=6.)
        self.assertEqual(shared_target_plan(context(state), TacticalMemory())
                         .assignments["u0"].target, "e0")

    def test_future_source_cannot_reserve_current_shooters_only_ready_target(self):
        state = scene([((12.5, 0.), "stalker"), ((0., 0.), "stalker")], {
            "e0": enemy((5.5, 0.), "stalker", health=13., max_health=150.),
            "e1": enemy((-6.7, 0.), "stalker", health=150., max_health=150.)},
            attack_range=6.)
        state["allies"]["u1"]["weapon_cooldown_loops"] = 8.
        c = context(state)
        self.assertFalse(c.can_issue_attack("u0", "e0"))
        self.assertTrue(c.can_issue_attack("u1", "e0"))
        self.assertEqual(c.fire_estimate("u1", "e0").window_shots, 1)
        self.assertFalse(c.can_issue_attack("u1", "e1"))
        allocation = shared_target_plan(c, TacticalMemory()).assignments
        self.assertEqual(allocation["u1"].target, "e0")
        self.assertTrue(allocation["u1"].immediate)
        self.assertFalse(allocation["u0"].immediate)

    def test_actor_with_current_shot_does_not_reserve_a_future_kill_first(self):
        state = scene([((0., 0.), "stalker")], {
            "e0": enemy((1., 0.), "stalker", health=150., max_health=150.),
            "e1": enemy((6.7, 0.), "stalker", health=13., max_health=150.)},
            attack_range=6.)
        c = context(state)
        self.assertFalse(c.can_issue_attack("u0", "e1"))
        future = c.projected_attack_opportunity(
            "u0", "e1", c.predict_position("u0", 4).position)
        self.assertTrue(future.available)
        self.assertGreaterEqual(future.damage_est, c.snapshot.enemies["e1"].effective_health)
        allocation = shared_target_plan(c, TacticalMemory()).assignments["u0"]
        self.assertEqual(allocation.target, "e0")
        self.assertTrue(allocation.immediate)

    def test_g3_keeps_four_directions_and_hands_selected_alternative_to_g1(self):
        state = self.pressure_scene()
        state["allies"]["u0"]["weapon_cooldown_loops"] = 12.
        memory = TacticalMemory()
        memory.attack_history["u0"] = {"target": "e0", "tag": "ally", "last_loop": 80}
        c = context(state)
        skill = CombatRepositionSkill({})
        plans = skill.propose(c, memory, Telemetry())
        self.assertLessEqual(len(plans), 6)
        self.assertEqual({p.current_actions["u0"] for p in plans if p.goal == "e0"},
                         {2, 3, 4, 5})
        alternative = next(p for p in plans if p.goal == "e9")
        self.assertEqual(next_shot_target(c, memory, "u0", {"e0", "e9"}), "e0")
        memory.commit(c.snapshot, alternative.current_actions, [alternative], context=c)
        deadline = memory.instance_for("combat_reposition", "u0").extra["next_shot_until_loop"]
        state["game_loop"] = 88
        state["allies"]["u0"]["weapon_cooldown_loops"] = 0.
        c = context(state)
        plans = FocusFireSkill({"search": {"max_candidates_per_unit": 1}}).propose(
            c, memory, Telemetry())
        self.assertEqual(plans[0].goal, "e9")
        self.assertEqual(memory.attack_history["u0"]["target"], "e0")
        state["game_loop"] = deadline
        self.assertEqual(next_shot_target(context(state), memory, "u0", {"e0", "e9"}), "e0")

    def test_unselected_or_replaced_actor_cannot_inherit_g3_intention(self):
        state = self.pressure_scene()
        state["allies"]["u0"]["weapon_cooldown_loops"] = 12.
        memory = TacticalMemory()
        memory.begin_step(context(state).snapshot)
        memory.attack_history["u0"] = {"target": "e0", "tag": "ally", "last_loop": 80}
        c = context(state)
        plan = next(p for p in CombatRepositionSkill({}).propose(c, memory, Telemetry())
                    if p.goal == "e9")
        memory.commit(c.snapshot, {"u0": 1}, [plan], context=c)
        self.assertIsNone(memory.instance_for("combat_reposition", "u0"))
        state["game_loop"] = 88
        c = context(state)
        memory.begin_step(c.snapshot)
        plan = next(p for p in CombatRepositionSkill({}).propose(c, memory, Telemetry())
                    if p.goal == "e9")
        memory.commit(c.snapshot, plan.current_actions, [plan], context=c)
        state["allies"]["u0"]["tag"] = "replacement"
        self.assertIsNone(next_shot_target(context(state), memory, "u0", {"e0", "e9"}))

    def test_continuing_g3_does_not_extend_next_shot_deadline(self):
        state = self.pressure_scene()
        state["allies"]["u0"]["weapon_cooldown_loops"] = 12.
        memory, skill = TacticalMemory(), CombatRepositionSkill({})
        c = context(state)
        memory.begin_step(c.snapshot)
        memory.attack_history["u0"] = {"target": "e0", "tag": "ally", "last_loop": 80}
        plan = next(p for p in skill.propose(c, memory, Telemetry()) if p.goal == "e9")
        memory.commit(c.snapshot, plan.current_actions, [plan], context=c)
        initial = memory.instance_for("combat_reposition", "u0")
        deadline = initial.extra["next_shot_until_loop"]
        state["game_loop"] = 88
        c = context(state)
        memory.begin_step(c.snapshot)
        plan = next(p for p in skill.propose(c, memory, Telemetry()) if p.goal == "e9")
        memory.commit(c.snapshot, plan.current_actions, [plan], context=c)
        continued = memory.instance_for("combat_reposition", "u0")
        self.assertEqual(continued.instance_id, initial.instance_id)
        self.assertEqual(continued.extra["next_shot_until_loop"], deadline)
        self.assertEqual(next_shot_target(c, memory, "u0", {"e0"}), "e0")


class ConservativeSplashTests(unittest.TestCase):
    def splash_context(self, cooldown=0.):
        state = scene([((0., 0.), "colossus")], {
            "e0": enemy((4., 0.), "stalker"),
            "e1": enemy((4., 1.), "stalker"),
            "e2": enemy((4., -1.), "stalker", is_flying=True)}, attack_range=7.)
        state["allies"]["u0"]["weapon_cooldown_loops"] = cooldown
        return context(state)

    def test_proxy_discount_changes_only_opportunity_not_primary_damage(self):
        c = self.splash_context()
        estimates, claims = [], []
        for reliability in (0., .5, 1.):
            provider = SplashFireSkill({"splash_fire": {"coverage_reliability": reliability}})
            estimates.append(provider.coverage_estimate(c, "u0", "e0"))
            claims.append(provider.impact_claims(c, "u0", "e0"))
            self.assertIsNone(provider.coverage_model()["calibration_id"])
        self.assertEqual([e.raw_opportunity for e in estimates], [.5, .5, .5])
        self.assertEqual([e.discounted_opportunity for e in estimates], [0., .25, .5])
        self.assertEqual([e.targets for e in estimates], [("e1",)] * 3)
        self.assertEqual(claims[0], claims[1])
        self.assertEqual(claims[1], claims[2])
        self.assertEqual([e.target for e in claims[0] if e.kind == "damage"], ["e0"])
        self.assertTrue(all(e.amount is None for e in claims[0] if e.kind == "coverage"))

    def test_current_and_future_coverage_keep_distinct_readiness(self):
        c, provider = self.splash_context(cooldown=12.), SplashFireSkill({})
        self.assertFalse(provider.coverage_estimate(c, "u0", "e0").qualified)
        future = provider.coverage_estimate(c, "u0", "e0", position=(0., 0.), horizon=16)
        self.assertTrue(future.qualified)
        self.assertGreater(future.discounted_opportunity, 0.)
        self.assertFalse(any(e.kind == "coverage" for e in provider.impact_claims(c, "u0", "e0")))


if __name__ == "__main__":
    unittest.main()
