"""Full-action coordination costs, bounded group search and truthful telemetry."""
import copy
import unittest

from skills.tactical_v1.context import Context, build_snapshot
from skills.tactical_v1.contracts import Commitment, ResourceClaims
from skills.tactical_v1.coordinator import Coordinator
from skills.tactical_v1.evaluator import Evaluator
from skills.tactical_v1.memory import TacticalMemory
from skills.tactical_v1.skill_manager import SkillManager
from skills.tactical_v1.skills.common import make_plan
from skills.tactical_v1.validator import validate_actions, validate_plans
from .fixtures import enemy, scene, with_questions


def context_for(state):
    return Context(build_snapshot(with_questions(state), state["game_loop"]))


def plan_for(context, actions, kind="none", instance_id=None, data=None):
    data = dict(data or {}, objective={"kind": kind})
    return make_plan(
        "precombat_rally" if kind == "rally" else "coordinated_engage",
        tuple(actions), "test-goal", actions, [],
        ResourceClaims(controls=tuple(actions)), Commitment("testing"), {},
        context.snapshot.snapshot_id, "test", instance_id=instance_id,
        atomic=len(actions) > 1, goal_kind="area", goal_data=data)


class JointCoordinationTests(unittest.TestCase):
    def test_escape_intent_does_not_duplicate_physical_risk_reduction(self):
        context = context_for(scene([((0., 0.), "marine")],
                                    {"e0": enemy((3., 0.))}))
        evaluator = Evaluator()
        actions = {"u0": 5}
        plain = evaluator.score_joint(context, [], actions)
        escape = evaluator.score_joint(context, [plan_for(context, actions, "escape")],
                                       actions)
        self.assertEqual(plain, escape)
        self.assertGreater(plain[1]["risk"], evaluator.score_joint(
            context, [], {"u0": 1})[1]["risk"])

    def test_local_cover_withdrawal_cost_uses_compatible_enemy_and_final_positions(self):
        state = scene([((0., 0.), "marine"), ((-5., 0.), "marine"),
                       ((50., 0.), "marine")], {"e0": enemy((2., 0.))})
        # Original geometric-link ablation. The peer at x=-5 cannot yet
        # shoot e0; default action-based cover is tested in joint_mechanisms.
        evaluator = Evaluator(common_combat={'joint_assessment_enabled': False})
        context = context_for(state)
        stationary = evaluator.score_joint(context, [], {"u0": 1, "u1": 1, "u2": 1})[1]
        retreat = evaluator.score_joint(context, [], {"u0": 1, "u1": 5, "u2": 1})[1]
        remote = evaluator.score_joint(context, [], {"u0": 1, "u1": 1, "u2": 5})[1]
        self.assertEqual(stationary["support_break"], 0.)
        self.assertLess(retreat["support_break"], 0.)
        self.assertEqual(remote["support_break"], 0.)
        state["relations"]["u1"]["e0"]["weapon_range_status"] = "no_compatible_weapon"
        unsupported = evaluator.score_joint(context_for(state), [],
                                           {"u0": 1, "u1": 5, "u2": 1})[1]
        self.assertEqual(unsupported["support_break"], 0.)

    def test_ready_attack_and_cooling_reentry_are_distinct_costs(self):
        state = scene([((0., 0.), "marine")], {"e0": enemy((5., 0.))}, distance=3.)
        state["allies"]["u0"]["orders"] = [{"kind": "attack", "target_unit": "e0"}]
        # Legacy ablation keeps the original split costs. The enabled common
        # budget is tested with real cooldown slack in test_firecycle_budget.
        evaluator = Evaluator(common_combat={'next_fire_budget_enabled': False})
        ready = evaluator.score_joint(context_for(state), [], {"u0": 5})[1]
        self.assertLess(ready["attack_continuity"], 0.)
        self.assertEqual(ready["contact_reentry"], 0.)
        state["allies"]["u0"].update(weapon_cooldown_loops=100., weapon_ready=False)
        cooling = evaluator.score_joint(context_for(state), [], {"u0": 5})[1]
        self.assertEqual(cooling["attack_continuity"], 0.)
        self.assertLess(cooling["contact_reentry"], 0.)
        # A critically injured unit may escape without protecting its next shot.
        state["allies"]["u0"].update(health=5., health_lost_since_previous_decision=10.)
        emergency = evaluator.score_joint(context_for(state), [], {"u0": 5})[1]
        self.assertEqual(emergency["contact_reentry"], 0.)

    def test_rally_rewards_group_readiness_and_prices_unproductive_wait(self):
        state = scene([((x, 0.), "stalker") for x in (0., 8., 16.)],
                      {"e0": enemy((30., 0.))})
        context = context_for(state)
        data = {"anchor": [8., 0.], "members": ["u0", "u1", "u2"],
                "ready_radius": 4., "ready_fraction": .8}
        move = {"u0": 4, "u1": 1, "u2": 5}
        hold = {"u0": 1, "u1": 1, "u2": 1}
        evaluator = Evaluator()
        gain = evaluator.score_joint(context, [plan_for(context, move, "rally", data=data)], move)[1]
        wait = evaluator.score_joint(context, [plan_for(context, hold, "rally", data=data)], hold)[1]
        self.assertGreater(gain["link"], 0.)
        self.assertEqual(gain["progress"], 0.)  # no duplicate per-unit point bonus
        self.assertEqual(wait["link"], 0.)
        self.assertLess(wait["rally_wait"], 0.)
        self.assertEqual(wait["hold_value"], 0.)

    def test_two_evaluations_still_compare_complete_atomic_rally(self):
        manager = SkillManager()
        manager.coordinator.max_evaluations = 2
        state = scene([((x, 0.), "stalker") for x in (0., 8., 16.)],
                      {"e0": enemy((30., 0.))})
        actions, summary = manager.step(with_questions(state), 3)
        self.assertEqual(actions, [4, 1, 5])
        self.assertEqual(summary["plan_evaluations_used"], 2)
        self.assertEqual({row["skill"] for row in summary["units"].values()},
                         {"precombat_rally"})
        self.assertEqual(len({row["plan"] for row in summary["units"].values()}), 1)

    def test_minimum_commitment_is_short_soft_and_interruptible(self):
        state = scene([((0., 0.), "marine")], {"e0": enemy((20., 0.))}, loop=88)
        memory = TacticalMemory()
        instance = memory.create_instance(
            "coordinated_engage", ("u0",), "e0", "approach", 80, 64,
            goal_kind="enemy", extra={"min_loops": 16,
                "interrupt_on": ("contact", "danger_change", "illegal"),
                "initial_threats": {"u0": []}})
        memory.last_committed_actions = {"u0": 4}
        evaluator = Evaluator()
        context = context_for(state)
        actions = {"u0": 4}
        abandon = evaluator.score_joint(context, [], actions, memory)[1]
        continued = evaluator.score_joint(context,
            [plan_for(context, actions, instance_id=instance.instance_id)], actions, memory)[1]
        self.assertLess(abandon["commitment_switch"], 0.)
        self.assertEqual(continued["commitment_switch"], 0.)
        state["game_loop"] = 96
        expired = evaluator.score_joint(context_for(state), [], actions, memory)[1]
        self.assertEqual(expired["commitment_switch"], 0.)
        state["game_loop"] = 88
        state["allies"]["u0"]["shield_lost_since_previous_decision"] = 1.
        danger = evaluator.score_joint(context_for(state), [], actions, memory)[1]
        self.assertEqual(danger["commitment_switch"], 0.)
        state["allies"]["u0"]["shield_lost_since_previous_decision"] = 0.
        memory.last_committed_actions["u0"] = 999
        illegal = evaluator.score_joint(context_for(state), [], actions, memory)[1]
        self.assertEqual(illegal["commitment_switch"], 0.)
        memory.last_committed_actions["u0"] = 4
        state = scene([((0., 0.), "marine")], {"e0": enemy((3., 0.))}, loop=88)
        contact = evaluator.score_joint(context_for(state), [], actions, memory)[1]
        self.assertEqual(contact["commitment_switch"], 0.)
        state = scene([((0., 0.), "marine")], loop=88)
        lost = evaluator.score_joint(context_for(state), [], actions, memory)[1]
        self.assertEqual(lost["commitment_switch"], 0.)

    def test_reported_alternatives_are_ranked_complete_valid_and_reproducible(self):
        state = scene([((x, 0.), "stalker") for x in (0., 8., 16.)],
                      {"e0": enemy((30., 0.))})
        manager = SkillManager()
        actions, summary = manager.step(with_questions(state), 3)
        alternatives = summary["top_alternates"]
        self.assertTrue(alternatives)
        self.assertEqual([a["score"] for a in alternatives],
                         sorted((a["score"] for a in alternatives), reverse=True))
        context = context_for(state)
        for alternative in alternatives:
            self.assertEqual(set(alternative["actions"]), {"u0", "u1", "u2"})
            values = [alternative["actions"][f"u{i}"] for i in range(3)]
            self.assertNotEqual(values, actions)
            self.assertEqual(validate_actions(values, context, 3), [])
            entries = [e for e in manager.coordinator._cache.values()
                       if e.actions == alternative["actions"]
                       and round(e.score, 4) == alternative["score"]]
            self.assertTrue(entries)
            entry = entries[0]
            self.assertEqual(validate_plans(entry.plans, context, entry.actions,
                                           manager.evaluator.effects(context, entry.actions)), [])
            score, breakdown = manager.evaluator.score_joint(
                context, entry.plans, entry.actions, manager.memory)
            self.assertEqual(round(score, 4), alternative["score"])
            self.assertEqual({k: round(v, 4) for k, v in breakdown.items()},
                             alternative["score_breakdown"])
        self.assertTrue(all(p["score"] == summary["joint_score"]
                            for p in summary["chosen_plans"]))

    def test_score_cache_distinguishes_retaining_a_live_commitment(self):
        context = context_for(scene([((0., 0.), "marine")],
                                    {"e0": enemy((20., 0.))}, loop=88))
        memory = TacticalMemory()
        instance = memory.create_instance("coordinated_engage", ("u0",),
            "e0", "approach", 80, 64, goal_kind="enemy", extra={"min_loops": 16})
        coordinator = Coordinator({})
        actions, budget = {"u0": 4}, [2]
        retained = coordinator.evaluate_once(context,
            [plan_for(context, actions, instance_id=instance.instance_id)],
            actions, memory, budget, 1)
        abandoned = coordinator.evaluate_once(context, [plan_for(context, actions)],
                                               actions, memory, budget, 1)
        self.assertIsNotNone(retained)
        self.assertIsNotNone(abandoned)
        self.assertGreater(retained.score, abandoned.score)
        self.assertEqual(budget[0], 0)

    def test_lifecycle_summary_counts_only_completed_commits(self):
        manager = SkillManager()
        state = scene([((x, 0.), "stalker") for x in (0., 8., 16.)],
                      {"e0": enemy((30., 0.))})
        actions, first = manager.step(with_questions(state), 3)
        self.assertEqual(first["lifecycle_counts"], {})
        self.assertIsNone(first["lifecycle_last_committed_loop"])
        manager.commit_final_actions(actions)
        state["game_loop"] = 88
        actions, second = manager.step(with_questions(copy.deepcopy(state)), 3)
        self.assertEqual(second["lifecycle_counts"]["precombat_rally"]["created"], 1)
        self.assertEqual(second["skill_usage"]["precombat_rally"]["continued"], 0)
        self.assertEqual(second["lifecycle_last_committed_loop"], 80)
        manager.commit_final_actions(actions)
        state["game_loop"] = 96
        _, third = manager.step(with_questions(state), 3)
        self.assertEqual(third["skill_usage"]["precombat_rally"]["continued"], 1)
        self.assertIn("current selection pending commit", third["lifecycle_semantics"])


if __name__ == "__main__":
    unittest.main()
