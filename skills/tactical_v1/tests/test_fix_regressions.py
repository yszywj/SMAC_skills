"""Round-4 end-to-end regression (task section 7): five property groups over
the REAL implementation. These check generate -> validate -> full scoring ->
selection -> commit -> next-observation connections.
    python -m unittest skills.tactical_v1.tests.test_fix_regressions
"""
import copy
import unittest

from skills.tactical_v1.actions import KEEP_HEAL_ORDER
from skills.tactical_v1.context import build_snapshot, Context
from skills.tactical_v1.evaluator import Evaluator
from skills.tactical_v1.effects import physical_effects
from skills.tactical_v1.skill_manager import SkillManager
from skills.tactical_v1.telemetry import Telemetry
from .fixtures import (MOVE_DELTAS, STOP, enemy, heal_candidates, scene,
                       with_questions)


def make_context(state):
    return Context(build_snapshot(with_questions(state), 1))


# ---------------------------------------------------------------------------
# Group 1: atomic joint plans through the generic validator
# ---------------------------------------------------------------------------

class Group1AtomicPlans(unittest.TestCase):
    def test_s2_joint_meeting_candidate_generated_for_out_of_range_patient(self):
        # medivac far from a wounded marine (NOT in the legal heal menu):
        # S2 must generate joint MEET candidates — generic member semantics.
        manager = SkillManager()
        state = scene([((0., 0.), "medivac"), ((9., 0.), "marine")],
                      enemies={"e0": enemy((12., 12.))})
        state["allies"]["u0"]["energy"] = 50.
        state["allies"]["u1"]["health"] = 20.
        context = make_context(state)
        plans = manager.skills["heal_support"].propose(
            context, manager.memory, Telemetry())
        meets = [p for p in plans if p.atomic and p.goal_kind == "meeting"]
        self.assertTrue(meets, "joint meet candidates must be generated for "
                               "an out-of-range patient")
        actions, telemetry = manager.step(with_questions(state), 2)
        manager.commit_final_actions(actions)
        # the whole final plan validated (runner rejects otherwise)
        self.assertEqual(len(actions), 2)

    def test_member_action_changed_invalidates_whole_atomic(self):
        from skills.tactical_v1.contracts import (CandidatePlan, Commitment,
                                                  Dependency, ResourceClaims)
        from skills.tactical_v1.validator import validate_plans
        context = make_context(scene([((0., 0.), "medivac"),
                                      ((9., 0.), "marine")]))

        def joint(actions):
            return CandidatePlan(
                proposal_id="meet-1", snapshot_id=1, config_fingerprint="t",
                primary_skill="heal_support", specialization_provider=None,
                instance_id=None, controlled_actors=tuple(sorted(actions)),
                beneficiaries=("u1",), goal="u1", goal_kind="meeting",
                goal_data={"healer": "u0", "patient": "u1"},
                current_actions=dict(actions),
                expected_effects=[], claims=ResourceClaims(
                    controls=tuple(sorted(actions))),
                dependencies=(Dependency("member", tuple(sorted(actions))),),
                commitment=Commitment("meet", 0), features={},
                proposed_memory_delta={}, atomic=True)
        plans = [joint({"u0": 4, "u1": 5})]
        self.assertEqual(validate_plans(plans, context, {"u0": 4, "u1": 5}),
                         [])
        broken = validate_plans(plans, context, {"u0": 4, "u1": 1})
        self.assertTrue(any(p["type"] == "member_action_lost" and
                            p["drop_whole"] for p in broken))

    def test_joint_kill_keeps_fire_window_check(self):
        manager = SkillManager()
        state = scene([((0., 0.), "marine"), ((0., 2.), "marine"),
                       ((2., 0.), "marine")],
                      enemies={"e0": enemy((4., 0.), "baneling",
                                           health=10.)})
        context = make_context(state)
        plans = manager.skills["focus_fire"].propose(
            context, manager.memory, Telemetry())
        joint = [p for p in plans if p.atomic]
        self.assertTrue(joint)
        self.assertTrue(any(d.kind == "fire_window"
                            for d in joint[0].dependencies))


# ---------------------------------------------------------------------------
# Group 2: tactical value enters the final score
# ---------------------------------------------------------------------------

class Group2TacticalValue(unittest.TestCase):
    def test_converging_beats_diverging_for_support_link(self):
        from skills.tactical_v1.contracts import (CandidatePlan, Commitment,
                                                  ResourceClaims)
        # u0 at (0,0), u1 at (10,0): east (4) closes the link, west (5) opens
        state = scene([((0., 0.), "marine"), ((10., 0.), "marine")])
        context = make_context(state)
        goal_data = {"group_a": ["u0"], "group_b": ["u1"],
                     "support_radius": 8.0, "scale": 4.0}

        def plan(action):
            return CandidatePlan(
                proposal_id=f"regroup-{action}", snapshot_id=1,
                config_fingerprint="t", primary_skill="regroup",
                specialization_provider=None, instance_id=None,
                controlled_actors=("u0",), beneficiaries=(), goal="merge",
                goal_kind="group_link", goal_data=dict(goal_data),
                current_actions={"u0": action}, expected_effects=[],
                claims=ResourceClaims(controls=("u0",)),
                dependencies=(), commitment=Commitment("rallying", 0),
                features={}, proposed_memory_delta={})
        evaluator = Evaluator()
        toward, tb_t = evaluator.score_joint(context, [plan(4)],
                                             {"u0": 4, "u1": 1})
        away, tb_a = evaluator.score_joint(context, [plan(5)],
                                           {"u0": 5, "u1": 1})
        self.assertGreater(toward, away,
                           "closing the support link must outrank opening it")
        self.assertGreater(tb_t["link"], 0.0)
        self.assertLess(tb_a["link"], 0.0)

    def test_role_position_improvement_vs_worsening(self):
        from skills.tactical_v1.contracts import (CandidatePlan, Commitment,
                                                  ResourceClaims)
        state = scene([((6., 0.), "marine"), ((6., 1.), "marine"),
                       ((7., 0.), "colossus")],
                      enemies={"e0": enemy((12., 0.))}, attack_range=9.0)
        context = make_context(state)
        goal_data = {"role": "colossus_protected", "actor": "u2",
                     "guards": ["u0", "u1"]}

        def plan(action):
            return CandidatePlan(
                proposal_id=f"role-{action}", snapshot_id=1,
                config_fingerprint="t", primary_skill="local_formation",
                specialization_provider=None, instance_id=None,
                controlled_actors=("u2",), beneficiaries=(), goal="formation",
                goal_kind="role", goal_data=dict(goal_data),
                current_actions={"u2": action}, expected_effects=[],
                claims=ResourceClaims(controls=("u2",)),
                dependencies=(), commitment=Commitment("adjusting", 0),
                features={}, proposed_memory_delta={})
        evaluator = Evaluator()
        back, tb_b = evaluator.score_joint(context, [plan(5)],
                                           {"u0": 1, "u1": 1, "u2": 5})
        forward, tb_f = evaluator.score_joint(context, [plan(4)],
                                              {"u0": 1, "u1": 1, "u2": 4})
        self.assertGreater(tb_b["role"], tb_f["role"],
                           "behind the friendly line beats past it")

    def test_meeting_gives_future_opportunity_not_present_heal(self):
        state = scene([((0., 0.), "medivac"), ((9., 0.), "marine")],
                      enemies={"e0": enemy((20., 20.))})
        state["allies"]["u1"]["health"] = 20.
        context = make_context(state)
        report = physical_effects(context, {"u0": 4, "u1": 5})
        self.assertNotIn("u1", report.heal,
                         "no heal opportunity before the pair is in range")
        from skills.tactical_v1.contracts import (CandidatePlan, Commitment,
                                                  ResourceClaims)
        meet = CandidatePlan(
            proposal_id="meet", snapshot_id=1, config_fingerprint="t",
            primary_skill="heal_support", specialization_provider=None,
            instance_id=None, controlled_actors=("u0", "u1"),
            beneficiaries=("u1",), goal="u1", goal_kind="meeting",
            goal_data={"healer": "u0", "patient": "u1"},
            current_actions={"u0": 4, "u1": 5}, expected_effects=[],
            claims=ResourceClaims(controls=("u0", "u1")),
            dependencies=(), commitment=Commitment("meet", 0), features={},
            proposed_memory_delta={}, atomic=True)
        evaluator = Evaluator()
        score, breakdown = evaluator.score_joint(
            context, [meet], {"u0": 4, "u1": 5})
        self.assertGreater(breakdown["meeting"], 0.0,
                           "closing the pair earns a future opportunity")
        self.assertEqual(breakdown["heal"], 0.0,
                         "the meet must not fabricate present-window healing")


# ---------------------------------------------------------------------------
# Group 3: lifecycle through the manager
# ---------------------------------------------------------------------------

class Group3Lifecycle(unittest.TestCase):
    def test_arrival_then_new_waypoint(self):
        manager = SkillManager()
        state = scene([((0., 0.), "marine")], search=(4., 0.))
        for step in range(3):
            state["game_loop"] = 80 + step * 8
            actions, _ = manager.step(with_questions(copy.deepcopy(state)), 1)
            manager.commit_final_actions(actions)
            self.assertEqual(manager.memory.visited_search_cells, {})
        state["allies"]["u0"]["position"] = [4., 0.]
        state["game_loop"] = 104
        actions, _ = manager.step(with_questions(copy.deepcopy(state)), 1)
        manager.commit_final_actions(actions)
        self.assertTrue(manager.memory.visited_search_cells)

    def test_timeout_records_unreachable_and_retry(self):
        manager = SkillManager()
        # waypoint far away and the unit can only Stop (no legal moves)
        state = scene([((0., 0.), "marine")], search=(20., 0.))
        state["movement_candidates"]["u0"] = {}
        saw_record = False
        for step in range(16):
            state["game_loop"] = 80 + step * 8
            pl = with_questions(copy.deepcopy(state))
            pl["questions"]["u0"]["criteria"] = {"1": "stop"}
            actions, _ = manager.step(pl, 1)
            manager.commit_final_actions(actions)
            if manager.memory.retry_records:
                saw_record = True
                break
        self.assertTrue(saw_record,
                        "a search that cannot progress must terminate with a "
                        "retry record, not restart silently")
        if manager.memory.retry_records:
            key = next(iter(manager.memory.retry_records))
            self.assertEqual(key[0], "cohesive_search")
            self.assertTrue(manager.memory.retry_blocked(
                key[0], key[1].split("+"), key[2],
                manager.memory.last_game_loop))

    def test_effect_danger_not_safe_for_g4(self):
        manager = SkillManager()
        skill = manager.skills["disengage_reenter"]
        memory = manager.memory
        danger = scene([((0., 0.), "marine")],
                       enemies={"e0": enemy((1.5, 0.), "baneling")})
        snapshot = build_snapshot(with_questions(danger), 1)
        memory.begin_step(snapshot)
        memory.observe(snapshot)
        instance = memory.create_instance(
            "disengage_reenter", ["u0"], "safe", "ESCAPE",
            snapshot.game_loop, 96, goal_kind="internal")
        phase = skill._phase(Context(snapshot), "u0", instance)
        self.assertIn(phase, ("ESCAPE", "STILL_DANGEROUS"),
                      "an adjacent effect weapon is not a safe state")
        safe = scene([((0., 0.), "marine"), ((2., 0.), "medivac")],
                     enemies={"e0": enemy((15., 15.), "baneling")})
        safe["allies"]["u0"]["health"] = 10.
        snapshot2 = build_snapshot(with_questions(safe), 2)
        memory.begin_step(snapshot2)
        memory.observe(snapshot2)
        phase2 = skill._phase(Context(snapshot2), "u0", instance)
        self.assertIn(phase2, ("STABILIZE", "REENTER"))


# ---------------------------------------------------------------------------
# Group 4: search frontier and completion
# ---------------------------------------------------------------------------

class Group4SearchCompletion(unittest.TestCase):
    def test_independent_improvements_combine_in_final_plan(self):
        state = scene([((0., 0.), "marine"), ((3., 3.), "medivac"),
                       ((4., 3.), "marine")],
                      enemies={"e0": enemy((9., 0.))})
        state["allies"]["u1"]["energy"] = 50.
        state["allies"]["u2"]["health"] = 30.
        heal_candidates(state, "u1")
        manager = SkillManager()
        actions, telemetry = manager.step(with_questions(state), 3)
        manager.commit_final_actions(actions)
        skills = {a: info["skill"] for a, info in telemetry["units"].items()}
        self.assertEqual(skills["u0"], "coordinated_engage")
        self.assertEqual(skills["u1"], "heal_support")

    def test_healer_refill_respects_shared_capacity(self):
        state = scene([((0., 0.), "medivac"), ((2., 0.), "medivac"),
                       ((4., 0.), "marine"), ((5., 0.), "marauder")],
                      enemies={"e0": enemy((9., 9.))})
        state["allies"]["u0"]["energy"] = 50.
        state["allies"]["u1"]["energy"] = 50.
        state["allies"]["u2"]["health"] = 30.
        state["allies"]["u3"]["health"] = 40.
        heal_candidates(state, "u0")
        heal_candidates(state, "u1")
        manager = SkillManager()
        actions, telemetry = manager.step(with_questions(state), 4)
        manager.commit_final_actions(actions)
        self.assertNotEqual(actions[0], actions[1],
                            "two new healers on one patient violates the "
                            "verified one-healer capacity")

    def test_tiny_budget_returns_complete_valid_plan(self):
        from skills.tactical_v1.validator import validate_actions
        manager = SkillManager()
        manager.coordinator.max_evaluations = 4
        state = scene([((0., 0.), "marine"), ((1., 0.), "zealot"),
                       ((2., 0.), "stalker")],
                      enemies={"e0": enemy((6., 0.))})
        pl = with_questions(state)
        context = Context(build_snapshot(pl, 1))
        actions, telemetry = manager.step(pl, 3)
        manager.commit_final_actions(actions)
        self.assertEqual(validate_actions(actions, context, 3), [])
        self.assertEqual(len(actions), 3)
        self.assertLessEqual(telemetry["plan_evaluations_used"], 4)


# ---------------------------------------------------------------------------
# Group 5: regression boundaries
# ---------------------------------------------------------------------------

class Group5Boundaries(unittest.TestCase):
    def test_sparse_enemy_ids_keep_real_slots(self):
        state = scene([((0., 0.), "marine")],
                      enemies={"e2": enemy((3., 0.)), "e5": enemy((4., 0.))})
        context = make_context(state)
        self.assertEqual(context.attack_action("u0", "e2"), 8)
        self.assertEqual(context.attack_action("u0", "e5"), 11)
        manager = SkillManager()
        actions, _ = manager.step(with_questions(state), 1)
        self.assertIn(actions[0], (8, 11))

    def test_keep_heal_semantics_unchanged(self):
        state = scene([((0., 0.), "medivac"), ((2., 0.), "marine")],
                      enemies={"e0": enemy((9., 9.))})
        state["allies"]["u1"]["health"] = 30.
        heal_candidates(state, "u0")
        state["action_contract"]["keep_heal_order"] = {"enabled": True,
                                                       "action_id": 16}
        state["keep_heal_order_candidates"] = {"u0": {"action_id": 16,
                                                      "target": "u1"}}
        pl = with_questions(state)
        pl["questions"]["u0"]["criteria"]["16"] = "keep"
        snapshot = build_snapshot(pl, 1)
        semantic = Context(snapshot).decode("u0", 16)
        self.assertEqual(semantic.kind, KEEP_HEAL_ORDER)
        self.assertEqual(semantic.target, "u1")

    def test_ready_damage_not_cancelled_by_cooling_source(self):
        state = scene([((0., 0.), "marine"), ((1., 0.), "marine")],
                      enemies={"e0": enemy((3., 0.))})
        state["allies"]["u1"]["weapon_cooldown_loops"] = 40.
        context = make_context(state)
        report = physical_effects(context, {"u0": 6, "u1": 6})
        self.assertEqual(report.damage_known["e0"], 6.0)

    def test_s1_switch_changes_final_coverage(self):
        on = Evaluator(mechanisms={"splash": True, "burst": True})
        off = Evaluator(mechanisms={"splash": False, "burst": True})
        state = scene([((0., 0.), "colossus")],
                      enemies={"e0": enemy((5., 0.)), "e1": enemy((5.5, 0.))},
                      attack_range=9.0)
        context = make_context(state)
        _, tb_on = on.score_joint(context, [], {"u0": 6})
        _, tb_off = off.score_joint(context, [], {"u0": 6})
        self.assertGreater(tb_on["coverage"], 0.0)
        self.assertEqual(tb_off["coverage"], 0.0)

    def test_unknown_baneling_keeps_legal_attack_attempt(self):
        state = scene([((4., 0.), "baneling")],
                      enemies={"e0": enemy((5., 0.))}, attack_range=3.0)
        state["unit_type_profiles"] = {}
        relation = state["relations"]["u0"]["e0"]
        relation["in_weapon_range"] = None
        relation["native_weapon_range"] = None
        relation["weapon_range_status"] = "unknown"
        context = make_context(state)
        self.assertEqual(context.contact_status("u0", "e0"), "unknown")
        manager = SkillManager()
        plans = manager.skills["burst_engage"].propose(
            context, manager.memory, Telemetry())
        self.assertTrue([p for p in plans
                         if p.current_actions.get("u0") == 6],
                        "legal attack attempt must survive unknown contact")

    def test_ground_coverage_skips_flying_neighbours(self):
        state = scene([((0., 0.), "colossus")],
                      enemies={"e0": enemy((5., 0.)),
                               "e1": enemy((5.5, 0.), is_flying=True)},
                      attack_range=9.0)
        context = make_context(state)
        report = physical_effects(context, {"u0": 6})
        self.assertNotIn("e1", report.coverage,
                         "a ground-only splash mechanism never credits a "
                         "flying neighbour")


class ManagerCycle(unittest.TestCase):
    def test_manager_cycle_legal_and_idempotent(self):
        from skills.tactical_v1.validator import validate_actions
        manager = SkillManager()
        state = scene([((0., 0.), "marine"), ((1., 0.), "zealot"),
                       ((2., 0.), "medivac")],
                      enemies={"e0": enemy((8., 0.)), "e1": enemy((9., 1.))})
        state["allies"]["u2"]["energy"] = 50.
        for step in range(5):
            state["game_loop"] = 80 + step * 8
            pl = with_questions(copy.deepcopy(state))
            actions, _ = manager.step(pl, 3)
            manager.commit_final_actions(actions)
            context = Context(build_snapshot(pl, step + 1))
            self.assertEqual(validate_actions(actions, context, 3), [])
            again, _ = manager.step(pl, 3)
            self.assertEqual(again, actions)
            manager.commit_final_actions(actions)


if __name__ == "__main__":
    unittest.main()
