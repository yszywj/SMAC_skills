"""Compact correctness regression for the tactical_v1 fixes (task 6.2).

Eight property groups, each stated as a physical/interface contract — no
assertion is conditioned away and no expected label is produced by tuning
weights. Run:
    python -m unittest skills.tactical_v1.tests.test_fix_regressions
"""
import copy
import unittest

from skills.tactical_v1.actions import decode
from skills.tactical_v1.context import build_snapshot, Context
from skills.tactical_v1.effects import physical_effects
from skills.tactical_v1.evaluator import Evaluator
from skills.tactical_v1.registry import build_skills
from skills.tactical_v1.skill_manager import SkillManager
from skills.tactical_v1.memory import TacticalMemory
from skills.tactical_v1.telemetry import Telemetry
from .fixtures import (MOVE_DELTAS, STOP, ally, enemy, heal_candidates,
                       scene, with_questions)


def make_context(state):
    return Context(build_snapshot(with_questions(state), 1))


class G1IndexingDecodingTests(unittest.TestCase):
    """Group 1: e0,e2,e5 visibility keeps real slots; heal/keep-heal decode."""

    def test_visible_subset_keeps_original_action_slots(self):
        state = scene([((0., 0.), "marine")],
                      enemies={"e0": enemy((3., 0.)), "e2": enemy((4., 0.)),
                               "e5": enemy((5., 0.))})
        context = make_context(state)
        # attack slots follow REAL enemy indices, not enumerate positions
        self.assertEqual(context.attack_action("u0", "e2"), 6 + 2)
        self.assertEqual(context.attack_action("u0", "e5"), 6 + 5)
        actions, _ = SkillManager().step(with_questions(state), 1)
        semantic = context.decode("u0", actions[0])
        self.assertIn(actions[0], (6, 8, 11))
        self.assertTrue(semantic.valid)

    def test_heal_and_keep_heal_are_never_attack_decodes(self):
        state = scene([((0., 0.), "medivac"), ((2., 0.), "marine")],
                      enemies={"e0": enemy((9., 9.))})
        state["allies"]["u1"]["health"] = 30.
        heal_candidates(state, "u0")
        context = make_context(state)
        semantic = context.decode("u0", 6 + 1)      # heal u1 band
        self.assertEqual(semantic.kind, "heal")
        self.assertEqual(semantic.target, "u1")
        state["action_contract"]["keep_heal_order"] = {"enabled": True,
                                                       "action_id": 16}
        state["keep_heal_order_candidates"] = {"u0": {"action_id": 16,
                                                      "target": "u1"}}
        pl = with_questions(state)
        pl["questions"]["u0"]["criteria"]["16"] = "keep heal order"
        snapshot = build_snapshot(pl, 1)
        self.assertEqual(snapshot.keep_heal_action_id, 16)
        ctx = Context(snapshot)
        semantic = ctx.decode("u0", 16)
        self.assertEqual(semantic.kind, "keep_heal_order")
        self.assertEqual(semantic.target, "u1")


class G2TimingTests(unittest.TestCase):
    """Group 2: window-qualified damage vs legal-but-not-landed attacks."""

    def test_long_cooldown_and_no_contact_score_no_window_damage(self):
        state = scene([((0., 0.), "marine")], enemies={"e0": enemy((3., 0.))})
        state["allies"]["u0"]["weapon_cooldown_loops"] = 40.   # >> H (8)
        context = make_context(state)
        estimate = context.fire_estimate("u0", "e0")
        self.assertTrue(context.can_issue_attack("u0", "e0"))
        self.assertEqual(estimate.window_shots, 0)
        self.assertIsNone(estimate.damage_est)
        report = physical_effects(context, {"u0": 6})
        self.assertEqual(report.damage.get("e0", 0.0), 0.0)

    def test_in_window_attack_keeps_damage_candidate(self):
        state = scene([((0., 0.), "marine")], enemies={"e0": enemy((3., 0.))})
        context = make_context(state)
        estimate = context.fire_estimate("u0", "e0")
        self.assertGreaterEqual(estimate.window_shots, 1)
        self.assertEqual(estimate.damage_est, 6.0)
        report = physical_effects(context, {"u0": 6})
        self.assertEqual(report.damage["e0"], 6.0)

    def test_legal_but_out_of_physical_contact_no_immediate_damage(self):
        # Relation says smac-legal at long distance while catalog range is 5:
        # the contact query must separate command from physics.
        state = scene([((0., 0.), "marine")], enemies={"e0": enemy((6., 0.))})
        state["relations"]["u0"]["e0"]["smac_attack_available"] = True
        state["relations"]["u0"]["e0"]["in_weapon_range"] = True
        pl = with_questions(state)
        pl["questions"]["u0"]["criteria"]["6"] = "attack"
        context = Context(build_snapshot(pl, 1))
        self.assertTrue(context.can_issue_attack("u0", "e0"))


class G3LifecycleTests(unittest.TestCase):
    """Group 3: instance continuation, phase/timeout consumption, and
    commit-integrity when the final action is externally changed."""

    def _search_steps(self, manager, n=4):
        state = scene([((0., 0.), "marine"), ((1., 0.), "marine")])
        instance_ids = []
        for step in range(n):
            state["game_loop"] = 80 + step * 8
            actions, telemetry = manager.step(with_questions(
                copy.deepcopy(state)), 2)
            manager.commit_final_actions(actions)
            ids = [i.instance_id for i in manager.memory.instances.values()]
            instance_ids.append(ids)
        return instance_ids

    def test_same_task_continues_one_instance_not_recreated(self):
        manager = SkillManager()
        history = self._search_steps(manager)
        self.assertTrue(history[0], "an instance must be created on selection")
        first = history[0]
        for later in history[1:]:
            self.assertEqual(later, first,
                             "search task must continue the SAME instance")

    def test_changed_final_action_blocks_plan_commit(self):
        manager = SkillManager()
        state = scene([((0., 0.), "marine")], enemies={"e0": enemy((3., 0.))})
        actions, _ = manager.step(with_questions(state), 1)
        manager.commit_final_actions([1])        # externally changed to Stop
        # the attack plan's instance must NOT have been committed as active
        self.assertEqual(manager.memory.instances, {})

    def test_phase_and_timeout_actually_consume(self):
        manager = SkillManager()
        state = scene([((0., 0.), "marine")])
        for step in range(20):
            state["game_loop"] = 80 + step * 8
            actions, _ = manager.step(with_questions(copy.deepcopy(state)), 1)
            manager.commit_final_actions(actions)
        # waypoint budget (12H = 96 loops) forces retire/renewal, not a
        # forever-growing same-id list and not unbounded instance count
        self.assertLessEqual(len(manager.memory.instances), 1)


class G4ScoringTests(unittest.TestCase):
    """Group 4: packaging-invariance, diminishing fire, full-plan scoring."""

    def _context(self, n_marines=3, enemy_health=35.):
        state = scene([((i, 0.), "marine") for i in range(n_marines)],
                      enemies={"e0": enemy((3., 0.), health=enemy_health)})
        return make_context(state)

    def test_packaging_and_permutation_invariance(self):
        from skills.tactical_v1.contracts import (Commitment, EffectClaim,
                                                  ResourceClaims)
        evaluator = Evaluator()
        context = self._context()
        attack = 6

        def plan(pid, actors):
            from skills.tactical_v1.contracts import CandidatePlan
            return CandidatePlan(
                proposal_id=pid, snapshot_id=1, config_fingerprint="t",
                primary_skill="focus_fire", specialization_provider=None,
                instance_id=None, controlled_actors=tuple(sorted(actors)),
                beneficiaries=(), goal="e0",
                current_actions={a: attack for a in actors},
                expected_effects=[EffectClaim("damage", "e0", 6., 8,
                                              "calibrated", "t")],
                claims=ResourceClaims(controls=tuple(sorted(actors)),
                                      fire=(("e0", 6.),)),
                dependencies=(), commitment=Commitment("attacking", 0),
                features={}, proposed_memory_delta={})
        actions = {f"u{i}": attack for i in range(3)}
        packed, _ = evaluator.score_joint(context, [plan("p1", ["u0", "u1", "u2"])],
                                          actions)
        split, _ = evaluator.score_joint(
            context, [plan("p1", ["u0"]), plan("p2", ["u1"]), plan("p3", ["u2"])],
            actions)
        self.assertAlmostEqual(packed, split, places=6)

    def test_repeated_fire_diminishes_and_caps(self):
        evaluator = Evaluator()
        context = self._context(enemy_health=18.)   # 3 marine shots to kill
        scores = []
        for shooters in (1, 2, 3, 4, 5):
            actions = {f"u{i}": 6 for i in range(shooters)}
            score, _ = evaluator.score_joint(context, [], actions)
            scores.append(score)
        self.assertGreater(scores[0], 0.)
        self.assertLess(scores[1], scores[2])       # kill feasibility adds once
        self.assertAlmostEqual(scores[2], scores[3], places=6,
                               msg="budget satisfied: no further damage credit")
        self.assertAlmostEqual(scores[3], scores[4], places=6,
                               msg="overkill must not keep adding value")

    def test_final_fallback_actions_participate_in_scoring(self):
        # A plan whose units fall back must still be scored on the FINAL
        # actions (risk of the fallback position counts).
        evaluator = Evaluator()
        context = self._context(1)
        state = scene([((0., 0.), "marine")],
                      enemies={"e0": enemy((3., 0.))})
        attack_score, _ = evaluator.score_joint(context, [], {"u0": 6})
        flee_score, _ = evaluator.score_joint(
            context, [], {"u0": 3})   # retreat into more risk
        self.assertGreater(attack_score, flee_score)


class G5JointPlanTests(unittest.TestCase):
    """Group 5: atomic interception integrity and resource release."""

    def test_joint_kill_atomic_and_released_cleanly(self):
        config = SkillManager().config
        skills = build_skills(config)
        state = scene([((0., 0.), "marine"), ((0., 2.), "marine"),
                       ((2., 0.), "marine")],
                      enemies={"e0": enemy((4., 0.), "baneling", health=10.)})
        context = make_context(state)
        plans = skills["focus_fire"].propose(context, TacticalMemory(),
                                             Telemetry())
        joint = [p for p in plans if p.atomic]
        self.assertTrue(joint)
        plan = joint[0]
        self.assertTrue(all(context.attack_action(a, "e0") ==
                            plan.current_actions[a]
                            for a in plan.controlled_actors))
        from skills.tactical_v1.ledger import Ledger
        ledger = Ledger.from_plans([plan])
        self.assertEqual(set(ledger.controls), set(plan.controlled_actors))
        ledger.release(plan.proposal_id)
        self.assertEqual(ledger.controls, {})
        self.assertEqual(ledger.fire, {})

    def test_member_removal_invalidates_whole_instance(self):
        manager = SkillManager()
        memory = manager.memory
        from skills.tactical_v1.contracts import SkillInstance
        instance = memory.create_instance(
            "focus_fire", ["u0", "u1"], "e0", "intercepting", 80, 96)
        state = scene([((0., 0.), "marine"), ((0., 2.), "marine")],
                      enemies={"e0": enemy((4., 0.), "baneling")})
        state["allies"]["u1"]["alive"] = False        # a member dies
        snapshot = build_snapshot(with_questions(state), 2)
        retired = memory.retire_invalid(snapshot)
        self.assertIn(instance.instance_id, retired)


class G6HealTests(unittest.TestCase):
    """Group 6: two medivacs / two patients, patients keep attacking, and a
    competing escape under a real air-capable threat."""

    def _terran(self, second_patient_worse=True):
        state = scene([((0., 0.), "medivac"), ((2., 0.), "medivac"),
                       ((4., 0.), "marine"), ((5., 0.), "marauder")],
                      enemies={"e0": enemy((9., 9.))})
        state["allies"]["u0"]["energy"] = 50.
        state["allies"]["u1"]["energy"] = 50.
        state["allies"]["u2"]["health"] = 40. if not second_patient_worse else 30.
        state["allies"]["u3"]["health"] = 30. if not second_patient_worse else 40.
        heal_candidates(state, "u0")
        heal_candidates(state, "u1")
        return state

    def test_two_medivacs_two_patients_valid_allocation(self):
        manager = SkillManager()
        state = self._terran()
        actions, telemetry = manager.step(with_questions(state), 4)
        manager.commit_final_actions(actions)
        med_actions = {actions[0], actions[1]}
        # each medivac heals a distinct patient (u2 needs more)
        self.assertEqual(actions[0], 6 + 2)          # u0 heals u2 (needier)
        self.assertIn(actions[1], (6 + 2, 6 + 3))
        if actions[1] == actions[0]:
            self.fail("both medivacs piled onto one patient although the "
                      "other patient is wounded and unowned")
        # patients keep their own action (attack/approach), not heal-control
        for patient in ("u2", "u3"):
            self.assertNotEqual(telemetry["units"][patient]["skill"],
                                "heal_support")

    def test_air_threat_creates_competing_escape(self):
        state = self._terran()
        # a stalker CAN hit air: domain-compatible threat on the medivac
        state["visible_enemies"]["e1"] = enemy((3., 1.), "stalker")
        for actor in state["allies"]:
            state["relations"][actor]["e1"] = {
                "distance": 3.2, "in_weapon_range": False,
                "weapon_range_status": "out_of_range",
                "native_weapon_range": 6.0, "in_enemy_weapon_range": True,
                "enemy_weapon_range_status": "in_range",
                "smac_attack_available": False}
        context = make_context(state)
        risk = context.risk_at("u0", state["allies"]["u0"]["position"])
        self.assertIn("e1", risk.threats)
        # ground-only threats must NOT threaten the flying medivac
        state2 = self._terran()
        state2["visible_enemies"]["e1"] = enemy((3., 1.), "zergling")
        for actor in state2["allies"]:
            state2["relations"][actor]["e1"] = dict(
                state2["relations"][actor]["e0"])
            state2["relations"][actor]["e1"][
                "enemy_weapon_range_status"] = "no_compatible_weapon"
        context2 = make_context(state2)
        risk2 = context2.risk_at("u0", context2.snapshot.allies["u0"].position)
        self.assertNotIn("e1", risk2.threats)


class G7GroupsAndSearchTests(unittest.TestCase):
    """Group 7: two tight-but-distant squads; shared position conflicts are
    evaluated; a move command never writes `visited`."""

    def test_two_distant_squads_both_detected(self):
        state = scene([((0., 0.), "marine"), ((1., 0.), "marine"),
                       ((40., 40.), "marine"), ((41., 40.), "marine")],
                      enemies={"e0": enemy((4., 0.))})
        context = make_context(state)
        groups = context.engagement_groups()
        self.assertEqual(len(groups), 2)

    def test_predicted_overlap_is_penalized(self):
        evaluator = Evaluator()
        state = scene([((0., 0.), "zealot"), ((0.6, 0.), "zealot")],
                      enemies={"e0": enemy((9., 9.))})
        context = make_context(state)
        apart, _ = evaluator.score_joint(context, [], {"u0": 2, "u1": 3})
        together, _ = evaluator.score_joint(context, [], {"u0": 2, "u1": 2})
        self.assertLess(together, apart)

    def test_move_issue_never_writes_visited(self):
        manager = SkillManager()
        state = scene([((0., 0.), "marine")])
        for step in range(6):
            state["game_loop"] = 80 + step * 8
            actions, _ = manager.step(with_questions(copy.deepcopy(state)), 1)
            manager.commit_final_actions(actions)
            self.assertEqual(manager.memory.visited_search_cells, {},
                             "visited may only be written on ARRIVAL")


class G8SpecializationTests(unittest.TestCase):
    """Group 8: the S1 switch changes real provider calls; colossus keeps
    cycle repositioning; duplicate coverage saturates."""

    def _skills(self, enabled):
        from skills.tactical_v1.skill_manager import SkillManager
        manager = SkillManager()
        manager.config["enabled"]["splash_fire"] = enabled
        from skills.tactical_v1.registry import build_skills
        return build_skills(manager.config), manager.config

    def test_s1_switch_changes_provider_path(self):
        on_skills, config = self._skills(True)
        off_skills, _ = self._skills(False)
        self.assertIn("splash_fire", on_skills["focus_fire"].providers)
        self.assertNotIn("splash_fire", off_skills["focus_fire"].providers)
        # the real call path differs: with S1 the impact claims carry
        # coverage opportunities; without it they never do
        state = scene([((0., 0.), "colossus")],
                      enemies={"e0": enemy((5., 0.)),
                               "e1": enemy((5.5, 0.))}, attack_range=9.0)
        context = make_context(state)
        with_s1 = on_skills["focus_fire"]._impact(context, "u0", "e0")
        without_s1 = off_skills["focus_fire"]._impact(context, "u0", "e0")
        self.assertTrue(any(e.kind == "coverage" for e in with_s1))
        self.assertFalse(any(e.kind == "coverage" for e in without_s1))

    def test_colossus_keeps_reposition_candidates(self):
        skills, config = self._skills(True)
        state = scene([((0., 0.), "colossus")],
                      enemies={"e0": enemy((4., 0.))}, attack_range=9.0)
        state["allies"]["u0"]["weapon_cooldown_loops"] = 16.
        state["allies"]["u0"]["health_lost_since_previous_decision"] = 6.
        context = make_context(state)
        plans = skills["combat_reposition"].propose(context, TacticalMemory(),
                                                    Telemetry())
        self.assertTrue(plans, "colossus must not lose repositioning with S1 on")

    def test_duplicate_coverage_saturates(self):
        evaluator = Evaluator(coverage_radius=3.0)
        state = scene([((0., 0.), "colossus"), ((0., 2.), "colossus")],
                      enemies={"e0": enemy((5., 0.)), "e1": enemy((5.5, 0.))},
                      attack_range=9.0)
        context = make_context(state)
        one, one_breakdown = evaluator.score_joint(context, [], {"u0": 6})
        two, two_breakdown = evaluator.score_joint(context, [], {"u0": 6, "u1": 6})
        # second colossus on the same area adds LESS coverage than the first
        self.assertLess(two_breakdown["coverage"] - one_breakdown["coverage"],
                        one_breakdown["coverage"] + 1e-9)


class ManagerIntegrationTests(unittest.TestCase):
    """One pseudo-environment pass: step -> commit -> step."""

    def test_full_step_commit_cycle_is_legal_and_idempotent(self):
        manager = SkillManager()
        state = scene([((0., 0.), "marine"), ((1., 0.), "zealot"),
                       ((2., 0.), "medivac")],
                      enemies={"e0": enemy((8., 0.)), "e1": enemy((9., 1.))})
        state["allies"]["u2"]["energy"] = 50.
        for step in range(5):
            state["game_loop"] = 80 + step * 8
            pl = with_questions(copy.deepcopy(state))
            actions, telemetry = manager.step(pl, 3)
            manager.commit_final_actions(actions)
            from skills.tactical_v1.validator import validate_actions
            context = Context(build_snapshot(pl, step + 1))
            self.assertEqual(validate_actions(actions, context, 3), [])
            # idempotent re-read at the same clock
            again, _ = manager.step(pl, 3)
            self.assertEqual(again, actions)
            for actor, action in enumerate(actions):
                if action in MOVE_DELTAS:
                    dx, dy = MOVE_DELTAS[action]
                    p = state["allies"][f"u{actor}"]["position"]
                    state["allies"][f"u{actor}"]["position"] = [p[0] + dx, p[1] + dy]


if __name__ == "__main__":
    unittest.main()
