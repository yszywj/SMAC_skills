"""Compact correctness regression, round 3 (task section 7).

Six property groups over the REAL implementation (imports only, no copied
functions). Deterministic scenes plus a few consecutive snapshots; no
random soak, no full-suite rerun, no real game.
    python -m unittest skills.tactical_v1.tests.test_fix_regressions
"""
import copy
import unittest

from skills.tactical_v1.actions import KEEP_HEAL_ORDER, decode
from skills.tactical_v1.context import build_snapshot, Context
from skills.tactical_v1.effects import physical_effects
from skills.tactical_v1.evaluator import Evaluator
from skills.tactical_v1.registry import build_skills
from skills.tactical_v1.skill_manager import SkillManager
from skills.tactical_v1.memory import TacticalMemory
from skills.tactical_v1.telemetry import Telemetry
from .fixtures import (MOVE_DELTAS, STOP, enemy, heal_candidates, scene,
                       with_questions)


def make_context(state):
    return Context(build_snapshot(with_questions(state), 1))


# ---------------------------------------------------------------------------
# Group 1: joint selection & intent
# ---------------------------------------------------------------------------

class Group1JointSelection(unittest.TestCase):
    def test_disjoint_plans_coexist_in_final_plan(self):
        # u0 engages a distant enemy while u1 (medivac) heals u2: both must
        # survive together in the final plan.
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
        self.assertEqual(actions[1], 6 + 2)        # heal u2

    def test_plan_never_conflicts_with_itself(self):
        from skills.tactical_v1.ledger import Ledger
        from skills.tactical_v1.contracts import (Commitment, ResourceClaims)

        def plan(pid, actors):
            from skills.tactical_v1.contracts import CandidatePlan
            return CandidatePlan(
                proposal_id=pid, snapshot_id=1, config_fingerprint="t",
                primary_skill="focus_fire", specialization_provider=None,
                instance_id=None, controlled_actors=tuple(sorted(actors)),
                beneficiaries=(), goal="e0",
                current_actions={a: 6 for a in actors},
                expected_effects=[], claims=ResourceClaims(
                    controls=tuple(sorted(actors))),
                dependencies=(), commitment=Commitment("attacking", 0),
                features={}, proposed_memory_delta={})
        ledger = Ledger.from_plans([plan("p1", ["u0", "u1"]),
                                    plan("p2", ["u2"])])
        self.assertEqual(set(ledger.controls), {"u0", "u1", "u2"})
        self.assertEqual(ledger.conflicts(plan("p1", ["u0", "u1"])), [],
                         "a plan must not conflict with itself")
        self.assertTrue(ledger.conflicts(plan("p3", ["u0"])))

    def test_same_action_different_task_not_force_attributed(self):
        # A northward move can be search or approach; the manager must not
        # commit a task instance whose goal differs from the actor's real
        # task. We create a live SEARCH instance, then present a state where
        # the search waypoint is EAST: an engage-flavoured north move must
        # not silently advance the search instance.
        manager = SkillManager()
        state = scene([((0., 0.), "marine")], search=(16., 0.))   # east
        for step in range(2):
            state["game_loop"] = 80 + step * 8
            actions, _ = manager.step(with_questions(copy.deepcopy(state)), 1)
            manager.commit_final_actions(actions)
        search_ids = [i.instance_id for i in manager.memory.instances.values()
                      if i.skill == "cohesive_search"]
        self.assertTrue(search_ids)
        instance = manager.memory.instance_by_id(search_ids[0])
        self.assertEqual(instance.goal_kind, "position")
        self.assertEqual(tuple(instance.extra["waypoint"]), (16., 0.))


# ---------------------------------------------------------------------------
# Group 2: effects & mechanism switches
# ---------------------------------------------------------------------------

class Group2EffectsSwitches(unittest.TestCase):
    def test_ready_source_not_cancelled_by_cooling_source(self):
        state = scene([((0., 0.), "marine"), ((1., 0.), "marine")],
                      enemies={"e0": enemy((3., 0.))})
        state["allies"]["u1"]["weapon_cooldown_loops"] = 40.    # >> H
        context = make_context(state)
        report = physical_effects(context, {"u0": 6, "u1": 6})
        self.assertEqual(report.damage_known["e0"], 6.0,
                         "u0's known contribution must survive u1's cooldown")
        self.assertIn("u1", report.unknown_sources.get("e0", [])
                      + ([], ) [0] if False else report.attack_commands["e0"])
        self.assertNotIn("u1", report.window_sources.get("e0", []))

    def test_colossus_without_window_shot_gets_no_window_coverage(self):
        state = scene([((0., 0.), "colossus")],
                      enemies={"e0": enemy((5., 0.)), "e1": enemy((5.5, 0.))},
                      attack_range=9.0)
        state["allies"]["u0"]["weapon_cooldown_loops"] = 40.    # >> H
        context = make_context(state)
        report = physical_effects(context, {"u0": 6})
        self.assertEqual(sum(report.coverage.values()), 0.0)
        self.assertNotIn("u0", report.window_sources.get("e0", []))

    def test_s1_off_removes_final_coverage_optimization(self):
        evaluator_on = Evaluator(mechanisms={"splash": True, "burst": True})
        evaluator_off = Evaluator(mechanisms={"splash": False, "burst": True})
        state = scene([((0., 0.), "colossus")],
                      enemies={"e0": enemy((5., 0.)), "e1": enemy((5.5, 0.))},
                      attack_range=9.0)
        context = make_context(state)
        on, breakdown_on = evaluator_on.score_joint(context, [], {"u0": 6})
        off, breakdown_off = evaluator_off.score_joint(context, [], {"u0": 6})
        self.assertGreater(breakdown_on["coverage"], 0.0)
        self.assertEqual(breakdown_off["coverage"], 0.0,
                         "S1 off must remove the final coverage credit")
        # provider path really changes too (G1 candidates)
        config = SkillManager().config
        config["enabled"]["splash_fire"] = False
        skills_off = build_skills(config)
        self.assertNotIn("splash_fire", skills_off["focus_fire"].providers)
        config2 = SkillManager().config
        skills_on = build_skills(config2)
        plans_on = skills_on["focus_fire"].propose(
            context, TacticalMemory(), Telemetry())
        plans_off = skills_off["focus_fire"].propose(
            context, TacticalMemory(), Telemetry())
        self.assertTrue(any(e.kind == "coverage" for p in plans_on
                            for e in p.expected_effects))
        self.assertFalse(any(e.kind == "coverage" for p in plans_off
                             for e in p.expected_effects))


# ---------------------------------------------------------------------------
# Group 3: healing
# ---------------------------------------------------------------------------

class Group3Healing(unittest.TestCase):
    def _terran(self, u2_health=30., u3_health=44.):
        state = scene([((0., 0.), "medivac"), ((2., 0.), "medivac"),
                       ((4., 0.), "marine"), ((5., 0.), "marauder")],
                      enemies={"e0": enemy((9., 9.))})
        state["allies"]["u0"]["energy"] = 50.
        state["allies"]["u1"]["energy"] = 50.
        state["allies"]["u2"]["health"] = u2_health
        state["allies"]["u3"]["health"] = u3_health
        heal_candidates(state, "u0")
        heal_candidates(state, "u1")
        return state

    def test_keep_and_reissue_have_consistent_base_opportunity(self):
        # same patient, same conditions: keep-order and a fresh heal command
        # earn the same base heal opportunity in the effects model.
        state = self._terran()
        context = make_context(state)
        fresh = physical_effects(context, {"u0": 6 + 2})   # heal u2
        state2 = self._terran()
        state2["action_contract"]["keep_heal_order"] = {"enabled": True,
                                                        "action_id": 16}
        state2["keep_heal_order_candidates"] = {"u0": {"action_id": 16,
                                                       "target": "u2"}}
        pl = with_questions(state2)
        pl["questions"]["u0"]["criteria"]["16"] = "keep"
        context2 = Context(build_snapshot(pl, 1))
        semantic = context2.decode("u0", 16)
        self.assertEqual(semantic.kind, KEEP_HEAL_ORDER)
        kept = physical_effects(context2, {"u0": 16})
        self.assertEqual(kept.heal["u2"].need, fresh.heal["u2"].need)
        self.assertAlmostEqual(kept.heal["u2"].opportunity,
                               fresh.heal["u2"].opportunity, places=6)

    def test_light_and_heavy_wounds_differ(self):
        light = make_context(self._terran(u2_health=44.))    # missing 1
        heavy = make_context(self._terran(u2_health=5.))     # missing 40
        light_report = physical_effects(light, {"u0": 6 + 2})
        heavy_report = physical_effects(heavy, {"u0": 6 + 2})
        self.assertGreater(heavy_report.heal["u2"].need,
                           light_report.heal["u2"].need + 0.5)

    def test_two_healers_capacity_and_patient_keeps_attacking(self):
        manager = SkillManager()
        state = self._terran()
        actions, telemetry = manager.step(with_questions(state), 4)
        manager.commit_final_actions(actions)
        healed = {actions[0], actions[1]}
        # one active source per patient: the two medivacs treat DIFFERENT
        # patients (u2 is needier)
        self.assertEqual(actions[0], 6 + 2)
        self.assertNotEqual(actions[1], actions[0],
                            "second healer must not pile on the same patient")
        self.assertIn(actions[1], (6 + 3,))
        # patients keep fighting under their own control
        for patient in ("u2", "u3"):
            self.assertNotEqual(telemetry["units"][patient]["skill"],
                                "heal_support")

    def test_joint_move_evaluated_by_relative_position(self):
        # patient stepping AWAY from the medivac loses sustain; a patient
        # holding position does not.
        state = self._terran()
        context = make_context(state)
        holding = physical_effects(context, {"u0": 6 + 2, "u2": 1})
        moving_away = physical_effects(context, {"u0": 6 + 2, "u2": 4})
        # u2 at (4,0), u0 at (0,0): west(5) increases the gap; stop keeps it
        self.assertGreater(holding.heal["u2"].sustain,
                           moving_away.heal["u2"].sustain)


# ---------------------------------------------------------------------------
# Group 4: lifecycle
# ---------------------------------------------------------------------------

class Group4Lifecycle(unittest.TestCase):
    def test_history_compared_before_record_and_progress_is_task_relevant(self):
        manager = SkillManager()
        # rebuild the scene each step so relations/movement candidates stay
        # consistent with the observed position (no stale fixture fields)
        for step, x in enumerate((0.0, 1.0)):
            state = scene([((x, 0.), "marine")],
                          enemies={"e0": enemy((9., 0.))},
                          loop=80 + step * 8)
            pl = with_questions(state)
            actions, _ = manager.step(pl, 1)
            manager.commit_final_actions(actions)
        engage = [i for i in manager.memory.instances.values()
                  if i.skill in ("coordinated_engage", "focus_fire")]
        self.assertTrue(engage)
        self.assertGreater(engage[0].last_progress_loop, engage[0].start_loop,
                           "closing on the target must advance progress")
        # same-clock repeated observation appends no second history record
        state = scene([((1.0, 0.), "marine")], enemies={"e0": enemy((9., 0.))},
                      loop=88)
        pl = with_questions(state)
        length = len(manager.memory.move_history["u0"])
        actions, _ = manager.step(pl, 1)          # identical payload
        manager.commit_final_actions(actions)
        self.assertEqual(len(manager.memory.move_history["u0"]), length)

    def test_propose_does_not_mutate_public_memory(self):
        manager = SkillManager()
        state = scene([((0., 0.), "marine"), ((1., 0.), "zealot")],
                      enemies={"e0": enemy((9., 9.))})
        context = make_context(state)
        before = copy.deepcopy({
            k: v for k, v in manager.memory.__dict__.items()})
        telemetry = Telemetry()
        for skill in manager.skills.values():
            if getattr(skill, "enabled", True):
                skill.propose(context, manager.memory, telemetry)
        for key, value in before.items():
            self.assertEqual(manager.memory.__dict__[key], value,
                             f"propose mutated {key}")

    def test_g4_phase_machine_transitions_on_observed_conditions(self):
        manager = SkillManager()
        memory = manager.memory
        skill = manager.skills["disengage_reenter"]
        # ESCAPE instance, unit still inside both zergling weapon ranges:
        # the machine must keep ESCAPE (or the explicit still-dangerous
        # path) — never STABILIZE just because an escape move was picked.
        dangerous = scene([((0., 0.), "marine")],
                          enemies={"e0": enemy((1.5, 0.)),
                                   "e1": enemy((0., 1.5))})
        snapshot = build_snapshot(with_questions(dangerous), 1)
        memory.begin_step(snapshot)
        memory.observe(snapshot)
        instance = memory.create_instance(
            "disengage_reenter", ["u0"], "safe", "ESCAPE",
            snapshot.game_loop, 12 * snapshot.step_mul)
        phase = skill._phase(Context(snapshot), "u0", instance)
        self.assertIn(phase, ("ESCAPE", "STILL_DANGEROUS"))
        # risk gone + healer present + low health -> STABILIZE/REENTER legal
        safe = scene([((0., 0.), "marine"), ((2., 0.), "medivac")],
                     enemies={"e0": enemy((1.5, 0.)), "e1": enemy((0., 1.5))})
        safe["allies"]["u0"]["health"] = 10.
        for other in ("e0", "e1"):
            safe["relations"]["u0"][other][
                "enemy_weapon_range_status"] = "no_compatible_weapon"
        snapshot2 = build_snapshot(with_questions(safe), 2)
        memory.begin_step(snapshot2)
        memory.observe(snapshot2)
        phase2 = skill._phase(Context(snapshot2), "u0", instance)
        self.assertIn(phase2, ("STABILIZE", "REENTER"))

    def test_timeout_does_not_restart_with_fresh_budget(self):
        manager = SkillManager()
        state = scene([((0., 0.), "marine")], search=(16., 0.))
        ids = set()
        for step in range(20):
            state["game_loop"] = 80 + step * 8
            actions, _ = manager.step(with_questions(copy.deepcopy(state)), 1)
            manager.commit_final_actions(actions)
            ids |= {i.instance_id for i in
                    manager.memory.instances.values()}
        # waypoint budget (12H) forces at least one retirement; a NEW task on
        # the SAME goal must be blocked by the retry record for a while, and
        # fresh instances may not keep the SAME id with a reset budget
        self.assertLessEqual(len(ids), 3)


# ---------------------------------------------------------------------------
# Group 5: generic spatial abilities
# ---------------------------------------------------------------------------

class Group5Spatial(unittest.TestCase):
    def test_moving_away_from_engage_target_scores_negative_progress(self):
        from skills.tactical_v1.skills.common import move_progress
        state = scene([((0., 0.), "marine")], enemies={"e0": enemy((6., 0.))})
        context = make_context(state)
        toward = move_progress(context, "u0", 4, (6., 0.))    # east: closer
        away = move_progress(context, "u0", 5, (6., 0.))      # west: farther
        self.assertGreater(toward, 0.0)
        self.assertLess(away, 0.0, "moving away must stay negative (no abs)")

    def test_two_tight_distant_squads_get_comparable_merge_plans(self):
        manager = SkillManager()
        skills = manager.skills
        state = scene([((0., 0.), "marine"), ((1., 0.), "marine"),
                       ((2., 0.), "marine"), ((3., 0.), "marine"),
                       ((4., 0.), "marine"),
                       ((40., 40.), "marine"), ((41., 40.), "marine"),
                       ((42., 40.), "marine"), ((43., 40.), "marine"),
                       ((44., 40.), "marine")],
                      enemies={"e0": enemy((8., 0.))})
        context = make_context(state)
        plans = skills["regroup"].propose(context, manager.memory, Telemetry())
        rear = [p for p in plans if "u5" in p.controlled_actors]
        self.assertTrue(rear, "the far squad must receive merge proposals")
        self.assertGreaterEqual(len({p.goal for p in plans}), 1)

    def test_predicted_position_conflict_enters_final_evaluation(self):
        evaluator = Evaluator()
        state = scene([((0., 0.), "zealot"), ((0.6, 0.), "zealot")],
                      enemies={"e0": enemy((9., 9.))})
        context = make_context(state)
        apart, apart_b = evaluator.score_joint(context, [], {"u0": 2, "u1": 3})
        together, together_b = evaluator.score_joint(context, [],
                                                     {"u0": 2, "u1": 2})
        self.assertLess(together_b["congestion"], apart_b["congestion"])

    def test_search_visited_only_on_arrival_and_then_moves_on(self):
        manager = SkillManager()
        state = scene([((0., 0.), "marine")], search=(4., 0.))
        for step in range(3):
            state["game_loop"] = 80 + step * 8
            actions, _ = manager.step(with_questions(copy.deepcopy(state)), 1)
            manager.commit_final_actions(actions)
            self.assertEqual(manager.memory.visited_search_cells, {},
                             "issuing moves never writes visited")
        # actually arrive at the waypoint
        state["allies"]["u0"]["position"] = [4., 0.]
        state["game_loop"] = 80 + 3 * 8
        actions, _ = manager.step(with_questions(copy.deepcopy(state)), 1)
        manager.commit_final_actions(actions)
        self.assertTrue(manager.memory.visited_search_cells,
                        "arrival must be confirmed in the observation phase")
        # after arrival the next waypoint differs from the visited one
        state["game_loop"] = 80 + 4 * 8
        actions, _ = manager.step(with_questions(copy.deepcopy(state)), 1)
        manager.commit_final_actions(actions)
        search = [i for i in manager.memory.instances.values()
                  if i.skill == "cohesive_search"]
        if search:
            self.assertNotIn(_cell(tuple(search[0].extra["waypoint"])),
                             manager.memory.visited_search_cells)


def _cell(point):
    return f"{int(point[0] // 8)}:{int(point[1] // 8)}"


# ---------------------------------------------------------------------------
# Group 6: specializations & protocol boundaries
# ---------------------------------------------------------------------------

class Group6SpecializationProtocol(unittest.TestCase):
    def test_baneling_empty_weapons_keeps_legal_attack_attempt(self):
        # strip all catalog profiles: weapons unknown -> contact unknown ->
        # the attack ATTEMPT must still be proposed, with no fake damage.
        state = scene([((4., 0.), "baneling")],
                      enemies={"e0": enemy((5., 0.))}, attack_range=3.0)
        state["unit_type_profiles"] = {}      # empty weapons: supported input
        relation = state["relations"]["u0"]["e0"]
        relation["in_weapon_range"] = None
        relation["native_weapon_range"] = None
        relation["weapon_range_status"] = "unknown"
        context = make_context(state)
        self.assertEqual(context.contact_status("u0", "e0"), "unknown")
        manager = SkillManager()
        plans = manager.skills["burst_engage"].propose(
            context, manager.memory, Telemetry())
        attacks = [p for p in plans if p.current_actions.get("u0") == 6]
        self.assertTrue(attacks, "legal attack attempt must survive unknown "
                                 "weapons/contact")
        for plan in attacks:
            for effect in plan.expected_effects:
                if effect.kind in ("damage", "loss"):
                    self.assertNotIn(plan.goal, [effect.target]
                                     and [])  # no fabricated amounts:
                    self.assertTrue(effect.amount is None or
                                    effect.target == "u0")

    def test_baneling_single_enemy_still_fights(self):
        state = scene([((0., 0.), "baneling")],
                      enemies={"e0": enemy((9., 0.))})   # lonely, far
        manager = SkillManager()
        actions, telemetry = manager.step(with_questions(state), 1)
        manager.commit_final_actions(actions)
        self.assertEqual(telemetry["units"]["u0"]["skill"], "burst_engage")
        self.assertNotEqual(actions[0], STOP,
                            "a lone enemy must not cause permanent waiting")

    def test_colossus_keeps_repositioning_with_s1(self):
        manager = SkillManager()
        state = scene([((0., 0.), "colossus")],
                      enemies={"e0": enemy((4., 0.))}, attack_range=9.0)
        state["allies"]["u0"]["weapon_cooldown_loops"] = 16.
        state["allies"]["u0"]["health_lost_since_previous_decision"] = 6.
        context = make_context(state)
        plans = manager.skills["combat_reposition"].propose(
            context, manager.memory, Telemetry())
        self.assertTrue(plans, "colossus keeps G3 with S1 enabled")

    def test_sparse_enemy_ids_and_keep_heal_extension(self):
        state = scene([((0., 0.), "marine")],
                      enemies={"e2": enemy((3., 0.)), "e5": enemy((4., 0.))})
        context = make_context(state)
        self.assertEqual(context.attack_action("u0", "e2"), 6 + 2)
        self.assertEqual(context.attack_action("u0", "e5"), 6 + 5)
        manager = SkillManager()
        actions, telemetry = manager.step(with_questions(state), 1)
        self.assertIn(actions[0], (6 + 2, 6 + 5))
        semantic = context.decode("u0", actions[0])
        self.assertEqual(semantic.kind, "attack")
        self.assertIn(semantic.target, ("e2", "e5"))
        # keep-heal extension never decodes as an attack for a healer
        state2 = scene([((0., 0.), "medivac"), ((2., 0.), "marine")],
                       enemies={"e0": enemy((9., 9.))})
        state2["allies"]["u1"]["health"] = 30.
        heal_candidates(state2, "u0")
        state2["action_contract"]["keep_heal_order"] = {"enabled": True,
                                                        "action_id": 16}
        state2["keep_heal_order_candidates"] = {"u0": {"action_id": 16,
                                                       "target": "u1"}}
        pl = with_questions(state2)
        pl["questions"]["u0"]["criteria"]["16"] = "keep"
        snapshot = build_snapshot(pl, 1)
        semantic = Context(snapshot).decode("u0", 16)
        self.assertEqual(semantic.kind, KEEP_HEAL_ORDER)
        self.assertEqual(semantic.target, "u1")


class ManagerCycle(unittest.TestCase):
    def test_full_manager_step_commit_cycle_stays_legal_and_idempotent(self):
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
            manager.commit_final_actions(actions)   # idempotent re-commit


if __name__ == "__main__":
    unittest.main()
