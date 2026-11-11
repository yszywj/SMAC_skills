"""D. The three mechanism specializations: Colossus splash, Medivac
sustained healing, Baneling burst trade."""
import copy
import unittest

from skills.tactical_v1.context import build_snapshot, Context
from skills.tactical_v1.registry import build_skills
from skills.tactical_v1.memory import TacticalMemory
from skills.tactical_v1.telemetry import Telemetry
from skills.tactical_v1.skill_manager import SkillManager
from .fixtures import (TacticalTestCase, ally, enemy, scene, with_questions,
                       heal_candidates, MOVE_DELTAS, STOP)


def colossus_context(self, enemies, allies=None):
    allies = allies or [((0., 0.), "colossus")]
    state = scene(allies, enemies=enemies, attack_range=9.0)
    return Context(build_snapshot(with_questions(state), 1)), state


class SplashTests(TacticalTestCase):
    def test_cluster_geometry_changes_target_value(self):
        from skills.tactical_v1.skills.splash_fire import SplashFireSkill
        skill = SplashFireSkill(SkillManager().config)
        context, _ = colossus_context(
            self, {"e0": enemy((8., 0.))})
        alone = skill.coverage_bonus(context, "u0", "e0")
        clustered, _ = colossus_context(self, {
            "e0": enemy((8., 0.)), "e1": enemy((8.5, 0.)), "e2": enemy((8., 0.5))})
        bonus = skill.coverage_bonus(clustered, "u0", "e0")
        self.assertEqual(alone, 0.0)
        self.assertGreater(bonus, alone)

    def test_secondary_effects_are_heuristic_and_uncredited(self):
        from skills.tactical_v1.skills.splash_fire import SplashFireSkill
        skill = SplashFireSkill(SkillManager().config)
        context, _ = colossus_context(self, {
            "e0": enemy((8., 0.)), "e1": enemy((8.5, 0.))})
        claims = skill.impact_claims(context, "u0", "e0")
        primary = [c for c in claims if c.target == "e0"]
        secondary = [c for c in claims if c.target != "e0"]
        self.assertTrue(primary and primary[0].amount >= 0)
        self.assertTrue(secondary)
        for claim in secondary:
            self.assertEqual(claim.evidence, "heuristic")
            self.assertEqual(claim.amount, 0.0)   # excluded from kill budget

    def test_multiple_colossus_marginal_coverage(self):
        # The second colossus aiming at an already-covered cluster earns
        # less evaluator credit than the first (marginal fire accounting).
        from skills.tactical_v1.evaluator import Evaluator
        from skills.tactical_v1.ledger import Ledger
        evaluator = Evaluator()
        context, state = colossus_context(
            self, {"e0": enemy((8., 0.)), "e1": enemy((8.5, 0.))},
            allies=[((0., 0.), "colossus"), ((0., 2.), "colossus")])
        skills = build_skills(SkillManager().config)
        plans = skills["focus_fire"].propose(context, TacticalMemory(),
                                             Telemetry())
        colossus_plans = [p for p in plans
                          if p.specialization_provider == "splash_fire"
                          and p.goal == "e0"]
        self.assertGreaterEqual(len(colossus_plans), 2)
        ledger = Ledger()
        first = evaluator.score_plan(colossus_plans[0], context, ledger)
        ledger.reserve(colossus_plans[0])
        second = evaluator.score_plan(colossus_plans[1], context, ledger)
        self.assertGreater(first, second)


class HealTests(TacticalTestCase):
    def terran_scene(self, **kwargs):
        state = scene([((0., 0.), "medivac"), ((2., 0.), "marine"),
                       ((3., 0.), "marauder")],
                      enemies={"e0": enemy((9., 9.))}, **kwargs)
        state["allies"]["u0"]["energy"] = 50.
        return state

    def test_no_patients_medivac_follows_group_not_enemy(self):
        state = self.terran_scene()
        actions, telemetry = self.step(self.manager(), state)
        self.assertEqual(telemetry["units"]["u0"]["skill"], "heal_support")
        # The medivac must not move toward the distant enemy: any legal move
        # it takes points at its group, or it holds.
        unit = state["allies"]["u0"]
        if actions[0] in MOVE_DELTAS:
            dx, dy = MOVE_DELTAS[actions[0]]
            after = (unit["position"][0] + dx, unit["position"][1] + dy)
            to_enemy_before = ((9 - 0) ** 2 + (9 - 0) ** 2) ** .5
            to_enemy_after = ((9 - after[0]) ** 2 + (9 - after[1]) ** 2) ** .5
            self.assertGreaterEqual(to_enemy_after, to_enemy_before)

    def test_multiple_patients_pick_neediest(self):
        state = self.terran_scene()
        state["allies"]["u1"]["health"] = 30.   # missing 15
        state["allies"]["u2"]["health"] = 40.   # missing 5
        heal_candidates(state, "u0")
        actions, telemetry = self.step(self.manager(), state)
        self.assertEqual(telemetry["units"]["u0"]["goal"], "u1")
        self.assertEqual(actions[0], 6 + 1)   # heal u1

    def test_shield_loss_is_not_healable_demand(self):
        state = self.terran_scene()
        # Marine at full HEALTH with lost shield only: no heal candidate
        # is legal for it in this protocol (missing_health == 0).
        state["allies"]["u1"]["shield"] = 20.
        state["allies"]["u1"]["max_shield"] = 60.
        state["allies"]["u1"]["shield_lost_since_previous_decision"] = 40.
        candidates = heal_candidates(state, "u0")
        self.assertNotIn("u1", [c["target"] for c in candidates.values()])

    def test_patient_move_joint_candidate(self):
        # A far wounded marauder: the medivac approaches. The patient keeps
        # its own action (combat/approach); the heal plan never claims it.
        state = scene([((0., 0.), "medivac"), ((2., 0.), "marine"),
                       ((6., 0.), "marauder")],
                      enemies={"e0": enemy((9., 9.))})
        state["allies"]["u0"]["energy"] = 50.
        state["allies"]["u2"]["health"] = 10.
        candidates = heal_candidates(state, "u0", max_distance=99.)
        self.assertIn("u2", [c["target"] for c in candidates.values()])
        manager = self.manager()
        actions, telemetry = self.step(manager, state)
        self.assertEqual(telemetry["units"]["u0"]["skill"], "heal_support")
        self.assertEqual(telemetry["units"]["u0"]["goal"], "u2")
        # The patient is controlled by a combat skill, not by healing.
        self.assertIn(telemetry["units"]["u2"]["skill"],
                      ("focus_fire", "coordinated_engage"))
        self.assertNotEqual(actions[2], STOP)

    def test_medivac_never_attacks_and_ignores_ground_only_threat(self):
        state = self.terran_scene()
        state["visible_enemies"]["e0"] = enemy((2., 1.))  # adjacent zergling
        for actor in state["allies"]:
            state["relations"][actor]["e0"]["distance"] = (
                1.4 if actor == "u0" else 8.0)
            state["relations"][actor]["e0"]["smac_attack_available"] = (
                actor != "u0")
        actions, telemetry = self.step(self.manager(), state)
        self.assertNotEqual(telemetry["units"]["u0"]["skill"], "focus_fire")

    def test_keep_heal_order_used_only_when_extension_present(self):
        state = self.terran_scene()
        state["allies"]["u1"]["health"] = 30.
        heal_candidates(state, "u0")
        # Without the extension: plain heal action.
        actions, telemetry = self.step(self.manager(), state)
        self.assertEqual(actions[0], 6 + 1)
        # With the extension: the dedicated keep action wins when observed.
        state["action_contract"]["keep_heal_order"] = {"enabled": True,
                                                       "action_id": 16}
        state["keep_heal_order_candidates"] = {
            "u0": {"action_id": 16, "target": "u1"}}
        pl = with_questions(state)
        pl["questions"]["u0"]["criteria"]["16"] = "keep heal order"
        actions2, telemetry2 = manager2 = SkillManager().step(pl, 3)[0:2]
        self.assertEqual(actions2[0], 16)


class BurstTests(TacticalTestCase):
    def baneling_scene(self, allies, enemies):
        return scene(allies, enemies=enemies)

    def test_burst_into_high_value_cluster(self):
        state = self.baneling_scene(
            [((5., 5.), "baneling")],
            {"e0": enemy((7., 5.)), "e1": enemy((7.5, 5.)), "e2": enemy((7., 5.5)),
             "e3": enemy((7.5, 5.5))})
        actions, telemetry = self.step(self.manager(), state)
        self.assertEqual(telemetry["units"]["u0"]["skill"], "burst_engage")
        self.assertEqual(actions[0], 6)   # detonate on adjacent cluster

    def test_low_value_contact_declined(self):
        state = self.baneling_scene([((5., 5.), "baneling")],
                                    {"e0": enemy((9., 9.))})
        actions, telemetry = self.step(self.manager(), state)
        # No cluster worth the trade in reach: never a forced charge at the
        # nearest enemy; hold/approach decisions are scored.
        self.assertEqual(telemetry["units"]["u0"]["skill"], "burst_engage")
        self.assertNotEqual(telemetry["units"]["u0"]["goal"], "e0"
                            ) if actions[0] == 6 else None

    def test_self_consumption_counted_as_cost(self):
        from skills.tactical_v1.skills.burst_engage import BurstEngageSkill
        skill = BurstEngageSkill(SkillManager().config)
        state = self.baneling_scene(
            [((5., 5.), "baneling")], {"e0": enemy((7., 5.))})
        context = Context(build_snapshot(with_questions(state), 1))
        plans = skill.propose(context, TacticalMemory(), Telemetry())
        burst = [p for p in plans if p.commitment.phase == "bursting"]
        if burst:
            loss = [e for e in burst[0].expected_effects if e.kind == "loss"]
            self.assertTrue(any(e.amount > 0 for e in loss))

    def test_disappearance_is_not_success(self):
        # The specialization must never record a successful burst merely
        # because the unit is gone: its effect claims stay heuristic.
        from skills.tactical_v1.skills.burst_engage import BurstEngageSkill
        skill = BurstEngageSkill(SkillManager().config)
        state = self.baneling_scene(
            [((5., 5.), "baneling")], {"e0": enemy((7., 5.))})
        context = Context(build_snapshot(with_questions(state), 1))
        for plan in skill.propose(context, TacticalMemory(), Telemetry()):
            for effect in plan.expected_effects:
                if effect.kind == "damage":
                    self.assertEqual(effect.evidence, "heuristic")
                    self.assertEqual(effect.amount, 0.0)

    def test_no_normal_kiting_for_banelings(self):
        state = self.baneling_scene([((5., 5.), "baneling")],
                                    {"e0": enemy((6., 5.))})
        state["allies"]["u0"]["weapon_cooldown_loops"] = 16.
        state["allies"]["u0"]["health_lost_since_previous_decision"] = 5.
        actions, telemetry = self.step(self.manager(), state)
        self.assertNotEqual(telemetry["units"]["u0"]["skill"], "combat_reposition")


if __name__ == "__main__":
    unittest.main()
