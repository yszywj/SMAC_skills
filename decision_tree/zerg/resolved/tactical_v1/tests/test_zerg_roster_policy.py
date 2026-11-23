import unittest
from types import SimpleNamespace

from skills.tactical_v1.context import Context, build_snapshot
from skills.tactical_v1.skill_manager import SkillManager
from skills.tactical_v1.zerg_roster_policy import (
    ZERGLING_SKILLS, is_zerg_roster, ling_candidate, zerg_fallback_action,
)
from .fixtures import enemy, scene, with_questions


def roster_scene(unit_types):
    allies = [((0., float(i)), name) for i, name in enumerate(unit_types)]
    enemies = {"e0": enemy((1., 0.), "zergling"), "e1": enemy((3., 1.), "hydralisk")}
    return scene(allies, enemies, map_name="10gen_zerg")


class ZergRosterPolicyTests(unittest.TestCase):
    def test_controlled_roster_alone_determines_scope(self):
        for names, applies in [
                (["zergling"], True), (["baneling"], True), (["hydralisk"], True),
                (["zergling", "hydralisk", "baneling"], True), (["stalker"], False),
                (["marine", "marauder"], False), (["hydralisk", "zealot"], False)]:
            with self.subTest(names=names):
                payload = with_questions(roster_scene(names))
                self.assertEqual(is_zerg_roster(Context(build_snapshot(payload, 1))), applies)

    def test_ling_group_skills_require_ling_actors_without_specialist_provider(self):
        state = roster_scene(["zergling", "zergling", "hydralisk", "baneling"])
        context = Context(build_snapshot(with_questions(state), 1))
        self.assertEqual(ZERGLING_SKILLS, {
            "cohesive_search", "precombat_rally", "coordinated_engage", "focus_fire",
            "local_formation", "combat_reposition", "disengage_reenter", "regroup"})
        for skill in ZERGLING_SKILLS:
            candidate = SimpleNamespace(primary_skill=skill, specialization_provider=None,
                                        controlled_actors=("u0", "u1"))
            self.assertIs(ling_candidate(context, candidate), candidate)
            for denied in ("u2", "u3"):
                candidate.controlled_actors = ("u0", denied)
                self.assertIsNone(ling_candidate(context, candidate))
        candidate.controlled_actors = ("u0",)
        candidate.primary_skill = "combat_rejoin"
        self.assertIsNone(ling_candidate(context, candidate))
        candidate.primary_skill = "focus_fire"
        candidate.specialization_provider = "burst_engage"
        self.assertIsNone(ling_candidate(context, candidate))

    def test_manager_enforces_unit_skill_permissions_in_real_decision(self):
        state = roster_scene(["zergling", "hydralisk", "baneling"])
        payload = with_questions(state)
        context = Context(build_snapshot(payload, 1))
        manager = SkillManager()
        actions, summary = manager.step(payload, 3)
        manager.commit_final_actions(actions)
        self.assertIn(summary["units"]["u0"]["skill"], ZERGLING_SKILLS | {"baseline"})
        self.assertEqual(summary["units"]["u1"]["skill"], "focus_fire")
        self.assertEqual(actions[1], context.attack_action("u1", "e1"))
        self.assertEqual(summary["units"]["u2"]["skill"], "baseline")
        self.assertEqual(actions[2], zerg_fallback_action(context, "u2"))
        for actor, unit in context.snapshot.allies.items():
            self.assertIn(actions[unit.index], context.legal_actions(actor))
