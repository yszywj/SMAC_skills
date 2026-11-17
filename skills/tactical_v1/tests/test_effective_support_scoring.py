"""Shared readiness cannot earn arbitrary credit by adding passive cover."""
import unittest

from skills.tactical_v1.context import Context, build_snapshot
from skills.tactical_v1.contracts import Commitment, ResourceClaims
from skills.tactical_v1.evaluator import Evaluator
from skills.tactical_v1.skills.common import make_plan
from .fixtures import enemy, scene, with_questions


class EffectiveSupportScoringTests(unittest.TestCase):
    def test_one_joiner_reward_does_not_grow_with_passive_cover_size(self):
        state = scene([((0., 0.), "stalker"), ((10., 0.), "stalker"),
                       ((10., 0.), "stalker"), ((10., 0.), "stalker")],
                      enemies={"e0": enemy((14., 0.), "stalker")}, attack_range=6.)
        context = Context(build_snapshot(with_questions(state), 1))
        # Receivers must actually fire; idle distant bodies no longer promise
        # cover. Adding extra receivers still cannot multiply joiner progress.
        actions = {"u0": 4, "u1": 6, "u2": 6, "u3": 6}
        evaluator = Evaluator()
        values = []
        for supporters in (["u1"], ["u1", "u2", "u3"]):
            actors = ("u0", *supporters)
            candidate = make_plan("combat_rejoin", actors, "join", {a: actions[a] for a in actors},
                [], ResourceClaims(controls=actors), Commitment("rejoining"), {}, 1, "test",
                atomic=True, goal_kind="area", goal_data={"joiners": ["u0"],
                    "supporters": supporters, "support_radius": 6.,
                    "objective": {"kind": "combat_rejoin"}})
            score = evaluator.score_joint(context, [candidate], actions)[1]["link"]
            values.append(score)
        self.assertGreater(values[0], 0.)
        self.assertAlmostEqual(values[0], values[1])
        self.assertLessEqual(values[1], evaluator.weights["link"])

    def test_idle_cost_requires_a_real_available_advance_not_native_auto_fire(self):
        evaluator = Evaluator()
        for enemy_x, expected_idle in ((6.5, True), (4., False), (30., False)):
            state = scene([((0., 0.), "stalker")],
                          enemies={"e0": enemy((enemy_x, 0.), "stalker")}, attack_range=6.)
            context = Context(build_snapshot(with_questions(state), 1))
            breakdown = evaluator.score_joint(context, [], {"u0": 1})[1]
            self.assertEqual(breakdown["combat_idle"] < 0., expected_idle)


if __name__ == "__main__":
    unittest.main()
