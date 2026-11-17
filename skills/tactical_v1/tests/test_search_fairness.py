"""Small real budgets must compare later actors and skill families."""
import copy
import unittest

from skills.tactical_v1.context import Context, build_snapshot
from skills.tactical_v1.contracts import Commitment, ResourceClaims
from skills.tactical_v1.coordinator import Coordinator
from skills.tactical_v1.memory import TacticalMemory
from skills.tactical_v1.skills.common import make_plan
from skills.tactical_v1.telemetry import Telemetry
from .fixtures import scene, with_questions


class FixedSkill:
    enabled = True

    def __init__(self, plans):
        self.plans = plans

    def propose(self, *args):
        return self.plans


class SearchFairnessTests(unittest.TestCase):
    def context(self, count=2):
        return Context(build_snapshot(with_questions(scene(
            [((i * 10., 0.), "marine") for i in range(count)])), 1))

    def plan(self, skill, actor, action, destination, goal="point"):
        return make_plan(
            skill, (actor,), goal, {actor: action}, [],
            ResourceClaims(controls=(actor,)), Commitment("moving"), {}, 1, "test",
            goal_kind="position", goal_data={"objective": {
                "kind": "waypoint", "position": list(destination)}})

    def test_many_early_actor_candidates_do_not_hide_last_actor(self):
        context = self.context()
        weak = [self.plan("early", "u0", 5, (20. + i, 0.), f"p{i}")
                for i in range(30)]
        useful = self.plan("late", "u1", 5, (0., 0.))
        coordinator = Coordinator({"search": {"max_plan_evaluations": 6}})
        actions, summary, _ = coordinator.decide(
            context, {"early": FixedSkill(weak), "late": FixedSkill([useful])},
            TacticalMemory(), Telemetry(), 2)
        self.assertEqual(actions[1], 5)
        self.assertEqual(summary["candidate_search"]["by_actor"]["u1"]["screened"], 1)
        self.assertLessEqual(summary["plan_evaluations_used"], 6)

    def test_first_option_of_later_skill_precedes_second_of_earlier_skill(self):
        context = self.context(1)
        weak = [self.plan("a_early", "u0", 5, (20. + i, 0.), f"p{i}")
                for i in range(20)]
        useful = self.plan("z_late", "u0", 4, (20., 0.))
        coordinator = Coordinator({"search": {"max_plan_evaluations": 5}})
        actions, summary, _ = coordinator.decide(context,
            {"a_early": FixedSkill(weak), "z_late": FixedSkill([useful])},
            TacticalMemory(), Telemetry(), 1)
        self.assertEqual(actions, [4])
        self.assertEqual(summary["candidate_search"]["by_skill"]["z_late"]["screened"], 1)
        self.assertLessEqual(summary["plan_evaluations_used"], 5)

    def test_dedup_preserves_different_objectives_and_task_contracts(self):
        coordinator = Coordinator({})
        first = self.plan("move", "u0", 4, (20., 0.))
        duplicate = copy.deepcopy(first)
        duplicate.proposal_id = "different-nonsemantic-id"
        other_goal = self.plan("move", "u0", 4, (-20., 0.))
        other_task = copy.deepcopy(first)
        other_task.proposed_memory_delta = {"kind": "create", "max_loops": 48}
        plans = coordinator._schedule_candidates(
            [first, duplicate, other_goal, other_task], self.context(1))
        self.assertEqual(len(plans), 3)
        self.assertIn(other_goal, plans)
        self.assertIn(other_task, plans)

    def test_duplicate_contracts_do_not_consume_all_screening_opportunities(self):
        context = self.context()
        first = self.plan("repeat", "u0", 5, (20., 0.))
        useful = self.plan("late", "u1", 5, (0., 0.))
        coordinator = Coordinator({"search": {"max_plan_evaluations": 5}})
        actions, summary, _ = coordinator.decide(context,
            {"repeat": FixedSkill([copy.deepcopy(first) for _ in range(100)]),
             "late": FixedSkill([useful])}, TacticalMemory(), Telemetry(), 2)
        self.assertEqual(actions[1], 5)
        self.assertEqual(summary["candidate_search"]["generated"], 101)
        self.assertEqual(summary["candidate_search"]["deduplicated"], 2)
        self.assertEqual(summary["candidate_search"]["unexamined"], 0)

    def test_frontier_preserves_distinct_deadlines_while_reusing_present_score(self):
        context = self.context(1)
        coordinator = Coordinator({})
        memory = TacticalMemory()
        short = self.plan("move", "u0", 4, (20., 0.))
        short.commitment = Commitment("moving", max_loops=16)
        short.proposed_memory_delta = {"kind": "create", "skill": "move",
            "actors": ["u0"], "goal": "point", "max_loops": 16}
        long = copy.deepcopy(short)
        long.commitment = Commitment("moving", max_loops=64)
        long.proposed_memory_delta["max_loops"] = 64
        budget = [4]
        entries = [coordinator.evaluate_once(context, [p], {"u0": 4}, memory, budget, 1)
                   for p in (short, long)]
        self.assertTrue(all(entries))
        self.assertEqual(budget[0], 3, "present scoring should still share a cache")
        frontier = coordinator._dedup_trim(entries, memory)
        self.assertEqual(len(frontier), 2)
        self.assertEqual({e.plans[0].commitment.max_loops for e in frontier}, {16, 64})


if __name__ == "__main__":
    unittest.main()
