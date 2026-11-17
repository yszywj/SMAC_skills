"""Run totals survive episode reset without counting repeated reads twice."""
import unittest

from skills.tactical_v1.skill_manager import SkillManager
from .fixtures import enemy, scene, with_questions


class RunStatisticsTests(unittest.TestCase):
    def test_budget_and_episode_totals_survive_reset(self):
        manager = SkillManager()
        manager.coordinator.max_evaluations = 2
        state = scene([((0., 0.), "stalker")], loop=80,
                      enemies={"e0": enemy((12., 0.), "stalker")})
        for loop in (80, 88):
            state["game_loop"] = loop
            actions, _ = manager.step(with_questions(state), 1)
            manager.commit_final_actions(actions)
        self.assertEqual(manager.stats["search_budget_events"], 2)
        manager.reset()
        state["game_loop"] = 80
        payload = with_questions(state)
        actions, summary = manager.step(payload, 1)
        manager.commit_final_actions(actions)
        self.assertEqual(summary["search_budget_events"], 1)
        self.assertEqual(manager.stats["search_budget_events"], 3)
        self.assertEqual(manager.stats["episodes"], 2)
        before = dict(manager.stats)
        manager.step(payload, 1)
        manager.commit_final_actions(actions)
        self.assertEqual(manager.stats, before)


if __name__ == "__main__":
    unittest.main()
