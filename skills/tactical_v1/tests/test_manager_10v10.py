"""Offline 10v10 action/commit checks for the real Adaptive manager."""
import copy
import unittest

from skills.tactical_v1.context import Context, build_snapshot
from skills.tactical_v1.skill_manager import SkillManager
from skills.tactical_v1.validator import validate_actions
from .fixtures import enemy, heal_candidates, scene, with_questions


class Manager10v10Tests(unittest.TestCase):
    def test_three_races_have_complete_legal_actions_and_idempotent_commit(self):
        rosters = {
            "protoss": ["zealot"] * 3 + ["stalker"] * 5 + ["colossus"] * 2,
            "terran": ["marine"] * 4 + ["marauder"] * 4 + ["medivac"] * 2,
            "zerg": ["zergling"] * 4 + ["hydralisk"] * 4 + ["baneling"] * 2,
        }
        for race, names in rosters.items():
            with self.subTest(race=race):
                state = scene(
                    [((float(i % 5), float(i // 5) * 3), name)
                     for i, name in enumerate(names)],
                    enemies={f"e{i}": enemy((8. + i % 5, float(i // 5) * 3))
                             for i in range(10)}, map_name=f"10gen_{race}")
                for actor, unit in state["allies"].items():
                    if unit["type"] == "medivac":
                        unit["energy"] = 50.
                        for relation in state["relations"][actor].values():
                            relation["smac_attack_available"] = False
                    else:
                        unit["health"] = 30.
                for actor, unit in state["allies"].items():
                    if unit["type"] == "medivac":
                        heal_candidates(state, actor)
                manager = SkillManager()
                for step in range(3):
                    state["game_loop"] = 80 + 8 * step
                    payload = with_questions(copy.deepcopy(state))
                    actions, _ = manager.step(payload, 10)
                    context = Context(build_snapshot(payload, step + 1))
                    self.assertEqual(len(actions), 10)
                    self.assertEqual(validate_actions(actions, context, 10), [])
                    manager.commit_final_actions(actions)
                    self.assertEqual(manager.memory.last_committed_actions,
                                     {f"u{i}": a for i, a in enumerate(actions)})
                    for instance in manager.memory.instances.values():
                        self.assertGreater(instance.executed_steps, 0)
                    committed = copy.deepcopy(manager.memory.__dict__)
                    repeated, _ = manager.step(payload, 10)
                    manager.commit_final_actions(repeated)
                    self.assertEqual(repeated, actions)
                    self.assertEqual(manager.memory.__dict__, committed)


if __name__ == "__main__":
    unittest.main()
