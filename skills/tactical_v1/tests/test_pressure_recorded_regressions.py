"""Recorded single-target regressions, rescoring only the changed pressure terms.

The rest of each score is the original logged value. This tests the isolated
ranking defect, not a policy replay, optimal action, or counterfactual victory.
"""
import copy
import json
from pathlib import Path
import unittest

from skills.tactical_v1.battle_pressure import joint_pressure
from skills.tactical_v1.context import Context, build_snapshot
from skills.tactical_v1.effects import physical_effects
from skills.tactical_v1.evaluator import Evaluator


class RecordedPressureRegressions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.examples = json.loads(Path(__file__).with_name(
            "pressure_snapshots.json").read_text(encoding="utf-8"))["examples"]

    def verify_local_ranking(self, case):
        selected, alternate = case["selected_actions"], case["alternate_actions"]
        changed = {a: [v, alternate[a]] for a, v in selected.items()
                   if alternate[a] != v}
        self.assertEqual(changed, case["action_difference"])
        self.assertEqual(len(changed), 1)
        self.assertLess(case["alternate_score"]["total"], case["selected_score"]["total"])
        c = Context(build_snapshot(case["observation"], 1))
        evaluator = Evaluator()
        recomputed, pressure_rows = {}, []
        for label, actions in (("selected", selected), ("alternate", alternate)):
            report = physical_effects(c, actions)
            original_report = copy.deepcopy(report)
            pressure = joint_pressure(c, actions, report)
            self.assertEqual(report, original_report)
            self.assertEqual(pressure["pressure_relief"], 0.)
            pressure_rows.append(pressure)
            logged = case[label + "_score"]
            recomputed[label] = (logged["total"] - logged["unsupported_frontline"]
                                - logged["target_threat"]
                                - evaluator.weights["unsupported_frontline"]
                                * pressure["unsupported_exposure"]
                                + evaluator.weights["target_threat"] * evaluator.urgency_value
                                * pressure["pressure_relief"])
        self.assertEqual(pressure_rows[0]["unsupported_actors"],
                         pressure_rows[1]["unsupported_actors"])
        self.assertGreater(recomputed["alternate"], recomputed["selected"])

    def test_6z4s_172_step35_does_not_reward_switching_to_full_health_stalker(self):
        self.verify_local_ranking(self.examples[0])

    def test_2z6s2c_171_step13_does_not_penalize_continued_colossus_focus(self):
        self.verify_local_ranking(self.examples[1])


if __name__ == "__main__":
    unittest.main()
