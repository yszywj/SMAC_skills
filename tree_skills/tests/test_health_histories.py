"""HealthHistories decision-step semantics tests (P0 §1.1)."""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from tree_skills.integration.pair_runtime import HealthHistories
from tree_skills.medivac.features import FEATURE_NAMES, build_pair_features


def allies(**ratios_alive):
    return ratios_alive


def snap(ratio, alive=True, max_health=45.0):
    return {"health": ratio * max_health, "max_health": max_health, "alive": alive}


class TestHealthHistories(unittest.TestCase):
    def test_one_entry_per_decision_step_even_when_unchanged(self):
        h = HealthHistories()
        for loop, ratio in enumerate([0.8, 0.8, 0.8, 0.6]):
            h.observe("ep1", loop, {"u0": snap(ratio)})
        self.assertEqual(h["u0"], [0.6, 0.8, 0.8, 0.8])  # newest first

    def test_net_change_over_three_steps(self):
        h = HealthHistories()
        for loop, ratio in enumerate([0.8, 0.8, 0.8, 0.6]):
            h.observe("ep1", loop, {"u0": snap(ratio)})
        ctx = {"histories": h, "state": {"allies": {
            "u0": {"health": 27, "max_health": 45, "type": "marine", "alive": True},
            "u4": {"health": 150, "max_health": 150, "type": "medivac", "alive": True,
                   "energy": 100, "orders": []}}},
            "relations": {}, "urgency": {"u0": 0.5}, "target_pool": {"u0": {"u4": {"distance": 2.0}}},
            "previous_assignments": {}, "original_edges": {"u4": [(1.0, "u0")]}}
        row = build_pair_features("u4", "u0", ctx)
        self.assertAlmostEqual(row[FEATURE_NAMES.index("target_net_health_change_3")],
                               0.6 - 0.8, places=6)
        self.assertEqual(row[FEATURE_NAMES.index("target_history3_missing")], 0.0)
        self.assertAlmostEqual(row[FEATURE_NAMES.index("target_net_health_change_1")],
                               0.6 - 0.8, places=6)

    def test_same_loop_idempotent(self):
        h = HealthHistories()
        h.observe("ep1", 7, {"u0": snap(0.8)})
        h.observe("ep1", 7, {"u0": snap(0.5)})  # duplicate loop: ignored
        self.assertEqual(h["u0"], [0.8])

    def test_dead_unit_records_zero(self):
        h = HealthHistories()
        h.observe("ep1", 0, {"u0": snap(0.4)})
        h.observe("ep1", 1, {"u0": snap(0.4, alive=False)})
        self.assertEqual(h["u0"][0], 0.0)

    def test_invalid_max_health_keeps_previous(self):
        h = HealthHistories()
        h.observe("ep1", 0, {"u0": snap(0.4)})
        h.observe("ep1", 1, {"u0": {"health": 30, "max_health": 0, "alive": True}})
        self.assertEqual(h["u0"], [0.4, 0.4])

    def test_reset_and_truncation(self):
        h = HealthHistories(length=3)
        for loop in range(6):
            h.observe("ep1", loop, {"u0": snap(0.5 + 0.01 * loop)})
        self.assertEqual(len(h["u0"]), 3)
        h.reset()
        self.assertEqual(len(h), 0)
        h.observe("ep2", 0, {"u0": snap(0.9)})  # fresh episode starts clean
        self.assertEqual(h["u0"], [0.9])

    def test_new_episode_reuses_loop_numbers_safely(self):
        h = HealthHistories()
        h.observe("ep1", 5, {"u0": snap(0.8)})
        h.observe("ep2", 5, {"u0": snap(0.2)})  # PairRuntime resets per episode
        h.reset()
        h.observe("ep2", 5, {"u0": snap(0.2)})
        self.assertEqual(h["u0"], [0.2])


if __name__ == "__main__":
    unittest.main()
