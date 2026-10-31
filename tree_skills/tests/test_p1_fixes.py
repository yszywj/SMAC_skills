"""P1-4 regression tests: pair distance per-healer, threat direction,
matching reference consistency, >8-healer strict rejection, history flags."""

import os
import random
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from skills.current.skills import heal_assign
from tree_skills.medivac.features import (FEATURE_NAMES, FeatureSchema,
                                          build_pair_features, _threat_flags)
from tree_skills.medivac.matching import forced_matching


def state_with_two_healers(d0, d1):
    """u4 at distance d0 from the target, u5 at distance d1."""
    allies = {
        "u0": {"health": 25, "max_health": 45, "type": "marine", "alive": True,
               "is_flying": False, "position": [0.0, 0.0],
               "health_lost_since_previous_decision": 10},
        "u4": {"health": 150, "max_health": 150, "type": "medivac", "alive": True,
               "is_flying": True, "position": [1.0, 0.0], "energy": 100, "orders": []},
        "u5": {"health": 150, "max_health": 150, "type": "medivac", "alive": True,
               "is_flying": True, "position": [3.0, 0.0], "energy": 100, "orders": []},
    }
    return {"allies": allies, "visible_enemies": {},
            "relations": {a: {} for a in allies},
            "heal_candidates": {"u4": {"6": {"target": "u0", "distance": d0,
                                              "missing_health": 20}},
                                "u5": {"6": {"target": "u0", "distance": d1,
                                             "missing_health": 20}}},
            "movement_candidates": {}}


def capture_context(state):
    captured = {}

    def capture(healer, target, context):
        captured.setdefault("context", context)
        captured.setdefault("healers", set()).add(healer)
        return 1.0

    heal_assign.assign_healing(state, 6, pair_scorer=capture, pair_scoring="on")
    return captured["context"]


class TestP1Fixes(unittest.TestCase):
    def test_pair_distance_is_per_healer(self):
        """Same target, two healers at clearly different distances: the
        pair_distance feature must differ between H0 and H1 (previously both
        got the first healer's distance)."""
        schema = FeatureSchema.current()
        context = capture_context(state_with_two_healers(1.0, 5.0))
        d_u4 = build_pair_features("u4", "u0", context)[FEATURE_NAMES.index("pair_distance")]
        d_u5 = build_pair_features("u5", "u0", context)[FEATURE_NAMES.index("pair_distance")]
        self.assertAlmostEqual(d_u4, 1.0, places=6)
        self.assertAlmostEqual(d_u5, 5.0, places=6)

    def test_threat_direction(self):
        """A medivac inside an enemy's range: in_enemy_weapon_range=True must
        mark it threatened even though its OWN weapon_range_status is
        'no_compatible_weapon' (no weapon says nothing about enemy threat)."""
        state = state_with_two_healers(1.0, 1.0)
        state["relations"]["u4"]["e0"] = {
            "distance": 3.0, "in_enemy_weapon_range": True,
            "enemy_weapon_range_status": "in_range",
            "weapon_range_status": "no_compatible_weapon",  # own side, ignored
            "in_weapon_range": False}
        threatened, known, missing = _threat_flags(state, "u4")
        self.assertEqual(threatened, 1)
        self.assertEqual(missing, 0)
        # enemy relation with unknown range status stays missing, not "safe"
        state["relations"]["u5"]["e0"] = {"distance": 3.0}
        threatened, known, missing = _threat_flags(state, "u5")
        self.assertEqual((threatened, known, missing), (0, 0, 1))

    def test_history_flags_via_injection(self):
        schema = FeatureSchema.current()
        state = state_with_two_healers(1.0, 1.0)
        captured = {}

        def capture(healer, target, context):
            captured.setdefault("ctx", context)
            return 1.0

        # with histories supplied through the injection layer
        contexts = []

        def grab(healer, target, context):
            contexts.append(context)
            return 1.0

        heal_assign.assign_healing(state, 6, pair_scorer=grab,
                                   pair_scoring="on",
                                   pair_histories={"u0": [0.6, 0.8, 0.7, 0.9]})
        row = build_pair_features("u4", "u0", contexts[-1])
        self.assertEqual(row[FEATURE_NAMES.index("target_history1_missing")], 0.0)
        self.assertAlmostEqual(row[FEATURE_NAMES.index("target_net_health_change_1")], -0.2, places=6)
        # without histories: flagged missing, value 0 (identifiable, not fake)
        heal_assign.assign_healing(state, 6, pair_scorer=grab, pair_scoring="on")
        row = build_pair_features("u4", "u0", contexts[-1])
        self.assertEqual(row[FEATURE_NAMES.index("target_history1_missing")], 1.0)
        self.assertEqual(row[FEATURE_NAMES.index("target_net_health_change_1")], 0.0)

    def test_matching_reference_consistency(self):
        """tree_skills reference forced matching == the injection-layer copy
        on random inputs (same tie handling)."""
        rng = random.Random(11)
        for _ in range(300):
            n = rng.randrange(1, 6)
            healers = [f"u{i}" for i in range(n)]
            targets = [f"t{j}" for j in range(rng.randrange(n, n + 4))]
            edges = {h: sorted(((round(rng.uniform(-3, 3), 3), t)
                                for t in targets if rng.random() < 0.8),
                               key=lambda r: (-r[0], r[1])) or [(0.0, targets[0])]
                     for h in healers}
            a = heal_assign._forced_matching(sorted(healers, key=lambda h: int(h[1:])), edges)
            b = forced_matching(healers, edges)
            self.assertEqual(a, b)

    def test_over_eight_healers_rejected(self):
        healers = [f"u{i}" for i in range(9)]
        edges = {h: [(1.0, f"t{j}") for j in range(20)] for h in healers}
        with self.assertRaises(ValueError):
            forced_matching(healers, edges)
        with self.assertRaises(ValueError):
            heal_assign._forced_matching(healers, edges)

    def test_infeasible_returns_none(self):
        self.assertIsNone(forced_matching(["u0"], {"u0": []}))


if __name__ == "__main__":
    unittest.main()
