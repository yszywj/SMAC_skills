"""Feature schema invariants: fixed length, missing-info flags, renumbering
and candidate-order invariance, schema-version rejection."""

import copy
import random
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from skills.current.skills import heal_assign
from tree_skills.medivac.features import (FEATURE_NAMES, SCHEMA_VERSION,
                                          FeatureSchema, build_pair_features)


def _state(seed=0, prefix="u"):
    rng = random.Random(seed)
    allies = {
        f"{prefix}0": {"health": 30, "max_health": 45, "type": "marine",
                       "alive": True, "is_flying": False, "position": [0.0, 0.0],
                       "health_lost_since_previous_decision": 8},
        f"{prefix}1": {"health": 40, "max_health": 45, "type": "marauder",
                       "alive": True, "is_flying": False, "position": [1.0, 0.0],
                       "health_lost_since_previous_decision": 4},
        f"{prefix}2": {"health": 25, "max_health": 45, "type": "marine",
                       "alive": True, "is_flying": False, "position": [2.0, 0.0],
                       "health_lost_since_previous_decision": 15},
        f"{prefix}4": {"health": 150, "max_health": 150, "type": "medivac",
                       "alive": True, "is_flying": True, "position": [0.5, 1.0],
                       "energy": 180, "orders": []},
    }
    state = {"allies": allies, "visible_enemies": {}, "relations": {a: {} for a in allies},
             "heal_candidates": {}, "movement_candidates": {}}
    state["heal_candidates"][f"{prefix}4"] = {
        str(6 + i): {"target": f"{prefix}{i}",
                     "distance": 1.5 + 0.5 * i,
                     "missing_health": allies[f"{prefix}{i}"]["max_health"] - allies[f"{prefix}{i}"]["health"]}
        for i in range(3)}
    return state


def _capture_context(state, previous=None):
    """Run the real assign_healing with a capturing scorer to obtain the
    exact context dict the injection layer provides."""
    captured = {}

    def capture(healer, target, context):
        captured.setdefault("context", context)
        return 1.0

    heal_assign.assign_healing(copy.deepcopy(state), 6,
                               previous_assignments=previous,
                               pair_scorer=capture, pair_scoring="on")
    return captured["context"]


class TestFeatures(unittest.TestCase):
    def test_fixed_length_and_schema(self):
        schema = FeatureSchema.current()
        self.assertEqual(len(FEATURE_NAMES), 33)
        context = _capture_context(_state())
        row = build_pair_features("u4", "u0", context, schema)
        self.assertEqual(len(row), schema.size)

    def test_schema_version_rejection(self):
        with self.assertRaises(ValueError):
            FeatureSchema({"version": 99, "names": FEATURE_NAMES})
        with self.assertRaises(ValueError):
            FeatureSchema({"version": 1, "names": FEATURE_NAMES[:-1]})

    def test_missing_history_flagged_not_zero(self):
        context = _capture_context(_state())
        row_with = build_pair_features("u4", "u0", dict(context, histories={"u0": [0.7, 0.9, 0.8, 0.95]}))
        row_without = build_pair_features("u4", "u0", context)
        # without history: missing flags set, changes reported as 0 AND flagged
        self.assertEqual(row_without[FEATURE_NAMES.index("target_history1_missing")], 1.0)
        self.assertEqual(row_without[FEATURE_NAMES.index("target_history3_missing")], 1.0)
        self.assertEqual(row_with[FEATURE_NAMES.index("target_history1_missing")], 0.0)
        self.assertAlmostEqual(row_with[FEATURE_NAMES.index("target_net_health_change_1")], -0.2, places=6)

    def test_renumbering_invariance(self):
        """Renumbering allies must not change a semantically identical pair."""
        schema = FeatureSchema.current()
        ctx_a = _capture_context(_state(prefix="u"))
        # semantic twin under a different numbering: u0->u7, u1->u3, u2->u9, u4->u5
        state_b = _state(prefix="u")
        ren = {"u0": "u7", "u1": "u3", "u2": "u9", "u4": "u5"}
        state_b["allies"] = {ren[a]: u for a, u in state_b["allies"].items()}
        state_b["relations"] = {ren[a]: r for a, r in state_b["relations"].items()}
        state_b["heal_candidates"] = {ren[h]: {aid: dict(e, target=ren[e["target"]])
                                               for aid, e in rows.items()}
                                      for h, rows in state_b["heal_candidates"].items()}
        ctx_b = _capture_context(state_b)
        row_a = build_pair_features("u4", "u0", ctx_a, schema)
        row_b = build_pair_features("u5", "u7", ctx_b, schema)
        self.assertEqual(row_a, row_b)

    def test_candidate_order_invariance(self):
        """Permuting heal_candidate dict order must not change features."""
        schema = FeatureSchema.current()
        state = _state()
        items = list(state["heal_candidates"]["u4"].items())
        random.Random(42).shuffle(items)
        state["heal_candidates"]["u4"] = dict(items)
        context = _capture_context(state)
        row = build_pair_features("u4", "u0", context, schema)
        base = build_pair_features("u4", "u0", _capture_context(_state()), schema)
        self.assertEqual(row, base)

    def test_more_candidates_only_more_rows(self):
        state = _state()
        state["allies"]["u6"] = dict(state["allies"]["u1"], health=20,
                                     health_lost_since_previous_decision=10)
        state["heal_candidates"]["u4"]["str"] = None
        del state["heal_candidates"]["u4"]["str"]
        state["heal_candidates"]["u4"]["9"] = {"target": "u6", "distance": 3.0,
                                               "missing_health": 25}
        schema = FeatureSchema.current()
        context = _capture_context(state)
        row = build_pair_features("u4", "u0", context, schema)
        self.assertEqual(len(row), schema.size)


if __name__ == "__main__":
    unittest.main()
