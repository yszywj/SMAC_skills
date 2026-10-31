"""Exported-rule numerical equivalence vs native sklearn GBDT.

Requires scikit-learn — run with a TRAINING python, never the SMACv2
adapter venv:  python tree_skills/tests/test_rule_model.py
Covers: random rows, every split threshold, both float32 neighbours of
each threshold, NaN missing values, extreme inputs, and joint-assignment
equality on candidate groups.
"""

import json
import math
import os
import random
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from sklearn.ensemble import GradientBoostingRegressor

from tree_skills.export.exporter import export_gbr, predict_reference
from tree_skills.export.rule_model import RuleEnsemble
from tree_skills.medivac.features import FEATURE_NAMES, SCHEMA_VERSION

TOLERANCE = 1e-6


def _f32(x):
    return struct.unpack("f", struct.pack("f", x))[0]


def _f32_neighbours(x):
    packed = struct.pack("f", x)
    bits = struct.unpack("I", packed)[0]
    return (_f32(struct.unpack("f", struct.pack("I", bits - 1))[0]),
            _f32(struct.unpack("f", struct.pack("I", bits + 1))[0]))


def make_model(seed=0):
    rng = random.Random(seed)
    X = [[rng.uniform(-3, 3) for _ in FEATURE_NAMES] for _ in range(400)]
    y = [0.5 * row[0] + (1.0 if row[7] > 0.5 else -1.0) * row[8]
         + 0.1 * sum(row[16:20]) + rng.gauss(0, 0.05) for row in X]
    model = GradientBoostingRegressor(n_estimators=32, max_depth=3,
                                      learning_rate=0.1, random_state=seed)
    model.fit(X, y)
    return model, X


class TestRuleModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.X = make_model()
        cls.tmp = tempfile.mkdtemp()
        cls.path = os.path.join(cls.tmp, "model.json")
        cls.meta = export_gbr(cls.model, FEATURE_NAMES, SCHEMA_VERSION, cls.path,
                              meta_extra={"test": True})
        cls.rules = RuleEnsemble.load(cls.path)

    def _assert_close(self, rows):
        for row in rows:
            native = float(predict_reference(self.model, [row])[0])
            exported = self.rules.predict(row)
            self.assertLessEqual(abs(native - exported), TOLERANCE,
                                 f"row {row[:4]}... native={native} exported={exported}")

    def test_random_rows(self):
        rng = random.Random(1)
        self._assert_close([[_f32(rng.uniform(-3, 3)) for _ in FEATURE_NAMES]
                            for _ in range(2000)])

    def test_threshold_boundaries(self):
        thresholds = set()

        def walk(node):
            if node["kind"] == "split":
                thresholds.add(node["threshold"])
                walk(node["left"]); walk(node["right"])
        for tree in self.meta["trees"]:
            walk(tree["root"])
        rng = random.Random(2)
        rows = []
        for threshold in list(thresholds)[:400]:
            for feature in range(len(FEATURE_NAMES)):
                base = [_f32(rng.uniform(-1, 1)) for _ in FEATURE_NAMES]
                for value in (threshold,
                              _f32(threshold),  # cast boundary
                              *_f32_neighbours(_f32(threshold)),
                              math.nextafter(threshold, math.inf),
                              math.nextafter(threshold, -math.inf)):
                    row = list(base)
                    row[feature] = value
                    rows.append(row)
        self._assert_close(rows)

    def test_missing_and_extreme(self):
        # sklearn's classic GBDT cannot predict NaN inputs at all, so NaN
        # semantics are defined ONLY by the exported model (deterministic
        # go-left) and by the feature builder never emitting NaN (missing
        # information is carried by explicit flag features instead).
        rng = random.Random(3)
        rows = []
        for _ in range(300):
            row = [_f32(rng.uniform(-3, 3)) for _ in FEATURE_NAMES]
            row[rng.randrange(len(FEATURE_NAMES))] = float("nan")
            first = self.rules.predict(row)
            second = self.rules.predict(row)
            self.assertEqual(first, second)
            self.assertTrue(math.isfinite(first))
            rows.append(row)
        rows = [r for r in rows if not any(v != v for v in r)]
        rows.extend(([0.0] * len(FEATURE_NAMES), [1e6] * len(FEATURE_NAMES),
                     [-1e6] * len(FEATURE_NAMES)))
        self._assert_close(rows)

    def test_joint_assignment_agreement(self):
        rng = random.Random(4)
        for _ in range(500):
            group = [[_f32(rng.uniform(-3, 3)) for _ in FEATURE_NAMES]
                     for _ in range(rng.randrange(2, 6))]
            native = predict_reference(self.model, group)
            native_top = max(range(len(group)), key=lambda i: native[i])
            exported = [self.rules.predict(r) for r in group]
            exported_top = max(range(len(group)), key=lambda i: exported[i])
            if abs(native[native_top] - exported[exported_top]) > TOLERANCE:
                self.assertEqual(native_top, exported_top)

    def test_format_roundtrip(self):
        with open(self.path, encoding="utf-8") as fh:
            meta = json.load(fh)
        self.assertEqual(meta["kind"], "exported_rules")
        self.assertEqual(meta["schema_version"], SCHEMA_VERSION)
        self.assertEqual(meta["n_trees"], 32)
        self.assertIn("base_score", meta)


if __name__ == "__main__":
    unittest.main()
