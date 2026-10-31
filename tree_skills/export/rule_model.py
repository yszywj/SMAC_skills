"""Exported if/else rule model: runtime representation and evaluator.

Deployment path (models/medivac_v1) must run without PyTorch, MARL-GPT,
scikit-learn or any training dependency — only this JSON format and the
standard library. The exported tree ensemble keeps full-precision split
thresholds, missing-value go-left/right flags, `<=` boundaries, per-tree
learning-rate scaling (applied once) and the initial base score.

Format (model.json):
{
  "kind": "exported_rules",
  "schema_version": 1,
  "feature_schema": {"version": 1, "names": [...]},
  "base_score": float,
  "learning_rate": float,
  "trees": [ { "root": node }, ... ]
}
node := {"kind": "split", "feature": int, "threshold": float,
         "missing_left": bool, "left": node, "right": node}
       | {"kind": "leaf", "value": float}
"""

import json
import math
import struct


def _cast(value, dtype):
    if dtype == "float32":
        return struct.unpack("f", struct.pack("f", value))[0]
    return value


class RuleEnsemble:
    """Deterministic evaluator for the exported rule format."""

    def __init__(self, meta):
        if meta.get("kind") != "exported_rules":
            raise ValueError("not an exported_rules model")
        self.base_score = float(meta["base_score"])
        self.learning_rate = float(meta["learning_rate"])
        self.trees = meta["trees"]
        self.dtype = meta.get("input_dtype", "float64")
        self.n_features = len(meta["feature_schema"]["names"])

    def _tree_value(self, node, row):
        while node["kind"] == "split":
            value = row[node["feature"]]
            go_left = (node["missing_left"] if value != value
                       else _cast(value, self.dtype) <= node["threshold"])
            node = node["left"] if go_left else node["right"]
        return node["value"]

    def predict(self, row):
        if len(row) != self.n_features:
            raise ValueError(f"row has {len(row)} features, model expects {self.n_features}")
        total = self.base_score
        for tree in self.trees:
            total += self.learning_rate * self._tree_value(tree["root"], row)
        if not math.isfinite(total):
            raise ValueError("non-finite prediction")
        return total

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as fh:
            return cls(json.load(fh))
