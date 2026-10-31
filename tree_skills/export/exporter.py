"""Deterministic exporter: sklearn GradientBoostingRegressor -> if/else rules.

Run in the TRAINING environment (needs scikit-learn); the produced
model.json runs anywhere via export.rule_model (stdlib only).

Numerical-equivalence contract (tests/test_rule_model.py):
- sklearn casts input to the estimator's native dtype (float32 for default
  GBDT) before threshold comparisons. The exported ``input_dtype`` records
  that cast and the evaluator applies the SAME cast; thresholds keep full
  float64 precision and are never truncated.
- Boundary semantics: cast(value) <= threshold goes left, NaN goes left
  (matching sklearn's missing-value handling), and both float32 neighbours
  of every threshold are tested in the equivalence suite.
"""

import json

import numpy as np


def native_input_dtype(model):
    """The dtype sklearn will cast raw input to before predicting.

    sklearn's tree ensemble ``predict`` validates input with the tree module
    DTYPE (np.float32) regardless of generic estimator tags — verified
    empirically in tests/test_rule_model.py: predicting with float64 input
    whose values sit on threshold midpoints disagrees with float32-cast
    input. GBDT therefore always reports float32 here.
    """
    from sklearn.ensemble import GradientBoostingRegressor
    if isinstance(model, GradientBoostingRegressor):
        return np.float32
    return np.float64


def _stages(model):
    """Yield one fitted DecisionTreeRegressor per boosting stage across
    sklearn versions (estimators_ is (n,1) of lists in <=1.6, flat in 1.7+)."""
    for entry in getattr(model, "estimators_", []):
        if hasattr(entry, "tree_"):
            yield entry
        else:
            for tree in entry:
                yield tree


def export_gbr(model, feature_names, schema_version, path, meta_extra=None):
    """Serialize a fitted GradientBoostingRegressor into model.json."""
    trees = [{"root": _export_tree(tree.tree_)} for tree in _stages(model)]
    meta = {
        "kind": "exported_rules",
        "schema_version": schema_version,
        "feature_schema": {"version": schema_version, "names": list(feature_names)},
        "base_score": float(model.init_.constant_[0][0]),
        "learning_rate": float(model.learning_rate),
        "input_dtype": "float32" if native_input_dtype(model) is np.float32 else "float64",
        "n_trees": len(trees),
        "trees": trees,
    }
    meta.update(meta_extra or {})
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1)
    return meta


def _export_tree(sk_tree):
    def node(i):
        if sk_tree.children_left[i] == -1:  # leaf
            return {"kind": "leaf", "value": float(sk_tree.value[i][0][0])}
        return {
            "kind": "split",
            "feature": int(sk_tree.feature[i]),
            "threshold": float(sk_tree.threshold[i]),
            "missing_left": True,  # sklearn criterion sends NaN left
            "left": node(sk_tree.children_left[i]),
            "right": node(sk_tree.children_right[i]),
        }
    return node(0)


def predict_reference(model, rows):
    """Reference prediction exactly as sklearn computes it, including the
    estimator's native input-dtype cast (used by equivalence tests)."""
    return model.predict(np.asarray(rows, dtype=np.float64).astype(native_input_dtype(model)))
