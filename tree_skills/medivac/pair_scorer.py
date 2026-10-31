"""Pair scorers for the medivac treatment-preference channel.

A pair scorer is a callable ``scorer(healer, target_id, context) -> float``
used exclusively through the default-off injection in
``skills/current/skills/heal_assign.py::_apply_pair_scorer``. It ranks the
LEGAL (environment-gated, order-reservation-filtered) candidates of a
healer that the original logic already decided to participate in healing.
Negative scores are meaningful; callers must not shift, clip or abs them.

Context dict keys (read-only):
  state                -- build_request snapshot (allies, relations, ...)
  urgency              -- original heal urgency per target id
  target_pool          -- {target: {healer: {action_id, distance, ...}}}
  previous_assignments -- observed heal orders used as previous assignments
  original_edges       -- {healer: [(original_score, target), ...]}
"""

import json
import math
import os

from .features import FeatureSchema, build_pair_features


class OriginalPairScorer:
    """A-plug scorer: returns the original edge score unchanged.

    Used to prove the injection channel itself does not change behaviour:
    ``assign_healing(..., pair_scorer=OriginalPairScorer(), pair_scoring="on")``
    must reproduce the default-off assignments bit-for-bit (see
    tests/test_pair_scorer_channel.py).
    """

    name = "original"

    def __call__(self, healer, target_id, context):
        for score, target in context["original_edges"].get(healer, []):
            if target == target_id:
                return score
        raise KeyError(f"no original edge for {healer}->{target_id}")


class TreePairScorer:
    """Scores a legal (healer, candidate) pair with a trained tree model.

    Accepts either an exported rule model (models/medivac_v1, no training
    dependency at all) or a native sklearn GradientBoostingRegressor bundle.
    The feature schema version must match exactly; unknown schema versions
    are rejected rather than silently reinterpreted.
    """

    name = "tree"

    def __init__(self, model_dir):
        self.model_dir = model_dir
        meta_path = os.path.join(model_dir, "model.json")
        with open(meta_path, "r", encoding="utf-8") as fh:
            self.meta = json.load(fh)
        if self.meta.get("kind") == "exported_rules":
            from ..export.rule_model import RuleEnsemble
            self._rules = RuleEnsemble(self.meta)
            self._native = None
        elif self.meta.get("kind") == "sklearn_gbr":
            try:
                import joblib
            except ImportError as exc:
                raise ImportError("native sklearn bundle requires joblib; "
                                  "prefer the exported rules deployment") from exc
            self._rules = None
            self._native = joblib.load(os.path.join(model_dir, "model.joblib"))
        else:
            raise ValueError(f"unknown model kind: {self.meta.get('kind')!r}")
        self.schema = FeatureSchema(self.meta["feature_schema"])
        self.version = self.meta["schema_version"]

    def __call__(self, healer, target_id, context):
        row = build_pair_features(healer, target_id, context, self.schema)
        if self._rules is not None:
            value = self._rules.predict(row)
        else:
            value = float(self._native.predict([row])[0])
        if not math.isfinite(value):
            raise ValueError(f"non-finite score for {healer}->{target_id}")
        return value


def make_pair_scorer(mode, model_dir=None):
    """Build the scorer for an integration mode.

    off            -> None (original behaviour)
    shadow         -> OriginalPairScorer plus recording is handled by the
                      injection layer; pass the real scorer to observe its
                      proposals without acting on them.
    teacher-score  -> requires the teacher inference server (blocked until
                      weights pass audit); raises with BLOCKED_TEACHER_WEIGHTS.
    tree-score     -> TreePairScorer on an exported/native model dir.
    """
    if mode == "off":
        return None
    if mode in ("a-plug", "a_plug"):
        return OriginalPairScorer()
    if mode == "teacher-score":
        from ..teacher.preflight import teacher_status
        status = teacher_status()
        if status.get("state") != "ready":
            raise RuntimeError(
                "BLOCKED_TEACHER_WEIGHTS: teacher inference unavailable "
                f"({status.get('state')}); see teacher/TEACHER_MANIFEST.json")
        from ..teacher.inference import TeacherPairScorer  # heavy import guard
        return TeacherPairScorer()
    if mode == "tree-score":
        if not model_dir:
            raise ValueError("tree-score requires --model-dir")
        return TreePairScorer(model_dir)
    raise ValueError(f"unknown integration mode: {mode!r}")
