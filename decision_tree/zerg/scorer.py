"""Skill-internal scoring interface.

``score_candidates`` is the single entry every scorer uses: one decision
point, one shared state feature vector, one fixed feature schema, many legal
candidates. Models only ever output a scalar preference per candidate id —
never actions. When the model is absent, unsupported at the reached leaf, or
fails, every candidate falls back to the rule score and the original ranking
survives unchanged.

Four scorers are declared (zergling / hydralisk / baneling / joint); each is
a configured view over the same core with its own unit-type routing and its
own model slot. Nothing here trains or requires a model to exist — an
unfitted scorer behaves as the pure rule fallback.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from .candidates import FAMILY_JOINT_PLAN, ZergCandidate
from .features import DEFAULT_SCHEMA, ZergFeatureSchema
from .schema import FeatureLayerMissing, build_model_features, missing_report

Score = Dict[str, Any]
FallbackFn = Callable[[Dict[str, Any], Sequence[ZergCandidate]], Dict[str, float]]

SCORER_NAMES = ("zergling_scorer", "hydralisk_scorer", "baneling_scorer",
                "joint_scorer")

# Units routed to type scorers; anything mixed-type or team-wide goes to the
# joint scorer.
SCORER_UNIT_TYPES = {
    "zergling_scorer": "zergling",
    "hydralisk_scorer": "hydralisk",
    "baneling_scorer": "baneling",
    "joint_scorer": None,
}


def default_fallback(state_features: Dict[str, Any],
                     candidates: Sequence[ZergCandidate]) -> Dict[str, float]:
    """Rule fallback: the evaluator score declared on each candidate, with
    list order as the deterministic tiebreaker. Never invents scores."""
    scores: Dict[str, float] = {}
    for index, candidate in enumerate(candidates):
        value = candidate.rule_score
        scores[candidate.candidate_id] = (
            float(value) if isinstance(value, (int, float)) and
            math.isfinite(float(value)) else -float(index) * 1e-6)
    return scores


@dataclass
class CandidateScores:
    candidate_id: str
    score: float
    supported: bool
    confidence: float
    reason: str
    rule_score: Optional[float] = None
    support_groups: int = 0
    support_samples: int = 0
    missing_count: int = 0
    missing_ratio: float = 0.0
    missing_features: Optional[List[str]] = None

    def to_record(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "score": self.score,
            "supported": self.supported,
            "confidence": self.confidence,
            "reason": self.reason,
            "rule_score": self.rule_score,
            "support_groups": self.support_groups,
            "support_samples": self.support_samples,
            "missing_count": self.missing_count,
            "missing_ratio": self.missing_ratio,
            "missing_features": self.missing_features or [],
        }


def _leaf_confidence(groups: int, min_groups: int) -> float:
    """Deterministic, monotone support confidence in [0, 1]."""
    if min_groups <= 0 or groups <= 0:
        return 0.0
    return min(1.0, groups / (2.0 * min_groups))


class ScorerModel:
    """Loaded scorer model: a threshold-tree ensemble, optionally split into
    per-``candidate_skill`` family bundles. Aggregation is ``mean`` (single /
    bagged trees) or ``weighted_sum`` (boosted trees with learning_rate
    weights and a base value). Missing feature values fall back to the
    recorded training-time imputation; a value with no imputation leaves the
    candidate unsupported."""

    def __init__(self, model_record: Dict[str, Any]):
        from .schema import validate_scorer_model
        self.record = validate_scorer_model(model_record)
        self.scorer = self.record["scorer"]
        self.feature_names = list(self.record["feature_names"])
        self.families: Optional[Dict[str, Any]] = self.record.get("families") \
            or None
        self.trees: List[Dict[str, Any]] = self.record.get("trees") or []
        self.aggregation = self.record.get("aggregation", "mean")
        self.base_value = float(self.record.get("base_value", 0.0))
        self.missing_imputation: Dict[str, float] = dict(
            self.record.get("missing_imputation", {}))
        if self.families:
            for bundle in self.families.values():
                self.missing_imputation.update(
                    bundle.get("missing_imputation", {}) or {})
        self.min_gain = float(self.record.get("min_gain", 0.0))
        self.tie_tolerance = float(self.record.get("tie_tolerance", 1e-9))

    def _bundle_for(self, candidate_skill: Optional[str]
                    ) -> Optional[Dict[str, Any]]:
        if self.families:
            bundle = self.families.get(candidate_skill or "")
            if bundle is None:
                return None
            return {"trees": bundle["trees"],
                    "aggregation": bundle.get("aggregation", "mean"),
                    "base_value": float(bundle.get("base_value", 0.0)),
                    "missing_imputation":
                        bundle.get("missing_imputation", {}) or {}}
        return {"trees": self.trees, "aggregation": self.aggregation,
                "base_value": self.base_value,
                "missing_imputation": self.missing_imputation}

    def estimate(self, flat_features: Dict[str, float],
                 candidate_skill: Optional[str] = None
                 ) -> List[Dict[str, Any]]:
        """Aggregate leaf estimate for one candidate; ``[]`` means the model
        does not support this candidate (unknown family, unrecorded missing
        feature, empty model)."""
        bundle = self._bundle_for(candidate_skill)
        if bundle is None or not bundle["trees"]:
            return []
        imputation = bundle["missing_imputation"]
        leaves: List[Dict[str, Any]] = []
        total_weight = 0.0
        weighted_value = 0.0
        for tree in bundle["trees"]:
            node = tree["root"]
            while "feature" in node:
                value = flat_features.get(node["feature"])
                if value is None or (isinstance(value, float)
                                     and math.isnan(value)):
                    value = imputation.get(node["feature"])
                if value is None:
                    return []
                node = node["left"] if value <= node["threshold"] \
                    else node["right"]
            weight = float(tree.get("weight", 1.0))
            total_weight += weight
            weighted_value += weight * float(node["value"])
            leaves.append(node)
        if bundle["aggregation"] == "mean":
            value = weighted_value / total_weight if total_weight else 0.0
        else:
            value = weighted_value + bundle["base_value"]
        return [{
            "value": value,
            "supported": all(bool(leaf.get("supported", False))
                             for leaf in leaves),
            "groups": min(int(leaf.get("groups", 0)) for leaf in leaves),
            "samples": min(int(leaf.get("samples", 0)) for leaf in leaves),
        }]


def _candidate_flat_features(candidate: ZergCandidate,
                             state_features: Dict[str, Any],
                             schema: ZergFeatureSchema) -> Dict[str, float]:
    """Task 1.1: online rows use the SAME join as offline training."""
    return build_model_features(state_features, candidate.features, schema)


def score_candidates(state_features: Dict[str, Any],
                     candidates: Sequence[ZergCandidate],
                     model: Optional[ScorerModel] = None,
                     fallback: Optional[FallbackFn] = None,
                     schema: ZergFeatureSchema = DEFAULT_SCHEMA,
                     ) -> List[CandidateScores]:
    """Core scoring entry.

    - ``state_features`` is the shared decision-point context (team+skill
      layers; candidate features travel on each candidate itself).
    - ``model`` is an optional loaded :class:`ScorerModel`.
    - ``fallback`` maps candidate_id -> rule score (defaults to the declared
      evaluator scores with stable order tiebreaking).
    Returns one :class:`CandidateScores` row per input candidate, in input
    order. The model NEVER emits actions and NEVER changes legality.
    """
    rule_scores = (fallback or default_fallback)(state_features, candidates)
    results: List[CandidateScores] = []
    if model is None:
        for candidate in candidates:
            rule = rule_scores.get(candidate.candidate_id)
            results.append(CandidateScores(
                candidate_id=candidate.candidate_id,
                score=float(rule) if rule is not None else 0.0,
                supported=False, confidence=0.0,
                reason="no_model_rule_fallback", rule_score=rule))
        return results

    for candidate in candidates:
        rule = rule_scores.get(candidate.candidate_id)
        row = CandidateScores(
            candidate_id=candidate.candidate_id,
            score=float(rule) if rule is not None else 0.0,
            supported=False, confidence=0.0,
            reason="rule_fallback", rule_score=rule)
        if not candidate.legal:
            row.reason = "candidate_not_legal_rule_fallback"
            results.append(row)
            continue
        try:
            flat = _candidate_flat_features(candidate, state_features, schema)
        except FeatureLayerMissing as exc:
            # A whole layer absent is a plumbing error, never imputed away.
            row.reason = f"layer_missing_{exc.layer}_rule_fallback"
            results.append(row)
            continue
        report = missing_report(flat, model.feature_names)
        row.missing_count = report["missing_count"]
        row.missing_ratio = report["missing_ratio"]
        row.missing_features = report["missing_features"]
        try:
            leaves = model.estimate(flat,
                                    candidate_skill=candidate.candidate_skill)
        except Exception:                       # prediction failure -> fallback
            leaves = []
        if not leaves:
            row.reason = "prediction_unsupported_rule_fallback"
            results.append(row)
            continue
        estimate = leaves[0]
        min_groups = int(model.record.get("capacity", {}).get("min_groups", 1))
        row.score = float(estimate["value"])
        row.supported = bool(estimate["supported"])
        row.confidence = (_leaf_confidence(int(estimate["groups"]), min_groups)
                          if estimate["supported"] else 0.0)
        row.support_groups = int(estimate["groups"])
        row.support_samples = int(estimate["samples"])
        row.reason = "model_score" if estimate["supported"] \
            else "leaf_below_support_rule_confidence"
        results.append(row)
    return results


def rank_by_scores(candidates: Sequence[ZergCandidate],
                   scores: Sequence[CandidateScores],
                   min_gain: float = 0.0,
                   tie_tolerance: float = 1e-9) -> List[str]:
    """Preference order over candidate ids.

    Supported model scores first (descending); the rule fallback order
    (original candidate order, rule score descending) fills the rest. Exact
    positive ties keep the earlier rule position — the model can only
    re-order, never invent, candidates. Returns ids only; actions always
    stay bound to the candidate's source plan.
    """
    by_id = {row.candidate_id: row for row in scores}
    rule_position = {candidate.candidate_id: index
                     for index, candidate in enumerate(candidates)}
    supported = [row for row in scores
                 if row.supported and row.score > min_gain + tie_tolerance]
    if supported:
        supported.sort(key=lambda row: (-row.score, rule_position[row.candidate_id]))
    else:
        supported = []
    supported_ids = {row.candidate_id for row in supported}
    rest = [row for row in scores if row.candidate_id not in supported_ids]
    rest.sort(key=lambda row: (
        -(row.rule_score if isinstance(row.rule_score, (int, float))
          else -1e9),
        rule_position[row.candidate_id]))
    return [row.candidate_id for row in supported + rest]


def route_candidates(candidates: Sequence[ZergCandidate],
                     state_features: Dict[str, Any]
                     ) -> Dict[str, List[ZergCandidate]]:
    """Split a decision point's candidates per scorer.

    Single-unit-type skill plans route to their type scorer; mixed rosters
    and joint plan candidates route to the joint scorer. Routing uses the
    decision point's ``unit_types`` map — no unit tags enter features.
    """
    unit_types = state_features.get("unit_types", {}) or {}
    routed: Dict[str, List[ZergCandidate]] = {
        name: [] for name in SCORER_NAMES}
    for candidate in candidates:
        if candidate.candidate_family == FAMILY_JOINT_PLAN:
            routed["joint_scorer"].append(candidate)
            continue
        types = {unit_types.get(actor) for actor in candidate.controlled_actors}
        types.discard(None)
        if len(types) == 1:
            unit_type = types.pop()
            for scorer_name, mapped in SCORER_UNIT_TYPES.items():
                if mapped == unit_type:
                    routed[scorer_name].append(candidate)
                    break
            else:
                routed["joint_scorer"].append(candidate)
        else:
            routed["joint_scorer"].append(candidate)
    return routed


@dataclass
class SkillScorer:
    """A named scorer slot: routing rule + optional loaded model.

    An unfitted scorer is a valid pure-fallback scorer; the interface does
    not require a trained model to exist.
    """
    name: str
    unit_type: Optional[str] = None
    model: Optional[ScorerModel] = None
    fallback: Optional[FallbackFn] = None
    min_gain: float = 0.0
    schema: ZergFeatureSchema = field(default=DEFAULT_SCHEMA)

    def score(self, state_features: Dict[str, Any],
              candidates: Sequence[ZergCandidate]) -> List[CandidateScores]:
        return score_candidates(state_features, candidates, model=self.model,
                                fallback=self.fallback, schema=self.schema)

    def rank(self, candidates: Sequence[ZergCandidate],
             scores: Sequence[CandidateScores]) -> List[str]:
        return rank_by_scores(candidates, scores,
                              min_gain=self.min_gain,
                              tie_tolerance=(self.model.tie_tolerance
                                             if self.model else 1e-9))

    def score_and_rank(self, state_features: Dict[str, Any],
                       candidates: Sequence[ZergCandidate]
                       ) -> Dict[str, Any]:
        scores = self.score(state_features, candidates)
        return {"scores": scores, "ranking": self.rank(candidates, scores)}


def build_default_scorers() -> Dict[str, SkillScorer]:
    """The four declared scorer slots; priority implementation order is
    hydralisk and zergling first, baneling and joint later (baneling keeps
    the slot without a dedicated model in v1)."""
    return {
        "zergling_scorer": SkillScorer("zergling_scorer", "zergling"),
        "hydralisk_scorer": SkillScorer("hydralisk_scorer", "hydralisk"),
        "baneling_scorer": SkillScorer("baneling_scorer", "baneling"),
        "joint_scorer": SkillScorer("joint_scorer", None),
    }


class PairwiseRankModel:
    """Loaded pairwise ranking model (Phase 6.3).

    Training unit = one ordered candidate comparison with the ``a.``/``b.``
    feature layout and a positive margin target. Inference scores a
    decision point by tournament counting: for each ordered pair (i, j),
    score_i += p(i,j) - p(j,i). Deterministic, no action output, same
    fallback semantics as the gain scorers."""

    def __init__(self, model_record: Dict[str, Any]):
        from .schema import validate_ranking_model
        self.record = validate_ranking_model(model_record)
        self.feature_names = list(self.record["feature_names"])
        self.trees: List[Dict[str, Any]] = self.record["trees"]
        self.aggregation = self.record.get("aggregation", "mean")
        self.base_value = float(self.record.get("base_value", 0.0))
        self.missing_imputation: Dict[str, float] = dict(
            self.record.get("missing_imputation", {}))
        self.tie_tolerance = float(self.record.get("tie_tolerance", 1e-9))

    def _compare(self, higher_flat: Dict[str, float],
                 lower_flat: Dict[str, float]) -> Tuple[float, bool]:
        features = {f"a.{k}": v for k, v in higher_flat.items()}
        features.update({f"b.{k}": v for k, v in lower_flat.items()})
        leaves = []
        total = 0.0
        for tree in self.trees:
            node = tree["root"]
            while "feature" in node:
                name = node["feature"]
                value = features.get(name)
                if value is None or (isinstance(value, float)
                                     and math.isnan(value)):
                    value = self.missing_imputation.get(name)
                if value is None:
                    return 0.0, False
                node = node["left"] if value <= node["threshold"] \
                    else node["right"]
            total += float(tree.get("weight", 1.0)) * float(node["value"])
            leaves.append(node)
        if self.aggregation == "mean":
            weights = math.fsum(float(t.get("weight", 1.0))
                                for t in self.trees)
            value = total / weights if weights else 0.0
        else:
            value = total + self.base_value
        supported = all(bool(leaf.get("supported", False)) for leaf in leaves)
        return value, supported

    def score_candidates(self, flat_by_candidate: Dict[str, Dict[str, float]],
                         candidate_ids: Optional[Sequence[str]] = None
                         ) -> Dict[str, Dict[str, Any]]:
        """Tournament scores per candidate id; unsupported pairs contribute
        0 and mark the whole point unsupported."""
        ids = list(candidate_ids or flat_by_candidate)
        scores = {cid: 0.0 for cid in ids}
        supported = True
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                ab, ab_ok = self._compare(flat_by_candidate[a],
                                          flat_by_candidate[b])
                ba, ba_ok = self._compare(flat_by_candidate[b],
                                          flat_by_candidate[a])
                if not (ab_ok and ba_ok):
                    supported = False
                    continue
                scores[a] += ab - ba
                scores[b] += ba - ab
        return {cid: {"score": scores[cid], "supported": supported}
                for cid in ids}
