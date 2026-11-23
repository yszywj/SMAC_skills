"""Skill-to-skill selection tree.

``select_skill_plan`` chooses among the COMPLETE legal joint plans the
Coordinator already produced (the selected plan plus its recorded legal
alternatives). It is a preference re-ranking, nothing more:

- it never constructs, merges, splits or repairs plans;
- it never touches resource claims, dependencies, execution duties or the
  final action arbitration — those all remain in tactical_v1's Coordinator,
  Validator and ``commit_final_actions`` path;
- when no candidate is supported (no model, unsupported leaf, feature
  mismatch, prediction failure) the original rule selection survives verbatim.

Only actually-legal candidates visible at the decision point may be ranked;
anything flagged illegal is excluded before the model is consulted.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from .candidates import FAMILY_JOINT_PLAN, ZergCandidate
from .features import DEFAULT_SCHEMA, ZergFeatureSchema
from .scorer import FallbackFn, ScorerModel, score_candidates

SKILL_PLAN_VOCABULARY = (
    # Every skill that may appear as a joint-plan primary skill. The actual
    # candidate set at a decision point is whatever the Coordinator surfaced
    # AND marked legal — vocabulary membership alone never makes a skill
    # selectable.
    "focus_fire", "coordinated_engage", "combat_reposition",
    "disengage_reenter", "regroup", "local_formation", "cohesive_search",
    "precombat_rally", "combat_rejoin", "ranged_fire_formation",
    "splash_fire", "heal_support", "burst_engage")


def legal_joint_plans(candidates: Sequence[ZergCandidate]
                      ) -> List[ZergCandidate]:
    """Filter to complete joint-plan candidates that passed the structural
    legality pre-check. The authoritative validation still belongs to
    tactical_v1's validator on the final action list."""
    return [candidate for candidate in candidates
            if candidate.candidate_family == FAMILY_JOINT_PLAN
            and candidate.legal]


def rule_preference_order(legal_plans: Sequence[ZergCandidate]
                          ) -> List[str]:
    """The original Coordinator preference: the rule-selected plan first
    (the candidate built from ``chosen_plans``), then recorded alternatives
    by their declared joint score, stable by input order on ties."""
    def key(index_and_candidate):
        index, candidate = index_and_candidate
        score = candidate.rule_score
        score_value = float(score) if isinstance(score, (int, float)) else -1e18
        is_chosen = candidate.source_plan_id == candidate.candidate_id \
            and index == 0
        return (0 if is_chosen else 1, -score_value, index)
    return [candidate.candidate_id for _, candidate
            in sorted(enumerate(legal_plans), key=key)]


def selector_fallback(state_features: Dict[str, Any],
                      candidates: Sequence[ZergCandidate]) -> Dict[str, float]:
    """Rule fallback scores derived from the Coordinator's own ranking."""
    order = rule_preference_order(candidates)
    return {candidate_id: -float(rank) for rank, candidate_id
            in enumerate(order)}


def select_skill_plan(global_features: Dict[str, Any],
                      legal_plans: Sequence[ZergCandidate],
                      model: Optional[ScorerModel] = None,
                      fallback: Optional[FallbackFn] = None,
                      min_gain: float = 0.0,
                      tie_tolerance: float = 1e-9,
                      schema: ZergFeatureSchema = DEFAULT_SCHEMA
                      ) -> Dict[str, Any]:
    """Choose one complete legal joint plan.

    ``legal_plans`` are joint ``ZergCandidate`` records (see
    ``candidates.joint_candidates_from_summary``); the first entry is expected
    to be the rule-selected plan. Returns a decision record:

    - ``selected_candidate_id`` / ``selected_actions``: the winning plan.
      Identical to the rule choice whenever the model cannot support a
      strictly-preferred alternative.
    - ``changed``: False unless a SUPPORTED model candidate with a strictly
      positive preference replaced the rule choice.
    - ``scores`` / ``ranking``: full score rows and preference order.

    Applying a changed selection still requires re-running tactical_v1's
    final validation before the actions leave the manager (runtime duty).
    """
    plans = legal_joint_plans(legal_plans)
    rule_order = rule_preference_order(plans)
    base = {
        "ranking_rule": rule_order,
        "scores": [],
        "reason": "no_candidates",
        "selected_candidate_id": None,
        "selected_actions": None,
        "changed": False,
    }
    if not plans:
        return base
    by_id = {candidate.candidate_id: candidate for candidate in plans}
    base["selected_candidate_id"] = rule_order[0]
    base["selected_actions"] = dict(by_id[rule_order[0]].proposed_actions)
    base["reason"] = "rule_choice"
    if len(plans) == 1:
        base["reason"] = "single_candidate"
        return base

    scores = score_candidates(global_features, plans,
                              model=model,
                              fallback=fallback or selector_fallback,
                              schema=schema)
    base["scores"] = [row.to_record() for row in scores]

    supported = [row for row in scores
                 if row.supported and row.score > min_gain + tie_tolerance]
    if model is None or not supported:
        base["reason"] = "no_supported_candidate_rule_fallback"
        return base
    rule_position = {candidate_id: index for index, candidate_id
                     in enumerate(rule_order)}
    supported.sort(key=lambda row: (-row.score,
                                    rule_position.get(row.candidate_id, 10**9)))
    winner = supported[0]
    if winner.candidate_id == rule_order[0]:
        base["reason"] = "model_confirms_rule_choice"
        return base
    base["selected_candidate_id"] = winner.candidate_id
    base["selected_actions"] = dict(by_id[winner.candidate_id].proposed_actions)
    base["changed"] = True
    base["reason"] = "model_preferred_legal_alternative"
    base["pending_validation"] = True
    return base
