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


# -- Phase 5: full-plan replacement -----------------------------------------

def apply_selected_plan(selected, manager, context=None, memory=None,
                        base_plans=None, base_actions=None):
    """Apply a model-selected COMPLETE joint plan with full semantics.

    The selected candidate must carry its real source ``CandidatePlan``
    (``candidate.plan``) from the CURRENT decision step. Replacement:

    1. releases the base plans that control the selected plan's actors
       (atomic plans are released whole, never split);
    2. inserts the selected CandidatePlan untouched — ResourceClaims,
       Dependencies, Commitment, controlled actors and atomicity all
       travel with the plan object, never a bare action copy;
    3. merges actions and re-runs the tactical_v1 Validator on the COMPLETE
       final action list, then the bounded ``repair`` (whole-plan release +
       baseline refill) when problems remain;
    4. falls back to the untouched BASE plans/actions when replacement or
       repair cannot produce a clean plan set — Tree Score never bypasses
       the Validator, and Memory commit stays entirely inside the
       SkillManager's own surviving-plan logic.

    Returns a decision record; ``applied`` is True only when the final
    validator run reported zero problems on the replaced plan set.
    """
    result = {
        "applied": False,
        "fallback_to_base": False,
        "reason": None,
        "replaced_plan_ids": [],
        "validator_problems": [],
        "repair_log": [],
        "final_plans": base_plans,
        "final_actions": base_actions,
        "commit_status": "base_unchanged",
    }
    if getattr(selected, "candidate_family", None) != FAMILY_JOINT_PLAN:
        result["reason"] = "not_a_joint_plan_candidate"
        return result
    plan = getattr(selected, "plan", None)
    if plan is None:
        # No real plan object available (summary-level candidate): full-plan
        # replacement cannot preserve semantics, so it is refused.
        result["reason"] = "no_source_plan_object_refused"
        result["fallback_to_base"] = True
        return result
    snapshot = getattr(context, "snapshot", None)
    if snapshot is None and manager is not None:
        pending = getattr(manager, "_pending_commit", None)
        snapshot = pending[0] if pending else None
    if snapshot is not None and getattr(plan, "snapshot_id", None) \
            != getattr(snapshot, "snapshot_id", None):
        result["reason"] = "plan_not_from_current_decision_step"
        result["fallback_to_base"] = True
        return result
    if base_plans is None or base_actions is None:
        if manager is None:
            result["reason"] = "missing_base_plan_set"
            return result
        pending = getattr(manager, "_pending_commit", None)
        cache = getattr(manager, "_cache", None)
        base_plans = list(pending[1]) if pending else (list(cache[3])
                                                       if cache else [])
        cached_actions = cache[1] if cache else {}
        if isinstance(cached_actions, list):
            cached_actions = {f"u{i}": action for i, action
                              in enumerate(cached_actions)}
        base_actions = dict(cached_actions)

    validator = _validator_module(manager)
    evaluator = getattr(manager, "evaluator", None)
    skills = getattr(manager, "skills", None)
    memory = memory if memory is not None else getattr(manager, "memory",
                                                       None)
    if validator is None or evaluator is None:
        result["reason"] = "validator_or_evaluator_unavailable"
        return result

    actors = set(getattr(plan, "controlled_actors", ()) or ())
    kept = []
    for base_plan in base_plans:
        if actors & set(getattr(base_plan, "controlled_actors", ()) or ()):
            result["replaced_plan_ids"].append(
                str(getattr(base_plan, "proposal_id", "")))
            continue
        kept.append(base_plan)
    new_plans = kept + [plan]
    new_actions = dict(base_actions)
    new_actions.update(dict(getattr(plan, "current_actions", {}) or {}))

    try:
        report = evaluator.effects(context, new_actions)
        problems = validator.complete_and_validate(
            new_plans, context, new_actions, report=report, memory=memory)
        if problems:
            new_plans, new_actions, repair_log = validator.repair(
                new_plans, context, new_actions, memory, evaluator,
                skills=skills)
            result["repair_log"] = list(repair_log)
            report = evaluator.effects(context, new_actions)
            problems = validator.complete_and_validate(
                new_plans, context, new_actions, report=report, memory=memory)
    except Exception as exc:               # any failure -> full BASE fallback
        result["reason"] = f"replacement_error:{type(exc).__name__}"
        result["validator_problems"] = [str(exc)]
        result["fallback_to_base"] = True
        result["final_plans"], result["final_actions"] = \
            base_plans, base_actions
        return result

    result["validator_problems"] = problems
    if problems:
        result["reason"] = "validation_failed_full_base_fallback"
        result["fallback_to_base"] = True
        result["final_plans"], result["final_actions"] = \
            base_plans, base_actions
        return result
    result.update({
        "applied": True,
        "reason": "replacement_validated",
        "final_plans": new_plans,
        "final_actions": new_actions,
        "commit_status": "pending_manager_commit",
    })
    return result


def _validator_module(manager):
    """The Validator module of the package the manager actually came from."""
    import importlib
    manager_module = getattr(type(manager), "__module__", "")
    if not manager_module:
        return None
    package = manager_module.rsplit(".", 1)[0]
    try:
        return importlib.import_module(f"{package}.validator")
    except Exception:
        return None


def apply_complete_plan(selected, manager, context=None, memory=None,
                        base_plans=None, base_actions=None):
    """Task 4.2: apply a model-selected COMPLETE branch.

    ``selected`` must be a :class:`candidates.CompletePlanCandidate` whose
    ``plans`` are the Coordinator's REAL complete plan objects for the
    CURRENT snapshot (``plan_objects_available`` true). The whole branch is
    replaced as a unit — plans, actions, ResourceClaims, Dependencies,
    Commitments, atomicity and execution duties all travel with the objects;
    there is no partial-plan splice that could orphan another unit's task.
    The complete Validator re-runs on the final action list, bounded repair
    runs when problems remain, and any failure returns the untouched BASE.
    """
    result = {
        "applied": False,
        "fallback_to_base": False,
        "reason": None,
        "replaced_plan_ids": [],
        "validator_problems": [],
        "repair_log": [],
        "final_plans": base_plans,
        "final_actions": base_actions,
        "commit_status": "base_unchanged",
    }
    from .candidates import CompletePlanCandidate
    if not isinstance(selected, CompletePlanCandidate):
        result["reason"] = "not_a_complete_plan_candidate"
        return result
    if not selected.plan_objects_available or not selected.plans:
        # Summary-level candidate (actions only): full-plan replacement
        # cannot preserve semantics, so it is refused outright.
        result["reason"] = "no_complete_plan_objects_refused"
        result["fallback_to_base"] = True
        return result
    if not selected.legal:
        result["reason"] = "candidate_not_legal"
        result["fallback_to_base"] = True
        return result

    snapshot = getattr(context, "snapshot", None)
    if snapshot is None and manager is not None:
        pending = getattr(manager, "_pending_commit", None)
        snapshot = pending[0] if pending else None
    if snapshot is not None \
            and selected.snapshot_id != getattr(snapshot, "snapshot_id", None):
        result["reason"] = "plan_not_from_current_decision_step"
        result["fallback_to_base"] = True
        return result

    if base_plans is None:
        pending = getattr(manager, "_pending_commit", None)
        cache = getattr(manager, "_cache", None)
        base_plans = list(pending[1]) if pending else (list(cache[3])
                                                       if cache else [])
    if base_actions is None:
        cache = getattr(manager, "_cache", None)
        cached = cache[1] if cache else {}
        if isinstance(cached, list):
            cached = {f"u{i}": a for i, a in enumerate(cached)}
        base_actions = dict(cached)

    result["replaced_plan_ids"] = sorted({str(getattr(p, "proposal_id", ""))
                                          for p in base_plans}
                                         - set(selected.source_plan_ids))
    validator = _validator_module(manager)
    evaluator = getattr(manager, "evaluator", None)
    skills = getattr(manager, "skills", None)
    memory = memory if memory is not None else getattr(manager, "memory",
                                                       None)
    if validator is None or evaluator is None:
        result["reason"] = "validator_or_evaluator_unavailable"
        return result

    new_plans = list(selected.plans)
    new_actions = dict(selected.actions)
    try:
        report = evaluator.effects(context, new_actions)
        problems = validator.complete_and_validate(
            new_plans, context, new_actions, report=report, memory=memory)
        if problems:
            new_plans, new_actions, repair_log = validator.repair(
                new_plans, context, new_actions, memory, evaluator,
                skills=skills)
            result["repair_log"] = list(repair_log)
            report = evaluator.effects(context, new_actions)
            problems = validator.complete_and_validate(
                new_plans, context, new_actions, report=report, memory=memory)
    except Exception as exc:               # any failure -> full BASE fallback
        result["reason"] = f"replacement_error:{type(exc).__name__}"
        result["validator_problems"] = [str(exc)]
        result["fallback_to_base"] = True
        result["final_plans"], result["final_actions"] = \
            base_plans, base_actions
        return result

    result["validator_problems"] = problems
    if problems:
        result["reason"] = "validation_failed_full_base_fallback"
        result["fallback_to_base"] = True
        result["final_plans"], result["final_actions"] = \
            base_plans, base_actions
        return result
    result.update({
        "applied": True,
        "reason": "replacement_validated",
        "final_plans": new_plans,
        "final_actions": new_actions,
        "commit_status": "pending_manager_commit",
    })
    return result
