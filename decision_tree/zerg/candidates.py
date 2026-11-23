"""Unified Zerg candidate extraction on top of tactical_v1 proposals.

The authoritative candidate generator is tactical_v1 itself: skills produce
``CandidatePlan`` objects through ``propose(...)``, the Coordinator composes
complete legal joint plans, the Validator and action arbitration decide what
finally executes. This module never re-implements any of that — it only
ADAPTS already-generated plans into a uniform, serialisable ``ZergCandidate``
record with fixed-schema features, so scorers and datasets can treat
skill-internal alternatives and whole joint plans the same way.

Two candidate families:

- ``skill_internal``: the legal alternative plans one skill proposed for the
  actors it is responsible for (scorer input).
- ``joint_plan``: complete legal plan sets the Coordinator produced (the
  selected one plus its recorded runner-up alternatives) — selector input.

Every candidate keeps the source plan's identity in ``source_plan_id``; the
final legality verdict always belongs to tactical_v1's validator, never to
the pre-check here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .features import (
    DEFAULT_SCHEMA, ZergFeatureSchema, extract_candidate_features)

CANDIDATE_SCHEMA = "zerg_candidate_v1"
COMPLETE_PLAN_SCHEMA = "zerg_complete_plan_candidate_v1"
FAMILY_SKILL_INTERNAL = "skill_internal"
FAMILY_JOINT_PLAN = "joint_plan"


def _claims_dict(claims: Any) -> Dict[str, Any]:
    if claims is None:
        return {}
    return {
        "controls": list(getattr(claims, "controls", ()) or ()),
        "fire": [[enemy, float(amount or 0.0)] for enemy, amount
                 in (getattr(claims, "fire", ()) or ())],
        "healing": [[ally, float(amount or 0.0)] for ally, amount
                    in (getattr(claims, "healing", ()) or ())],
        "space": [
            {"actor": getattr(claim, "actor", ""),
             "position": list(getattr(claim, "position", ()) or ()),
             "window_loops": int(getattr(claim, "window_loops", 0) or 0),
             "collision_class": getattr(claim, "collision_class", ""),
             "radius": float(getattr(claim, "radius", 0.0) or 0.0),
             "slot_id": getattr(claim, "slot_id", None)}
            for claim in (getattr(claims, "space", ()) or ())],
    }


def _dependencies_list(dependencies: Any) -> List[Dict[str, Any]]:
    rows = []
    for dependency in dependencies or ():
        rows.append({"kind": getattr(dependency, "kind", ""),
                     "refs": list(getattr(dependency, "refs", ()) or ())})
    return rows


@dataclass
class ZergCandidate:
    """Uniform candidate record; ``plan`` keeps the live source object and is
    excluded from serialisation and equality. ``instance_id``/``task_key``
    carry the skill-internal task identity used to build A-class proposal
    groups (never model features)."""
    candidate_id: str
    candidate_family: str
    candidate_skill: str
    controlled_actors: Tuple[str, ...]
    target: Optional[str]
    phase: Optional[str]
    current_actions: Dict[str, int]
    proposed_actions: Dict[str, int]
    features: Dict[str, float]
    resource_claims: Dict[str, Any]
    dependencies: List[Dict[str, Any]]
    atomic: bool
    legal: bool
    rejection_reason: Optional[str]
    source_plan_id: str
    rule_score: Optional[float] = None
    instance_id: Optional[str] = None
    task_key: Optional[str] = None
    plan: Any = field(default=None, repr=False, compare=False)

    def to_record(self) -> Dict[str, Any]:
        return {
            "schema": CANDIDATE_SCHEMA,
            "candidate_id": self.candidate_id,
            "candidate_family": self.candidate_family,
            "candidate_skill": self.candidate_skill,
            "controlled_actors": list(self.controlled_actors),
            "target": self.target,
            "phase": self.phase,
            "current_actions": dict(self.current_actions),
            "proposed_actions": dict(self.proposed_actions),
            "features": dict(self.features),
            "resource_claims": self.resource_claims,
            "dependencies": self.dependencies,
            "atomic": bool(self.atomic),
            "legal": bool(self.legal),
            "rejection_reason": self.rejection_reason,
            "source_plan_id": self.source_plan_id,
            "rule_score": self.rule_score,
            "instance_id": self.instance_id,
            "task_key": self.task_key,
        }


@dataclass
class CompletePlanCandidate:
    """Task 4.1: a B-class candidate bound to the Coordinator's REAL
    complete plan objects for the current decision step.

    ``plans`` must be the full plan set of one Coordinator branch (BASE from
    the manager's pending commit; a Coordinator alternative when the runner
    exposes those objects). ``plan_objects_available=False`` marks
    summary-level candidates (actions only) — they may be scored and
    recorded but can NEVER trigger a Tree Score replacement, and a set of
    per-skill local plans must never be presented as one joint branch.
    """
    candidate_id: str
    snapshot_id: int
    plans: List[Any]
    actions: Dict[str, int]
    resource_claims: Dict[str, Any]
    dependencies: List[Dict[str, Any]]
    execution_duties: Dict[str, Any]
    rule_score: Optional[float]
    candidate_features: Dict[str, float]
    legal: bool
    source_plan_ids: List[str]
    candidate_skill: str = "joint"
    plan_objects_available: bool = False

    @property
    def atomic(self) -> bool:
        return True        # a complete branch is replaced as a whole

    def to_record(self) -> Dict[str, Any]:
        return {
            "schema": COMPLETE_PLAN_SCHEMA,
            "candidate_id": self.candidate_id,
            "snapshot_id": self.snapshot_id,
            "actions": dict(self.actions),
            "resource_claims": self.resource_claims,
            "dependencies": self.dependencies,
            "execution_duties": self.execution_duties,
            "rule_score": self.rule_score,
            "candidate_features": dict(self.candidate_features),
            "legal": bool(self.legal),
            "source_plan_ids": list(self.source_plan_ids),
            "candidate_skill": self.candidate_skill,
            "plan_objects_available": bool(self.plan_objects_available),
            "n_plans": len(self.plans),
        }


def check_candidate_legality(proposed_actions: Dict[str, int],
                             snapshot, context) -> Tuple[bool, Optional[str]]:
    """Cheap structural pre-check BEFORE scoring.

    Only membership checks (alive actor, action inside the snapshot's legal
    menu, one action per controlled actor). This is a scoring guard, NOT the
    authoritative validation — full plan validation, resource arbitration,
    dependency resolution and repair stay in tactical_v1's
    ``validator``/Coordinator and always run on the final plan set.
    """
    allies = getattr(snapshot, "allies", {}) or {}
    for actor, action in proposed_actions.items():
        unit = allies.get(actor)
        if unit is None:
            return False, f"unknown_actor:{actor}"
        if not getattr(unit, "alive", False):
            return False, f"dead_actor:{actor}"
        legal = None
        if context is not None:
            try:
                legal = context.legal_actions(actor)
            except Exception:
                legal = None
        if legal is None:
            legal = (getattr(snapshot, "legal", {}) or {}).get(actor)
        if legal is not None and action not in legal:
            return False, f"illegal_action:{actor}:{action}"
    return True, None


def candidate_from_plan(plan, snapshot, context=None,
                        family: str = FAMILY_SKILL_INTERNAL,
                        peer_plans: Optional[Sequence[Any]] = None,
                        duty: Optional[str] = None,
                        rule_score: Optional[float] = None,
                        schema: ZergFeatureSchema = DEFAULT_SCHEMA
                        ) -> ZergCandidate:
    """Adapt one tactical_v1 ``CandidatePlan`` into a ``ZergCandidate``."""
    proposed = dict(getattr(plan, "current_actions", {}) or {})
    previous = getattr(snapshot, "previous_action_ids", {}) or {}
    legal, reason = check_candidate_legality(proposed, snapshot, context)
    features = extract_candidate_features(
        plan, snapshot, context=context, peer_plans=peer_plans, duty=duty)
    commitment = getattr(plan, "commitment", None)
    goal = getattr(plan, "goal", None)
    target = goal if isinstance(goal, str) and goal else None
    return ZergCandidate(
        candidate_id=str(getattr(plan, "proposal_id", "")),
        candidate_family=family,
        candidate_skill=str(getattr(plan, "primary_skill", "") or "baseline"),
        controlled_actors=tuple(getattr(plan, "controlled_actors", ()) or ()),
        target=target,
        phase=getattr(commitment, "phase", None) if commitment is not None else None,
        current_actions=dict(previous),
        proposed_actions=proposed,
        features=features,
        resource_claims=_claims_dict(getattr(plan, "claims", None)),
        dependencies=_dependencies_list(getattr(plan, "dependencies", None)),
        atomic=bool(getattr(plan, "atomic", False)),
        legal=legal,
        rejection_reason=reason,
        source_plan_id=str(getattr(plan, "proposal_id", "")),
        rule_score=rule_score,
        instance_id=(str(getattr(plan, "instance_id", None))
                     if getattr(plan, "instance_id", None) else None),
        task_key=(str(getattr(plan, "task_key", None))
                  if getattr(plan, "task_key", None) else None),
        plan=plan)


def skill_internal_candidates(plans: Sequence[Any], snapshot, context=None,
                              duties: Optional[Dict[str, Any]] = None,
                              rule_scores: Optional[Dict[str, float]] = None,
                              schema: ZergFeatureSchema = DEFAULT_SCHEMA
                              ) -> List[ZergCandidate]:
    """A-type: legal alternative plans of the skill(s) that already own the
    task. ``plans`` must come from a real ``propose(...)`` call — this adapter
    does not synthesise battle plans.

    ``duties`` maps actor -> duty row (from the step summary's
    ``execution_duties.actors``); the dominant duty of a candidate is the
    highest-priority non-generic duty among its controlled actors.
    """
    duties = duties or {}
    scores = rule_scores or {}
    priority = ("rescue", "intercept", "burst", "fire", "crossfire",
                "support", "entry")
    candidates: List[ZergCandidate] = []
    for plan in plans:
        duty = None
        for actor in getattr(plan, "controlled_actors", ()) or ():
            row = duties.get(actor)
            kind = row.get("kind") if isinstance(row, dict) else None
            if kind and kind in priority:
                if duty is None or priority.index(kind) < priority.index(duty):
                    duty = kind
        candidates.append(candidate_from_plan(
            plan, snapshot, context=context,
            family=FAMILY_SKILL_INTERNAL, peer_plans=plans, duty=duty,
            rule_score=scores.get(str(getattr(plan, "proposal_id", ""))),
            schema=schema))
    return candidates


def _alternative_row(row: Dict[str, Any]) -> Dict[str, Any]:
    return row if isinstance(row, dict) else {}


def joint_candidates_from_summary(summary: Dict[str, Any], snapshot,
                                  context=None,
                                  payload: Optional[Dict[str, Any]] = None,
                                  schema: ZergFeatureSchema = DEFAULT_SCHEMA
                                  ) -> List[ZergCandidate]:
    """B-type: complete legal plan sets from one coordinator decision.

    ``summary`` is the step summary returned by ``SkillManager.step``:
    ``chosen_plans`` + ``actions`` describe the rule-selected joint plan;
    ``top_alternates`` rows carry the runner-up complete plans the
    Coordinator already built and validated during its beam search. Nothing
    here invents plans; candidates are exactly the legal options the
    Coordinator surfaced.
    """
    candidates: List[ZergCandidate] = []
    actions = dict(summary.get("actions", {}) or {})
    chosen = summary.get("chosen_plans", []) or []
    primary = chosen[0] if chosen else {}
    chosen_id = str(primary.get("id") or "joint_rule_choice")
    previous = getattr(snapshot, "previous_action_ids", {}) or {}

    def _actor_legal_ok(actor: str, action: int) -> bool:
        if payload is None:
            return True
        questions = payload.get("questions", {}) or {}
        question = questions.get(actor)
        if not isinstance(question, dict):
            return True
        criteria = question.get("criteria")
        return True if criteria is None else action in criteria

    chosen_score = primary.get("score")
    candidates.append(ZergCandidate(
        candidate_id=chosen_id,
        candidate_family=FAMILY_JOINT_PLAN,
        candidate_skill=str(primary.get("skill") or "joint"),
        controlled_actors=tuple(primary.get("actors") or ()),
        target=primary.get("goal") or None,
        phase=None,
        current_actions=dict(previous),
        proposed_actions=actions,
        features=extract_candidate_features(
            _JointPlanProxy(primary, actions), snapshot, context=context,
            duty=None),
        resource_claims={},
        dependencies=[],
        atomic=True,
        legal=all(_actor_legal_ok(f"u{i}", action)
                  for i, action in actions.items()),
        rejection_reason=None,
        source_plan_id=chosen_id,
        rule_score=float(chosen_score) if isinstance(chosen_score,
                                                     (int, float)) else None,
        plan=None))

    seen = {chosen_id}
    for row in summary.get("top_alternates", []) or []:
        row = _alternative_row(row)
        alt_id = str(row.get("id") or "")
        if not alt_id or alt_id in seen:
            continue
        seen.add(alt_id)
        alt_actions = dict(row.get("actions", {}) or {})
        if not alt_actions:
            continue
        legal = all(_actor_legal_ok(f"u{i}", action)
                    for i, action in alt_actions.items())
        score = row.get("score")
        candidates.append(ZergCandidate(
            candidate_id=alt_id,
            candidate_family=FAMILY_JOINT_PLAN,
            candidate_skill=str(row.get("skill") or "joint"),
            controlled_actors=tuple(row.get("actors") or ()),
            target=None,
            phase=None,
            current_actions=dict(previous),
            proposed_actions=alt_actions,
            features=extract_candidate_features(
                _JointPlanProxy(row, alt_actions), snapshot,
                context=context, duty=None),
            resource_claims={},
            dependencies=[],
            atomic=True,
            legal=legal,
            rejection_reason=None if legal else "alternative_not_fully_legal",
            source_plan_id=alt_id,
            rule_score=float(score) if isinstance(score, (int, float)) else None,
            plan=None))
    return candidates


class _JointPlanProxy:
    """Minimal plan-shaped view so ``extract_candidate_features`` can also
    describe a summary-level joint candidate (actions + declared score)."""

    def __init__(self, row: Dict[str, Any], actions: Dict[str, int]):
        self.primary_skill = str(row.get("skill") or "joint")
        self.controlled_actors = tuple(row.get("actors") or ())
        self.goal = row.get("goal")
        self.goal_kind = None
        self.current_actions = dict(actions)
        self.expected_effects = ()
        self.commitment = None
        self.features = {}


def collect_step_candidates(summary: Dict[str, Any], snapshot, context=None,
                            skill_plans: Optional[Dict[str, Sequence[Any]]] = None,
                            payload: Optional[Dict[str, Any]] = None,
                            schema: ZergFeatureSchema = DEFAULT_SCHEMA
                            ) -> Dict[str, List[ZergCandidate]]:
    """One decision point -> both candidate families.

    ``skill_plans`` maps skill name -> the plans that skill proposed this
    step (only needed for A-type scoring; pass None for joint-only runs).
    """
    duties = ((summary.get("execution_duties") or {}).get("actors") or {})
    result: Dict[str, List[ZergCandidate]] = {}
    for skill_name, plans in (skill_plans or {}).items():
        result[f"{FAMILY_SKILL_INTERNAL}:{skill_name}"] = \
            skill_internal_candidates(plans, snapshot, context=context,
                                      duties=duties, schema=schema)
    result[FAMILY_JOINT_PLAN] = joint_candidates_from_summary(
        summary, snapshot, context=context, payload=payload, schema=schema)
    return result
