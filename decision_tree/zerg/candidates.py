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

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .features import (
    DEFAULT_SCHEMA, ZergFeatureSchema, extract_candidate_features)
from .schema import IDENTIFIER

CANDIDATE_SCHEMA = "zerg_candidate_v1"
COMPLETE_PLAN_SCHEMA = "zerg_complete_plan_candidate_v1"
FAMILY_SKILL_INTERNAL = "skill_internal"
FAMILY_JOINT_PLAN = "joint_plan"


def _record_id(source_id: str) -> str:
    # Local-engagement plans can contain '/' between SC2 tags. Preserve
    # the original in source_plan_id, and keep join keys schema-safe.
    if not source_id or IDENTIFIER.fullmatch(source_id):
        return source_id
    return "plan_sha256_" + hashlib.sha256(source_id.encode()).hexdigest()


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
        candidate_id=_record_id(str(getattr(plan, "proposal_id", ""))),
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
        actor = str(actor)
        question = questions.get(actor if actor.startswith("u") else f"u{actor}")
        if not isinstance(question, dict):
            return True
        criteria = question.get("criteria")
        return criteria is None or action in criteria or str(action) in criteria

    chosen_score = primary.get("score")
    candidates.append(ZergCandidate(
        candidate_id=_record_id(chosen_id),
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
        legal=all(_actor_legal_ok(i, action)
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
        legal = all(_actor_legal_ok(i, action)
                    for i, action in alt_actions.items())
        score = row.get("score")
        candidates.append(ZergCandidate(
            candidate_id=_record_id(alt_id),
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


# -- V1.3 Phase 2: complete-branch candidates from the Coordinator cache ----

def build_branch_candidates(manager, snapshot, summary: Dict[str, Any],
                            max_branches: int = 8
                            ) -> List["CompletePlanCandidate"]:
    """Task 2.1/2.2: export the Coordinator's OWN complete legal branches as
    :class:`CompletePlanCandidate` records.

    Source of truth: ``manager.coordinator._cache`` — ``evaluate_once``
    caches ONLY branches that passed ``validate_actions``/``validate_plans``
    at scoring time, and every entry carries THIS step's real
    ``CandidatePlan`` objects (plans / actions / claims / dependencies /
    commitment / atomic flags all travel on the objects). Nothing is
    re-generated, no skill re-runs, no fabricated plans from JSON actions:
    branches with no cached objects simply do not exist here. The selected
    branch (actions == summary actions) is the BASE.
    """
    pending = getattr(manager, "_pending_commit", None)
    if not pending:
        return []
    snapshot_id = int(getattr(snapshot, "snapshot_id", 0) or 0)
    final_actions = dict(summary.get("actions", {}) or {})
    chosen = (summary.get("chosen_plans") or [{}])[0] \
        if summary.get("chosen_plans") else {}
    branches: List[CompletePlanCandidate] = []

    def _entry_to_branch(candidate_id, plans, actions, score, execution):
        claims: Dict[str, Any] = {"controls": [], "fire": [], "healing": [],
                                  "space": []}
        dependencies: List[Dict[str, Any]] = []
        for plan in plans:
            plan_claims = getattr(plan, "claims", None)
            if plan_claims is not None:
                claims["controls"].extend(
                    getattr(plan_claims, "controls", ()) or ())
                claims["fire"].extend(getattr(plan_claims, "fire", ()) or ())
                claims["healing"].extend(
                    getattr(plan_claims, "healing", ()) or ())
                claims["space"].extend(getattr(plan_claims, "space", ()) or ())
            dependencies.extend(
                {"kind": getattr(d, "kind", ""),
                 "refs": list(getattr(d, "refs", ()) or ())}
                for d in getattr(plan, "dependencies", ()) or ())
        duties = dict((execution or {}).get("actors") or {})
        return CompletePlanCandidate(
            candidate_id=candidate_id,
            snapshot_id=snapshot_id,
            plans=list(plans),
            actions=dict(actions),
            resource_claims=claims,
            dependencies=dependencies,
            execution_duties=duties,
            rule_score=float(score) if isinstance(score, (int, float))
            else None,
            candidate_features={},
            legal=True,
            source_plan_ids=[str(getattr(p, "proposal_id", ""))
                             for p in plans],
            candidate_skill="joint",
            plan_objects_available=bool(plans))

    # BASE: the branch the Coordinator actually returned (pending commit) —
    # always authoritative and complete.
    score = chosen.get("score")
    branches.append(_entry_to_branch(
        "joint_base", list(pending[1] or ()), final_actions, score,
        (summary.get("execution_duties") or {})))

    # Alternatives: the Coordinator's own validated evaluate-once cache,
    # excluding the returned branch (deduplicated by actions).
    coordinator = getattr(manager, "coordinator", None)
    cache = getattr(coordinator, "_cache", None)
    seen_action_keys = {tuple(sorted(final_actions.items()))}
    for entry in (cache.values() if cache else ()):
        actions = dict(getattr(entry, "actions", {}) or {})
        key = tuple(sorted(actions.items()))
        if not actions or key in seen_action_keys:
            continue
        seen_action_keys.add(key)
        branches.append(_entry_to_branch(
            f"joint_alt_{len(branches)}", list(getattr(entry, "plans", ()) or ()),
            actions, getattr(entry, "score", None),
            getattr(entry, "execution", {})))
    return branches[:max_branches]
# BASE first; keep the declared rule order (coordinator rank) after it.
    branches.sort(key=lambda b: 0 if b.candidate_id == "joint_base" else 1)
    return branches[:max_branches]


def branch_scoring_features(branches: List["CompletePlanCandidate"],
                            snapshot) -> None:
    """Task 3 (V1.3 fix): B-class candidate features from REAL evidence.

    - ``candidate_action_change_count`` compares each branch's actions with
      the SNAPSHOT's ``previous_action_ids`` (the last executed actions) —
      no more an empty ``previous={}``;
    - ``candidate_target_type`` encodes the modal plan ``goal_kind``;
    - ``candidate_predicted_damage`` / ``candidate_predicted_loss`` sum the
      plans' DECLARED effect amounts (declared predictions, never outcomes);
    - ``candidate_progress`` averages the plans' declared progress channels;
    - ``candidate_risk`` is the fraction of the branch's actors currently
      inside enemy weapon range (snapshot relations);
    - ``candidate_support_gain`` has no reliable estimator and stays MISSING
      (NaN) — unknown values are never filled with 0.

    The SAME function runs at branch construction (inference) and at any
    training-side branch feature extraction, so train/inference vectors
    agree. Feature names are the candidate-layer schema entries;
    ``schema.build_model_features`` prefixes them identically on both sides.
    """
    from .features import (GOAL_KIND_VOCABULARY, MISSING,
                           PHASE_VOCABULARY, SKILL_VOCABULARY, encode_index)
    previous_actions = getattr(snapshot, "previous_action_ids", {}) or {}
    relations = getattr(snapshot, "relations", {}) or {}
    allies = getattr(snapshot, "allies", {}) or {}
    for branch in branches:
        actors = sorted({actor for plan in branch.plans
                         for actor in getattr(plan, "controlled_actors", ())})
        changes = sum(1 for actor, action in branch.actions.items()
                      if previous_actions.get(actor) != action)
        damage = 0.0
        loss = 0.0
        goal_kinds: List[str] = []
        skills: List[str] = []
        phases: List[str] = []
        progress_values: List[float] = []
        for plan in branch.plans:
            goal_kind = getattr(plan, "goal_kind", "")
            if goal_kind:
                goal_kinds.append(str(goal_kind))
            skills.append(str(getattr(plan, "primary_skill", "")))
            commitment = getattr(plan, "commitment", None)
            if commitment is not None and getattr(commitment, "phase", ""):
                phases.append(str(commitment.phase))
            declared_progress = (getattr(plan, "features", {}) or {}).get(
                "progress")
            if isinstance(declared_progress, (int, float)) \
                    and math.isfinite(float(declared_progress)):
                progress_values.append(float(declared_progress))
            for effect in getattr(plan, "expected_effects", ()) or ():
                amount = getattr(effect, "amount", None)
                if amount is None or not isinstance(amount, (int, float)) \
                        or not math.isfinite(float(amount)):
                    continue
                if getattr(effect, "kind", "") == "damage":
                    damage += float(amount)
                elif getattr(effect, "kind", "") == "loss":
                    loss += float(amount)
        # risk: branch actors currently inside enemy weapon range
        exposed = counted = 0
        for actor in actors:
            unit = allies.get(actor)
            if unit is None or not getattr(unit, "alive", False):
                continue
            counted += 1
            rows = relations.get(actor, {}) or {}
            if any(getattr(row, "in_enemy_weapon_range", False)
                   for row in rows.values()):
                exposed += 1
        def _modal(values: List[str]) -> Optional[str]:
            if not values:
                return None
            counts: Dict[str, int] = {}
            for value in values:
                counts[value] = counts.get(value, 0) + 1
            return max(counts.items(), key=lambda kv: (kv[1], kv[0]))[0]
        modal_goal = _modal(goal_kinds)
        modal_skill = _modal(skills)
        phase = phases[0] if phases else None
        branch.candidate_features = {
            "candidate_skill": float(encode_index(modal_skill,
                                                  SKILL_VOCABULARY)),
            "candidate_phase": float(encode_index(phase, PHASE_VOCABULARY)),
            "candidate_actor_count": float(len(actors)),
            "candidate_action_change_count": float(changes),
            "candidate_target_type": float(encode_index(
                modal_goal, GOAL_KIND_VOCABULARY)),
            "candidate_predicted_damage": damage,
            "candidate_predicted_loss": loss,
            "candidate_risk": (exposed / counted) if counted else MISSING,
            "candidate_support_gain": MISSING,   # no estimator: MISSING
            "candidate_progress": (math.fsum(progress_values)
                                   / len(progress_values))
            if progress_values else MISSING,
            # Task 4: no branch-level conflict estimator exists — MISSING,
            # never a fabricated 0
            "candidate_resource_conflict": MISSING,
            "candidate_execution_duty": float(len(sorted({
                row.get("kind") for row
                in branch.execution_duties.values()
                if isinstance(row, dict) and row.get("kind")}))),
        }
        branch.source_plan_ids = [str(getattr(p, "proposal_id", ""))
                                  for p in branch.plans]
