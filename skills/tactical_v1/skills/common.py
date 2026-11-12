"""Shared candidate-construction helpers for tactical_v1 skills."""
import hashlib
import math
from typing import Dict, List, Optional, Tuple

from ..context import Context, distance
from ..contracts import (CandidatePlan, Commitment, Dependency, EffectClaim,
                         ResourceClaims, SpaceClaim)


def plan_id(skill: str, snapshot_id: int, actors, goal: str, phase: str,
            actions: Dict[str, int]) -> str:
    """Stable digest of snapshot + plan semantics (R4): re-running the same
    decision yields the same id; it never depends on call order."""
    payload = "|".join([
        skill, str(snapshot_id), ",".join(sorted(actors)), str(goal), phase,
        ",".join(f"{a}:{v}" for a, v in sorted(actions.items()))])
    return f"{skill}-{hashlib.sha1(payload.encode()).hexdigest()[:10]}"


def space_claim(context: Context, actor: str, action: int) -> SpaceClaim:
    unit = context.snapshot.allies[actor]
    prediction = context.predict_position(actor, action)
    return SpaceClaim(actor=actor, position=prediction.position,
                      window_loops=context.H,
                      collision_class="air" if unit.is_flying else "ground",
                      radius=unit.radius)


def move_progress(context: Context, actor: str, action: Optional[int],
                  target) -> float:
    """Distance closure toward ``target``, normalized and capped (A3)."""
    if action is None:
        return 0.0
    destinations = context.move_destinations(actor)
    destination = destinations.get(action)
    if destination is None:
        return 0.0
    unit = context.snapshot.allies[actor]
    before = distance(unit.position, target)
    after = distance(destination, target)
    return max(min((before - after) / PROGRESS_SCALE, 1.0), -1.0)


PROGRESS_SCALE = 2.0


def best_move_toward(context: Context, actor: str, target) -> Optional[int]:
    """Legal move minimizing distance to ``target`` (deterministic ties)."""
    destinations = context.move_destinations(actor)
    if not destinations:
        return None
    return min(sorted(destinations),
               key=lambda a: (round(distance(destinations[a], target), 3), a))


def best_move_away(context: Context, actor: str, threats: List[str]) -> Optional[int]:
    """Legal move minimizing the risk AT THE PREDICTED position — the same
    risk standard as the evaluator (R3), never 'max distance from nearest'."""
    destinations = context.move_destinations(actor)
    if not destinations:
        return None
    unit = context.snapshot.allies[actor]
    allies = [u.position for a, u in sorted(context.snapshot.allies.items())
              if a != actor and u.alive]

    def score(action):
        prediction = context.predict_position(actor, action)
        risk = context.risk_at(actor, prediction.position)
        support = (min(distance(prediction.position, p) for p in allies)
                   if allies else 0.0)
        return (len(risk.threats), -round(distance(prediction.position,
                                                  unit.position), 3),
                round(support, 3), action)

    return min(sorted(destinations), key=score)


def task_key_for(skill: str, actors, goal_kind: str, goal: str,
                 extra: str = "") -> str:
    """Stable duty key: skill + sorted actors + normalized goal (1.2).
    Independent of proposal_id and call order."""
    return "|".join((skill, ",".join(sorted(actors)), goal_kind, str(goal),
                     extra))


def make_plan(primary_skill: str, actors, goal: str, actions: Dict[str, int],
              effects: List[EffectClaim], claims: ResourceClaims,
              commitment: Commitment, features: Dict[str, float],
              snapshot_id: int, fingerprint: str,
              provider: Optional[str] = None, instance_id: Optional[str] = None,
              beneficiaries=(), dependencies=(), memory_delta=None,
              atomic: bool = False, rationale: str = "",
              goal_kind: str = "internal",
              goal_data: Optional[Dict] = None,
              task_key: str = "") -> CandidatePlan:
    named = {k: float(v) for k, v in features.items()}
    if set(named) - {"progress", "hold_value", "switch_cost", "reentry_cost",
                     "urgency"}:
        raise ValueError(
            f"unnamed feature channel in {primary_skill}: "
            f"{sorted(set(named) - {'progress', 'hold_value', 'switch_cost', 'reentry_cost', 'urgency'})}")
    return CandidatePlan(
        proposal_id=plan_id(primary_skill, snapshot_id, actors, goal,
                            commitment.phase, actions),
        snapshot_id=snapshot_id, config_fingerprint=fingerprint,
        primary_skill=primary_skill, specialization_provider=provider,
        instance_id=instance_id, controlled_actors=tuple(sorted(actions)),
        beneficiaries=tuple(beneficiaries), goal=goal,
        current_actions=dict(actions), expected_effects=effects, claims=claims,
        dependencies=tuple(dependencies), commitment=commitment,
        features=named, proposed_memory_delta=memory_delta or {},
        atomic=atomic, rationale=rationale, goal_kind=goal_kind,
        goal_data=dict(goal_data or {}),
        task_key=task_key or task_key_for(
            primary_skill, actors, goal_kind, goal))


def h_loops(context: Context, multiple: float) -> int:
    """Convert an H-multiple lifecycle budget into game loops (2.3)."""
    return int(round(context.H * multiple))


def no_progress(context: Context, memory, actor: str, window_h: float) -> bool:
    instance = memory.displacement(actor, h_loops(context, window_h),
                                   context.snapshot)
    return instance < 0.5
