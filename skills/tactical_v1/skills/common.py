"""Shared candidate-construction helpers for tactical_v1 skills."""
import math
from typing import Dict, List, Optional, Tuple

from ..context import Context, STOP, distance
from ..contracts import (CALIBRATED, CandidatePlan, Commitment, EffectClaim,
                         HEURISTIC, ResourceClaims, VERIFIED)

_plan_counter = 0


def next_plan_id(prefix: str) -> str:
    global _plan_counter
    _plan_counter += 1
    return f"{prefix}-{_plan_counter:05d}"


def enemy_weapon_range(context: Context, enemy_id: str) -> Optional[float]:
    """Public static data only: the adapter's action contract publishes
    candidate-center distances per unit type (bridge RACE_RANGES)."""
    contract = context.snapshot.raw_state.get("action_contract", {})
    table = contract.get("candidate_center_distance_by_type", {})
    enemy = context.snapshot.enemies[enemy_id]
    for key, value in table.items():
        if key == enemy.type_name or str(enemy.unit_type_id) == str(key):
            return float(value)
    return None


def threats_reaching(context: Context, position, ignore_flying_domain=True) -> List[str]:
    """Visible enemies whose weapon plausibly reaches ``position``."""
    result = []
    for enemy_id, enemy in context.snapshot.enemies.items():
        relation_range = enemy_weapon_range(context, enemy_id)
        if relation_range is None:
            # Unknown range: treat near enemies (<=5) as potential threats.
            if distance(position, enemy.position) <= 5.0:
                result.append(enemy_id)
            continue
        if distance(position, enemy.position) <= relation_range:
            result.append(enemy_id)
    return result


def best_move_toward(context: Context, actor: str, target) -> Optional[int]:
    """Legal move minimizing distance to ``target`` (deterministic ties)."""
    destinations = context.move_destinations(actor)
    if not destinations:
        return None
    return min(sorted(destinations),
               key=lambda a: (round(distance(destinations[a], target), 3), a))


def best_move_away(context: Context, actor: str, threats: List[str]) -> Optional[int]:
    """Legal move maximizing min distance to visible threats, preferring
    destinations that also move toward the main ally group."""
    destinations = context.move_destinations(actor)
    if not destinations or not threats:
        return None
    allies = [u.position for u in context.snapshot.allies.values()
              if u.actor != actor and u.alive]
    def score(action):
        destination = destinations[action]
        threat_distance = min(distance(destination, context.snapshot.enemies[e].position)
                              for e in threats)
        support = (min(distance(destination, p) for p in allies) if allies else 0.0)
        return (-round(threat_distance, 3), round(support, 3), action)
    return min(sorted(destinations), key=score)


def move_progress(context: Context, actor: str, action: Optional[int],
                  target) -> float:
    if action is None:
        return 0.0
    destinations = context.move_destinations(actor)
    destination = destinations.get(action)
    if destination is None:
        return 0.0
    unit = context.snapshot.allies[actor]
    before = distance(unit.position, target)
    after = distance(destination, target)
    return before - after


def make_plan(primary_skill: str, actors, goal: str, actions: Dict[str, int],
              effects: List[EffectClaim], claims: ResourceClaims,
              commitment: Commitment, features: Dict[str, float],
              snapshot_id: int, fingerprint: str,
              provider: Optional[str] = None, instance_id: Optional[str] = None,
              beneficiaries=(), dependencies=(), memory_delta=None,
              rationale: str = "") -> CandidatePlan:
    return CandidatePlan(
        proposal_id=next_plan_id(primary_skill), snapshot_id=snapshot_id,
        config_fingerprint=fingerprint, primary_skill=primary_skill,
        specialization_provider=provider, instance_id=instance_id,
        controlled_actors=tuple(sorted(actions)), beneficiaries=tuple(beneficiaries),
        goal=goal, current_actions=dict(actions), expected_effects=effects,
        claims=claims, dependencies=tuple(dependencies), commitment=commitment,
        features=dict(features), proposed_memory_delta=memory_delta or {},
        rationale=rationale)
