"""Final validation and bounded repair (P4.5)."""
from typing import Dict, List, Optional, Tuple

from .context import Context, DEAD_NOOP
from .contracts import CandidatePlan


def validate_actions(actions: List[int], context: Context,
                     n_agents: int) -> List[str]:
    """Legality of the complete per-unit list against the raw mask view."""
    problems = []
    actors = sorted(context.snapshot.allies)
    if len(actions) != n_agents:
        problems.append(f"length:{len(actions)}!={n_agents}")
    for i, action in enumerate(actions):
        actor = f"u{i}"
        if not isinstance(action, int) or isinstance(action, bool):
            problems.append(f"type:{actor}")
            continue
        unit = context.snapshot.allies.get(actor)
        legal = context.legal_actions(actor)
        if unit is not None and not unit.alive:
            if action != DEAD_NOOP:
                problems.append(f"dead:{actor}:{action}")
            continue
        if action not in legal:
            problems.append(f"illegal:{actor}:{action}")
    return problems


def validate_joint(plans: List[CandidatePlan], context: Context) -> List[str]:
    """Structural checks on a joint plan before scoring."""
    problems = []
    seen = set()
    for plan in plans:
        for actor in plan.controlled_actors:
            if actor in seen:
                problems.append(f"double_control:{actor}:{plan.proposal_id}")
            seen.add(actor)
        for actor, action in plan.current_actions.items():
            if action not in context.legal_actions(actor):
                problems.append(f"plan_illegal:{actor}:{action}")
        if set(plan.current_actions) != set(plan.controlled_actors):
            problems.append(f"actions_actors_mismatch:{plan.proposal_id}")
    return problems


def fallback_action(actor: str, context: Context) -> int:
    """Deterministic, always-legal completion for one unit: best legal
    attack, else a legal move toward the nearest enemy, else Stop, else 0."""
    legal = context.legal_actions(actor)
    unit = context.snapshot.allies.get(actor)
    if unit is None or not unit.alive:
        return DEAD_NOOP
    best = None
    for enemy_id in context.attackable_enemies(actor):
        score = (-context.distance(actor, enemy_id), enemy_id)
        if best is None or score < best[0]:
            best = (score, enemy_id)
    if best is not None:
        return 6 + context.snapshot.enemies[best[1]].index
    nearest = context.nearest_enemy(actor)
    if nearest is not None:
        destinations = context.move_destinations(actor)
        enemy = context.snapshot.enemies[nearest]
        if destinations:
            best_move = min(
                sorted(destinations),
                key=lambda a: (_dist(destinations[a], enemy.position), a))
            return best_move
    return 1 if 1 in legal else (next(iter(sorted(legal))) if legal else DEAD_NOOP)


def _dist(a, b) -> float:
    import math
    return math.hypot(a[0] - b[0], a[1] - b[1])
