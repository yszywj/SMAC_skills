"""A1: the complete, always-legal baseline plan P0.

Built BEFORE any search from the shared queries, so beam entries are always
compared against a full executable plan — never an empty 0-score plan.
Conservative per class: keep/choose legal attacks by window rules, approach,
search; medivacs heal/follow with occupancy respected (never chase enemies);
banelings get basic engage/wait (no burst optimization when S3 is off).
"""
from typing import Dict, List, Optional

from .context import Context, STOP
from .contracts import CALIBRATED, HEURISTIC


def baseline_actions(context: Context, memory, skills) -> Dict[str, int]:
    snapshot = context.snapshot
    actions: Dict[str, int] = {}
    search_waypoint = None
    if not snapshot.enemies:
        # deterministic shared search direction: toward the map search point
        search_waypoint = snapshot.search_location
    for actor, unit in sorted(snapshot.allies.items()):
        if not unit.alive:
            actions[actor] = 0
            continue
        capability = context.capabilities[actor]
        if capability.healer:
            actions[actor] = _medivac_baseline(context, actor, memory)
            continue
        if capability.suicide_attack:
            actions[actor] = _baneling_baseline(context, actor, memory,
                                                skills)
            continue
        actions[actor] = _combat_baseline(context, actor, memory,
                                          search_waypoint)
    return actions


def _combat_baseline(context: Context, actor: str, memory,
                     search_waypoint) -> int:
    # keep a still-valid attack target (instance), else nearest issuable
    instance = memory.instance_for("focus_fire", actor)
    if instance and instance.goal.startswith("e") \
            and instance.goal in context.snapshot.enemies \
            and context.can_issue_attack(actor, instance.goal):
        return context.attack_action(actor, instance.goal)
    attackable = context.attackable_enemies(actor)
    if attackable:
        return context.attack_action(actor, attackable[0])
    nearest = context.nearest_enemy(actor)
    if nearest is not None:
        from .context import distance
        move = _move_toward(context, actor,
                            context.snapshot.enemies[nearest].position)
        if move is not None:
            return move
    if search_waypoint is not None:
        move = _move_toward(context, actor, search_waypoint)
        if move is not None:
            return move
    return STOP if STOP in context.legal_actions(actor) else \
        min(context.legal_actions(actor) or {STOP})


def _medivac_baseline(context: Context, actor: str, memory) -> int:
    keep = context.keep_heal_action(actor)
    if keep is not None:
        return keep
    owners = context.heal_occupancy()
    candidates = sorted(context.snapshot.heal_candidates.get(actor, {}).items(),
                        key=lambda item: (-_urgency(context, item[1].get("target")),
                                          item[0]))
    for action, info in candidates:
        if action not in context.legal_actions(actor):
            continue
        patient = info.get("target")
        if owners.get(patient) not in (None, actor):
            continue      # observed occupancy: the order still owns this patient
        return action
    # approach the neediest unowned patient, else follow the fighting group
    from .context import distance
    patients = sorted((a for a, u in context.snapshot.allies.items()
                       if u.alive and 0 < u.health < u.max_health
                       and not context.capabilities[a].healer
                       and owners.get(a) is None),
                      key=lambda a: (-_urgency(context, a), a))
    if patients:
        move = _move_toward(context, actor,
                            context.snapshot.allies[patients[0]].position)
        if move is not None:
            return move
    fighters = [a for a, u in context.snapshot.allies.items()
                if u.alive and not context.capabilities[a].healer]
    if fighters:
        centre = (sum(context.snapshot.allies[a].position[0] for a in fighters)
                  / len(fighters),
                  sum(context.snapshot.allies[a].position[1] for a in fighters)
                  / len(fighters))
        if distance(context.snapshot.allies[actor].position, centre) > 5.0:
            move = _move_toward(context, actor, centre)
            if move is not None:
                return move
    return STOP


def _baneling_baseline(context: Context, actor: str, memory, skills) -> int:
    burst = skills.get("burst_engage") if skills else None
    instance = memory.instance_for("burst_engage", actor)
    goal = instance.goal if instance and instance.goal.startswith("e") else None
    for enemy_id in sorted(context.snapshot.enemies):
        if goal and enemy_id != goal:
            continue
        if context.can_issue_attack(actor, enemy_id) \
                and context.contact_status(actor, enemy_id) == "in_range":
            return context.attack_action(actor, enemy_id)
    if goal and goal in context.snapshot.enemies:
        move = _move_toward(context, actor,
                            context.snapshot.enemies[goal].position)
        if move is not None:
            return move
    nearest = context.nearest_enemy(actor)
    if nearest is not None:
        move = _move_toward(context, actor,
                            context.snapshot.enemies[nearest].position)
        if move is not None:
            return move
    return STOP


def _urgency(context: Context, patient: Optional[str]) -> float:
    if patient is None or patient not in context.snapshot.allies:
        return 0.0
    unit = context.snapshot.allies[patient]
    missing = max(unit.max_health - unit.health, 0.0)
    return missing / max(unit.max_health, 1.0)


def _move_toward(context: Context, actor: str, target) -> Optional[int]:
    destinations = context.move_destinations(actor)
    if not destinations:
        return None
    from .context import distance
    return min(sorted(destinations),
               key=lambda a: (round(distance(destinations[a], target), 3), a))
