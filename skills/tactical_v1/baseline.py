"""A1/A3: the complete, always-legal baseline plan P0 and the context-aware
refill used by search/repair. Refill knows what is already kept: observed
heal ownership, this step's kept heal assignments, committed window fire and
kept contact slots — it never re-assigns an occupied patient or re-creates
an already-satisfied kill.
"""
from typing import Dict, Iterable, List, Optional

from .actions import KEEP_HEAL_ORDER
from .context import Context, STOP, distance


def complete_plan(context: Context, memory, skills,
                  locked_plans: Iterable = ()) -> Dict[str, int]:
    """A3: ONE shared-resource completion for P0, neighborhood replacement
    and repair. locked_plans' actions are protected verbatim; every free
    actor is filled in a single resource context (observed heal ownership,
    this step's new heal capacity, committed window fire, and necessary
    space reservations grow as each action is assigned)."""
    return baseline_actions(context, memory, skills,
                            reserved_plans=tuple(locked_plans))


def baseline_actions(context: Context, memory, skills,
                     reserved: Optional[Dict[str, int]] = None,
                     reserved_plans: Iterable = (),
                     only: Optional[Iterable[str]] = None
                     ) -> Dict[str, int]:
    """Conservative complete policy (used by complete_plan and repair).

    reserved / reserved_plans: already-kept final actions of the joint plan
    being repaired or substituted. only: compute just these actors (the
    released ones); default computes everyone (P0).
    """
    snapshot = context.snapshot
    if reserved is None and reserved_plans:
        reserved = {a: p.current_actions[a] for p in reserved_plans
                    for a in p.current_actions}
    targets = set(only) if only is not None else None
    # -- context from kept actions ------------------------------------------
    kept_actions = {a: act for a, act in (reserved or {}).items()
                    if targets is None or a not in targets}
    heal_taken = _kept_heal_assignments(context, kept_actions)
    committed_fire = _committed_fire(context, kept_actions)
    owners = context.heal_occupancy()
    actions: Dict[str, int] = {}
    search_waypoint = snapshot.search_location if not snapshot.enemies else None
    # pre-reserve kept healer actions so the WHOLE call shares one capacity
    for actor, action in kept_actions.items():
        if actor in context.snapshot.allies                 and context.capabilities.get(actor, object()).healer:
            semantic = context.decode(actor, action)
            if semantic.kind in ("heal", KEEP_HEAL_ORDER) and semantic.target:
                heal_taken.setdefault(semantic.target, actor)
    for actor, unit in sorted(snapshot.allies.items()):
        if targets is not None and actor not in targets:
            continue
        if not unit.alive:
            actions[actor] = 0
            continue
        capability = context.capabilities[actor]
        if capability.healer:
            action = _medivac_baseline(context, actor, memory,
                                       owners, heal_taken)
            semantic = context.decode(actor, action)
            if semantic.kind in ("heal", KEEP_HEAL_ORDER) and semantic.target:
                heal_taken.setdefault(semantic.target, actor)
            actions[actor] = action
            continue
        if capability.suicide_attack:
            actions[actor] = _baneling_baseline(context, actor, memory,
                                                skills, committed_fire)
            continue
        actions[actor] = _combat_baseline(context, actor, memory,
                                          search_waypoint, committed_fire)
    return actions


def _committed_fire(context: Context, kept: Dict[str, int]) -> Dict[str, float]:
    """Window-qualified known damage already committed by kept actions."""
    totals: Dict[str, float] = {}
    for actor, action in kept.items():
        unit = context.snapshot.allies.get(actor)
        if unit is None or not unit.alive:
            continue
        semantic = context.decode(actor, action)
        if semantic.kind != "attack" or semantic.target not in context.snapshot.enemies:
            continue
        estimate = context.fire_estimate(actor, semantic.target)
        if estimate.window_shots >= 1 and estimate.damage_est:
            totals[semantic.target] = totals.get(semantic.target, 0.0) \
                + estimate.damage_est
    return totals


def _kept_heal_assignments(context: Context,
                           kept: Dict[str, int]) -> Dict[str, str]:
    assigned: Dict[str, str] = {}
    for actor, action in kept.items():
        if actor not in context.snapshot.allies:
            continue
        if not context.capabilities.get(actor, object()).healer:
            continue
        semantic = context.decode(actor, action)
        if semantic.kind in ("heal", KEEP_HEAL_ORDER) and semantic.target:
            assigned.setdefault(semantic.target, actor)
    return assigned


def _urgency(context: Context, patient: Optional[str]) -> float:
    if patient is None or patient not in context.snapshot.allies:
        return 0.0
    unit = context.snapshot.allies[patient]
    missing = max(unit.max_health - unit.health, 0.0)
    return missing / max(unit.max_health, 1.0)


def _combat_baseline(context: Context, actor: str, memory,
                     search_waypoint, committed_fire) -> int:
    # keep a still-valid attack target (typed goal: enemy), else nearest
    # issuable target that is NOT already covered by committed fire
    instance = memory.instance_for("focus_fire", actor)
    if instance is not None and instance.goal_kind == "enemy" \
            and instance.goal in context.snapshot.enemies \
            and context.can_issue_attack(actor, instance.goal):
        return context.attack_action(actor, instance.goal)
    best = None
    for enemy_id in context.attackable_enemies(actor):
        enemy = context.snapshot.enemies[enemy_id]
        if committed_fire.get(enemy_id, 0.0) >= enemy.effective_health:
            continue      # kept fire already covers this target
        best = enemy_id
        break
    if best is None and context.attackable_enemies(actor):
        best = context.attackable_enemies(actor)[0]
    if best is not None:
        return context.attack_action(actor, best)
    nearest = context.nearest_enemy(actor)
    if nearest is not None:
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


def _medivac_baseline(context: Context, actor: str, memory,
                      owners, heal_taken) -> int:
    keep = context.keep_heal_action(actor)
    if keep is not None:
        return keep
    candidates = sorted(context.snapshot.heal_candidates.get(actor, {}).items(),
                        key=lambda item: (-_urgency(context, item[1].get("target")),
                                          item[0]))
    for action, info in candidates:
        if action not in context.legal_actions(actor):
            continue
        patient = info.get("target")
        if owners.get(patient) not in (None, actor):
            continue          # observed order still owns this patient
        if heal_taken.get(patient, actor) != actor:
            continue          # this step's capacity: one new source per patient
        return action
    patients = sorted((a for a, u in context.snapshot.allies.items()
                       if u.alive and 0 < u.health < u.max_health
                       and not context.capabilities[a].healer
                       and owners.get(a) is None
                       and heal_taken.get(a) is None),
                      key=lambda a: (-_urgency(context, a), a))
    if patients:
        move = _move_toward(context, actor,
                            context.snapshot.allies[patients[0]].position)
        if move is not None:
            return move
    fighters = [a for a, u in context.snapshot.allies.items()
                if u.alive and not context.capabilities[a].healer
                and not u.is_flying]
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


def _baneling_baseline(context: Context, actor: str, memory, skills,
                       committed_fire) -> int:
    instance = memory.instance_for("burst_engage", actor)
    goal = instance.goal if instance is not None and \
        instance.goal_kind == "enemy" else None
    for enemy_id in sorted(context.snapshot.enemies):
        if goal and enemy_id != goal:
            continue
        if context.can_issue_attack(actor, enemy_id):
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


def _move_toward(context: Context, actor: str, target) -> Optional[int]:
    destinations = context.move_destinations(actor)
    if not destinations:
        return None
    return min(sorted(destinations),
               key=lambda a: (round(distance(destinations[a], target), 3), a))
