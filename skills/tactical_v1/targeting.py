"""Shared, short-lived target allocation for firing and cooldown movement.

An allocation is a candidate-generation preference, not a damage ledger or
an order. Only one primary volley per available source is reserved here.
Prospective movement/visibility remain uncertain, and ordinary nonlethal
fire never removes incoming weapon exposure. The final evaluator must
reconstruct all physical effects from the selected actions.
"""
from dataclasses import dataclass
from typing import Dict, Optional

from .battle_pressure import enemy_pressure
from .context import IN_RANGE, distance
from .unit_value_view import target_value as _object_value


@dataclass(frozen=True)
class TargetAssignment:
    target: str
    previous_target: Optional[str]
    damage_est: Optional[float]
    immediate: bool
    requires_move: bool
    horizon_loops: int


@dataclass(frozen=True)
class TargetPlan:
    assignments: Dict[str, TargetAssignment]
    horizon_loops: int


def planning_enabled(config=None):
    return (config or {}).get("focus_fire", {}).get("target_planning_enabled", True)


def local_front_targets(context, memory, actors, *, seed_targets=(), radius=6., config=None):
    """Visible targets on one observed local front, with shared-plan ordering.

    Contact and incoming threats anchor the front. An explicit distant seed
    cannot add a second front; neither can a chain of nearby enemy units.
    Without contact, one nearest planned/seed target supplies the anchor.
    This is candidate scope, not a commitment or a claim of suppression.
    """
    actors = tuple(sorted(a for a in actors if a in context.snapshot.allies
                          and context.snapshot.allies[a].alive))
    enemies = context.snapshot.enemies
    if not actors or not enemies:
        return ()
    plan = shared_target_plan(context, memory, config)
    assigned = tuple(dict.fromkeys(plan.assignments[a].target for a in actors
                                  if a in plan.assignments))
    anchors = {e for a in actors for e in context.threats_to(a) if e in enemies}
    anchors.update(e for a in actors for e in context.attackable_enemies(a)
                   if e in enemies and context.contact_status(a, e) == IN_RANGE)
    seeds = {e for e in seed_targets if e in enemies}
    proximity = lambda e: min(distance(context.snapshot.allies[a].position,
                                      enemies[e].position) for a in actors)
    if not anchors:
        pool = seeds or set(assigned) or set(enemies)
        anchors = {min(pool, key=lambda e: (proximity(e), e))}
    # One hop from physical anchors, never recursively from newly added units.
    front = {e for e in enemies if e in anchors or any(
        distance(enemies[e].position, enemies[anchor].position) <= radius
        for anchor in anchors)}
    priority = {e: index for index, e in enumerate(assigned)}
    return tuple(sorted(front, key=lambda e: (priority.get(e, len(priority)), proximity(e), e)))


def next_shot_target(context, memory, actor, candidates):
    """Read a selected G3 intention without renewing it during a query.

    The existing attack history remains the fallback. A G3 preference lasts
    at most two decision windows, and only a committed plan can create it.
    An obsolete target, replaced actor or expired task is not a hard lock.
    """
    candidates = set(candidates)
    instance = memory.instance_for("combat_reposition", actor)
    if instance is not None:
        extra = instance.extra
        target = extra.get("next_shot_target")
        tags = tuple(context.snapshot.allies[a].tag for a in instance.actors
                     if a in context.snapshot.allies)
        if target in candidates and tags == instance.ally_tags \
                and context.snapshot.game_loop < extra.get("next_shot_until_loop", -1):
            return target
    return memory.recent_attack_target(context, actor, candidates)


def _opportunity(context, actor, target, horizon):
    """Return one possible primary volley; never promote an illegal shot."""
    unit = context.snapshot.allies[actor]
    if context.can_issue_attack(actor, target) \
            and context.contact_status(actor, target) == IN_RANGE:
        estimate = context.fire_estimate(actor, target, horizon=horizon)
        if estimate.window_shots:
            immediate = context.fire_estimate(actor, target).window_shots >= 1
            return estimate.damage_est, immediate, False
    if unit.weapon_cooldown_loops > horizon:
        return None
    # The existing position or one legal control-window move must already
    # open the next shot. No long-range chase is manufactured by this plan.
    positions = [unit.position]
    positions.extend(context.predict_position(actor, action).position
                     for action in sorted(context.move_destinations(actor)))
    for index, point in enumerate(positions):
        projected = context.projected_attack_opportunity(
            actor, target, point, horizon=horizon)
        if projected.available and projected.ready:
            return projected.damage_est, False, index > 0
    return None


def shared_target_plan(context, memory, config=None):
    """Deterministic allocation shared by G1, G3 and the final evaluator.

    The context-local cache includes the read-only memory view and the
    enabling switch. Candidate order and earlier score evaluations cannot
    change allocations. Earlier-ready sources reserve first, so prospective
    shots cannot displace a currently available primary volley.
    """
    horizon = 2 * max(context.H, 1)
    if not planning_enabled(config) or memory is None:
        return TargetPlan({}, horizon)
    enemies = tuple(sorted(context.snapshot.enemies))
    actors = tuple(a for a, u in sorted(context.snapshot.allies.items())
                   if u.alive and not context.capabilities[a].healer
                   and not context.capabilities[a].suicide_attack)
    previous = {a: next_shot_target(context, memory, a, enemies) for a in actors}
    key = (id(memory), tuple(previous.items()))
    cache = getattr(context, "_short_target_plan_cache", None)
    if cache is None:
        cache = context._short_target_plan_cache = {}
    if key in cache:
        return cache[key]
    options = {a: {} for a in actors}
    for actor in actors:
        for target in enemies:
            opportunity = _opportunity(context, actor, target, horizon)
            if opportunity is not None:
                options[actor][target] = opportunity
    remaining = {e: max(context.snapshot.enemies[e].effective_health, 0.)
                 for e in enemies}
    pressure = {e: enemy_pressure(context, e) for e in enemies}
    assignments = {}
    immediate_options = {a: {e: option for e, option in options[a].items()
                             if option[1]} for a in actors}
    # Allocate actual current-window sources first, restricted to their
    # current shots. Only then may future movement sources reserve remaining
    # budgets. Raw cooldown alone cannot distinguish these two phases.
    order = sorted(actors, key=lambda a: (
        not bool(immediate_options[a]),
        context.snapshot.allies[a].weapon_cooldown_loops,
        len(immediate_options[a] or options[a]), a))
    for actor in order:
        choices = immediate_options[actor] or options[actor]
        if not choices:
            continue

        def rank(target):
            amount, immediate, moving = choices[target]
            enemy = context.snapshot.enemies[target]
            available = remaining[target]
            useful = min(max(amount or 0., 0.), available)
            maximum = max(enemy.max_health + enemy.max_shield,
                          enemy.effective_health, 1.)
            # Main value is useful primary damage and feasible completion.
            # Pressure is a bounded objective preference, never suppression.
            value = 3. * useful / maximum
            if amount is not None and available > 0 and amount >= available:
                value += 2.
            # Object value (adapted legacy evaluator) is a bounded ORDERING
            # preference for equally useful volleys, not a second damage,
            # kill or relief ledger entry.
            value += getattr(context, 'unit_value_parameters', {}).get(
                'focus_value_weight', .2) * _object_value(context, target) if useful > 0 else 0.
            # A useful primary volley advances the local objective even
            # against a large health pool. Normalize this separate bounded
            # preference by the source's volley, not by enemy EHP again.
            value += pressure[target] * useful / max(amount or 0., 1.)
            if not immediate:
                value *= .5
            if moving:
                value *= .75
            if target == previous[actor] and available > 0:
                value += .08
            return (-value, moving, target != previous[actor], target)

        target = min(choices, key=rank)
        amount, immediate, moving = choices[target]
        assignments[actor] = TargetAssignment(
            target, previous[actor], amount, immediate, moving, horizon)
        if amount is not None:
            remaining[target] = max(0., remaining[target] - max(amount, 0.))
    result = TargetPlan(assignments, horizon)
    cache[key] = result
    return result


def allocation_progress(context, actions, report, memory, config=None):
    """Bounded strategic value of actual assigned primary damage.

    Recompute from final actions, not candidate claims. Each actor supplies
    at most its current-window primary shot; each target earns at most its
    pressure (<=1), normalized by its assigned primary volley budget.
    Already lethal targets earn
    zero here because the separate lethal-pressure term handles them. Thus
    this is neither another kill bonus nor a reduction in incoming risk.
    """
    plan = shared_target_plan(context, memory, config)
    allocated = {}
    budgets = {}
    for assignment in plan.assignments.values():
        if assignment.damage_est is not None:
            budgets[assignment.target] = budgets.get(assignment.target, 0.) \
                + max(assignment.damage_est, 0.)
    for actor, assignment in plan.assignments.items():
        action = actions.get(actor)
        if action is None:
            continue
        command = context.decode(actor, action)
        target = assignment.target
        if command.kind != "attack" or command.target != target \
                or not context.can_issue_attack(actor, target):
            continue
        estimate = context.fire_estimate(actor, target)
        if estimate.window_shots < 1 or estimate.damage_est is None:
            continue
        allocated[target] = allocated.get(target, 0.) + max(estimate.damage_est, 0.)
    total = 0.
    for target, damage in allocated.items():
        enemy = context.snapshot.enemies[target]
        known = max(report.damage_known.get(target, 0.), 0.)
        if known >= enemy.effective_health:
            continue
        useful = min(damage, known, enemy.effective_health)
        budget = min(budgets.get(target, 0.), enemy.effective_health)
        if budget > 0:
            total += enemy_pressure(context, target) * min(useful / budget, 1.)
    return total
