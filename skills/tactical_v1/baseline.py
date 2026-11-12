"""A1 (round 5): the ONE completion entry point.

complete_plan(context, memory, skills, locked_plans) returns the WHOLE team's
actions. Locked units are never re-decided: their actions are validated,
conflict-checked and reserved FIRST; free actors are then assigned one at a
time in a single shared resource context (observed heal ownership, this
step's new heal capacity from already-assigned actions, window fire budget
and exclusive contact slots). Each assignment updates the resources the
next free actor sees. Candidate trial runs never touch public memory.
"""
from typing import Dict, Iterable, List, Optional, Set

from .actions import KEEP_HEAL_ORDER
from .context import Context, STOP, distance


class LockedPlanConflict(RuntimeError):
    """Locked plans clash with each other (control right, hard slot) or a
    locked action is illegal — the caller must reject that search branch,
    never silently rewrite a locked task."""


class _Shim:
    """Minimal locked-plan stand-in for a plain {actor: action} dict."""

    def __init__(self, actions, claims_space=()):
        self.current_actions = dict(actions)
        self.proposal_id = "reserved-" + ",".join(sorted(
            f"{a}:{v}" for a, v in actions.items()))
        self.claims = type("C", (), {"space": tuple(claims_space)})()


class _Resources:
    """Shared per-completion resource context (1.3).

    heal_taken: observed owners (until release is observed) + this step's
    new heal/keep assignments — meeting MOVES never occupy heal capacity.
    fire: window-qualified known damage per target from assigned actions.
    slots: exclusive contact-slot ids from locked plans.
    """

    def __init__(self, context: Context):
        self.context = context
        self.heal_taken: Dict[str, str] = dict(context.heal_occupancy())
        self.fire: Dict[str, float] = {}
        self.slots: Set[str] = set()

    def add_locked_tasks(self, locked_plans) -> None:
        for plan in locked_plans:
            for claim in getattr(plan.claims, "space", ()):
                if getattr(claim, "slot_id", None):
                    self.slots.add(claim.slot_id)

    def reserve_action_once(self, actor: str, action: int) -> None:
        """Register one assigned action's effects, deduplicated per actor —
        locked resources are registered here exactly once, never again by
        plan."""
        context = self.context
        unit = context.snapshot.allies.get(actor)
        if unit is None or not unit.alive:
            return
        semantic = context.decode(actor, action)
        capability = context.capabilities.get(actor)
        if capability is not None and capability.healer:
            if semantic.kind in ("heal", KEEP_HEAL_ORDER) and semantic.target:
                self.heal_taken.setdefault(semantic.target, actor)
            return
        if semantic.kind == "attack" and semantic.target in context.snapshot.enemies:
            estimate = context.fire_estimate(actor, semantic.target)
            if estimate.window_shots >= 1 and estimate.damage_est:
                self.fire[semantic.target] = \
                    self.fire.get(semantic.target, 0.0) + estimate.damage_est

    def target_covered(self, enemy_id: str) -> bool:
        enemy = self.context.snapshot.enemies.get(enemy_id)
        return enemy is not None and \
            self.fire.get(enemy_id, 0.0) >= enemy.effective_health


def complete_plan(context: Context, memory, skills,
                  locked_plans: Iterable = ()) -> Dict[str, int]:
    locked_plans = tuple(sorted(locked_plans,
                                key=lambda p: p.proposal_id))
    locked_actions: Dict[str, int] = {}
    # -- 1. validate + reserve locked tasks (control rights, hard slots) --
    from .ledger import Ledger
    ledger = Ledger()
    for plan in locked_plans:
        for actor, action in plan.current_actions.items():
            if actor in locked_actions:
                raise LockedPlanConflict(f"double control: {actor}")
            if action not in context.legal_actions(actor):
                raise LockedPlanConflict(f"locked action illegal: {actor}")
        if ledger.conflicts(plan):
            raise LockedPlanConflict(plan.proposal_id)
        ledger.reserve(plan)
        locked_actions.update(plan.current_actions)
    # -- 2. shared resources from real observations + locked actions -------
    resources = _Resources(context)
    resources.add_locked_tasks(locked_plans)
    for actor, action in locked_actions.items():
        resources.reserve_action_once(actor, action)
    # -- 3. free actors: one shared context, assign-and-reserve one by one -
    actions = dict(locked_actions)
    free_actors = sorted(set(context.snapshot.allies) - set(locked_actions))
    for actor in free_actors:
        unit = context.snapshot.allies[actor]
        if not unit.alive:
            actions[actor] = 0
            continue
        capability = context.capabilities[actor]
        if capability.healer:
            action = _medivac_baseline(context, actor, memory, resources)
        elif capability.suicide_attack:
            action = _baneling_baseline(context, actor, memory, skills,
                                        resources)
        else:
            action = _combat_baseline(context, actor, memory, resources)
        actions[actor] = action
        resources.reserve_action_once(actor, action)
    assert set(actions) == set(context.snapshot.allies)
    assert all(actions[a] == v for a, v in locked_actions.items())
    return actions


# ---------------------------------------------------------------------------
# Free-actor policies (operate ONLY on the shared resources)
# ---------------------------------------------------------------------------

def _urgency(context: Context, patient: Optional[str]) -> float:
    if patient is None or patient not in context.snapshot.allies:
        return 0.0
    unit = context.snapshot.allies[patient]
    missing = max(unit.max_health - unit.health, 0.0)
    return missing / max(unit.max_health, 1.0)


def _search_goal(context: Context, memory) -> Optional[tuple]:
    """Search target that skips waypoints already visited or temporarily
    unreachable — a failed waypoint is not quietly re-walked by baseline."""
    if context.snapshot.enemies:
        return None
    base = context.snapshot.search_location
    loop = context.snapshot.game_loop

    def fresh(point):
        cell = f"{int(point[0] // 8)}:{int(point[1] // 8)}"
        if loop - memory.unreachable_search_cells.get(cell, -10**9) < 160:
            return False
        return memory.visited_search_cells.get(cell, -10**9) < loop - 160
    if fresh(base):
        return base
    radius = max(base[0], base[1]) / 2.0 or 8.0
    for dx, dy in ((.5, .5), (.5, -.5), (-.5, .5), (-.5, -.5),
                   (1., 0.), (0., 1.), (-1., 0.), (0., -1.)):
        point = (base[0] + dx * radius, base[1] + dy * radius)
        if fresh(point):
            return point
    return None


def _combat_baseline(context: Context, actor: str, memory,
                     resources: _Resources) -> int:
    instance = memory.instance_for("focus_fire", actor)
    if instance is not None and instance.goal_kind == "enemy" \
            and instance.goal in context.snapshot.enemies \
            and context.can_issue_attack(actor, instance.goal):
        return context.attack_action(actor, instance.goal)
    best = None
    for enemy_id in context.attackable_enemies(actor):
        if resources.target_covered(enemy_id):
            continue      # window fire already assigned by earlier actors
        best = enemy_id
        break
    if best is None and context.attackable_enemies(actor):
        best = context.attackable_enemies(actor)[0]   # keep firing is legal
    if best is not None:
        return context.attack_action(actor, best)
    nearest = context.nearest_enemy(actor)
    if nearest is not None:
        move = _move_toward(context, actor,
                            context.snapshot.enemies[nearest].position)
        if move is not None:
            return move
    goal = _search_goal(context, memory)
    if goal is not None:
        move = _move_toward(context, actor, goal)
        if move is not None:
            return move
    return STOP if STOP in context.legal_actions(actor) else \
        min(context.legal_actions(actor) or {STOP})


def _medivac_baseline(context: Context, actor: str, memory,
                      resources: _Resources) -> int:
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
        if resources.heal_taken.get(patient, actor) != actor:
            continue    # observed order or an earlier free healer owns it
        return action
    patients = sorted((a for a, u in context.snapshot.allies.items()
                       if u.alive and 0 < u.health < u.max_health
                       and not context.capabilities[a].healer
                       and resources.heal_taken.get(a) is None),
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
                       resources: _Resources) -> int:
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


def baseline_actions(context, memory, skills, reserved=None,
                     reserved_plans=(), only=None):
    """Back-compat shim: every caller routes through complete_plan."""
    if reserved and not reserved_plans:
        return complete_plan(context, memory, skills, [_Shim(reserved)])
    return complete_plan(context, memory, skills, reserved_plans)
