"""Skill eligibility for the three playable SMACv2 Zerg units."""

from .actions import DEAD_NOOP, STOP
from .context import distance
from .skills.common import best_move_toward
from .zerg_hydra_policy import zerg_hydra_plans


ZERG_UNITS = frozenset({"zergling", "hydralisk", "baneling"})

# Selected by Zerglings in the 6-ling/4-hydra contact-response trace.
ZERGLING_SKILLS = frozenset({
    "cohesive_search", "precombat_rally", "coordinated_engage", "focus_fire",
    "local_formation", "combat_reposition", "disengage_reenter", "regroup",
})


def is_zerg_roster(context):
    """Use the controlled roster, including dead units, independently of opponents."""
    allies = context.snapshot.allies
    return bool(allies) and all(unit.type_name in ZERG_UNITS for unit in allies.values())


class ZergRosterPolicy:
    """One decision window. Every joint candidate uses the same per-unit rules."""

    def __init__(self, context, skills, fingerprint):
        self.context = context
        self.fingerprint = fingerprint
        self.lings = tuple(a for a, cap in context.capabilities.items() if cap.key == 'zergling')
        self.focus = bool(getattr(skills.get('focus_fire'), 'enabled', False))
        self.named_moves = {}
        self.fallback = {a: zerg_fallback_action(context, a, self.named_moves)
                         for a, cap in context.capabilities.items()
                         if cap.key != 'zergling' and (cap.key != 'hydralisk' or not self.focus)}
        self.hydra_plans = {}

    def complete(self, plans, actions):
        """Recompute Hydra orders when a candidate changes its Lings' actions or targets."""
        ling_plans = [p for p in plans if all(a in self.lings for a in p.controlled_actors)]
        ling_actions = {a: actions[a] for a in self.lings}
        ling_targets = {}
        for plan in ling_plans:
            target = plan.goal_data.get('target_id', plan.goal)
            if target in self.context.snapshot.enemies:
                ling_targets.update(dict.fromkeys(plan.controlled_actors, target))
        key = (tuple(ling_actions.values()), tuple(sorted(ling_targets.items())))
        if key not in self.hydra_plans:
            self.hydra_plans[key] = (zerg_hydra_plans(
                self.context, ling_actions, ling_targets, self.fingerprint) if self.focus else [])
        orders = {**actions, **self.fallback}
        for plan in self.hydra_plans[key]:
            orders.update(plan.current_actions)
        return [*ling_plans, *self.hydra_plans[key]], orders


def ling_candidate(context, plan):
    """Keep whole legal skill assignments; partial atomic plans invalidate their claims."""
    if plan.primary_skill not in ZERGLING_SKILLS or plan.specialization_provider:
        return None
    if not plan.controlled_actors or any(
            context.snapshot.allies[actor].type_name != "zergling"
            for actor in plan.controlled_actors):
        return None
    return plan


def zerg_fallback_action(context, actor, named_moves=None):
    """Basic nearest-enemy policy for units with no enabled tactical skill."""
    unit = context.snapshot.allies[actor]
    if not unit.alive:
        return DEAD_NOOP
    enemies = sorted(context.snapshot.enemies.values(),
                     key=lambda enemy: (distance(unit.position, enemy.position), enemy.index))
    for enemy in enemies:
        attack = context.attack_action(actor, enemy.enemy_id)
        if attack is not None:
            return attack
    goal = enemies[0].position if enemies else context.snapshot.search_location
    move = best_move_toward(context, actor, goal)
    if move is not None:
        destination = context.move_destinations(actor)[move]
        if distance(destination, goal) < distance(unit.position, goal):
            if named_moves is not None and enemies and unit.type_name == 'baneling':
                # Preserve the actual origin of this order. Do not reconstruct
                # an enemy later from the common cardinal action or proximity.
                named_moves[actor] = (move, enemies[0].enemy_id)
            return move
    legal = context.legal_actions(actor)
    return STOP if STOP in legal else min(legal)
