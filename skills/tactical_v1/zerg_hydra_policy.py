"""Focus-fire orders for hydralisks in the Zerg tactical profile."""
from .actions import STOP
from .battle_line import _observed_target
from .context import CONTACT_UNKNOWN, IN_RANGE, distance
from .contracts import Commitment, EffectClaim, ResourceClaims
from .skills.common import make_plan, space_claim


def _rear_point(context, actor, ling_actions, ling_targets):
    units = context.snapshot.allies
    enemies = context.snapshot.enemies
    origin = units[actor].position
    lings = (u for u in units.values() if u.alive and u.type_name == 'zergling')
    ling = min(lings, key=lambda u: (distance(u.position, origin), u.index), default=None)
    if ling is None:
        return None, None

    order = ling_actions.get(ling.actor)
    semantic = context.decode(ling.actor, order) if order is not None else None
    targets = (semantic.target if semantic and semantic.kind == 'attack' else None,
               ling_targets.get(ling.actor), _observed_target(context, ling.actor))
    target = next((enemy for enemy in targets if enemy in enemies), None)
    if target is not None:
        aim = enemies[target].position
    elif enemies:
        nearest = min(enemies.values(), key=lambda e: (distance(ling.position, e.position), e.index))
        aim = nearest.position
    elif semantic and semantic.kind == 'move':
        # A searching ling can establish a forward direction before first contact.
        aim = context.move_destinations(ling.actor).get(order)
    else:
        aim = None
    if aim is None:
        return ling.actor, None
    length = distance(ling.position, aim)
    if length < 1e-6:
        return ling.actor, None
    step = min((distance(origin, p) for p in context.move_destinations(actor).values()),
               default=0.)
    clearance = ling.radius + units[actor].radius + step
    rear = tuple(p + (p - q) * clearance / length for p, q in zip(ling.position, aim))
    return ling.actor, rear


def _move_toward(context, actor, point):
    moves = context.move_destinations(actor)
    legal = context.legal_actions(actor)
    hold = STOP if STOP in legal else min(legal)
    if point is None or not moves:
        return hold
    best = min(moves, key=lambda a: (distance(moves[a], point), a))
    if STOP in legal and distance(moves[best], point) >= distance(
            context.snapshot.allies[actor].position, point) - 1e-6:
        return STOP
    return best


def _reach(context, actor, target):
    reach = context.effective_attack_reach(actor, target)
    if reach is not None and reach >= 0:
        return reach
    # Older traces may omit the pair's native range or command-distance cap.
    # Recover weapon geometry from the shooter's own catalog and target domain.
    snap = context.snapshot
    ally, enemy = snap.allies[actor], snap.enemies[target]
    relation = snap.relations.get(actor, {}).get(target)
    if relation and relation.weapon_range_status == 'no_compatible_weapon':
        return None
    native = relation.native_weapon_range if relation else None
    if native is None:
        profile = snap.raw_state.get('unit_type_profiles', {}).get(str(ally.unit_type_id), {})
        domain = 'air' if enemy.is_flying else 'ground'
        ranges = (weapon['range'] for weapon in profile.get('weapons', ())
                  if weapon.get('target_type') in (domain, 'any')
                  and isinstance(weapon.get('range'), (int, float)))
        native = max(ranges, default=None) if profile.get('known') else None
    if native is None:
        return None
    physical = native + ally.radius + enemy.radius
    limit = snap.raw_state.get('relations', {}).get(actor, {}).get(target, {}).get(
        'smac_attack_distance_limit')
    if limit is None:
        limit = snap.raw_state.get('action_contract', {}).get(
            'candidate_center_distance_by_type', {}).get(ally.type_name)
    return min(physical, limit) if isinstance(limit, (int, float)) else physical


def _in_range(context, actor, target):
    status = context.contact_status(actor, target)
    if status != CONTACT_UNKNOWN:
        return status == IN_RANGE
    reach = _reach(context, actor, target)
    return reach is not None and context.distance(actor, target) <= reach


def _near_hydra(context, actor):
    candidates = []
    for target, enemy in context.snapshot.enemies.items():
        if enemy.type_name != 'hydralisk':
            continue
        reach = _reach(context, actor, target)
        separation = distance(context.snapshot.allies[actor].position, enemy.position)
        if reach is not None and reach > 0. and separation <= 1.3 * reach + 1e-6:
            candidates.append((separation, enemy.index, target))
    return min(candidates)[2] if candidates else None


def zerg_hydra_plans(context, ling_actions, ling_targets, fingerprint):
    """Allocate one shot or positioning order per living hydra after ling selection."""
    enemies = context.snapshot.enemies
    remaining = {e: unit.effective_health for e, unit in enemies.items()}
    for ling, action in ling_actions.items():
        order = context.decode(ling, action)
        if order.kind == 'attack' and order.target in remaining:
            remaining[order.target] -= context.fire_estimate(ling, order.target).damage_est or 0.

    plans = []
    for actor, unit in sorted(context.snapshot.allies.items(), key=lambda pair: pair[1].index):
        if not unit.alive or unit.type_name != 'hydralisk':
            continue
        in_range = [e for e in context.attackable_enemies(actor) if _in_range(context, actor, e)]
        effects = []
        fire = ()
        if unit.weapon_ready and unit.weapon_cooldown_loops <= 0. and in_range:
            # Concentrate a volley until assigned damage covers the current HP.
            # An in-range enemy hydra retains priority even over a killable ling.
            target = min(in_range, key=lambda e: (enemies[e].type_name != 'hydralisk',
                                                   remaining[e] <= 0., abs(remaining[e]),
                                                   context.distance(actor, e), enemies[e].index))
            action = context.attack_action(actor, target)
            shot = context.fire_estimate(actor, target)
            amount = shot.damage_est
            effects.append(EffectClaim('damage', target, amount, context.H, shot.evidence,
                                       'focus_fire_volley', lower=shot.damage_lo, upper=shot.damage_hi))
            if amount is not None:
                remaining[target] -= amount
                fire = ((target, amount),)
            phase, kind, goal = 'attacking', 'enemy', target
            data = {'target_id': target}
        else:
            target = _near_hydra(context, actor)
            if target is not None and target not in in_range:
                goal, kind, phase = target, 'enemy', 'approach_hydralisk'
                point = enemies[target].position
                data = {'target_id': target}
            else:
                ling, point = _rear_point(context, actor, ling_actions, ling_targets)
                goal, kind = (ling, 'ally') if ling else (actor, 'internal')
                phase = 'follow_zergling' if point is not None else 'hold'
                data = {'anchor': ling, 'position': point}
            action = _move_toward(context, actor, point)
        actions = {actor: action}
        claims = ResourceClaims(controls=(actor,), fire=fire, space=(space_claim(context, actor, action),))
        plans.append(make_plan(
            'focus_fire', (actor,), goal, actions, effects, claims, Commitment(phase),
            {}, context.snapshot.snapshot_id, fingerprint, goal_kind=kind, goal_data=data,
            rationale=phase))
    return plans
