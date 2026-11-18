"""Bounded, snapshot-local explosion defense; all amounts are predictions.

Target preferences are never prepaid damage. Only members of a complete
interception contribute fire, and final actions must preserve that fire.
"""
import math

from .burst import damage as burst_damage, center_reach
from .context import GAME_LOOPS_PER_SPEED_SECOND, distance

SAFETY_MARGIN = .5


def _enabled(context):
    return getattr(context, 'intercept_config', {}).get('enabled', True)


def _enemy_speed(context, enemy_id):
    value = context.snapshot.raw_state.get('unit_type_profiles', {}).get(
        str(context.snapshot.enemies[enemy_id].unit_type_id), {}).get('movement_speed_native')
    return float(value) if isinstance(value, (int, float)) and math.isfinite(value) and value >= 0 else None


def _entry(context, actor, enemy_id, point):
    reach, known = center_reach(context, context.snapshot.allies[actor])
    speed = _enemy_speed(context, enemy_id)
    gap = distance(point, context.snapshot.enemies[enemy_id].position) - reach
    if not known or speed is None:
        return None
    return 0. if gap <= 0 else gap * GAME_LOOPS_PER_SPEED_SECOND / speed if speed else float('inf')


def threat_facts(context):
    cached = getattr(context, '_intercept_threat_facts', None)
    if cached is not None:
        return cached
    facts = {'enabled': _enabled(context), 'bombers': []}
    if facts['enabled']:
        for enemy, cap in sorted(context.enemy_capabilities.items()):
            if not cap.suicide_attack:
                continue
            victims = []
            for actor, unit in sorted(context.snapshot.allies.items()):
                if not unit.alive or unit.is_flying:
                    continue
                entry = _entry(context, actor, enemy, unit.position)
                amount = burst_damage(context, actor, enemy_target=False)
                victims.append({'actor': actor, 'entry_h': entry / max(context.H, 1) if entry is not None else None,
                                'loss': min(amount, unit.effective_health) if amount is not None else None})
            known = [v for v in victims if v['entry_h'] is not None]
            facts['bombers'].append({'enemy': enemy, 'victims': victims,
                'entry_h': min((v['entry_h'] for v in known), default=float('inf')),
                'unknown_damage': any(v['loss'] is None or v['entry_h'] is None for v in victims),
                'known_loss_hp': sum(v['loss'] or 0. for v in known if v['entry_h'] <= 1.)})
    facts['bombers'].sort(key=lambda row: (row['entry_h'], -row['known_loss_hp'], row['enemy']))
    context._intercept_threat_facts = facts
    return facts


def explosion_loss(context, actor, point, horizon=None):
    """Conservative one-window reachable coverage, once per enemy explosion."""
    total = 0.
    for row in threat_facts(context)['bombers']:
        entry = _entry(context, actor, row['enemy'], point)
        amount = burst_damage(context, actor, enemy_target=False)
        if entry is not None and amount is not None and entry <= (context.H if horizon is None else horizon):
            total += min(amount, context.snapshot.allies[actor].effective_health)
    return total


def spread_valid(context, actor, action, memory=None):
    if action not in context.move_destinations(actor):
        return False
    if memory is not None and memory.movement_blockage(context, actor, action) >= .5:
        return False
    unit = context.snapshot.allies[actor]
    point = context.predict_position(actor, action).position
    middle = tuple((x + y) / 2. for x, y in zip(unit.position, point))
    before = explosion_loss(context, actor, unit.position)
    if before <= 0 or explosion_loss(context, actor, point) >= before:
        return False
    if explosion_loss(context, actor, middle) > before:
        return False
    # Moving out of one explosion must not enter another or fatal gunfire.
    for row in threat_facts(context)['bombers']:
        initial = _entry(context, actor, row['enemy'], unit.position)
        if initial is None:
            return False
        if initial > context.H and any((_entry(context, actor, row['enemy'], p) or 0.) <= context.H
                                      for p in (middle, point)):
            return False
    from .combat_exchange import danger_evidence
    origin = danger_evidence(context, actor, horizon_h=1.)
    for p in (middle, point):
        risk = danger_evidence(context, actor, p, horizon_h=1.)
        if risk['unknown'] or risk['supported_amount'] >= unit.effective_health \
                or risk['supported_amount'] > origin['supported_amount'] + .1:
            return False
    return True


def _safe_shooter(context, actor, enemy):
    cap = context.capabilities[actor]
    unit = context.snapshot.allies[actor]
    reach, known = center_reach(context, unit)
    if cap.combat_class != 'ranged' or not known or not unit.alive \
            or distance(unit.position, context.snapshot.enemies[enemy].position) <= reach + SAFETY_MARGIN:
        return None
    if not context.can_issue_attack(actor, enemy) or context.contact_status(actor, enemy) != 'in_range':
        return None
    estimate = context.fire_estimate(actor, enemy)
    if estimate.window_shots < 1 or estimate.damage_lo is None or estimate.damage_lo <= 0:
        return None
    return estimate


def entry_guard_actions(context, actor, memory=None):
    """Keep a productive route outside an observed enemy death-blast circle.

    Applies to final ATTACK implicit movement as well as MOVE. It never
    introduces a wait or retreat: if no safe firing/advancing alternative
    exists, ordinary exchange scoring retains control.
    """
    cache = getattr(context, '_blast_entry_guards', None)
    if cache is None:
        cache = context._blast_entry_guards = {}
    key = actor, id(memory)
    if key in cache:
        return cache[key]
    unit, cap = context.snapshot.allies[actor], context.capabilities[actor]
    if not _enabled(context) or not unit.alive or unit.is_flying or cap.healer or cap.suicide_attack:
        cache[key] = ()
        return ()
    bombers = [r['enemy'] for r in threat_facts(context)['bombers']
               if (entry := _entry(context, actor, r['enemy'], unit.position)) is not None and entry > 0.]
    from .joint_combat import action_position
    safe, firing_actions, unsafe = [], [], False
    targets = [e for e in context.snapshot.enemies
               if not context.enemy_capabilities[e].suicide_attack
               and context.effective_attack_reach(actor, e) is not None]
    for action in context.legal_actions(actor):
        command = context.decode(actor, action)
        if command.kind not in ('attack', 'move'):
            continue
        point, _ = action_position(context, actor, action)
        middle = tuple((x + y) / 2. for x, y in zip(unit.position, point))
        enters = any(_entry(context, actor, enemy, p) == 0.
                     for enemy in bombers for p in (middle, point))
        unsafe = unsafe or enters
        if enters:
            continue
        firing = command.kind == 'attack' and context.contact_status(actor, command.target) == 'in_range'
        advancing = any(distance(point, context.snapshot.enemies[e].position) + .05
                        < distance(unit.position, context.snapshot.enemies[e].position) for e in targets)
        if not firing and not advancing:
            continue
        if command.kind == 'move' and memory is not None \
                and memory.movement_blockage(context, actor, action) >= .5:
            continue
        safe.append(action)
        if firing:
            firing_actions.append(action)
    # The guard must not grant a new right to drop an existing melee shot.
    preferred = firing_actions if firing_actions and unit.weapon_cooldown_loops <= context.H else safe
    cache[key] = tuple(preferred) if unsafe else ()
    return cache[key]


def intercept_plan(context, memory=None, config=None):
    cache = getattr(context, '_intercept_plans', {})
    key = id(memory)
    if key in cache:
        return cache[key]
    plans, used = [], set()
    from .action_duties import rescue_actions
    from .unit_value_view import preservation_value
    for row in threat_facts(context)['bombers']:
        enemy = row['enemy']
        # Only imminent approaches get a duty; distant bombers retain ordinary fire candidates.
        if row['entry_h'] > 2. or row['unknown_damage']:
            continue
        pool = []
        for actor in sorted(context.snapshot.allies):
            if actor in used:
                continue
            estimate = _safe_shooter(context, actor, enemy)
            if estimate is not None and not rescue_actions(context, actor, memory):
                delay = estimate.effective_after_loops
                pool.append((float(context.H if delay is None else delay), -estimate.damage_lo, actor))
        pool.sort()
        remaining, shooters, kill_loops = context.snapshot.enemies[enemy].effective_health, [], 0.
        for delay, negative_damage, actor in pool:
            shooters.append((actor, context.attack_action(actor, enemy), -negative_damage))
            remaining += negative_damage
            kill_loops = max(kill_loops, delay)
            if remaining <= 0:
                break
        safe = bool(shooters) and remaining <= 0 and kill_loops + 1. < row['entry_h'] * max(context.H, 1)
        # Failed/late attempts do not reserve shooters or create hard attack duties.
        if not safe:
            shooters = []
        used.update(a for a, _, _ in shooters)
        spread = []
        if not safe:
            for victim in sorted(row['victims'], key=lambda v: (
                    v['entry_h'] if v['entry_h'] is not None else float('inf'),
                    -preservation_value(context, v['actor']), v['actor'])):
                actor = victim['actor']
                cap = context.capabilities[actor]
                if actor in used or cap.healer or cap.suicide_attack or victim['entry_h'] is None \
                        or victim['entry_h'] > 1. or not victim['loss']:
                    continue
                moves = [a for a in context.move_destinations(actor) if spread_valid(context, actor, a, memory)]
                if moves:
                    action = min(moves, key=lambda a: (
                        explosion_loss(context, actor, context.predict_position(actor, a).position),
                        distance(context.snapshot.allies[actor].position, context.predict_position(actor, a).position), a))
                    spread.append((actor, action))
                    used.add(actor)
        if shooters or spread:
            plans.append({**row, 'shooters': shooters, 'spread': spread, 'safe': safe,
                          'kill_loops': kill_loops, 'victims': [v['actor'] for v in row['victims']
                              if v['entry_h'] is not None and v['entry_h'] <= 1.]})
    cache[key] = plans
    context._intercept_plans = cache
    return plans


def actor_intercept_duty(context, actor, memory=None, config=None):
    for plan in intercept_plan(context, memory, config):
        choices = [(a, action, 'safe_intercept_window') for a, action, _ in plan['shooters'] if plan['safe']]
        choices += [(a, action, 'spread_from_enemy_burst_cover') for a, action in plan['spread']]
        for member, action, reason in choices:
            if member == actor:
                return {'required': True, 'actions': (action,), 'kind': 'intercept', 'reason': reason}
    guard = entry_guard_actions(context, actor, memory)
    if guard:
        return {'required': True, 'actions': guard, 'kind': 'intercept',
                'reason': 'productive_route_outside_enemy_death_blast'}
    return {'required': False, 'actions': (), 'kind': None, 'reason': 'no_intercept_duty'}


def final_valid(context, actions, actor, memory):
    """Validate whole shooting group after combination/repair, never prepaid fire."""
    for plan in intercept_plan(context, memory):
        if any(a == actor for a, _, _ in plan['shooters']):
            damage = 0.
            for member, command, _ in plan['shooters']:
                estimate = _safe_shooter(context, member, plan['enemy'])
                if actions.get(member) != command or estimate is None:
                    return False
                damage += estimate.damage_lo
            if damage < context.snapshot.enemies[plan['enemy']].effective_health:
                return False
            from .joint_combat import action_position
            for member, command in actions.items():
                unit = context.snapshot.allies[member]
                if not unit.alive or unit.is_flying:
                    continue
                point = action_position(context, member, command)[0]
                middle = tuple((x + y) / 2. for x, y in zip(unit.position, point))
                for position in (unit.position, middle, point):
                    entry = _entry(context, member, plan['enemy'], position)
                    if entry is None or entry <= plan['kill_loops'] + 1.:
                        return False
            return plan['safe']
        if any(a == actor for a, _ in plan['spread']):
            command = actions.get(actor)
            if not spread_valid(context, actor, command, memory):
                return False
            from .joint_combat import action_position, assess_joint
            positions = {a: action_position(context, a, c)[0] for a, c in actions.items()
                         if context.snapshot.allies[a].alive}
            if memory is not None and memory.movement_blockage(context, actor, command,
                    joint_actions=actions, joint_positions=positions) >= .5:
                return False
            unit = context.snapshot.allies[actor]
            if any(a != actor and context.snapshot.allies[a].is_flying == unit.is_flying
                   and distance(positions[actor], point) < unit.radius + context.snapshot.allies[a].radius
                   for a, point in positions.items()):
                return False
            exchange = assess_joint(context, actions)['team_exchange']['actors'].get(actor)
            return not exchange or exchange['after']['amount'] < unit.effective_health
    return actions.get(actor) in entry_guard_actions(context, actor, memory)


def summary(context, memory=None, limit=3):
    rows = []
    for plan in intercept_plan(context, memory)[:limit]:
        rows.append({'enemy': plan['enemy'], 'safe': plan['safe'],
                     'shooters': [a for a, _, _ in plan['shooters']],
                     'allocated_fire': sum(d for _, _, d in plan['shooters']),
                     'spread': [a for a, _ in plan['spread']], 'entry_h': plan['entry_h'],
                     'predicted_loss_hp': plan['known_loss_hp'], 'victims': plan['victims']})
    return {'enabled': _enabled(context), 'bombers': rows,
            'semantics': 'predicted reachable coverage, not observed damage'}
