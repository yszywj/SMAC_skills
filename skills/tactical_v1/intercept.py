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


def spread_moves(context, actor, memory=None):
    """Legal moves that reduce this ranged unit's imminent explosion loss."""
    cache = getattr(context, '_blast_spread_moves', None)
    if cache is None:
        cache = context._blast_spread_moves = {}
    key = actor, id(memory)
    if key not in cache:
        unit = context.snapshot.allies[actor]
        if context.capabilities[actor].combat_class == 'ranged' and unit.alive \
                and not unit.is_flying and explosion_loss(context, actor, unit.position) > 0.:
            cache[key] = tuple(a for a in context.move_destinations(actor)
                               if spread_valid(context, actor, a, memory))
        else:
            cache[key] = ()
    return cache[key]


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
    if not bombers:
        cache[key] = ()
        return ()
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
    # A route reducing imminent blast loss can be worthwhile even when an
    # ATTACK is legal. Keep both choices for the final exchange comparison.
    if unsafe:
        escapes = spread_moves(context, actor, memory)
        if memory is not None and firing_actions:
            budget = memory.fire_delay_available(context, actor)
            escapes = [a for a in escapes if memory._fire_delay_postponement(
                context, actor, a) <= budget + 1e-6]
        cache[key] = tuple(dict.fromkeys((*preferred, *escapes)))
    else:
        cache[key] = ()
    return cache[key]


def intercept_plan(context, memory=None, config=None):
    cache = getattr(context, '_intercept_plans', {})
    key = id(memory)
    if key in cache:
        return cache[key]
    plans, alternatives, committed = [], [], set()
    from .impact_timing import attack_impact_timing, joint_kill_time
    from .unit_value_view import preservation_value
    timing_guard = getattr(context, 'intercept_config', {}).get(
        'timing_guard_enabled', True)
    loss_control = getattr(context, 'intercept_config', {}).get(
        'loss_control_enabled', True)
    for row in threat_facts(context)['bombers']:
        enemy = row['enemy']
        # Only imminent approaches get a duty; distant bombers retain ordinary fire candidates.
        if row['entry_h'] > 2. or row['unknown_damage']:
            continue
        exposed = [v for v in row['victims']
                   if v['entry_h'] is not None and v['entry_h'] <= 0.]
        protected = [v for v in row['victims']
                     if v['entry_h'] is not None and 0. < v['entry_h'] <= 2.]
        pool = []
        for actor in sorted(context.snapshot.allies):
            if actor in committed:
                continue
            estimate = _safe_shooter(context, actor, enemy)
            # Rescue-eligible shooters are NOT unconditionally excluded here:
            # whether the group substitution is admissible is decided by the
            # joint rescue check and final group validation, not per member.
            if estimate is not None:
                delay = attack_impact_timing(context, actor, enemy).ready_after_loops
                pool.append((delay, -estimate.damage_lo, actor))
        pool.sort()
        remaining, shooters, kill_loops = context.snapshot.enemies[enemy].effective_health, [], 0.
        for delay, negative_damage, actor in pool:
            shooters.append((actor, context.attack_action(actor, enemy), -negative_damage))
            remaining += negative_damage
            kill_loops = max(kill_loops, delay)
            if remaining <= 0:
                break
        if remaining <= 0:
            # An early small hit can become redundant when a heavier shot
            # joins. Release such members without delaying the lethal volley.
            for member in sorted(shooters, key=lambda item: item[2]):
                if remaining + member[2] <= 0.:
                    remaining += member[2]
                    shooters.remove(member)
        # Timing split (task 1): cooldown end is when the gun CAN fire, not
        # when damage LANDS. Without wind-up/projectile data the kill-time
        # has a lower bound only — that can never prove a timely intercept.
        kill = joint_kill_time(context, [a for a, _, _ in shooters], enemy)
        conservative_kill = None
        if all(attack_impact_timing(context, a, enemy).impact_after_loops_upper
               is not None for a, _, _ in shooters) and shooters:
            conservative_kill = max(
                attack_impact_timing(context, a, enemy).impact_after_loops_upper
                for a, _, _ in shooters)
        if not timing_guard:
            # Legacy rollback comparison (cooldown treated as impact time).
            conservative_kill = kill_loops if shooters else None
        safe = bool(shooters) and remaining <= 0 \
            and conservative_kill is not None \
            and conservative_kill + 1. < row['entry_h'] * max(context.H, 1)
        # Loss-control interception: members are ALREADY exposed, so killing
        # the bomber now can still cap the team loss even though "safe"
        # timing can no longer be proven. The bomber's death explosion at its
        # current position stays priced (evaluator enemy_death_burst floor);
        # killing it never zeroes that cost here.
        loss_plan = bool(loss_control and exposed and shooters and remaining <= 0)
        plan_type = ('safe_intercept' if safe else 'loss_control_intercept'
                     if loss_plan else 'ordinary_joint_fire' if shooters else 'spread_only')
        firing = {a for a, _, _ in shooters}
        spread_options = []
        if safe:
            committed.update(firing)
        else:
            for victim in sorted(row['victims'], key=lambda v: (
                    v['entry_h'] if v['entry_h'] is not None else float('inf'),
                    -preservation_value(context, v['actor']), v['actor'])):
                actor = victim['actor']
                cap = context.capabilities[actor]
                if actor in committed or cap.healer or cap.suicide_attack or victim['entry_h'] is None \
                        or victim['entry_h'] > 1. or not victim['loss']:
                    continue
                moves = (spread_moves(context, actor, memory) if cap.combat_class == 'ranged'
                         else [a for a in context.move_destinations(actor)
                               if spread_valid(context, actor, a, memory)])
                if moves:
                    action = min(moves, key=lambda a: (
                        explosion_loss(context, actor, context.predict_position(actor, a).position),
                        distance(context.snapshot.allies[actor].position, context.predict_position(actor, a).position), a))
                    spread_options.append((actor, action))
        spread = [(a, move) for a, move in spread_options if a not in firing]
        if shooters or spread:
            primary = {**row, 'shooters': shooters, 'spread': spread,
                       'safe': safe, 'type': plan_type,
                       'kill_loops': kill_loops, 'kill_time_status': kill['timing_status'],
                       'conservative_kill_loops': conservative_kill,
                       'already_exposed': [v['actor'] for v in exposed],
                       'protected_members': [v['actor'] for v in protected],
                       'death_burst_priced_elsewhere': True,
                       'victims': [v['actor'] for v in row['victims']
                                   if v['entry_h'] is not None and v['entry_h'] <= 1.]}
            plans.append(primary)
            # A blocked spread member must not invalidate the whole firing
            # opportunity. The beam can combine these smaller contracts
            # with legal moves from other actors.
            if shooters and spread:
                alternatives.append(dict(primary, spread=[]))
            if len(spread) > 1:
                singles = sorted(spread, key=lambda item: (
                    explosion_loss(context, item[0],
                        context.predict_position(*item).position)
                    - explosion_loss(context, item[0], context.snapshot.allies[item[0]].position),
                    item[0]))
                for movement in singles[:2]:
                    alternatives.append(dict(primary, spread=[movement]))
            if shooters and spread_options:
                alternatives.append(dict(primary, shooters=[], spread=spread_options,
                                         type='spread_only', kill_loops=None,
                                         conservative_kill_loops=None, kill_time_status='no_fire_claim'))
            if remaining <= 0. and conservative_kill is None:
                for delay, negative_damage, actor in pool:
                    if delay == 0. and actor not in firing:
                        backup = (actor, context.attack_action(actor, enemy), -negative_damage)
                        # Leave the other members free. A redundant shot
                        # must be comparable even if one spread route fails.
                        alternatives.append(dict(primary, shooters=[*shooters, backup],
                                                 backup_shooter=actor, spread=[]))
                        break
    # Every imminent enemy gets its first candidate before spending the
    # atomic search slots on a redundant shot or a spread-only branch.
    plans.extend(alternatives)
    cache[key] = plans
    context._intercept_plans = cache
    return plans


def actor_intercept_duty(context, actor, memory=None, config=None):
    """Only SAFE interceptions become hard duties.

    loss_control_intercept and spread-only plans stay COMPETING atomic
    candidates: because the final ranking compares duty-violation counts
    BEFORE joint_score, granting them required=True here would lock an
    unproven plan ahead of the comparison. Their spread members with real
    emergency evidence are picked up by the ordinary rescue path instead.
    """
    for plan in intercept_plan(context, memory, config):
        if not plan.get('safe'):
            continue
        for member, action, _ in plan['shooters']:
            if member == actor:
                return {'required': True, 'actions': (action,),
                        'kind': 'intercept', 'reason': 'safe_intercept_window'}
    guard = entry_guard_actions(context, actor, memory)
    if guard:
        return {'required': True, 'actions': guard, 'kind': 'intercept',
                'reason': 'productive_route_outside_enemy_death_blast'}
    return {'required': False, 'actions': (), 'kind': None, 'reason': 'no_intercept_duty'}


def validate_intercept_plan(context, actions, plan, memory=None, report=None):
    """Whole-group validation of ONE candidate's OWN bomber contract.

    Reads ``plan.goal_data['validation_kind'] == 'bomber_intercept'`` and
    checks the DECLARED members against the actual final actions — never
    the first matching row of intercept_plan(), because candidates may
    differ in members, spread or target. Pure read-only: no coordinator
    calls, no search, no task mutations, no fire pre-reservation.

    Returns (passed: bool, reason: str, target: str).
    """
    data = getattr(plan, 'goal_data', None) or {}
    if data.get('validation_kind') != 'bomber_intercept':
        return True, 'not_bomber_contract', ''
    target = data.get('target_id')
    if target not in context.snapshot.enemies:
        return False, 'target_lost', target or ''
    shooter_actions = data.get('shooter_actions') or {}
    spread_actions = data.get('spread_actions') or {}
    # ---- shooters: alive, final action intact, legal, window-qualified ----
    cumulative = 0.
    for member, command in shooter_actions.items():
        unit = context.snapshot.allies.get(member)
        if unit is None or not unit.alive:
            return False, 'shooter_dead', target
        if actions.get(member) != command:
            return False, 'shooter_action_lost', target
        if command not in context.legal_actions(member):
            return False, 'shooter_action_illegal', target
        estimate = context.fire_estimate(member, target)
        if estimate.window_shots < 1:
            return False, 'shooter_lost_window', target
        if estimate.damage_lo is not None:
            cumulative += max(0., estimate.damage_lo)
    lethal_required = bool(data.get('requires_lethal_damage'))
    if lethal_required and cumulative \
            < context.snapshot.enemies[target].effective_health:
        return False, 'intercept_fire_insufficient', target
    # safe plans additionally need their conservative kill time to exist
    # and keep relevant members outside the kill window (same conservative
    # time everywhere; no cooldown fallback). Ordinary joint fire never
    # claims a kill and is not forced lethal here.
    if data.get('intercept_type') == 'safe_intercept':
        conservative = data.get('conservative_kill_loops')
        if conservative is None:
            return False, 'safe_timing_unknown', target
        from .joint_combat import action_position
        protected = set(data.get('protected_members') or ())
        for member, command in actions.items():
            unit = context.snapshot.allies.get(member)
            if not unit.alive or unit.is_flying:
                continue
            point = action_position(context, member, command)[0]
            middle = tuple((x + y) / 2. for x, y in zip(unit.position, point))
            for position in (unit.position, middle, point):
                entry = _entry(context, member, target, position)
                if entry is None or entry <= conservative + 1.:
                    if member in protected or member in shooter_actions \
                            or member in spread_actions:
                        return False, 'safe_window_violated', target
    # ---- spread: every declared member, against one shared joint state ----
    if spread_actions:
        from .joint_combat import action_position, assess_joint
        positions = {a: action_position(context, a, c)[0] for a, c in actions.items()
                     if context.snapshot.allies[a].alive}
        joint = assess_joint(context, actions)   # once; per-member reads below
        for member, command in spread_actions.items():
            unit = context.snapshot.allies.get(member)
            if unit is None or not unit.alive:
                return False, 'spread_member_dead', target
            if actions.get(member) != command:
                return False, 'spread_action_lost', target
            if command not in context.legal_actions(member):
                return False, 'spread_action_illegal', target
            if memory is not None and memory.movement_blockage(
                    context, member, command, joint_actions=actions,
                    joint_positions=positions) >= .5:
                return False, 'spread_route_blocked', target
            if any(a != member and context.snapshot.allies[a].is_flying == unit.is_flying
                   and distance(positions[member], point)
                   < unit.radius + context.snapshot.allies[a].radius
                   for a, point in positions.items()):
                return False, 'spread_position_overlap', target
            exchange = joint['team_exchange']['actors'].get(member)
            if exchange and exchange['after']['amount'] >= unit.effective_health:
                return False, 'spread_route_still_lethal', target
    return True, 'bomber_contract_valid', target


def final_valid(context, actions, actor, memory):
    """Validate the hard duty issued by actor_intercept_duty.

    Optional interception contracts are checked by validate_intercept_plan
    when selected. They do not constrain a member's independent entry guard.
    """
    for plan in intercept_plan(context, memory):
        if plan['safe'] and any(a == actor for a, _, _ in plan['shooters']):
            damage = 0.
            for member, command, _ in plan['shooters']:
                estimate = _safe_shooter(context, member, plan['enemy'])
                if actions.get(member) != command or estimate is None:
                    return False
                damage += estimate.damage_lo
            if damage < context.snapshot.enemies[plan['enemy']].effective_health:
                return False
            conservative = plan.get('conservative_kill_loops')
            if conservative is None:
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
                    if entry is None or entry <= conservative + 1.:
                        return False
            return True
    return actions.get(actor) in entry_guard_actions(context, actor, memory)


def summary(context, memory=None, limit=3):
    rows, seen = [], set()
    for plan in intercept_plan(context, memory):
        if len(rows) >= limit:
            break
        if plan['enemy'] in seen:
            continue
        seen.add(plan['enemy'])
        rows.append({'enemy': plan['enemy'], 'type': plan.get('type'),
                     'safe': plan['safe'],
                     'kill_time_status': plan.get('kill_time_status'),
                     'already_exposed': plan.get('already_exposed', []),
                     'protected_members': plan.get('protected_members', []),
                     'shooters': [a for a, _, _ in plan['shooters']],
                     'allocated_fire': sum(d for _, _, d in plan['shooters']),
                     'spread': [a for a, _ in plan['spread']], 'entry_h': plan['entry_h'],
                     'predicted_loss_hp': plan['known_loss_hp'], 'victims': plan['victims']})
    return {'enabled': _enabled(context), 'bombers': rows,
            'semantics': 'predicted reachable coverage, not observed damage; '
                         'kill timing has a lower bound only unless stated otherwise'}
