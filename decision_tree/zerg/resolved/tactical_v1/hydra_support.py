"""Counter-fire and screening candidates for local Hydra support."""
from .action_duties import ready_hydralisk_shots
from .context import STOP, distance
from .contracts import Commitment, Dependency, EffectClaim, HEURISTIC, ResourceClaims
from .deployment import SUPPORT_ETA_H, _fire_eta_h, safe_step
from .memory import TacticalMemory
from .skills.common import make_plan

SUPPORT_RADIUS = 9.0
MAX_PAIRS = 2
MAX_SUPPORTERS = 2
# Extra ordinary threats a firing-position grab may accept. Crossing into
# several fresh firing lines to open one shot is never worth it.
GRAB_NEW_THREATS = 1


def _can_fire_now(context, actor, enemy_id):
    return (context.can_issue_attack(actor, enemy_id)
            and context.fire_estimate(actor, enemy_id).window_shots >= 1)


def _enemy_covers(context, actor, target):
    if context.enemy_capabilities[target].suicide_attack:
        return False
    relation = context.snapshot.relations.get(actor, {}).get(target)
    if relation is not None:
        if relation.enemy_weapon_range_status in ('out_of_range', 'no_compatible_weapon'):
            return False
        if relation.in_enemy_weapon_range is not None:
            return relation.in_enemy_weapon_range
        if relation.enemy_weapon_range_status == 'in_range':
            return True
    unit, enemy = context.snapshot.allies[actor], context.snapshot.enemies[target]
    reach = context._enemy_weapon_range(target, unit.is_flying)
    return reach is not None and distance(unit.position, enemy.position) <= reach + unit.radius + enemy.radius


def _threats_of(context, actor):
    return sorted((e for e in context.snapshot.enemies if _enemy_covers(context, actor, e)),
                  key=lambda e: (context.distance(actor, e), e))


def pressured_hydras(context):
    """Rank observed or catalog-supported ordinary weapon coverage by urgency."""
    rows = []
    for actor, unit in context.snapshot.allies.items():
        if unit.alive and unit.type_name == 'hydralisk':
            threats = _threats_of(context, actor)
            if threats:
                rows.append((unit.health_fraction, context.distance(actor, threats[0]), actor, threats))
    return [(actor, threats) for _, _, actor, threats in sorted(rows)]


def _safe_moves(context, memory, actor):
    origin = context.snapshot.allies[actor].position
    origin_bombers = set(context.risk_at(actor, origin).effect_weapon_near)
    moves = []
    for action in sorted(context.move_destinations(actor)):
        if memory is not None and memory.movement_blockage(context, actor, action) >= .5:
            continue
        end = context.predict_position(actor, action).position
        middle = tuple((x + y) * .5 for x, y in zip(origin, end))
        if safe_step(context, actor, end) and all(
                not set(context.risk_at(actor, p).effect_weapon_near) - origin_bombers
                for p in (middle, end)):
            moves.append((action, end))
    return moves


def support_paths_overlap(first, first_end, second, second_end):
    """Synchronized closest approach, permitting existing bodies to separate."""
    before = tuple(x - y for x, y in zip(first.position, second.position))
    after = tuple(x - y for x, y in zip(first_end, second_end))
    clearance = min(first.radius + second.radius, distance(first.position, second.position))
    return TacticalMemory._segment_distance((0., 0.), before, after) < clearance - .02


def _pair_clear(context, protected, supporter, end_a, end_b):
    a, b = context.snapshot.allies[protected], context.snapshot.allies[supporter]
    return (distance(end_a, end_b) <= SUPPORT_RADIUS
            and not support_paths_overlap(a, end_a, b, end_b))


def support_path_clear(context, actor, end, actions):
    """Check a reinforcement path against the actual orders of all ground allies."""
    from .joint_combat import action_position
    if not hasattr(context, '_support_path_conflicts'):
        context._support_path_conflicts = {}
    cache = context._support_path_conflicts
    units = context.snapshot.allies
    for peer, other in units.items():
        if peer == actor or not other.alive or other.is_flying:
            continue
        key = actor, tuple(end), peer, actions[peer]
        if key not in cache:
            peer_end = action_position(context, peer, actions[peer])[0]
            cache[key] = support_paths_overlap(units[actor], end, other, peer_end)
        if cache[key]:
            return False
    return True


def _support_orders(context, supporter, target, shots, moves):
    """Order one-shot, one-window, then bounded two-window closing options."""
    unit = context.snapshot.allies[supporter]
    if target in shots:
        return [((0, 0.), context.attack_action(supporter, target), unit.position)]
    if shots or _fire_eta_h(context, supporter, (target,)) > SUPPORT_ETA_H:
        return []
    reach = context.effective_attack_reach(supporter, target)
    if reach is None or reach < 0:
        return []
    enemy = context.snapshot.enemies[target]
    before = max(0., distance(unit.position, enemy.position) - reach)
    orders = []
    for action, end in moves:
        gap = max(0., distance(end, enemy.position) - reach)
        if gap < before - .05:
            orders.append(((1 if gap <= 1e-6 else 2, gap), action, end))
    return sorted(orders)


def _withdrawals(context, actor, threats, moves):
    """Prefer leaving physical coverage, then increasing nearest threat distance."""
    unit = context.snapshot.allies[actor]
    positions = [context.snapshot.enemies[t].position for t in threats]
    before = min(distance(unit.position, p) for p in positions)
    options = []
    for action, end in moves:
        gap = min(distance(end, p) for p in positions)
        if gap <= before + .05:
            continue
        remaining = set(context.risk_at(actor, end).threats)
        options.append((len(remaining), -gap, action, end))
    return sorted(options)


def _isolated_shooter_routes(context, memory):
    """First support steps for a pressured Hydra with sole contact on a threat."""
    units = context.snapshot.allies
    hydras = [a for a, u in units.items() if u.alive and u.type_name == 'hydralisk']
    routes = []
    if len(hydras) < 2:
        return routes
    from .action_duties import actor_duty
    moves = {}
    firing = {a for a in hydras if any(_can_fire_now(context, a, e)
                                        for e in context.attackable_enemies(a))}
    for lead, threats in pressured_hydras(context):
        for target in threats:
            if context.snapshot.enemies[target].type_name != 'hydralisk':
                continue
            shooters = [a for a in hydras if context.can_issue_attack(a, target)
                        and context.contact_status(a, target) == 'in_range']
            if lead not in shooters:
                continue
            damage = context.fire_estimate(lead, target)
            if damage.window_shots and damage.damage_lo is not None and (
                    damage.damage_lo >= context.snapshot.enemies[target].effective_health):
                continue
            if len(shooters) > 1:
                # A reinforcement that really moved last window can now
                # complete its order, provided THIS observation permits fire.
                previous = getattr(memory, 'last_hydra_support_move', None)
                if previous and previous[:2] == (lead, target) and previous[2] in shooters:
                    peer = previous[2]
                    shot = context.attack_action(peer, target)
                    duty = actor_duty(context, peer, memory)
                    if _can_fire_now(context, peer, target) and (
                            not duty['required'] or shot in duty['actions']):
                        routes.append(((0, units[lead].health_fraction, 0.), lead,
                                       target, peer, shot, units[peer].position))
                continue
            for peer in hydras:
                if peer == lead or peer in firing or distance(
                        units[lead].position, units[peer].position) > SUPPORT_RADIUS:
                    continue
                if peer not in moves:
                    duty = actor_duty(context, peer, memory)
                    options = _safe_moves(context, memory, peer)
                    if duty['required'] and duty['kind'] != 'entry':
                        options = [(cmd, end) for cmd, end in options if cmd in duty['actions']]
                    before = set(context.risk_at(peer, units[peer].position).threats)
                    moves[peer] = [(cmd, end) for cmd, end in options
                                   if len(set(context.risk_at(peer, end).threats) - before) <= GRAB_NEW_THREATS]
                for rank, cmd, end in _support_orders(context, peer, target, (), moves[peer]):
                    # Prefer an open firing lane, then an urgent lead and
                    # the smallest residual gap. The next mask confirms it.
                    key = (rank[0], units[lead].health_fraction, rank[1])
                    routes.append((key, lead, target, peer, cmd, end))
    return sorted(routes)


def isolated_shooter_support(context, memory, actions):
    """Choose one collision-free reinforcement against the final team orders."""
    if (getattr(context, 'formation_orders', {}) or
            not getattr(context, 'ranged_fire_formation_enabled', False)):
        return None
    if not hasattr(context, '_isolated_shooter_routes'):
        context._isolated_shooter_routes = _isolated_shooter_routes(context, memory)
    best = None
    for rank, lead, target, peer, cmd, end in context._isolated_shooter_routes:
        if best and rank > best[0]:
            break
        order = context.decode(lead, actions[lead])
        if order.kind == 'attack' and order.target != target:
            continue
        if rank[0] != 0 and not support_path_clear(context, peer, end, actions):
            continue
        row = (rank, actions[peer] != cmd, lead, target, peer, cmd)
        if best is None or row < best:
            best = row
    if best is None:
        return None
    _, needs_change, lead, target, peer, cmd = best
    return {'lead': lead, 'target': target, 'supporter': peer, 'action': cmd,
            'satisfied': not needs_change}


def _fire_grabs(context, memory, actor, target, moves):
    """One-step endpoints that OPEN a legal firing position vs ``target``.

    A grab may accept at most GRAB_NEW_THREATS additional ordinary threats
    at the endpoint: crossing into several fresh firing lines to enable one
    shot is rejected outright. The opened shot is a prediction — the next
    real action menu confirms it; no damage or kill is claimed here.
    Returns ordered (rank, action, end) rows.
    """
    unit = context.snapshot.allies[actor]
    enemy = context.snapshot.enemies[target]
    reach = context.effective_attack_reach(actor, target)
    if reach is None or reach < 0:
        return []
    before_threats = set(context.risk_at(actor, unit.position).threats)
    origin_gap = max(0., distance(unit.position, enemy.position) - reach)
    rows = []
    for action, end in moves:
        opportunity = context.projected_attack_opportunity(actor, target, end)
        if not opportunity.available:
            continue                    # still no legal firing position there
        new_threats = len(set(context.risk_at(actor, end).threats) - before_threats)
        if new_threats > GRAB_NEW_THREATS:
            continue                    # charging into a focus fire line
        gap = max(0., distance(end, enemy.position) - reach)
        if gap >= origin_gap - 1e-6:
            continue                    # not actually approaching the band
        rows.append(((0 if opportunity.ready else 1, new_threats, gap),
                     action, end))
    return sorted(rows)


def approach_support_plans(context, memory, fingerprint):
    """Compare the incumbent entry step with a locally covered entry or a brief wait."""
    from .baseline import complete_plan
    from .joint_combat import action_position
    from .local_entry import wait_remaining_loops

    units = context.snapshot.allies
    hydras = sorted(a for a, u in units.items() if u.alive and u.type_name == 'hydralisk')
    if len(hydras) < 2:
        return []
    incumbent = getattr(context, 'baseline_actions', None)
    if incumbent is None:
        incumbent = complete_plan(context, memory, None)
    covered = {actor for actor, _ in pressured_hydras(context)}
    shots = {a: any(_can_fire_now(context, a, e) for e in context.attackable_enemies(a))
             for a in hydras}
    plans = []
    for actor in hydras:
        if actor in covered or shots[actor]:
            continue
        action = incumbent[actor]
        origin = units[actor].position
        end = action_position(context, actor, action)[0]
        if distance(origin, end) < .05 or not safe_step(context, actor, end):
            continue
        danger = [e for e, enemy in context.snapshot.enemies.items()
                  if enemy.type_name == 'hydralisk' and _endpoint_covered(context, actor, end, e)]
        if not danger:
            continue
        safe = _safe_moves(context, memory, actor)
        if context.decode(actor, action).kind == 'move' and action not in dict(safe):
            continue
        for target in sorted(danger, key=lambda e: context.distance(actor, e)):
            variants = [('proceed', action, end)]
            if STOP in context.legal_actions(actor) and safe_step(context, actor, origin) and not (
                    context.risk_at(actor, origin).effect_weapon_near) and (
                    wait_remaining_loops(context, memory, actor, max_h=1.) >= context.H):
                variants.append(('hold', STOP, origin))
            variants.extend(('sidestep', move, point) for move, point in safe
                            if not context.risk_at(actor, point).threats
                            and not _endpoint_covered(context, actor, point, target))
            best = {}
            for peer in hydras:
                if peer == actor or shots[peer] or distance(origin, units[peer].position) > SUPPORT_RADIUS:
                    continue
                if context.projected_attack_opportunity(peer, target, units[peer].position).available:
                    continue
                for fill, point in _safe_moves(context, memory, peer):
                    shot = context.projected_attack_opportunity(peer, target, point)
                    if not (shot.available and shot.ready):
                        continue
                    for label, order, dest in variants:
                        if not _pair_clear(context, actor, peer, dest, point):
                            continue
                        rank = (len(context.risk_at(peer, point).threats),
                                distance(dest, context.snapshot.enemies[target].position),
                                distance(point, context.snapshot.enemies[target].position), peer, fill)
                        if label not in best or rank < best[label][0]:
                            best[label] = rank, peer, {actor: order, peer: fill}
            if best:
                for label, (_, peer, orders) in best.items():
                    plans.append(_support_plan(context, fingerprint, actor, peer, target,
                                               orders, False, prospective=True, label=label))
                break
    return plans


def _endpoint_covered(context, actor, endpoint, enemy_id):
    """Would ``endpoint`` sit inside enemy ``enemy_id``'s coverage?"""
    enemy = context.snapshot.enemies[enemy_id]
    if enemy.is_flying:
        return False
    reach = context._enemy_weapon_range(enemy_id, False)
    if reach is None:
        return False
    return distance(endpoint, enemy.position) <= float(reach) + enemy.radius + context.snapshot.allies[actor].radius


def _support_plan(context, fingerprint, actor, supporter, target, actions,
                  withdrawing, prospective=False, label=None):
    effects, fire = [], []
    for member, action in actions.items():
        order = context.decode(member, action)
        if order.kind == 'attack':
            estimate = context.fire_estimate(member, order.target)
            if estimate.window_shots < 1:
                effects.append(EffectClaim('coverage', order.target, None, context.H, HEURISTIC,
                                           'continuing_attack_on_cooldown'))
                continue
            effects.append(EffectClaim('damage', order.target, estimate.damage_est, context.H,
                estimate.evidence, 'shared_damage_model_window_shot',
                lower=estimate.damage_lo, upper=estimate.damage_hi))
            if estimate.damage_lo is not None:
                fire.append((order.target, estimate.damage_lo))
        elif order.kind == 'move':
            if member == actor:
                reason = 'pressured_withdraw_step' if withdrawing \
                    else 'firing_position_grab_not_confirmed_fire'
            else:
                reason = 'support_approach_not_confirmed_fire'
            effects.append(EffectClaim('position', member, None, context.H, HEURISTIC, reason))
    approaching = any(context.decode(m, a).kind == 'move' and (m != actor or not withdrawing)
                      for m, a in actions.items())
    plan = make_plan('coordinated_engage', actions, target, actions, effects,
        ResourceClaims(controls=tuple(actions), fire=tuple(fire)), Commitment('supporting', 0, context.H),
        {}, context.snapshot.snapshot_id, fingerprint, atomic=True, goal_kind='enemy',
        dependencies=(Dependency('member', tuple(actions)), Dependency('target_valid', (target,))),
        goal_data={'objective': {'kind': 'counter_fire_support', 'target': target},
                   'support_pair': [actor, supporter] if supporter else [actor], 'prospective_fire': prospective or approaching,
                   'withdrawing': withdrawing,
                   'approach_label': label},
        rationale='counter-fire support vs threat of pressured ' + actor
                  + (' (withdraw)' if withdrawing else '')
                  + (f' [{label}]' if label else ''))
    plan.proposal_id += '-' + actor
    return plan


def counter_fire_support_plans(context, memory, fingerprint):
    pressured = pressured_hydras(context)
    if not pressured:
        return []
    hydras = [a for a, u in context.snapshot.allies.items() if u.alive and u.type_name == 'hydralisk']
    shots = {a: [e for e in context.attackable_enemies(a) if _can_fire_now(context, a, e)] for a in hydras}
    moves = {}

    def safe_moves(actor):
        if actor not in moves:
            moves[actor] = _safe_moves(context, memory, actor)
        return moves[actor]

    plans_by_actor = []
    reserved = False
    for actor, threats in pressured:
        unanswered = not any(e in threats for peer, targets in shots.items()
                             if peer != actor for e in targets)
        if len(plans_by_actor) == MAX_PAIRS and not unanswered:
            continue
        unit = context.snapshot.allies[actor]
        # [3] Independent firing-position grabs: an exposed hydra that
        # cannot legally fire still compares a forward/diagonal step that
        # opens its own shot next window (bounded extra exposure) against
        # the withdrawal options — with or without any supporter.
        grabs = {} if ready_hydralisk_shots(context, actor) else {
            target: _fire_grabs(context, memory, actor, target, safe_moves(actor))
            for target in threats}
        retreats = () if ready_hydralisk_shots(context, actor) else \
            _withdrawals(context, actor, threats, safe_moves(actor))
        candidates = []
        for supporter in hydras:
            other = context.snapshot.allies[supporter]
            if supporter == actor or distance(unit.position, other.position) > SUPPORT_RADIUS:
                continue
            for target in threats:
                b_orders = _support_orders(context, supporter, target, shots[supporter],
                                           () if shots[supporter] else safe_moves(supporter))
                if not b_orders:
                    continue
                a_target = next((e for e in shots[actor] if e in threats), None)
                if target in shots[actor]:
                    a_target = target
                elif a_target is None and shots[actor]:
                    a_target = shots[actor][0]
                stationary = context.attack_action(actor, a_target) if a_target else None
                if (stationary is None and context.can_issue_attack(actor, target)
                        and context.contact_status(actor, target) == 'in_range'):
                    stationary = context.attack_action(actor, target)
                if stationary is None and STOP in context.legal_actions(actor):
                    stationary = STOP
                variants = {}
                for rank, b_action, b_end in b_orders:
                    if stationary is not None and 'stationary' not in variants and _pair_clear(
                            context, actor, supporter, unit.position, b_end):
                        variants['stationary'] = (rank, {actor: stationary, supporter: b_action}, False)
                    if 'grab' not in variants:
                        for _, grab_action, grab_end in grabs.get(target, ())[:2]:
                            if _pair_clear(context, actor, supporter, grab_end, b_end):
                                variants['grab'] = (rank, {actor: grab_action,
                                                           supporter: b_action}, False)
                                break
                    # Paying for a ready A shot is an alternative only when B
                    # supplies fire now. B closing alone cannot replace it.
                    if (not a_target or rank[0] == 0) and 'withdraw' not in variants:
                        for _, _, a_action, a_end in retreats:
                            if _pair_clear(context, actor, supporter, a_end, b_end):
                                variants['withdraw'] = (rank, {actor: a_action, supporter: b_action}, True)
                                break
                if variants:
                    best = min(row[0] for row in variants.values())
                    candidates.append((best, context.distance(actor, target), supporter, target, variants))
        # Reserve budget for distinct feasible supporters, not candidates
        # that merely passed a radial ETA before routes were checked.
        selected = set()
        actor_plans = []
        for _, _, supporter, target, variants in sorted(candidates, key=lambda row: row[:4]):
            if supporter in selected:
                continue
            selected.add(supporter)
            actor_plans.extend(
                _support_plan(context, fingerprint, actor, supporter, target,
                              actions, withdrawing)
                for _, actions, withdrawing in variants.values())
            if len(selected) >= MAX_SUPPORTERS:
                break
        if not actor_plans and not ready_hydralisk_shots(context, actor):
            # [3] No feasible supporter right now: A still gets its own
            # independent comparison (grab vs withdrawal) as single-actor
            # candidates, so it never waits in place just for company.
            solo = sorted((rank, context.distance(actor, target), target, move)
                          for target, rows in grabs.items() for rank, move, _ in rows)
            for _, _, target, grab_action in solo[:1]:
                actor_plans.append(_support_plan(
                    context, fingerprint, actor, None, target,
                    {actor: grab_action}, False, prospective=True, label='solo_grab'))
            for _, _, retreat_action, _ in retreats[:1]:
                actor_plans.append(_support_plan(
                    context, fingerprint, actor, None, threats[0],
                    {actor: retreat_action}, True, label='solo_withdraw'))
        if actor_plans:
            # Keep the earliest feasible urgent group and reserve one place
            # for the first uncovered group with an executable support route.
            if len(plans_by_actor) == MAX_PAIRS:
                plans_by_actor[-1] = actor_plans
            else:
                plans_by_actor.append(actor_plans)
            reserved = reserved or unanswered
            if len(plans_by_actor) == MAX_PAIRS and reserved:
                break
    return [plan for group in plans_by_actor for plan in group]


def _ready_melee(context, ling):
    """Keep an existing legal melee contact, including during cooldown."""
    return any(context.contact_status(ling, e) == 'in_range'
               for e in context.attackable_enemies(ling))


def _guard_step(context, memory, ling, goal, shooter=None):
    """Find a progressing safe route, preserving the covering shooter's space."""
    unit = context.snapshot.allies[ling]
    origin = unit.position
    choices = []
    for action, point in _safe_moves(context, memory, ling):
        gap = distance(point, goal)
        if gap >= distance(origin, goal) - .05:
            continue
        if shooter is not None and support_paths_overlap(unit, point, shooter, shooter.position):
            continue
        choices.append((gap, action))
    return min(choices)[1] if choices else None


def _covered_flank(context, memory, fingerprint, pressured, lings):
    import math
    from .skills.common import space_claim
    units, enemies = context.snapshot.allies, context.snapshot.enemies
    hydras = [a for a, u in sorted(units.items()) if u.alive and u.type_name == 'hydralisk']
    for aid, threats in pressured[:MAX_PAIRS]:
        for front in threats:
            if enemies[front].is_flying:
                continue
            shooters = [h for h in hydras if _can_fire_now(context, h, front)]
            if not shooters:
                continue
            shooter = min(shooters, key=lambda h: (h != aid, context.distance(h, front), h))
            target = enemies[front].position
            base = math.atan2(units[shooter].position[1] - target[1],
                              units[shooter].position[0] - target[0])
            for ling in sorted(lings, key=lambda lid: (context.distance(lid, front), lid)):
                reach = context.effective_attack_reach(ling, front)
                if reach is None or reach <= 0. or _fire_eta_h(context, ling, (front,)) > SUPPORT_ETA_H:
                    continue
                if sum((x - t) * (y - t) for x, y, t in
                       zip(units[ling].position, units[shooter].position, target)) < 0.:
                    continue  # do not cross through the enemy to reach the friendly flank
                radius = max(0., reach - .05)
                # Both flanks lie on the shooter's side and inside real melee reach.
                flanks = [(target[0] + radius * math.cos(base + turn),
                           target[1] + radius * math.sin(base + turn)) for turn in (-.65, .65)]
                for point in sorted(flanks, key=lambda p: distance(units[ling].position, p)):
                    move = _guard_step(context, memory, ling, point, units[shooter])
                    if move is None:
                        continue
                    shot = context.attack_action(shooter, front)
                    members = {ling: move, shooter: shot}
                    estimate = context.fire_estimate(shooter, front)
                    fire = ((front, estimate.damage_lo),) if estimate.damage_lo is not None else ()
                    return make_plan(
                        'coordinated_engage', members, front, members,
                        [EffectClaim('position', ling, None, context.H, HEURISTIC, 'covered_flank'),
                         EffectClaim('damage', front, estimate.damage_est, context.H,
                                     estimate.evidence, 'window_shot', lower=estimate.damage_lo,
                                     upper=estimate.damage_hi)],
                        ResourceClaims(controls=tuple(members), fire=fire,
                                       space=(space_claim(context, ling, move),)),
                        Commitment('guarding', 0, max_loops=context.H), {},
                        context.snapshot.snapshot_id, fingerprint, atomic=True,
                        goal_kind='enemy', goal_data={'validation_kind': 'zerg_screen',
                            'role': 'harass', 'pressured': aid,
                            'supporter': shooter, 'objective': {'kind': 'contact_point', 'position': point}},
                        rationale='flank the threatening shooter under friendly fire')
    return None


def zergling_guard_plans(context, memory, fingerprint):
    """One covered flank and one local screen, both optional for the final search."""
    import math
    from .skills.common import space_claim
    pressured = pressured_hydras(context)
    units, enemies = context.snapshot.allies, context.snapshot.enemies
    lings = [lid for lid, unit in sorted(units.items())
             if unit.alive and unit.type_name == 'zergling' and not _ready_melee(context, lid)]
    if not pressured or not lings:
        return []
    flank = _covered_flank(context, memory, fingerprint, pressured, lings)
    plans = [flank] if flank else []
    if flank:
        lings = [a for a in lings if a not in flank.current_actions]
    aid, threats = pressured[0]
    anchor, threat = units[aid].position, enemies[threats[0]].position
    angle = math.atan2(threat[1] - anchor[1], threat[0] - anchor[0])
    for ling in sorted(lings, key=lambda lid: (distance(units[lid].position, anchor), lid)):
        if distance(units[ling].position, anchor) > SUPPORT_RADIUS:
            continue
        radius = units[aid].radius + units[ling].radius + .1
        point = (anchor[0] + radius * math.cos(angle), anchor[1] + radius * math.sin(angle))
        if any(unit.alive and (unit.type_name == 'baneling' or
               unit.type_name == 'zergling' and peer != ling)
               and distance(unit.position, point) < radius for peer, unit in units.items()):
            break  # the threatened side already has a body; avoid duplicating its slot
        move = _guard_step(context, memory, ling, point)
        if move is not None:
            plans.append(make_plan(
                'coordinated_engage', [ling], aid, {ling: move},
                [EffectClaim('position', ling, None, context.H, HEURISTIC, 'rear_guard')],
                ResourceClaims(controls=(ling,), space=(space_claim(context, ling, move),)),
                Commitment('guarding', 0, max_loops=context.H), {},
                context.snapshot.snapshot_id, fingerprint, goal_kind='ally',
                goal_data={'validation_kind': 'zerg_screen',
                           'objective': {'kind': 'waypoint', 'position': point},
                           'role': 'rear_guard', 'pressured': aid},
                rationale='screen the threatened side of a nearby hydralisk'))
            break
    return plans
