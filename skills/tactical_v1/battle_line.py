"""Shared local contact and firing lanes, derived from visible capabilities.

The lanes are candidate destinations, not walls that absorb enemy fire.
No race or unit name chooses a tank. Final commands, target budgets and
arrival windows decide whether a proposed line can provide useful fire.
"""
import math

from .context import GAME_LOOPS_PER_SPEED_SECOND, distance
from .combat_exchange import exposure_window, escape_urgency, danger_evidence
from .joint_combat import action_position
from .local_entry import _fronts, PEER_RADIUS


def _ordinary(context, actor):
    unit, cap = context.snapshot.allies.get(actor), context.capabilities.get(actor)
    return bool(unit and unit.alive and cap and not cap.healer and not cap.suicide_attack)


def ready_targets(context, actor, targets=None):
    cache = getattr(context, '_battle_line_ready', None)
    if cache is None:
        cache = context._battle_line_ready = {}
    if actor not in cache:
        cache[actor] = tuple(e for e in context.attackable_enemies(actor)
                            if context.contact_status(actor, e) == 'in_range'
                            and context.fire_estimate(actor, e).window_shots >= 1)
    if targets is None:
        return cache[actor]
    allowed = set(targets)
    return tuple(e for e in cache[actor] if e in allowed)


def _eta(context, actor, enemy, point):
    reach = context.effective_attack_reach(actor, enemy)
    if reach is None or reach < 0.:
        return float('inf')
    stride = (context.speed(actor) or 0.) * max(context.H, 1) / GAME_LOOPS_PER_SPEED_SECOND
    gap = max(0., distance(point, context.snapshot.enemies[enemy].position) - reach)
    return max(context.snapshot.allies[actor].weapon_cooldown_loops / max(context.H, 1),
               gap / stride if stride > 0. else 0. if gap <= .05 else float('inf'))


def _volley(context, actor, target):
    """Known compatible volley, cached independently of proposed movement."""
    cache = getattr(context, '_battle_line_volleys', None)
    if cache is None:
        cache = context._battle_line_volleys = {}
    key = actor, target
    if key not in cache:
        damage = context._damage(actor, target)
        cache[key] = max(0., damage[0]) if damage is not None else 0.
    return cache[key]


def _observed_target(context, actor):
    orders = context.snapshot.raw_state.get('allies', {}).get(actor, {}).get('orders', ())
    return orders[0].get('target_unit') if orders and orders[0].get('kind') == 'attack' else None


def _front_fire(context, actors, front, etas):
    """At most one timely volley per source, bounded by visible target HP.

    This ranks a common local objective. It does not predict kills or remove
    hostile exposure, and unknown damage supplies no invented firepower.
    """
    remaining = {e: context.snapshot.enemies[e].effective_health for e in front}
    value = 0.
    for actor in sorted(actors, key=lambda a: (etas[a].get(front, float('inf')), a)):
        choices = [(min(_volley(context, actor, e), remaining[e]) /
                    (1. + _eta(context, actor, e, context.snapshot.allies[actor].position)), e)
                   for e in front if _eta(context, actor, e,
                       context.snapshot.allies[actor].position) <= 3.]
        if not choices:
            continue
        useful, target = max(choices)
        value += useful
        remaining[target] = max(0., remaining[target] - _volley(context, actor, target))
    return value


def battle_line(context):
    """Snapshot-local assignments; target death/visibility rebuilds the line.

    A front is spatially bounded rather than transitively merging an enemy
    chain. Existing fire and nearby exposed allies take precedence over a
    distant attractive target. The actual legal/native reach sets each lane.
    """
    cached = getattr(context, '_battle_line_facts', None)
    if cached is not None:
        return cached
    enemies, units = context.snapshot.enemies, context.snapshot.allies
    fronts = _fronts(context)
    members = tuple(a for a in sorted(units) if _ordinary(context, a))
    ready = {a: ready_targets(context, a) for a in members}
    contact = {a: tuple(e for e in context.attackable_enemies(a)
                        if context.contact_status(a, e) == 'in_range') for a in members}
    threats = {a: set(context.forecast_risk(a, units[a].position).reachable_threats)
               for a in members}
    etas = {a: {front: min((_eta(context, a, e, units[a].position) for e in front),
                          default=float('inf')) for front in fronts} for a in members}
    assigned = {}
    # Established contact anchors the local group even while cooling. An
    # idle rear unit follows that fire, rather than choosing its own nearer
    # front and calling another equally idle mover "support".
    for actor in members:
        current = [front for front in fronts if set(front) & set(contact[actor])]
        if current:
            observed = _observed_target(context, actor)
            assigned[actor] = min(current, key=lambda f: (
                observed not in f, -_front_fire(context, (actor,), f, etas), etas[actor][f], f))
    pending = set(members) - set(assigned)
    while pending:
        # Anchor-bounded friendly pockets cannot grow into a transitive chain.
        anchor = min(pending)
        pocket = tuple(sorted(a for a in pending if distance(
            units[a].position, units[anchor].position) <= PEER_RADIUS / 2.))
        nearby = tuple(a for a in members if distance(
            units[a].position, units[anchor].position) <= PEER_RADIUS)
        covers = tuple(a for a in nearby if contact[a])
        options = []
        for front in fronts:
            feasible = tuple(a for a in pocket if math.isfinite(etas[a][front]))
            if not feasible:
                continue
            shooters = tuple(a for a in covers if assigned.get(a) == front)
            covered_fire = _front_fire(context, shooters, front, etas)
            joint_fire = _front_fire(context, (*shooters, *feasible), front, etas)
            eta = sum(min(6., etas[a][front]) for a in feasible) / len(feasible)
            threatened = any(set(front) & threats[a] for a in nearby)
            # Cover counts only if this pocket can reach the same fight soon;
            # distant ready fire must not pull a flank across the enemy army.
            supported = covered_fire > 0. and any(etas[a][front] <= 3. for a in feasible)
            options.append((not supported, not threatened, -joint_fire,
                            -len(feasible), eta, front))
        if options:
            front = min(options)[-1]
            for actor in pocket:
                if math.isfinite(etas[actor][front]):
                    assigned[actor] = front
                else:
                    compatible = [f for f in fronts if math.isfinite(etas[actor][f])]
                    if compatible:
                        assigned[actor] = min(compatible, key=lambda f: (etas[actor][f], f))
        pending.difference_update(pocket)
    result = {'actors': {}, 'fronts': []}
    for front in fronts:
        actors = tuple(a for a in members if assigned.get(a) == front)
        if not actors:
            continue
        center = tuple(sum(enemies[e].position[i] for e in front) / len(front) for i in (0, 1))
        friendly = tuple(sum(units[a].position[i] for a in actors) / len(actors) for i in (0, 1))
        dx, dy = friendly[0] - center[0], friendly[1] - center[1]
        norm = math.hypot(dx, dy)
        # Encircled or coincident groups have no reliable global "rear".
        axis = (dx / norm, dy / norm) if norm > .25 else None
        result['fronts'].append({'targets': front, 'members': actors, 'axis': axis})
        for actor in actors:
            result['actors'][actor] = _assignment(context, actor, front, actors, axis, ready[actor])
    context._battle_line_facts = result
    return result


def _assignment(context, actor, targets, members, axis, ready):
    units, enemies = context.snapshot.allies, context.snapshot.enemies
    unit = units[actor]
    targets = tuple(e for e in targets if e in enemies
                    and context.effective_attack_reach(actor, e) is not None
                    and context.effective_attack_reach(actor, e) >= 0.)
    if not targets:
        return None
    peers = tuple(a for a in members if a != actor and distance(
        unit.position, units[a].position) <= PEER_RADIUS)
    receiving = set().union(*(set(context.forecast_risk(a, units[a].position).current_threats)
                              for a in peers)) if peers else set()
    nearest_eta = min(_eta(context, actor, e, unit.position) for e in targets)
    observed = _observed_target(context, actor)
    target = min(targets, key=lambda e: (
        0 if e in ready else 1,
        # Intercept a nearby source threatening this group's shooters. A
        # distant threat cannot send a short-range unit past reachable foes.
        0 if e in receiving and _eta(context, actor, e, unit.position) <= nearest_eta + 1. else 1,
        0 if e == observed and _eta(context, actor, e, unit.position) <= nearest_eta + .5 else 1,
        _eta(context, actor, e, unit.position), e))
    enemy = enemies[target]
    reach = context.effective_attack_reach(actor, target)
    role = 'contact' if context.capabilities[actor].combat_class == 'melee' else 'fire'
    # Both melee and ranged stand just inside their own effective weapon band.
    # Longer reach never requires a unit to advance to another unit's shorter band.
    radius = min(reach, max(unit.radius + enemy.radius, reach - min(.15, .1 * reach)))
    vx, vy = unit.position[0] - enemy.position[0], unit.position[1] - enemy.position[1]
    angle = math.atan2(vy, vx) if math.hypot(vx, vy) > .01 else (
        math.atan2(axis[1], axis[0]) if axis else 0.)
    if axis is not None:
        facing = math.atan2(axis[1], axis[0])
        delta = math.atan2(math.sin(angle - facing), math.cos(angle - facing))
        # Do not send a flanker through the enemy to enforce a global rear.
        if abs(delta) <= math.pi / 2:
            angle = facing + max(-1.2, min(1.2, delta))
    # Small local lane offsets separate bodies pursuing the same contact band.
    same_band = sorted((a for a in (actor, *peers)
                        if context.capabilities[a].combat_class == context.capabilities[actor].combat_class),
                       key=lambda a: (math.atan2(units[a].position[1] - enemy.position[1],
                                                units[a].position[0] - enemy.position[0]), a))
    if len(same_band) > 1:
        rank = same_band.index(actor) - (len(same_band) - 1) / 2.
        angle += max(-.45, min(.45, rank * (2. * unit.radius + .05) / max(radius, .1)))
    goal = (enemy.position[0] + math.cos(angle) * radius,
            enemy.position[1] + math.sin(angle) * radius)
    if role == 'contact':
        # Other wounded melee may favor the outer edge of their contact
        # band. Zerglings keep their attacking position through ordinary
        # injury; their explicit blast-escape rule owns emergency movement.
        # Every melee still uses the same timely ranged-support geometry.
        from .unit_value_view import preservation_value
        protection = preservation_value(context, actor)
        # A bounded positioning preference, never a new retreat trigger.
        if context.capabilities[actor].key != 'zergling' \
                and unit.effective_health < (.5 + .1 * protection) * max(unit.max_health + unit.max_shield, 1.):
            radius = min(reach, radius + .35)
            goal = (enemy.position[0] + math.cos(angle) * radius,
                    enemy.position[1] + math.sin(angle) * radius)
        ranged_peers = [a for a in members if a != actor
                        and context.capabilities[a].combat_class == 'ranged'
                        and distance(units[a].position, enemy.position) <= 14.]
        if ranged_peers:
            anchor = min(ranged_peers, key=lambda a: distance(units[a].position, goal))
            anchor_unit = units[anchor]
            peer_reach = 0.
            relation = context.snapshot.relations.get(anchor, {}).get(target)
            if relation is not None and relation.native_weapon_range is not None:
                peer_reach = relation.native_weapon_range
            # Stay inside one short sprint of the nearest rear shooter's own
            # firing band: this bounds short local flanks while forbidding
            # lone deep charges and long back-around routes (the shared axis
            # clamp above already blocks crossing through the enemy).
            support_limit = peer_reach + anchor_unit.radius + (
                (context.speed(anchor) or 0.) * context.H / 16.)
            gap = distance(goal, anchor_unit.position)
            if gap > support_limit:
                fraction = support_limit / gap
                goal = (anchor_unit.position[0] + (goal[0] - anchor_unit.position[0]) * fraction,
                        anchor_unit.position[1] + (goal[1] - anchor_unit.position[1]) * fraction)
    return {'front': tuple(targets), 'target': target, 'role': role, 'goal': goal,
            'reach': reach, 'eta_h': _eta(context, actor, target, unit.position),
            'ready_targets': tuple(e for e in ready if e in targets), 'peers': peers}


def actor_line(context, actor, targets=None):
    """Optionally restrict an existing G2/rejoin duty to its local enemies."""
    base = battle_line(context)['actors'].get(actor)
    if base is None or targets is None:
        return base
    selected = tuple(sorted(e for e in targets if e in context.snapshot.enemies))
    if not selected:
        return None
    key = actor, selected
    cache = getattr(context, '_battle_line_restricted', None)
    if cache is None:
        cache = context._battle_line_restricted = {}
    if key not in cache:
        cache[key] = _assignment(context, actor, selected, (actor, *base['peers']),
                                 None, ready_targets(context, actor, selected))
    return cache[key]


def line_action_rank(context, actor, action, targets=None, memory=None):
    """Lower ranks offer productive lanes; None does not forbid other skills.

    Current attacks are retained. A movement lane needs real progress or
    urgent pressure relief and a nonblocked route. An uncertain all-enemy
    focus-fire bound alone cannot remove every contested entry candidate.
    """
    row = actor_line(context, actor, targets)
    if row is None or action not in context.legal_actions(actor):
        return None
    semantic = context.decode(actor, action)
    if semantic.kind not in ('move', 'attack'):
        return None
    if semantic.kind == 'attack' and semantic.target not in row['front']:
        return None
    unit = context.snapshot.allies[actor]
    point, certain = action_position(context, actor, action)
    moving = semantic.kind == 'move' or distance(point, unit.position) > .01
    urgent = escape_urgency(context, actor)['immediate']
    ready = ready_targets(context, actor)
    if moving and ready and not urgent:
        return None
    if semantic.kind == 'move' and memory is not None and (
            memory.movement_blockage(context, actor, action) >= .5
            or memory.movement_conflict(context, actor, action) > 0.):
        return None
    risk = context.forecast_risk(actor, point)
    if risk.effect_weapon_near and not context.forecast_risk(actor, unit.position).effect_weapon_near:
        return None
    target = semantic.target if semantic.kind == 'attack' else row['target']
    before_eta = _eta(context, actor, target, unit.position)
    after_eta = _eta(context, actor, target, point)
    before = exposure_window(context, actor, unit.position, horizon_h=max(1., before_eta))
    after = exposure_window(context, actor, point, horizon_h=max(1., after_eta), certain=certain)
    relief = before['amount'] - after['amount'] >= max(.1, .15 * before['amount'])
    evidence = danger_evidence(context, actor, point, horizon_h=max(1., after_eta))
    if moving and evidence['immediate'] and not (urgent and relief):
        return None
    lane_progress = distance(unit.position, row['goal']) - distance(point, row['goal'])
    if moving and after_eta >= before_eta - .05 and lane_progress <= .05 and not (urgent and relief):
        return None
    # A contact unit may not gain a free first charge from a distant healthy
    # ally. Prefer arrivals other nearby sources can plausibly accompany.
    peer_eta = min((actor_line(context, peer)['eta_h'] for peer in row['peers']
                    if actor_line(context, peer) is not None
                    and set(actor_line(context, peer)['front']) & set(row['front'])), default=3.)
    unsupported = max(0., min(3., peer_eta) - max(1., after_eta)) if risk.reachable_threats else 0.
    # This is one candidate, not a hard override: G1's attack still competes
    # in the complete plan. An urgently threatened receiver must be allowed
    # to withdraw while this small group supplies replacement fire.
    priority = (-1 if moving and urgent and relief else
                0 if semantic.kind == 'attack' and semantic.target in ready else 1)
    return (priority,
            round(after_eta + unsupported, 4),
            round(evidence['supported_amount'] / max(unit.effective_health, 1.), 4),
            round(after['reserve_ratio'], 4),
            round(distance(point, row['goal']), 4),
            0 if target == row['target'] else 1, action)


def assess_battle_line(context, actions, joint_facts, memory=None):
    """Positive per-actor joint participation gains, without another risk cost.

    Consumers merge this with their existing participation value, never sum
    two rewards for the same approach. Ready actors gain no geometry credit;
    future target budgets and receiver viability are taken from final facts.
    """
    cache = getattr(context, '_battle_line_assessments', None)
    if cache is None:
        cache = context._battle_line_assessments = {}
    # Memory is observed once before proposals and committed after selection.
    # Joint facts identity distinguishes optional forecast configurations.
    key = tuple(sorted(actions.items())), id(joint_facts), id(memory)
    if key in cache:
        return cache[key]
    facts = battle_line(context)
    rows = joint_facts['actors']
    gains, details = {}, {}
    for actor, lane in facts['actors'].items():
        if lane is None or actor not in rows:
            continue
        row = rows[actor]
        if not row.get('ordinary') or not row.get('moving') or ready_targets(context, actor):
            continue
        front = set(lane['front'])
        usable = tuple(e for e in row['target_etas'] if e in front)
        after = min((row['target_etas'][e] for e in usable), default=float('inf'))
        unit = context.snapshot.allies[actor]
        stride = (context.speed(actor) or 0.) * max(context.H, 1) / GAME_LOOPS_PER_SPEED_SECOND
        if stride <= 0.:
            continue
        # Compare executing this command with spending this window without
        # advancing. Both ETAs are measured from the CURRENT observation.
        # Comparing after's 1H MOVE overhead with a zero-overhead baseline
        # would erase every ordinary one-window approach's progress.
        before = min((max(unit.weapon_cooldown_loops / max(context.H, 1),
            1. + context.projected_attack_opportunity(actor, e, unit.position).center_gap / stride)
            for e in usable), default=float('inf'))
        if not math.isfinite(before) or not math.isfinite(after) or after >= before:
            continue
        if row['kind'] == 'move' and memory is not None and memory.movement_blockage(
                context, actor, actions[actor], joint_actions=actions,
                joint_positions={a: r['position'] for a, r in rows.items()}) >= .5:
            continue
        peers, fire = [], {}
        remaining = {e: context.snapshot.enemies[e].effective_health for e in front}
        for peer in sorted(lane['peers']):
            other = rows.get(peer)
            if not other or other.get('cover_after_viability', 0.) <= 0.:
                continue
            if other['kind'] == 'move' and memory is not None and memory.movement_blockage(
                    context, peer, actions[peer], joint_actions=actions,
                    joint_positions={a: r['position'] for a, r in rows.items()}) >= .5:
                continue
            opportunities = []
            for enemy, eta in other['target_etas'].items():
                if enemy not in front or eta > min(3., after + 1.):
                    continue
                # A current ranged volley may be followed by its native next
                # cycle when a melee partner arrives. A MOVE cannot promise
                # recurring fire or cover merely by being geometrically near.
                if eta < after - 1.:
                    period = context.weapon_period(peer, enemy)
                    if other['kind'] != 'attack' or period is None or period <= 0.:
                        continue
                    period_h = period / max(context.H, 1)
                    eta += math.ceil((after - 1. - eta) / period_h) * period_h
                if eta > min(3., after + 1.):
                    continue
                useful = min(_volley(context, peer, enemy), remaining[enemy])
                useful *= other.get('target_credits', {}).get(enemy, 0.)
                useful *= other['cover_after_viability'] / (1. + abs(after - eta))
                if useful > 0.:
                    opportunities.append((useful, enemy))
            if not opportunities:
                continue
            amount, enemy = max(opportunities)
            remaining[enemy] = max(0., remaining[enemy] - amount)
            peers.append(peer)
            fire[peer] = amount
        if not fire:
            continue
        viability = max(0., min(1., row.get('cover_after_viability', 0.)))
        # Two fragile melee movers are not a full covering force merely
        # because they have matching ETAs. Scale by bounded useful fire in
        # this same arrival window, without subtracting it from enemy risk.
        reference = max(1., row.get('team_exposure_after', row['exposure'])['amount'],
                        max((_volley(context, actor, e) for e in usable), default=0.))
        support_fraction = min(1., sum(fire.values()) / reference)
        gain = (min(1., before - after) * viability * support_fraction
                / (1. + .25 * min(before, after)))
        if memory is not None:
            gain *= getattr(memory, 'advance_credit_scale', lambda *_: 1.)(
                context, actor, actions[actor])
        gains[actor] = gain
        details[actor] = {'front': lane['front'], 'role': lane['role'],
                         'before_eta_h': before, 'after_eta_h': after,
                         'arrival_eta_h': after,
                         'timely_peers': tuple(peers), 'timely_fire': fire,
                         'support_fraction': support_fraction, 'gain': gain}
    result = {'actor_gains': gains, 'actors': details, 'fronts': facts['fronts']}
    cache[key] = result
    return result
