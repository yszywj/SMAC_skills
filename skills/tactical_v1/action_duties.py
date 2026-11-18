"""Execution priorities for complete final commands, including baseline.

These are ordinal requirements, not extra damage rewards. A requirement is
activated only when a legal, observed-mechanism-compatible alternative exists.
Queries cache snapshot facts; they never create or renew persistent tasks.
"""
from .combat_exchange import danger_evidence, next_fire_timing
from .context import distance
from .joint_combat import action_position


PRIORITIES = ('rescue', 'intercept', 'burst', 'fire', 'entry')


def zergling_blast_exposure(context, actor, position=None):
    """Known one-window explosion exposure, separate from ordinary gunfire.

    Zerglings keep contact through ordinary injury. Only a calibrated blast
    can justify their emergency departure; unknown effects do not establish
    a safe escape. The original route checks still price ordinary fire.
    """
    unit = context.snapshot.allies[actor]
    point = unit.position if position is None else position
    sources = tuple(context.forecast_risk(actor, point, horizon=context.H).effect_weapon_near)
    if not sources:
        return 0., sources
    if not context.burst_profile().geometry_known:
        return None, sources
    amounts = [context.enemy_attack_damage(enemy, actor) for enemy in sources]
    if any(amount is None for amount in amounts):
        return None, sources
    return sum(max(0., amount) for amount in amounts), sources


def rescue_actions(context, actor, memory=None):
    """Legal ordinary-unit moves with evidence-backed immediate relief.

    Low HP or all-enemy possible fire alone is not a retreat instruction.
    A dying actor without a survivable route retains its attacking alternatives.
    """
    cache = getattr(context, '_rescue_actions', None)
    if cache is None:
        cache = context._rescue_actions = {}
    key = actor, id(memory)
    if key in cache:
        return cache[key]
    unit, cap = context.snapshot.allies.get(actor), context.capabilities.get(actor)
    if unit is None or not unit.alive or cap is None or cap.healer or cap.suicide_attack:
        cache[key] = ()
        return ()
    zergling = cap.key == 'zergling'
    blast_before, _ = zergling_blast_exposure(context, actor) if zergling else (0., ())
    if zergling and (blast_before is None or blast_before <= 0.):
        cache[key] = ()
        return ()
    before = danger_evidence(context, actor, horizon_h=2.)
    if not zergling and not before['immediate']:
        cache[key] = ()
        return ()
    origin_effects = set(context.forecast_risk(actor, unit.position).effect_weapon_near)
    options = []
    for action in context.move_destinations(actor):
        if memory is not None and memory.movement_blockage(context, actor, action) >= .5:
            continue
        point, _ = action_position(context, actor, action)
        if distance(unit.position, point) <= .05:
            continue
        midpoint = tuple((x + y) / 2. for x, y in zip(unit.position, point))
        effects = set(context.forecast_risk(actor, point).effect_weapon_near)
        mid_effects = set(context.forecast_risk(actor, midpoint).effect_weapon_near)
        if (effects | mid_effects) - origin_effects:
            continue
        if zergling:
            blast_after, _ = zergling_blast_exposure(context, actor, point)
            blast_middle, _ = zergling_blast_exposure(context, actor, midpoint)
            if blast_after is None or blast_middle is None \
                    or blast_after >= blast_before or blast_middle > blast_before:
                continue  # ordinary pressure relief alone cannot pull a ling out
        after = danger_evidence(context, actor, point, horizon_h=2.)
        middle = danger_evidence(context, actor, midpoint, horizon_h=1.)
        # A relative improvement is not a rescue. Both the traversed route
        # and the destination must leave health for the next observation.
        # This is a necessary model check, not a native survival guarantee.
        if max(middle['supported_amount'], after['supported_amount']) >= unit.effective_health:
            continue
        if middle['unknown'] or after['unknown']:
            continue
        if middle['supported_amount'] > before['supported_amount'] + max(1., .15 * before['supported_amount']):
            continue
        reduced = before['supported_amount'] - after['supported_amount'] >= max(
            1., .15 * before['supported_amount'])
        effect_relief = bool(origin_effects and len(effects) < len(origin_effects))
        if not reduced and not effect_relief:
            continue
        # Losing one explosion source is not relief if the destination adds
        # a new, independently supported lethal ordinary volley.
        if effect_relief and after['supported_amount'] > before['supported_amount'] + 1.:
            continue
        options.append((len(effects), after['supported_amount'],
                        after['upper_amount'], action))
    cache[key] = tuple(row[-1] for row in sorted(options))
    return cache[key]


def sustained_fire_actions(context, actor, memory=None):
    """Keep real contact across cooldowns, with only zero-delay movement.

    Native ATTACK waits through cooldown without cancelling the order. A
    MOVE must finish before readiness, preserve contact, and improve safety
    or bring the shooter closer to an exposed ally. No perpetual shot lock
    or new task deadline is created; targets are checked each observation.
    """
    unit = context.snapshot.allies[actor]
    targets = tuple(e for e in context.attackable_enemies(actor)
                    if context.contact_status(actor, e) == 'in_range')
    if not targets:
        return ()
    attacks = tuple(context.attack_action(actor, e) for e in targets)
    if unit.weapon_cooldown_loops <= context.H:
        return attacks
    before = danger_evidence(context, actor, horizon_h=1.)
    peers = [u for a, u in context.snapshot.allies.items() if a != actor and u.alive
             and u.health_lost + u.shield_lost > 0.
             and distance(unit.position, u.position) <= 10.]
    moves = []
    for action in context.move_destinations(actor):
        if memory is not None and memory.movement_blockage(context, actor, action) >= .5:
            continue
        point, _ = action_position(context, actor, action)
        middle = tuple((x + y) / 2. for x, y in zip(unit.position, point))
        retained = [e for e in targets if all(context.projected_attack_opportunity(
            actor, e, p).available for p in (middle, point))]
        if not retained:
            continue
        timing = next_fire_timing(context, actor, action, position=point,
                                  eligible_targets=retained)
        if timing['delay_h'] > 1e-6 or timing['lost_cycle_opportunities'] > 0:
            continue
        # A cooldown move must not widen ORDINARY enemy weapon coverage
        # unless a teammate can actually deliver window fire on the new
        # threat (the lone-advancing-shooter failure). The same standard is
        # applied to an out-of-range ATTACK's implied approach point by the
        # guard's point-based interface.
        from .deployment import unsupported_cooldown_advance
        if unsupported_cooldown_advance(context, actor, point)['blocked']:
            continue
        after = danger_evidence(context, actor, point, horizon_h=1.)
        midway = danger_evidence(context, actor, middle, horizon_h=1.)
        if max(after['supported_amount'], midway['supported_amount']) > before['supported_amount'] + .1:
            continue
        effects = set(context.forecast_risk(actor, unit.position).effect_weapon_near)
        if any(set(context.forecast_risk(actor, p).effect_weapon_near) - effects for p in (middle, point)):
            continue
        relief = after['supported_amount'] < before['supported_amount'] - .1
        closes_support = any(distance(point, u.position) + .25 < distance(unit.position, u.position)
                             for u in peers)
        if relief or closes_support:
            moves.append(action)
    return attacks + tuple(moves)


def actor_duty(context, actor, memory):
    """One active duty per actor, with explicit precedence and fallback."""
    cache = getattr(context, '_execution_actor_duties', None)
    if cache is None:
        cache = context._execution_actor_duties = {}
    key = actor, id(memory)
    if key in cache:
        return cache[key]
    unit, cap = context.snapshot.allies.get(actor), context.capabilities.get(actor)
    result = {'required': False, 'actions': (), 'reason': 'no_execution_duty', 'kind': None}
    if unit is not None and unit.alive and cap is not None and not cap.healer:
        if cap.suicide_attack:
            progress = getattr(memory, 'burst_progress', None)
            if progress is not None:
                result = dict(progress.execution(context, actor))
                result['kind'] = 'burst'
        else:
            rescue = rescue_actions(context, actor, memory)
            if rescue:
                threats = set(context.forecast_risk(actor, unit.position).reachable_threats)
                finishes = []
                for enemy in context.attackable_enemies(actor):
                    if enemy not in threats or context.contact_status(actor, enemy) != 'in_range':
                        continue
                    estimate = context.fire_estimate(actor, enemy)
                    if estimate.window_shots >= 1 and estimate.damage_lo is not None \
                            and estimate.damage_lo >= context.snapshot.enemies[enemy].effective_health:
                        finishes.append(context.attack_action(actor, enemy))
                # Let the complete exchange compare a known finishing shot
                # with withdrawal. This does not remove predicted enemy fire
                # or enemy death explosions from either branch's risk.
                result = {'required': True, 'actions': rescue + tuple(finishes), 'kind': 'rescue',
                          'reason': 'survivable_rescue_or_threat_finish' if finishes
                                    else 'lethal_pressure_with_survivable_route'}
            else:
                # Safe interception of an approaching enemy bomber precedes
                # ordinary sustained fire: a window-qualified kill before the
                # bomber first covers a victim prevents a multi-unit loss
                # that one more volley of ordinary output cannot buy back.
                # The duty set is rebuilt from this snapshot's facts only, so
                # it releases itself when the window closes. Spread moves
                # here are exactly the bounded override of the ready-weapon
                # fire constraint: only threatened members move, and each
                # single explosion is not lethal on its own.
                from .intercept import actor_intercept_duty
                intercept = actor_intercept_duty(context, actor, memory)
                if intercept["required"]:
                    result = intercept
                else:
                    fire = sustained_fire_actions(context, actor, memory)
                    if fire:
                        result = {'required': True, 'actions': fire, 'kind': 'fire',
                                  'reason': 'current_executable_shot' if unit.weapon_cooldown_loops <= context.H
                                            else 'preserve_contact_through_cooldown'}
                    else:
                        from .deployment import deployment_duty
                        deployment = deployment_duty(context, actor, memory)
                        progress = getattr(memory, 'contact_progress', None)
                        if deployment is not None:
                            result = deployment
                        elif progress is not None:
                            result = dict(progress.execution(context, actor, memory))
                            result['kind'] = 'entry'
    result['actions'] = tuple(a for a in result.get('actions', ()) if a in context.legal_actions(actor))
    result['required'] = bool(result.get('required') and result['actions'])
    cache[key] = result
    return result


def assess(context, actions, memory):
    """Smaller violation counts win BEFORE the existing numeric score.

    No branch is rejected for an unavoidable duty: legal least-violation
    fallback remains available even when all combined routes conflict.
    """
    cache = getattr(context, '_execution_assessments', None)
    if cache is None:
        cache = context._execution_assessments = {}
    key = tuple(sorted(actions.items())), id(memory)
    if key in cache:
        return cache[key]
    counts = dict.fromkeys(PRIORITIES, 0)
    rows = {}
    joint = None
    burst_allowed = memory.burst_progress.joint_allowed(context, actions)
    for actor, action in actions.items():
        duty = actor_duty(context, actor, memory)
        satisfied = not duty['required'] or action in duty['actions']
        branch_reason = None
        if actor in burst_allowed and burst_allowed[actor]:
            valid = burst_allowed[actor]
            if action not in valid:
                satisfied = False
                branch_reason = 'joint_burst_route_or_residual_invalid'
            elif not any(a in valid for a in duty['actions']):
                satisfied = True  # original forced target lost its marginal value
        if satisfied and duty['required'] and duty['kind'] == 'intercept':
            from .intercept import final_valid
            satisfied = final_valid(context, actions, actor, memory)
            if not satisfied:
                branch_reason = 'interception_or_spread_invalid_after_combination'
        if satisfied and duty['required'] and duty['kind'] == 'rescue' \
                and context.decode(actor, action).kind == 'move':
            from .joint_combat import assess_joint
            if joint is None:
                joint = assess_joint(context, actions)
            exchange = joint['team_exchange']['actors'].get(actor)
            # Recheck the whole final route after teammate actions change
            # exposure. A rescue that the joint model still predicts lethal
            # cannot defeat standing fire merely by carrying a rescue label.
            if exchange and exchange['after']['amount'] >= context.snapshot.allies[actor].effective_health:
                satisfied = False
                branch_reason = 'joint_route_still_lethal_compare_exchange'
        if not satisfied:
            counts[duty['kind'] or 'burst'] += 1
        rows[actor] = {**duty, 'selected_action': action, 'satisfied': satisfied,
                       'branch_reason': branch_reason}
    result = {'priority': tuple(counts[k] for k in PRIORITIES),
              'violations': counts, 'actors': rows,
              'semantics': 'survivable rescue, worthwhile burst, continuous fire, coordinated entry; then utility'}
    cache[key] = result
    return result


def seed_actions(context, actions, memory):
    """Construct an actual complete alternative, not just penalize P0.

    Preserve satisfied commands and heal orders. Within each allowed set,
    prefer nonblocked, nonoverlapping routes before its local preference.
    The resulting full branch still passes normal validation and scoring.
    """
    chosen = dict(actions)
    duties = {a: actor_duty(context, a, memory) for a in actions}
    from .unit_value_view import preservation_value
    ordered = sorted(actions, key=lambda a: (
        PRIORITIES.index(duties[a]['kind']) if duties[a]['required'] else len(PRIORITIES),
        -preservation_value(context, a), a))
    for actor in ordered:
        duty = duties[actor]
        if not duty['required'] or chosen[actor] in duty['actions']:
            continue
        unit = context.snapshot.allies[actor]
        positions = {a: action_position(context, a, command)[0]
                     for a, command in chosen.items() if context.snapshot.allies[a].alive}
        def route_rank(action):
            point, _ = action_position(context, actor, action)
            moving = distance(unit.position, point) > .05
            overlaps = sum(distance(point, other) < unit.radius + context.snapshot.allies[a].radius
                for a, other in positions.items() if a != actor
                and context.snapshot.allies[a].is_flying == unit.is_flying) if moving else 0
            trial = {**chosen, actor: action}
            blockage = memory.movement_blockage(context, actor, action,
                joint_actions=trial, joint_positions={**positions, actor: point}) if memory is not None else 0.
            return blockage >= .5, overlaps, duty['actions'].index(action)
        chosen[actor] = min(duty['actions'], key=route_rank)
    # Repair residual conflicts using the very same branch-local check.
    for actor in sorted(chosen):
        allowed = memory.burst_progress.joint_allowed(context, chosen).get(actor, ())
        if allowed and chosen[actor] not in allowed:
            from .burst import outcome, exchange_value
            def burst_rank(action):
                result = outcome(context, actor, action)
                value = exchange_value(context, actor, result.position,
                                       opportunity=result.opportunity)['net_hp'] if result.opportunity else 0.
                semantic = context.decode(actor, action)
                return (-value, semantic.kind == 'stop', action)
            chosen[actor] = min(allowed, key=burst_rank)
    return chosen
