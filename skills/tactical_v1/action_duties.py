"""Execution priorities for complete final commands, including baseline.

These are ordinal requirements, not extra damage rewards. A requirement is
activated only when a legal, observed-mechanism-compatible alternative exists.
Queries cache snapshot facts; they never create or renew persistent tasks.
"""
from .combat_exchange import danger_evidence, next_fire_timing
from .context import distance
from .joint_combat import action_position


PRIORITIES = ('rescue', 'intercept', 'burst', 'fire', 'crossfire', 'support', 'entry')


def ready_hydralisk_shots(context, actor):
    """Legal, native-range shots available at this observation, before retreat."""
    unit = context.snapshot.allies.get(actor)
    if unit is None or not unit.alive or unit.type_name != 'hydralisk':
        return ()
    raw = context.snapshot.raw_state.get('allies', {}).get(actor, {})
    cooldown = raw.get('weapon_cooldown_loops')
    if cooldown is None:
        if raw.get('weapon_ready') is not True:
            return ()
    elif not unit.weapon_ready or unit.weapon_cooldown_loops > 0.:
        return ()
    cache = getattr(context, '_ready_hydralisk_shots', None)
    if cache is None:
        cache = context._ready_hydralisk_shots = {}
    if actor not in cache:
        cache[actor] = tuple(context.attack_action(actor, enemy)
                             for enemy in context.attackable_enemies(actor)
                             if context.contact_status(actor, enemy) == 'in_range')
    return cache[actor]


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
    before = danger_evidence(context, actor, horizon_h=2.)
    # A zergling qualifies for rescue through EITHER evidence channel:
    # observed imminent ordinary fire (same as every ordinary unit — being a
    # zergling alone vetoes nothing) or a calibrated explosion exposure.
    # Blast checks below are an ADDITIONAL condition when a blast threat
    # actually exists, never the only gate to rescue eligibility.
    melee_rescue_ok = getattr(context, 'common_config', {}).get(
        'melee_survivable_rescue_enabled', True)
    immediate = (before['immediate'] and (not zergling or melee_rescue_ok)) \
        or (zergling and (blast_before or 0.) > 0.)
    if not immediate:
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
        if zergling and (blast_before or 0.) > 0.:
            blast_after, _ = zergling_blast_exposure(context, actor, point)
            blast_middle, _ = zergling_blast_exposure(context, actor, midpoint)
            if blast_after is None or blast_middle is None \
                    or blast_after >= blast_before or blast_middle > blast_before:
                continue  # with a real blast threat, the route must also reduce it
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
    if ready_hydralisk_shots(context, actor):
        return attacks
    # Bounded "delay one shot" budget in LOOPS: zero-delay repositioning
    # keeps the legacy safety/support criteria; moves that POSTPONE the next
    # shot need a concrete damage reduction (supported-fire relief or
    # explosion-coverage loss) AND must fit the remaining budget. "Neater
    # formation" or "closer to a wounded teammate" is never enough for a
    # costly delay. At most ONE expected attack opportunity may be lost,
    # and the loss stays priced in the full-plan scoring.
    delay_budget = memory.fire_delay_available(context, actor) \
        if memory is not None else 0.
    if unit.weapon_cooldown_loops <= context.H and delay_budget <= 1e-9:
        return attacks
    before = danger_evidence(context, actor, horizon_h=1.)
    peers = [u for a, u in context.snapshot.allies.items() if a != actor and u.alive
             and u.health_lost + u.shield_lost > 0.
             and distance(unit.position, u.position) <= 10.]
    from .intercept import explosion_loss, spread_moves
    origin_blast_loss = explosion_loss(context, actor, unit.position)
    moves = []
    for action in context.move_destinations(actor):
        if memory is not None and memory.movement_blockage(context, actor, action) >= .5:
            continue
        point, _ = action_position(context, actor, action)
        middle = tuple((x + y) / 2. for x, y in zip(unit.position, point))
        retained = [e for e in targets if all(context.projected_attack_opportunity(
            actor, e, p).available for p in (middle, point))]
        if not retained:
            # A surplus shooter can leave contact for this window if doing
            # so actually reduces explosion loss and its shot delay is funded.
            if origin_blast_loss > 0. and memory is not None \
                    and action in spread_moves(context, actor, memory) \
                    and memory._fire_delay_postponement(context, actor, action) <= delay_budget + 1e-6:
                moves.append(action)
            continue
        timing = next_fire_timing(context, actor, action, position=point,
                                  eligible_targets=retained)
        # Unified postponement query (same target set, position and
        # next_fire_timing semantics as the commit-side charge).
        delay_loops = memory._fire_delay_postponement(context, actor, action) \
            if memory is not None else max(0., timing['delay_h']) * max(context.H, 1)
        if timing['lost_cycle_opportunities'] > 1:
            continue                    # one postponed opportunity at most
        if timing['lost_cycle_opportunities'] > 0 and delay_loops > delay_budget + 1e-6:
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
        blast_relief = origin_blast_loss > 0 and \
            explosion_loss(context, actor, point) < origin_blast_loss
        closes_support = any(distance(point, u.position) + .25 < distance(unit.position, u.position)
                             for u in peers)
        if delay_loops <= 1e-9:
            # Zero-delay repositioning: legacy safety/support criteria.
            if relief or blast_relief or closes_support:
                moves.append(action)
        else:
            # Costly delay: concrete damage reduction ONLY, within budget.
            if (relief or blast_relief) and delay_loops <= delay_budget + 1e-6:
                moves.append(action)
    return attacks + tuple(moves)


def actor_duty(context, actor, memory):
    """One active duty per actor, with explicit precedence and fallback."""
    owned = getattr(context, 'formation_orders', {})
    if actor in owned:
        cmd = owned[actor]
        return {'required': True, 'actions': (cmd,),
                'kind': 'fire' if context.decode(actor, cmd).kind == 'attack' else 'entry',
                'reason': 'exclusive_hydra_formation'}
    cache = getattr(context, '_execution_actor_duties', None)
    if cache is None:
        cache = context._execution_actor_duties = {}
    key = actor, id(memory)
    if key in cache:
        return cache[key]
    unit, cap = context.snapshot.allies.get(actor), context.capabilities.get(actor)
    result = {'required': False, 'actions': (), 'reason': 'no_execution_duty', 'kind': None}
    if unit is not None and unit.alive and cap is not None and not cap.healer:
        targets = context.forced_hydra_targets(actor)
        if targets:
            result = {'required': True, 'actions': tuple(context.attack_action(actor, e) for e in targets),
                      'kind': 'fire', 'reason': 'hydralisk_priority_over_zergling'}
        elif cap.suicide_attack:
            progress = getattr(memory, 'burst_progress', None)
            if progress is not None:
                result = dict(progress.execution(context, actor))
                result['kind'] = 'burst'
        else:
            rescue = rescue_actions(context, actor, memory)
            shots = ready_hydralisk_shots(context, actor) if rescue else ()
            if shots:
                result = {'required': True, 'actions': shots, 'kind': 'rescue',
                          'reason': 'ready_shot_before_withdrawal'}
            elif rescue:
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


def joint_rescue_substitution(context, actions, actor, memory):
    """A COMPLETE final joint kill of the threatening enemy can satisfy a
    rescue duty without withdrawal.

    Read-only, final-actions based: the actor's own final command must be a
    legal in-range ATTACK on an enemy currently threatening it, and the full
    group whose final actions attack that same enemy must have cumulative
    window-qualified fire reaching its remaining health with a stated
    timing basis. Members without an executable contribution do not count,
    and a missing timing/damage basis grants NO exemption (the plain joint
    attack candidate still competes through scoring). Pre-kill damage and
    the target's death effects stay priced in the evaluator.
    """
    if not getattr(context, 'common_config', {}).get('joint_rescue_enabled', True):
        return None
    unit = context.snapshot.allies.get(actor)
    action = actions.get(actor)
    if unit is None or not unit.alive or action is None \
            or action not in context.legal_actions(actor):
        return None
    threats = set(context.forecast_risk(actor, unit.position).reachable_threats) \
        | set(context.risk_at(actor, unit.position).threats)
    semantic = context.decode(actor, action)
    if semantic.kind != 'attack' or semantic.target not in threats \
            or semantic.target not in context.snapshot.enemies:
        return None
    target = semantic.target
    members = []
    for member, command in actions.items():
        other = context.snapshot.allies.get(member)
        if other is None or not other.alive:
            continue
        decoded = context.decode(member, command)
        if decoded.kind == 'attack' and decoded.target == target:
            members.append(member)
    if not members:
        return None
    from .impact_timing import joint_kill_time
    kill = joint_kill_time(context, members, target)
    # Strong exemption requires BOTH: sufficient executable fire AND a
    # conservative completion time (upper bound). Today the upper bound is
    # always unknown (no wind-up/projectile data), so the strong rescue
    # exemption is not granted; the plain joint attack candidate stays in
    # the comparison. Pre-kill damage, the target's death effects and the
    # actor's other threats all remain priced by the evaluator.
    if not kill['damage_sufficient'] or kill['kill_after_loops_upper'] is None:
        return {'substituted': False, 'target': target,
                'members': kill['contributors'],
                'timing_status': kill['timing_status'],
                'damage_sufficient': kill['damage_sufficient'],
                'reason': 'timing_unknown_no_strong_exemption'}
    return {'substituted': True, 'target': target,
            'members': kill['contributors'],
            'timing_status': kill['timing_status'],
            'kill_after_loops_upper': kill['kill_after_loops_upper']}


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
    from .skills.ranged_fire_formation import opening_crossfire, opening_entry_action
    crossfire = opening_crossfire(context, actions)
    counts['crossfire'] = len(crossfire)
    rows = {}
    joint = None
    burst_allowed = memory.burst_progress.joint_allowed(context, actions)
    for actor, action in actions.items():
        duty = actor_duty(context, actor, memory)
        satisfied = not duty['required'] or action in duty['actions']
        branch_reason = None
        if not satisfied and duty.get('kind') == 'entry' and actor not in crossfire:
            if opening_entry_action(context, actor, actions):
                satisfied = True
                branch_reason = 'shared_first_engagement'
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
        substitution = None
        if not satisfied and duty['required'] and duty['kind'] == 'rescue':
            # Complete-plan alternative: a final joint kill of the actor's
            # actual threatening enemy satisfies the rescue duty without
            # withdrawal — but ONLY with a conservative completion time
            # (see joint_rescue_substitution). The plain retreat options
            # stay available; both branches compete at the same duty level.
            substitution = joint_rescue_substitution(context, actions, actor, memory)
            if substitution is not None and substitution.get('substituted'):
                satisfied = True
                branch_reason = 'joint_kill_rescue_substitution'
        if not satisfied:
            counts[duty['kind'] or 'burst'] += 1
        if actor in crossfire:
            branch_reason = 'unsupported_entry_into_crossfire'
        rows[actor] = {**duty, 'selected_action': action, 'satisfied': satisfied,
                       'branch_reason': branch_reason,
                       **({'joint_rescue': substitution} if substitution else {})}
    from .hydra_support import isolated_shooter_support
    support = isolated_shooter_support(context, memory, actions)
    if support:
        counts['support'] = int(not support['satisfied'])
        rows[support['supporter']]['isolated_shooter_support'] = support
    result = {'priority': tuple(counts[k] for k in PRIORITIES),
              'violations': counts, 'actors': rows,
              'semantics': 'rescue, intercept, burst, continuous fire, crossfire, isolated shooter support, coordinated entry; then utility'}
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
                                       opportunity=result.opportunity,
                                       enemy_positions=result.enemy_positions)['net_hp'] if result.opportunity else 0.
                semantic = context.decode(actor, action)
                return (-value, semantic.kind == 'stop', action)
            chosen[actor] = min(allowed, key=burst_rank)
    from .hydra_support import isolated_shooter_support
    support = isolated_shooter_support(context, memory, chosen)
    if support:
        chosen[support['supporter']] = support['action']
    from .skills.ranged_fire_formation import repair_crossfire
    return repair_crossfire(context, chosen, duties)
