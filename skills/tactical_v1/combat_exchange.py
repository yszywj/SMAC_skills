"""Short visible exposure windows shared by movement and receiving cover.

This is a bounded risk proxy, not a simulator of enemy targeting/cooldowns.
An observed loss trend may extend a one-volley estimate over a short wait;
ordinary friendly fire never removes an enemy's exposure. No hidden state.
"""
import math

from .context import GAME_LOOPS_PER_SPEED_SECOND


class PressureMemory:
    """At most two windows of observed loss from the same visible exposure.

    This remembers evidence that an exchange was dangerous, not an enemy's
    weapon cooldown or attribution of damage to an individual attacker.
    """
    def __init__(self, config=None):
        self.enabled = (config or {}).get("common_combat", {}).get(
            "pressure_memory_enabled", True)
        self.records = {}
        self.loop = None

    def __eq__(self, other):
        return isinstance(other, PressureMemory) and self.__dict__ == other.__dict__

    @staticmethod
    def _exposure(context, actor):
        from .battle_intent import enemy_identity
        unit = context.snapshot.allies[actor]
        risk = context.forecast_risk(actor, unit.position)
        current = set(risk.current_threats)
        def amount(enemy):
            damage = context.enemy_attack_damage(enemy, actor)
            # Same explicitly uncertain proxy as exposure_window. A missing
            # catalog entry cannot erase an actually observed loss.
            return 14. if damage is None else max(damage, 0.)
        return {e: (enemy_identity(context, e),
                    (1. if e in current else .5) * amount(e))
                for e in risk.reachable_threats}

    def observe(self, context):
        loop, H = context.snapshot.game_loop, max(context.H, 1)
        if self.loop is not None and loop <= self.loop:
            context._observed_pressure = self.evidence(context)
            return
        self.loop = loop
        for actor in set(self.records) | set(context.snapshot.allies):
            unit = context.snapshot.allies.get(actor)
            cap = context.capabilities.get(actor)
            if not self.enabled or unit is None or not unit.alive or cap is None \
                    or cap.healer or cap.suicide_attack:
                self.records.pop(actor, None)
                continue
            record = self.records.get(actor)
            current = self._exposure(context, actor)
            if record:
                shared = sum(min(weight, current[e][1]) for e, (tag, weight)
                             in record['threats'].items()
                             if e in current and current[e][0] == tag)
                original = sum(weight for _, weight in record['threats'].values())
                if unit.tag != record['tag'] or loop >= record['until_loop'] \
                        or shared < .85 * original or original <= 0.:
                    self.records.pop(actor, None)
            loss = max(unit.health_lost + unit.shield_lost, 0.)
            if loss > 0. and sum(weight for _, weight in current.values()) > 0.:
                self.records[actor] = {'tag': unit.tag, 'loop': loop,
                    'until_loop': loop + 2 * H, 'loss': loss, 'threats': current}
        context._observed_pressure = self.evidence(context)

    def evidence(self, context):
        """Pure, decaying view; expiry is never renewed by a danger query."""
        loop = context.snapshot.game_loop
        return {actor: record['loss'] * max(0., min(1.,
                    (record['until_loop'] - loop) /
                    max(record['until_loop'] - record['loop'], 1)))
                for actor, record in self.records.items()
                if actor in context.snapshot.allies
                and context.snapshot.allies[actor].tag == record['tag']
                and loop < record['until_loop']}


def observed_pressure(context, actor, *, enabled=True):
    unit = context.snapshot.allies[actor]
    fresh = max(unit.health_lost + unit.shield_lost, 0.)
    return max(fresh, getattr(context, '_observed_pressure', {}).get(actor, 0.)) \
        if enabled else fresh


def attack_requires_approach(context, actor, action):
    """A legal ATTACK can spend its next window chasing instead of firing.

    Unknown native reach is not evidence of movement. Current in-range
    attacks remain ordinary firing commands, independent of their provider.
    """
    semantic = context.decode(actor, action)
    return bool(semantic.kind == 'attack'
        and semantic.target in context.snapshot.enemies
        and context.can_issue_attack(actor, semantic.target)
        and context.contact_status(actor, semantic.target) == 'out_of_range')


def attack_approach_position(context, actor, action):
    """Bounded straight-line endpoint, not a collision/path guarantee."""
    unit = context.snapshot.allies[actor]
    if not attack_requires_approach(context, actor, action):
        return context.predict_position(actor, action).position
    target = context.decode(actor, action).target
    reach, speed = context.effective_attack_reach(actor, target), context.speed(actor)
    if reach is None or reach < 0. or speed is None:
        return unit.position
    enemy = context.snapshot.enemies[target]
    length = math.dist(unit.position, enemy.position)
    travel = min(max(0., length - reach), speed * context.H / GAME_LOOPS_PER_SPEED_SECOND)
    ratio = travel / length if length > 0. else 0.
    return tuple(x + ratio * (y - x) for x, y in zip(unit.position, enemy.position))


def attack_approach_relief(context, actor, action, position=None,
                           pressure_memory_enabled=True):
    """A dangerous current state does not itself make a chase an escape.

    Use the same exposure comparison as movement decisions. A pursuit may
    keep the emergency exemption when its prospective path ends in genuine
    relief, including clearing an effect, without claiming survival.
    """
    if not attack_requires_approach(context, actor, action):
        return False
    point = attack_approach_position(context, actor, action) if position is None else position
    facts = movement_exchange(context, actor, point, moving=True,
                              pressure_memory_enabled=pressure_memory_enabled)
    return bool(not facts['after']['effect_danger'] and (
        facts['relief'] or facts['before']['effect_danger']))


def next_fire_timing(context, actor, action, *, position=None, eligible_targets=None,
                     allow_outside=False, use_native_period=True):
    """Pure next-shot timing under the SAME usable target-health budget.

    The baseline is a currently legal shot/implicit attack approach or the
    best executable one-window approach. Necessary cardinal/blocked-route
    movement is not compared with an impossible straight-line command.
    An implicit ATTACK approach commits to its selected target; other nearby
    enemies cannot stand in for that command's next shot. Unlike MOVE, it
    may fire on arrival without spending a whole window before shooting.
    Known native cycles can increase the existing delay-in-H floor, never
    reduce it. The fixed 2H count is an opportunity, not hits or damage.
    """
    empty = {'before_eta_by_target': {}, 'after_eta_by_target': {},
             'before_eta_h': float('inf'), 'after_eta_h': float('inf'),
             'delay_h': 0., 'period_known': False, 'period_by_target': {},
             'lost_cycle_opportunities': 0., 'cost': 0.}
    unit = context.snapshot.allies[actor]
    cap = context.capabilities[actor]
    speed = context.speed(actor)
    semantic = context.decode(actor, action)
    approaching = attack_requires_approach(context, actor, action)
    if not unit.alive or cap.healer or cap.suicide_attack or speed is None or speed <= 0. \
            or not (semantic.kind == 'move' or approaching):
        return empty
    point = tuple(context.predict_position(actor, action).position if position is None else position)
    H = max(context.H, 1)
    scale, cooldown = speed * H / GAME_LOOPS_PER_SPEED_SECOND, unit.weapon_cooldown_loops / H
    # Final branches often differ only in their target-health budget. Cache
    # physical timing independently, then filter both sides identically.
    cache = getattr(context, '_next_fire_geometry_cache', None)
    if cache is None:
        cache = context._next_fire_geometry_cache = {}
    key = actor, action, point
    if key not in cache:
        before, after, periods, contact = {}, {}, {}, set()
        destinations = [context.predict_position(actor, move).position
                        for move in context.move_destinations(actor)]
        for enemy in context.snapshot.enemies:
            reach = context.effective_attack_reach(actor, enemy)
            if reach is None or reach < 0.:
                continue
            gap = lambda at: context.projected_attack_opportunity(actor, enemy, at).center_gap
            if context.can_issue_attack(actor, enemy) \
                    and context.contact_status(actor, enemy) == 'in_range':
                contact.add(enemy)
            if context.can_issue_attack(actor, enemy):
                # A legal ATTACK may already auto-approach native range.
                baseline = max(cooldown, gap(unit.position) / scale)
            else:
                baseline = min((max(cooldown, 1. + gap(z) / scale)
                                for z in destinations), default=float('inf'))
            if not math.isfinite(baseline):
                continue
            before[enemy] = baseline
            if approaching:
                # Evaluate the selected command from the observed origin.
                # An approximate endpoint must not erase time already spent
                # reaching it, nor invent a switch to a different target.
                if enemy == semantic.target:
                    after[enemy] = max(cooldown, gap(unit.position) / scale)
            else:
                resulting = max(cooldown, 1. + gap(point) / scale)
                if math.isfinite(resulting):
                    after[enemy] = resulting
            periods[enemy] = context.weapon_period(actor, enemy)
        cache[key] = before, after, periods, contact
    all_before, all_after, all_periods, contact = cache[key]
    targets = set(all_before) if eligible_targets is None else set(eligible_targets) & set(all_before)
    if not targets:
        return empty
    before = {e: all_before[e] for e in sorted(targets)}
    after = {e: all_after[e] for e in sorted(targets) if e in all_after}
    periods = {e: all_periods[e] for e in sorted(targets)}
    # If the selected pursuit target is already reserved by earlier peer
    # fire, the command has no guaranteed primary shot. Remaining alternative
    # targets still define the forgone opportunity; this must not become a
    # zero-cost chase merely because the after map is empty.
    eta_before, eta_after = min(before.values()), min(after.values(), default=float('inf'))
    delay = max(0., eta_after - eta_before)
    known = all(p is not None and math.isfinite(p) and p > 0. for p in periods.values())
    lost = 0.
    if use_native_period and known and delay > 1e-9:
        def opportunities(etas):
            return max((0 if eta > 2. + 1e-9 else
                        1 + math.floor(max(0., (2. - eta) * H) / periods[e] + 1e-9)
                        for e, eta in etas.items()), default=0)
        lost = max(0, opportunities(before) - opportunities(after))
    enabled = bool(contact & targets) or allow_outside or eligible_targets is not None
    return {'before_eta_by_target': before, 'after_eta_by_target': after,
            'before_eta_h': eta_before, 'after_eta_h': eta_after, 'delay_h': delay,
            'period_known': known, 'period_by_target': periods,
            'lost_cycle_opportunities': lost,
            'cost': min(1., max(delay, lost)) if enabled and delay > 1e-9 else 0.}


def next_fire_delay(context, actor, action, *, position=None, eligible_targets=None,
                    allow_outside=False, use_native_period=True):
    """Bounded timing cost; survival/blockage exemptions belong to the caller."""
    return next_fire_timing(context, actor, action, position=position,
        eligible_targets=eligible_targets, allow_outside=allow_outside,
        use_native_period=use_native_period)['cost']


def reserve_fraction(unit):
    return unit.effective_health / max(unit.max_health + unit.max_shield, 1.)


def next_fire_horizon(context, actor, position=None):
    unit = context.snapshot.allies[actor]
    point = unit.position if position is None else position
    cache = getattr(context, "_next_fire_horizon_cache", None)
    if cache is None:
        cache = context._next_fire_horizon_cache = {}
    key = actor, tuple(point)
    if key in cache:
        return cache[key]
    scale = (context.speed(actor) or 0.) * context.H / GAME_LOOPS_PER_SPEED_SECOND
    gap = min((context.projected_attack_opportunity(actor, e, point).center_gap
               for e in context.snapshot.enemies), default=float("inf"))
    travel = gap / scale if scale > 0 else (0. if gap <= .05 else float("inf"))
    eta = max(travel, unit.weapon_cooldown_loops / max(context.H, 1))
    result = min(3., max(1., eta + 1.))
    cache[key] = result
    return result


def exposure_window(context, actor, position=None, *, horizon_h=1., certain=True,
                    forecast_config=None, pressure_memory_enabled=True):
    """Known volley or observed short loss trend, whichever is greater.

    Unknown damage uses an explicitly uncertain proxy and limits viability.
    Waiting does not invent repeated enemy shots. The loss trend is scaled
    down when the final exposure is lower, and vanishes in safe positions.
    """
    unit = context.snapshot.allies[actor]
    point = unit.position if position is None else tuple(position)
    params = forecast_config or {}
    discount = float(params.get("forecast_reach_discount", .5))
    floor = float(params.get("uncertain_position_floor", .35))
    horizon_h = min(3., max(1., horizon_h if math.isfinite(horizon_h) else 3.))
    cache = getattr(context, "_exchange_window_cache", None)
    if cache is None:
        cache = context._exchange_window_cache = {}
    recent = observed_pressure(context, actor, enabled=pressure_memory_enabled)
    key = actor, tuple(point), horizon_h, certain, discount, floor, recent
    if key in cache:
        return cache[key]
    current = context.forecast_risk(actor, unit.position)
    future = context.forecast_risk(actor, point)

    def weights(risk):
        now = set(risk.current_threats)
        return {e: 1. if e in now else discount for e in risk.reachable_threats}

    origin, final = weights(current), weights(future)
    if not certain:
        for enemy, weight in origin.items():
            final[enemy] = max(final.get(enemy, 0.), floor * weight)
    unknown = False

    def amount(exposures):
        nonlocal unknown
        result = 0.
        for enemy, weight in exposures.items():
            damage = context.enemy_attack_damage(enemy, actor)
            unknown |= damage is None
            result += weight * (max(damage, 0.) if damage is not None else 14.)
        return result

    before, volley = amount(origin), amount(final)
    trend = recent * horizon_h * min(1., volley / max(before, 1.)) if final else 0.
    exposure = max(volley, trend)
    ratio = exposure / max(unit.effective_health, 1.)
    viability = max(0., 1. - ratio)
    if unknown:
        viability = min(viability, .5)
    # Effect damage is already included in volley/exposure. Proximity alone
    # must not erase the output value of a healthy unit surviving that blast.
    result = {"amount": exposure, "volley": volley, "observed_trend": trend,
              "reserve_ratio": ratio, "viability": viability,
              "unknown": unknown, "horizon_h": horizon_h,
              "effect_danger": bool(future.effect_weapon_near)}
    cache[key] = result
    return result


def danger_evidence(context, actor, position=None, *, horizon_h=1.,
                    pressure_memory_enabled=True, critical_fraction=.35):
    """Separate supported urgent danger from an all-enemy exposure bound.

    A possible volley from every reachable enemy is useful as an upper
    bound, but cannot prove that every one targets this actor. Evidence is
    recent observed loss, an explicitly observed in-range attack order, or
    a local effect danger. Missing enemy orders/cooldowns invent no facts.
    The short loss projection is still a warning, never damage attribution.
    """
    unit = context.snapshot.allies[actor]
    point = tuple(unit.position if position is None else position)
    horizon_h = min(2., max(1., horizon_h if math.isfinite(horizon_h) else 2.))
    cache = getattr(context, '_danger_evidence', None)
    if cache is None:
        cache = context._danger_evidence = {}
    key = actor, point, horizon_h, pressure_memory_enabled, critical_fraction
    if key in cache:
        return cache[key]
    facts = exposure_window(context, actor, point, horizon_h=horizon_h,
                            pressure_memory_enabled=pressure_memory_enabled)
    origin = exposure_window(context, actor, unit.position, horizon_h=horizon_h,
                             pressure_memory_enabled=pressure_memory_enabled)
    pressure = observed_pressure(context, actor, enabled=pressure_memory_enabled)
    # With no visible source at the origin, a new position cannot prove that
    # already observed incoming damage has disappeared. Keep that evidence
    # uncertain until a later observation; only known exposure admits relief.
    unknown_loss_source = pressure > 0. and origin['volley'] <= 0.
    retained = min(1., facts['volley'] / origin['volley']) if origin['volley'] > 0. else 1.
    projected_loss = pressure * horizon_h * retained
    targeted, targeted_sources = 0., []
    raw_enemies = context.snapshot.raw_state.get('visible_enemies', {})
    for enemy in context.risk_at(actor, point).threats:
        raw = raw_enemies.get(enemy, {})
        orders = raw.get('orders') or ()
        order = orders[0] if orders and isinstance(orders[0], dict) else {}
        attack = order.get('kind') == 'attack' or order.get('ability_id') == 23
        target_tag = order.get('target_tag')
        owns_target = order.get('target_unit') == actor or (
            target_tag is not None and str(target_tag) == str(unit.tag))
        if not attack or not owns_target:
            continue
        cooldown = raw.get('weapon_cooldown_loops')
        if isinstance(cooldown, (int, float)) and cooldown > horizon_h * context.H:
            continue
        damage = context.enemy_attack_damage(enemy, actor)
        if damage is not None and damage > 0.:
            targeted += damage
            targeted_sources.append(enemy)
    fresh = max(0., unit.health_lost + unit.shield_lost)
    shock = fresh > 0. and fresh >= unit.effective_health and reserve_fraction(unit) <= critical_fraction
    # A recent collapse is retained at the origin only. A genuinely safer
    # destination is evaluated through retained loss and targeted fire.
    shock_here = shock and math.dist(point, unit.position) <= .05
    observed_lethal = pressure > 0. and projected_loss >= unit.effective_health
    targeted_lethal = targeted > 0. and targeted >= unit.effective_health
    supported = max(projected_loss, targeted, fresh if shock_here else 0.)
    # A reachable explosion is a risk, not automatically a lethal emergency.
    # Count its calibrated damage once; the observed loss projection may
    # already include it, so do not add that projection a second time.
    effect_sources = context.forecast_risk(actor, point).effect_weapon_near
    effect_damage = [context.enemy_attack_damage(e, actor) for e in effect_sources]
    effect_amount = sum(max(0., value) for value in effect_damage if value is not None)
    effect_lethal = bool(effect_sources and max(projected_loss, targeted + effect_amount)
                         >= unit.effective_health)
    supported = max(supported, targeted + effect_amount)
    result = {'immediate': bool(effect_lethal or shock_here or
                               observed_lethal or targeted_lethal),
              'effect_danger': facts['effect_danger'],
              'effect_lethal': effect_lethal, 'effect_amount': effect_amount,
              'observed_lethal': observed_lethal, 'targeted_lethal': targeted_lethal,
              'observed_collapse': shock_here, 'targeted_sources': tuple(targeted_sources),
              'supported_amount': supported, 'upper_amount': facts['amount'],
              'unknown': facts['unknown'] or unknown_loss_source, 'pressure': pressure}
    cache[key] = result
    return result


def imminent_exchange_danger(context, actor, *, pressure_memory_enabled=True):
    return danger_evidence(context, actor, horizon_h=next_fire_horizon(context, actor),
                           pressure_memory_enabled=pressure_memory_enabled)['immediate']


def escape_urgency(context, actor, *, critical_fraction=.35, min_threats=2,
                   pressure_memory_enabled=True):
    """Survival evidence for an escape, distinct from a task interruption.

    Potential lethal exposure remains available in ``exposure``. It does
    not itself grant a survival exemption. Full-health ordinary units with
    no observed loss/targeting are not forced out by hypothetical focus fire.
    """
    unit = context.snapshot.allies[actor]
    facts = exposure_window(context, actor, horizon_h=next_fire_horizon(context, actor),
                            pressure_memory_enabled=pressure_memory_enabled)
    risk = context.risk_at(actor, unit.position)
    vectors = [(context.snapshot.enemies[e].position[0] - unit.position[0],
                context.snapshot.enemies[e].position[1] - unit.position[1])
               for e in risk.threats if e in context.snapshot.enemies]
    crossfire = len(vectors) >= max(2, min_threats) and any(
        ax * bx + ay * by <= 0. and math.hypot(ax, ay) > .05 and math.hypot(bx, by) > .05
        for i, (ax, ay) in enumerate(vectors) for bx, by in vectors[i + 1:])
    evidence = danger_evidence(context, actor, horizon_h=next_fire_horizon(context, actor),
                               pressure_memory_enabled=pressure_memory_enabled,
                               critical_fraction=critical_fraction)
    pressure = evidence['pressure']
    immediate = evidence['immediate']
    remaining = max(0., unit.effective_health - evidence['supported_amount']) / max(
        unit.max_health + unit.max_shield, 1.)
    urgent = immediate or bool(risk.threats and (
        remaining <= critical_fraction and (pressure > 0. or evidence['targeted_sources'])
        or pressure > 0. and crossfire and facts['reserve_ratio'] >= critical_fraction))
    return {'urgent': urgent, 'immediate': immediate, 'crossfire': crossfire,
            'observed_collapse': evidence['observed_collapse'],
            'remaining_reserve_fraction': remaining, 'pressure': pressure,
            'exposure': facts, 'evidence': evidence}


def movement_exchange(context, actor, position=None, *, targets=None, moving=True,
                      pressure_memory_enabled=True, forecast_config=None):
    """One common, bounded window until a legal shot after a movement.

    Potential exposure is not attributed damage. An ordinary contested move
    remains feasible when its own EHP covers this window; neither a healthy
    teammate nor a large HP percentage supplies reserve to the moving unit.
    Escape callers may independently prefer a real reduction even when no
    available route is survivable. No enemy cooldown or future kill is used.
    """
    unit = context.snapshot.allies[actor]
    point = unit.position if position is None else tuple(position)
    enemies = tuple(context.snapshot.enemies if targets is None else targets)
    enemies = tuple(e for e in enemies if e in context.snapshot.enemies
                    and context.effective_attack_reach(actor, e) is not None
                    and context.effective_attack_reach(actor, e) >= 0.)
    def gap(at):
        return min((context.projected_attack_opportunity(actor, e, at).center_gap
                    for e in enemies), default=float('inf'))
    old_gap, new_gap = gap(unit.position), gap(point)
    scale = (context.speed(actor) or 0.) * context.H / GAME_LOOPS_PER_SPEED_SECOND
    travel = new_gap / scale if scale > 0 else (0. if new_gap <= .05 else float('inf'))
    delay = 1. if moving else 0.
    eta = max(delay + travel, unit.weapon_cooldown_loops / max(context.H, 1))
    if moving and scale <= 0.:
        eta = float('inf')  # an endpoint alone is not observed travel capability
    horizon = min(3., max(1., eta + 1.))
    before = exposure_window(context, actor, horizon_h=horizon,
                             pressure_memory_enabled=pressure_memory_enabled, forecast_config=forecast_config)
    after = exposure_window(context, actor, point, horizon_h=horizon,
                            certain=not moving, pressure_memory_enabled=pressure_memory_enabled,
                            forecast_config=forecast_config)
    relief = before['amount'] - after['amount'] >= max(1., .15 * before['amount'])
    return {'before': before, 'after': after, 'eta_h': eta, 'gap': new_gap,
            'closing': math.isfinite(old_gap) and new_gap < old_gap - .05,
            'ready_after_move': new_gap <= .05 and unit.weapon_cooldown_loops <= context.H,
            'relief': relief,
            'admissible': math.isfinite(eta) and not after['effect_danger']
                          and after['reserve_ratio'] < 1.}


def safe_reentry_window(context, actor, *, pressure_memory_enabled=True):
    facts = exposure_window(context, actor, horizon_h=next_fire_horizon(context, actor),
                            pressure_memory_enabled=pressure_memory_enabled)
    physical = context.risk_at(actor, context.snapshot.allies[actor].position)
    return not physical.threats and not physical.effect_weapon_near \
        and not facts['effect_danger'] and facts['reserve_ratio'] < .75


class ReentryWindowMemory:
    """A three-window guard against undoing an observed escape before firing.

    Only actual movement with actual exposure reduction starts this memory.
    Queries cannot renew it. Continued escape retains its original deadline;
    real firing, identity changes, disappearing threats and expiry clear it.
    """
    def __init__(self, config=None):
        self.enabled = (config or {}).get('common_combat', {}).get('reentry_window_enabled', True)
        self.records, self.pending = {}, {}
        self.loop = None

    def __eq__(self, other):
        return isinstance(other, ReentryWindowMemory) and self.__dict__ == other.__dict__

    def observe(self, context):
        from .battle_intent import enemy_identity
        from .context import distance
        loop, H = context.snapshot.game_loop, max(context.H, 1)
        if self.loop is not None and loop <= self.loop:
            return
        self.loop = loop
        for actor in set(self.records) | set(self.pending):
            unit = context.snapshot.allies.get(actor)
            record, pending = self.records.get(actor), self.pending.get(actor)
            if unit is None or not unit.alive or not self.enabled:
                self.records.pop(actor, None)
                continue
            if record and (unit.tag != record['tag'] or loop >= record['until_loop']
                    or not any(e in context.snapshot.enemies and enemy_identity(context, e) == tag
                               for e, tag in record['threats'].items())):
                self.records.pop(actor, None)
                record = None
            if not pending or unit.tag != pending['tag'] or loop <= pending['loop']:
                continue
            fired = unit.weapon_cooldown_loops > max(0., pending['cooldown'] -
                                                      (loop - pending['loop'])) + .5
            if fired:
                self.records.pop(actor, None)
                continue
            if pending['kind'] != 'move' or distance(unit.position, pending['position']) <= .1:
                continue
            amount = exposure_window(context, actor)['amount']
            if pending['amount'] - amount < max(1., .15 * pending['amount']):
                continue
            if record is None and pending['threats']:
                self.records[actor] = {'tag': unit.tag, 'origin': pending['position'],
                    'threats': pending['threats'], 'until_loop': pending['loop'] + 3 * H}
        self.pending.clear()

    def commit(self, context, actions):
        from .battle_intent import enemy_identity
        if not self.enabled:
            return
        for actor, action in actions.items():
            unit = context.snapshot.allies.get(actor)
            cap = context.capabilities.get(actor)
            if unit is None or not unit.alive or cap is None or cap.healer or cap.suicide_attack:
                continue
            risk = context.forecast_risk(actor, unit.position)
            self.pending[actor] = {'tag': unit.tag, 'loop': context.snapshot.game_loop,
                'position': unit.position, 'cooldown': unit.weapon_cooldown_loops,
                'kind': context.decode(actor, action).kind,
                'amount': exposure_window(context, actor)['amount'],
                'threats': {e: enemy_identity(context, e) for e in risk.reachable_threats}}

    def allowed(self, context, actor, position, targets=None):
        from .battle_intent import enemy_identity
        from .context import distance
        record = self.records.get(actor)
        unit = context.snapshot.allies.get(actor)
        if not self.enabled or record is None or unit is None or unit.tag != record['tag'] \
                or context.snapshot.game_loop >= record['until_loop']:
            return True
        if distance(position, record['origin']) >= distance(unit.position, record['origin']) - .1:
            return True
        risk = context.forecast_risk(actor, position)
        if not any(e in record['threats'] and enemy_identity(context, e) == record['threats'][e]
                   for e in risk.reachable_threats):
            return True
        facts = movement_exchange(context, actor, position, targets=targets)
        if facts['relief'] or facts['ready_after_move'] and facts['admissible']:
            return True
        # Hysteresis reserves a quarter of the unit's own EHP while it still
        # cannot fire. This is not an added risk charge or a permanent lock.
        return facts['after']['reserve_ratio'] < .75 and not facts['after']['effect_danger']
