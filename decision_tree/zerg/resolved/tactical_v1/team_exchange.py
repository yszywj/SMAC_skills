"""Finite hostile volley allocations and observed team-delay costs.

The allocation is a planning estimate, not an enemy targeting probability or
a promise of safety. Each known single-target enemy owns one volley per
scene; it cannot spend that whole volley on every possible victim. AOE and
unknown mechanisms retain their per-victim upper exposure. Observed damage
is a separate floor. Friendly attacks never erase hostile damage here.
"""
import math

from .combat_exchange import observed_pressure
from .context import GAME_LOOPS_PER_SPEED_SECOND, distance


HORIZON_H = 2.


def _observed_target(context, enemy):
    raw = context.snapshot.raw_state.get('visible_enemies', {}).get(enemy, {})
    orders = raw.get('orders') or ()
    if not orders or orders[0].get('kind') != 'attack':
        return None
    order = orders[0]
    target = order.get('target_unit')
    if target in context.snapshot.allies:
        return target
    tag = order.get('target_tag')
    return next((a for a, u in context.snapshot.allies.items()
                 if tag is not None and str(u.tag) == str(tag)), None)


def _exposures(context, actor, point, discount):
    cache = getattr(context, '_team_exposure_inputs', None)
    if cache is None:
        cache = context._team_exposure_inputs = {}
    key = actor, tuple(point), discount
    if key not in cache:
        forecast = context.forecast_risk(actor, point)
        current = set(forecast.current_threats)
        sources = {}
        for enemy in forecast.reachable_threats:
            damage = context.enemy_attack_damage(enemy, actor)
            known = damage is not None and math.isfinite(damage)
            sources[enemy] = {
                'damage': max(0., damage) if known else 14.,
                'known': known and enemy not in forecast.heuristic_enemies,
                'reach_weight': 1. if enemy in current else discount}
        cache[key] = (sources, bool(forecast.effect_weapon_near))
    return cache[key]


def _scene(context, positions, *, pressure_memory_enabled, discount):
    """One simultaneous scene: each ordinary source's shares sum to one."""
    key = (tuple(sorted((a, tuple(p)) for a, p in positions.items())),
           pressure_memory_enabled, discount)
    cache = getattr(context, '_team_exchange_scenes', None)
    if cache is None:
        cache = context._team_exchange_scenes = {}
    if key in cache:
        return cache[key]
    units, enemies = context.snapshot.allies, context.snapshot.enemies
    inputs, by_enemy = {}, {}
    for actor, point in positions.items():
        sources, effect = _exposures(context, actor, point, discount)
        inputs[actor] = {'sources': sources, 'effect': effect}
        for enemy, source in sources.items():
            by_enemy.setdefault(enemy, {})[actor] = source
    allocated = {a: 0. for a in positions}
    shares = {a: {} for a in positions}
    source_details = {}
    for enemy, victims in by_enemy.items():
        capability = context.enemy_capabilities[enemy]
        # A single-target budget is justified by a known compatible catalog,
        # not by the absence of damage data. An unknown effect is never safe.
        single = (not capability.splash_attack and not capability.suicide_attack
                  and capability.data_completeness != 'unknown'
                  and all(row['known'] for row in victims.values()))
        target = _observed_target(context, enemy)
        weights = {}
        for actor, source in victims.items():
            proximity = 1. / (1. + distance(positions[actor], enemies[enemy].position))
            # A visible attack order is stronger evidence than mere reach,
            # but may change before the next observation. Recent loss is not
            # attributed to this enemy; it only weakly biases the allocation.
            focus = 4. if actor == target else 1.
            loss = observed_pressure(context, actor, enabled=pressure_memory_enabled)
            receiving = 1. + min(1., loss / max(units[actor].effective_health, 1.))
            weights[actor] = max(.001, source['reach_weight'] * proximity * focus * receiving)
        total_weight = sum(weights.values())
        details = {}
        for actor, source in victims.items():
            fraction = weights[actor] / total_weight if single else 1.
            amount = source['damage'] * source['reach_weight'] * fraction
            allocated[actor] += amount
            shares[actor][enemy] = fraction
            details[actor] = {'share': fraction, 'amount': amount}
        source_details[enemy] = {
            'single_target_budget': single, 'observed_target': target,
            'victim_count': len(details),
            'share_sum': sum(row['share'] for row in details.values()),
            'allocated_amount': sum(row['amount'] for row in details.values()),
            'max_compatible_volley': max(s['damage'] for s in victims.values())}
    rows = {}
    for actor, point in positions.items():
        unit = units[actor]
        entry = inputs[actor]
        upper = sum(s['damage'] * s['reach_weight'] for s in entry['sources'].values())
        origin_sources, _ = _exposures(context, actor, unit.position, discount)
        origin_upper = sum(s['damage'] * s['reach_weight'] for s in origin_sources.values())
        recent = observed_pressure(context, actor, enabled=pressure_memory_enabled)
        # Leaving all known exposure can reduce the trend. When its source is
        # unobserved, movement cannot assert that the recent loss is now gone.
        retained = min(1., upper / origin_upper) if origin_upper > 0. else 1.
        observed_floor = recent * HORIZON_H * retained
        unknown = any(not s['known'] for s in entry['sources'].values()) or (
            recent > 0. and origin_upper == 0.)
        amount = max(allocated[actor], observed_floor)
        ratio = amount / max(unit.effective_health, 1.)
        viability = max(0., 1. - ratio)
        if unknown:
            viability = min(viability, .5)
        # AOE damage is allocated per victim above; being in an effect's
        # possible reach is not itself zero survival (e.g. one 35-damage
        # explosion against a healthy 90-health shooter).
        rows[actor] = {'amount': amount, 'reserve_ratio': ratio,
                       'viability': viability, 'unknown': unknown,
                       'effect_danger': entry['effect'], 'upper_amount': upper,
                       'allocated_volley': allocated[actor], 'observed_floor': observed_floor,
                       'horizon_h': HORIZON_H, 'source_shares': shares[actor]}
    result = {'actors': rows, 'sources': source_details}
    cache[key] = result
    return result


def assess_team_exchange(context, actions, *, pressure_memory_enabled=True, forecast_config=None):
    """Origin/midpoint/endpoint estimates for a complete final action set.

    Per-actor after exposure is the worst sampled route scene; these maxima
    need not be simultaneous. Source budgets are retained per scene so that
    route conservatism is not mistaken for actual combined enemy damage.
    """
    settings_input = forecast_config or {}
    discount = max(0., min(1., float(settings_input.get('forecast_reach_discount', .5))))
    uncertainty_floor = max(0., min(1., float(settings_input.get('uncertain_position_floor', .35))))
    key = tuple(sorted(actions.items())), pressure_memory_enabled, discount, uncertainty_floor
    cache = getattr(context, '_team_exchange_cache', None)
    if cache is None:
        cache = context._team_exchange_cache = {}
    if key in cache:
        return cache[key]
    # Importing the geometry helper is safe: never call assess_joint here.
    from .joint_combat import action_position
    origin = {a: u.position for a, u in context.snapshot.allies.items() if u.alive}
    predicted = {a: action_position(context, a, actions[a]) if a in actions else (point, True)
                 for a, point in origin.items()}
    final = {a: item[0] for a, item in predicted.items()}
    middle = {a: tuple((x + y) / 2 for x, y in zip(point, final[a]))
              for a, point in origin.items()}
    settings = {'pressure_memory_enabled': pressure_memory_enabled, 'discount': discount}
    before = _scene(context, origin, **settings)
    midway = _scene(context, middle, **settings)
    after = _scene(context, final, **settings)
    actors = {}
    for actor in origin:
        # Origin is the already-observed state; charging it forever would
        # make a genuine escape indistinguishable from standing still.
        route = [midway['actors'][actor], after['actors'][actor]]
        worst = dict(max(route, key=lambda row: row['amount']))
        worst['effect_danger'] = any(row['effect_danger'] for row in route)
        worst['unknown'] = any(row['unknown'] for row in route)
        worst['viability'] = min(row['viability'] for row in route)
        # A guessed endpoint is not proof of escape through terrain/bodies.
        retained = uncertainty_floor * before['actors'][actor]['amount'] if not predicted[actor][1] else 0.
        worst['uncertain_route_floor'] = retained
        worst['amount'] = max(worst['amount'], retained)
        worst['reserve_ratio'] = worst['amount'] / max(context.snapshot.allies[actor].effective_health, 1.)
        worst['viability'] = min(worst['viability'], max(0., 1. - worst['reserve_ratio']))
        actors[actor] = {'before': before['actors'][actor], 'after': worst,
                         'endpoint': after['actors'][actor]}
    result = {'actors': actors, 'source_budgets': {
        'before': before['sources'], 'midpoint': midway['sources'], 'after': after['sources']},
        'pressure_memory_enabled': pressure_memory_enabled,
        'semantics': 'finite single-target volley allocation; AOE/unknown retained; observed loss floor; no suppression'}
    cache[key] = result
    return result


def _earliest_entry(context, actor, enemy):
    """Best legal one-command approach, before any survival-score filtering."""
    cache = getattr(context, '_team_earliest_entry', None)
    if cache is None:
        cache = context._team_earliest_entry = {}
    key = actor, enemy
    if key in cache:
        return cache[key]
    unit, H = context.snapshot.allies[actor], max(context.H, 1)
    reach = context.effective_attack_reach(actor, enemy)
    stride = (context.speed(actor) or 0.) * H / GAME_LOOPS_PER_SPEED_SECOND
    if reach is None or reach < 0.:
        cache[key] = float('inf')
        return cache[key]
    gap = lambda point: max(0., distance(point, context.snapshot.enemies[enemy].position) - reach)
    if context.can_issue_attack(actor, enemy) and gap(unit.position) <= .05:
        cache[key] = unit.weapon_cooldown_loops / H
        return cache[key]
    if stride <= 0.:
        cache[key] = float('inf')
        return cache[key]
    options = []
    if context.can_issue_attack(actor, enemy):
        options.append(gap(unit.position) / stride)
    for action in context.move_destinations(actor):
        point = context.predict_position(actor, action).position
        if gap(point) < gap(unit.position) - .05:
            options.append(1. + gap(point) / stride)
    cache[key] = max(unit.weapon_cooldown_loops / H, min(options, default=float('inf')))
    return cache[key]


def delay_exchange_cost(context, actions, joint_facts, exchange):
    """Cost of avoidable lost participation while nearby teammates lose HP.

    Uses final-command target ETAs and the same residual target credits as
    joint_combat. It never says fire prevents incoming damage. Combine with
    other contact-delay costs per actor using max, not by adding duplicates.
    """
    units, rows = context.snapshot.allies, joint_facts['actors']
    pressure_enabled = exchange.get('pressure_memory_enabled', True)
    from .fire_opportunities import final_action_budget
    budget = final_action_budget(context, actions)
    costs, receivers = {}, {}
    for actor, row in rows.items():
        if not row.get('ordinary'):
            continue
        shared, pressure = set(), 0.
        local_receivers = []
        for peer, other in rows.items():
            if peer == actor or not other.get('ordinary') or distance(
                    units[actor].position, units[peer].position) > 10.:
                continue
            loss = observed_pressure(context, peer, enabled=pressure_enabled)
            if loss <= 0.:
                continue
            threats = set(context.forecast_risk(peer, other['position']).reachable_threats)
            compatible = {e for e in threats if context.effective_attack_reach(actor, e) is not None
                          and context.effective_attack_reach(actor, e) >= 0.}
            if not compatible:
                continue
            shared.update(compatible)
            pressure += min(1., loss / max(units[peer].effective_health, 1.))
            local_receivers.append(peer)
        if not shared:
            continue
        earliest_by_target = {e: _earliest_entry(context, actor, e) for e in shared}
        eligible = budget.future_targets(actor, earliest_by_target)['primary']
        earliest = min(eligible.values(), default=float('inf'))
        if not math.isfinite(earliest) or earliest > 3.:
            continue
        selected = min((eta for e, eta in row['target_etas'].items() if e in eligible),
                       default=float('inf'))
        delay = max(0., min(3., selected) - earliest) / 3.
        if delay <= 0.:
            continue
        before, after = exchange['actors'][actor]['before'], exchange['actors'][actor]['after']
        # Evidence-backed escape is useful participation, even if a shot is
        # deferred. Merely having a high all-enemies upper bound is not this.
        recent = observed_pressure(context, actor, enabled=pressure_enabled)
        emergency = before['effect_danger'] or (recent > 0. and before['reserve_ratio'] >= 1.)
        relief = before['amount'] - after['amount'] >= max(.1, .15 * before['amount'])
        if emergency and relief:
            continue
        costs[actor] = min(1., pressure) * delay
        receivers[actor] = tuple(local_receivers)
    return {'cost': sum(costs.values()), 'actors': costs, 'receivers': receivers}
