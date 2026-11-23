"""Action-level combat facts shared by every complete branch, including P0.

The facts describe opportunities, not enemy targeting or confirmed damage.
No skill name is used, and friendly attacks never erase hostile exposure.
"""
import math

from .context import GAME_LOOPS_PER_SPEED_SECOND, distance
from .combat_exchange import exposure_window, movement_exchange
from .fire_opportunities import final_action_budget


# Both sides compare the same bounded next-volley window. This is opportunity,
# not suppression or a promise that the recipient survives until cover arrives.
SUPPORT_HORIZON_H = 3.


def _cover_weight(etas, relevant, established, credits=None):
    timely = [(credits or {}).get(enemy, 1.) / (1. + eta) for enemy, eta in etas.items()
              if enemy in relevant and eta <= SUPPORT_HORIZON_H]
    if not timely:
        return 0.
    return (1. if established else .5) * max(timely)


def action_position(context, actor, action):
    """Include implicit approach in ATTACK, without claiming path certainty."""
    if context.capabilities[actor].suicide_attack:
        from .burst import action_position as burst_position
        return burst_position(context, actor, action), False
    prediction = context.predict_position(actor, action)
    semantic = context.decode(actor, action)
    unit = context.snapshot.allies[actor]
    if semantic.kind != 'attack' or semantic.target not in context.snapshot.enemies:
        return prediction.position, prediction.certain
    target = context.snapshot.enemies[semantic.target]
    reach = context.effective_attack_reach(actor, semantic.target)
    speed = context.speed(actor)
    if reach is None or reach < 0 or speed is None:
        return prediction.position, prediction.certain
    length = distance(unit.position, target.position)
    travel = min(max(0., length - reach), speed * context.H / GAME_LOOPS_PER_SPEED_SECOND)
    if travel <= 0 or length <= 0:
        return unit.position, True
    ratio = travel / length
    return tuple(x + ratio * (y - x) for x, y in zip(unit.position, target.position)), False


def assess_joint(context, actions, *, pressure_memory_enabled=True, forecast_config=None):
    """Pure cached facts from FINAL commands, never from proposed labels."""
    key = (tuple(sorted(actions.items())), pressure_memory_enabled,
           tuple(sorted((forecast_config or {}).items())))
    cache = getattr(context, '_joint_combat_cache', None)
    if cache is None:
        cache = context._joint_combat_cache = {}
    if key in cache:
        return cache[key]
    actor_cache = getattr(context, '_joint_actor_cache', None)
    if actor_cache is None:
        actor_cache = context._joint_actor_cache = {}
    actors, current = {}, {}
    H = max(context.H, 1)
    for actor, action in actions.items():
        unit = context.snapshot.allies.get(actor)
        if unit is None or not unit.alive:
            continue
        actor_key = (actor, action, key[1], key[2])
        base = actor_cache.get(actor_key)
        if base is not None:
            # Candidate branches recombine a small number of physical unit
            # actions. Only support/service facts depend on the whole team.
            actors[actor], current[actor] = dict(base[0]), base[1]
            continue
        cap = context.capabilities[actor]
        semantic = context.decode(actor, action)
        point, certain = action_position(context, actor, action)
        moving = distance(unit.position, point) > .01 or semantic.kind == 'move'
        ordinary = not cap.healer and not cap.suicide_attack
        facts = movement_exchange(context, actor, point, moving=moving,
                                  pressure_memory_enabled=pressure_memory_enabled,
                                  forecast_config=forecast_config)
        # A dangerous intermediate point cannot disappear behind a safe endpoint.
        route = [facts['after']]
        if moving:
            middle = tuple((x + y) / 2 for x, y in zip(unit.position, point))
            route.append(exposure_window(context, actor, middle,
                horizon_h=facts['after']['horizon_h'], certain=False,
                pressure_memory_enabled=pressure_memory_enabled, forecast_config=forecast_config))
        route_exposure = max(route, key=lambda x: x['amount'])
        targets, before = {}, {}
        orders = context.snapshot.raw_state.get('allies', {}).get(actor, {}).get('orders', ())
        observed_target = (orders[0].get('target_unit') if orders
                           and orders[0].get('kind') == 'attack' else None)
        scale = (context.speed(actor) or 0.) * H / GAME_LOOPS_PER_SPEED_SECOND
        if ordinary:
            for enemy in context.snapshot.enemies:
                reach = context.effective_attack_reach(actor, enemy)
                if reach is None or reach < 0:
                    continue
                old = context.projected_attack_opportunity(actor, enemy, unit.position)
                new = context.projected_attack_opportunity(actor, enemy, point)
                # A cooling weapon still supplies an upcoming volley if it
                # stays. Use the same horizon and ETA weight as the final plan.
                if (old.center_gap <= .05
                        and unit.weapon_cooldown_loops / H <= SUPPORT_HORIZON_H
                        and (observed_target is None or observed_target == enemy)):
                    before[enemy] = unit.weapon_cooldown_loops / H
                if semantic.kind == 'attack' and semantic.target != enemy:
                    continue
                if semantic.kind not in ('attack', 'move', 'stop'):
                    continue
                if semantic.kind == 'stop':
                    if not orders or orders[0].get('kind') != 'attack' or orders[0].get('target_unit') != enemy:
                        continue  # STOP cannot promise arbitrary new fire targets
                if semantic.kind == 'move' and new.center_gap > old.center_gap + .05:
                    continue       # withdrawing peers cannot promise cover
                travel = new.center_gap / scale if scale > 0 else (
                    0. if new.center_gap <= .05 else float('inf'))
                if semantic.kind == 'attack':
                    # Native ATTACK may fire as soon as it reaches the target.
                    # Count travel from the observed origin, not a mandatory
                    # full MOVE window plus travel from the predicted endpoint.
                    command_travel = old.center_gap / scale if scale > 0 else (
                        0. if old.center_gap <= .05 else float('inf'))
                    eta = max(command_travel, unit.weapon_cooldown_loops / H)
                else:
                    eta = max((1. if moving else 0.) + travel,
                              unit.weapon_cooldown_loops / H)
                if eta <= SUPPORT_HORIZON_H:
                    targets[enemy] = eta
        current[actor] = before
        cover_before = exposure_window(context, actor, unit.position,
            horizon_h=SUPPORT_HORIZON_H, pressure_memory_enabled=pressure_memory_enabled,
            forecast_config=forecast_config)
        cover_after = exposure_window(context, actor, point,
            horizon_h=SUPPORT_HORIZON_H, certain=certain,
            pressure_memory_enabled=pressure_memory_enabled, forecast_config=forecast_config)
        cover_viability = cover_after['viability']
        if moving:
            cover_viability = min(cover_viability, exposure_window(context, actor, middle,
                horizon_h=SUPPORT_HORIZON_H, certain=False,
                pressure_memory_enabled=pressure_memory_enabled,
                forecast_config=forecast_config)['viability'])
        actors[actor] = {
            'position': point, 'certain': certain, 'kind': semantic.kind,
            'ordinary': ordinary, 'moving': moving, 'target_etas': targets,
            'eta_h': min(targets.values(), default=float('inf')),
            'exposure': route_exposure, 'relief': facts['relief'],
            'initial_viability': facts['before']['viability'],
            'cover_before_viability': cover_before['viability'],
            'cover_after_viability': cover_viability,
            'observed_attack_target': observed_target,
            'support_window_h': SUPPORT_HORIZON_H,
            'closing': facts['closing'], 'support_before': 0., 'support_after': 0.,
        }
        actor_cache[actor_key] = (dict(actors[actor]), dict(before))
    # Allocate ordinary enemy firing opportunities once across the FINAL
    # team positions. Keep the old per-unit worst case as a diagnostic bound;
    # it must not simultaneously veto every member of a joint entry.
    from .team_exchange import assess_team_exchange
    exchange = assess_team_exchange(context, actions,
        pressure_memory_enabled=pressure_memory_enabled, forecast_config=forecast_config)
    for actor, row in actors.items():
        entry = exchange['actors'].get(actor)
        if entry is None or not row['ordinary']:
            continue
        row['team_exposure_before'] = entry['before']
        row['team_exposure_after'] = entry['after']
        row['initial_viability'] = entry['before']['viability']
        row['cover_before_viability'] = entry['before']['viability']
        row['cover_after_viability'] = entry['after']['viability']
    # Actor caches contain geometry only. A peer's final ATTACK can consume
    # a target budget, so recompute all credits AFTER the branch is combined.
    budget = final_action_budget(context, actions)
    before_targets, after_targets = {}, {}
    for actor, row in actors.items():
        before_targets[actor] = budget.future_targets(actor, current[actor])
        after_targets[actor] = budget.future_targets(actor, row['target_etas'])
        selected = after_targets[actor]
        row['raw_target_etas'] = row['target_etas']
        row['target_etas'] = selected['primary']
        row['contingency_target_etas'] = selected['contingency']
        row['target_credits'] = selected['credits']
        row['eta_h'] = selected['eta_h']
    for actor, row in actors.items():
        if not row['ordinary']:
            continue
        unit = context.snapshot.allies[actor]
        # Support has to concern a threatening/local compatible enemy, not
        # another fight elsewhere. Healers and one-use burst are separate facts.
        threats = set(context.forecast_risk(actor, row['position']).reachable_threats)
        original_threats = set(context.forecast_risk(actor, unit.position).reachable_threats)
        relevant = threats | original_threats | set(current[actor]) | set(row['raw_target_etas'])
        backup_before, backup_after = {}, {}
        for peer, other in actors.items():
            if peer == actor or not other['ordinary']:
                continue
            if distance(unit.position, context.snapshot.allies[peer].position) <= 10.:
                established = other['observed_attack_target'] in relevant
                selected = before_targets[peer]
                value = _cover_weight(selected['primary'], relevant, established, selected['credits'])
                row['support_before'] += value * other['cover_before_viability']
                if not value:
                    for target, eta in selected['contingency'].items():
                        if target in relevant and eta <= SUPPORT_HORIZON_H:
                            weight = ((1. if established else .5) * selected['credits'][target]
                                      * other['cover_before_viability'] / (1. + eta))
                            backup_before[target] = max(backup_before.get(target, 0.), weight)
            if distance(row['position'], other['position']) > 10.:
                continue
            selected = after_targets[peer]
            value = _cover_weight(selected['primary'], relevant, other['kind'] == 'attack', selected['credits'])
            row['support_after'] += value * other['cover_after_viability']
            if not value:
                for target, eta in selected['contingency'].items():
                    if target in relevant and eta <= SUPPORT_HORIZON_H:
                        weight = ((1. if other['kind'] == 'attack' else .5) * selected['credits'][target]
                                  * other['cover_after_viability'] / (1. + eta))
                        backup_after[target] = max(backup_after.get(target, 0.), weight)
        # Ten backup actors cannot manufacture ten full support links to an
        # already budgeted kill. One bounded contingency credit per target.
        row['support_before'] += sum(backup_before.values())
        row['support_after'] += sum(backup_after.values())
        row['support_loss'] = (max(0., row['support_before'] - row['support_after'])
                               / max(1., row['support_before'])) if threats else 0.
    from .healing import planned_healing_services
    result = {'actors': actors, 'healing_services': planned_healing_services(context, actions),
              'team_exchange': exchange}
    from .team_exchange import delay_exchange_cost
    result['team_delay'] = delay_exchange_cost(context, actions, result, exchange)
    cache[key] = result
    return result


def summary(facts):
    """JSON-safe final-branch explanation, kept out of the numeric objective."""
    return {'actors': {actor: {
        'eta_h': row['eta_h'] if math.isfinite(row['eta_h']) else None,
        'exposure_ratio': row['exposure']['reserve_ratio'],
        'allocated_exposure_ratio': row.get('team_exposure_after', row['exposure'])['reserve_ratio'],
        'exposure_unknown': row['exposure']['unknown'], 'relief': row['relief'],
        'support_before': row['support_before'], 'support_after': row['support_after'],
        'support_window_h': row['support_window_h'],
        'primary_targets': tuple(sorted(row['target_etas'])),
        'contingency_targets': tuple(sorted(row['contingency_target_etas'])),
    } for actor, row in facts['actors'].items()},
        'healing_services': facts['healing_services'],
        'team_exchange': facts.get('team_exchange', {}),
        'team_delay': facts.get('team_delay', {})}
