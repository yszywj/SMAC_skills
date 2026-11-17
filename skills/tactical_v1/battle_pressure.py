"""Bounded visible exposure and lethal threat relief, not a battle simulator.

Enemy weapon amounts describe one possible volley, not enemy DPS or timing.
Ordinary shots do not suppress weapons: neither target allocation nor nearby
potential shooters reduce incoming exposure. Unknown weapon amounts remain
unknown; the evaluator's separate risk term handles uncertainty.
"""


def _exposure(context, actor, point):
    """Cache known/unknown exposure by snapshot-local actor and final point."""
    cache = getattr(context, "_battle_pressure_exposures", None)
    if cache is None:
        cache = context._battle_pressure_exposures = {}
    key = actor, tuple(point)
    if key in cache:
        return cache[key]
    forecast = context.forecast_risk(actor, point)
    current = set(forecast.current_threats)
    exposure = {}
    for enemy in forecast.reachable_threats:
        exposure[enemy] = (1. if enemy in current else .5,
                           context.enemy_attack_damage(enemy, actor))
    cache[key] = exposure
    return exposure


def _pressure(context, enemy, positions):
    """An enemy threatening a fragile frontline matters more than its ID.

    Only the two largest compatible exposures count. This bounds the number
    of simultaneous victims attributed to one weapon; it does not assume a
    single enemy can fire a separate full volley at every friendly unit.
    """
    values, threatened = [], []
    for actor, unit in context.snapshot.allies.items():
        if not unit.alive:
            continue
        point = positions.get(actor, unit.position)
        exposure = _exposure(context, actor, point).get(enemy)
        if exposure is None:
            continue
        threatened.append(actor)
        weight, damage = exposure
        if damage is not None and damage > 0:
            values.append(min(1., weight * damage / max(unit.effective_health, 1.)))
    values.sort(reverse=True)
    value = min(1., (values[0] if values else 0.)
                + .25 * (values[1] if len(values) > 1 else 0.))
    return value, tuple(threatened)


def enemy_pressure(context, enemy_id, positions=None):
    """Snapshot objective pressure in [0,1], independent of unit type names."""
    if enemy_id not in context.snapshot.enemies:
        return 0.
    if positions is not None:
        return _pressure(context, enemy_id, positions)[0]
    cache = getattr(context, "_battle_pressure_targets", None)
    if cache is None:
        cache = context._battle_pressure_targets = {}
    if enemy_id not in cache:
        cache[enemy_id] = _pressure(context, enemy_id, {})[0]
    return cache[enemy_id]


def joint_pressure(context, actions, report):
    """Use final positions and known lethal damage to price threat pressure.

    The legacy ``unsupported_*`` keys now mean unresolved volley exposure,
    normalized by friendly EHP. Firing one ordinary shot cannot cancel the
    threatening weapon, and a future firing opportunity is not current cover.
    These costs consequently do not depend on how nonlethal fire is allocated.

    Only a window-qualified known damage lower bound reaching enemy EHP earns
    ``pressure_relief``. It is bounded once per enemy and is not also deducted
    from exposure: enemy shot timing is unknown, so no guaranteed first-strike
    protection is assumed. Partial damage already has its own evaluator term.
    This helper never changes the damage or kill ledgers. ``actions`` remains
    an interface argument; the report already describes those final actions.
    """
    units = context.snapshot.allies
    positions = {a: point for a, (point, _) in report.positions.items()}
    exposures = {}
    pressure_values = {e: [] for e in context.snapshot.enemies}
    threatened = {e: [] for e in context.snapshot.enemies}
    unknown = 0
    for actor, unit in units.items():
        if not unit.alive:
            continue
        exposure = _exposure(context, actor, positions.get(actor, unit.position))
        exposures[actor] = exposure
        unknown += sum(damage is None for _, damage in exposure.values())
        for enemy, (weight, damage) in exposure.items():
            threatened[enemy].append(actor)
            if damage is not None and damage > 0:
                pressure_values[enemy].append(min(
                    1., weight * damage / max(unit.effective_health, 1.)))
    relieved = 0.
    for enemy, unit in context.snapshot.enemies.items():
        known = max(report.damage_known.get(enemy, 0.), 0.)
        lower = min(known, report.damage_bounds.get(enemy, (known, known))[0])
        if unit.effective_health <= 0 or lower < unit.effective_health:
            continue
        values = sorted(pressure_values[enemy], reverse=True)
        pressure = min(1., (values[0] if values else 0.)
                       + .25 * (values[1] if len(values) > 1 else 0.))
        relieved += pressure
    unsupported = {}
    for actor, exposure in exposures.items():
        incoming = 0.
        for enemy, (weight, damage) in exposure.items():
            if damage is None:
                continue
            incoming += weight * max(damage, 0.)
        unsupported[actor] = min(1., incoming / max(units[actor].effective_health, 1.))
    return {"unsupported_exposure": sum(unsupported.values()),
            "pressure_relief": relieved, "unknown_enemy_pairs": unknown,
            "unsupported_actors": unsupported, "threatened_allies": threatened}
