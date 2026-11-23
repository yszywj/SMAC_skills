"""Visible group support facts shared by combat joining and task observation."""
import math

from .context import distance, GAME_LOOPS_PER_SPEED_SECOND
from .encounters import combat_members
from .combat_exchange import exposure_window, imminent_exchange_danger, movement_exchange


def connected_groups(context, radius, members=None):
    members = combat_members(context) if members is None else members
    pending = {a for a in members if a in context.snapshot.allies
               and context.snapshot.allies[a].alive}
    groups = []
    while pending:
        actor = min(pending)
        pending.remove(actor)
        todo, group = [actor], []
        while todo:
            actor = todo.pop()
            group.append(actor)
            near = {a for a in pending if distance(
                context.snapshot.allies[actor].position,
                context.snapshot.allies[a].position) <= radius}
            pending -= near
            todo.extend(sorted(near))
        groups.append(tuple(sorted(group)))
    return sorted(groups)


def forecast(context, actor, position):
    predict = getattr(context, "forecast_risk", context.risk_at)
    return predict(actor, position)


def group_status(context, members):
    """Explicit heuristic, not a simulated battle or calibrated DPS ratio.

    Only physical current contact marks fighting. Forecast threats inform
    pressure; effective-health and window-ready fire distinguish a supported
    group from two wounded/cooling units standing next to one another.
    """
    threats, current_threats, ready_damage, ready_shooters = set(), set(), 0., 0
    strength, fighting = 0., False
    for actor in members:
        unit = context.snapshot.allies[actor]
        current = context.risk_at(actor, unit.position)
        current_threats.update(current.threats)
        threats.update(forecast(context, actor, unit.position).threats)
        contacts = [e for e in context.attackable_enemies(actor)
                    if context.contact_status(actor, e) == "in_range"]
        ready = [context.fire_estimate(actor, e) for e in contacts]
        ready = [estimate for estimate in ready if estimate.window_shots >= 1]
        ready_shooters += bool(ready)
        ready_damage += max((estimate.damage_est or 0. for estimate in ready), default=0.)
        health = unit.effective_health / max(unit.max_health + unit.max_shield, 1.)
        strength += health * (1. if ready else .5)
        fighting |= bool(contacts or current.threats or current.effect_weapon_near
                         or unit.health_lost > 0 or unit.shield_lost > 0)
    return {"members": tuple(members), "threats": tuple(sorted(threats)),
            "ready_shooters": ready_shooters, "ready_damage": ready_damage,
            "available_strength": strength, "pressure": len(threats),
            "current_pressure": len(current_threats),
            "support_ratio": strength / max(len(threats), 1), "fighting": fighting}


def support_readiness(context, joiners, supporters, radius, positions=None):
    positions = positions or {}
    units = context.snapshot.allies
    cover = [positions.get(a, units[a].position) for a in supporters
             if a in units and units[a].alive]
    moving = [positions.get(a, units[a].position) for a in joiners
              if a in units and units[a].alive]
    if not cover or not moving:
        return {"fraction": 0., "gap": float("inf"), "max_gap": float("inf")}
    gaps = [max(min(distance(point, other) for other in cover) - radius, 0.)
            for point in moving]
    return {"fraction": sum(gap <= .05 for gap in gaps) / len(gaps),
            "gap": sum(gaps) / len(gaps), "max_gap": max(gaps)}


def effective_support(context, joiners, supporters, positions=None, radius=6.,
                      targets_by_actor=None, roles=None):
    """Observable weapon/movement readiness, not simulated future damage.

    ETA uses this actor's actual weapon AND command-mask reach, movement speed
    and cooldown. Cover survival is a conservative volley-pressure heuristic:
    enemy targeting/cooldown and future deaths are unknown. It is only a
    viability discount, never a promise that a receiver lives that long.
    """
    positions = positions or {}
    units = context.snapshot.allies
    members = tuple(dict.fromkeys((*joiners, *supporters)))
    cache = getattr(context, "_effective_support_cache", None)
    if cache is None:
        cache = context._effective_support_cache = {}
    cache_key = (tuple(joiners), tuple(supporters), radius,
                 None if targets_by_actor is None else tuple(
                     (a, tuple(targets_by_actor.get(a, ()))) for a in members),
                 None if roles is None else tuple((a, roles.get(a)) for a in joiners),
                 tuple((a, tuple(positions.get(a, units[a].position)))
                       for a in members if a in units))
    if cache_key in cache:
        return cache[cache_key]
    eta, gaps = {}, {}
    for actor in members:
        unit = units.get(actor)
        if unit is None or not unit.alive:
            eta[actor], gaps[actor] = float("inf"), float("inf")
            continue
        point = positions.get(actor, unit.position)
        targets = (context.snapshot.enemies if targets_by_actor is None
                   else targets_by_actor.get(actor, ()))
        possibilities = [context.projected_attack_opportunity(actor, enemy, point)
                         for enemy in targets if enemy in context.snapshot.enemies]
        gap = min((p.center_gap for p in possibilities), default=float("inf"))
        speed = context.speed(actor)
        scale = speed * context.H / GAME_LOOPS_PER_SPEED_SECOND if speed else 0.
        travel = gap / scale if scale > 0 else (0. if gap <= .05 else float("inf"))
        gaps[actor] = gap
        eta[actor] = max(travel, unit.weapon_cooldown_loops / max(context.H, 1))
    geometry = support_readiness(context, joiners, supporters, radius, positions)
    arrival = []
    for actor in joiners:
        unit = units.get(actor)
        if unit is None or not unit.alive:
            continue
        point = positions.get(actor, unit.position)
        gap = min((max(distance(point, positions.get(b, units[b].position)) - radius, 0.)
                   for b in supporters if b in units and units[b].alive), default=float("inf"))
        speed = context.speed(actor)
        scale = speed * context.H / GAME_LOOPS_PER_SPEED_SECOND if speed else 0.
        travel = gap / scale if scale > 0 else (0. if gap <= .05 else float("inf"))
        # A combat receiver must last until the joining fire can arrive, not
        # merely until a friendly body crosses its support ring. Escape has
        # a different completion condition: arrival at safe support suffices.
        arrival.append(max(travel, eta[actor]) if roles is not None
                       and roles.get(actor) == "join" else travel)
    arrival_windows = max(arrival, default=float("inf"))
    cover_health, volleys = 0., {}
    critical = False
    for actor in supporters:
        unit = units.get(actor)
        if unit is None or not unit.alive:
            continue
        cover_health += unit.effective_health
        risk = context.risk_at(actor, positions.get(actor, unit.position))
        critical |= bool(risk.effect_weapon_near or
                         (risk.threats and unit.health_fraction <= .25))
        for enemy in risk.threats:
            amount = context.enemy_attack_damage(enemy, actor)
            volleys[enemy] = max(volleys.get(enemy, 0.), amount if amount is not None else 14.)
    pressure = sum(volleys.values())
    survival = cover_health / pressure if pressure > 0 else float("inf")
    viability = (0. if critical or cover_health <= 0 else
                 1. if not pressure else min(1., survival / max(arrival_windows + 1., 1.)))
    # Healthy receivers cannot lend their reserve to an exposed frontline.
    # Bound aggregate cover by each exposed member's own short loss window.
    local_cover = {}
    for actor in supporters:
        if actor not in units or not units[actor].alive:
            continue
        point = positions.get(actor, units[actor].position)
        local_cover[actor] = 1.
        if forecast(context, actor, point).threats:
            facts = exposure_window(context, actor, point,
                horizon_h=arrival_windows + 1., certain=actor not in positions)
            local_cover[actor] = facts["viability"]
    if local_cover:
        viability = min(viability, min(local_cover.values()))
    ready = sum(value <= 2. for value in eta.values())
    result = {"effective_ready_count": ready, "fraction": ready / max(len(members), 1),
            "eta_by_actor": eta, "gap_by_actor": gaps,
            "joiner_gap": geometry["gap"], "arrival_windows": arrival_windows,
            "cover_viability": viability, "cover_survival_windows": survival,
            "cover_viability_by_actor": local_cover}
    cache[cache_key] = result
    return result


def _member_interrupt(context, actor, data):
    """An individual cannot borrow another receiver's survival reserve."""
    unit = context.snapshot.allies[actor]
    risk = context.risk_at(actor, unit.position)
    if risk.effect_weapon_near:
        return "effect_danger"
    if actor in data["supporters"] and imminent_exchange_danger(context, actor):
        return "cover_emergency"
    if actor in data["supporters"] and unit.health_fraction <= .25 and risk.threats:
        return "cover_emergency"
    initial = data.get("initial_threat_counts", {}).get(actor, 0)
    if unit.health_lost + unit.shield_lost > 0 and len(risk.threats) >= max(3, initial + 2):
        return "danger_escalated"
    return None


def viable_rejoin_data(context, data, horizon_h, instance=None):
    """Read-only, shrinking projection of a duty; identity changes on commit.

    No new ally may enter a running duty, and replacement tags never inherit
    membership. Each retained receiver must survive its own exposure window.
    Removing a slow joiner can make a smaller group feasible, but cannot give
    remaining members a fresh deadline or progress baseline.
    """
    if not data.get("local_cover_enabled", False):
        return data
    units = context.snapshot.allies
    tags = dict(zip(instance.actors, instance.ally_tags)) if instance is not None else {}
    valid = lambda a: (a in units and units[a].alive and
                       (a not in tags or units[a].tag == tags[a]) and
                       not _member_interrupt(context, a, data) and
                       (a in data["supporters"] or not (
                           imminent_exchange_danger(context, a) or
                           (units[a].health_fraction <= .25 and context.threats_to(a)))))
    joiners = tuple(a for a in data["joiners"] if valid(a))
    cover = tuple(a for a in data["supporters"] if valid(a))
    radius, targets = data["support_radius"], data.get("targets_by_actor")
    while joiners and cover:
        facts = effective_support(context, joiners, cover, radius=radius,
                                  targets_by_actor=targets, roles=data.get("roles"))
        feasible = []
        for actor in joiners:
            if data.get('arrival_exposure_enabled') and data.get('roles', {}).get(actor) == 'join':
                exchange = movement_exchange(context, actor, moving=False,
                    targets=None if targets is None else targets.get(actor, ()))
                if not data.get('joint_assessment_enabled') and not exchange['admissible']:
                    continue  # receivers cannot lend reserve to the arriving member
                if data.get('joint_assessment_enabled') and (not math.isfinite(exchange['eta_h'])
                        or exchange['after']['effect_danger']):
                    continue
            geometry = support_readiness(context, (actor,), cover, radius)
            speed = context.speed(actor)
            scale = speed * context.H / GAME_LOOPS_PER_SPEED_SECOND if speed else 0.
            travel = (geometry["gap"] / scale if scale > 0 else
                      0. if geometry["gap"] <= .05 else float("inf"))
            engaged = geometry["gap"] <= .05 and facts["gap_by_actor"][actor] <= .05
            eta = (travel if data.get("roles", {}).get(actor) == "escape"
                   else max(travel, facts["eta_by_actor"][actor]))
            if engaged or eta <= horizon_h:
                feasible.append(actor)
        if tuple(feasible) != joiners:
            joiners = tuple(feasible)
            continue
        # Whether the *selected* receiver fires, moves or can survive to cover
        # arrival is a final joint-action fact, not an observation-only veto.
        if data.get('joint_assessment_enabled'):
            break
        safe_cover = tuple(a for a in cover
                           if facts["cover_viability_by_actor"].get(a, 0.) >= .5)
        if safe_cover != cover:
            cover = safe_cover
            continue
        if facts["cover_viability"] < .5:
            return None
        break
    if not joiners or not cover:
        return None
    if joiners == tuple(data["joiners"]) and cover == tuple(data["supporters"]):
        return data
    members = set((*joiners, *cover))
    result = dict(data, joiners=list(joiners), supporters=list(cover))
    for key in ("roles", "joiner_progress", "initial_threat_counts", "targets_by_actor", "allocated_targets"):
        if key in data:
            result[key] = {a: value for a, value in data[key].items() if a in members}
    return result


def active_rejoin_data(context, instance):
    """Current feasible members of the original bounded task, without writes."""
    remaining = ((instance.start_loop + instance.max_loops - context.snapshot.game_loop)
                 / max(context.H, 1) if instance.max_loops is not None else 0.)
    return viable_rejoin_data(context, instance.extra, max(remaining, 0.), instance)


def task_interrupt(context, instance):
    data = instance.extra
    if data.get("shared_front_enabled") and not any(
            e in context.snapshot.enemies for e in data.get("front_targets", ())):
        return "front_lost"
    if data.get("local_cover_enabled"):
        if active_rejoin_data(context, instance) is not None:
            return None
        # Leave an otherwise intact duty's elapsed budget to the ordinary
        # deadline hook; lack of remaining travel time is not a new threat.
        if instance.max_loops is not None and context.snapshot.game_loop >= \
                instance.start_loop + instance.max_loops and all(
                    a in context.snapshot.allies and context.snapshot.allies[a].alive
                    and not _member_interrupt(context, a, data) for a in instance.actors):
            return None
        return "cover_emergency"
    units = context.snapshot.allies
    if any(a not in units or not units[a].alive for a in instance.actors):
        return "member_lost"
    return next((reason for a in instance.actors
                 if (reason := _member_interrupt(context, a, data))), None)


def joiner_progress_errors(context, joiners, supporters, radius, positions=None,
                           targets_by_actor=None, roles=None):
    """Distance still needed for support AND useful weapon reach, per joiner.

    Cooldown elapsing alone is not spatial progress. The same metric is used
    at task creation and observation, so the first observation cannot invent
    a progress event merely by changing the definition.
    """
    positions = positions or {}
    units = context.snapshot.allies
    cover = [positions.get(a, units[a].position) for a in supporters
             if a in units and units[a].alive]
    effective = effective_support(context, joiners, supporters, positions, radius,
                                  targets_by_actor, roles)
    return {actor: max(min((distance(positions.get(actor, units[actor].position), p)
                          for p in cover), default=float("inf")) - radius, 0.)
                    + (0. if roles is not None and roles.get(actor) == "escape" else
                       min(effective["gap_by_actor"].get(actor, float("inf")), 40.))
            for actor in joiners if actor in units and units[actor].alive}


def rejoin_continuation_opportunity(context, instance):
    """Bounded opportunity to finish an observed, progressing live duty.

    These are per-joiner eligibility weights, not a bonus for retaining an
    instance ID. A final-plan evaluator must charge only abandonment of
    progress, and must exempt actual firing and safety improvements. New,
    stalled, completed, infeasible and dangerous duties have no entitlement.
    Neither cover population nor time already spent increases the weights.
    """
    if instance.skill != "combat_rejoin" or instance.termination_reason \
            or task_interrupt(context, instance):
        return {}
    data, loop = active_rejoin_data(context, instance), context.snapshot.game_loop
    if instance.max_loops is None:
        return {}  # no unbounded promise
    remaining = instance.start_loop + instance.max_loops - loop
    if remaining <= 0:
        return {}
    joiners, supporters = data["joiners"], data["supporters"]
    radius = data["support_radius"]
    facts = effective_support(context, joiners, supporters, radius=radius,
                              targets_by_actor=data.get("targets_by_actor"),
                              roles=data.get("roles") if data.get("local_cover_enabled") else None)
    if facts["cover_viability"] < .5:
        return {}
    units, weights = context.snapshot.allies, {}
    for actor in joiners:
        record = data.get("joiner_progress", {}).get(actor, {})
        last = record.get("last_progress_loop", instance.start_loop)
        if not record.get("observed_progress") or loop - last > 2 * context.H:
            continue
        unit = units[actor]
        gap = max(min(distance(unit.position, units[a].position)
                      for a in supporters) - radius, 0.)
        escaping = data.get("local_cover_enabled") and data.get("roles", {}).get(actor) == "escape"
        weapon_gap = 0. if escaping else facts["gap_by_actor"][actor]
        if gap <= .05 and weapon_gap <= .05:
            continue  # already engaged; cooldown is not an unfinished journey
        speed = context.speed(actor)
        scale = speed * context.H / GAME_LOOPS_PER_SPEED_SECOND if speed else 0.
        travel = gap / scale if scale > 0 else (0. if gap <= .05 else float("inf"))
        eta = travel if escaping else max(travel, facts["eta_by_actor"][actor])
        stall_left = last + data["no_progress_loops"] - loop
        feasible = min(remaining, stall_left) / max(context.H, 1)
        if not math.isfinite(eta) or eta > feasible or feasible <= 0:
            continue
        weights[actor] = facts["cover_viability"] / (1. + max(eta, 0.))
    return weights


def task_outcome(context, instance):
    """Observation hook; proposals must not call this mutating function."""
    if instance.skill != "combat_rejoin":
        return None
    reason = task_interrupt(context, instance)
    if reason:
        return reason
    data = active_rejoin_data(context, instance)
    if data is None:
        return None
    if tuple(sorted((*data["joiners"], *data["supporters"]))) != tuple(instance.actors):
        return None  # pruning alone is neither observed progress nor completion
    errors = joiner_progress_errors(context, data["joiners"], data["supporters"],
                                   data["support_radius"], targets_by_actor=data.get("targets_by_actor"),
                                   roles=data.get("roles") if data.get("local_cover_enabled") else None)
    progress = data.setdefault("joiner_progress", {})
    progressed = False
    for actor, error in errors.items():
        record = progress.setdefault(actor, {"best_error": error,
            "last_progress_loop": context.snapshot.game_loop, "observed_progress": False})
        if error < record["best_error"] - .05:
            progressed = True
            record.update(best_error=error, last_progress_loop=context.snapshot.game_loop,
                          observed_progress=True)
    ready = support_readiness(context, data["joiners"], data["supporters"],
                              data["support_radius"])
    effective = effective_support(context, data["joiners"], data["supporters"],
                                  radius=data["support_radius"],
                                  targets_by_actor=data.get("targets_by_actor"),
                                  roles=data.get("roles") if data.get("local_cover_enabled") else None)
    # A near melee unit must still reach a real engagement position. A
    # retreating member may complete at safe support without attacking.
    participation = all(any(context.can_issue_attack(a, e)
                            and context.contact_status(a, e) == "in_range"
                            for e in (data.get("targets_by_actor", {}).get(a, ())
                                      if data.get("shared_front_enabled") else context.snapshot.enemies)
                            if e in context.snapshot.enemies)
        or (data.get("roles", {}).get(a) == "escape" and not context.threats_to(a))
        for a in data["joiners"])
    if not context.snapshot.enemies:
        participation = True
    # Both squads must actually form a connected supporting group. Command
    # endpoints and the shrinking alive population cannot complete this duty.
    if participation and ready["fraction"] >= data["readiness_fraction"] and len(connected_groups(
            context, data["support_radius"], instance.actors)) == 1:
        return "support_connected"
    effective_gap = sum(0. if data.get("local_cover_enabled") and
                        data.get("roles", {}).get(a) == "escape" else
                        min(effective["gap_by_actor"].get(a, 0.), 40.)
                        for a in data["joiners"]) / max(len(data["joiners"]), 1)
    progress_gap = ready["gap"] + effective_gap
    improving = (progressed if data.get("local_cover_enabled") else
                 progress_gap < data.get("best_support_gap", float("inf")) - .05)
    if improving:
        data["best_support_gap"] = progress_gap
        data["last_support_progress_loop"] = context.snapshot.game_loop
        instance.last_progress_loop = context.snapshot.game_loop
    if context.snapshot.game_loop - data.get(
            "last_support_progress_loop", instance.start_loop) >= data["no_progress_loops"]:
        return "no_progress"
    return None
