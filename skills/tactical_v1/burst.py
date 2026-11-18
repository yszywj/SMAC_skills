"""One-use, ground-only explosion model shared by S3 and threat queries.

The protocol's empty Baneling weapon list is not a zero-damage weapon. Native
effect calibration is separate from command legality. A projected trigger is
an opportunity, never an observed hit, kill, or proof of self-detonation.
"""
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Dict, Optional, Tuple


@dataclass(frozen=True)
class BurstProfile:
    blast_radius: Optional[float] = None
    blast_radius_upper: Optional[float] = None
    trigger_reach: Optional[float] = None
    base_damage: Optional[float] = None
    light_bonus: Optional[float] = None
    evidence: str = "unknown_native_effect"
    geometry_known: bool = False
    damage_known: bool = False


@dataclass(frozen=True)
class BurstOutcome:
    actor: str
    trigger: str
    position: Tuple[float, float]
    opportunity: float
    damage_potential: Dict[str, Optional[float]] = field(default_factory=dict)
    geometry_known: bool = False
    evidence: str = "unknown_native_effect"
    target_id: Optional[str] = None
    trigger_loops: float = 0.
    uncertain: bool = True


@dataclass(frozen=True)
class BurstRegion:
    """One action's reference, shared by proposals, baseline and scoring."""
    target_id: str
    position: Tuple[float, float]
    progress: float
    arrival_loops: float
    incoming: float
    unknown: int
    value: float


_PROFILE_PATH = Path(__file__).with_name("burst_native_profile.json")
_CALIBRATION = json.loads(_PROFILE_PATH.read_text(encoding="utf8")) \
    if _PROFILE_PATH.exists() else {}


def profile(context) -> BurstProfile:
    """Installation calibration guarded by the visible native catalog shape.

    Payloads currently omit native build identity. The benchmark separately
    verifies its build against the calibration; an optional supplied version
    mismatch is rejected, but this is not strict runtime version validation.
    """
    data = _CALIBRATION
    if not data.get("verified"):
        return BurstProfile()
    raw = context.snapshot.raw_state
    native = raw.get("unit_type_profiles", {}).get("9", {})
    identity = data.get("catalog_contract", {})
    if any(native.get(key) != value for key, value in identity.items()):
        return BurstProfile(evidence="native_effect_catalog_mismatch")
    observed_version = raw.get("native_game_version")
    if observed_version is not None and observed_version != data.get("game_version"):
        return BurstProfile(evidence="native_effect_version_mismatch")
    upgrade_supported = all(unit.get("attack_upgrade_level") == 0
                            for side in ("allies", "visible_enemies")
                            for unit in raw.get(side, {}).values()
                            if unit.get("unit_type_id") == 9)
    return BurstProfile(
        blast_radius=data.get("blast_radius"),
        blast_radius_upper=data.get("blast_radius_upper"),
        trigger_reach=data.get("trigger_reach"),
        base_damage=data.get("base_damage"), light_bonus=data.get("light_bonus"),
        evidence=data["evidence"], geometry_known=data.get("blast_radius") is not None,
        damage_known=data.get("base_damage") is not None and upgrade_supported)


def _distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def affects(profile, center, target) -> bool:
    """Native effect geometry; an unknown radius only establishes body contact.

    The fallback has finite scope and is explicitly uncalibrated. It cannot
    create numeric damage or make every visible Baneling an imminent threat.
    """
    if target.is_flying:
        return False
    radius = profile.blast_radius if profile.geometry_known else 0.
    return _distance(center, target.position) <= radius + target.radius


def damage(context, target_id, *, enemy_target=True) -> Optional[float]:
    """Calibrated unshielded, zero-upgrade potential at a verified trigger.

    Armor/Light are native target attributes. Unsupported shield/upgrade
    combinations remain unknown, rather than reusing a guessed gun weapon.
    """
    effect = profile(context)
    targets = context.snapshot.enemies if enemy_target else context.snapshot.allies
    target = targets.get(target_id)
    if target is None or target.is_flying:
        return 0. if target is not None else None
    if not effect.damage_known:
        return None
    raw = context.snapshot.raw_state
    unit = raw.get("visible_enemies" if enemy_target else "allies", {}).get(target_id, {})
    catalog = raw.get("unit_type_profiles", {}).get(str(target.unit_type_id), {})
    if not catalog.get("known") or target.shield > 0 \
            or unit.get("armor_upgrade_level") != 0:
        return None
    armor, attributes = catalog.get("armor"), catalog.get("attributes")
    if not isinstance(armor, (int, float)) or not math.isfinite(armor) \
            or not isinstance(attributes, list):
        return None
    if 1 in attributes and effect.light_bonus is None:
        return None
    raw_damage = effect.base_damage + (effect.light_bonus if 1 in attributes else 0.)
    return max(raw_damage - armor, .5)


def center_reach(context, target, *, trigger=False) -> Tuple[float, bool]:
    effect = profile(context)
    reach = effect.trigger_reach if trigger else effect.blast_radius_upper
    # This is a finite body-contact fallback, not a made-up effect radius.
    return (max(float(reach), 0.) + target.radius if reach is not None
            else target.radius, reach is not None)


def trigger_position(context, actor, enemy_id, horizon=None):
    """Projected source center, bounded by native speed and trigger reach."""
    source = context.snapshot.allies[actor]
    target = context.snapshot.enemies.get(enemy_id)
    effect = profile(context)
    if target is None or target.is_flying or effect.trigger_reach is None:
        return source.position, False
    gap = _distance(source.position, target.position)
    trigger = source.radius + target.radius + effect.trigger_reach
    travel = max(gap - trigger, 0.)
    speed = context.speed(actor)
    budget = 0. if speed is None else speed * (horizon or context.H) / 16.
    amount = min(travel, budget)
    fraction = amount / gap if gap > 0 else 0.
    point = (source.position[0] + (target.position[0] - source.position[0]) * fraction,
             source.position[1] + (target.position[1] - source.position[1]) * fraction)
    return point, travel <= budget + 1e-6


def area_damage(context, center):
    effect = profile(context)
    return {enemy_id: damage(context, enemy_id)
            for enemy_id, enemy in context.snapshot.enemies.items()
            if affects(effect, center, enemy)}


def _exposure(context, actor, point, horizon=None, *, physical=False):
    """Visible volley bound, with native cooldown exclusion when available.

    Enemy targeting and projectile flight are unknown. This never establishes
    death, and a normalized exposure discount below is NOT a probability.
    """
    horizon = context.H if horizon is None else max(float(horizon), 0.)
    risk = context.forecast_risk(actor, point, max(1, int(math.ceil(horizon))))
    ids = set(risk.current_threats if physical else risk.reachable_threats)
    incoming, unknown = 0., 0
    for enemy_id in ids:
        raw = context.snapshot.raw_state.get('visible_enemies', {}).get(enemy_id, {})
        cooldown = raw.get('weapon_cooldown_loops')
        suicide = context.enemy_capabilities[enemy_id].suicide_attack
        if not suicide and isinstance(cooldown, (int, float)) and cooldown > horizon:
            continue
        amount = risk.damage_estimates.get(enemy_id)
        if amount is None:
            unknown += 1
        else:
            incoming += amount
        if not suicide and not isinstance(cooldown, (int, float)):
            unknown += 1
    return incoming, unknown


def action_position(context, actor, action):
    semantic = context.decode(actor, action)
    if semantic.kind == 'attack' and semantic.target in context.snapshot.enemies:
        return trigger_position(context, actor, semantic.target)[0]
    return tuple(context.predict_position(actor, action).position)


def supported_death_point(context, actor, destination, horizon=None):
    """First sampled lethal warning backed by loss, targeting or an effect.

    The all-enemy volley remains a scoring bound in ``_exposure``. It does
    not replace a reachable contact point with a hypothetical earlier death.
    This pure query deliberately does not call region/assignment allocation.
    """
    from .combat_exchange import danger_evidence
    source = context.snapshot.allies[actor]
    horizon_h = max(1., float(context.H if horizon is None else horizon) / max(context.H, 1))
    for fraction in (0., .5, 1.):
        point = tuple(source.position[i] + (destination[i] - source.position[i]) * fraction
                      for i in (0, 1))
        facts = danger_evidence(context, actor, point, horizon_h=horizon_h)
        if facts['immediate'] and facts['supported_amount'] >= source.effective_health:
            return point, fraction, facts
    return None


def outcome(context, actor, action, position=None) -> BurstOutcome:
    """Snapshot-local cached mutually exclusive trigger opportunity."""
    cache = getattr(context, '_burst_outcome_cache', None)
    if cache is None:
        cache = context._burst_outcome_cache = {}
    # Native attack auto-approach is reconstructed independently of the
    # ordinary gun predictor's stationary placeholder.
    key = (actor, action, tuple(position) if position is not None
           and context.decode(actor, action).kind != 'attack' else None)
    if key not in cache:
        cache[key] = _outcome(context, actor, action, position)
    return cache[key]


def _outcome(context, actor, action, position=None) -> BurstOutcome:
    """One mutually exclusive opportunity: contact, origin, or path death.

    Prefer an immediate legal trigger. Otherwise supported lethal danger
    bounds the useful source center, rather than also granting a later arrival.
    An all-enemy volley only discounts value; it cannot prove an earlier death.
    Path sampling is explicitly uncertain; it is not a native timing model.
    """
    source = context.snapshot.allies[actor]
    effect = profile(context)
    semantic = context.decode(actor, action)
    predicted = action_position(context, actor, action)
    if semantic.kind != 'attack' and position is not None:
        predicted = tuple(position)
    trigger, weight, center = "none", 0., predicted
    target_id = semantic.target if semantic.kind == 'attack' else None
    reachable, arrival = False, float(context.H)
    speed = context.speed(actor)
    if semantic.kind == "attack" and semantic.target in context.snapshot.enemies:
        center, reachable = trigger_position(context, actor, semantic.target)
        reachable = reachable and context.can_issue_attack(actor, semantic.target)
        arrival = _distance(source.position, center) * 16. / speed if speed else float(context.H)
    if reachable and arrival <= 1e-6:
        trigger, weight = 'active_contact_opportunity', 1.
    else:
        # Origin, middle, endpoint are exclusive ordered hypotheses. They do
        # not each receive the complete damage or consumption budget.
        horizon = min(arrival, context.H) if reachable else context.H
        maximum_incoming, maximum_unknown = 0., 0
        for fraction in (0., .5, 1.):
            point = tuple(source.position[i] + (predicted[i] - source.position[i]) * fraction
                          for i in (0, 1))
            incoming, unknown = _exposure(context, actor, point, horizon, physical=True)
            maximum_incoming = max(maximum_incoming, incoming)
            maximum_unknown = max(maximum_unknown, unknown)
        death = supported_death_point(context, actor, predicted, horizon)
        if death is not None:
            center, fraction, facts = death
            trigger = ('death_at_origin_opportunity' if fraction == 0.
                       else 'death_on_path_opportunity')
            supported = facts['supported_amount']
            weight = supported / (supported + max(source.effective_health, 1.))
            arrival = horizon * fraction
        elif reachable:
            center = predicted
            trigger = 'active_contact_opportunity'
            weight = source.effective_health / max(source.effective_health + maximum_incoming, 1.)
            weight /= 1. + maximum_unknown
    # Unify the burst point and the enemy-group prediction MOMENT (task 6):
    # for an attack-trigger hypothesis, shift the trigger center with the
    # target's short-window predicted position at the arrival loop. Death
    # hypotheses keep the sampled death location (that is where WE are).
    # This one entry point serves both candidate ranking and the final
    # effects rebuild (context.burst_outcome), so neither can drift back to
    # static positions while the other uses motion. Unreliable or missing
    # motion keeps the static result — legal actions are never cancelled.
    if weight and trigger == 'active_contact_opportunity' \
            and target_id in context.snapshot.enemies:
        from .enemy_motion import predicted_position
        future = predicted_position(context, target_id,
                                    context.snapshot.game_loop + max(arrival, 0.))
        if future is not None:
            enemy = context.snapshot.enemies[target_id]
            center = (center[0] + future[0] - enemy.position[0],
                      center[1] + future[1] - enemy.position[1])
    potential = area_damage(context, center) if weight else {}
    if trigger.startswith('death_') and potential:
        target_id = max(potential, key=lambda enemy_id: (
            (potential[enemy_id] or 0.) / max(context.snapshot.enemies[enemy_id].effective_health, 1.),
            -_distance(center, context.snapshot.enemies[enemy_id].position), enemy_id))
    return BurstOutcome(actor, trigger, center, weight, potential,
                        effect.geometry_known, effect.evidence, target_id,
                        arrival if weight else 0., trigger != 'active_contact_opportunity'
                        or arrival > 1e-6)


def useful_area_value(context, center, remaining=None):
    """Residual capped damage fractions for deterministic multi-source allocation."""
    result = 0.
    for enemy_id, amount in area_damage(context, center).items():
        enemy = context.snapshot.enemies[enemy_id]
        health = enemy.effective_health if remaining is None else remaining.get(enemy_id, 0.)
        if amount is not None and health > 0:
            result += min(amount, health) / max(enemy.effective_health, 1.)
    return result


def exchange_value(context, actor, center, remaining=None, *, opportunity=1.,
                   consumed=(), positions=None):
    """Conservative HP exchange gate for a compulsory one-use commitment.

    Unlike damage/current-HP utility, a nearly dead victim is not worth a
    whole healthy source. This is a gate, not a replacement for final joint
    scoring: marginal or unknown exchanges remain legal soft alternatives.
    Enemy death effects are conditional costs, never observed damage.
    """
    source = context.snapshot.allies[actor]
    effect = profile(context)
    potential = area_damage(context, center)
    useful, known, death_sources = 0., True, []
    for enemy_id, amount in potential.items():
        enemy = context.snapshot.enemies[enemy_id]
        health = enemy.effective_health if remaining is None else remaining.get(enemy_id, 0.)
        if amount is None:
            known = False
            continue
        useful += min(max(health, 0.), amount) * opportunity
        if health > 0. and amount >= health and context.enemy_capabilities[enemy_id].suicide_attack:
            death_sources.append(enemy_id)
    collateral, endangered = 0., set()
    positions = positions or {}
    for enemy_id in death_sources:
        enemy = context.snapshot.enemies[enemy_id]
        for peer, unit in context.snapshot.allies.items():
            if peer == actor or peer in consumed or not unit.alive or unit.is_flying:
                continue
            point = positions.get(peer, unit.position)
            radius = effect.blast_radius_upper if effect.blast_radius_upper is not None else 0.
            if _distance(point, enemy.position) > radius + unit.radius:
                continue
            amount = damage(context, peer, enemy_target=False)
            if amount is None:
                known = False
            else:
                collateral += min(unit.effective_health, amount)
                if amount >= unit.effective_health:
                    endangered.add(peer)
    # Its remaining explosion is still a full resource when the source is
    # wounded. Cheap last-gasp trades may win utility, but are not mandatory.
    consumption = max(source.max_health + source.max_shield, source.effective_health, 1.)
    net = useful - consumption - collateral
    return {'useful_hp': useful, 'source_cost_hp': consumption,
            'counter_burst_hp': collateral, 'net_hp': net, 'known': known,
            'worthwhile': bool(known and net > 1e-6 and not endangered),
            'endangered_peers': tuple(sorted(endangered)),
            'death_sources': tuple(death_sources)}


def _better_point_within_window(context, actor, exchange, remaining, margin):
    """Best strictly-better detonation point reachable within ONE decision
    window, or None. A projected point only defers the current positive
    exchange when it is legally reachable now, the straight route is not
    plausibly lethal, and the net gain clears an explicit margin. Dying
    sources, dispersing clusters and thin gains therefore still detonate
    immediately — this never invents an unreachable cluster centre, every
    point is a real native attack trigger anchor.
    """
    source = context.snapshot.allies[actor]
    speed = context.speed(actor)
    effect = profile(context)
    if effect.trigger_reach is None or not speed:
        return None
    progress = getattr(context, 'burst_progress', None)
    if progress is not None and (progress.overdue(context, actor)
            or progress.records.get(actor, {}).get('cashout_reason')):
        return None
    from .combat_exchange import danger_evidence
    if danger_evidence(context, actor, horizon_h=1.)['immediate']:
        return None      # dying now: cash in the current positive exchange
    budget = speed * context.H / 16.
    threshold = exchange['net_hp'] + margin * max(exchange['source_cost_hp'], 1.)
    best, best_net = None, exchange['net_hp']
    for move in sorted(context.move_destinations(actor)):
        point = context.predict_position(actor, move).position
        # Only actual legal move endpoints with a compatible trigger anchor;
        # a straight-line projection through an obstacle is not a route.
        anchored = any(not enemy.is_flying and _distance(point, enemy.position)
                       <= source.radius + enemy.radius + effect.trigger_reach
                       for enemy in context.snapshot.enemies.values())
        if not anchored or _distance(source.position, point) > budget + .1:
            continue
        alternative = exchange_value(context, actor, point, remaining)
        if not alternative['known'] or alternative['net_hp'] <= best_net \
                or alternative['net_hp'] <= threshold:
            continue
        midpoint = tuple((source.position[i] + point[i]) / 2. for i in (0, 1))
        incoming, unknown = _exposure(context, actor, midpoint, context.H, physical=True)
        end_incoming, end_unknown = _exposure(context, actor, point, context.H, physical=True)
        if max(incoming, end_incoming) >= source.effective_health or unknown or end_unknown:
            continue
        best, best_net = point, alternative['net_hp']
    return best


def trigger_assignments(context):
    """SELECTIVE strong-cashout duties (task 7).

    "Currently net-positive" alone no longer forces consumption: the plain
    attack stays a normal candidate competing with side-steps, synchronized
    advances and bounded waits. A STRONG (forced) assignment needs state
    evidence:
      * evidence-backed imminent death while the exchange has value; or
      * the approach deadline / cashout marker fired (waiting has no
        reliable remaining payoff); or
      * a key team-level exchange completing right now (cover fire ready,
        arrival immediate).
    Timing basis is stated per assignment; unknown data degrades the claim
    instead of inventing one. Reservations stay per-snapshot: unselected
    candidates never hold budget, and a changed final action elsewhere is
    re-marginalized on the next observation.
    """
    cache = getattr(context, '_burst_trigger_assignments', None)
    if cache is not None:
        return cache
    selective = getattr(context, 'burst_config', {}).get(
        'selective_commitment_enabled', True)
    margin = float(getattr(context, 'intercept_config', {}).get(
        'burst_defer_margin', .5) or 0.)
    defer_log = getattr(context, '_burst_defer_reasons', None)
    if defer_log is None:
        defer_log = context._burst_defer_reasons = {}
    commit_log = getattr(context, '_burst_commit_reasons', None)
    if commit_log is None:
        commit_log = context._burst_commit_reasons = {}
    from .combat_exchange import danger_evidence
    progress = getattr(context, 'burst_progress', None)
    candidates = []
    for actor, cap in sorted(context.capabilities.items()):
        if not cap.suicide_attack or not context.snapshot.allies[actor].alive:
            continue
        for action in context.legal_actions(actor):
            if context.decode(actor, action).kind != 'attack':
                continue
            result = outcome(context, actor, action)
            if result.trigger == 'active_contact_opportunity' and result.opportunity > 0.:
                candidates.append((actor, action, result))
    remaining = {e: u.effective_health for e, u in context.snapshot.enemies.items()}
    selected, positions = {}, {}
    while candidates:
        ranked = []
        for actor, action, result in candidates:
            if actor in selected:
                continue
            exchange = exchange_value(context, actor, result.position, remaining,
                opportunity=result.opportunity, consumed=selected, positions=positions)
            if not exchange['worthwhile']:
                continue
            better = _better_point_within_window(context, actor, exchange,
                                                 remaining, margin)
            if better is not None:
                defer_log[actor] = {
                    'reason': 'better_reachable_point_within_one_window',
                    'current_net_hp': round(exchange['net_hp'], 2), 'point': list(better)}
                continue
            defer_log.pop(actor, None)
            basis = _strong_commitment_basis(context, actor, result, progress)
            if selective and basis is None:
                # Positive but not forced: keep the attack as a normal
                # candidate; no duty, no budget reservation.
                defer_log[actor] = {
                    'reason': 'positive_exchange_not_forced',
                    'net_hp': round(exchange['net_hp'], 2)}
                continue
            commit_log[actor] = basis or {'reason': 'forced_by_rollback_switch'}
            ranked.append((exchange['net_hp'], -result.trigger_loops,
                           actor, action, result, exchange))
        if not ranked:
            break
        _, _, actor, action, result, exchange = max(ranked, key=lambda row: row[:4])
        selected[actor] = {'action': action, 'exchange': exchange,
                           'target': context.decode(actor, action).target,
                           'position': result.position, 'basis': commit_log.get(actor)}
        positions[actor] = result.position
        # Reserve the whole potential to prevent two compulsory sources
        # banking on the same victim. Final actions still recompute effects.
        for enemy_id, amount in result.damage_potential.items():
            if amount is not None:
                remaining[enemy_id] = max(0., remaining[enemy_id] - amount)
    context._burst_trigger_assignments = selected
    context._burst_trigger_remaining = remaining
    return selected


def _strong_commitment_basis(context, actor, result, progress):
    """State-based reason a positive exchange MUST be cashed now, or None."""
    unit = context.snapshot.allies[actor]
    if danger_evidence(context, actor, horizon_h=1.)['immediate']:
        return {'reason': 'evidence_backed_imminent_death'}
    record = progress.records.get(actor) if progress is not None else None
    if record is not None and (record.get('deadline_expired')
                               or record.get('cashout_reason')
                               or record.get('last_failure')):
        return {'reason': 'approach_deadline_or_cashout_reached',
                'marker': record.get('cashout_reason') or record.get('last_failure')}
    if result.trigger_loops <= 1e-6:
        ready, arriving = cover_window(context, actor, result.target_id or '')
        if ready > 0:
            return {'reason': 'key_exchange_completing_now_with_cover',
                    'ready_cover': ready}
    return None


def cover_window(context, actor, enemy_id):
    """Ready shots and allies actually approaching this region within 2H.

    Cover is an opportunity to coordinate timing, never a discount on incoming
    damage or a promise that the enemy shoots somebody else.
    """
    cache = getattr(context, '_burst_cover_windows', None)
    if cache is None:
        cache = context._burst_cover_windows = {}
    key = (actor, enemy_id)
    if key not in cache:
        cache[key] = _cover_window(context, actor, enemy_id)
    return cache[key]


def _cover_window(context, actor, enemy_id):
    enemy = context.snapshot.enemies.get(enemy_id)
    if enemy is None:
        return 0, 0
    ready, arriving = 0, 0
    progress = getattr(context, 'burst_progress', None)
    previous = progress.last_actions if progress is not None else {}
    for peer, unit in context.snapshot.allies.items():
        cap = context.capabilities[peer]
        if peer == actor or not unit.alive or cap.healer or cap.suicide_attack:
            continue
        if context.can_issue_attack(peer, enemy_id) and context.fire_estimate(peer, enemy_id).window_shots > 0:
            ready += 1
            continue
        action = previous.get(peer)
        if action is None or context.decode(peer, action).kind not in ('move', 'attack'):
            continue
        point = context.predict_position(peer, action).position
        if _distance(point, enemy.position) + .2 >= _distance(unit.position, enemy.position):
            continue
        if context.projected_attack_opportunity(peer, enemy_id, point, horizon=2 * context.H).available:
            arriving += 1
    return ready, arriving


def _area_priority(value):
    # Preserve a cluster's marginal value while keeping future geometry below
    # the scale of an unbounded damage or kill claim.
    return min(3., value / (1. + .25 * value))


def action_region(context, actor, action):
    """One fixed regional reference for all directions in this observation.

    Region ownership is chosen before comparing actions. Opposite moves
    cannot each select a different convenient target and claim progress.
    A useful immediate native trigger may interrupt the regional commitment.
    """
    cache = getattr(context, '_burst_action_regions', None)
    if cache is None:
        cache = context._burst_action_regions = {}
    key = (actor, action)
    if key in cache:
        return cache[key]
    source = context.snapshot.allies[actor]
    effect = profile(context)
    semantic = context.decode(actor, action)
    predicted = action_position(context, actor, action)
    travel = _distance(source.position, predicted)
    incoming, unknown = _exposure(context, actor, predicted)
    discount = source.effective_health / max(source.effective_health + incoming, 1.) / (1. + unknown)
    speed = context.speed(actor)
    if effect.trigger_reach is None:
        # Command legality is independent of our calibration coverage. Keep
        # a legal native attack attempt, without an invented trigger point,
        # area damage, or numeric movement reward.
        target = context.snapshot.enemies.get(semantic.target) if semantic.kind == 'attack' else None
        result = (BurstRegion(semantic.target, source.position, 0.,
                  _distance(source.position, target.position) * 16. / speed if speed else float('inf'),
                  incoming, unknown, 0.) if target is not None and not target.is_flying else None)
        cache[key] = result
        return result
    immediate = outcome(context, actor, action)
    if immediate.opportunity > 0. and immediate.target_id in context.snapshot.enemies \
            and useful_area_value(context, immediate.position) > 0.:
        result = BurstRegion(immediate.target_id, immediate.position, 0.,
                             immediate.trigger_loops, incoming, unknown,
                             useful_area_value(context, immediate.position) * immediate.opportunity)
        cache[key] = result
        return result
    memory = getattr(context, 'burst_progress', None)
    if memory is None:
        from .burst_progress import BurstProgress
        memory = BurstProgress()  # pure snapshot fallback, no installed duty
    cashout = memory.cashout_plan(context, actor)
    fixed = cashout if cashout is not None else memory.region(context, actor)
    if fixed is not None and (fixed.get('target_id') is None or fixed.get('position') is None):
        # Following allies is an area movement duty, not an enemy burst region.
        cache[key] = None
        return None
    if fixed is None:
        # Unsupported damage (e.g. uncalibrated shield interactions) must
        # not erase a legal native attempt. It receives no numeric promise.
        target = context.snapshot.enemies.get(semantic.target) if semantic.kind == 'attack' else None
        result = (BurstRegion(semantic.target, predicted, 0., context.H,
                             incoming, unknown, 0.)
                  if target is not None and not target.is_flying else None)
        cache[key] = result
        return result
    if semantic.kind == 'attack' and semantic.target != fixed['target_id']:
        cache[key] = None
        return None
    center = fixed['position']
    before, after = _distance(source.position, center), _distance(predicted, center)
    progress = max(0., min((before - after) / max(travel, 1e-6), 1.))
    urgency = min(1., travel / max(before, travel, 1e-6))
    value = _area_priority(useful_area_value(context, center)) * progress * urgency * discount
    arrival = before * 16. / speed if speed else float('inf')
    result = BurstRegion(fixed['target_id'], center, progress * urgency * discount,
                         arrival, incoming, unknown, value)
    cache[key] = result
    return result


def action_value(context, actor, action, remaining=None, weights=None):
    """Mechanism-based fallback ordering, not a second joint score.

    The final evaluator still performs the team's real residual allocation.
    Here an uncertain source consumption competes with actual useful area;
    movement toward an unproductive region cannot win on enemy ID alone.
    """
    from .evaluator import DEFAULT_WEIGHTS
    weights = dict(DEFAULT_WEIGHTS, **(weights or {}))
    result = outcome(context, actor, action)
    region = action_region(context, actor, action)
    value = weights['useful_damage'] * useful_area_value(context, result.position, remaining) * result.opportunity
    consumption = weights['planned_loss'] * result.opportunity
    if result.opportunity <= 0. and region is not None:
        value += weights['tactic_progress'] * approach_exchange(context, actor, action, remaining)
    incoming, unknown = _exposure(context, actor, action_position(context, actor, action))
    exposure = incoming / max(context.snapshot.allies[actor].effective_health + incoming, 1.)
    value -= max(consumption, weights['risk'] * exposure * (1. + min(unknown, 1)))
    memory = getattr(context, 'burst_progress', None)
    if memory is not None:
        value -= weights['movement_continuity'] * memory.action_cost(context, actor, action)
    return value


def candidate_actions(context, actor, *, allow_wait=True, weights=None, remaining=None):
    """Small independently generated alternatives; no source pre-reservation."""
    legal = context.legal_actions(actor)
    actions = [action for action in legal if action_region(context, actor, action) is not None]
    rank = lambda action: (-action_value(context, actor, action, remaining, weights),
                           action_region(context, actor, action).arrival_loops, action)
    memory = getattr(context, 'burst_progress', None)
    duty = memory.execution(context, actor) if memory is not None else None
    if duty and duty['required']:
        required = [a for a in duty['actions'] if a in actions]
        if required:
            return sorted(required, key=rank)
    attacks = sorted((action for action in actions if context.decode(actor, action).kind == 'attack'), key=rank)
    moves = [action for action in actions if context.decode(actor, action).kind == 'move']
    advancing = sorted((action for action in moves if action_region(context, actor, action).progress > 0.
                        or outcome(context, actor, action).opportunity > 0.), key=rank)
    selected = attacks[:2] + advancing[:1]
    deferred = getattr(context, '_burst_defer_reasons', {}).get(actor)
    if deferred is not None:
        selected.extend(a for a in moves if _distance(action_position(context, actor, a),
                                                       deferred['point']) < .05)
    source = context.snapshot.allies[actor]
    origin_incoming, origin_unknown = _exposure(context, actor, source.position)
    if moves:
        # One alternate can flank or escape a real exposure, even when it
        # does not immediately close distance. It earns no false progress.
        alternate = min(moves, key=lambda action: (
            action_region(context, actor, action).incoming,
            action_region(context, actor, action).unknown, *rank(action)))
        region = action_region(context, actor, alternate)
        if region.incoming < origin_incoming or region.unknown < origin_unknown \
                or region.progress > 0.:
            selected.append(alternate)
    stop = next((action for action in actions if context.decode(actor, action).kind == 'stop'), None)
    productive_now = any(outcome(context, actor, action).opportunity > 0.
                         and useful_area_value(context, outcome(context, actor, action).position) > 0.
                         for action in selected)
    memory = getattr(context, 'burst_progress', None)
    allow_wait = allow_wait and memory is not None and memory.wait_available(context, actor)
    if allow_wait and stop is not None and not productive_now and origin_unknown == 0 \
            and origin_incoming < source.effective_health:
        region = action_region(context, actor, stop)
        ready_cover, arriving_cover = cover_window(context, actor, region.target_id)
        if ready_cover == 0 and arriving_cover > 0 and region.arrival_loops > context.H \
                and useful_area_value(context, region.position) > 0.:
            selected.append(stop)
    return sorted(set(selected), key=rank)


def continuation_state(context, actor, action, goal):
    """Action-semantic continuation across providers; never renews a TTL."""
    region = action_region(context, actor, action)
    if region is None or region.target_id != goal:
        return None
    result = outcome(context, actor, action)
    if result.opportunity > 0. and useful_area_value(context, result.position) > 0.:
        phase = 'bursting'
    elif region.progress > 1e-6:
        phase = 'approaching'
    else:
        return None
    return {'phase': phase, 'target': region.target_id, 'center': region.position}


def approach_value(context, actions, report):
    """Bounded residual-region progress from FINAL actions, once per actor.

    This is a tactical opportunity score, not damage or survival probability.
    Fixed current enemy positions exclude enemy movement as actor progress.
    The exposure discount compares one visible volley with source health;
    unknown damage cannot earn an optimistic region value.
    """
    effect = profile(context)
    if not effect.geometry_known or not effect.damage_known or effect.trigger_reach is None:
        return 0.
    from .fire_opportunities import final_action_budget
    budget = final_action_budget(context, actions)
    reserved = {}
    total = 0.
    for actor, action in sorted(actions.items()):
        cap = context.capabilities.get(actor)
        source = context.snapshot.allies.get(actor)
        if cap is None or not cap.suicide_attack or source is None or not source.alive:
            continue
        existing = report.burst_outcomes.get(actor)
        if existing is not None and existing.opportunity > 0:
            continue
        region = action_region(context, actor, action)
        if region is None:
            continue
        remaining = {enemy_id: max(0., enemy.effective_health * budget.remaining_fraction(
                                  actor, enemy_id, region.arrival_loops / max(context.H, 1))
                                  - report.burst_damage_potential.get(enemy_id, 0.)
                                  - reserved.get(enemy_id, 0.))
                     for enemy_id, enemy in context.snapshot.enemies.items()}
        value = approach_exchange(context, actor, action, remaining)
        total += value
        for enemy_id, amount in area_damage(context, region.position).items():
            if amount is not None:
                reserved[enemy_id] = reserved.get(enemy_id, 0.) + amount * min(value, 1.)
    return total


def approach_exchange(context, actor, action, remaining=None):
    """2–4H regional exchange opportunity, separate from immediate effects.

    Sample the selected route; potentially lethal early exposure limits the
    useful center to the same native death explosion, never a later arrival.
    The result remains a discounted tactical value, not future damage.
    """
    region = action_region(context, actor, action)
    if region is None or region.progress <= 0:
        return 0.
    source = context.snapshot.allies[actor]
    value = _area_priority(useful_area_value(context, region.position, remaining))
    if region.arrival_loops > 4 * context.H:
        return value * region.progress
    cache = getattr(context, '_burst_route_windows', None)
    if cache is None:
        cache = context._burst_route_windows = {}
    key = (actor, action)
    if key not in cache:
        cache[key] = _route_window(context, actor, region)
    useful_center, discount, window = cache[key]
    value = _area_priority(useful_area_value(context, useful_center, remaining))
    return value * min(region.progress * window, discount)


def _route_window(context, actor, region):
    source = context.snapshot.allies[actor]
    horizon = max(context.H, min(4 * context.H, region.arrival_loops))
    greatest_exposure, unknown = 0., 0
    useful_center = region.position
    for fraction in (0., .5, 1.):
        point = tuple(source.position[i] + (region.position[i] - source.position[i]) * fraction
                      for i in (0, 1))
        incoming, uncertain = _exposure(context, actor, point, horizon, physical=True)
        greatest_exposure, unknown = max(greatest_exposure, incoming), max(unknown, uncertain)
    death = supported_death_point(context, actor, region.position, horizon)
    if death is not None:
        useful_center = death[0]
    cover, _ = cover_window(context, actor, region.target_id)
    # Immediate cover makes the window urgent, but never reduces exposure.
    window = 1. + min(cover, 2) * .15
    discount = source.effective_health / max(source.effective_health + greatest_exposure, 1.) / (1. + unknown)
    # region.progress already includes its endpoint exposure discount; use
    # the smaller route result, rather than charging the same risk twice.
    return useful_center, discount, window
