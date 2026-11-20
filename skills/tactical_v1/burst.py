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
    # Unified space-time facts (repair 5): everything in this outcome is
    # evaluated at ONE moment, evaluation_after_loops from now. enemy_positions
    # holds the position each visible enemy was assumed to hold AT that
    # moment (motion prediction where reliable, explicitly static otherwise);
    # motion_used marks whether any prediction actually applied.
    evaluation_after_loops: float = 0.
    enemy_positions: Optional[Dict[str, Tuple[float, float]]] = None
    motion_used: bool = False
    # Enemies whose position in enemy_positions is an explicitly STATIC
    # fallback (no reliable motion record), not a claimed future position.
    static_fallback_enemies: tuple = ()


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


def affects(profile, center, target, position=None) -> bool:
    """Native effect geometry; an unknown radius only establishes body contact.

    ``position`` optionally overrides the target's location so coverage can
    be judged at ONE shared evaluation moment instead of mixing current and
    future positions. The fallback has finite scope and is explicitly
    uncalibrated: it cannot create numeric damage or make every visible
    Baneling an imminent threat.
    """
    if target.is_flying:
        return False
    point = target.position if position is None else position
    radius = profile.blast_radius if profile.geometry_known else 0.
    return _distance(center, point) <= radius + target.radius


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


def _predicted_enemy_positions(context, at_loop):
    """Enemy positions at ONE evaluation moment, with a static fallback.

    ``at_loop`` is an ABSOLUTE loop count; all callers use one convention.
    Returns (map, motion_used, fallback_ids): reliable ≤1H motion
    predictions where they exist, otherwise the CURRENT (static) position —
    never a mixture of different moments. ``fallback_ids`` names the enemies
    whose entry is an explicitly static fallback, so consumers can tell
    predicted from conservative positions without trusting the whole map.
    Pure query.
    """
    from .enemy_motion import predicted_position
    positions, motion_used, fallback = {}, False, []
    for enemy_id, enemy in context.snapshot.enemies.items():
        future = predicted_position(context, enemy_id, at_loop)
        if future is not None:
            positions[enemy_id] = future
            motion_used = True
        else:
            positions[enemy_id] = enemy.position
            fallback.append(enemy_id)
    return positions, motion_used, tuple(fallback)


def _contact_point(context, actor, enemy_id, enemy_position):
    """Trigger contact point toward the enemy AT ``enemy_position``.

    P(T) = S + (E(T)-S) * travel/d with R = radii + trigger reach;
    d = 0 keeps the source center (no division by zero). Returns
    (point, travel): travel is the straight-line distance the bomber must
    still cover by T — the caller compares it with its own speed budget.
    """
    source = context.snapshot.allies[actor]
    enemy = context.snapshot.enemies[enemy_id]
    effect = profile(context)
    reach = source.radius + enemy.radius + (effect.trigger_reach or 0.)
    dx, dy = enemy_position[0] - source.position[0], enemy_position[1] - source.position[1]
    d = math.hypot(dx, dy)
    if d <= 1e-9:
        return tuple(source.position), 0.
    travel = max(0., d - reach)
    fraction = travel / d
    return (source.position[0] + dx * fraction,
            source.position[1] + dy * fraction), travel


def area_damage(context, center, enemy_positions=None):
    """Coverage at ONE moment: every enemy uses the supplied position map.

    Omitted map = pure static T=0 query (current positions); callers inside
    a prediction chain must pass the same-moment map rather than mixing.
    """
    effect = profile(context)
    positions = enemy_positions or {}
    return {enemy_id: damage(context, enemy_id)
            for enemy_id, enemy in context.snapshot.enemies.items()
            if affects(effect, center, enemy, positions.get(enemy_id))}


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


def action_position_at(context, actor, action, after_loops):
    """Predicted ally position at ONE moment under its FINAL action.

    after_loops is RELATIVE to the current game_loop: 0 returns the current
    OBSERVED position; 0 < T <= H advances the unit along its action's
    semantics limited by its native speed (speed*T/16); out-of-range or
    non-finite T degrades to the conservatively-certain current position,
    never silently to the full-window endpoint. No mutation, no coordinator
    calls, no game commands; cache keyed by snapshot/actor/action/T.
    Returns (position, uncertain): uncertain=True marks model estimates
    (missing speed/path data or bomber trigger geometry), never an
    observed fact.
    """
    H = float(context.H)
    try:
        T = float(after_loops)
    except (TypeError, ValueError):
        return tuple(context.snapshot.allies[actor].position), True
    if not math.isfinite(T) or T < 0.:
        return tuple(context.snapshot.allies[actor].position), True
    T = min(T, H)
    cache = getattr(context, "_action_position_at_cache", None)
    if cache is None:
        cache = context._action_position_at_cache = {}
    key = (context.snapshot.snapshot_id, actor, action, round(T, 6))
    if key in cache:
        return cache[key]
    unit = context.snapshot.allies[actor]
    current = tuple(unit.position)
    if T <= 1e-9:
        cache[key] = (current, False)
        return cache[key]
    speed = context.speed(actor)
    semantic = context.decode(actor, action)
    if semantic.kind == 'move':
        info = context.snapshot.movement_candidates.get(actor, {}).get(action)
        destination = info.get('destination') if info else None
        if destination is None or not speed:
            cache[key] = (current, True)       # unknown speed/path data
            return cache[key]
        destination = (float(destination[0]), float(destination[1]))
        gap = _distance(current, destination)
        max_travel = speed * T / 16.
        if gap <= max_travel:
            point = destination                # arrived by T (model estimate)
        else:
            fraction = max_travel / gap if gap > 1e-9 else 0.
            point = (current[0] + (destination[0] - current[0]) * fraction,
                     current[1] + (destination[1] - current[1]) * fraction)
        cache[key] = (point, True)
        return cache[key]
    if semantic.kind == 'attack' and semantic.target in context.snapshot.enemies:
        cap = context.capabilities.get(actor)
        if cap is not None and cap.suicide_attack:
            # Bomber: reuse the outcome's already-selected route/position;
            # no recursion into effects or outcome building.
            result = outcome(context, actor, action)
            if result.opportunity > 0. and result.trigger \
                    != 'active_contact_opportunity' \
                    and T >= result.trigger_loops - 1e-6:
                cache[key] = (tuple(result.position), True)
                return cache[key]
            point_t, travel = _contact_point(context, actor, semantic.target,
                                             context.snapshot.enemies[
                                                 semantic.target].position)
            budget_t = speed * T / 16. if speed else 0.
            if travel <= budget_t + 1e-6:
                cache[key] = (tuple(point_t), True)
                return cache[key]
            fraction = budget_t / travel if travel > 1e-9 else 0.
            point = (current[0] + (point_t[0] - current[0]) * fraction,
                     current[1] + (point_t[1] - current[1]) * fraction)
            cache[key] = (point, True)
            return cache[key]
        contact = context.contact_status(actor, semantic.target)
        if contact == "in_range":
            cache[key] = (current, False)      # keeps existing contact model
            return cache[key]
        enemy = context.snapshot.enemies[semantic.target]
        reach = context.effective_attack_reach(actor, semantic.target)
        if reach is None or reach < 0 or not speed:
            cache[key] = (current, True)
            return cache[key]
        gap = max(_distance(current, enemy.position) - reach, 0.)
        max_travel = speed * T / 16.
        advance = min(gap, max_travel)
        direction = (enemy.position[0] - current[0], enemy.position[1] - current[1])
        norm = math.hypot(*direction)
        if norm <= 1e-9:
            cache[key] = (current, True)
            return cache[key]
        point = (current[0] + direction[0] / norm * advance,
                 current[1] + direction[1] / norm * advance)
        cache[key] = (point, True)
        return cache[key]
    # STOP and other semantics: the existing model keeps them in place.
    cache[key] = (current, False)
    return cache[key]


def supported_death_point(context, actor, destination, horizon=None):
    """First SAMPLED lethal-opportunity point backed by danger evidence.

    Samples the straight route current position -> ``destination`` at
    fractions 0/.5/1 and returns (point, fraction, facts) for the first
    position whose danger evidence is immediate AND supported lethal.
    ``fraction * horizon`` is the death OPPORTUNITY's sampled moment — a
    model estimate, never an observed death time. The all-enemy volley
    stays a scoring bound in ``_exposure`` and never proves an earlier
    death by itself. Pure query, no allocation calls.
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
    """ONE mutually exclusive result: contact OR death, never both.

    Selection order (task 1): build the model-feasible CONTACT candidate
    first (earliest feasible trigger moment, contact point recomputed from
    scratch per moment), but do NOT cash in its coverage yet. When the
    contact needs travel (contact_time > 0), check the SAME NEW route
    (current position -> contact position) for an EVIDENCE-BACKED earlier
    death via supported_death_point; if found, the death hypothesis wins
    and the distant contact coverage is dropped. Selection rules:
      contact_time == 0            -> immediate contact (no invented risk
                                      ordering against a later-window death);
      contact_time > 0, earlier
      supported death found        -> death_at/path_opportunity at the
                                      SAMPLED death point and its sampled
                                      moment (uncertain, not a known death
                                      time);
      contact candidate, no death  -> active_contact_opportunity with the
                                      new-route risk discount;
      no contact candidate         -> death on the actual action path only.
    One unified exit builds a single BurstOutcome; damage is evaluated once
    at the selected position/moment. health/(health+incoming) stays an
    OPPORTUNITY discount only — it never substitutes for relocating the
    burst point to an earlier death.
    """
    source = context.snapshot.allies[actor]
    effect = profile(context)
    semantic = context.decode(actor, action)
    predicted = action_position(context, actor, action)
    if semantic.kind != 'attack' and position is not None:
        predicted = tuple(position)
    target_id = semantic.target if semantic.kind == 'attack' else None
    speed = context.speed(actor)
    Hf = float(context.H)

    # ---- 1. contact candidate (model-feasible earliest trigger moment) ----
    contact = None        # (time, position, enemy_map, motion_used, fallbacks)
    if semantic.kind == "attack" and semantic.target in context.snapshot.enemies \
            and context.can_issue_attack(actor, semantic.target):
        target_enemy = context.snapshot.enemies[semantic.target]
        if not target_enemy.is_flying:
            static_center, static_ok = trigger_position(context, actor,
                                                        semantic.target)
            candidates_T = [0., Hf * .25, Hf * .5, Hf * .75, Hf]
            if static_ok and speed:
                static_arrival = (_distance(source.position, static_center)
                                  * 16. / speed)
                if 0. <= static_arrival <= Hf:
                    candidates_T.append(static_arrival)
            for moment in sorted(set(round(t, 6) for t in candidates_T)):
                if moment > Hf + 1e-6:
                    continue
                enemy_map_t, motion_t, fallback_t = _predicted_enemy_positions(
                    context, context.snapshot.game_loop + moment)
                future = enemy_map_t.get(semantic.target)
                if future is None:
                    continue
                point_t, travel = _contact_point(context, actor,
                                                 semantic.target, future)
                budget_t = speed * moment / 16. if speed else 0.
                if travel <= budget_t + 1e-6:
                    contact = (moment, point_t, enemy_map_t, motion_t, fallback_t)
                    break

    # ---- 2. evidence-backed earlier death along the actual route ----
    death = None
    if contact is not None and contact[0] > 1e-6:
        # New route only: current position -> NEW contact position.
        death = supported_death_point(context, actor, contact[1],
                                      horizon=max(contact[0], 1.))
    elif contact is None:
        # No contact candidate: death on the action's own path.
        death = supported_death_point(context, actor, predicted, horizon=None)

    # ---- 3. mutually exclusive selection ----
    if contact is not None and contact[0] <= 1e-6:
        selected_trigger = 'active_contact_opportunity'
        selected_position = contact[1]
        selected_time = 0.
        selected_map = contact[2]
        motion_used = contact[3]
        fallbacks = contact[4]
        weight = 1.
    elif death is not None:
        selected_position, fraction, facts = death
        selected_trigger = ('death_at_origin_opportunity' if fraction == 0.
                            else 'death_on_path_opportunity')
        # fraction*horizon is the SAMPLED death-opportunity moment — a
        # model estimate, never an observed death time.
        selected_time = max(0., min(float(context.H), contact[0] if contact
                                    else float(context.H)) * fraction) \
            if contact is not None else float(context.H) * fraction
        selected_map, motion_used, fallbacks = _predicted_enemy_positions(
            context, context.snapshot.game_loop + selected_time)
        supported = facts['supported_amount']
        weight = supported / (supported + max(source.effective_health, 1.))
    elif contact is not None:
        moment, point_t, enemy_map_t, motion_t, fallback_t = contact
        selected_trigger = 'active_contact_opportunity'
        selected_position = point_t
        selected_time = moment
        selected_map = enemy_map_t
        motion_used = motion_t
        fallbacks = fallback_t
        maximum_incoming, maximum_unknown = 0., 0
        for fraction in (0., .5, 1.):
            sample = tuple(source.position[i] + (point_t[i] - source.position[i]) * fraction
                           for i in (0, 1))
            incoming, unknown = _exposure(context, actor, sample,
                                          max(moment, 1.), physical=True)
            maximum_incoming = max(maximum_incoming, incoming)
            maximum_unknown = max(maximum_unknown, unknown)
        weight = source.effective_health / max(source.effective_health + maximum_incoming, 1.)
        weight /= 1. + maximum_unknown
    else:
        selected_trigger, selected_position = 'none', predicted
        selected_time, selected_map = 0., {}
        motion_used, fallbacks, weight = False, (), 0.
        target_id = None

    # ---- 4. single unified exit: one damage evaluation ----
    evaluation_after = selected_time if weight > 0. else 0.
    if selected_trigger == 'active_contact_opportunity':
        # contact_time semantics: 0 means already in contact NOW.
        trigger_loops = selected_time
    else:
        trigger_loops = selected_time
    potential = area_damage(context, selected_position, selected_map) \
        if weight > 0. and selected_trigger != 'none' else {}
    if potential and selected_trigger.startswith('death_'):
        target_id = max(potential, key=lambda enemy_id: (
            (potential[enemy_id] or 0.) / max(context.snapshot.enemies[enemy_id].effective_health, 1.),
            -_distance(selected_position, selected_map.get(
                enemy_id, context.snapshot.enemies[enemy_id].position)), enemy_id))
    if selected_trigger == 'none':
        # No trigger opportunity: leave NO position/damage metadata behind.
        selected_map, motion_used, fallbacks = {}, False, ()
    return BurstOutcome(actor, selected_trigger, selected_position, weight,
                        potential, effect.geometry_known, effect.evidence,
                        target_id, trigger_loops,
                        selected_trigger != 'active_contact_opportunity'
                        or trigger_loops > 1e-6,
                        evaluation_after_loops=evaluation_after,
                        enemy_positions=dict(selected_map) if potential else None,
                        motion_used=bool(motion_used) and weight > 0.,
                        static_fallback_enemies=tuple(fallbacks)
                        if potential else ())


def useful_area_value(context, center, remaining=None, enemy_positions=None):
    """Residual capped damage fractions at ONE evaluation moment."""
    result = 0.
    for enemy_id, amount in area_damage(context, center, enemy_positions).items():
        enemy = context.snapshot.enemies[enemy_id]
        health = enemy.effective_health if remaining is None else remaining.get(enemy_id, 0.)
        if amount is not None and health > 0:
            result += min(amount, health) / max(enemy.effective_health, 1.)
    return result


def exchange_value(context, actor, center, remaining=None, *, opportunity=1.,
                   consumed=(), positions=None, enemy_positions=None):
    """Conservative HP exchange gate for a compulsory one-use commitment.

    ``enemy_positions`` carries the SAME-moment enemy position map used by
    the outcome being evaluated; without it this is a pure static T=0
    query (explicitly not part of a prediction chain). Collateral from
    enemy death bursts is also measured at that shared moment. This is a
    gate, not a replacement for final joint scoring: marginal or unknown
    exchanges remain legal soft alternatives. Enemy death effects are
    conditional costs, never observed damage.
    """
    source = context.snapshot.allies[actor]
    effect = profile(context)
    potential = area_damage(context, center, enemy_positions)
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
    shared = enemy_positions or {}
    for enemy_id in death_sources:
        enemy = context.snapshot.enemies[enemy_id]
        for peer, unit in context.snapshot.allies.items():
            if peer == actor or peer in consumed or not unit.alive or unit.is_flying:
                continue
            point = positions.get(peer, unit.position)
            radius = effect.blast_radius_upper if effect.blast_radius_upper is not None else 0.
            if _distance(point, shared.get(enemy_id, enemy.position)) > radius + unit.radius:
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


def reserve_actions(context, actor, actions, valid, reservations, residual_for):
    """Compare a second burst with a concrete screen or alternate contact."""
    from .combat_exchange import danger_evidence
    if not reservations or not valid:
        return valid
    redundant = set()
    claimed = {e for _, _, hits in reservations for e, hp in hits.items() if hp > 0.}
    if not claimed:
        return valid
    hydras = {peer: context.predict_position(peer, actions[peer]).position
              for peer, unit in context.snapshot.allies.items()
              if unit.alive and unit.type_name == 'hydralisk' and peer in actions}
    # A surviving melee unit already in reach of a hydra has breached the
    # screen. An observed attacker or one-shot threat also justifies a finish.
    breakers = set()
    for peer, point in hydras.items():
        risk = context.forecast_risk(peer, point)
        for enemy in set(risk.current_threats) - set(risk.heuristic_enemies):
            cap = context.enemy_capabilities[enemy]
            amount = risk.damage_estimates.get(enemy)
            if (cap.combat_class == 'melee' or cap.suicide_attack
                    or amount is not None and amount >= context.snapshot.allies[peer].effective_health
                    or enemy in danger_evidence(context, peer, point)['targeted_sources']):
                breakers.add(enemy)
    for action in context.legal_actions(actor):
        result = outcome(context, actor, action)
        if result.trigger != 'active_contact_opportunity':
            continue
        overlap = any(t <= result.trigger_loops and any(
            hp > 0. and e in result.damage_potential for e, hp in hits.items())
            for t, _, hits in reservations)
        if not overlap:
            continue
        remaining = residual_for(result)
        exchange = exchange_value(context, actor, result.position, remaining,
            opportunity=result.opportunity, enemy_positions=result.enemy_positions,
            consumed={source for t, source, _ in reservations if t <= result.trigger_loops})
        # A second blast may pay for itself or finish the surviving group.
        finishes = sum(1 for e, hp in result.damage_potential.items()
                       if hp is not None and 0. < remaining[e] <= hp * result.opportunity)
        urgent_finish = any(e in breakers
                            and hp is not None and 0. < remaining[e] <= hp * result.opportunity
                            for e, hp in result.damage_potential.items())
        if (exchange['known'] and not exchange['worthwhile']
                and finishes < 2 and not urgent_finish):
            redundant.add(action)
    if not redundant:
        return valid
    source = context.snapshot.allies[actor]
    alternatives = []
    screen_moves = [a for a in valid if context.decode(actor, a).kind in ('move', 'stop')
                    and outcome(context, actor, a).opportunity == 0.]
    for action in valid:
        semantic = context.decode(actor, action)
        if action not in redundant and (outcome(context, actor, action).opportunity > 0.
                or semantic.kind == 'attack' and semantic.target not in claimed):
            alternatives.append(action)
    others = [e for e, unit in context.snapshot.enemies.items()
              if not unit.is_flying and e not in claimed]
    if others and screen_moves:
        target = min(others, key=lambda e: _distance(source.position, context.snapshot.enemies[e].position))
        point, reachable = trigger_position(context, actor, target, 4 * max(context.H, 1))
        if reachable:
            advance = min(screen_moves, key=lambda a: _distance(action_position(context, actor, a), point))
            if _distance(action_position(context, actor, advance), point) < _distance(source.position, point) - .05:
                alternatives.append(advance)
    # Screen the exposed side of a living hydra at its commanded position.
    # The role lasts only for this joint branch, while a first burst is due.
    ground = [u.position for u in context.snapshot.enemies.values() if not u.is_flying]
    if screen_moves and hydras and ground:
        anchor = min(hydras.values(), key=lambda p: _distance(source.position, p))
        threat = min(ground, key=lambda p: _distance(anchor, p))
        gap = _distance(anchor, threat)
        # One body width ahead of the anchor keeps the bomber close enough
        # to intercept without sending it down the first bomber's route.
        offset = min(gap, 2. * source.radius)
        point = tuple(a + (b - a) * offset / max(gap, 1e-6) for a, b in zip(anchor, threat))
        safe_screen = []
        for action in screen_moves:
            end = action_position(context, actor, action)
            risk = danger_evidence(context, actor, end)
            if (not risk['immediate'] and not risk['effect_danger']
                    and _distance(end, point) <= _distance(source.position, point) + .05):
                safe_screen.append((_distance(end, point), action))
        if safe_screen:
            alternatives.append(min(safe_screen)[1])
    return tuple(dict.fromkeys(alternatives)) or valid


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
    # Alternatives reachable within one window are compared at the SAME
    # future moment (T=H) as their own evaluation, never against a mixture
    # of current and predicted enemy positions.
    horizon_map, _, _ = _predicted_enemy_positions(
        context, context.snapshot.game_loop + max(context.H, 1))
    for move in sorted(context.move_destinations(actor)):
        point = context.predict_position(actor, move).position
        # Trigger anchor measured against the SAME-TIME enemy positions, not
        # against current positions while coverage uses the future map.
        anchored = any(not enemy.is_flying
                       and _distance(point, horizon_map[enemy_id])
                       <= source.radius + enemy.radius + effect.trigger_reach
                       for enemy_id, enemy in context.snapshot.enemies.items())
        if not anchored or _distance(source.position, point) > budget + .1:
            continue
        # A MOVE arriving at H only CREATES a later trigger opportunity; the
        # deferred value keeps the cost of needing that follow-up action and
        # its uncertainty. This is not a same-window detonation.
        alternative = exchange_value(context, actor, point, remaining,
                                     enemy_positions=horizon_map)
        alternative_net = alternative['net_hp'] * 0.5   # follow-up-action discount
        if not alternative['known'] or alternative_net <= best_net \
                or alternative_net <= threshold:
            continue
        midpoint = tuple((source.position[i] + point[i]) / 2. for i in (0, 1))
        incoming, unknown = _exposure(context, actor, midpoint, context.H, physical=True)
        end_incoming, end_unknown = _exposure(context, actor, point, context.H, physical=True)
        if max(incoming, end_incoming) >= source.effective_health or unknown or end_unknown:
            continue
        best, best_net = point, alternative_net
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
                opportunity=result.opportunity, consumed=selected, positions=positions,
                enemy_positions=result.enemy_positions)
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
    """State-based reason a positive exchange MUST be cashed now, or None.

    Tightened criteria (no label-level shortcuts):
      * imminent death uses danger_evidence's IMMEDIATE flag AND a
        supported incoming amount that plausibly removes the source's
        remaining effective health — any damage or any threat alone is
        not "dying now";
      * deadline/cashout markers only: a plain last_failure (blocked
        route, detour) never forces an explosion;
      * a completing key exchange needs an immediate trigger (trigger_loops
        ~ 0) AND more than one ready covering shot AND the covered value
        to actually be team-level (multi-enemy coverage) — one nearby
        shooting opportunity alone proves nothing.
    Returns None when no basis holds; the attack then stays a normal
    positive candidate and is never forced.
    """
    from .combat_exchange import danger_evidence
    unit = context.snapshot.allies[actor]
    evidence = danger_evidence(context, actor, horizon_h=1.)
    if evidence['immediate'] and not evidence['unknown'] \
            and evidence['supported_amount'] >= .5 * max(unit.effective_health, 1.):
        return {'reason': 'evidence_backed_imminent_death',
                'supported_amount': round(evidence['supported_amount'], 2),
                'effective_health': round(unit.effective_health, 2)}
    record = progress.records.get(actor) if progress is not None else None
    if record is not None and (record.get('deadline_expired')
                               or record.get('cashout_reason')):
        return {'reason': 'approach_deadline_or_cashout_reached',
                'marker': record.get('cashout_reason') or 'deadline_expired'}
    if result.trigger_loops <= 1e-6:
        ready, arriving = cover_window(context, actor, result.target_id or '')
        covered = sum(1 for amount in result.damage_potential.values()
                      if amount and amount > 0.)
        if ready > 1 and covered > 1:
            return {'reason': 'key_exchange_completing_now_with_cover',
                    'ready_cover': ready, 'covered_enemies': covered}
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
    if deferred is not None and 'point' in deferred:
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
