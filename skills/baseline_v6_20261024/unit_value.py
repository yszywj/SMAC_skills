"""Pure, local tactical-importance estimates over a team-visible snapshot.

The design follows UnitValue's base-weight/health decomposition, but the
weights below are *manual engineering priors*, not copied Sharpy data,
official game values, calibrated win probabilities, DPS or causal effects.
No actions, environment objects, hidden enemies, clocks or caches are used.

Native IDs were checked against the installed ``pysc2.lib.units`` enum.
The explicit _RL aliases were checked against ``LeaderAlias`` entries in
the bundled SMAC_HARD_maps' Base.SC2Data\\GameData\\UnitData.xml. Custom IDs
are never inferred from their numeric range or from string prefixes.
"""

import math
from numbers import Real


EVALUATION_VERSION = "unit-value-v2"
DEFAULT_PARAMETERS = {
    "mode": "off",
    "spatial_context": False,
    "w_matchup": 0.2,
    "w_support": 0.2,
    "w_scarcity": 0.2,
    "scale": 2.0,
    "focus_value_weight": 0.2,
    "focus_switch_margin": 0.05,
    "retreat_value_margin": 0.05,
    "retreat_threshold_upper_bound": 0.6,
    "heal_value_weight": 0.1,
}

# Dimensionless, manually chosen initial weights: Marine = 1.0. The table
# covers public roster types in every registered map and relevant tests.
# It does not assign value to unsupported economic or spell-casting roles.
BASE_VALUES = {
    "Marine": 1.0, "Marauder": 2.0, "Medivac": 2.0,
    "Reaper": 1.5, "Ghost": 2.5, "Hellion": 1.5,
    "SiegeTank": 3.0, "SiegeTankSieged": 3.0,
    "Battlecruiser": 6.0, "Liberator": 3.0, "VikingFighter": 2.5,
    "Zealot": 2.0, "Stalker": 2.5, "Colossus": 5.0,
    "Sentry": 2.0, "HighTemplar": 2.5, "Immortal": 4.0,
    "Disruptor": 4.0, "Phoenix": 2.5, "VoidRay": 4.0,
    "Zergling": 0.5, "Baneling": 1.0, "Hydralisk": 2.0,
    "Queen": 2.5, "SpineCrawler": 2.0, "SporeCrawler": 2.0,
}
NATIVE_TYPES = {
    48: "Marine", 51: "Marauder", 54: "Medivac", 49: "Reaper",
    50: "Ghost", 53: "Hellion", 33: "SiegeTank", 32: "SiegeTankSieged",
    57: "Battlecruiser", 689: "Liberator", 35: "VikingFighter",
    73: "Zealot", 74: "Stalker", 4: "Colossus", 77: "Sentry",
    75: "HighTemplar", 83: "Immortal", 694: "Disruptor", 78: "Phoenix",
    80: "VoidRay", 105: "Zergling", 9: "Baneling", 107: "Hydralisk",
    126: "Queen", 98: "SpineCrawler", 99: "SporeCrawler",
}
TYPE_ALIASES = {name.lower(): name for name in BASE_VALUES}
TYPE_ALIASES.update({(name + "_RL").lower(): name for name in (
    "Marine", "Stalker", "Zealot", "Zergling", "Baneling", "Colossus",
    "Hydralisk", "VoidRay")})
BIOLOGICAL_TYPES = frozenset({
    "Marine", "Marauder", "Reaper", "Ghost", "Zealot", "HighTemplar",
    "Zergling", "Baneling", "Hydralisk", "Queen", "SpineCrawler", "SporeCrawler",
})
STRUCTURE_TYPES = frozenset({"SpineCrawler", "SporeCrawler"})
SUPPORT_DISTANCE = 6.0  # Local proximity estimate, not a claim of native healing range.
SUPPORT_DEMAND_SCALE = 2.0  # Two health bars of eligible local missing health saturate S.


def _number(value):
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    try:
        value = float(value)
    except (OverflowError, ValueError):
        return None
    return value if math.isfinite(value) else None


def validate_parameters(parameters=None):
    """Return a fresh full config; reject unknown, nonfinite and unsafe values."""
    supplied = dict(parameters or {})
    unexpected = set(supplied) - set(DEFAULT_PARAMETERS)
    if unexpected:
        raise ValueError("Unknown unit-value parameters: " + ", ".join(sorted(unexpected)))
    result = {**DEFAULT_PARAMETERS, **supplied}
    if result["mode"] not in ("off", "basic", "contextual"):
        raise ValueError("unit-value mode must be off, basic, or contextual")
    if not isinstance(result["spatial_context"], bool):
        raise ValueError("unit-value spatial_context must be boolean")
    bounds = {
        "w_matchup": (0, 1), "w_support": (0, 1), "w_scarcity": (0, 1),
        "focus_value_weight": (0, 0.5), "focus_switch_margin": (0, 1),
        "retreat_value_margin": (0, 0.2), "retreat_threshold_upper_bound": (0, 1),
        "heal_value_weight": (0, 0.5),
    }
    for key in set(DEFAULT_PARAMETERS) - {"mode", "spatial_context"}:
        value = _number(result[key])
        if value is None:
            raise ValueError(f"unit-value {key} must be finite")
        if key == "scale":
            valid = value > 0
        else:
            low, high = bounds[key]
            valid = low <= value <= high
            if key == "retreat_threshold_upper_bound":
                valid = valid and value > 0
        if not valid:
            raise ValueError(f"unit-value {key} is outside its permitted range")
        result[key] = value
    return result


def enabled_values(unit_values):
    return bool(unit_values and unit_values.get("mode") in ("basic", "contextual"))


def normalized_value(unit_values, side, unit_id):
    """Safe allocator lookup; missing/off/malformed sidecar values are neutral."""
    if not enabled_values(unit_values):
        return 0.0
    row = unit_values.get(side, {}).get(unit_id, {})
    field = "preservation_value" if side == "allies" else "target_priority_value"
    value = _number(row.get(field, row.get("normalized_importance")))
    return min(1.0, max(0.0, value)) if value is not None else 0.0


def _identity(unit):
    raw = unit.get("unit_type_id")
    if isinstance(raw, (int, str)) and not isinstance(raw, bool):
        try:
            native = NATIVE_TYPES.get(int(raw))
        except (ValueError, OverflowError):
            native = None
        if native:
            return native, "verified_native_type_id"
    name = TYPE_ALIASES.get(str(unit.get("type", "")).lower())
    return (name, "verified_exact_type_name") if name else (None, "unknown_type_neutral_fallback")


def _dead(unit):
    health = _number(unit.get("health"))
    return unit.get("alive") is False or (health is not None and health <= 0)


def _component(value, basis, availability=None, **details):
    return {"value": value, "applied_value": value if value is not None else 0.0,
            "availability": availability or ("known" if value is not None else "unknown"),
            "basis": basis, **details}


def _base_row(unit):
    name, source = _identity(unit)
    base = BASE_VALUES.get(name, 1.0)
    unknown = [] if name else ["unit_type"]
    reasons = [source, "base_weight_is_manual_prior"]
    health, maximum = _number(unit.get("health")), _number(unit.get("max_health"))
    shield, max_shield = _number(unit.get("shield")), _number(unit.get("max_shield"))
    for field, value in (("health", health), ("max_health", maximum),
                         ("shield", shield), ("max_shield", max_shield)):
        if value is None or value < 0 or (field == "max_health" and value <= 0):
            unknown.append(field)
    if _dead(unit):
        factor = 0.0
        reasons.append("confirmed_dead_zero_value")
    elif not unknown or set(unknown) == {"unit_type"}:
        # Scale first so even very large finite input values cannot overflow.
        divisor = max(health, shield, maximum, max_shield, 1.0)
        numerator = health / divisor + shield / divisor
        denominator = maximum / divisor + max_shield / divisor
        # A tiny positive maximum may underflow after scaling by enormous
        # observed health. It is already an over-full state, whose clamp is 1.
        ratio = 1.0 if numerator >= denominator else numerator / denominator
        factor = 0.5 + 0.5 * ratio
        reasons.append("health_and_shield_fraction")
    else:
        factor = 1.0
        reasons.append("invalid_or_missing_health_data_neutral_factor_1")
    return {"base_value": base, "health_factor": factor,
            "combat_value": base * factor, "importance": base * factor,
            "normalized_importance": 0.0, "components": {},
            "confidence": "partial" if unknown else "complete",
            "reasons": reasons, "unknown_fields": unknown}


def _capabilities(state, unit):
    name, _ = _identity(unit)
    if name == "Medivac":
        return frozenset(), "confirmed_medivac_no_attack_weapon"
    profile = state.get("unit_type_profiles", {}).get(str(unit.get("unit_type_id")), {})
    weapons = profile.get("weapons")
    if profile.get("known") is not True or not isinstance(weapons, list) or not weapons:
        return None, "missing_or_empty_weapon_profile_is_not_proof_of_no_weapon"
    targets = set()
    for weapon in weapons:
        target = weapon.get("target_type") if isinstance(weapon, dict) else None
        if target not in ("ground", "air", "any"):
            return None, "weapon_target_type_unknown"
        targets.update(("air", "ground") if target == "any" else (target,))
    return frozenset(targets), "observed_weapon_target_types"


def _target_classes(unit):
    flying = unit.get("is_flying")
    name, _ = _identity(unit)
    if name == "Colossus":
        return {"ground", "air"}
    return {"air" if flying else "ground"} if isinstance(flying, bool) else None


def _matchup(capability, opponents):
    if not opponents:
        return _component(0.0, "no_current_visible_opponents")
    if capability is None:
        return _component(None, "weapon_target_types_unknown")
    if not capability:
        return _component(0.0, "confirmed_noncombat_unit")
    targets = [_target_classes(unit) for unit in opponents.values()]
    if any(target is None for target in targets):
        return _component(None, "opponent_flight_status_unknown")
    return _component(sum(bool(capability & target) for target in targets) / len(targets),
                      "fraction_of_current_visible_opponents_with_compatible_target_type",
                      compatible_targets=sum(bool(capability & target) for target in targets),
                      visible_targets=len(targets))


def _missing_fraction(unit):
    health, maximum = _number(unit.get("health")), _number(unit.get("max_health"))
    if health is None or maximum is None or maximum <= 0 or health < 0:
        return None
    return max(0.0, min(1.0, 1.0 - health / maximum))


def _biological_target(unit):
    name, _ = _identity(unit)
    flying = unit.get("is_flying")
    if name in STRUCTURE_TYPES or flying is True:
        return False
    if name is None or not isinstance(flying, bool):
        return None
    return name in BIOLOGICAL_TYPES


def _nearby(first, second):
    left, right = first.get("position"), second.get("position")
    if not isinstance(left, (list, tuple)) or not isinstance(right, (list, tuple)):
        return None
    if len(left) < 2 or len(right) < 2:
        return None
    coords = [_number(x) for x in (*left[:2], *right[:2])]
    if any(x is None for x in coords):
        return None
    return math.hypot(coords[0] - coords[2], coords[1] - coords[3]) <= SUPPORT_DISTANCE


def _support(state, side, actor, unit, roster):
    name, _ = _identity(unit)
    if name != "Medivac":
        return _component(0.0, "no_confirmed_supported_medical_ability") if name else _component(
            None, "unit_support_capabilities_unknown")
    energy = _number(unit.get("energy"))
    if energy is None or energy < 0:
        return _component(None, "observed_healer_energy_unknown")
    if energy == 0:
        return _component(0.0, "observed_zero_energy_no_current_healing_bonus")
    eligible, unknown = [], []
    if side == "allies":
        candidates = state.get("heal_candidates", {}).get(actor)
        if not isinstance(candidates, dict):
            return _component(None, "legal_heal_candidates_unknown")
        # Legal candidates are authoritative for target suitability. Extra
        # checks reject full health, shield-only loss, dead and known machines.
        for candidate in candidates.values():
            target_id = candidate.get("target") if isinstance(candidate, dict) else None
            target = roster.get(target_id)
            if target is None or target_id == actor or _dead(target):
                continue
            name_target, _ = _identity(target)
            if target.get("is_flying") is True or (name_target and _biological_target(target) is False):
                continue
            missing = _missing_fraction(target)
            if missing is None:
                unknown.append(target_id)
            elif missing > 0 and target_id not in eligible:
                eligible.append(target_id)
        basis = "current_legal_health_only_heal_candidates"
    else:
        # Enemy candidates are never read from a hidden-state source. This is
        # only a nearby visible biological-health-demand estimate.
        for target_id, target in roster.items():
            if target_id == actor or _dead(target):
                continue
            eligible_type, near = _biological_target(target), _nearby(unit, target)
            missing = _missing_fraction(target)
            if eligible_type is False or near is False or missing == 0:
                continue
            if eligible_type is None or near is None or missing is None:
                unknown.append(target_id)
            else:
                eligible.append(target_id)
        basis = "visible_nearby_confirmed_biological_health_demand_estimate"
    if unknown:
        return _component(None, basis + "; incomplete_target_data", unknown_targets=sorted(unknown))
    demand = sum(_missing_fraction(roster[target]) for target in eligible)
    return _component(min(1.0, demand / SUPPORT_DEMAND_SCALE), basis,
                      eligible_targets=sorted(eligible), missing_health_fractions=demand,
                      demand_scale=SUPPORT_DEMAND_SCALE)


def _air_demand(state, opponents, roster):
    """Require an observed airborne combat capability relevant to this roster."""
    uncertain = False
    for unit in opponents.values():
        if unit.get("is_flying") is False:
            continue
        caps, _ = _capabilities(state, unit)
        if caps == frozenset():
            continue  # Flying support alone is not a confirmed combat threat.
        if unit.get("is_flying") is not True or caps is None:
            uncertain = True
            continue
        for target in roster.values():
            classes = _target_classes(target)
            if classes is None:
                uncertain = True
            elif classes & caps:
                return True
    return None if uncertain else False


def _scarcity(state, actor, roster, opponents, capabilities, supports, side):
    scope = "complete_current_allied_roster" if side == "allies" else "current_visible_enemy_roster_only"
    air = _air_demand(state, opponents, roster)
    capable = [key for key, caps in capabilities.items() if caps is not None and "air" in caps]
    uncertain_air = [key for key, caps in capabilities.items() if caps is None]
    support_active = [key for key, support in supports.items()
                      if support["value"] is not None and support["value"] > 0]
    uncertain_support = [key for key, support in supports.items() if support["value"] is None]
    own_capability = capabilities[actor]
    air_value = 0.0
    air_availability = "known"
    if air is None or (air and own_capability is None):
        air_value, air_availability = None, "unknown"
    elif air and "air" in (own_capability or ()):
        # Unknown units count as potential substitutes, so no unjustified
        # assertion that this is the unique provider is made.
        air_value = 1.0 / (len(capable) + len(uncertain_air))
        air_availability = "partial" if uncertain_air else "known"
    healing_value = 0.0
    healing_availability = "known"
    if actor in support_active:
        healing_value = 1.0 / (len(support_active) + len(uncertain_support))
        healing_availability = "partial" if uncertain_support else "known"
    elif actor in uncertain_support:
        healing_value, healing_availability = None, "unknown"
    values = [value for value in (air_value, healing_value) if value is not None]
    # Known positive evidence can be used conservatively despite another
    # unknown channel; completely unknown evidence stays null, not known zero.
    score = max(values) if values and (max(values) > 0 or len(values) == 2) else None
    availability = ("unknown" if score is None else "partial" if
                    air_availability != "known" or healing_availability != "known" else "known")
    return _component(score, "demand_and_available_capability_required; " + scope,
                      availability, air={"value": air_value, "demand": air,
                                         "confirmed_providers": len(capable),
                                         "potential_substitutes": len(uncertain_air),
                                         "availability": air_availability},
                      healing={"value": healing_value, "confirmed_providers": len(support_active),
                               "potential_substitutes": len(uncertain_support),
                               "availability": healing_availability})


def _distance(first, second):
    left, right = first.get("position"), second.get("position")
    if not isinstance(left, (list, tuple)) or not isinstance(right, (list, tuple)):
        return None
    if len(left) < 2 or len(right) < 2:
        return None
    coordinates = [_number(x) for x in (*left[:2], *right[:2])]
    if any(value is None for value in coordinates):
        return None
    value = math.hypot(coordinates[0] - coordinates[2], coordinates[1] - coordinates[3])
    return value if math.isfinite(value) else None


def _cluster_potential(target_ids, opponents):
    """Visible-center proximity proxy only; no claim of actual splash damage."""
    if len(target_ids) < 2:
        return 0.0
    sizes = []
    for center in target_ids:
        distances = [_distance(opponents[center], opponents[other]) for other in target_ids]
        if any(value is None for value in distances):
            return None
        sizes.append(sum(value <= 2.5 for value in distances))
    return min(1.0, max(0, max(sizes) - 1) / 3.0)


def _spatial_values(state, side, actor, unit, roster, opponents, row, scale):
    """Separate bounded local preservation and target scores from intrinsic value.

    This optional layer never creates legal targets, path plans or enemy intent.
    Its coefficients and 2.5/6.0 proximity radii are engineering priors, not DPS.
    Missing relationships/positions contribute no bonus and are marked unknown.
    """
    relations = state.get("relations", {})
    unknown, targets, attackable = [], [], []
    if side == "allies":
        pairs = relations.get(actor)
        if not isinstance(pairs, dict):
            pairs = {}
            unknown.append("relations")
        for enemy in opponents:
            relation = pairs.get(enemy)
            if not isinstance(relation, dict):
                unknown.append("relations")
                continue
            if not isinstance(relation.get("smac_attack_available"), bool):
                unknown.append("smac_attack_available")
            if relation.get("smac_attack_available") is True:
                attackable.append(enemy)
                if not isinstance(relation.get("in_weapon_range"), bool):
                    unknown.append("in_weapon_range")
                if relation.get("in_weapon_range") is True:
                    targets.append(enemy)
        nearby = 0
        for ally, other in roster.items():
            if ally == actor:
                continue
            distance = _distance(unit, other)
            if distance is None:
                unknown.append("positions")
            elif distance <= SUPPORT_DISTANCE:
                nearby += 1
        engagement = min(1.0, len(targets) / 3.0)
        support = min(1.0, nearby / 3.0) if targets else 0.0
        splash = _cluster_potential(targets, opponents) if _identity(unit)[0] == "Colossus" else 0.0
        if splash is None:
            unknown.append("cluster_positions")
        bonus = .15 * engagement + .10 * support + .15 * (splash or 0.0)
        components = {"engagement": engagement, "nearby_support": support, "cluster_potential": splash}
        field = "preservation_value"
    else:
        for ally in opponents:
            pairs = relations.get(ally)
            pair = pairs.get(actor) if isinstance(pairs, dict) else None
            if not isinstance(pair, dict):
                unknown.append("relations")
                continue
            if not isinstance(pair.get("in_enemy_weapon_range"), bool):
                unknown.append("in_enemy_weapon_range")
            if not isinstance(pair.get("smac_attack_available"), bool):
                unknown.append("smac_attack_available")
            if pair.get("in_enemy_weapon_range") is True:
                targets.append(ally)
            if pair.get("smac_attack_available") is True:
                attackable.append(ally)
        pressure = min(1.0, len(targets) / 3.0)
        splash = _cluster_potential(targets, opponents) if _identity(unit)[0] == "Colossus" else 0.0
        if splash is None:
            unknown.append("cluster_positions")
        bonus = .20 * pressure + .15 * (splash or 0.0)
        components = {"observed_range_pressure": pressure, "cluster_potential": splash}
        field = "target_priority_value"
    importance = row["importance"] * (1.0 + bonus)
    return {
        field: importance / (importance + scale),
        "spatial_context": {
            "components": components, "bonus_fraction": bonus,
            "score_semantics": field, "local_importance": importance,
            "range_covered_opponents": sorted(targets),
            "currently_attackable_relations": sorted(attackable),
            "data_completeness": "partial" if unknown else "complete",
            "unknown_fields": sorted(set(unknown)),
            "basis": "bounded_local_geometry_proxy_not_damage_or_intent; no_path_or_legality_changes",
        },
    }


def evaluate_unit_values(state, parameters=None):
    """Return deterministic JSON-safe estimates; never mutate the snapshot.

    ``off`` returns an empty sidecar. Callers should skip even this call in
    off mode. Invalid/missing health data uses factor 1; unknown contextual
    components have null value and neutral applied_value 0. Confidence is
    a completeness label, never a probability or multiplier.
    """
    params = validate_parameters(parameters)
    result = {"mode": params["mode"], "version": EVALUATION_VERSION,
              "evaluation_scope": "team_visible", "estimate_kind": "heuristic_tactical_importance",
              "parameters": params, "allies": {}, "visible_enemies": {}}
    if params["mode"] == "off":
        return result
    rosters = {side: dict(state.get(side) or {}) for side in ("allies", "visible_enemies")}
    for side, roster in rosters.items():
        live = {actor: unit for actor, unit in roster.items() if not _dead(unit)}
        opponent_side = "visible_enemies" if side == "allies" else "allies"
        opponents = {actor: unit for actor, unit in rosters[opponent_side].items() if not _dead(unit)}
        contextual = params["mode"] == "contextual"
        capabilities = ({actor: _capabilities(state, unit)[0] for actor, unit in live.items()}
                        if contextual else {})
        supports = ({actor: _support(state, side, actor, unit, live) for actor, unit in live.items()}
                    if contextual else {})
        for actor, unit in sorted(roster.items()):
            row = _base_row(unit)
            components = row["components"]
            if params["mode"] == "contextual" and actor in live:
                components["matchup"] = _matchup(capabilities[actor], opponents)
                components["support"] = supports[actor]
                components["scarcity"] = _scarcity(
                    state, actor, live, opponents, capabilities, supports, side)
                for key, component in components.items():
                    if component["availability"] != "known":
                        row["unknown_fields"].append(key)
                if capabilities[actor] is None:
                    row["unknown_fields"].append("weapon_target_types")
                if _identity(unit)[0] == "Medivac" and _number(unit.get("energy")) is None:
                    row["unknown_fields"].append("energy")
                multiplier = 1.0 + sum(params["w_" + key] * component["applied_value"]
                                       for key, component in components.items())
                row["importance"] = row["combat_value"] * multiplier
                row["reasons"].append("bounded_contextual_corrections")
            else:
                components.update({key: _component(0.0, "disabled_in_basic_or_dead", "disabled")
                                   for key in ("matchup", "support", "scarcity")})
            row["normalized_importance"] = row["importance"] / (row["importance"] + params["scale"])
            row["unknown_fields"] = sorted(set(row["unknown_fields"]))
            row["confidence"] = "partial" if row["unknown_fields"] else "complete"
            row["data_completeness"] = row["confidence"]
            if contextual and params["spatial_context"]:
                if actor in live:
                    row.update(_spatial_values(state, side, actor, unit, live, opponents,
                                               row, params["scale"]))
                    if row["spatial_context"]["data_completeness"] == "partial":
                        row["data_completeness"] = "partial"
                else:
                    row["preservation_value" if side == "allies" else "target_priority_value"] = 0.0
            result[side][actor] = row
    return result
