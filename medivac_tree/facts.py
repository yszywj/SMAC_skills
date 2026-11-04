"""Observed facts and native legality. No old controller imports or scores."""

import math
import re


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def number(value, name):
    if not finite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def actor_index(actor):
    if not isinstance(actor, str) or re.fullmatch(r"u\d+", actor) is None:
        raise ValueError(f"invalid actor ID {actor!r}")
    return int(actor[1:])


def position(unit):
    point = unit.get("position")
    if not isinstance(point, (list, tuple)) or len(point) != 2:
        raise ValueError("unit position must contain two coordinates")
    return tuple(number(v, "position") for v in point)


def is_medivac(unit):
    return str(unit.get("type", "")).lower().startswith("medivac") or unit.get("unit_type_id") == 54


def hp_ratio(unit):
    maximum = number(unit.get("max_health"), "max_health")
    if maximum <= 0:
        raise ValueError("max_health must be positive")
    return number(unit.get("health"), "health") / maximum


def loss_ratio(unit):
    loss = unit.get("health_lost_since_previous_decision")
    return max(0., float(loss)) / number(unit["max_health"], "max_health") if finite(loss) else 0.


def enough_energy(unit):
    return finite(unit.get("energy")) and unit["energy"] >= 1


def observed_target(state, actor):
    unit = state["allies"][actor]
    orders = unit.get("orders") or []
    if not unit.get("alive") or not orders or orders[0].get("ability_id") != 386:
        return None
    target = orders[0].get("target_unit")
    if target in state["allies"] and state["allies"][target].get("alive"):
        return target
    return None


def legal_actions(state, actor):
    raw = (state.get("legal_actions") or {}).get(actor)
    if not isinstance(raw, (list, tuple, set, frozenset)):
        raise ValueError(f"{actor}: explicit legal_actions is required")
    if any(isinstance(a, bool) or not isinstance(a, int) or a < 0 for a in raw):
        raise ValueError(f"{actor}: legal_actions must contain nonnegative integers")
    if not raw:
        raise ValueError(f"{actor}: no legal action")
    return set(raw)


def moves(state, actor, legal):
    result = {}
    for key, entry in state.get("movement_candidates", {}).get(actor, {}).items():
        action = int(key)
        if action not in legal or action not in (2, 3, 4, 5):
            continue
        result[action] = position({"position": entry.get("destination")})
    return result


def wounded_patient(unit):
    return bool(unit.get("alive") and unit.get("is_flying") is False and not is_medivac(unit)
                and 0 < number(unit.get("health"), "patient health") < number(unit.get("max_health"), "patient max_health"))


def heal_candidates(state, actor, legal):
    if not enough_energy(state["allies"][actor]):
        return {}
    result = {}
    for key, entry in state.get("heal_candidates", {}).get(actor, {}).items():
        action = int(key)
        target = entry.get("target")
        if action not in legal or target == actor:
            continue
        if target not in state["allies"]:
            raise ValueError(f"{actor}: unknown heal target {target}")
        if action != 6 + actor_index(target):
            raise ValueError(f"{actor}: heal slot does not match target {target}")
        if wounded_patient(state["allies"][target]):
            result[target] = action
    return result


def threats(state, actor, parameters):
    """Only visible enemies; known ground-only weapons never threaten air."""
    result = []
    unit = state["allies"][actor]
    origin = position(unit)
    for enemy, relation in state.get("relations", {}).get(actor, {}).items():
        target = state.get("visible_enemies", {}).get(enemy)
        if target is None or target.get("alive") is False:
            continue
        status = relation.get("enemy_weapon_range_status")
        if status == "no_compatible_weapon":
            continue
        profile = state.get("unit_type_profiles", {}).get(str(target.get("unit_type_id")), {})
        weapons = profile.get("weapons") or []
        air_weapons = [w for w in weapons if w.get("target_type") in ("air", "any", 2, 3)]
        if profile.get("known") and weapons and not air_weapons:
            continue
        reach_values = [w["range"] for w in air_weapons if finite(w.get("range"))]
        reach = max(reach_values, default=parameters["threat_distance"])
        reach += float(unit.get("radius", 0) or 0) + float(target.get("radius", 0) or 0)
        point = position(target)
        distance = math.dist(origin, point)
        result.append({"enemy": enemy, "position": list(point), "reach": reach,
                       "distance": distance, "known_compatible": bool(air_weapons or status in ("in_range", "out_of_range")),
                       "near": bool(relation.get("in_enemy_weapon_range") is True or distance <= parameters["threat_distance"])})
    return result


def exposure(rows, point):
    return sum(math.dist(point, r["position"]) <= r["reach"] for r in rows)


def clearance(rows, point):
    return min((math.dist(point, r["position"]) - r["reach"] for r in rows), default=1000.)
