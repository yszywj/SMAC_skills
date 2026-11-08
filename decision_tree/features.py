"""Pure pre-decision features from public observations and caller-owned memory.

Distances describe observed geometry, not path reachability. No outcome, hidden
enemy state, or evaluation metadata is accepted as a model feature.
"""

import math

FEATURE_NAMES = (
    "game_loop", "ally_count", "ally_alive_fraction", "ally_effective_health_ratio",
    "ally_health_ratio", "ally_shield_ratio", "ally_recent_loss_ratio",
    "ally_wounded_fraction", "weapon_ready_fraction", "attackable_ally_fraction",
    "visible_enemy_count", "visible_enemy_effective_health_sum",
    "visible_enemy_effective_health_mean", "main_group_share", "group_radius",
    "mean_pair_distance", "position_coverage", "shield_capable_fraction",
    "healer_fraction", "flying_fraction", "melee_fraction", "ranged_fraction",
    "weapon_profile_coverage", "can_attack_air_fraction", "can_attack_ground_fraction",
    "race_protoss_fraction", "race_terran_fraction", "race_zerg_fraction",
    "race_unknown_fraction", "contacted", "has_last_visible", "loops_since_visible",
    "phase_search", "phase_approach", "phase_engage", "phase_lost_contact",
)

_RACE_TYPES = {
    "protoss": ("zealot", "stalker", "colossus", "sentry", "adept", "immortal",
                "hightemplar", "darktemplar", "archon", "phoenix", "voidray",
                "oracle", "tempest", "carrier", "mothership", "observer", "warpprism", "probe"),
    "terran": ("marine", "marauder", "medivac", "reaper", "ghost", "hellion",
               "hellbat", "siegetank", "cyclone", "widowmine", "thor", "viking",
               "banshee", "raven", "liberator", "battlecruiser", "scv"),
    "zerg": ("zergling", "baneling", "roach", "ravager", "hydralisk", "lurker",
             "infestor", "swarmhost", "ultralisk", "mutalisk", "corruptor",
             "broodlord", "viper", "queen", "overlord", "overseer", "drone"),
}


def _num(value, default=0.0):
    if isinstance(value, bool):
        return default
    try:
        value = float(value)
        return value if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def _ratio(value, maximum):
    return min(1.0, max(0.0, value / maximum)) if maximum > 0 else 0.0


def unit_race(unit):
    race = str(unit.get("race", "")).lower()
    if race in _RACE_TYPES:
        return race
    name = str(unit.get("type", "")).lower().replace("_", "").replace(" ", "")
    for race, names in _RACE_TYPES.items():
        if any(name.startswith(prefix) for prefix in names):
            return race
    return "unknown"


def infer_races(state):
    """Roster races, including dead units; unknown types are never guessed."""
    return tuple(sorted({unit_race(u) for u in state.get("allies", {}).values()}))


def phase_name(features):
    for phase in ("engage", "approach", "lost_contact", "search"):
        if features.get("phase_" + phase, 0) > 0:
            return phase
    return "search"


def extract_features(state, memory=None):
    """Return a finite, fixed numeric schema without mutating state or memory."""
    memory = memory or {}
    roster = list(state.get("allies", {}).values())
    living = {a: u for a, u in state.get("allies", {}).items() if u.get("alive", False)}
    units = list(living.values())
    n = len(units)
    denom = max(1, n)
    enemies = [e for e in state.get("visible_enemies", {}).values()
               if e.get("alive", True) and _num(e.get("health")) + _num(e.get("shield")) > 0]
    roster_max = sum(max(0, _num(u.get("max_health"))) + max(0, _num(u.get("max_shield"))) for u in roster)
    hp = sum(max(0, _num(u.get("health"))) for u in units)
    shield = sum(max(0, _num(u.get("shield"))) for u in units)
    max_hp = sum(max(0, _num(u.get("max_health"))) for u in units)
    max_shield = sum(max(0, _num(u.get("max_shield"))) for u in units)
    loss = sum(max(0, _num(u.get("health_lost_since_previous_decision")))
               + max(0, _num(u.get("shield_lost_since_previous_decision"))) for u in units)
    attackable = sum(any(r.get("smac_attack_available") is True
                         and e in state.get("visible_enemies", {})
                         for e, r in state.get("relations", {}).get(a, {}).items()) for a in living)
    loop = max(0, _num(state.get("game_loop")))
    previous_contact = bool(memory.get("contacted", False))
    contacted = previous_contact or bool(enemies)
    last = memory.get("last_visible_loop")
    has_last = last is not None and 0 <= _num(last, -1) <= loop
    phase = "engage" if attackable else "approach" if enemies else "lost_contact" if previous_contact else "search"
    positions = []
    for unit in units:
        position = unit.get("position")
        if isinstance(position, (tuple, list)) and len(position) == 2:
            xy = [_num(v, float("nan")) for v in position]
            if all(math.isfinite(v) for v in xy):
                positions.append(xy)
    # Connected components at a fixed six-unit observation radius.
    parents = list(range(len(positions)))
    def find(i):
        while parents[i] != i:
            i = parents[i]
        return i
    distances = []
    for i in range(len(positions)):
        for j in range(i):
            d = math.dist(positions[i], positions[j])
            distances.append(d)
            if d <= 6.0:
                parents[find(i)] = find(j)
    sizes = {}
    for i in range(len(positions)):
        root = find(i)
        sizes[root] = sizes.get(root, 0) + 1
    center = [sum(p[k] for p in positions) / len(positions) for k in range(2)] if positions else [0, 0]
    radius = max((math.dist(p, center) for p in positions), default=0.0)
    melee = ranged = profiled = air = ground = 0
    for actor, unit in living.items():
        profile = state.get("unit_type_profiles", {}).get(str(unit.get("unit_type_id")), {})
        weapons = profile.get("weapons", []) or []
        ranges = [_num(w.get("range"), -1) for w in weapons if _num(w.get("range"), -1) >= 0]
        if not ranges:
            ranges = [_num(r.get("native_weapon_range"), -1)
                      for r in state.get("relations", {}).get(actor, {}).values()
                      if _num(r.get("native_weapon_range"), -1) >= 0]
        if ranges:
            profiled += 1
            melee += min(ranges) <= 1.5
            ranged += min(ranges) > 1.5
        targets = {str(w.get("type", w.get("target_type", ""))).lower() for w in weapons}
        air += bool(targets.intersection({"air", "any", "both", "2", "3"}))
        ground += bool(targets.intersection({"ground", "any", "both", "1", "3"}))
    enemy_hp = sum(max(0, _num(e.get("health"))) + max(0, _num(e.get("shield"))) for e in enemies)
    features = {
        "game_loop": loop, "ally_count": n, "ally_alive_fraction": n / max(1, len(roster)),
        "ally_effective_health_ratio": _ratio(hp + shield, roster_max),
        "ally_health_ratio": _ratio(hp, max_hp), "ally_shield_ratio": _ratio(shield, max_shield),
        "ally_recent_loss_ratio": _ratio(loss, max_hp + max_shield),
        "ally_wounded_fraction": sum(_num(u.get("health")) + _num(u.get("shield"))
                                     < _num(u.get("max_health")) + _num(u.get("max_shield")) for u in units) / denom,
        "weapon_ready_fraction": sum(bool(u.get("weapon_ready")) for u in units) / denom,
        "attackable_ally_fraction": attackable / denom,
        "visible_enemy_count": len(enemies), "visible_enemy_effective_health_sum": enemy_hp,
        "visible_enemy_effective_health_mean": enemy_hp / max(1, len(enemies)),
        "main_group_share": max(sizes.values(), default=0) / denom,
        "group_radius": radius, "mean_pair_distance": sum(distances) / max(1, len(distances)),
        "position_coverage": len(positions) / denom,
        "shield_capable_fraction": sum(_num(u.get("max_shield")) > 0 for u in units) / denom,
        "healer_fraction": sum(str(u.get("type", "")).lower().startswith("medivac") for u in units) / denom,
        "flying_fraction": sum(bool(u.get("is_flying", False)) for u in units) / denom,
        "melee_fraction": melee / denom, "ranged_fraction": ranged / denom,
        "weapon_profile_coverage": profiled / denom,
        "can_attack_air_fraction": air / denom, "can_attack_ground_fraction": ground / denom,
        "contacted": float(contacted), "has_last_visible": float(bool(enemies) or has_last),
        "loops_since_visible": 0.0 if enemies or not has_last else loop - _num(last),
    }
    for race in ("protoss", "terran", "zerg", "unknown"):
        features["race_" + race + "_fraction"] = sum(unit_race(u) == race for u in units) / denom
    for name in ("search", "approach", "engage", "lost_contact"):
        features["phase_" + name] = float(name == phase)
    return {name: _num(features[name]) for name in FEATURE_NAMES}
