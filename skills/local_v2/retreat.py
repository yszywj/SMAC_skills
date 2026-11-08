"""Single-actor retreat. Enumerate every allowed step before recommending one.

Distances and weapon envelopes are estimates against currently visible targets
at their current positions. No future enemy movement, hit or return is assumed.
"""
from math import dist, isfinite

from skills.current.skills.cover_retreat import PARAMS, detect_retreat_candidates, select_retreat_action
from skills.current.skills.focus_fire import _compatible_weapon
from skills.current.skills.movement_scoring import predicted_move
from skills.local_v1.library import ATTACK_ROLES, _healer, _id, _menu, _moves, _position


def _known_number(value):
    return type(value) in (int, float) and isfinite(value) and value >= 0


def _point(unit):
    position = unit.get("position")
    if not isinstance(position, (list, tuple)) or len(position) < 2 or None in position[:2]:
        return None
    return _position(unit)


def _enemy_reach(state, actor, unit, enemy_id, enemy):
    """Return the observed native envelope, -1 for incompatible, None for unknown."""
    relation = state["relations"][actor].get(enemy_id, {})
    if relation.get("enemy_weapon_range_status") == "no_compatible_weapon":
        return -1
    profile = state.get("unit_type_profiles", {}).get(str(enemy.get("unit_type_id")), {})
    if (not profile.get("known") or not profile.get("weapons")
            or "is_flying" not in unit or unit.get("unit_type_id") is None):
        return None
    weapons = profile["weapons"]
    if any(w.get("target_type") not in ("any", "ground", "air") for w in weapons):
        return None
    usable = [w for w in weapons if _compatible_weapon({"weapons": [w]}, unit)]
    if not usable:
        return -1
    if (not _known_number(unit.get("radius")) or not _known_number(enemy.get("radius"))
            or any(not _known_number(w.get("range")) for w in usable)):
        return None
    return max(w["range"] for w in usable) + unit["radius"] + enemy["radius"]


def _own_reach(state, actor, unit, enemy_id, enemy):
    relation = state["relations"][actor].get(enemy_id, {})
    if relation.get("weapon_range_status") == "no_compatible_weapon":
        return -1
    values = [relation.get("native_weapon_range"), relation.get("smac_attack_distance_limit"),
              unit.get("radius"), enemy.get("radius")]
    if not all(_known_number(v) for v in values):
        return None
    native, limit, own_radius, enemy_radius = values
    return min(native + own_radius + enemy_radius, limit)


def _counterfire(state, actor, distances, enemies):
    """Possible shot next window with static enemies, subject to runtime masks."""
    unit = state["allies"][actor]
    cooldown, window = unit.get("weapon_cooldown_loops"), state.get("step_mul")
    ready = True if unit.get("weapon_ready") is True else None
    if _known_number(cooldown) and _known_number(window):
        ready = cooldown <= window
    if ready is False:
        return False
    unknown = False
    for key, enemy in enemies.items():
        reach = _own_reach(state, actor, unit, key, enemy)
        if reach == -1:
            continue
        if reach is None or distances[key] is None:
            unknown = True
        elif distances[key] <= reach:
            return ready
    return None if unknown else False


def _distances(position, enemy_positions):
    return {key: dist(position, pt) if position is not None and pt is not None else None
            for key, pt in enemy_positions.items()}


def _assess(state, actor, action, position, enemies, enemy_positions, friends, reaches, next_position):
    distances = _distances(position, enemy_positions)
    covering = [key for key in enemies if _known_number(reaches[key]) and distances[key] is not None
                and distances[key] <= reaches[key]]
    unknown = [key for key in enemies if reaches[key] is None or (reaches[key] != -1 and distances[key] is None)]
    possible_threats = [d for key, d in distances.items() if d is not None and reaches[key] != -1]
    next_distances = _distances(next_position, enemy_positions)
    return {"action": action, "destination": position, "estimated_next_position": next_position,
            "known_covering_enemies": sorted(covering), "unknown_enemy_coverage": sorted(unknown),
            "nearest_threat_distance": min(possible_threats, default=None),
            "nearest_ally_distance": min((dist(position, friend) for friend in friends), default=None)
            if position is not None else None,
            "next_counterfire_possible": _counterfire(state, actor, next_distances, enemies)}


def _next_position(state, actor, action):
    unit = state["allies"][actor]
    speed = state.get("unit_type_profiles", {}).get(str(unit.get("unit_type_id")), {}).get("movement_speed_native")
    if (not _known_number(speed) or not _known_number(state.get("step_mul")) or _point(unit) is None
            or any(_point(u) is None for u in state.get("visible_enemies", {}).values())):
        return None
    return predicted_move(state, actor, action)["destination"]


def _safer(move, base, loses_shot):
    if move["unknown_enemy_coverage"] or base["unknown_enemy_coverage"]:
        return False
    after, before = move["nearest_threat_distance"], base["nearest_threat_distance"]
    if after is None or before is None or after < before:
        return False
    threats, original = set(move["known_covering_enemies"]), set(base["known_covering_enemies"])
    if not threats.issubset(original):
        return False
    if len(threats) < len(original):
        return True
    if loses_shot or after <= before + PARAMS["improvement_min"]:
        return False
    if base["next_counterfire_possible"] is True and move["next_counterfire_possible"] is not True:
        return False
    supported = base["nearest_ally_distance"]
    return not (supported is not None and supported <= PARAMS["support_distance"]
                and move["nearest_ally_distance"] > PARAMS["support_distance"])


def analyze(payload, baseline, actor):
    """Describe original action and all masked movement directions for a danger binding."""
    state = payload["state"]
    unit = state["allies"].get(actor)
    slot = _id(actor, "u")
    if (not unit or not unit.get("alive") or _healer(unit) or slot >= len(baseline["actions"])
            or baseline.get("roles", {}).get(actor) not in ATTACK_ROLES | {"reposition"}
            or actor not in detect_retreat_candidates(state)):
        return None
    base_action = baseline["actions"][slot]
    menu = _menu(payload, actor)
    if base_action not in menu:
        return None
    moves = {key: move for key, move in _moves(payload, actor).items()
             if _point({"position": move["destination"]}) is not None}
    movement_state = {**state, "movement_candidates": {**state.get("movement_candidates", {}), actor: moves},
                      "visible_enemies": state.get("visible_enemies", {})}
    enemies = {key: enemy for key, enemy in state.get("visible_enemies", {}).items() if enemy.get("alive", True)}
    enemy_positions = {key: _point(enemy) for key, enemy in enemies.items()}
    friends = [pt for key, friend in state["allies"].items() if key != actor and friend.get("alive")
               and (pt := _point(friend)) is not None]
    reaches = {key: _enemy_reach(state, actor, unit, key, enemy) for key, enemy in enemies.items()}
    origin = _point(unit)
    actions = [base_action, *sorted(menu.intersection({2, 3, 4, 5}) - {base_action})]
    options = []
    for action in actions:
        if action in (2, 3, 4, 5):
            destination = _point({"position": moves.get(str(action), {}).get("destination")})
            next_position = _next_position(movement_state, actor, action) if destination is not None else None
        else:
            # Compare a stationary attack/stop as the geometric reference. An
            # attack order can track its target; this is not a simulated step.
            destination = next_position = origin
        options.append(_assess(state, actor, action, destination, enemies, enemy_positions,
                               friends, reaches, next_position))
    base = options[0]
    relation = state["relations"][actor].get(f"e{base_action-6}", {})
    loses_shot = (base_action >= 6 and unit.get("weapon_ready") is not False
                  and relation.get("in_weapon_range") is not False)
    safer = [move for move in options[1:] if _safer(move, base, loses_shot)]
    def rank(move):
        support = move["nearest_ally_distance"]
        return (len(move["known_covering_enemies"]), move["next_counterfire_possible"] is not True,
                support is not None and support > PARAMS["support_distance"],
                -move["nearest_threat_distance"], support if support is not None else 0, move["action"])
    recommended = min(safer, key=rank) if safer else base
    # v1 needs an explicit distance map for every move and positions for all
    # supporters. A v2 caller may have a partial observation; keep the v1
    # recommendation unknown without shrinking that caller's legal menu.
    legacy = None
    if (all(isinstance(m.get("distances_to_visible_enemies"), dict)
            and all(_known_number(d) for d in m["distances_to_visible_enemies"].values()) for m in moves.values())
            and all(_point(u) is not None for u in state["allies"].values() if u.get("alive"))):
        legacy, _, _ = select_retreat_action(movement_state, actor, {actor})
    return {"family": "retreat_step", "actor": actor, "base_action": base_action,
            "recommended_action": recommended["action"], "legacy_action": legacy, "options": options}
