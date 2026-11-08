"""One Protoss unit's pre-contact movement, evaluated before committing adaptive.

Compare adaptive, the original centroid step and every legal direction. The v2
recommendation trades distance to local support against progress toward the
same visible contact. All distances are predictions from current coordinates.
"""

from skills.local_v1.library import (
    DEFAULTS, _attack_targets, _distance, _healer, _id, _menu,
    _moves, _position,
)


EPSILON = 1e-9


def _observed_position(unit):
    coords = unit.get("position")
    if not isinstance(coords, (tuple, list)) or len(coords) < 2 or None in coords[:2]:
        return None
    return _position(unit)


def _nearest(position, units):
    # IDs are already in numeric order, so equally close units have a stable binding.
    return min(((unit, _distance(position, point)) for unit, point in units),
               key=lambda item: item[1])


def _legacy_action(origin, friends, moves, base_action):
    center = tuple(sum(p[axis] for _, p in friends) / len(friends) for axis in (0, 1))
    distance, action = min((_distance(dest, center), move) for move, dest in moves.items())
    return action if distance + EPSILON < _distance(origin, center) else base_action


def analyze(payload, baseline, actor):
    """Return all movement bindings and the heuristic choice for an eligible actor.

    The caller owns episode timing and action submission. Recommendations equal
    to base_action preserve adaptive's entire proposal, including its histories.
    """
    state = payload["state"]
    unit = state["allies"].get(actor)
    if not unit or not unit.get("alive") or _healer(unit):
        return None
    if baseline.get("roles", {}).get(actor) != "reposition" or _attack_targets(payload, actor):
        return None

    origin = _observed_position(unit)
    friends = [(key, pos) for key, ally in state["allies"].items()
               if key != actor and ally.get("alive") and (pos := _observed_position(ally)) is not None]
    friends.sort(key=lambda item: _id(item[0], "u"))
    if origin is None or not friends or _nearest(origin, friends)[1] <= DEFAULTS["regroup_distance"]:
        return None
    menu = _menu(payload, actor)
    legal_moves = menu.intersection((2, 3, 4, 5))
    if not legal_moves:
        return None
    observed_moves = _moves(payload, actor)
    moves = {int(action): pos for action, row in observed_moves.items()
             if (pos := _observed_position({"position": row["destination"]})) is not None}
    if set(moves) != legal_moves:
        raise ValueError("A legal regroup movement is missing its observed destination")
    base_action = baseline["actions"][_id(actor, "u")]
    if base_action not in menu:
        raise ValueError("Adaptive action is outside the actor's legal menu")
    base_dest = moves.get(base_action, origin)

    enemies = [(key, pos) for key, enemy in state.get("visible_enemies", {}).items()
               if enemy.get("alive", True) and (pos := _observed_position(enemy)) is not None]
    enemies.sort(key=lambda item: _id(item[0], "e"))
    enemy_id = _nearest(origin, enemies)[0] if enemies else None
    contact = dict(enemies).get(enemy_id)
    available = [(key, point) for key, point in friends
                 if not _healer(state["allies"][key])
                 and baseline.get("roles", {}).get(key) not in ("retreat", "recover")]
    covering = [(key, point) for key, point in available
                if enemy_id and enemy_id in _attack_targets(payload, key)]
    support = covering or available

    # Keep the contact and eligible support group fixed when comparing directions.
    # A one-unit gain toward support offsets at most one unit of extra contact distance.
    support_before = _nearest(origin, support)[1] if support else None
    support_base = _nearest(base_dest, support)[1] if support else None
    contact_base = _distance(base_dest, contact) if contact is not None else None
    options = []
    for action in sorted(legal_moves | {base_action}):
        dest = moves.get(action, base_dest)
        partner, support_after = _nearest(dest, support) if support else (None, None)
        enemy_after = _distance(dest, contact) if contact is not None else None
        support_gain = support_base - support_after if support else None
        contact_cost = max(0.0, enemy_after - contact_base) if contact is not None else None
        margin = support_gain - contact_cost if support and contact is not None else None
        options.append({"action": action, "destination": list(dest),
                        "nearest_support_id": partner, "support_distance": support_after,
                        "contact_distance": enemy_after, "support_gain_vs_base": support_gain,
                        "contact_cost_vs_base": contact_cost, "margin_vs_base": margin})

    choices = [row for row in options if row["margin_vs_base"] is not None
               and row["margin_vs_base"] > EPSILON
               and row["support_distance"] + EPSILON < support_before]
    best = min(choices, key=lambda row: (-row["margin_vs_base"], row["contact_cost_vs_base"],
                                        row["action"])) if choices else None
    return {"family": "regroup_step", "actor": actor, "base_action": base_action,
            "legacy_action": _legacy_action(origin, friends, moves, base_action),
            "recommended_action": best["action"] if best else base_action,
            "contact_id": enemy_id, "support_ids": [key for key, _ in support],
            "support_basis": "can_attack_contact" if covering else "available_allies",
            "support_distance_before": support_before, "options": options}
