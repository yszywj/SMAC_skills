"""CombatPositioning: engagement-geometry sub-skill inside the combat mode.

Not a top-level mode. The combat controller (AdaptiveFocusFire) decides who
attacks / retreats / heals; this module only places the units it explicitly
releases (reposition-role actors with no legal attack) into spaced engagement
positions. Protected actions (attack, retreat, heal, recovery, promised
re-entry) are passed as ``fixed_actions`` and never changed; those units still
reserve space. Falls back to the existing coordinated approach when no enemy
cluster geometry or no useful position exists — never re-ranks forever.
"""

import math

from .coordinated_approach import coordinate_approaches
from .group_geometry import (PARAMS as GEOMETRY_PARAMS, best_move_toward,
                             connected_components, unit_role)

PARAMS = {
    **GEOMETRY_PARAMS,
    "combat_positioning": False,     # master switch (set via combat_params)
    "positioning_spacing": "normal",  # compact | normal | spread
    "positioning_layout": "free",     # free | front_back
}

SPACING_METRES = {"compact": 1.0, "normal": 1.8, "spread": 3.0}


def _enemy_clusters(state, link_radius):
    enemies = list(state["visible_enemies"])
    positions = {e: state["visible_enemies"][e]["position"] for e in enemies}
    parent = {e: e for e in enemies}

    def find(e):
        while parent[e] != e:
            parent[e] = parent[parent[e]]
            e = parent[e]
        return e

    for i, a in enumerate(enemies):
        for b in enemies[i + 1:]:
            if math.dist(positions[a], positions[b]) <= link_radius:
                parent[find(a)] = find(b)
    groups = {}
    for e in enemies:
        groups.setdefault(find(e), []).append(e)
    return sorted(groups.values(), key=lambda g: (-len(g), min(g)))


def allocate_combat_positioning(state, actors, histories, fixed_actions=None, parameters=None):
    """Positioning proposal for released actors; returns coordinate_approaches-
    compatible (actions, diagnostics, updates). Unknown enemy positions never
    enter the geometry: only visible enemies form clusters.
    """
    params = {**PARAMS, **(parameters or {})}
    fixed_actions = fixed_actions or {}
    allies = state["allies"]
    eligible = [a for a in actors
                if allies.get(a, {}).get("alive")
                and a not in fixed_actions
                and not any(r.get("smac_attack_available")
                            for r in state.get("relations", {}).get(a, {}).values())]
    if not eligible or not state["visible_enemies"]:
        return coordinate_approaches(state, actors, histories, fixed_actions, params)

    spacing = SPACING_METRES.get(params["positioning_spacing"], 1.8)
    front_back = params["positioning_layout"] == "front_back"
    clusters = _enemy_clusters(state, params["link_radius"])

    # Reservations from fixed actions (they occupy their predicted or current space).
    from .movement_scoring import predicted_move
    reservations = {}
    for actor, action in fixed_actions.items():
        unit = allies.get(actor)
        if unit is None or not unit.get("alive"):
            continue
        reservations[actor] = (predicted_move(state, actor, action)["destination"]
                               if str(action) in state.get("movement_candidates", {}).get(actor, {})
                               else unit["position"])

    actions, diagnostics, updates = {}, {}, {}
    ordered = sorted(eligible, key=lambda a: (unit_role(state, a) != "melee", a))
    ranged_rank = {}
    for actor in ordered:
        if unit_role(state, actor) == "ranged":
            ranged_rank[actor] = len(ranged_rank)
    for actor in ordered:
        unit = allies[actor]
        origin = unit["position"]
        role = unit_role(state, actor)
        # Multi-direction enemies: engage the NEAREST cluster, not the global
        # centroid — no blind charge into the middle of two enemy groups.
        best = None
        for cluster in clusters:
            center = [sum(state["visible_enemies"][e]["position"][i] for e in cluster) / len(cluster)
                      for i in range(2)]
            reach = None
            for e in cluster:
                relation = state.get("relations", {}).get(actor, {}).get(e, {})
                native = relation.get("native_weapon_range")
                if native is not None:
                    reach = (native + unit.get("radius", .5)
                             + state["visible_enemies"][e].get("radius", .5))
                    break
            reach = reach if reach is not None else params["link_radius"]
            distance = math.dist(origin, center) - reach
            if best is None or distance < best[0] - 1e-6:
                best = (distance, center, reach)
        _, center, reach = best
        direction = [center[0] - origin[0], center[1] - origin[1]]
        length = math.hypot(*direction) or 1.0
        direction = [direction[0] / length, direction[1] / length]
        # Perpendicular lateral offset spreads ranged units sideways to gain
        # firing lines; melee keeps the central forward channel.
        lateral = [ -direction[1], direction[0]]
        slot_distance = max(0.0, math.dist(origin, center) - reach)
        if role == "melee":
            goal = [origin[0] + direction[0] * 2.0, origin[1] + direction[1] * 2.0]
        elif role == "healer":
            # Support without a heal task keeps a supportable position: the
            # friendly centroid, approached if too far behind.
            friends = [u["position"] for a, u in allies.items()
                       if a != actor and u.get("alive")]
            goal = [sum(p[i] for p in friends) / len(friends) for i in range(2)] if friends \
                else [origin[0], origin[1]]
        else:
            spread_index = ranged_rank.get(actor, 0)
            side = 1 if spread_index % 2 == 0 else -1
            offset = side * spacing * (1 + spread_index // 2)
            goal = [origin[0] + direction[0] * 2.0 + lateral[0] * offset,
                    origin[1] + direction[1] * 2.0 + lateral[1] * offset]
        if front_back and role != "melee":
            # Front-back layout: non-melee hold one spacing behind melee line.
            goal = [goal[0] - direction[0] * spacing, goal[1] - direction[1] * spacing]
        action, info = best_move_toward(state, actor, goal, reservations, params,
                                        (histories.get(actor) or {}).get("positioning_last_action"))
        if action is None:
            continue
        actions[actor] = action
        reservations[actor] = predicted_move(state, actor, action)["destination"]
        updates[actor] = {"positioning_last_action": action}
        diagnostics[actor] = {"role": role, "goal": [round(g, 3) for g in goal],
                              "nearest_cluster_size": len(clusters[0]),
                              "cluster_count": len(clusters), **info,
                              "meaning": "Positional proposal for released units only; "
                                         "protected combat actions untouched."}
    if not actions:
        # No useful position: degrade to the original coordinated approach.
        return coordinate_approaches(state, actors, histories, fixed_actions, params)
    return actions, diagnostics, updates
