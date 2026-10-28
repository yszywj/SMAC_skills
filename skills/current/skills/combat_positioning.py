"""CombatPositioning: engagement-geometry sub-skill inside the combat mode.

Not a top-level mode. The combat controller (AdaptiveFocusFire) decides who
attacks / retreats / heals; this module only places the units it explicitly
releases (reposition-role actors with no legal attack) into spaced engagement
positions. Protected actions (attack, retreat, heal, recovery, promised
re-entry) are passed as ``fixed_actions`` and never changed; those units still
reserve space.

Geometry (v8.1): slots are anchored to the SQUAD reference (team anchor,
frontline ally, selected enemy cluster) — never to each unit's own current
position, so a unit can no longer chase a "step forward, step back" goal.
``front_back`` keeps non-melee behind the REAL frontline (or, with no melee,
behind the engagement line itself) instead of retreating every step. Spacing
presets only widen the lateral spread.

Fallback: per-unit and whole-set fallback to the legacy coordinated approach
when positioning yields no useful improvement. When the caller enables
``approach_progress_guard``, its guard runs on the same candidates here, so
A-B-A / blocked-move protection survives the new path.
"""

import math

from .approach_progress import guard_approach
from .coordinated_approach import coordinate_approaches
from .group_geometry import (PARAMS as GEOMETRY_PARAMS, base_reservations,
                             best_move_toward, connected_components, predicted_destination,
                             unit_role)
from .movement_scoring import engagement_reach

PARAMS = {
    **GEOMETRY_PARAMS,
    "combat_positioning": False,      # master switch (set via combat_params)
    "positioning_spacing": "normal",  # compact | normal | spread
    "positioning_layout": "free",     # free | front_back
    "positioning_no_progress_limit": 3,  # windows without slot progress -> per-unit fallback
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


def _cluster_center(state, cluster):
    points = [state["visible_enemies"][e]["position"] for e in cluster]
    return [sum(p[i] for p in points) / len(points) for i in range(len(points[0]))]


def _engagement_line(state, actor, cluster, params):
    """Distance from the selected cluster's centre at which this unit should
    stand. Reuses engagement_reach()/relation data; unknown capability never
    becomes a promised range (falls back to a purely geometric hold distance).
    """
    unit = state["allies"][actor]
    reach = None
    for enemy in cluster:
        value = engagement_reach(state, actor, enemy)
        if value is not None:
            reach = value if reach is None else max(reach, value)
    if reach is None:
        return None, "unknown_reach_geometric_hold"
    buffer = {"compact": 0.2, "normal": 0.6, "spread": 1.0}[params["positioning_spacing"]]
    return max(0.5, reach - buffer), "engagement_reach"


def allocate_combat_positioning(state, actors, histories, fixed_actions=None, parameters=None):
    """Positioning proposal for released actors; returns coordinate_approaches-
    compatible (actions, diagnostics, updates). Unknown enemy positions never
    enter the geometry: only visible enemies form clusters.
    """
    params = {**PARAMS, **(parameters or {})}
    fixed_actions = dict(fixed_actions or {})
    allies = state["allies"]
    eligible = [a for a in actors
                if allies.get(a, {}).get("alive")
                and a not in fixed_actions
                and not any(r.get("smac_attack_available")
                            for r in state.get("relations", {}).get(a, {}).values())]
    if not eligible or not state["visible_enemies"]:
        return coordinate_approaches(state, [a for a in actors if a not in fixed_actions],
                                     histories, fixed_actions, params)

    spacing = SPACING_METRES.get(params["positioning_spacing"], 1.8)
    front_back = params["positioning_layout"] == "front_back"
    clusters = _enemy_clusters(state, params["link_radius"])
    live_allies = [a for a, u in allies.items() if u.get("alive")]

    # Reservations: every living ally's current position by default; decided
    # (fixed and already-proposed) actions overwrite with predicted endpoints.
    reservations = base_reservations(state)
    for actor, action in fixed_actions.items():
        unit = allies.get(actor)
        if unit is None or not unit.get("alive"):
            continue
        reservations[actor] = (predicted_destination(state, actor, action),
                               unit.get("is_flying", False))

    actions, diagnostics, updates = {}, {}, {}
    # Stable membership: deterministic ordering by role then id; cluster
    # assignment is per-unit (nearest cluster), so members dying do not
    # renumber other clusters' slots.
    ordered = sorted(eligible, key=lambda a: (unit_role(state, a) != "melee", a))
    fallback_units = []
    lateral_rank = {}
    for actor in ordered:
        if unit_role(state, actor) == "ranged":
            lateral_rank[actor] = len(lateral_rank)
    for actor in ordered:
        unit = allies[actor]
        origin = unit["position"]
        role = unit_role(state, actor)
        # Multi-direction enemies: engage the NEAREST cluster, not the global
        # centroid — no blind charge between two enemy groups.
        best = None
        for cluster in clusters:
            center = _cluster_center(state, cluster)
            reach = engagement_reach(state, actor, cluster[0])
            distance = math.dist(origin, center) - (reach or params["link_radius"])
            if best is None or distance < best[0] - 1e-6:
                best = (distance, cluster, center)
        _, cluster, center = best
        hold_distance, hold_reason = _engagement_line(state, actor, cluster, params)
        if hold_distance is None:
            hold = params["link_radius"]
        else:
            hold = hold_distance
        # Squad reference: team anchor + real frontline (closest live ally to
        # this cluster), NOT the unit's own position.
        anchor_points = [u["position"] for u in allies.values() if u.get("alive")]
        anchor = [sum(p[i] for p in anchor_points) / len(anchor_points) for i in range(2)]
        frontline = min(live_allies,
                        key=lambda a: (math.dist(allies[a]["position"], center), a))
        frontline_position = allies[frontline]["position"]
        # Heading from the frontline toward the enemy cluster centre.
        heading = [center[0] - frontline_position[0], center[1] - frontline_position[1]]
        length = math.hypot(*heading) or 1.0
        heading = [heading[0] / length, heading[1] / length]
        lateral = [-heading[1], heading[0]]
        # Slot: stand on the engagement arc around the cluster centre.
        base = [center[0] - heading[0] * hold, center[1] - heading[1] * hold]
        if role == "melee":
            slot = [center[0] - heading[0] * max(0.5, hold * 0.4),
                    center[1] - heading[1] * max(0.5, hold * 0.4)]
        elif role == "healer":
            # Support without a heal task keeps a supportable position near
            # the squad anchor (never cancels an ongoing heal: those units are
            # in fixed_actions).
            slot = list(anchor)
        else:
            rank = lateral_rank.get(actor, 0)
            side = 1 if rank % 2 == 0 else -1
            offset = side * spacing * (1 + rank // 2)
            slot = [base[0] + lateral[0] * offset, base[1] + lateral[1] * offset]
        if front_back and role != "melee":
            # Behind the REAL frontline; with no melee at all this is still
            # the engagement arc, never behind the unit's own position.
            back = spacing if any(unit_role(state, a) == "melee" for a in live_allies) else 0.0
            slot = [slot[0] - heading[0] * back, slot[1] - heading[1] * back]

        # Per-unit improvement check: if the best legal action still moves the
        # unit AWAY from its slot (and there is no exposure justification),
        # fall back to the legacy coordinated approach for this unit.
        current_slot_distance = math.dist(origin, slot)
        action, info = best_move_toward(state, actor, slot, reservations, params,
                                        (histories.get(actor) or {}).get("positioning_last_action"))
        if action is None:
            fallback_units.append(actor)
            continue
        action_distance = math.dist(predicted_destination(state, actor, action), slot)
        history = histories.get(actor) or {}
        stall = history.get("positioning_no_progress", 0)
        at_slot = current_slot_distance <= spacing * 0.75
        if at_slot:
            stall = 0  # standing on the slot is success, not a stall
        elif action_distance >= current_slot_distance - 0.1:
            stall += 1  # no useful improvement this window
        else:
            stall = 0
        if stall >= params["positioning_no_progress_limit"]:
            fallback_units.append(actor)
            diagnostics[actor] = {"role": role, "slot": [round(s, 3) for s in slot],
                                  "fallback": "no_slot_progress",
                                  "no_progress_windows": stall}
            continue

        guarded_action, guard_row, guard_memory = action, None, None
        if params.get("approach_progress_guard"):
            target = cluster[0]
            candidates = {str(a): predicted_destination(state, actor, a)
                          for a in state.get("movement_candidates", {}).get(actor, {})}
            features = {str(a): {"base_score": info["candidates"].get(str(a), {}).get("score", 0.0),
                                 "estimated_contact_windows": 0.0, "collision_cost":
                                     info["candidates"].get(str(a), {}).get("collision_cost", 0.0),
                                 "visible_danger_count": 0, "blocked_move_penalty": 0.0}
                        for a in candidates}
            eligible_moves = list(candidates)
            guarded_action, guard_row, guard_memory = guard_approach(
                state, actor, history, target, candidates, features, action, eligible_moves)
        actions[actor] = guarded_action
        reservations[actor] = (predicted_destination(state, actor, guarded_action),
                               unit.get("is_flying", False))
        updates[actor] = {"positioning_last_action": guarded_action,
                          "positioning_no_progress": stall}
        if guard_memory is not None:
            updates[actor]["approach_progress"] = guard_memory
        diagnostics[actor] = {"role": role, "slot": [round(s, 3) for s in slot],
                              "selected_cluster": list(cluster),
                              "selected_cluster_size": len(cluster),
                              "cluster_count": len(clusters),
                              "hold_reason": hold_reason,
                              "frontline": frontline, **info,
                              "meaning": "Positional proposal for released units only; "
                                         "protected combat actions untouched."}
        if guard_row is not None:
            diagnostics[actor]["progress_guard"] = guard_row
            # The shared finalizer reconciles the actual action with these
            # same predictions after all combat filters have run.
            diagnostics[actor]["predicted_destinations"] = candidates

    if fallback_units:
        # The legacy coordinated approach is the explicit per-unit fallback;
        # positioning-decided units participate as fixed actions so no unit
        # ever gets two proposals in one step.
        fallback_fixed = {**fixed_actions, **actions}
        fb_actions, fb_diagnostics, fb_updates = coordinate_approaches(
            state, fallback_units, histories, fallback_fixed, params)
        actions.update(fb_actions)
        for actor, row in fb_diagnostics.items():
            # Keep the original fallback reason (e.g. no_slot_progress); the
            # legacy approach is where the unit went, not why it left.
            diagnostics[actor] = {**row, "fallback": diagnostics.get(actor, {}).get(
                "fallback", "coordinated_approach")}
        updates.update(fb_updates)
    return actions, diagnostics, updates
