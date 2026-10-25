"""Shared spatial utilities for regroup / cohesive search / combat positioning.

Pure geometric helpers over the snapshot's public ally data only. Distances
are Euclidean approximations on observed positions: without terrain or a
pathing grid they never claim true reachability or supportability. Ideas
follow Sharpy's group combat manager and Ares squad behaviors (grouping by
proximity, main-group selection, explainable cohesion metrics), re-implemented
for SMAC's discrete action space with no external dependency.
"""

import math

PARAMS = {
    "link_radius": 6.0,          # allies closer than this (Euclidean) share an edge
    "main_group_min_share": 0.6, # main group must hold this share of live allies
    "target_radius": 7.0,        # quantile radius below which the team counts as compact
    "outlier_distance": 8.0,     # a unit farther than this from the main-group anchor is isolated
    "spacing_factor": 1.6,       # ring slot spacing in multiples of combined radius
    "melee_range": 1.5,          # native weapon range below this counts as melee
}


def _finite(value, fallback=None):
    try:
        value = float(value)
        return value if math.isfinite(value) else fallback
    except (TypeError, ValueError):
        return fallback


def living_allies(state):
    """Alive allies sorted by numeric id suffix; ids are opaque strings."""
    return sorted((a for a, u in state["allies"].items() if u.get("alive")),
                  key=lambda a: (len(a), a))


def unit_role(state, actor):
    """Recognize melee / ranged / healer from public data; conservative fallback.

    Healers are identified by the same public type convention as the combat
    skills (Medivac). Melee/ranged comes from own native weapon range when a
    relation exposes it, else from the unit-type profile; unknown types are
    treated as generic ranged (conservative: keep with the group, no special
    forward role).
    """
    unit = state["allies"][actor]
    if str(unit.get("type", "")).lower().startswith("medivac"):
        return "healer"
    ranges = [r.get("native_weapon_range") for r in state.get("relations", {}).get(actor, {}).values()
              if r.get("native_weapon_range") is not None]
    if not ranges:
        profile = state.get("unit_type_profiles", {}).get(str(unit.get("unit_type_id")), {})
        ranges = [w.get("range") for w in profile.get("weapons", [])
                  if w.get("range") is not None]
    if ranges:
        return "melee" if min(ranges) <= PARAMS["melee_range"] else "ranged"
    return "ranged"  # unknown type: conservative generic role


def connected_components(state, actors=None, link_radius=None):
    """Union-find grouping of the given allies by Euclidean proximity.

    A long single-file chain is one component but is NOT compact; use
    compactness metrics separately before declaring the team concentrated.
    """
    radius = link_radius if link_radius is not None else PARAMS["link_radius"]
    members = actors if actors is not None else living_allies(state)
    positions = {a: state["allies"][a]["position"] for a in members}
    parent = {a: a for a in members}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, a in enumerate(members):
        for b in members[i + 1:]:
            if math.dist(positions[a], positions[b]) <= radius:
                parent[find(a)] = find(b)
    groups = {}
    for a in members:
        groups.setdefault(find(a), []).append(a)
    return sorted(groups.values(), key=lambda g: (-len(g), min(g)))


def centroid(positions):
    points = list(positions)
    if not points:
        return None
    return [sum(p[i] for p in points) / len(points) for i in range(len(points[0]))]


def compactness(state, actors):
    """Explainable cohesion metrics for one set of actors.

    Returns dict with span (max pairwise distance), median/quantile radius
    around the centroid, and the centroid itself. Geometry only.
    """
    positions = [state["allies"][a]["position"] for a in actors]
    if not positions:
        return {"count": 0, "centroid": None, "span": None,
                "median_radius": None, "quantile_radius": None}
    center = centroid(positions)
    distances = sorted(math.dist(p, center) for p in positions)
    span = max((math.dist(p, q) for p in positions for q in positions), default=0.0)
    quantile = distances[max(0, math.ceil(0.9 * len(distances)) - 1)]
    return {"count": len(actors), "centroid": [round(c, 3) for c in center],
            "span": round(span, 3),
            "median_radius": round(distances[len(distances) // 2], 3),
            "quantile_radius": round(quantile, 3)}


def main_group(state, actors=None):
    """Largest proximity component; ties broken by lowest member id."""
    groups = connected_components(state, actors)
    return list(groups[0]) if groups else []


def cohesion_report(state, parameters=None):
    """Full observation-level cohesion snapshot shared by the spatial skills."""
    params = {**PARAMS, **(parameters or {})}
    live = living_allies(state)
    groups = connected_components(state, live, params["link_radius"])
    main = list(groups[0]) if groups else []
    anchor = centroid(state["allies"][a]["position"] for a in main) if main else None
    others = [a for a in live if a not in main]
    isolated = [a for a in others
                if anchor is not None
                and math.dist(state["allies"][a]["position"], anchor) >= params["outlier_distance"]]
    main_metrics = compactness(state, main) if main else compactness(state, [])
    # One long chain is a single component but NOT concentrated: coherence
    # requires BOTH the main-share threshold and a compact main group.
    coherent = bool(live
                    and len(main) / len(live) >= params["main_group_min_share"]
                    and main_metrics["quantile_radius"] is not None
                    and main_metrics["quantile_radius"] <= params["target_radius"])
    return {
        "live_units": live,
        "groups": [list(g) for g in groups],
        "main_group": main,
        "main_share": round(len(main) / len(live), 3) if live else 0.0,
        "anchor": [round(c, 3) for c in anchor] if anchor else None,
        "main_compactness": main_metrics,
        "outliers": others,
        "isolated": isolated,
        "team_compactness": compactness(state, live),
        "coherent": coherent,
    }


def ring_slots(count, spacing, roles=None):
    """Deterministic spaced local goals around an anchor (0,0 offsets).

    Simple expanding-ring layout: the first ring holds up to 6 slots, further
    members go on the next ring. Roles may push melee roles to inner slots.
    """
    slots = []
    ring, placed, ring_capacity = 0, 0, 6
    for index in range(count):
        if placed >= ring_capacity:
            ring += 1
            placed = 0
            ring_capacity = 6 * (ring + 1)
        angle = (2 * math.pi * placed) / min(ring_capacity, max(1, count - index + placed))
        radius = spacing * (ring + 1)
        slots.append((radius * math.cos(angle), radius * math.sin(angle)))
        placed += 1
    if roles:
        # Melee first on the inner slots: stable sort keeps determinism.
        order = sorted(range(count), key=lambda i: (roles[i] != "melee", i))
        slots = [slots[i] for i in order]
    return slots


def best_move_toward(state, actor, target, fixed_reservations=None, parameters=None,
                     previous_action=None):
    """Pick the legal movement candidate whose predicted destination is closest
    to ``target`` with small collision / direction-flip penalties.

    Returns (action_id or None, info). Action 1 (SMAC Stop) is a legal wait
    candidate when already near the target. Only legal snapshot actions are
    returned; predictions are scoring aids, never outputs.
    """
    params = {**PARAMS, **(parameters or {})}
    unit = state["allies"][actor]
    moves = state.get("movement_candidates", {}).get(actor, {})
    if not moves:
        return None, {"reason": "no_legal_moves"}
    origin = unit["position"]
    from .movement_scoring import predicted_move  # local import avoids a cycle at load
    reservations = dict(fixed_reservations or {})
    scored = []
    info = {"candidates": {}}
    for action in moves:
        destination = predicted_move(state, actor, action)["destination"]
        distance = math.dist(destination, target)
        collision = 0.0
        for other, reserved in reservations.items():
            if other == actor:
                continue
            friend = state["allies"].get(other)
            if friend is None or not friend.get("alive"):
                continue
            if friend.get("is_flying") != unit.get("is_flying"):
                continue
            clearance = (unit.get("radius", .5) + friend.get("radius", .5)
                         + params["spacing_factor"] * 0.1)
            gap = math.dist(destination, reserved)
            if gap < clearance:
                collision += (clearance - gap) / clearance
        flip = 0.25 if previous_action is not None and str(action) != str(previous_action) else 0.0
        score = distance + 2.0 * collision + flip
        scored.append((score, int(action)))
        info["candidates"][str(action)] = {"destination": [round(d, 3) for d in destination],
                                           "distance": round(distance, 3),
                                           "collision_cost": round(collision, 3), "score": round(score, 3)}
    if not scored:
        return None, {"reason": "no_legal_moves"}
    scored.sort()
    info["selected_action"] = scored[0][1]
    return scored[0][1], info
