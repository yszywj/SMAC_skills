"""Pure local interventions. No clocks, memory, environment calls or model I/O.

The caller commits the selected result through the existing manager *before*
its history update. Every proposal lasts one decision step and owns one actor.
Unselected candidates cannot update memory. Legal action menus are mandatory.
"""
from copy import deepcopy
from hashlib import sha256
import json
import math
import re

from skills.current.skills.cover_retreat import detect_retreat_candidates, select_retreat_action
from skills.current.skills.focus_fire import expected_damage

FAMILIES = ("retarget", "retreat_veto", "retreat_step", "regroup_step")
FEATURE_NAMES = (
    "actor_health_ratio", "actor_recent_loss_ratio", "actor_weapon_ready",
    "ally_alive_count", "team_health_ratio", "visible_enemy_count",
    "actor_legal_attack_count", "actor_threat_count", "changed_actor_fraction",
    "baseline_attack", "baseline_retreat", "baseline_recover", "baseline_reposition",
    "candidate_attack", "candidate_move", "target_health_ratio", "target_health_known",
    "target_effective_health", "target_distance", "target_distance_known",
    "nearest_enemy_before", "nearest_enemy_after", "nearest_ally_before",
    "nearest_ally_after", "assigned_damage_before", "assigned_damage_known",
)
DEFAULTS = {"max_candidates": 24, "max_per_family": 8, "regroup_distance": 6.0}
ATTACK_ROLES = frozenset(("attack", "all_in_attack"))


def _id(value, prefix):
    if not isinstance(value, str) or not re.fullmatch(prefix + r"\d+", value):
        raise ValueError("malformed actor/target id")
    return int(value[1:])


def _number(value, default=0.0):
    if value is None:
        return float(default)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("finite numeric observation required")
    return float(value)


def _position(unit):
    p = unit.get("position")
    if not isinstance(p, (list, tuple)) or len(p) < 2:
        return None
    return (_number(p[0]), _number(p[1]))


def _distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _health(unit):
    return _number(unit.get("health")) + _number(unit.get("shield"))


def _maximum(unit):
    return _number(unit.get("max_health")) + _number(unit.get("max_shield"))


def _healer(unit):
    return str(unit.get("type", "")).lower().startswith("medivac")


def _menu(payload, actor):
    menu = payload.get("questions", {}).get(actor, {}).get("criteria")
    if not isinstance(menu, dict):
        return frozenset()
    result = set()
    for key in menu:
        if isinstance(key, bool) or not re.fullmatch(r"\d+", str(key)):
            raise ValueError("malformed action menu")
        result.add(int(key))
    return frozenset(result)


def _attack_targets(payload, actor):
    state = payload["state"]
    menu = _menu(payload, actor)
    return sorted((e for e, r in state.get("relations", {}).get(actor, {}).items()
                   if e in state.get("visible_enemies", {})
                   and state["visible_enemies"][e].get("alive", True)
                   and r.get("smac_attack_available")
                   and 6 + _id(e, "e") in menu), key=lambda e: _id(e, "e"))


def _moves(payload, actor):
    menu = _menu(payload, actor)
    return {str(k): v for k, v in payload["state"].get("movement_candidates", {}).get(actor, {}).items()
            if str(k) in ("2", "3", "4", "5") and int(k) in menu
            and isinstance(v, dict) and _position({"position": v.get("destination")}) is not None}


def _destination(payload, actor, action):
    origin = _position(payload["state"]["allies"][actor])
    if action in (2, 3, 4, 5):
        row = _moves(payload, actor).get(str(action))
        return _position({"position": row["destination"]}) if row else None
    return origin


def effect_key(candidate):
    """Fingerprint committed effect, including role-history semantics, not aliases."""
    value = {"actions": candidate["actions"], "roles": candidate["roles"],
             "duration_steps": candidate.get("duration_steps", 1)}
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def deduplicate_candidates(candidates):
    """First deterministic representative wins; aliases never become extra evidence."""
    unique = {}
    for source in candidates:
        candidate = deepcopy(source)
        key = effect_key(candidate)
        candidate["effect_key"] = key
        candidate.setdefault("aliases", [])
        if key in unique:
            row = unique[key]
            row["aliases"] = sorted(set(row["aliases"] + [candidate["id"]] + candidate["aliases"]))
        else:
            unique[key] = candidate
    return list(unique.values())


def _balanced_family(rows, family, limit):
    """Spread a bounded family menu across actors before adding more bindings.

    Actor rank uses public need; stable IDs only break equal-score ties and are
    not model features. This avoids the first actor consuming a target menu.
    """
    buckets = {}
    for row in rows:
        buckets.setdefault(row["actors"][0], []).append(row)
    def actor_order(actor):
        f = buckets[actor][0]["features"]
        if family in ("retreat_step", "retreat_veto"):
            return (f["actor_health_ratio"], -f["actor_recent_loss_ratio"], _id(actor, "u"))
        if family == "regroup_step":
            return (-f["nearest_ally_before"], _id(actor, "u"))
        return (-f["actor_weapon_ready"], _id(actor, "u"))
    actors = sorted(buckets, key=actor_order)
    for actor in actors:
        buckets[actor].sort(key=lambda row: (row["features"]["target_effective_health"], row["id"]))
    result = []
    for binding in range(max((len(xs) for xs in buckets.values()), default=0)):
        for actor in actors:
            if binding < len(buckets[actor]):
                result.append(buckets[actor][binding])
                if len(result) == limit:
                    return result
    return result


def _features(payload, baseline, actor, action):
    state = payload["state"]
    unit = state["allies"][actor]
    alive = [u for u in state["allies"].values() if u.get("alive")]
    before, after = _position(unit), _destination(payload, actor, action)
    enemy_positions = [p for u in state.get("visible_enemies", {}).values()
                       if (p := _position(u)) is not None]
    friend_positions = [p for a, u in state["allies"].items() if a != actor and u.get("alive")
                        and (p := _position(u)) is not None]
    nearest = lambda p, xs: min((_distance(p, q) for q in xs), default=-1.0) if p else -1.0
    target_id = "e" + str(action - 6) if action >= 6 else None
    target = state.get("visible_enemies", {}).get(target_id, {})
    relation = state.get("relations", {}).get(actor, {}).get(target_id, {})
    maximum = _maximum(unit)
    role = baseline.get("roles", {}).get(actor)
    team_max = sum(_maximum(u) for u in alive)
    target_max = _maximum(target)
    budget = baseline.get("predicted_damage") or {}
    budget_value = budget.get(target_id)
    result = {
        "actor_health_ratio": _health(unit) / maximum if maximum > 0 else 1.0,
        "actor_recent_loss_ratio": (_number(unit.get("health_lost_since_previous_decision"))
                                      + _number(unit.get("shield_lost_since_previous_decision"))) / (maximum or 1.0),
        "actor_weapon_ready": float(bool(unit.get("weapon_ready"))),
        "ally_alive_count": float(len(alive)),
        "team_health_ratio": sum(_health(u) for u in alive) / team_max if team_max > 0 else 1.0,
        "visible_enemy_count": float(len(state.get("visible_enemies", {}))),
        "actor_legal_attack_count": float(len(_attack_targets(payload, actor))),
        "actor_threat_count": float(sum(r.get("in_enemy_weapon_range") is True
                                          for r in state.get("relations", {}).get(actor, {}).values())),
        "changed_actor_fraction": 1.0 / max(1, len(alive)),
        "baseline_attack": float(role in ATTACK_ROLES),
        "baseline_retreat": float(role == "retreat"),
        "baseline_recover": float(role == "recover"),
        "baseline_reposition": float(role == "reposition"),
        "candidate_attack": float(action >= 6), "candidate_move": float(action in (2, 3, 4, 5)),
        "target_health_ratio": _health(target) / target_max if target_max > 0 else 0.0,
        "target_health_known": float(target_max > 0),
        "target_effective_health": _health(target),
        "target_distance": _number(relation.get("distance"), -1.0),
        "target_distance_known": float(relation.get("distance") is not None),
        "nearest_enemy_before": nearest(before, enemy_positions),
        "nearest_enemy_after": nearest(after, enemy_positions),
        "nearest_ally_before": nearest(before, friend_positions),
        "nearest_ally_after": nearest(after, friend_positions),
        "assigned_damage_before": _number(budget_value),
        "assigned_damage_known": float(budget_value is not None),
    }
    return {name: float(result[name]) for name in FEATURE_NAMES}


def _candidate(payload, baseline, actor, action, family, role):
    index = _id(actor, "u")
    if baseline["actions"][index] == action:
        return None
    row = {"id": f"{family}:{actor}:{action}", "family": family, "actors": [actor],
           "actions": {actor: int(action)}, "roles": {actor: role},
           "duration_steps": 1, "aliases": [],
           "features": _features(payload, baseline, actor, action)}
    row["effect_key"] = effect_key(row)
    return row


def generate_candidates(payload, baseline_result, config=None):
    """Generate bounded proposals; absent menus or unsupported actors abstain.

    No skill enable flag changes observation memory. Thresholds affect proposal
    availability only. Safety/benefit is learned from paired real rollouts.
    """
    config = dict(config or {})
    if set(config) - set(DEFAULTS):
        raise ValueError("unknown local candidate option")
    cfg = {**DEFAULTS, **config}
    for key in ("max_candidates", "max_per_family"):
        if type(cfg[key]) is not int or not 1 <= cfg[key] <= 256:
            raise ValueError("candidate limits must be integers in [1,256]")
    if _number(cfg["regroup_distance"]) <= 0:
        raise ValueError("regroup_distance must be positive")
    state = payload["state"]
    proposals = []
    endangered = detect_retreat_candidates(state)
    for actor in sorted(state["allies"], key=lambda a: _id(a, "u")):
        unit = state["allies"][actor]
        index = _id(actor, "u")
        if not unit.get("alive") or _healer(unit) or index >= len(baseline_result["actions"]):
            continue
        role = baseline_result.get("roles", {}).get(actor)
        targets = _attack_targets(payload, actor)
        if role in ATTACK_ROLES or role in ("retreat", "recover"):
            family = "retarget" if role in ATTACK_ROLES else "retreat_veto"
            final_role = role if role in ATTACK_ROLES else "attack"
            for target in targets:
                row = _candidate(payload, baseline_result, actor, 6 + _id(target, "e"), family, final_role)
                if row:
                    proposals.append(row)
        moves = _moves(payload, actor)
        if actor in endangered and role in ATTACK_ROLES | {"reposition"} and moves:
            # Filter against the authoritative mask before scoring; the shared
            # scorer cannot select a masked movement or use unseen positions.
            local_state = {**state, "movement_candidates": {**state.get("movement_candidates", {}), actor: moves}}
            action, _, _ = select_retreat_action(local_state, actor, {actor})
            if action in (2, 3, 4, 5):
                row = _candidate(payload, baseline_result, actor, action, "retreat_step", "retreat")
                if row:
                    proposals.append(row)
        if role == "reposition" and not targets and moves:
            origin = _position(unit)
            friends = [p for a, u in state["allies"].items()
                       if a != actor and u.get("alive") and (p := _position(u)) is not None]
            if origin and friends and min(_distance(origin, p) for p in friends) > cfg["regroup_distance"]:
                center = tuple(sum(p[i] for p in friends) / len(friends) for i in range(2))
                scored = sorted((_distance(_position({"position": entry["destination"]}), center), int(action))
                                for action, entry in moves.items())
                distance, action = scored[0]
                if distance + 1e-9 < _distance(origin, center):
                    row = _candidate(payload, baseline_result, actor, action, "regroup_step", "reposition")
                    if row:
                        proposals.append(row)
    # Round robin avoids exhausting the total cap on one family's many targets.
    rows = deduplicate_candidates(proposals)
    buckets = {family: _balanced_family([r for r in rows if r["family"] == family],
                                        family, cfg["max_per_family"]) for family in FAMILIES}
    result = []
    for i in range(cfg["max_per_family"]):
        for family in FAMILIES:
            if i < len(buckets[family]):
                result.append(buckets[family][i])
                if len(result) == cfg["max_candidates"]:
                    return result
    return result


def _validate_candidate(payload, baseline, candidate):
    if candidate.get("family") not in FAMILIES or type(candidate.get("duration_steps")) is not int or candidate["duration_steps"] != 1:
        raise ValueError("unsupported local skill or duration")
    actors = candidate.get("actors")
    if not isinstance(actors, list) or len(actors) != 1:
        raise ValueError("v1 local skill owns exactly one actor")
    actor = actors[0]
    index = _id(actor, "u")
    state = payload["state"]
    unit = state["allies"].get(actor)
    if not unit or not unit.get("alive") or _healer(unit) or index >= len(baseline["actions"]):
        raise ValueError("unsupported local actor")
    if set(candidate.get("actions", {})) != {actor} or set(candidate.get("roles", {})) != {actor}:
        raise ValueError("local writes exceed declared ownership")
    action = candidate["actions"][actor]
    if type(action) is not int or action not in _menu(payload, actor):
        raise ValueError("local action is outside observed legal menu")
    role = candidate["roles"][actor]
    old_role = baseline.get("roles", {}).get(actor)
    family = candidate["family"]
    if family in ("retarget", "retreat_veto"):
        eligible = old_role in ATTACK_ROLES if family == "retarget" else old_role in ("retreat", "recover")
        if not eligible or action < 6 or "e" + str(action - 6) not in _attack_targets(payload, actor):
            raise ValueError("local attack is not a visible legal family binding")
        expected_role = old_role if family == "retarget" else "attack"
        if role != expected_role:
            raise ValueError("invalid local attack role")
    else:
        if action not in (2, 3, 4, 5) or str(action) not in _moves(payload, actor):
            raise ValueError("local move has no legal candidate geometry")
        if family == "retreat_step" and (old_role not in ATTACK_ROLES | {"reposition"} or role != "retreat"):
            raise ValueError("invalid retreat scope")
        if family == "regroup_step" and (old_role != "reposition" or role != "reposition" or _attack_targets(payload, actor)):
            raise ValueError("regroup cannot overwrite available attacks")
    if baseline["actions"][index] == action:
        raise ValueError("local candidate is a no-op")
    if candidate.get("effect_key") != effect_key(candidate):
        raise ValueError("local candidate fingerprint mismatch")
    return actor, index, action, role


def _rebuild_budget(state, result):
    """Recount ready, known direct-hit estimates from the final actions only.

    Unknown damage is recorded as unknown, not guessed. These values are
    estimates, never delivered damage or a counterfactual reward label.
    """
    assignments, predicted, unknown = {}, {}, []
    for actor, unit in state["allies"].items():
        index = _id(actor, "u")
        if not unit.get("alive") or _healer(unit) or index >= len(result["actions"]):
            continue
        action = result["actions"][index]
        target = "e" + str(action - 6) if action >= 6 else None
        relation = state.get("relations", {}).get(actor, {}).get(target, {})
        if target not in state.get("visible_enemies", {}) or not relation.get("smac_attack_available"):
            continue
        assignments[actor] = target
        predicted.setdefault(target, 0.0)
        if not unit.get("weapon_ready") or relation.get("in_weapon_range") is False:
            continue
        damage = expected_damage(state, actor, target)
        if damage is None:
            unknown.append(actor)
        else:
            predicted[target] += damage
    result["attack_assignments"] = assignments
    result["predicted_damage"] = predicted
    result["immediate_damage"] = deepcopy(predicted)
    result["window_only_damage"] = {}
    result["estimated_overkill"] = {e: max(0.0, damage - _health(state["visible_enemies"][e]))
                                   for e, damage in predicted.items()}
    result["damage_model"] = {"kind": "local_final_ready_known_direct_hit", "unknown_actors": unknown,
                              "window_prediction_replaced": True, "observed_damage": False}


def apply_candidate(payload, baseline_result, candidate):
    """Return the scoped final proposal; normal manager commit updates history."""
    actor, index, action, role = _validate_candidate(payload, baseline_result, candidate)
    result = deepcopy(baseline_result)
    result["actions"][index] = action
    result["roles"][actor] = role
    for key in ("approach_state_updates", "engagement_state_updates", "movement_diagnostics",
                "approach_diagnostics", "safe_reentry_diagnostics", "critical_diagnostics",
                "critical_escape_diagnostics", "planned_reentry_diagnostics", "engagement_diagnostics",
                "reentry_decisions", "retreat_actions", "retreat_reasons"):
        if isinstance(result.get(key), dict):
            result[key].pop(actor, None)
    # Existing value diagnostics describe the old proposal, so retain them
    # only in explicitly labelled baseline provenance instead of effective stats.
    if "unit_value_diagnostics" in result:
        result["baseline_unit_value_diagnostics"] = result["unit_value_diagnostics"]
        result["unit_value_diagnostics"] = {}
    roles = result["roles"]
    order = lambda xs: sorted(xs, key=lambda a: _id(a, "u"))
    result["attack_units"] = order(a for a, r in roles.items() if r in ATTACK_ROLES)
    result["retreat_units"] = order(a for a, r in roles.items() if r == "retreat")
    result["support_units"] = order(a for a, r in roles.items() if r == "support")
    result["reengaged"] = [a for a in result.get("reengaged", []) if a != actor]
    if baseline_result["roles"][actor] in ("retreat", "recover") and role in ATTACK_ROLES:
        result["reengaged"].append(actor)
    if role == "retreat":
        result.setdefault("retreat_actions", {})[actor] = action
        result.setdefault("retreat_reasons", {})[actor] = "selected_local_retreat_step"
    _rebuild_budget(payload["state"], result)
    result["phase"] = "mixed" if result["retreat_units"] else ("all_in" if result.get("all_in") else "engage")
    result["local_patch"] = {"candidate_id": candidate["id"], "family": candidate["family"],
                             "actors": [actor], "effect_key": candidate["effect_key"],
                             "baseline_action": baseline_result["actions"][index],
                             "final_action": action, "duration_steps": 1}
    result["local_changed_actors"] = [actor]
    return result
