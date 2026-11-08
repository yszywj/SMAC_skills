"""Observable gates for repeated one-step interventions."""
from copy import deepcopy
import math


DEFAULT = {
    "schema_version": 1,
    "kind": "conditional_local_skills",
    "priority": ["retarget", "regroup_step"],
    "actor_cooldown_steps": 2,
    "max_interventions": None,
    "regroup_max_enemy_distance": 12.0,
    "regroup_min_alive": 6,
    "regroup_unknown": "wait",
    "corrections": [],
}


def normalize_policy(raw=None):
    raw = {} if raw is None else raw
    if not isinstance(raw, dict) or set(raw) - set(DEFAULT):
        raise ValueError("Unknown conditional policy fields")
    policy = {**deepcopy(DEFAULT), **deepcopy(raw)}
    if policy["schema_version"] != 1 or policy["kind"] != DEFAULT["kind"]:
        raise ValueError("Expected conditional_local_skills schema 1")
    order = policy["priority"]
    if (not isinstance(order, list) or not order or
            any(x not in ("retarget", "regroup_step") for x in order) or len(set(order)) != len(order)):
        raise ValueError("Priority must contain unique supported execution skills")
    cooldown = policy["actor_cooldown_steps"]
    if type(cooldown) is not int or cooldown < 1:
        raise ValueError("Cooldown must be at least one decision step")
    budget = policy["max_interventions"]
    if budget is not None and (type(budget) is not int or budget < 1):
        raise ValueError("Intervention budget must be null or a positive integer")
    distance = policy["regroup_max_enemy_distance"]
    if type(distance) not in (int, float) or not math.isfinite(distance) or distance < 0:
        raise ValueError("Invalid regroup distance")
    if type(policy["regroup_min_alive"]) is not int or policy["regroup_min_alive"] < 2:
        raise ValueError("Invalid regroup alive count")
    if policy["regroup_unknown"] not in ("wait", "execute"):
        raise ValueError("Unknown contact rule")
    from .corrections import validate_layers
    validate_layers(policy["corrections"])
    return policy


def gate(row, candidate, policy):
    if row["family"] not in policy["priority"]:
        return "catalog_only"
    if candidate is None:
        return row.get("reason", "keep_adaptive")
    if row["family"] == "regroup_step":
        features = candidate["features"]
        if features["ally_alive_count"] < policy["regroup_min_alive"]:
            return "wait_alive_count"
        distance = features["nearest_enemy_before"]
        if distance < 0 and policy["regroup_unknown"] == "wait":
            return "wait_unknown_contact"
        if distance > policy["regroup_max_enemy_distance"]:
            return "wait_distant_contact"
    return "eligible"
