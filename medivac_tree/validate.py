"""Validation for a closed, data-only binary tree language (never eval)."""

import math

INTENTS = frozenset({"NOOP", "RETREAT", "HEAL_RESCUE", "KEEP_HEAL", "HEAL_CURRENT",
                     "HEAL_NEW", "APPROACH", "SUPPORT", "STOP"})
FEATURES = frozenset({"alive", "endangered", "can_retreat", "has_rescue", "can_keep",
                      "has_current", "has_new", "has_approach", "has_support", "has_support_move", "can_stop",
                      "health_ratio", "energy", "energy_known", "threat_count",
                      "candidate_count", "keep_stalled", "keep_unconfirmed",
                      "approach_windows", "rescue_cooldown_ready"})
DEFAULT_PARAMETERS = {
    "danger_hp_ratio": .35, "threat_distance": 8.0,
    "stable_hp_ratio": .55, "stable_loss_ratio": .10,
    "critical_hp_ratio": .25, "rapid_loss_ratio": .20, "rapid_loss_hp_ratio": .50,
    "rescue_cooldown_windows": 3, "keep_stall_windows": 2,
    "keep_max_unconfirmed_windows": 4, "energy_drop": .05,
    "approach_min_progress": .1, "approach_max_windows": 4,
    "approach_cooldown_windows": 1,
}
OPS = frozenset({"<=", ">=", "<", ">", "==", "!="})


def validate_tree(document):
    if not isinstance(document, dict) or type(document.get("schema_version")) is not int or document.get("schema_version") != 1:
        raise ValueError("tree schema_version must be 1")
    if set(document) - {"schema_version", "feature_schema_version", "version", "description", "parameters", "root"}:
        raise ValueError("unknown top-level tree fields (all nodes must be reachable from root)")
    if not isinstance(document.get("version"), str) or not document["version"]:
        raise ValueError("tree version must be a nonempty string")
    if type(document.get("feature_schema_version", 1)) is not int or document.get("feature_schema_version", 1) != 1:
        raise ValueError("unsupported feature schema")
    parameters = document.get("parameters", {})
    if not isinstance(parameters, dict) or set(parameters) - set(DEFAULT_PARAMETERS):
        raise ValueError("unknown tree parameters")
    for key, value in parameters.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"invalid parameter {key}")
        if key.endswith("windows") and (not isinstance(value, int) or value < 1):
            raise ValueError(f"{key} must be a positive integer")
        if key.endswith("ratio") and value > 1:
            raise ValueError(f"{key} must be within [0,1]")
    seen = set()

    def visit(node, depth=0):
        if depth > 32 or len(seen) >= 255 or not isinstance(node, dict):
            raise ValueError("tree must have at most 255 nodes and depth 32")
        identifier = node.get("id")
        if not isinstance(identifier, str) or not identifier or identifier in seen:
            raise ValueError("tree node IDs must be nonempty and unique")
        seen.add(identifier)
        if "intent" in node:
            if set(node) != {"id", "intent"} or node["intent"] not in INTENTS:
                raise ValueError(f"invalid leaf {identifier}")
            return
        if set(node) != {"id", "feature", "op", "threshold", "yes", "no"}:
            raise ValueError(f"invalid split fields at {identifier}")
        if node["feature"] not in FEATURES or node["op"] not in OPS:
            raise ValueError(f"unknown feature/operator at {identifier}")
        value = node["threshold"]
        if not isinstance(value, (bool, int, float)) or not math.isfinite(value):
            raise ValueError(f"invalid threshold at {identifier}")
        visit(node["yes"], depth + 1)
        visit(node["no"], depth + 1)

    visit(document.get("root"))
    return {"node_count": len(seen), "parameters": {**DEFAULT_PARAMETERS, **parameters}}


if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", default=str(Path(__file__).with_name("tree_v0.json")))
    args = parser.parse_args()
    document = json.loads(Path(args.path).read_text(encoding="utf-8"))
    if isinstance(document, dict) and document.get("policy_kind") == "paired_return_selector":
        from .selector import validate_policy
        result = validate_policy(document)
    else:
        result = validate_tree(document)
    print(json.dumps(result, indent=2))
