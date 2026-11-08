"""JSON contracts for observable-state skill selection and return trees."""

import hashlib
import json
import math
import re
from pathlib import Path

from .features import FEATURE_NAMES

SKILLS = {"adaptive_focus_fire", "focus_fire_budget", "cover_retreat", "heal_assign",
          "regroup", "cohesive_search"}
PHASES = {"search", "approach", "engage", "lost_contact"}
TARGET_RULES = {"default", "lowest_effective_health", "highest_threat", "easiest_focus"}
POSTURES = {"aggressive", "balanced", "defensive"}


def _finite(value, name, minimum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _integer(value, name, low, high):
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")


def _object(value, name):
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")


def _identifier(value):
    return (isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,64}", value)
            and value.lower() not in {"base", "model", "selector"})


def _json_copy(value):
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (ValueError, TypeError, OverflowError, RecursionError) as exc:
        raise ValueError("policy/model must be finite acyclic JSON") from exc


def _validate_call(call, partial=False):
    _object(call, "call")
    allowed = {"operation", "skill", "actor_scope", "target_rule", "priority_target", "posture",
               "adaptive_params", "max_steps", "movement_profile"}
    if set(call) - allowed:
        raise ValueError(f"unsupported call fields: {sorted(set(call) - allowed)}")
    if call.get("operation", "start") not in ("start", "keep"):
        raise ValueError("call operation must be start or keep")
    if call.get("operation") == "keep":
        if set(call) - {"operation"}:
            raise ValueError("keep cannot silently change invocation parameters")
        return
    if not isinstance(call.get("skill"), str) or call["skill"] not in SKILLS:
        raise ValueError("unknown skill")
    if call.get("actor_scope", "whole_squad") != "whole_squad":
        raise ValueError("only whole_squad actor_scope is implemented")
    if not isinstance(call.get("target_rule", "default"), str) or call.get("target_rule", "default") not in TARGET_RULES:
        raise ValueError("unknown target_rule")
    if call.get("target_rule", "default") != "default" and call["skill"] not in ("adaptive_focus_fire", "focus_fire_budget"):
        raise ValueError("target rules require an attack allocator")
    if call.get("posture") is not None and (not isinstance(call["posture"], str) or call["posture"] not in POSTURES):
        raise ValueError("unknown posture")
    if call.get("priority_target") is not None:
        if not isinstance(call["priority_target"], str) or not re.fullmatch(r"e\d+", call["priority_target"]):
            raise ValueError("priority_target must be an e<number> identifier or null")
        if call["skill"] not in ("adaptive_focus_fire", "focus_fire_budget"):
            raise ValueError("priority_target requires an attack allocator")
    if call.get("max_steps") is not None:
        _integer(call["max_steps"], "max_steps", 1, 64)
    params = call.get("adaptive_params", {})
    _object(params, "adaptive_params")
    for name, value in params.items():
        if name in ("max_retreat_steps", "max_recover_steps", "max_retreat_cycles", "stall_steps"):
            bounds = {"max_retreat_steps": (1, 8), "max_recover_steps": (0, 8),
                      "max_retreat_cycles": (1, 10), "stall_steps": (1, 30)}
            _integer(value, name, *bounds[name])
        elif name in ("all_in_team_ratio", "critical_escape_ratio"):
            low = .05 if name == "all_in_team_ratio" else 0.0
            if not low <= _finite(value, name) <= 1:
                raise ValueError(f"{name} must be in [{low},1]")
        elif name in ("disable_retreat", "disable_all_in"):
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be boolean")
        else:
            raise ValueError(f"unsupported invocation parameter: {name}")
    if params and call["skill"] != "adaptive_focus_fire":
        raise ValueError("adaptive_params require adaptive_focus_fire")
    if "movement_profile" in call:
        movement = call["movement_profile"]
        _object(movement, "movement_profile")
        if call["skill"] != "adaptive_focus_fire":
            raise ValueError("movement_profile requires adaptive_focus_fire")
        if set(movement) - {"spacing", "layout"}:
            raise ValueError("unsupported movement_profile fields")
        if movement.get("spacing", "normal") not in ("compact", "normal", "spread"):
            raise ValueError("unknown positioning_spacing")
        if movement.get("layout", "free") not in ("free", "front_back"):
            raise ValueError("unknown positioning_layout")


def _walk_tree(node, leaf_validator, depth=0, counter=None, feature_names=FEATURE_NAMES):
    counter = [0] if counter is None else counter
    counter[0] += 1
    if depth > 16 or counter[0] > 511:
        raise ValueError("tree exceeds maximum depth/node count")
    _object(node, "tree node")
    if "feature" in node:
        if node["feature"] not in feature_names:
            raise ValueError(f"unknown pre-decision feature: {node['feature']}")
        _finite(node.get("threshold"), "threshold")
        if "left" not in node or "right" not in node or any(k in node for k in ("gains", "gain", "estimates", "call", "delegate", "skill")):
            raise ValueError("split needs exactly two branches and no leaf output")
        _walk_tree(node["left"], leaf_validator, depth + 1, counter, feature_names)
        _walk_tree(node["right"], leaf_validator, depth + 1, counter, feature_names)
    else:
        if any(k in node for k in ("left", "right", "threshold")):
            raise ValueError("malformed leaf")
        leaf_validator(node)


def validate_model(value, candidate_ids=None):
    model = _json_copy(value)
    _object(model, "model")
    if type(model.get('schema_version')) is int and model['schema_version'] == 3 and model.get('kind') == 'skill_vector_gain_tree':
        from .choice_features import CHOICE_FEATURE_NAMES
        names = model.get('feature_names')
        if names not in (list(FEATURE_NAMES), list(CHOICE_FEATURE_NAMES)):
            raise ValueError('vector model requires an exact public state feature schema')
        ids = model.get('candidate_ids')
        if not isinstance(ids, list) or not ids or any(not _identifier(i) for i in ids) or len(ids) != len(set(ids)):
            raise ValueError('vector candidate_ids must be unique declared identifiers')
        if candidate_ids is not None and ids != list(candidate_ids):
            raise ValueError('vector candidate_ids differ from policy declaration order')
        minimum = model.get('min_candidate_groups')
        _integer(minimum, 'min_candidate_groups', 1, 2**31-1)
        _finite(model.setdefault('tie_tolerance', 1e-9), 'tie_tolerance', 0)
        def vector_leaf(node):
            if any(key in node for key in ('gain', 'gains', 'call', 'delegate', 'skill')):
                raise ValueError('vector leaf must use independent candidate estimates')
            _integer(node.get('n_samples'), 'vector leaf n_samples', 1, 2**31-1)
            _integer(node.get('n_groups'), 'vector leaf n_groups', 1, node['n_samples'])
            estimates = node.get('estimates')
            if not isinstance(estimates, dict) or set(estimates) != set(ids):
                raise ValueError('vector estimates must cover every declared candidate; missing observations use null')
            groups, samples = set(), 0
            for cid, estimate in estimates.items():
                _object(estimate, 'candidate estimate')
                _integer(estimate.get('n_samples'), 'n_samples', 0, 2**31-1)
                _integer(estimate.get('n_groups'), 'n_groups', 0, estimate['n_samples'])
                gids = estimate.get('group_ids')
                if (not isinstance(gids, list) or any(not isinstance(g, str) or not g for g in gids)
                        or len(gids) != len(set(gids)) or len(gids) != estimate['n_groups']):
                    raise ValueError('candidate group support metadata mismatch')
                if bool(estimate['n_samples']) != bool(estimate['n_groups']):
                    raise ValueError('observed candidate samples require scenario groups')
                if estimate['n_samples']:
                    _finite(estimate.get('gain'), 'candidate gain')
                elif estimate.get('gain') is not None:
                    raise ValueError('unobserved candidate gain must be null, never zero')
                if type(estimate.get('supported')) is not bool or estimate['supported'] != (estimate['n_groups'] >= minimum):
                    raise ValueError('candidate support must use its own independent scenario count')
                groups.update(gids)
                samples += estimate['n_samples']
            if node.get('n_samples') != samples or node.get('n_groups') != len(groups):
                raise ValueError('vector leaf totals differ from candidate evidence')
        _walk_tree(model.get('root'), vector_leaf, feature_names=names)
        return model
    if type(model.get('schema_version')) is int and model['schema_version'] == 2 and model.get('kind') == 'skill_candidate_gain_tree':
        from .descriptors import MODEL_FEATURE_NAMES
        names = model.get('feature_names')
        if names != list(MODEL_FEATURE_NAMES):
            raise ValueError('candidate gain model requires the public state and call feature schema')
        def scalar_leaf(node):
            if any(key in node for key in ('gains', 'call', 'delegate', 'skill')):
                raise ValueError('scalar model leaf must output gain only')
            _finite(node.get('gain'), 'gain')
        _walk_tree(model.get('root'), scalar_leaf, feature_names=MODEL_FEATURE_NAMES)
        return model
    if type(model.get("schema_version")) is not int or model["schema_version"] != 1 or model.get("kind") != "skill_gain_tree":
        raise ValueError("unsupported gain model schema")
    ids = model.get("candidate_ids")
    if not isinstance(ids, list) or not ids or any(not _identifier(i) for i in ids) or len(ids) != len(set(ids)):
        raise ValueError("model candidate_ids must be nonempty unique strings")
    if candidate_ids is not None and set(ids) != set(candidate_ids):
        raise ValueError("model candidate_ids differ from policy candidates")
    names = model.get("feature_names", list(FEATURE_NAMES))
    if not isinstance(names, list) or any(n not in FEATURE_NAMES for n in names) or len(names) != len(set(names)):
        raise ValueError("invalid model feature_names")
    def leaf(node):
        gains = node.get("gains")
        if not isinstance(gains, dict) or set(gains) != set(ids):
            raise ValueError("leaf gains must cover every candidate exactly once")
        for key, gain in gains.items():
            _finite(gain, f"gain[{key}]")
    _walk_tree(model.get("root"), leaf)
    def check_names(node):
        if "feature" in node:
            if node["feature"] not in names:
                raise ValueError("split feature missing from feature_names")
            check_names(node["left"])
            check_names(node["right"])
    check_names(model["root"])
    return model


def validate_policy(value):
    policy = _json_copy(value)
    _object(policy, "policy")
    if type(policy.get("schema_version")) is not int or policy["schema_version"] != 1 or policy.get("kind") != "skill_return_policy":
        raise ValueError("unsupported skill policy schema")
    allowed = {"schema_version", "kind", "description", "name", "version", "supported_races",
               "gate", "candidates", "selection", "initial_tree", "metadata", "exploration"}
    if set(policy) - allowed:
        raise ValueError(f"unknown policy fields: {sorted(set(policy) - allowed)}")
    races = policy.get("supported_races", ["protoss"])
    if not isinstance(races, list) or not races or any(not isinstance(r, str) or r not in ("protoss", "terran", "zerg") for r in races) or len(races) != len(set(races)):
        raise ValueError("supported_races must list supported roster races")
    policy["supported_races"] = races
    exploration = policy.get('exploration', {})
    _object(exploration, 'exploration')
    if set(exploration) - {'enabled', 'seed', 'feature_schema'}:
        raise ValueError('unknown exploration fields')
    if 'enabled' in exploration and not isinstance(exploration['enabled'], bool):
        raise ValueError('exploration.enabled must be boolean')
    expanded = exploration.get('enabled', False)
    if exploration:
        _integer(exploration.setdefault('seed', 0), 'exploration.seed', 0, 2**31 - 1)
    feature_names = FEATURE_NAMES
    if 'feature_schema' in exploration:
        if not expanded or type(exploration['feature_schema']) is not int or exploration['feature_schema'] != 3:
            raise ValueError('feature_schema must be 3 and requires expanded exploration')
        from .choice_features import CHOICE_FEATURE_NAMES
        feature_names = CHOICE_FEATURE_NAMES
    gate = policy.setdefault("gate", {"phases": ["engage"], "min_game_loop": 0})
    _object(gate, "gate")
    if set(gate) - {"phases", "min_game_loop", "effect_check"}:
        raise ValueError("unknown gate fields")
    if gate.setdefault("effect_check", "first_action") not in ("first_action", "call_config"):
        raise ValueError("effect_check must be first_action or call_config")
    phases = gate.setdefault("phases", ["engage"])
    if not isinstance(phases, list) or not phases or any(not isinstance(p, str) or p not in PHASES for p in phases) or len(phases) != len(set(phases)):
        raise ValueError("invalid gate phases")
    _integer(gate.setdefault("min_game_loop", 0), "min_game_loop", 0, 2**31 - 1)
    candidates = policy.get("candidates")
    if not isinstance(candidates, list) or not 1 <= len(candidates) <= 16:
        raise ValueError("policy requires 1..16 candidate configurations")
    ids = []
    for candidate in candidates:
        _object(candidate, "candidate")
        identifier = candidate.get("id")
        if not _identifier(identifier) or identifier in ids:
            raise ValueError("candidate ids must be unique nonempty strings other than BASE")
        ids.append(identifier)
        if candidate.get("operation", "start") != "start":
            raise ValueError("candidate must start a declared invocation")
        if 'phases' in candidate:
            phases = candidate['phases']
            if (not expanded or not isinstance(phases, list) or not phases
                    or any(not isinstance(p, str) or p not in PHASES for p in phases)
                    or len(phases) != len(set(phases))):
                raise ValueError('candidate phases require unique expanded phases')
        if 'when' in candidate:
            conditions = candidate['when']
            if not expanded or not isinstance(conditions, list) or not 1 <= len(conditions) <= 16:
                raise ValueError('candidate when requires 1..16 public conditions')
            for condition in conditions:
                _object(condition, 'candidate condition')
                if (set(condition) != {'feature', 'op', 'threshold'}
                        or condition['feature'] not in feature_names or condition['op'] not in ('le', 'ge')):
                    raise ValueError('invalid candidate pre-decision condition')
                _finite(condition['threshold'], 'candidate threshold')
        _validate_call({k: v for k, v in candidate.items() if k not in ('id', 'phases', 'when')}, partial=True)
    selection = policy.setdefault("selection", {"mode": "base", "max_interventions_per_episode": 1})
    _object(selection, "selection")
    if set(selection) - {"mode", "candidate_id", "max_interventions_per_episode", "model", "min_gain",
                         "anchor_call_number", "candidate_rank"}:
        raise ValueError("unknown selection fields")
    if selection.get("mode") not in (("base", "candidate", "model", "free") if expanded else ("base", "candidate", "model")):
        raise ValueError("unknown selection mode")
    _integer(selection.setdefault("max_interventions_per_episode", 1), "max_interventions_per_episode", 1, 64 if expanded else 1)
    if 'anchor_call_number' in selection:
        if not expanded:
            raise ValueError('anchor_call_number requires expanded exploration')
        _integer(selection['anchor_call_number'], 'anchor_call_number', 1, 2**31 - 1)
    if 'candidate_rank' in selection:
        if not expanded or selection['mode'] != 'candidate' or 'candidate_id' in selection:
            raise ValueError('candidate_rank requires expanded candidate mode without candidate_id')
        _integer(selection['candidate_rank'], 'candidate_rank', 0, len(ids) - 1)
    _finite(selection.setdefault("min_gain", 0.0), "min_gain", 0)
    if selection["mode"] == "candidate" and 'candidate_rank' not in selection and selection.get("candidate_id") not in ids:
        raise ValueError("forced candidate_id is not declared")
    if "model" in selection:
        selection["model"] = validate_model(selection["model"], ids)
        if selection['model']['schema_version'] in (2, 3) and not expanded:
            raise ValueError('candidate gain model requires expanded exploration')
        if selection['model']['schema_version'] == 3 and selection['model']['feature_names'] != list(feature_names):
            raise ValueError('vector model must match the active policy state feature schema')
    elif selection["mode"] == "model":
        raise ValueError("model selection requires a model")
    if "initial_tree" in policy:
        def initial_leaf(node):
            if "delegate" in node:
                if node != {"delegate": "legacy"}:
                    raise ValueError("only legacy delegation is supported")
            elif "call" in node:
                if set(node) != {"call"}:
                    raise ValueError("initial call leaf has unexpected fields")
                _validate_call(node["call"])
                if node["call"].get("operation") == "keep":
                    raise ValueError("initial tree is evaluated at an inactive boundary and cannot KEEP")
            else:
                _validate_call(node)
                if node.get("operation") == "keep":
                    raise ValueError("initial tree is evaluated at an inactive boundary and cannot KEEP")
        _walk_tree(policy["initial_tree"], initial_leaf, feature_names=feature_names)
    return policy


def load_policy(path_or_dict):
    if isinstance(path_or_dict, dict):
        return validate_policy(path_or_dict)
    try:
        with Path(path_or_dict).open(encoding="utf-8-sig") as source:
            return validate_policy(json.load(source))
    except (OSError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load skill policy: {exc}") from exc


def policy_hash(policy):
    canonical = json.dumps(load_policy(policy), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
