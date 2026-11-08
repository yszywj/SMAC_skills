"""Pure invocation proposals and return-based selection; no executor side effects."""

from copy import deepcopy
import math
import hashlib
import random

from .features import extract_features, infer_races, phase_name
from .schema import load_policy, validate_model, _validate_call
from .descriptors import candidate_features
from .targeting import choose_target


def extract_policy_features(policy, state, memory=None):
    if policy.get('exploration', {}).get('feature_schema') == 3:
        from .choice_features import extract_choice_features
        return extract_choice_features(state, memory)
    return extract_features(state, memory)


def _walk(node, features):
    while "feature" in node:
        name = node["feature"]
        value = features.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"missing or invalid pre-decision feature: {name}")
        node = node["left"] if value <= node["threshold"] else node["right"]
    return node


def initial_call(policy, features):
    """Resolve the optional initial tree; availability is enforced by the caller."""
    policy = load_policy(policy)
    leaf = _walk(policy.get("initial_tree", {"delegate": "legacy"}), features)
    return deepcopy(leaf.get("call", leaf))


def _effective_health(unit):
    value = 0.0
    for key in ("health", "shield"):
        part = unit.get(key, 0)
        if isinstance(part, bool) or not isinstance(part, (int, float)) or not math.isfinite(part):
            return None
        value += max(0.0, part)
    return value


def bind_call(call, state, base_call=None):
    """Resolve semantic target rules to an executable invocation, or None.

    Runtime deployment checks still enforce enabled skills and capabilities.
    No controller or shared memory is mutated while binding a proposal.
    """
    _validate_call(call)
    base_call = base_call or {}
    bound = deepcopy(call)
    if bound.get("operation") == "keep":
        return bound
    bound.setdefault("operation", "start")
    bound.setdefault("actor_scope", "whole_squad")
    for key in ("posture", "max_steps"):
        if bound.get(key) is None:
            bound.pop(key, None)
            if base_call.get(key) is not None:
                bound[key] = deepcopy(base_call[key])
    params = deepcopy(base_call.get("adaptive_params", {})) if bound["skill"] == "adaptive_focus_fire" else {}
    params.update(bound.get("adaptive_params", {}))
    if params or bound["skill"] == "adaptive_focus_fire":
        bound["adaptive_params"] = params
    if "movement_profile" not in bound and bound["skill"] == base_call.get("skill") and "movement_profile" in base_call:
        bound["movement_profile"] = deepcopy(base_call["movement_profile"])
    rule = bound.pop("target_rule", "default")
    if rule != "default":
        target = choose_target(rule, state)
        if target is None:
            return None
        bound["priority_target"] = target
    elif bound["skill"] in ("adaptive_focus_fire", "focus_fire_budget"):
        bound.setdefault("priority_target", base_call.get("priority_target"))
    target = bound.get("priority_target")
    if target is not None:
        enemy = state.get("visible_enemies", {}).get(target)
        health = _effective_health(enemy) if enemy is not None else None
        if enemy is None or not enemy.get("alive", True) or health is None or health <= 0:
            return None
        if not any(state.get("relations", {}).get(actor, {}).get(target, {}).get("smac_attack_available") is True
                   for actor, ally in state.get("allies", {}).items() if ally.get("alive", False)):
            return None
    _validate_call(bound)
    return bound


def candidate_calls(policy, state, base_call, features=None):
    """Bind bounded proposals from the current snapshot, never running a skill.

    A proposal is an available invocation, not proof that its execution changes
    an action or improves reward. The runtime audits those facts separately.
    """
    policy = load_policy(policy)
    features = extract_policy_features(policy, state) if features is None else features
    races = infer_races(state)
    if not races or "unknown" in races or not set(races).issubset(policy["supported_races"]):
        return []
    gate = policy["gate"]
    if phase_name(features) not in gate["phases"] or features.get("game_loop", 0) < gate["min_game_loop"]:
        return []
    if base_call.get("operation", "start") != "start":
        return []  # A live KEEP is not silently restarted by a proposal.
    available = []
    for candidate in policy["candidates"]:
        if phase_name(features) not in candidate.get('phases', tuple(('search', 'approach', 'engage', 'lost_contact'))):
            continue
        allowed = True
        for condition in candidate.get('when', []):
            value = features.get(condition['feature'])
            if (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                    or (value > condition['threshold'] if condition['op'] == 'le' else value < condition['threshold'])):
                allowed = False
                break
        if not allowed:
            continue
        call = bind_call({k: v for k, v in candidate.items() if k not in ('id', 'phases', 'when')}, state, base_call)
        if call is None:
            continue
        # Capability gates for the other existing modes remain conservative;
        # deployment enable flags and phase-specific execution gates are root-owned.
        if call["skill"] == "heal_assign" and features.get("healer_fraction", 0) <= 0:
            continue
        available.append({"id": candidate["id"], "call": call})
    return available


def predict_gains(model, features):
    model = validate_model(model)
    if model['kind'] != 'skill_gain_tree':
        raise ValueError('candidate gain model must be scored with a call descriptor')
    return deepcopy(_walk(model["root"], features)["gains"])


def predict_candidate_gain(model, features, call, base_call=None):
    model = validate_model(model)
    if model['kind'] != 'skill_candidate_gain_tree':
        raise ValueError('expected a scalar candidate gain model')
    return float(_walk(model['root'], candidate_features(features, call, base_call))['gain'])


def predict_vector_estimates(model, features):
    model = validate_model(model)
    if model['kind'] != 'skill_vector_gain_tree':
        raise ValueError('expected independent candidate estimates')
    return deepcopy(_walk(model['root'], features)['estimates'])


def choose(policy, features, available, base_call, *, sampling_key=0):
    """Choose a declared available plan; ties/unsupported evidence keep BASE."""
    policy = load_policy(policy)
    selection = policy["selection"]
    by_id = {row["id"]: row["call"] for row in available}
    declared = [c["id"] for c in policy["candidates"]]
    if len(by_id) != len(available) or set(by_id) - set(declared):
        raise ValueError("available candidate ids differ from policy")
    result = {"candidate_id": "BASE", "call": deepcopy(base_call),
              "predicted_gains": {}, "reason": "base_mode"}
    if not by_id:
        result["reason"] = "no_available_candidate"
        return result
    if selection["mode"] == "base":
        return result
    if selection['mode'] == 'free':
        seed = policy['exploration']['seed']
        key = hashlib.sha256(f'{seed}:{sampling_key}:free'.encode()).hexdigest()
        selected = available[random.Random(key).randrange(len(available))]['id']
        result.update(candidate_id=selected, call=deepcopy(by_id[selected]), reason='free_exploration',
                      sampling_probability=1.0 / len(available))
        return result
    if selection["mode"] == "candidate":
        rank = selection.get('candidate_rank')
        selected = (available[rank]['id'] if rank is not None and rank < len(available)
                    else selection.get('candidate_id'))
        if selected not in by_id:
            result["reason"] = "forced_candidate_unavailable"
            return result
        result.update(candidate_id=selected, call=deepcopy(by_id[selected]), reason="forced_candidate")
        return result
    try:
        if selection['model']['kind'] == 'skill_vector_gain_tree':
            estimates = predict_vector_estimates(selection['model'], features)
            gains = {cid: estimates[cid]['gain'] if estimates[cid]['supported'] else None for cid in by_id}
            result['predicted_gains'] = gains
            result['candidate_support'] = {cid: {key: estimates[cid][key]
                for key in ('n_samples', 'n_groups', 'supported')} for cid in by_id}
            tolerance = selection['model'].get('tie_tolerance', 1e-9)
            choices = [cid for cid in by_id if gains[cid] is not None
                       and gains[cid] > selection['min_gain'] + tolerance]
            if not choices:
                result['reason'] = 'no_positive_supported_gain'
                return result
            best = max(choices, key=lambda cid: gains[cid])
            if any(cid != best and abs(gains[cid]-gains[best]) <= tolerance for cid in choices):
                result['reason'] = 'ambiguous_candidate_gain'
                return result
            result.update(candidate_id=best, call=deepcopy(by_id[best]), reason='supported_candidate_gain')
            return result
        if selection['model']['kind'] == 'skill_candidate_gain_tree':
            gains = {cid: predict_candidate_gain(selection['model'], features, call, base_call)
                     for cid, call in by_id.items()}
        else:
            gains = predict_gains(selection["model"], features)
    except ValueError:
        result["reason"] = "invalid_model_features"
        return result
    result["predicted_gains"] = gains
    choices = [i for i in declared if i in by_id and gains[i] > selection["min_gain"]]
    if not choices:
        result["reason"] = "no_positive_supported_gain"
        return result
    selected = max(choices, key=lambda i: gains[i])
    result.update(candidate_id=selected, call=deepcopy(by_id[selected]), reason="predicted_gain")
    return result
