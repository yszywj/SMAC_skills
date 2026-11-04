"""Pure proposals and observed features for one paired medical intervention.

The existing binding implementation is deliberately frozen. A proposal neither
reserves its patient nor advances any history; only the chosen action commits.
"""

from copy import deepcopy
import math

from .facts import exposure, hp_ratio, loss_ratio, moves, position
from .idle_recovery import evaluate as evaluate_idle, observed_context, validate_event
from .validate import FEATURES, INTENTS

FEATURE_SCHEMA_VERSION = 2
FEATURE_NAMES = tuple(sorted(FEATURES)) + (
    "base_target_has_target", "base_target_hp_ratio", "base_target_loss_ratio", "base_target_distance",
    "candidate_target_has_target", "candidate_target_hp_ratio", "candidate_target_loss_ratio", "candidate_target_distance",
) + tuple("base_intent_" + intent for intent in sorted(INTENTS)) + (
    "stop_windows", "wounded_unreserved_count", "nearest_unreserved_gap", "nearest_unreserved_gap_known",
    "current_exposure", "candidate_next_gap", "candidate_progress", "candidate_exposure_delta",
    "candidate_movement_known",
)


def validate_candidate(config):
    """Return a canonical gate, rejecting unknown or ambiguous settings."""
    basic = {"intent", "base_intents", "min_game_loop"}
    if not isinstance(config, dict) or set(config) not in (basic, basic | {"generator", "event"}):
        raise ValueError("candidate requires intent, base_intents and min_game_loop")
    if not isinstance(config["intent"], str) or config["intent"] not in INTENTS:
        raise ValueError("unknown candidate intent")
    bases = config["base_intents"]
    if not isinstance(bases, list) or not bases or any(not isinstance(i, str) or i not in INTENTS for i in bases):
        raise ValueError("candidate base_intents must be a nonempty list of known intents")
    if len(set(bases)) != len(bases):
        raise ValueError("duplicate candidate base_intents")
    loop = config["min_game_loop"]
    if type(loop) is not int or loop < 0:
        raise ValueError("candidate min_game_loop must be a nonnegative integer")
    result = {"intent": config["intent"], "base_intents": sorted(bases), "min_game_loop": loop}
    if "generator" in config:
        if config["generator"] != "idle_recovery" or config["intent"] != "APPROACH" or bases != ["STOP"]:
            raise ValueError("idle_recovery requires intent APPROACH and base_intents ['STOP']")
        result.update(generator="idle_recovery", event=validate_event(config["event"]))
    return result


def evaluate_candidate(config, bindings, base_intent, base_binding, game_loop, *, context=None):
    if config.get("generator") == "idle_recovery":
        if context is None:
            raise ValueError("idle_recovery needs observed candidate context")
        return evaluate_idle(config, context["state"], context["actor"], context["legal"], context["old"],
                             context["reservations"], context["threat_rows"], base_intent, base_binding,
                             context["stop_windows"], context.get("observed"))
    return propose_candidate(config, bindings, base_intent, base_binding, game_loop), None


def propose_candidate(config, bindings, base_intent, base_binding, game_loop, *, context=None):
    """Return a copy of an existing legal binding at an eligible opportunity."""
    if config.get("generator") == "idle_recovery":
        return evaluate_candidate(config, bindings, base_intent, base_binding, game_loop, context=context)[0]
    if game_loop < config["min_game_loop"] or base_intent not in config["base_intents"]:
        return None
    binding = bindings.get(config["intent"])
    if binding is None or binding["action"] == base_binding["action"]:
        return None
    return {"intent": config["intent"], "binding": deepcopy(binding)}


def build_features(state, actor, facts, base_intent, base_binding, candidate_binding, *, context=None):
    """Fixed numeric schema, containing only information observed pre-action."""
    values = {name: float(facts[name]) for name in sorted(FEATURES)}
    origin = position(state["allies"][actor])
    for prefix, binding in (("base_target", base_binding), ("candidate_target", candidate_binding)):
        target = binding.get("target")
        patient = state["allies"].get(target) if target is not None else None
        values[prefix + "_has_target"] = float(patient is not None)
        values[prefix + "_hp_ratio"] = hp_ratio(patient) if patient is not None else 0.
        values[prefix + "_loss_ratio"] = loss_ratio(patient) if patient is not None else 0.
        values[prefix + "_distance"] = math.dist(origin, position(patient)) if patient is not None else 0.
    values.update({"base_intent_" + intent: float(base_intent == intent) for intent in sorted(INTENTS)})
    context = context or {}
    rows = context.get("threat_rows", [])
    observed = context.get("observed") or observed_context(state, actor, context.get("reservations", {}), rows)
    values.update({"stop_windows": float(context.get("stop_windows", 0)),
                   **{name: float(observed[name]) for name in
                      ("wounded_unreserved_count", "nearest_unreserved_gap", "nearest_unreserved_gap_known", "current_exposure")},
                   "candidate_next_gap": 0., "candidate_progress": 0., "candidate_exposure_delta": 0.,
                   "candidate_movement_known": 0.})
    target = candidate_binding.get("target")
    legal = context.get("legal", state.get("legal_actions", {}).get(actor, ()))
    destination = moves(state, actor, legal).get(candidate_binding["action"])
    if target in state["allies"] and destination is not None:
        gap = math.dist(origin, position(state["allies"][target]))
        next_gap = math.dist(destination, position(state["allies"][target]))
        values.update(candidate_next_gap=next_gap, candidate_progress=gap - next_gap,
                      candidate_exposure_delta=float(exposure(rows, destination) - observed["current_exposure"]),
                      candidate_movement_known=1.)
    if any(not math.isfinite(value) for value in values.values()):
        raise ValueError("return selector features must be finite")
    return values
