"""Observed, single-step approach proposals after repeated committed STOPs.

These defaults are explicit development hypotheses, not fitted parameters or a
safety guarantee. A proposal predicts geometric progress only; actual movement,
healing and return must be measured after submitting the selected command.
"""

import math

from .facts import (actor_index, exposure, finite, heal_candidates, hp_ratio,
                    is_medivac, legal_actions, loss_ratio, moves, position, wounded_patient)

DEFAULT_EVENT = {
    "min_stop_windows": 3, "min_energy": 5, "min_health_ratio": .6,
    "max_target_distance": 12, "min_progress": .1, "max_extra_exposure": 1,
}


def validate_event(event):
    if not isinstance(event, dict) or set(event) - set(DEFAULT_EVENT):
        raise ValueError("unknown idle_recovery event fields")
    result = {**DEFAULT_EVENT, **event}
    for name, value in result.items():
        if not finite(value):
            raise ValueError(f"idle_recovery {name} must be finite")
    if type(result["min_stop_windows"]) is not int or result["min_stop_windows"] < 1:
        raise ValueError("min_stop_windows must be a positive integer")
    if result["min_energy"] < 1 or not 0 <= result["min_health_ratio"] <= 1:
        raise ValueError("min_energy must be at least 1 and min_health_ratio within [0,1]")
    if result["max_target_distance"] <= 0 or result["min_progress"] <= 0:
        raise ValueError("max_target_distance and min_progress must be positive")
    if type(result["max_extra_exposure"]) is not int or result["max_extra_exposure"] < 0:
        raise ValueError("max_extra_exposure must be a nonnegative integer")
    return result


def has_heal_order(unit):
    return any(order.get("ability_id") == 386 for order in (unit.get("orders") or []))


def stop_windows(state, actor, old, base_intent, base_binding, event):
    """Prospective STOP count: two submitted stops plus current BASE_STOP = 3.

The caller commits this count only if STOP is the selected final command.
An unavailable/noncontiguous prior record starts a new count rather than
claiming an earlier STOP. Active healing and insufficient energy break it.
"""
    unit = state["allies"][actor]
    if (not unit.get("alive") or not finite(unit.get("energy"))
            or unit["energy"] < event["min_energy"] or has_heal_order(unit)
            or base_intent != "STOP" or base_binding["action"] != 1):
        return 0
    contiguous = old.get("tag") == unit.get("tag") and old.get("loop") == state["game_loop"] - state.get("step_mul", 8)
    previous = old.get("idle_stop_windows", 0)
    continuing = (contiguous and old.get("last_action") == 1 and old.get("last_intent") == "STOP"
                  and type(previous) is int and previous > 0)
    return previous + 1 if continuing else 1


def observed_context(state, actor, reservations, threat_rows):
    """Observable patient availability; reservations are read without changes."""
    origin = position(state["allies"][actor])
    patients, reserved = [], []
    for target, patient in state["allies"].items():
        if not wounded_patient(patient):
            continue
        row = {"target": target, "gap": math.dist(origin, position(patient)),
               "hp_ratio": hp_ratio(patient), "loss_ratio": loss_ratio(patient)}
        owner = reservations.get(target)
        if owner is not None and owner != actor:
            reserved.append({**row, "reserved_by": owner})
        else:
            patients.append(row)
    patients.sort(key=lambda row: (row["gap"], actor_index(row["target"])))
    return {"patients": patients, "reserved_patients": reserved,
            "wounded_unreserved_count": len(patients),
            "nearest_unreserved_gap": patients[0]["gap"] if patients else 0.,
            "nearest_unreserved_gap_known": bool(patients),
            "current_exposure": exposure(threat_rows, origin)}


def other_heal_coverage(state, actor):
    """Current legal direct-heal options, not promises of actual treatment.

An idle, distant medic must not reserve a patient another living medic can
already heal this step. Unknown action availability is handled conservatively
instead of assuming that the other medic has no heal option.
"""
    coverage, uncertain = {}, []
    for other, unit in sorted(state["allies"].items(), key=lambda row: actor_index(row[0])):
        if other == actor or not unit.get("alive") or not is_medivac(unit):
            continue
        if not finite(unit.get("energy")):
            uncertain.append({"actor": other, "reason": "unknown_energy"})
            continue
        if unit["energy"] < 1:
            continue
        try:
            allowed = legal_actions(state, other)
            if not isinstance(state.get("heal_candidates", {}).get(other), dict):
                raise ValueError("explicit heal_candidates is required")
            candidates = heal_candidates(state, other, allowed)
        except (ValueError, TypeError, KeyError) as error:
            uncertain.append({"actor": other, "reason": "unknown_heal_legality", "detail": str(error)})
            continue
        for target, action in candidates.items():
            coverage.setdefault(target, []).append({"actor": other, "action": action})
    return coverage, uncertain


def evaluate(config, state, actor, legal, old, reservations, threat_rows,
             base_intent, base_binding, windows, context=None):
    """Return (proposal or None, audit) without writing memory or reservations."""
    event = config["event"]
    unit = state["allies"][actor]
    context = observed_context(state, actor, reservations, threat_rows) if context is None else context
    audit = {"generator": "idle_recovery", "eligible": False, "stop_windows": windows,
             "wounded_unreserved_count": context["wounded_unreserved_count"],
             "nearest_unreserved_gap": context["nearest_unreserved_gap"],
             "nearest_unreserved_gap_known": context["nearest_unreserved_gap_known"],
             "current_exposure": context["current_exposure"],
             "reserved_patients": context["reserved_patients"], "covered_patients": [],
             "coverage_uncertain_medics": [], "options": [], "rejections": []}
    rejections = audit["rejections"]
    if state["game_loop"] < config["min_game_loop"]:
        rejections.append("min_game_loop")
    if base_intent not in config["base_intents"] or base_binding["action"] != 1:
        rejections.append("base_intent_mismatch")
    if not unit.get("alive"):
        rejections.append("dead")
    if not finite(unit.get("energy")) or unit["energy"] < event["min_energy"]:
        rejections.append("insufficient_energy")
    if has_heal_order(unit):
        rejections.append("active_heal_order")
    if hp_ratio(unit) < event["min_health_ratio"]:
        rejections.append("low_health")
    if windows < event["min_stop_windows"]:
        rejections.append("stop_window_threshold")
    if rejections:
        audit["reason"] = rejections[0]
        return None, audit
    if not context["patients"]:
        audit["reason"] = "no_unreserved_wounded"
        return None, audit
    coverage, uncertain = other_heal_coverage(state, actor)
    audit["covered_patients"] = [{**row, "healers": coverage[row["target"]]}
                                 for row in context["patients"] if row["target"] in coverage]
    audit["coverage_uncertain_medics"] = uncertain
    if uncertain:
        audit["reason"] = "other_medical_legality_unknown"
        return None, audit
    uncovered = [row for row in context["patients"] if row["target"] not in coverage]
    if not uncovered:
        audit["reason"] = "no_uncovered_wounded"
        return None, audit
    eligible = [row for row in uncovered if row["gap"] <= event["max_target_distance"]]
    if not eligible:
        audit["reason"] = "no_target_within_distance"
        return None, audit
    destinations = moves(state, actor, legal)
    if not destinations:
        audit["reason"] = "no_legal_move"
        return None, audit
    options = []
    for patient in eligible:
        target = patient["target"]
        for action, point in sorted(destinations.items()):
            next_gap = math.dist(point, position(state["allies"][target]))
            progress = patient["gap"] - next_gap
            extra = exposure(threat_rows, point) - context["current_exposure"]
            reasons = []
            if progress < event["min_progress"]:
                reasons.append("insufficient_progress")
            if extra > event["max_extra_exposure"]:
                reasons.append("exposure_budget_exceeded")
            # Lexicographic choice is declared policy, not a learned score.
            key = (extra, patient["gap"], patient["hp_ratio"], -patient["loss_ratio"],
                   next_gap, actor_index(target), action)
            row = {"target": target, "action": action, "gap": patient["gap"], "next_gap": next_gap,
                   "progress": progress, "exposure_delta": extra, "selection_key": list(key),
                   "rejections": reasons}
            audit["options"].append(row)
            if not reasons and action != base_binding["action"]:
                options.append((key, row))
    if not options:
        audit["reason"] = "no_move_passes_progress_and_exposure"
        return None, audit
    _, chosen = min(options, key=lambda entry: entry[0])
    audit.update(eligible=True, reason="proposed", **{k: v for k, v in chosen.items() if k != "rejections"})
    binding = {k: v for k, v in chosen.items() if k != "rejections"}
    binding.update(reason="idle_recovery_bounded_one_step", generator="idle_recovery")
    # gap is the pre-command observed distance, never the hoped-for next gap.
    approach = {"target": binding["target"], "target_tag": state["allies"][binding["target"]].get("tag"),
                "gap": binding["gap"], "windows": 1, "generator": "idle_recovery"}
    return {"intent": "APPROACH", "binding": binding, "approach_memory": approach}, audit


def diagnose_recorded_events(path):
    """Read old executed BASE traces to report coverage, never simulated return.

Later eligible snapshots remain snapshots from the recorded BASE trajectory;
they cannot describe what would happen after an earlier intervention.
"""
    from collections import Counter
    from copy import deepcopy
    import json
    from pathlib import Path

    events = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    observations = {(e["episode"], e["step"]): e["payload"] for e in events if e.get("event") == "skill_observation"}
    config = {"intent": "APPROACH", "base_intents": ["STOP"], "min_game_loop": 0,
              "generator": "idle_recovery", "event": dict(DEFAULT_EVENT)}
    episodes, history = {}, {}
    for frame in events:
        if frame.get("event") != "medivac_tree_step":
            continue
        if frame.get("mode") != "tree":
            raise ValueError("coverage diagnostic requires executed tree traces")
        episode = frame["episode"]
        payload = observations[(episode, frame["step"])]
        state = deepcopy(payload["state"])
        # The real SkillManager installs these public per-medic menus before
        # calling the tree. Raw observation logs precede that installation.
        state["legal_actions"] = dict(state.get("legal_actions", {}))
        for medic, unit in state["allies"].items():
            if is_medivac(unit):
                state["legal_actions"][medic] = ([int(k) for k in payload["questions"].get(medic, {}).get("criteria", {})]
                                                  if unit.get("alive") else [0])
        result = episodes.setdefault(episode, {"eligible_states_on_recorded_base": 0, "first_opportunity": None,
                                               "reasons": Counter(), "stop_decisions": []})
        for decision in frame["decisions"]:
            if decision.get("executed") is not True or decision.get("final_action") != decision["action"]:
                raise ValueError("coverage diagnostic requires unmodified executed decisions")
            if "return_selection" in decision and decision["return_selection"].get("selected") != "BASE":
                raise ValueError("coverage diagnostic cannot treat an intervened trajectory as BASE")
            actor = decision["actor"]
            old = history.get((episode, actor), {})
            binding = decision["binding"]
            windows = stop_windows(state, actor, old, decision["intent"], binding, config["event"])
            legal = {int(k) for k in payload["questions"].get(actor, {}).get("criteria", {})} if state["allies"][actor].get("alive") else {0}
            proposal, audit = evaluate(config, state, actor, legal, old, decision["reservation_before"],
                                       decision["visible_threats"], decision["intent"], binding, windows)
            result["reasons"][audit["reason"]] += 1
            if proposal:
                result["eligible_states_on_recorded_base"] += 1
                if result["first_opportunity"] is None:
                    result["first_opportunity"] = {"actor": actor, "game_loop": frame["game_loop"],
                                                   "stop_windows": windows, "binding": proposal["binding"]}
            if decision["intent"] == "STOP":
                result["stop_decisions"].append({"actor": actor, "game_loop": frame["game_loop"],
                                                 "stop_windows": windows, "reason": audit["reason"],
                                                 "target": proposal["binding"]["target"] if proposal else None,
                                                 "reserved_targets": [r["target"] for r in audit["reserved_patients"]]})
            history[(episode, actor)] = {**decision["memory_after"],
                                         "idle_stop_windows": windows if decision["intent"] == "STOP" and decision["final_action"] == 1 else 0}
    return {"schema_version": 1, "kind": "recorded_base_candidate_coverage", "candidate": config,
            "episodes": episodes, "predicted_returns": None,
            "limitations": ["Only coverage on previously executed observations is checked.",
                            "No alternative action is executed; return or safety cannot be inferred.",
                            "Later recorded states do not predict the continuation after intervention."]}


if __name__ == "__main__":
    import argparse
    import json
    parser = argparse.ArgumentParser(description="Read-only coverage diagnostic; never starts SC2")
    parser.add_argument("--events", required=True)
    args = parser.parse_args()
    print(json.dumps(diagnose_recorded_events(args.events), ensure_ascii=False, indent=2, allow_nan=False))
