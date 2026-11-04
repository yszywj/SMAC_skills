"""Audit independently rerun, one-intervention episode pairs.

This is matched *observed* prefix evidence, not a simulator snapshot fork or
proof of identical hidden state. Never attach an episode return to every step.
Old logs without the explicit opportunity contract are rejected, not guessed.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import re

SCHEMA_VERSION = 1
EVIDENCE_KIND = "matched_observed_prefix"
FILES = ("run.json", "manifest.json", "events.jsonl", "episodes.json", "summary.json")


class AuditError(ValueError):
    """An input cannot support an audited return label."""


def _require(condition, reason):
    if not condition:
        raise AuditError(reason)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _hash(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _load(path):
    path = Path(path)
    out = {"path": str(path.resolve()), "file_hashes": {}}
    for name in FILES:
        raw = (path / name).read_bytes()
        out["file_hashes"][name] = hashlib.sha256(raw).hexdigest()
        if name.endswith("jsonl"):
            out[name] = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
        else:
            out[name] = json.loads(raw)
    _require(all(isinstance(out[name], dict) for name in ("run.json", "manifest.json", "summary.json")),
             "invalid_run_document")
    _require(isinstance(out["episodes.json"], list) and all(isinstance(e, dict) for e in out["episodes.json"]),
             "invalid_episode_document")
    _require(all(isinstance(e, dict) for e in out["events.jsonl"]), "invalid_event_document")
    return out


def _policy_source(key, metadata):
    """Only the explicitly declared policy JSON may differ between arms."""
    declared = str(metadata.get("tree_path", "")).replace("\\", "/")
    key = str(key).replace("\\", "/")
    return bool(declared.endswith(".json") and (declared == key or declared.endswith("/" + key)))


def _run_contract(base, candidate, noise, *, other_mode=None):
    from medivac_tree.candidates import FEATURE_NAMES, FEATURE_SCHEMA_VERSION

    expected_other_mode = other_mode or ("base" if noise else "candidate")
    _require(expected_other_mode in ("base", "candidate", "model"), "unsupported_comparison_mode")

    b, c = base["run.json"], candidate["run.json"]
    for run in (base, candidate):
        manifest, summary = run["manifest.json"], run["summary.json"]
        metadata = run["run.json"]
        _require(summary.get("status") == "complete" and summary.get("error") is None, "run_not_complete")
        _require(not any(e.get("event") == "run_failed" for e in run["events.jsonl"]), "run_failed_event")
        _require(manifest.get("version") == 1 and manifest.get("sha256") == digest(
            {k: v for k, v in manifest.items() if k != "sha256"}), "manifest_hash_invalid")
        n = len(manifest.get("episodes", []))
        _require(n > 0 and isinstance(run["episodes.json"], list), "missing_episodes")
        _require(len(run["episodes.json"]) == n == summary.get("completed_episodes")
                 == summary.get("expected_episodes") == metadata.get("scenario_count"), "episode_count_mismatch")
        _require(metadata.get("manifest_sha256") == summary.get("manifest_sha256") == manifest["sha256"],
                 "manifest_binding_mismatch")
        _require(metadata.get("medivac_policy") == "tree", "tree_not_executed")
        _require(metadata.get("native_teacher_control") == "off" and summary.get("pair_mode") == "off",
                 "other_intervention_enabled")
        for field in ("source_hashes", "skill_source_hashes", "versions", "env_args", "skill_parameters",
                      "observation_protocol", "action_extensions"):
            _require(field in metadata and metadata[field] is not None, "missing_run_contract:" + field)
        _require(metadata["source_hashes"] and metadata["versions"], "empty_source_or_version_contract")
        for field in ("source_hashes", "skill_source_hashes"):
            _require(isinstance(metadata[field], dict) and all(_hash(v) for v in metadata[field].values()),
                     "invalid_source_hashes:" + field)
        _require(metadata["env_args"] == manifest.get("env_args"), "environment_manifest_mismatch")
        for field in ("skill_source_hashes", "skill_parameters", "action_extensions"):
            _require(summary.get(field) == metadata[field], "summary_contract_mismatch:" + field)
        for record in manifest["episodes"]:
            _require(record.get("sha256") == digest(record.get("episode_config")), "scenario_hash_invalid")
            _require(isinstance(record.get("sc2_seed"), int), "missing_scenario_seed")
    _require(base["manifest.json"] == candidate["manifest.json"], "manifest_mismatch")
    bm, cm = b.get("medivac_tree_metadata", {}), c.get("medivac_tree_metadata", {})
    for meta, mode in ((bm, "base"), (cm, expected_other_mode)):
        _require(meta.get("policy_kind") == "paired_return_selector", "missing_paired_selector_contract")
        _require(meta.get("selection_mode") == mode, "selection_mode_mismatch")
        _require(_hash(meta.get("comparison_contract_hash")), "invalid_comparison_contract_hash")
        _require(_hash(meta.get("base_tree_hash")) and _hash(meta.get("tree_hash")), "missing_tree_hash")
        _require(isinstance(meta.get("candidate"), dict) and meta["candidate"], "missing_candidate_contract")
        _require(isinstance(meta.get("runtime_source_hashes"), dict) and meta["runtime_source_hashes"],
                 "missing_runtime_source_hashes")
        _require(all(_hash(v) for v in meta["runtime_source_hashes"].values()), "invalid_runtime_source_hashes")
        _require(meta.get("feature_schema") == {"version": FEATURE_SCHEMA_VERSION, "names": list(FEATURE_NAMES)},
                 "feature_schema_mismatch")
    # All non-declared metadata and all executable source hashes must agree.
    def normalized(meta):
        return {k: v for k, v in meta.items() if k not in ("tree_hash", "tree_path", "version", "selection_mode")}
    _require(normalized(bm) == normalized(cm), "policy_contract_mismatch")
    _require({k: v for k, v in b.items() if k not in ("medivac_tree_metadata", "source_hashes")} ==
             {k: v for k, v in c.items() if k not in ("medivac_tree_metadata", "source_hashes")}, "run_configuration_mismatch")
    bh = {k.replace("\\", "/"): v for k, v in b["source_hashes"].items() if not _policy_source(k, bm)}
    ch = {k.replace("\\", "/"): v for k, v in c["source_hashes"].items() if not _policy_source(k, cm)}
    _require(bh == ch, "source_hashes_mismatch")
    return bm


def _index(events, event_type, field):
    out = {}
    for e in events:
        if e.get("event") == event_type:
            key = e.get(field)
            _require(isinstance(key, int) and key not in out, "duplicate_or_missing:" + event_type)
            out[key] = e
    return out


def _episode(run, episode, scenario):
    events = [e for e in run["events.jsonl"] if e.get("episode") == episode]
    records = [e for e in run["episodes.json"] if e.get("episode") == episode]
    starts = [e for e in events if e.get("event") == "episode_started"]
    ends = [e for e in events if e.get("event") == "episode_completed"]
    _require(len(records) == len(starts) == len(ends) == 1, "missing_or_duplicate_episode_boundary")
    record, start, end = records[0], starts[0], ends[0]
    _require(record.get("outcome") in ("win", "loss", "draw", "timeout") and _finite(record.get("reward")), "invalid_episode_result")
    for k in ("scenario_sha256", "initial_state_sha256", "outcome", "steps", "reward", "info"):
        _require(k in record and record[k] == end.get(k), "episode_end_mismatch:" + k)
    _require(record["scenario_sha256"] == start.get("scenario_sha256") == scenario["sha256"], "scenario_binding_mismatch")
    _require(_hash(record["initial_state_sha256"]) and record["initial_state_sha256"] == start.get("initial_state_sha256"),
             "initial_state_binding_mismatch")
    _require(bool(start.get("game_version")), "missing_game_version")
    steps = _index(events, "step", "step")
    observations = _index(events, "skill_observation", "step")
    trees = _index(events, "medivac_tree_step", "step")
    n = record["steps"]
    _require(isinstance(n, int) and n > 0 and set(steps) == set(range(1, n + 1)), "incomplete_step_sequence")
    _require(set(observations) == set(trees) == set(range(n)), "incomplete_observation_or_tree_sequence")
    _require(steps[n].get("terminated") is True and all(steps[i].get("terminated") is False for i in range(1, n)),
             "invalid_termination_sequence")
    _require(all(_finite(s.get("reward")) for s in steps.values()) and
             math.isclose(sum(s["reward"] for s in steps.values()), record["reward"], rel_tol=1e-10, abs_tol=1e-8),
             "reward_sum_mismatch")
    opportunities = []
    for i in range(n):
        step, observation, tree = steps[i + 1], observations[i], trees[i]
        _require(step.get("game_loop") == tree.get("game_loop"), "step_loop_mismatch")
        payload = observation.get("payload", {})
        _require(isinstance(payload.get("state"), dict) and isinstance(payload.get("questions"), dict), "missing_observation")
        actions = step.get("actions")
        _require(isinstance(actions, list) and actions, "missing_final_actions")
        _require(set(payload["state"].get("allies", {})) == {"u" + str(j) for j in range(len(actions))},
                 "action_roster_mismatch")
        for j, action in enumerate(actions):
            actor = "u" + str(j)
            q = payload["questions"].get(actor)
            ally = payload["state"].get("allies", {}).get(actor, {})
            legal = str(action) in (q or {}).get("criteria", {}) or (action == 0 and ally.get("alive") is False)
            _require(isinstance(action, int) and not isinstance(action, bool) and legal, "illegal_final_action")
        _require(tree.get("mode") == "tree" and tree.get("candidate_metadata") == run["run.json"]["medivac_tree_metadata"],
                 "tree_event_metadata_mismatch")
        decisions = tree.get("decisions")
        _require(isinstance(decisions, list), "missing_decisions")
        for d in decisions:
            _require(isinstance(d, dict), "malformed_decision")
            if "return_selection" in d:
                selection = d["return_selection"]
                _require(isinstance(selection, dict), "malformed_return_selection")
                if selection.get("opportunity") is True:
                    opportunities.append((i, tree["game_loop"], d, selection))
    return {"record": record, "start": start, "steps": steps, "observations": observations,
            "trees": trees, "opportunities": opportunities}


def _feedback(ep, index, d, selected, *, allow_existing_order=False):
    step = ep["steps"][index + 1]
    actor = d.get("actor")
    _require(isinstance(actor, str) and re.fullmatch(r"u\d+", actor), "invalid_actor")
    aid = int(actor[1:])
    _require(aid < len(step["actions"]), "actor_not_in_final_actions")
    _require(d.get("executed") is True and d.get("override_reason") is None, "decision_not_executed")
    _require(all(d.get(k) == selected.get(k) for k in ("intent", "action", "target")), "selected_decision_mismatch")
    _require(d.get("proposed_action") == d.get("final_action") == step["actions"][aid] == selected["action"],
             "selected_action_not_submitted")
    feedback = step.get("feedback", {})
    _require(feedback.get("response_mapping") == "exact" and feedback.get("from_game_loop") == step["game_loop"],
             "feedback_not_exact")
    _require(not feedback.get("unmatched_observation_errors"), "unmatched_feedback_errors")
    f = feedback.get("actors", {}).get(actor)
    _require(isinstance(f, dict) and str(f.get("requested_action_id")) == str(selected["action"]), "feedback_action_mismatch")
    _require(f.get("observation_errors") == [], "feedback_observation_error")
    submission = f.get("submission")
    acceptance = {"acceptance_semantics": "engine_accepted"}
    if submission == "accepted":
        _require(selected.get("intent") != "KEEP_HEAL", "keep_must_retain_order")
        _require(isinstance(f.get("response"), dict) and f["response"].get("code") == 1
                 and f["response"].get("name") == "Success", "feedback_failed")
        _require(isinstance(f.get("command"), dict), "feedback_command_missing")
        if selected.get("intent") in ("HEAL_NEW", "HEAL_CURRENT", "HEAL_RESCUE"):
            _require(f["command"].get("kind") == "heal"
                     and f["command"].get("target_unit") == selected.get("target"), "feedback_target_mismatch")
        if selected["action"] in (1, 2, 3, 4, 5):
            _require(f["command"].get("kind") == ("stop" if selected["action"] == 1 else "move"),
                     "feedback_command_mismatch")
    elif submission == "rejected" and allow_existing_order:
        # A redundant BASE heal command can receive AlreadyTargeted. This is
        # evidence of an existing order, never confirmation that a new command
        # was accepted or that healing occurred. Candidate changes still need
        # Success; no other rejection has this restricted interpretation.
        _require(selected.get("intent") == "HEAL_CURRENT"
                 and f.get("response") == {"code": 206, "name": "AlreadyTargeted"}, "feedback_failed")
        allies = ep["observations"][index]["payload"]["state"]["allies"]
        orders = allies[actor].get("orders", [])
        target = selected.get("target")
        patient = allies.get(target, {})
        tag = patient.get("tag")
        _require(isinstance(tag, str) and bool(tag) and bool(orders) and isinstance(orders[0], dict)
                 and orders[0].get("ability_id") == 386 and orders[0].get("target_unit") == target
                 and orders[0].get("target_tag") == tag, "existing_heal_order_not_observed")
        command = f.get("command")
        _require(isinstance(command, dict) and command.get("ability_id") == 386 and command.get("kind") == "heal"
                 and command.get("target_unit") == target and command.get("target_tag") == tag,
                 "existing_heal_command_mismatch")
        acceptance = {"acceptance_semantics": "observed_existing_order", "observed_target": target,
                      "observed_target_tag": tag, "new_command_accepted": False,
                      "healing_effect_confirmed": False}
    else:
        _require(submission == "retained_order" and selected.get("intent") == "KEEP_HEAL"
                 and f.get("command") is None and f.get("response") is None, "feedback_failed")
        orders = ep["observations"][index]["payload"]["state"]["allies"][actor].get("orders", [])
        _require(bool(orders) and orders[0].get("ability_id") == 386
                 and orders[0].get("target_unit") == selected.get("target"), "retained_target_not_observed")
        acceptance = {"acceptance_semantics": "explicit_retained_order"}
    return copy.deepcopy(f), acceptance


def _matched_row(base, candidate, episode, scenario, meta):
    b, c = _episode(base, episode, scenario), _episode(candidate, episode, scenario)
    _require(b["start"] == c["start"], "initial_state_or_game_version_mismatch")
    _require(len(b["opportunities"]) == len(c["opportunities"]) == 1, "not_exactly_one_opportunity")
    bi, bl, bd, bs = b["opportunities"][0]
    ci, cl, cd, cs = c["opportunities"][0]
    _require((bi, bl, bd.get("actor")) == (ci, cl, cd.get("actor")), "opportunity_boundary_mismatch")
    from medivac_tree.selector import fingerprint
    for decision, selection in ((bd, bs), (cd, cs)):
        _require(selection.get("comparison_contract_hash") == meta["comparison_contract_hash"], "opportunity_contract_mismatch")
        _require("predicted_gain" in selection and selection["predicted_gain"] is None
                 and selection.get("path") == [], "unexpected_forced_arm_prediction")
        features = selection.get("features", {})
        _require(isinstance(features, dict) and set(features) == set(meta["feature_schema"]["names"])
                 and all(_finite(v) for v in features.values()),
                 "invalid_opportunity_features")
        for field in ("state_fingerprint", "memory_fingerprint", "reservations_fingerprint", "context_fingerprint"):
            _require(_hash(selection.get(field)), "missing_fingerprint:" + field)
        _require(isinstance(decision.get("reservation_before"), dict) and selection["reservations_fingerprint"]
                 == fingerprint(decision["reservation_before"]), "reservation_fingerprint_invalid")
        context = {k: selection[k] for k in ("state_fingerprint", "memory_fingerprint", "reservations_fingerprint")}
        context.update(actor=decision.get("actor"), game_loop=bl)
        _require(selection["context_fingerprint"] == fingerprint(context), "context_fingerprint_invalid")
        for field in ("base", "candidate"):
            choice = selection.get(field)
            _require(isinstance(choice, dict) and all(k in choice for k in ("intent", "action", "target")), "missing_choice:" + field)
        _require(selection["base"]["action"] != selection["candidate"]["action"], "no_action_contrast")
    for field in ("features", "base", "candidate", "state_fingerprint", "memory_fingerprint", "reservations_fingerprint",
                  "context_fingerprint", "comparison_contract_hash"):
        _require(bs[field] == cs[field], "opportunity_mismatch:" + field)
    _require(bs.get("selected") == "BASE" and cs.get("selected") == "CANDIDATE", "arm_selection_mismatch")
    for i in range(bi + 1):
        _require(b["observations"][i]["payload"] == c["observations"][i]["payload"], "observed_prefix_mismatch")
        if i < bi:
            _require(b["steps"][i + 1] == c["steps"][i + 1], "action_or_feedback_prefix_mismatch")
    ba, ca = b["steps"][bi + 1]["actions"], c["steps"][bi + 1]["actions"]
    actor = bd["actor"]
    _require(len(ba) == len(ca) and [i for i, (a, z) in enumerate(zip(ba, ca)) if a != z] == [int(actor[1:])],
             "multiple_actor_intervention")
    bf, be = _feedback(b, bi, bd, bs["base"], allow_existing_order=True)
    cf, ce = _feedback(c, ci, cd, cs["candidate"])
    pair_id = digest({"base_events": base["file_hashes"]["events.jsonl"],
                      "candidate_events": candidate["file_hashes"]["events.jsonl"], "episode": episode})
    return {"pair_id": pair_id, "group_id": scenario["sha256"], "scenario_sha256": scenario["sha256"],
            "episode": episode, "sc2_seed": scenario["sc2_seed"], "game_loop": bl, "actor": actor,
            "features": copy.deepcopy(bs["features"]), "target_gain": c["record"]["reward"] - b["record"]["reward"],
            "base_return": b["record"]["reward"], "candidate_return": c["record"]["reward"],
            "base_outcome": b["record"]["outcome"], "candidate_outcome": c["record"]["outcome"],
            "base": bs["base"], "candidate": bs["candidate"], "context_fingerprint": bs["context_fingerprint"],
            "initial_state_sha256": b["record"]["initial_state_sha256"], "prefix_steps": bi,
            "prefix_sha256": digest([b["observations"][i]["payload"] for i in range(bi + 1)]),
            "feedback": {"base": bf, "candidate": cf}, "feedback_audit": {"base": be, "candidate": ce},
            "comparison_contract_hash": meta["comparison_contract_hash"],
            "evidence_kind": EVIDENCE_KIND}


def build_dataset(base_run, candidate_run, *, baseline_noise=False):
    """Return auditable labels plus explicit rejections; never silently pair by row order."""
    result = {"schema_version": SCHEMA_VERSION, "evidence_kind": EVIDENCE_KIND,
              "comparison_contract_hash": None, "feature_schema": None,
              "rows": [], "rejected": [], "noise_audit": [],
              "limitations": ["Independently rerun observed prefixes; hidden simulator/RNG state is not proven equal.",
                              "One completed episode pair supplies at most one return label; no per-step credit claim.",
                              "State/memory fingerprints are matched across arms; their complete runtime inputs are not reconstructed from logs."]}
    try:
        base, candidate = _load(base_run), _load(candidate_run)
        result["sources"] = {"base": {k: base[k] for k in ("path", "file_hashes")},
                             "candidate": {k: candidate[k] for k in ("path", "file_hashes")}}
        meta = _run_contract(base, candidate, baseline_noise)
        from medivac_tree.experiment import context_hash, experiment_context
        context = experiment_context(base["run.json"])
        _require(context == experiment_context(candidate["run.json"]), "experiment_context_mismatch")
        result["experiment_context"] = context
        result["experiment_context_hash"] = context_hash(context)
        result["comparison_contract_hash"] = meta["comparison_contract_hash"]
        result["feature_schema"] = meta["feature_schema"]
        result["runtime_source_hashes"] = meta["runtime_source_hashes"]
    except (AuditError, OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        result["rejected"].append({"episode": None, "reasons": [str(exc)]})
        return result
    for episode, scenario in enumerate(base["manifest.json"]["episodes"], 1):
        try:
            if baseline_noise:
                b, c = _episode(base, episode, scenario), _episode(candidate, episode, scenario)
                _require(b["start"] == c["start"], "initial_state_or_game_version_mismatch")
                for ep in (b, c):
                    _require(all(s.get("selected") == "BASE" for _, _, _, s in ep["opportunities"]), "noise_arm_not_base")
                common = min(len(b["steps"]), len(c["steps"]))
                first = next((i for i in range(common) if b["observations"][i]["payload"] != c["observations"][i]["payload"]
                              or b["steps"][i + 1] != c["steps"][i + 1]), None)
                result["noise_audit"].append({"episode": episode, "group_id": scenario["sha256"],
                    "return_difference": c["record"]["reward"] - b["record"]["reward"],
                    "first_divergent_step": first, "episode_lengths_equal": len(b["steps"]) == len(c["steps"]),
                    "training_label": False})
            else:
                result["rows"].append(_matched_row(base, candidate, episode, scenario, meta))
        except (AuditError, ValueError, KeyError, TypeError, IndexError, AttributeError) as exc:
            result["rejected"].append({"episode": episode, "group_id": scenario["sha256"], "reasons": [str(exc)]})
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-run", required=True, type=Path)
    parser.add_argument("--candidate-run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--baseline-noise", action="store_true", help="BASE/BASE noise audit; emits no training rows")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("output already exists; refusing to overwrite")
    result = build_dataset(args.base_run, args.candidate_run, baseline_noise=args.baseline_noise)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"rows": len(result["rows"]), "rejected": len(result["rejected"]),
                      "noise_pairs": len(result["noise_audit"])}))
    return 0 if result["rows"] or result["noise_audit"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
