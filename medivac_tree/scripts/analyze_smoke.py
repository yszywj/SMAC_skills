"""Audit recorded direct-tree runs without importing a policy or starting SC2."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=True).encode()).hexdigest()


def decisions_list(trace):
    values = trace.get("decisions", [])
    if isinstance(values, dict):
        return [dict(value, actor=actor) for actor, value in values.items()]
    return values


def actor_index(actor):
    return int(str(actor).removeprefix("u"))


def audit_run(path):
    path = Path(path)
    manifest, metadata = read(path / "manifest.json"), read(path / "run.json")
    summary = read(path / "summary.json") if (path / "summary.json").exists() else {}
    expected = {index + 1: row for index, row in enumerate(manifest["episodes"])}
    completed, started, steps, observations, traces, nested = {}, {}, {}, {}, {}, {}
    errors, failures = [], []
    manifest_body = {key: value for key, value in manifest.items() if key != "sha256"}
    if manifest.get("sha256") != canonical_hash(manifest_body):
        errors.append("manifest content hash mismatch")
    if metadata.get("manifest_sha256") != manifest.get("sha256"):
        errors.append("run metadata manifest hash mismatch")
    with (path / "events.jsonl").open(encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, 1):
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"events line {line_no}: invalid JSON: {exc}")
                continue
            kind, episode = event.get("event"), event.get("episode")
            if kind == "episode_started":
                if episode in started:
                    errors.append(f"duplicate episode_started: {episode}")
                started[episode] = event
            elif kind == "episode_completed":
                if episode in completed:
                    errors.append(f"duplicate episode_completed: {episode}")
                completed[episode] = event
            elif kind == "run_failed":
                failures.append(event.get("error", "run_failed"))
            elif kind in ("skill_observation", "policy_observation"):
                payload = event.get("payload", {})
                loop = payload.get("state", {}).get("game_loop")
                observations[(episode, loop)] = payload
            elif kind in ("step", "medivac_tree_step"):
                key = (episode, event.get("game_loop"))
                store = steps if kind == "step" else traces
                if key in store:
                    errors.append(f"duplicate {kind}: {key}")
                store[key] = event
            elif kind == "skill_step":
                trace = event.get("race_telemetry", {}).get("medivac_tree")
                if trace is not None:
                    nested[(episode, event.get("game_loop"))] = trace
    # Dedicated post-mask events take precedence over earlier skill proposals.
    for key, trace in nested.items():
        traces.setdefault(key, trace)
    if set(completed) - set(expected):
        errors.append("completed episodes outside manifest")
    if summary.get("status") != "complete":
        errors.append(f"run status is {summary.get('status', 'missing')!r}")
    if failures:
        errors.append(f"run_failed events: {len(failures)}")
    rows, per_episode = [], {}
    for episode, scenario in expected.items():
        n_medivacs = sum(str(x).lower() == "medivac" for x in scenario["episode_config"]["team_gen"]["ally_team"])
        finish, begin = completed.get(episode), started.get(episode)
        state = "complete" if finish else "invalid" if begin else "not_started"
        if finish:
            if finish.get("scenario_sha256") != scenario.get("sha256"):
                errors.append(f"episode {episode}: scenario hash does not match manifest")
            if not begin or not finish.get("initial_state_sha256") or finish.get("initial_state_sha256") != begin.get("initial_state_sha256"):
                errors.append(f"episode {episode}: missing/inconsistent initial state hash")
        ep_steps = {loop: value for (ep, loop), value in steps.items() if ep == episode}
        leaves, intents, feedback_counts, counters = Counter(), Counter(), Counter(), Counter()
        trace_actors = set()
        for (ep, loop), trace in traces.items():
            if ep != episode:
                continue
            counters["tree_steps"] += 1
            actual = steps.get((ep, loop))
            if actual is None:
                errors.append(f"episode {ep} loop {loop}: tree decision has no submitted step")
                continue
            target_counts = Counter()
            for decision in decisions_list(trace):
                actor = str(decision.get("actor", ""))
                try:
                    index = actor_index(actor)
                    if index < 0:
                        raise ValueError("negative actor index")
                    action = actual["actions"][index]
                except (ValueError, KeyError, IndexError, TypeError):
                    errors.append(f"episode {ep} loop {loop}: invalid tree actor {actor!r}")
                    continue
                trace_actors.add(actor)
                counters["tree_decisions"] += 1
                leaf = decision.get("leaf_id", decision.get("leaf"))
                if leaf is None:
                    errors.append(f"episode {ep} loop {loop} {actor}: missing leaf")
                leaves[str(leaf)] += 1
                intents[str(decision.get("intent", "unknown"))] += 1
                proposed = decision.get("proposed_action", decision.get("action"))
                counters["proposal_matches_submitted"] += proposed == action
                counters["proposal_differs_from_submitted"] += proposed != action
                mode = trace.get("mode", metadata.get("medivac_policy"))
                applied = decision.get("executed", mode == "tree")
                counters["applied_tree_decisions"] += bool(applied)
                if decision.get("final_action", action) != action:
                    errors.append(f"episode {ep} loop {loop} {actor}: trace final action disagrees with step")
                if applied and proposed != action and not decision.get("override_reason"):
                    errors.append(f"episode {ep} loop {loop} {actor}: unexplained proposal override")
                counters["recorded_override_reasons"] += bool(decision.get("override_reason"))
                target = decision.get("requested_target", decision.get("target"))
                healing = (isinstance(proposed, int) and not isinstance(proposed, bool)
                           and 6 <= proposed < 6 + len(actual["actions"]))
                healing = healing or str(decision.get("intent", "")).upper() == "KEEP_HEAL"
                reserves_patient = healing or str(decision.get("intent", "")).upper() == "APPROACH"
                if target is not None and reserves_patient:
                    target_counts[str(target)] += 1
                    owner = decision.get("reservation_before", {}).get(str(target))
                    if owner is not None and owner != actor:
                        errors.append(f"episode {ep} loop {loop} {actor}: chose another actor's reserved patient")
                    if "reservation_after" in decision and decision["reservation_after"].get(str(target)) != actor:
                        errors.append(f"episode {ep} loop {loop} {actor}: selected patient was not reserved")
                blocked = decision.get("binding_details", {}).get("blocked_targets", decision.get("facts", {}).get("blocked_targets"))
                if isinstance(blocked, (list, dict)):
                    counters["reservation_candidate_exclusions"] += len(blocked)
            counters["reservation_conflicts"] += sum(max(0, value - 1) for value in target_counts.values())
            if any(value > 1 for value in target_counts.values()):
                errors.append(f"episode {ep} loop {loop}: duplicate tree patient reservation")
        for loop, step in ep_steps.items():
            payload = observations.get((episode, loop))
            if payload is None:
                errors.append(f"episode {episode} loop {loop}: missing observation")
            medic_actors = {actor for actor, value in (payload or {}).get("state", {}).get("allies", {}).items()
                            if value.get("unit_type_id") == 54}
            feedback = step.get("feedback", {})
            if feedback.get("response_mapping") != "exact" or feedback.get("unmatched_observation_errors"):
                errors.append(f"episode {episode} loop {loop}: engine feedback mapping is not exact")
            for actor in medic_actors:
                entry = feedback.get("actors", {}).get(actor)
                if entry is None:
                    errors.append(f"episode {episode} loop {loop}: missing feedback for {actor}")
                    continue
                feedback_counts[(entry.get("response") or {}).get("name") or entry.get("submission", "unknown")] += 1
                counters["engine_observation_errors"] += len(entry.get("observation_errors", []))
                try:
                    if int(entry["requested_action_id"]) != step["actions"][actor_index(actor)]:
                        errors.append(f"episode {episode} loop {loop}: feedback request differs for {actor}")
                except (KeyError, ValueError, IndexError, TypeError):
                    errors.append(f"episode {episode} loop {loop}: malformed action feedback for {actor}")
        if metadata.get("medivac_policy") in ("tree", "tree-shadow") and n_medivacs and finish and not counters["tree_decisions"]:
            errors.append(f"episode {episode}: tree policy has no decision records")
        record = {"episode": episode, "sc2_seed": scenario["sc2_seed"],
                  "study_group": scenario.get("study_group"), "medivac_count": n_medivacs,
                  "status": state, "win": finish.get("outcome") == "win" if finish else None,
                  "outcome": finish.get("outcome") if finish else None,
                  "initial_state_sha256": finish.get("initial_state_sha256") if finish else None,
                  "scenario_sha256": scenario.get("sha256"), "steps": len(ep_steps),
                  "wall_seconds": finish.get("wall_seconds") if finish else None,
                  "action_sequence_sha256": canonical_hash([[loop, value["actions"]] for loop, value in sorted(ep_steps.items())]),
                  "leaf_coverage": dict(leaves), "intents": dict(intents),
                  "engine_feedback": dict(feedback_counts),
                  **{key: counters[key] for key in ("tree_steps", "tree_decisions", "applied_tree_decisions",
                      "proposal_matches_submitted", "proposal_differs_from_submitted", "recorded_override_reasons",
                      "reservation_conflicts", "reservation_candidate_exclusions", "engine_observation_errors")}}
        rows.append(record)
        per_episode[episode] = record
    total = sum(row["status"] == "complete" for row in rows)
    aggregate_leaves, aggregate_intents, aggregate_feedback = Counter(), Counter(), Counter()
    for row in rows:
        aggregate_leaves.update(row["leaf_coverage"])
        aggregate_intents.update(row["intents"])
        aggregate_feedback.update(row["engine_feedback"])
    coordination = metadata.get("medical_patient_coordination")
    if isinstance(coordination, bool):
        coordination = "on" if coordination else "off"
    report = {"run": str(path.resolve()), "policy": metadata.get("medivac_policy"),
              "coordination": coordination,
              "candidate_metadata": metadata.get("medivac_tree_metadata"),
              "manifest_sha256": manifest.get("sha256"), "expected_episodes": len(expected),
              "completed_episodes": total, "invalid_episodes": sum(row["status"] == "invalid" for row in rows),
              "not_started_episodes": sum(row["status"] == "not_started" for row in rows),
              "wins": sum(row["win"] is True for row in rows),
              "win_rate_valid_only": sum(row["win"] is True for row in rows) / total if total else None,
              "tree_decisions": sum(row["tree_decisions"] for row in rows),
              "applied_tree_decisions": sum(row["applied_tree_decisions"] for row in rows),
              "proposal_matches_submitted": sum(row["proposal_matches_submitted"] for row in rows),
              "proposal_differs_from_submitted": sum(row["proposal_differs_from_submitted"] for row in rows),
              "reservation_conflicts": sum(row["reservation_conflicts"] for row in rows),
              "leaf_coverage": dict(aggregate_leaves), "intents": dict(aggregate_intents),
              "engine_feedback": dict(aggregate_feedback),
              "audit_errors": errors, "failures": failures, "episodes": rows,
              "engine_feedback_meaning": "Acknowledgement is not proof of successful healing; repeated commands are not independent samples"}
    return report, {"manifest": manifest, "metadata": metadata, "episodes": per_episode, "steps": steps, "observations": observations}


def compare(a, b):
    if a["manifest"] != b["manifest"]:
        return {"audit_errors": ["run manifests differ"], "pairs": []}
    errors, rows = [], []
    source_a, source_b = a["metadata"].get("source_hashes", {}), b["metadata"].get("source_hashes", {})
    common_sources = set(source_a) & set(source_b)
    mismatch = [name for name in sorted(common_sources)
                if not name.replace("\\", "/").startswith("medivac_tree/") and source_a[name] != source_b[name]]
    if not common_sources:
        errors.append("no recorded common runtime source hashes")
    if mismatch:
        errors.append(f"common non-candidate runtime sources differ: {mismatch}")
    for episode, left in a["episodes"].items():
        right = b["episodes"][episode]
        valid = left["status"] == right["status"] == "complete"
        hashes_match = bool(valid and left["initial_state_sha256"] and left["initial_state_sha256"] == right["initial_state_sha256"])
        if valid and not hashes_match:
            errors.append(f"episode {episode}: initial states differ or hash missing")
        first = None
        for key in sorted(set(a["steps"]) & set(b["steps"])):
            if key[0] == episode and a["steps"][key]["actions"] != b["steps"][key]["actions"]:
                obs_a, obs_b = a["observations"].get(key), b["observations"].get(key)
                first = {"game_loop": key[1], "observation_payloads_equal": bool(obs_a is not None and obs_b is not None and canonical_hash(obs_a) == canonical_hash(obs_b))}
                break
        rows.append({"episode": episode, "valid_pair": valid and hashes_match,
                     "initial_states_match": hashes_match, "a_win": left["win"], "b_win": right["win"],
                     "win_difference_b_minus_a": int(right["win"]) - int(left["win"]) if valid and hashes_match else None,
                     "no_medivac_control": left["medivac_count"] == 0,
                     "action_sequences_equal": left["action_sequence_sha256"] == right["action_sequence_sha256"] if valid else None,
                     "first_action_divergence": first})
    valid_rows = [row for row in rows if row["valid_pair"]]
    return {"audit_errors": errors, "pairs": rows, "valid_pairs": len(valid_rows),
            "common_runtime_source_hashes_match": bool(common_sources) and not mismatch,
            "paired_win_difference": sum(row["win_difference_b_minus_a"] for row in valid_rows) / len(valid_rows) if valid_rows else None,
            "scope": "Four composition-selected pairs are integration evidence only; no population win-rate or transfer inference"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, metavar="LABEL=PATH")
    parser.add_argument("--baseline", default="legacy_off")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    runs, evidence = {}, {}
    for value in args.run:
        label, separator, path = value.partition("=")
        if not separator or not label or label in runs:
            raise ValueError("Each --run must have a unique LABEL=PATH")
        runs[label], evidence[label] = audit_run(Path(path))
        contract = {"legacy_off": ("legacy", "off"), "tree": ("tree", "off"),
                    "legacy_on": ("legacy", "on"), "shadow": ("tree-shadow", "off")}.get(label)
        if contract and (runs[label]["policy"], runs[label]["coordination"]) != contract:
            runs[label]["audit_errors"].append(f"arm metadata differs from expected policy/coordination {contract}")
    if args.baseline not in runs:
        raise ValueError(f"Baseline {args.baseline!r} not among runs")
    comparisons = {label: compare(evidence[args.baseline], value)
                   for label, value in evidence.items() if label != args.baseline}
    if args.baseline == "legacy_off":
        for label, item in comparisons.items():
            for pair in item["pairs"]:
                if pair.get("valid_pair") and (label == "shadow" or pair.get("no_medivac_control")) and not pair.get("action_sequences_equal"):
                    item["audit_errors"].append(f"episode {pair['episode']}: expected unchanged control action sequence differs")
    errors = [f"{label}: {error}" for label, run in runs.items() for error in run["audit_errors"]]
    errors += [f"{args.baseline}->{label}: {error}" for label, item in comparisons.items() for error in item["audit_errors"]]
    complete = all(run["completed_episodes"] == run["expected_episodes"] for run in runs.values())
    result = {"status": "passed" if complete and not errors else "failed", "baseline": args.baseline,
              "scope": "Integration smoke, no statistical superiority claim", "runs": runs,
              "comparisons": comparisons, "audit_errors": errors}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "analysis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    csv_rows = [dict(arm=label, **{key: value for key, value in row.items() if not isinstance(value, (list, dict))})
                for label, run in runs.items() for row in run["episodes"]]
    with (args.output_dir / "episodes.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(json.dumps({"status": result["status"], "output": str(args.output_dir.resolve()),
                      "runs": {key: {field: row[field] for field in ("completed_episodes", "invalid_episodes", "not_started_episodes", "wins")}
                               for key, row in runs.items()}, "audit_errors": errors}, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
