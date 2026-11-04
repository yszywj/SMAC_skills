"""Reproduce the corrected-token experiment analysis after BOTH runs finish.

The old trajectory and its corrected labels are compared on identical
observations. New live B labels are kept in a separate shard, never joined
with old trajectories by their locally numbered episode IDs.
"""

import argparse
from collections import Counter, defaultdict
import csv
import json
import statistics
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from tree_skills.data.build_shard import (analyze, build_shard, iter_rows,
                                         native_teacher_diagnostics)
from tree_skills.evaluation.paired_report import (build_report, load_episodes,
                                                 load_manifest_medivac)
from tree_skills.scripts.relabel_collection import compare_labels, sha256
from tree_skills.teacher.action_mapping import SmacActionMap
from tree_skills.training.train_tree import (ConstantZeroScorer, evaluate_ranking,
                                            load_groups)


def read_json(path):
    with Path(path).open(encoding="utf-8") as source:
        return json.load(source)


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as out:
        json.dump(value, out, indent=2)


def load_run_evidence(run):
    summary = read_json(run / "summary.json")
    if summary.get("status") != "complete":
        raise ValueError(f"run is not complete: {run}")
    steps, configurations = {}, []
    with (run / "events.jsonl").open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            if row.get("event") == "step":
                key = (int(row["episode"]), row["game_loop"])
                if key in steps:
                    raise ValueError(f"duplicate environment step: {key}")
                steps[key] = {k: row[k] for k in ("step", "game_loop", "actions")}
                steps[key]["feedback"] = row.get("feedback")
            elif row.get("event") == "pair_configuration":
                configurations.append(row)
    return summary, steps, configurations


def action_divergences(a_steps, b_steps):
    """Compare matched loops; later divergences can be cascading consequences."""
    episodes = sorted({ep for ep, _ in a_steps} | {ep for ep, _ in b_steps})
    details = []
    for episode in episodes:
        a = {loop: row for (ep, loop), row in a_steps.items() if ep == episode}
        b = {loop: row for (ep, loop), row in b_steps.items() if ep == episode}
        common = sorted(set(a) & set(b))
        differing = [loop for loop in common if a[loop]["actions"] != b[loop]["actions"]]
        first = differing[0] if differing else None
        details.append({"episode": episode, "matched_loops": len(common),
                        "divergent_matched_loops": len(differing),
                        "loops_only_in_a": sorted(set(a) - set(b)),
                        "loops_only_in_b": sorted(set(b) - set(a)),
                        "first_action_divergence": None if first is None else {
                            "game_loop": first, "step_a": a[first]["step"],
                            "step_b": b[first]["step"],
                            "actions_a": a[first]["actions"], "actions_b": b[first]["actions"],
                            "changed_unit_indices": [i for i, (x, y) in
                                enumerate(zip(a[first]["actions"], b[first]["actions"])) if x != y]}})
    return {"interpretation": "Only the first divergence precedes action-induced trajectory branching. Later mismatches include cascades and are not independent teacher interventions.",
            "episodes_with_action_divergence": sum(d["first_action_divergence"] is not None for d in details),
            "matched_loops": sum(d["matched_loops"] for d in details),
            "divergent_matched_loops_including_cascades": sum(d["divergent_matched_loops"] for d in details),
            "episodes": details}


def attach_live_provenance(rows, run, config, source_files, b_steps):
    if config.get("env_indicator") != 0 or config.get("last_token") != "on":
        raise ValueError("live run does not declare the corrected token=0 / last_token=on teacher")
    if config.get("failure_policy") != "strict":
        raise ValueError("live teacher run is not strict")
    if not config.get("checkpoint_sha256"):
        raise ValueError("live teacher checkpoint hash is missing")
    provenance = {"kind": "live_corrected_teacher_policy_trajectory",
                  "run_directory": str(run), "run_metadata_sha256": sha256(run / "run.json"),
                  "run_summary_sha256": sha256(run / "summary.json"),
                  "source_collections": {str(p): sha256(p) for p in source_files},
                  "teacher_configuration": config,
                  "episode_namespace": run.name,
                  "dataset_role": "screening/diagnostics; not independent final-test evidence"}
    for row in rows:
        episode = int(row["episode"].removeprefix("ep"))
        key = (episode, row["game_loop"])
        if key not in b_steps or b_steps[key]["actions"] != row["final_actions"]:
            raise ValueError(f"collection actions do not match live B event log: {key}")
        row["label_provenance"] = provenance
    return provenance


def zero_baseline(groups):
    metrics = evaluate_ranking(ConstantZeroScorer(), groups)
    metrics.pop("constant_zero_baseline", None)
    return metrics


def teacher_intent(rows, mapping):
    """Contrast full legal-action preferences with restricted heal candidates."""
    scopes = {name: {"groups": 0, "stop": 0, "move": 0, "heal": 0, "noop": 0,
                     "preferred_heal_inside_candidates": 0,
                     "preferred_heal_outside_candidates": 0,
                     "stop_probabilities": [], "legal_heal_probabilities": []}
              for name in ("all", "multi_candidate")}
    for row in rows:
        for healer, entry in (row.get("healers") or {}).items():
            candidates = entry.get("candidates") or {}
            diag = native_teacher_diagnostics(row, healer, mapping)
            action = diag["teacher_native_argmax"]
            kind = "noop" if action == 0 else "stop" if action == 1 else "move" if action < 6 else "heal"
            candidate_slots = {mapping.heal_slot(int(t[1:])) for t in candidates}
            for name in (("all", "multi_candidate") if len(candidates) > 1 else ("all",)):
                scope = scopes[name]
                scope["groups"] += 1
                scope[kind] += 1
                if kind == "heal":
                    key = "preferred_heal_inside_candidates" if action in candidate_slots else "preferred_heal_outside_candidates"
                    scope[key] += 1
                scope["stop_probabilities"].append(diag["teacher_stop_probability"])
                scope["legal_heal_probabilities"].append(diag["teacher_legal_heal_probability"])
    for scope in scopes.values():
        for key in ("stop_probabilities", "legal_heal_probabilities"):
            values = scope.pop(key)
            scope[key] = {"mean": statistics.mean(values) if values else None,
                          "median": statistics.median(values) if values else None}
    return {"interpretation": "Masked actor preferences and temperature-1 softmax probabilities, not Q-values or evidence that the preferred action is tactically correct.", **scopes}


def intervention_execution_feedback(rows, b_steps, mapping):
    """Join each changed recommendation to that actor's actual SC2 response.

    Submission of an action, engine acceptance and delivered healing are
    different facts. AlreadyTargeted can mean a pre-existing heal order is
    continuing; neither it nor Success establishes actual healing benefit.
    """
    counts = dict(recommended_changes=0, submitted=0, not_submitted=0,
                  accepted_success=0, already_targeted=0, rejected_other=0,
                  unknown=0, feedback_error=0, feedback_mismatch=0,
                  with_observation_errors=0, observation_error_count=0)
    outcomes, response_distribution, records = Counter(), Counter(), []
    for row in rows:
        episode = int(row["episode"].removeprefix("ep"))
        key = (episode, row["game_loop"])
        event = b_steps.get(key) or {}
        actors = (event.get("feedback") or {}).get("actors") or {}
        for healer, change in ((row.get("pair_scoring") or {}).get("changed") or {}).items():
            counts["recommended_changes"] += 1
            index = int(healer[1:])
            actions = row.get("final_actions") or []
            final_action = actions[index] if index < len(actions) else None
            expected = mapping.heal_slot(int(change["to"][1:]))
            actor = actors.get(healer)
            actor = actor if isinstance(actor, dict) else {}
            response = actor.get("response") or {}
            command = actor.get("command") or {}
            errors = actor.get("observation_errors") or []
            submitted = final_action == expected
            counts["submitted" if submitted else "not_submitted"] += 1
            issue = None
            if not submitted:
                outcome = "not_submitted"
            elif not actor:
                outcome, issue = "unknown", "missing_actor_feedback"
            elif (str(actor.get("requested_action_id")) != str(expected)
                  or command.get("target_unit") != change["to"]
                  or command.get("kind") != "heal"):
                outcome, issue = "feedback_mismatch", "actor_feedback_does_not_describe_recommended_heal"
            elif (actor.get("submission") == "accepted"
                  and response.get("code") == 1 and response.get("name") == "Success"):
                outcome = "accepted_success"
            elif (actor.get("submission") == "rejected"
                  and response.get("code") == 206 and response.get("name") == "AlreadyTargeted"):
                outcome = "already_targeted"
            elif actor.get("submission") in ("error", "failed") or actor.get("error"):
                outcome = "feedback_error"
            elif actor.get("submission") == "rejected":
                outcome = "rejected_other"
            else:
                outcome, issue = "unknown", "unrecognized_or_incomplete_submission_response"
            if submitted:
                counts[outcome] += 1
                response_distribution[f"{actor.get('submission')} / {response.get('name')} / {response.get('code')}"] += 1
            if errors:
                counts["with_observation_errors"] += 1
                counts["observation_error_count"] += len(errors)
            outcomes[outcome] += 1
            records.append({"episode": episode, "game_loop": row["game_loop"],
                "healer": healer, "original_target": change["from"], "proposed_target": change["to"],
                "expected_heal_action": expected, "final_action": final_action,
                "submitted": submitted, "execution_outcome": outcome, "diagnostic_issue": issue,
                "submission": actor.get("submission"), "response": response,
                "command": command, "observation_errors": errors})
    classified = sum(counts[name] for name in ("accepted_success", "already_targeted",
                     "rejected_other", "unknown", "feedback_error", "feedback_mismatch"))
    if classified != counts["submitted"]:
        raise AssertionError("execution outcomes do not partition submitted recommendations")
    return {"interpretation": "Submitted means the recommended action ID reached the final action vector. Success means SC2 accepted the command, not that healing occurred. AlreadyTargeted is a rejected redundant targeting request and may leave an existing heal order running; it is not evidence of zero effect. Observation errors can concern orders already active during the interval and are reported separately.",
            "counts": counts, "outcomes": dict(outcomes),
            "submitted_response_distribution": dict(response_distribution), "records": records}


def analyze_experiment(run_a, run_b, old_collect, relabel_dir, new_collect, output_dir, rounds=10000):
    paths = [Path(p).resolve() for p in (run_a, run_b, old_collect, relabel_dir, new_collect, output_dir)]
    run_a, run_b, old_collect, relabel_dir, new_collect, output_dir = paths
    if output_dir.exists():
        raise ValueError(f"refusing to overwrite existing output directory: {output_dir}")
    # Check completeness and pair identity BEFORE writing any output.
    _, a_steps, _ = load_run_evidence(run_a)
    _, b_steps, configs = load_run_evidence(run_b)
    if len(configs) != 1:
        raise ValueError("expected exactly one teacher configuration event")
    paired = build_report(str(run_a), str(run_b), rounds=rounds)
    relabel_manifest = read_json(relabel_dir / "relabel_manifest.json")
    if relabel_manifest.get("status") != "complete":
        raise ValueError("old-trajectory relabel is incomplete")
    if relabel_manifest["source_sha256"] != sha256(old_collect):
        raise ValueError("old collection hash does not match relabel provenance")
    if relabel_manifest["checkpoint_sha256"] != configs[0]["checkpoint_sha256"]:
        raise ValueError("offline relabel and live teacher use different checkpoints")
    relabelled_path = relabel_dir / "collect-corrected-labels-on-old-trajectory.jsonl"
    if relabel_manifest["collection_sha256"] != sha256(relabelled_path):
        raise ValueError("relabelled collection hash mismatch")
    old_rows, relabelled = list(iter_rows(str(old_collect))), list(iter_rows(str(relabelled_path)))
    if [(r["episode"], r["game_loop"]) for r in old_rows] != [(r["episode"], r["game_loop"]) for r in relabelled]:
        raise ValueError("old/relabelled collections are not the same trajectory")
    for old, corrected in zip(old_rows, relabelled):
        if old["teacher_raw"]["agents"] != corrected["teacher_raw"]["agents"] or old["final_actions"] != corrected["final_actions"]:
            raise ValueError("old/relabelled observations, availability masks or recorded actions changed")
    source_files = [new_collect] if new_collect.is_file() else sorted(new_collect.glob("collect-*.jsonl"))
    if len(source_files) != 1:
        raise ValueError("new collection must contain exactly one run, preventing episode-ID collisions")
    live = list(iter_rows(str(source_files[0])))
    provenance = attach_live_provenance(live, run_b, configs[0], source_files, b_steps)
    metadata = read_json(run_b / "run.json")
    capability = metadata["env_args"]["capability_config"]
    mapping = SmacActionMap(n_agents=capability["n_units"], n_enemies=capability["n_enemies"])
    comparison = compare_labels(old_rows, relabelled, mapping)
    comparison["scope"] = "identical old-policy observations; label change, not a new rollout"
    interventions = analyze(live, mapping)
    execution = intervention_execution_feedback(live, b_steps, mapping)
    divergence = action_divergences(a_steps, b_steps)
    output_dir.mkdir(parents=True)
    shard = output_dir / "live_corrected_teacher_shard.jsonl"
    written, skipped = build_shard(live, mapping, str(shard))
    if sum(skipped.values()):
        raise ValueError(f"live label construction skipped groups: {skipped}")
    old_shard = output_dir / "old_token1_labels_diagnostic_only.jsonl"
    old_written, old_skipped = build_shard(old_rows, mapping, str(old_shard))
    baselines = {
        "interpretation": "All scorers return zero without training. Each dataset is evaluated separately; no episode IDs are joined across trajectories.",
        "old_token1_on_old_trajectory": zero_baseline(load_groups(str(old_shard))),
        "corrected_token0_on_old_trajectory": zero_baseline(load_groups(str(relabel_dir / "corrected_labels_shard.jsonl"))),
        "corrected_token0_on_new_live_trajectory": zero_baseline(load_groups(str(shard)))}
    a, b = load_episodes(str(run_a)), load_episodes(str(run_b))
    medivac = load_manifest_medivac(str(run_a))
    live_by_episode = defaultdict(list)
    for row in live:
        live_by_episode[int(row["episode"].removeprefix("ep"))].append(row)
    episode_stats = []
    for d in divergence["episodes"]:
        ep = d["episode"]
        stats = analyze(live_by_episode[ep], mapping)
        execution_counts = intervention_execution_feedback(live_by_episode[ep], b_steps, mapping)["counts"]
        first = d["first_action_divergence"]
        episode_stats.append({"episode": ep, "n_medivac": medivac[ep],
            "win_a": a[ep]["win"], "win_b": b[ep]["win"],
            "win_difference_b_minus_a": int(b[ep]["win"]) - int(a[ep]["win"]),
            "steps_a": a[ep]["steps"], "steps_b": b[ep]["steps"],
            "enemy_elimination_a": a[ep]["enemy_elimination_fraction"],
            "enemy_elimination_b": b[ep]["enemy_elimination_fraction"],
            "allied_survival_a": a[ep]["allied_survival_fraction"],
            "allied_survival_b": b[ep]["allied_survival_fraction"],
            "multi_candidate_groups": stats["multi_candidate_groups"],
            "recommended_changes": stats["recommended_changes"],
            "recommendations_submitted": stats["recommendations_submitted"],
            "recommendations_kept": stats["recommendations_kept"],
            "engine_accepted_success": execution_counts["accepted_success"],
            "engine_already_targeted": execution_counts["already_targeted"],
            "engine_rejected_other": execution_counts["rejected_other"],
            "engine_feedback_unknown": execution_counts["unknown"],
            "engine_feedback_error": execution_counts["feedback_error"],
            "engine_feedback_mismatch": execution_counts["feedback_mismatch"],
            "first_divergent_loop": first["game_loop"] if first else None,
            "divergent_matched_loops_including_cascades": d["divergent_matched_loops"]})
    results = {"paired_report": paired, "same_observation_label_comparison": comparison,
               "live_interventions": interventions, "constant_zero_baselines": baselines,
               "intervention_execution_feedback": execution,
               "teacher_intent": {"old_token1_on_old_trajectory": teacher_intent(old_rows, mapping),
                                  "corrected_token0_on_old_trajectory": teacher_intent(relabelled, mapping),
                                  "corrected_token0_on_new_live_trajectory": teacher_intent(live, mapping)},
               "action_divergence": divergence, "episodes": episode_stats,
               "live_shard": {"path": str(shard), "rows": written, "skipped": skipped,
                              "sha256": sha256(shard), "provenance": provenance},
               "old_diagnostic_shard": {"rows": old_written, "skipped": old_skipped},
               "analysis_script_sha256": sha256(Path(__file__))}
    for name, value in results.items():
        if isinstance(value, (dict, list)):
            write_json(output_dir / f"{name}.json", value)
    write_json(output_dir / "analysis.json", results)
    with (output_dir / "episodes.csv").open("x", encoding="utf-8", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=list(episode_stats[0]))
        writer.writeheader()
        writer.writerows(episode_stats)
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run-a", "run-b", "old-collect", "relabel-dir", "new-collect", "output-dir"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--bootstrap", type=int, default=10000)
    args = parser.parse_args(argv)
    results = analyze_experiment(args.run_a, args.run_b, args.old_collect,
        args.relabel_dir, args.new_collect, args.output_dir, args.bootstrap)
    print(json.dumps({"paired": results["paired_report"]["groups"],
                      "same_observation": results["same_observation_label_comparison"],
                      "interventions": results["live_interventions"],
                      "execution_feedback": results["intervention_execution_feedback"]["counts"],
                      "live_shard": results["live_shard"]["path"]}, indent=2))


if __name__ == "__main__":
    main()
