"""Read-only v3 trace statistics. Prints JSON; never launches games or writes runs.

This is descriptive analysis, not a replacement for native/counterfactual audit.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import statistics


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def distribution(values):
    values = list(values)
    return {"n": len(values), "min": min(values), "median": statistics.median(values),
            "mean": statistics.mean(values), "max": max(values)} if values else {"n": 0}


def family(binding):
    return binding.split(":")[0]


def summarize_run(path, group_ids=None):
    path = Path(path).resolve()
    if read(path / "summary.json").get("status") != "complete":
        raise ValueError("Only completed runs may be summarized: " + str(path))
    episodes = read(path / "episodes.json")
    if group_ids is not None:
        episodes = [row for row in episodes if row["scenario_sha256"] in set(group_ids)]
    included_episodes = {row["episode"] for row in episodes}
    metadata = read(path / "run.json")
    count, by_family, gates, family_options = Counter(), Counter(), Counter(), Counter()
    layer_counts = defaultdict(Counter)
    episode_counts = defaultdict(Counter)
    episode_family = defaultdict(Counter)
    per_actor = defaultdict(Counter)
    actor_previous, global_previous, previous_decision = {}, {}, {}
    actor_gaps, global_gaps, loop_gaps = [], [], []
    pending = {}
    anomalies = []
    digest = hashlib.sha256()

    def anomaly(kind, event, **extra):
        count["anomalies"] += 1
        if len(anomalies) < 30:
            anomalies.append({"kind": kind, "episode": event.get("episode"),
                              "step": event.get("step"), **extra})

    with (path / "events.jsonl").open("rb") as stream:
        for line in stream:
            digest.update(line)
            event = json.loads(line)
            kind, episode = event["event"], event.get("episode")
            if episode not in included_episodes:
                continue
            if kind == "conditional_local_decision":
                step, selected = event["step"], event["selected_id"]
                base, final = event["baseline_actions"], event["final_actions"]
                changed = [i for i, (a, b) in enumerate(zip(base, final)) if a != b]
                count["decision_steps"] += 1
                episode_counts[episode]["decision_steps"] += 1
                count["forced_steps"] += int(event.get("forced", False))
                count["empty_catalog_steps"] += int(len(event["catalog"]["effects"]) == 1)
                count["catalog_nonbase_effects"] += len(event["catalog"]["effects"]) - 1
                for row in event["catalog"]["bindings"]:
                    family_options[row["family"]] += len(row["options"])
                for row in event.get("considered", []):
                    gates[row["family"] + ":" + row["reason"]] += 1
                if len(base) != len(final) or len(changed) > 1 or bool(changed) != (selected != "BASE"):
                    anomaly("changed_command_count", event, changed=changed, selected=selected)
                previous = previous_decision.get(episode)
                if previous and previous["step"] + 1 == step and previous["selected_id"] != "BASE":
                    actor_index = int(previous["selected_id"].split(":")[1][1:])
                    old_base = previous["baseline_actions"][actor_index]
                    old_final = previous["final_actions"][actor_index]
                    if final[actor_index] == old_base and base[actor_index] == old_base and old_base != old_final:
                        count["next_step_same_actor_restored_prior_baseline_action_id"] += 1
                    if final[actor_index] == old_final:
                        count["next_step_same_actor_repeated_prior_action_id"] += 1
                previous_decision[episode] = event
                diagnostics = event.get("corrections", [])
                modifications = 0
                for index, diagnostic in enumerate(diagnostics):
                    before, after = diagnostic["before"], diagnostic["after"]
                    layer_counts[index]["evaluations"] += 1
                    if index and before != (diagnostics[index - 1]["before"] if
                                           diagnostics[index - 1]["after"] == "retain" else
                                           diagnostics[index - 1]["after"]):
                        anomaly("broken_correction_chain", event, layer=index)
                    if after == "retain" or after == before:
                        layer_counts[index]["retained"] += 1
                        continue
                    modifications += 1
                    layer_counts[index]["changed_selection"] += 1
                    action = "cancelled_to_BASE" if after == "BASE" else "activated_from_BASE" if before == "BASE" else "replaced_nonbase"
                    layer_counts[index][action] += 1
                    episode_counts[episode]["correction_" + action] += 1
                if diagnostics:
                    old, last = diagnostics[0]["before"], diagnostics[-1]
                    new = last["before"] if last["after"] == "retain" else last["after"]
                    count["steps_with_correction_selection_change"] += int(new != old)
                    count["steps_with_multiple_layer_changes"] += int(modifications > 1)
                    count["steps_corrections_cancelled_original_nonbase"] += int(old != "BASE" and new == "BASE")
                    count["steps_corrections_replaced_original_nonbase"] += int(old != "BASE" and new not in ("BASE", old))
                    if new != event.get("natural_selected_id", new):
                        anomaly("natural_selection_chain_mismatch", event)
                if changed:
                    actor = "u" + str(changed[0])
                    _, selected_actor, action_id = selected.split(":")
                    if actor != selected_actor or final[changed[0]] != int(action_id):
                        anomaly("selected_binding_mismatch", event)
                    count["changed_command_submissions"] += 1
                    episode_counts[episode]["changed_command_submissions"] += 1
                    by_family[family(selected)] += 1
                    episode_family[episode][family(selected)] += 1
                    per_actor[episode][actor] += 1
                    actor_key = episode, actor
                    if actor_key in actor_previous:
                        prev = actor_previous[actor_key]
                        gap = step - prev["step"]
                        actor_gaps.append(gap)
                        count["consecutive_actor_interventions_same_action_id"] += int(prev["selected_id"].split(":")[-1] == action_id)
                        if gap < event["conditional_policy"]["actor_cooldown_steps"]:
                            field = "forced_cooldown_bypasses" if event.get("forced") else "unforced_cooldown_violations"
                            count[field] += 1
                            if not event.get("forced"):
                                anomaly("unforced_cooldown", event, actor=actor, gap=gap)
                    actor_previous[actor_key] = event
                    if episode in global_previous:
                        prev = global_previous[episode]
                        global_gaps.append(step - prev["step"])
                        loop_gaps.append(event["game_loop"] - prev["game_loop"])
                    global_previous[episode] = event
                    pending[episode, step + 1] = (actor, final)
                if event["interventions"] != episode_counts[episode]["changed_command_submissions"]:
                    anomaly("intervention_counter", event)
            elif kind == "step":
                key = episode, event["step"]
                if key in pending:
                    actor, submitted = pending.pop(key)
                    receipt = event.get("feedback", {}).get("actors", {}).get(actor, {})
                    count["changed_command_receipt_" + receipt.get("submission", "missing")] += 1
                    count["changed_command_receipts_with_observation_errors"] += int(bool(receipt.get("observation_errors")))
                    if event["actions"] != submitted:
                        anomaly("native_action_mismatch", event)
    count["missing_changed_command_receipts"] = len(pending)
    rows = []
    for row in episodes:
        episode = row["episode"]
        rows.append({"episode": episode, "scenario_sha256": row["scenario_sha256"],
                     "outcome": row["outcome"], "return": row["reward"], "steps": row["steps"],
                     **episode_counts[episode], "by_family": dict(episode_family[episode]),
                     "by_actor": dict(per_actor[episode])})
    return {"run": str(path), "events_sha256": digest.hexdigest(),
            "policy_sha256": metadata.get("conditional_skills", {}).get("policy_sha256"),
            "scene_count": len(episodes), "outcomes": dict(Counter(r["outcome"] for r in episodes)),
            "selected_group_ids": sorted(r["scenario_sha256"] for r in episodes),
            "mean_return": statistics.mean(r["reward"] for r in episodes) if episodes else None,
            "step_mul": metadata.get("env_args", {}).get("step_mul"), "counts": dict(count),
            "executed_by_family": dict(by_family), "considered_binding_reasons": dict(gates),
            "family_catalog_option_occurrences": dict(family_options),
            "correction_layers": {str(k): dict(v) for k, v in layer_counts.items()},
            "interventions_per_episode": distribution(episode_counts[r["episode"]]["changed_command_submissions"] for r in episodes),
            "intervention_gap_decision_steps": distribution(global_gaps),
            "intervention_gap_game_loops": distribution(loop_gaps),
            "same_actor_intervention_gap_decision_steps": distribution(actor_gaps),
            "anomaly_examples": anomalies, "episodes": rows,
            "interpretation": ["Layer selection changes are not additional native commands; only the final selected patch executes.",
                               "Repeated action IDs and next-step restoration are descriptive, not evidence of wasted actions or causal harm.",
                               "Native acceptance does not establish firing, damage, or winning.",
                               "Statistics do not replace the campaign's full replay and paired-state audit."]}


def compare_runs(candidate, reference, group_ids=None):
    candidate = {r["scenario_sha256"]: r for r in read(Path(candidate) / "episodes.json")}
    reference = {r["scenario_sha256"]: r for r in read(Path(reference) / "episodes.json")}
    rows, counts = [], Counter()
    groups = candidate.keys() & reference.keys()
    if group_ids is not None:
        groups &= set(group_ids)
    for group in sorted(groups):
        a, b = candidate[group], reference[group]
        if a["initial_state_sha256"] != b["initial_state_sha256"]:
            raise ValueError("Mismatched initial state: " + group)
        won_a, won_b = a["outcome"] == "win", b["outcome"] == "win"
        kind = "rescued_win" if won_a and not won_b else "lost_win" if won_b and not won_a else "both_win" if won_a else "both_nonwin"
        counts[kind] += 1
        rows.append({"scenario_sha256": group, "candidate_episode": a["episode"], "reference_episode": b["episode"],
                     "kind": kind, "candidate_outcome": a["outcome"], "reference_outcome": b["outcome"],
                     "return_gain": a["reward"] - b["reward"]})
    return {"paired_scenes": len(rows), "counts": dict(counts),
            "mean_return_gain": statistics.mean(r["return_gain"] for r in rows) if rows else None,
            "episodes": rows}


def summarize_round(root, index, partition="train"):
    folder = Path(root).resolve() / f"round_{index}"
    if partition not in ("train", "holdout", "all"):
        raise ValueError("Unknown partition")
    if partition != "train" and not (folder / "freeze.json").exists():
        raise ValueError("Freeze the candidate policy before summarizing holdout results")
    split = read(folder / "split.json")
    groups = set(split["training_group_ids"] if partition == "train" else
                 split["validation_group_ids"] if partition == "holdout" else
                 split["training_group_ids"] + split["validation_group_ids"])
    runs = {}
    for name in ("adaptive", "incumbent", "candidate_train", "candidate_holdout"):
        if (partition == "train" and name == "candidate_holdout") or (partition == "holdout" and name == "candidate_train"):
            continue
        path = folder / name
        if (path / "summary.json").exists() and read(path / "summary.json").get("status") == "complete":
            runs[name] = summarize_run(path, groups)
    comparisons = {}
    for name in ("incumbent", "candidate_train", "candidate_holdout"):
        if name in runs:
            for reference in ("adaptive", "incumbent"):
                if name != reference and reference in runs:
                    comparisons[name + "_vs_" + reference] = compare_runs(folder / name, folder / reference, groups)
    plan_path = folder / "branch_plan.json"
    plan = read(plan_path) if plan_path.exists() else None
    coverage = None
    if plan:
        measured = [r for r in (read(p) for p in sorted((folder / "branches").glob("*.pair.json"))) if r["scene_hash"] in groups]
        jobs = [r for r in plan["jobs"] if r["scene_hash"] in groups]
        selected_coverage = [r for r in plan["coverage"] if r["scene_hash"] in groups]
        coverage = {"planned": len(jobs), "completed_audited_pairs": len(measured),
                    "planned_by_family_and_label": dict(Counter(r["family"] + ":" + r["label_kind"] for r in jobs)),
                    "complete_by_family_and_label": dict(Counter(r["family"] + ":" + r["label_kind"] for r in measured)),
                    "scene_family_coverage": dict(Counter(r["family"] + ":" + r["status"] for r in selected_coverage)),
                    "planned_temporal_phases": dict(Counter(r["phase"] for r in jobs)),
                    "bounds": plan["bounds"], "census": {k: v for k, v in plan["census"].items() if partition == "all" or k == partition}}
    return {"root": str(Path(root).resolve()), "round": index, "partition": partition,
            "selected_group_ids": sorted(groups), "runs": runs,
            "whole_policy_comparisons": comparisons, "exploration": coverage}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run", type=Path)
    source.add_argument("--root", type=Path)
    parser.add_argument("--round", type=int, choices=(1, 2, 3))
    parser.add_argument("--partition", choices=("train", "holdout", "all"), default="train",
                        help="Root mode defaults to train; holdout/all must wait until candidate freeze.")
    parser.add_argument("--group-ids", type=Path, help="Run mode: JSON list of allowed scene hashes.")
    args = parser.parse_args()
    if args.root and args.round is None:
        parser.error("--root requires --round")
    if args.root and args.group_ids:
        parser.error("--group-ids is for --run; use --partition with --root")
    groups = read(args.group_ids) if args.group_ids else None
    if groups is not None and (not isinstance(groups, list) or not all(isinstance(x, str) for x in groups)):
        parser.error("--group-ids must contain a JSON list of scene hashes")
    result = summarize_run(args.run, groups) if args.run else summarize_round(args.root, args.round, args.partition)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
