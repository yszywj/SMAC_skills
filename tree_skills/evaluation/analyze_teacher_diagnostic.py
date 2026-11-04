"""Validate and compare the four frozen-teacher diagnostic arms by cohort."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tree_skills.evaluation.paired_report import build_report, load_episodes, paired_stats
from tree_skills.integration.native_teacher import masked_native_argmax
from tree_skills.data.build_shard import analyze as analyze_collection, iter_rows
from tree_skills.teacher.action_mapping import SmacActionMap
from tree_skills.evaluation.analyze_tokenfix import intervention_execution_feedback


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def audit_events(run, arm):
    counts = Counter()
    medic = Counter()
    feedback_counts = Counter()
    current_obs = native = None
    episode_metrics = {}
    action_hashes = {}
    with (run / "events.jsonl").open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            event = row["event"]
            counts[event] += 1
            if event == "run_failed":
                raise ValueError(f"Failed event in {arm}")
            if event in ("skill_observation", "policy_observation"):
                current_obs = row
            elif event == "native_teacher_step":
                native = row
            elif event == "step":
                key = (row["episode"], row["game_loop"])
                if current_obs is None or (current_obs["episode"], current_obs["payload"]["state"]["game_loop"]) != key:
                    raise ValueError("Observation and submitted action boundary mismatch")
                payload = current_obs["payload"]
                state = payload["state"]
                actions = row["actions"]
                action_hashes.setdefault(str(row["episode"]), hashlib.sha256()).update(
                    json.dumps([row["game_loop"], actions], separators=(",", ":")).encode())
                step_feedback = row["feedback"]
                if step_feedback["response_mapping"] != "exact" or step_feedback["unmatched_observation_errors"]:
                    raise ValueError("Invalid engine feedback mapping")
                if arm in ("C", "D"):
                    if native is None or (native["episode"], native["game_loop"]) != key:
                        raise ValueError("Missing native decision")
                    decisions = native["decisions"]
                    if len(decisions) != len(actions) or {d["agent"] for d in decisions} != set(range(len(actions))):
                        raise ValueError("Incomplete native decision trace")
                    for d in decisions:
                        a = d["agent"]
                        expected_control = arm == "D" or d["unit_type"] == 54
                        if d["controlled"] != expected_control:
                            raise ValueError("Wrong controlled actor")
                        selected = masked_native_argmax(d["logits"], d["native_avail"], len(d["native_avail"]))
                        if selected != d["teacher_action"] or d["final_action"] != actions[a]:
                            raise ValueError("Teacher decision differs from submitted action")
                        if expected_control and actions[a] != selected:
                            raise ValueError("Teacher action overridden")
                        if not expected_control and actions[a] != d["rule_action"]:
                            raise ValueError("Ground rule action changed by medivac teacher")
                        if arm == "D" and d["rule_action"] is not None:
                            raise ValueError("Full teacher unexpectedly received rule actions")
                    native = None
                ep = episode_metrics.setdefault(str(row["episode"]), Counter())
                requested_heal_targets = Counter()
                for i, action in enumerate(actions):
                    actor = f"u{i}"
                    unit = state["allies"][actor]
                    f = step_feedback["actors"][actor]
                    if int(f["requested_action_id"]) != action:
                        raise ValueError("Engine feedback request mismatch")
                    if unit["unit_type_id"] != 54 or not unit["alive"]:
                        continue
                    kind = ("stop" if action == 1 else "move" if 2 <= action <= 5 else
                            "heal" if 6 <= action < 6 + len(actions) else
                            "keep" if action == 6 + len(actions) else "other")
                    medic[kind] += 1
                    ep[kind] += 1
                    feedback_counts[(f.get("response") or {}).get("name", f["submission"])] += 1
                    medic["observation_errors"] += len(f["observation_errors"])
                    legal = payload["questions"][actor]["criteria"]
                    wounded = [j for j in range(len(actions))
                               if str(6 + j) in legal and 0 < state["allies"][f"u{j}"]["health"]
                               < state["allies"][f"u{j}"]["max_health"]]
                    if len(wounded) >= 2:
                        medic["multiple_wounded_legal_targets"] += 1
                        ep["multiple_wounded_legal_targets"] += 1
                    if kind == "heal":
                        requested_heal_targets[f"u{action - 6}"] += 1
                        patient = state["allies"][f"u{action - 6}"]
                        if patient["health"] >= patient["max_health"]:
                            medic["heal_full_health_target"] += 1
                            ep["heal_full_health_target"] += 1
                        if unit["orders"] and unit["orders"][0].get("target_unit") == f"u{action - 6}":
                            medic["heal_same_observed_order_target"] += 1
                    elif kind == "keep" and unit["orders"]:
                        target = unit["orders"][0].get("target_unit")
                        if target is not None:
                            requested_heal_targets[target] += 1
                if sum(requested_heal_targets.values()) >= 2:
                    medic["steps_with_multiple_healing_medivacs"] += 1
                    ep["steps_with_multiple_healing_medivacs"] += 1
                    if max(requested_heal_targets.values()) >= 2:
                        medic["steps_with_shared_heal_target"] += 1
                        ep["steps_with_shared_heal_target"] += 1
                current_obs = None
    if arm == "D" and (counts["skill_step"] or counts["skill_started"]):
        raise ValueError("Rule policy executed in D")
    if arm in ("C", "D") and counts["native_teacher_step"] != counts["step"]:
        raise ValueError("Teacher/engine step count mismatch")
    if arm in ("A", "B", "C") and counts["skill_step"] != counts["step"]:
        raise ValueError("Missing rule decision trace")
    if current_obs is not None or native is not None:
        raise ValueError("Unsubmitted decision at end of run")
    return {"events": dict(counts), "medivac_alive_actions": dict(medic),
            "medivac_engine_feedback": dict(feedback_counts),
            "episode_medivac_metrics": {k: dict(v) for k, v in episode_metrics.items()},
            "episode_action_sha256": {k: v.hexdigest() for k, v in action_hashes.items()},
            "submitted_actions_and_teacher_permissions_verified": True}


def analyze(study, runs, output):
    if output.exists():
        raise FileExistsError(output)
    protocol = read(study / "protocol.json")
    manifest = read(study / "manifest.json")
    if protocol["manifest_sha256"] != manifest["sha256"]:
        raise ValueError("Protocol manifest mismatch")
    cohorts = {name: [i for i, e in enumerate(manifest["episodes"], 1) if e["study_group"] == name]
               for name in protocol["report_cohorts_separately"]}
    cohorts["diagnostic_medivac_combined"] = cohorts["diagnostic_one_medivac"] + cohorts["diagnostic_two_plus_medivac"]
    for label, present in (("native_with_medivac", True), ("native_without_medivac", False)):
        keys = [i for i in cohorts["native_distribution"]
                if ("medivac" in manifest["episodes"][i - 1]["episode_config"]["team_gen"]["ally_team"]) == present]
        if keys:
            cohorts[label] = keys
    episodes, summaries, audits, metadata = {}, {}, {}, {}
    for arm, run in runs.items():
        if read(run / "manifest.json") != manifest:
            raise ValueError(f"Different manifest: {arm}")
        summaries[arm] = read(run / "summary.json")
        metadata[arm] = read(run / "run.json")
        if summaries[arm]["status"] != "complete" or summaries[arm]["completed_episodes"] != len(manifest["episodes"]):
            raise ValueError(f"Incomplete arm: {arm}")
        episodes[arm] = load_episodes(run)
        if len(episodes[arm]) != len(manifest["episodes"]):
            raise ValueError("Incomplete episode events")
        audits[arm] = audit_events(run, arm)
    for arm in "BCD":
        if metadata[arm]["source_hashes"] != metadata["A"]["source_hashes"]:
            raise ValueError("Adapter/environment/skill source hashes changed between arms")
        teacher = summaries[arm]["pair_metadata"] if arm == "B" else summaries[arm]["native_teacher_metadata"]
        if (teacher["checkpoint_sha256"] != protocol["checkpoint_sha256"] or
                teacher["env_indicator"] != 0 or teacher["effective_history_len"] != 4):
            raise ValueError("Different teacher configuration")
    for i in cohorts.get("native_without_medivac", []):
        reference = audits["A"]["episode_action_sha256"][str(i)]
        for arm in "BC":
            if audits[arm]["episode_action_sha256"][str(i)] != reference:
                raise ValueError("A/B/C actions diverge without any medivac")
    pair_reports = {}
    for a, b in (("A", "B"), ("A", "C"), ("B", "C"), ("A", "D"), ("C", "D")):
        report = build_report(runs[a], runs[b])
        report["cohorts"] = {name: paired_stats(episodes[a], episodes[b], keys) for name, keys in cohorts.items()}
        pair_reports[f"{b}-{a}"] = report
    cohort_summary = {}
    for name, keys in cohorts.items():
        cohort_summary[name] = {}
        for arm in runs:
            e = episodes[arm]
            cohort_summary[name][arm] = {"n": len(keys), "wins": sum(e[k]["win"] for k in keys),
                "win_rate": sum(e[k]["win"] for k in keys) / len(keys),
                **{field: sum(e[k][field] for k in keys) / len(keys)
                   for field in ("enemy_elimination_fraction", "allied_survival_fraction", "steps")}}
    collection_paths = list((study / "B-collection").glob("collect-*.jsonl"))
    if len(collection_paths) != 1:
        raise ValueError("Expected one unambiguous B collection")
    collected = list(iter_rows(str(collection_paths[0])))
    b_steps = {}
    b_feedback = {}
    with (runs["B"] / "events.jsonl").open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["event"] == "step":
                b_steps[(row["episode"], row["game_loop"])] = row["actions"]
                b_feedback[(row["episode"], row["game_loop"])] = row
    seen = set()
    keep_recommendations = []
    for row in collected:
        key = (int(row["episode"][2:]), row["game_loop"])
        if key in seen or b_steps.get(key) != row["final_actions"]:
            raise ValueError("B collection duplicate or submitted-action mismatch")
        seen.add(key)
        for healer, change in ((row.get("pair_scoring") or {}).get("changed") or {}).items():
            if row["final_actions"][int(healer[1:])] == 16:
                keep = (row.get("healing", {}).get("keep_decisions") or {}).get(healer) or {}
                keep_recommendations.append({"episode": key[0], "game_loop": key[1],
                    "healer": healer, "proposed_target": change["to"], "retained_target": keep.get("target"),
                    "same_as_proposed": keep.get("target") == change["to"], "keep_decision": keep})
    if seen != set(b_steps):
        raise ValueError("Incomplete B collection")
    collection_stats = {name: analyze_collection(
        (row for row in collected if int(row["episode"][2:]) in keys), SmacActionMap())
        for name, keys in cohorts.items()}
    intervention_feedback = intervention_execution_feedback(collected, b_feedback, SmacActionMap())
    result = {"protocol": protocol, "cohorts": cohort_summary, "pairs": pair_reports,
              "audits": audits, "run_summaries": summaries,
              "B_collection_by_cohort": collection_stats,
              "B_changed_recommendations_execution": intervention_feedback["counts"],
              "B_keep_recommendations": keep_recommendations,
              "source_hashes_match": True,
              "interpretation": "Small new-seed mechanism screen. Native draws and composition-selected diagnostic strata are reported separately. These are not independent samples across arms. No training or zero-shot qualification claim."}
    output.mkdir(parents=True)
    (output / "analysis.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (output / "B_intervention_execution_feedback.json").write_text(
        json.dumps(intervention_feedback, indent=2), encoding="utf-8")
    with (output / "episodes.csv").open("w", newline="", encoding="utf-8-sig") as f:
        fields = ["episode", "seed", "cohort", "medivacs"] + [f"{a}_{k}" for a in runs for k in ("win", "steps", "enemy_elimination_fraction", "allied_survival_fraction")]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i, e in enumerate(manifest["episodes"], 1):
            row = {"episode": i, "seed": e["sc2_seed"], "cohort": e["study_group"],
                   "medivacs": e["episode_config"]["team_gen"]["ally_team"].count("medivac")}
            row.update({f"{a}_{k}": episodes[a][i][k] for a in runs
                        for k in ("win", "steps", "enemy_elimination_fraction", "allied_survival_fraction")})
            writer.writerow(row)
    print(json.dumps({"output": str(output), "cohorts": cohort_summary}, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study-dir", type=Path, required=True)
    p.add_argument("--run-prefix", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    analyze(args.study_dir, {a: Path(str(args.run_prefix) + a) for a in "ABCD"}, args.output_dir)
