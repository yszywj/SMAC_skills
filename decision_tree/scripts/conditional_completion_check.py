"""Read-only evidence checks for a finished bounded conditional 3 x 40 campaign.

This does not launch SC2 or replay large traces. It verifies the small artifacts
and the native attempt ledger, retaining the original audit-only exit-1 entry.
Only a fully successful check may create completion_checks.json.
"""
import argparse
from collections import Counter
from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

from decision_tree.conditional.objective import adoption_verdict, pair_labels, summarize_pairs
from decision_tree.conditional.planning import check_budget
from decision_tree.local.runtime import digest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
AUDIT_ERROR = "Conditional policy, actions, candidates or counters differ from replay"
SMALL_EVIDENCE = ("run.json", "summary.json", "manifest.json", "episodes.json")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    path = Path(path)
    require(path.is_file(), "Missing completion evidence: " + str(path))
    require(path.stat().st_size <= 64 * 1024 ** 2, "Unexpectedly large JSON evidence: " + str(path))
    return json.loads(path.read_text(encoding="utf-8-sig"))


def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inside(root, path):
    path = Path(path).resolve()
    require(path.is_relative_to(root) and path != root, "Evidence path escaped campaign: " + str(path))
    return path


def manifest_groups(manifest):
    require(manifest.get("version") == 1 and manifest.get("sha256") == digest(
        {key: value for key, value in manifest.items() if key != "sha256"}), "Manifest digest mismatch")
    scenes = manifest["episodes"]
    require([scene["index"] for scene in scenes] == list(range(len(scenes))), "Manifest indices changed")
    for scene in scenes:
        require(scene["sha256"] == digest(scene["episode_config"]), "Scene configuration digest mismatch")
    groups = [scene["sha256"] for scene in scenes]
    require(len(groups) == len(set(groups)), "Duplicate scene in manifest")
    return groups


def subset_manifest(manifest, groups):
    selected = set(groups)
    body = {key: value for key, value in manifest.items() if key != "sha256"}
    body["episodes"] = [dict(scene, index=index) for index, scene in enumerate(
        scene for scene in manifest["episodes"] if scene["sha256"] in selected)]
    return dict(body, sha256=digest(body))


def whole_policy(candidate, reference, groups):
    rows = []
    for group in groups:
        left, right = candidate[group], reference[group]
        require(left["initial_state_sha256"] == right["initial_state_sha256"], "Whole-policy initial state mismatch")
        rows.append({"group_id": group, "pair_id": group, "scope": "whole_policy",
                     "candidate": {"return": left["reward"], "outcome": left["outcome"]},
                     "reference": {"return": right["reward"], "outcome": right["outcome"]}})
    return summarize_pairs(rows, scope="whole_policy", expected_groups=groups)


def check_campaign(root, *, project_root=PROJECT_ROOT):
    """Return checks only after all three rounds pass; never mutate input files."""
    root, project_root = Path(root).resolve(), Path(project_root).resolve()
    # Fail early while a campaign is incomplete, before examining its live ledger.
    for index in range(1, 4):
        read(root / f"round_{index}" / "evaluation.json")
        read(root / f"round_{index}" / "adopted_policy.json")
    protocol = read(root / "protocol.json")
    require(protocol["kind"] == "conditional_v3_bounded_3x40", "Wrong campaign protocol")
    require([item["round"] for item in protocol["rounds"]] == [1, 2, 3], "Expected exactly three rounds")
    expected_budget = {"planned_games_max": 720, "scenes_per_round": 40, "train_scenes": 30,
                       "holdout_scenes": 10, "candidate_branches_per_train_scene": 4,
                       "max_steps_per_scene": 4, "max_effects_per_step": 8,
                       "max_effects_per_family_per_step": 2}
    require(all(protocol["budget"].get(key) == value for key, value in expected_budget.items()), "Campaign budget changed")
    limit = protocol["max_native_attempts"]
    require(type(limit) is int and 0 < limit <= 780, "Native episode-slot cap exceeds 780")
    revision_path = root / "source_revision.json"
    revision = read(revision_path) if revision_path.exists() else None
    if revision is not None:
        require(revision["original_source_hashes"] == protocol["source_hashes"], "Source revision lost original freeze")
    sources = revision["source_hashes"] if revision else protocol["source_hashes"]
    require(bool(sources), "Empty frozen source list")
    for name, expected in sources.items():
        path = inside(project_root, project_root / name)
        require(checksum(path) == expected, "Frozen source changed: " + name)

    database = root / "native_attempts.sqlite3"
    require(database.is_file(), "Native attempt ledger missing")
    with closing(sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA query_only=ON")
        conn.execute("BEGIN")
        meta = dict(conn.execute("SELECT key, value FROM meta"))
        attempts = [dict(row) for row in conn.execute("SELECT * FROM attempts ORDER BY id")]
        job_records = {str(inside(root, row["output"])): json.loads(row["identity"])
                       for row in conn.execute("SELECT output, identity FROM jobs")}
    require(meta.get("protocol") == digest(protocol), "Native ledger protocol differs from freeze")
    require(bool(attempts), "Empty native attempt ledger")
    require(not any(row["state"] in ("running", "reserved") for row in attempts), "Native ledger still has running/reserved attempts")
    require(all(type(row["slots"]) is int and row["slots"] > 0 for row in attempts), "Invalid episode-slot reservation")
    total_slots = sum(row["slots"] for row in attempts)
    require(total_slots <= limit, "Native episode-slot cap exceeded")
    by_output, by_path = {}, {}
    for row in attempts:
        output, path = inside(root, row["output"]), inside(root, row["path"])
        require(str(path) not in by_path, "Duplicate native attempt path")
        by_path[str(path)] = row
        by_output.setdefault(str(output), []).append(row)
    require(set(by_output) == set(job_records), "Jobs and attempts differ in the native ledger")
    for output, history in by_output.items():
        require(len(history) <= 2 and [row["attempt"] for row in history] == list(range(1, len(history) + 1)), "Invalid per-job retry history")
        identity = job_records[output]
        require(all(row["slots"] == identity["episode_slots"] for row in history), "Ledger slots differ from job identity")
        for name, expected in identity["inputs"].items():
            require(checksum(inside(root, name)) == expected, "Registered native input changed: " + name)

    compatibility = read(root / "audit_compatibility.json")
    require(compatibility["kind"] == "json_tuple_list_audit_compatibility"
            and compatibility["runtime_sources_modified"] is False
            and compatibility["native_decision_rules_modified"] is False, "Invalid audit compatibility registration")
    auditor_path = inside(project_root, compatibility["script"])
    require(checksum(auditor_path) == compatibility["sha256"], "Registered compatibility auditor changed")

    used_paths, used_outputs, recoveries, run_cache = set(), set(), [], {}

    def inspect_run(path, expected_manifest, policy=None, forced=None):
        path = inside(root, path)
        key = str(path)
        require(key not in used_paths, "One native run reused for two campaign jobs")
        require(key in by_path, "Run has no native attempt: " + key)
        attempt = by_path[key]
        output = str(inside(root, attempt["output"]))
        require(output not in used_outputs, "Two completed runs for one native job")
        headers = {name: read(path / name) for name in SMALL_EVIDENCE}
        manifest, summary, native, episodes = (headers[name] for name in
                                                ("manifest.json", "summary.json", "run.json", "episodes.json"))
        require(manifest == expected_manifest, "Native manifest differs from registered cohort: " + key)
        groups = manifest_groups(manifest)
        require(summary.get("status") == "complete" and not summary.get("error"), "Native run incomplete: " + key)
        require(summary["completed_episodes"] == summary["expected_episodes"] == len(groups)
                == attempt["slots"] == len(episodes), "Native completed episode count mismatch")
        require(native["manifest_sha256"] == summary["manifest_sha256"] == manifest["sha256"], "Native manifest stamp mismatch")
        require([item["scenario_sha256"] for item in episodes] == groups
                and [item["episode"] for item in episodes] == list(range(1, len(groups) + 1)), "Native episode IDs or order mismatch")
        for item in episodes:
            pair_labels({"candidate": {"return": item["reward"], "outcome": item["outcome"]}, "reference": None})
            require(bool(item["initial_state_sha256"]), "Missing native initial-state stamp")
        trace = path / "events.jsonl"
        require(trace.is_file() and trace.stat().st_size > 0, "Native trace missing")
        if policy is not None:
            metadata = native["conditional_skills"]
            require(summary["conditional_skills"] == metadata, "Conditional metadata differs between headers")
            require(metadata["policy"] == policy and metadata["policy_sha256"] == digest(policy), "Native policy differs from frozen policy")
            require(metadata["source_hashes"] == sources and metadata.get("forced") == forced, "Conditional source or override mismatch")
        hashes = {name: checksum(path / name) for name in SMALL_EVIDENCE}
        if attempt["state"] == "failed" and attempt["returncode"] == 1:
            recovery = read(path / "audit_recovery.json")
            require(recovery["kind"] == "json_representation_audit_recovery"
                    and recovery["native_attempt_id"] == attempt["id"]
                    and recovery["original_native_returncode"] == 1
                    and recovery["native_ledger_modified"] is False, "Invalid audit-only recovery record")
            require(policy is not None and recovery["source_hashes"] == sources
                    and recovery["auditor_sha256"] == compatibility["sha256"]
                    and recovery["completed_episodes"] == len(groups), "Audit recovery provenance mismatch")
            require(all(recovery["evidence_sha256"].get(name) == value for name, value in hashes.items()), "Recovered small evidence changed")
            require(len(recovery["evidence_sha256"].get("events.jsonl", "")) == 64, "Missing recovered trace digest")
            console = root / "launch_state" / f"attempt_{attempt['id']:05d}" / "console.log"
            require(AUDIT_ERROR in console.read_text(encoding="utf-8", errors="replace"), "Exit 1 was not the recognized audit error")
            require(len(by_output[output]) == 1, "Audit-only recovery unexpectedly reran the native job")
            recoveries.append({"native_attempt_id": attempt["id"], "path": key,
                               "episodes_reused": len(groups), "original_state": "failed",
                               "original_returncode": 1, "classification": "audit_representation_only",
                               "native_ledger_preserved": True, "additional_native_games": 0})
        else:
            require(attempt["state"] == "exited" and attempt["returncode"] == 0, "Required native run did not exit successfully")
        used_paths.add(key)
        used_outputs.add(output)
        run_cache[key] = (headers, hashes)
        return {item["scenario_sha256"]: item for item in episodes}

    all_groups, all_seeds, round_reports = set(), set(), []
    previous_policy = read(root / "initial_policy.json")
    for index in range(1, 4):
        folder = root / f"round_{index}"
        manifest, split = read(folder / "manifest.json"), read(folder / "split.json")
        groups = manifest_groups(manifest)
        require(len(groups) == 40 and not set(groups) & all_groups, "Rounds do not contain 120 independent scenes")
        seeds = [scene["sc2_seed"] for scene in manifest["episodes"]]
        require(len(set(seeds)) == 40 and not set(seeds) & all_seeds, "SC2 seed reused across scenes")
        all_groups.update(groups)
        all_seeds.update(seeds)
        env = manifest["env_args"]
        require(env["map_name"] == "10gen_protoss" and env["capability_config"]["n_units"]
                == env["capability_config"]["n_enemies"] == 10, "Wrong Protoss 10v10 scenario")
        registered = protocol["rounds"][index - 1]
        require(registered["manifest_sha256"] == manifest["sha256"] and registered["split"] == split, "Round differs from protocol freeze")
        train, holdout = split["training_group_ids"], split["validation_group_ids"]
        require(len(train) == len(set(train)) == 30 and len(holdout) == len(set(holdout)) == 10
                and not set(train) & set(holdout) and set(train + holdout) == set(groups), "Expected disjoint 30/10 split")
        train_manifest, holdout_manifest = read(folder / "train_manifest.json"), read(folder / "holdout_manifest.json")
        require(train_manifest == subset_manifest(manifest, train)
                and holdout_manifest == subset_manifest(manifest, holdout), "Partition manifest differs from split")
        incumbent, candidate = read(folder / "incumbent_policy.json"), read(folder / "candidate_policy.json")
        require(incumbent == previous_policy, "Incumbent does not follow previous adoption")
        plan, measured, freeze = read(folder / "branch_plan.json"), read(folder / "measured.json"), read(folder / "freeze.json")
        require(freeze == {"candidate_sha256": digest(candidate), "training_sha256": digest(measured),
                           "split_sha256": digest(split)}, "Candidate/data/split freeze digest mismatch")
        require(plan["round_index"] == index and plan["training_groups"] == sorted(train)
                and plan["holdout_groups"] == sorted(holdout), "Branch plan cohort mismatch")
        jobs = plan["jobs"]
        check_budget(jobs)
        require(len(jobs) == plan["planned_branch_games"] == len(measured) <= 120, "Measured branch count differs from plan or cap")
        paths, evaluation = read(folder / "run_paths.json"), read(folder / "evaluation.json")
        require(inside(root, plan["incumbent"]) == inside(root, paths["incumbent"]), "Plan incumbent path mismatch")
        adaptive_rows = inspect_run(paths["adaptive"], manifest)
        incumbent_rows = inspect_run(paths["incumbent"], manifest, incumbent)
        incumbent_headers, incumbent_hashes = run_cache[str(inside(root, paths["incumbent"]))]
        require(plan["source_hashes"] == sources and plan["incumbent_policy_sha256"] == digest(incumbent)
                and plan["incumbent_manifest_sha256"] == incumbent_hashes["manifest.json"], "Plan incumbent provenance mismatch")
        seen_ids = set()
        for number, (job, row) in enumerate(zip(jobs, measured), 1):
            require(row["id"] == f"r{index}_{number}" and row["id"] not in seen_ids, "Measured ID mismatch or duplicate")
            seen_ids.add(row["id"])
            require(all(row.get(key) == value for key, value in job.items()), "Measured row differs from planned job")
            require(job["scene_hash"] in train and row["group_id"] == job["scene_hash"], "Measured data contains held-out or wrong scene")
            require(type(job["episode"]) is int and 1 <= job["episode"] <= 40, "Invalid planned episode")
            scene = manifest["episodes"][job["episode"] - 1]
            require(scene["sha256"] == job["scene_hash"] and scene["sc2_seed"] == job["seed"], "Branch episode/scene/seed mismatch")
            require(row["scope"] == "single_override_then_same_conditional_policy" and row["different_action"] is True,
                    "Branch lacks a distinct measured override")
            require(pair_labels(row) == {"win_gain": row["win_gain"], "return_gain": row["return_gain"]}
                    and row["candidate"] == row["branch"] and row["reference"] == row["incumbent"], "Branch labels disagree")
            slot = folder / "branches" / f"{number:03d}"
            require(read(slot.with_suffix(".pair.json")) == row, "Measured row differs from saved pair")
            history = by_output.get(str(slot.resolve()), [])
            successful = [item for item in history if item["state"] == "exited" and item["returncode"] == 0]
            require(len(successful) == 1, "Branch must have one successful native attempt")
            branch_path = inside(root, successful[0]["path"])
            branch_manifest = subset_manifest(manifest, [job["scene_hash"]])
            branch_rows = inspect_run(branch_path, branch_manifest, incumbent, {"step": job["step"], "binding": job["binding"]})
            branch_headers, branch_hashes = run_cache[str(branch_path)]
            for name, hashes in (("incumbent_sources", incumbent_hashes), ("branch_sources", branch_hashes)):
                require(all(row[name].get(key) == value for key, value in hashes.items()), "Pair small-evidence provenance mismatch")
                require(len(row[name].get("events.jsonl", "")) == 64, "Missing audited pair trace digest")
            require(row["incumbent_sources"]["events.jsonl"] == plan["incumbent_events_sha256"], "Plan and pair trace stamps differ")
            branch, reference = branch_rows[job["scene_hash"]], incumbent_rows[job["scene_hash"]]
            require(branch["initial_state_sha256"] == reference["initial_state_sha256"], "Branch initial state mismatch")
            require(row["candidate"] == {"return": branch["reward"], "outcome": branch["outcome"]}
                    and row["reference"] == {"return": reference["reward"], "outcome": reference["outcome"]}, "Pair terminal labels differ from episodes")
        check_budget(measured)
        train_rows = inspect_run(evaluation["candidate_train"], train_manifest, candidate)
        holdout_rows = inspect_run(evaluation["candidate_holdout"], holdout_manifest, candidate)
        computed = {"training": whole_policy(train_rows, incumbent_rows, train),
                    "holdout_vs_incumbent": whole_policy(holdout_rows, incumbent_rows, holdout),
                    "holdout_vs_adaptive": whole_policy(holdout_rows, adaptive_rows, holdout)}
        require(all(evaluation[key] == value for key, value in computed.items()), "Whole-policy evaluation differs from terminal records")
        require(read(folder / "training_whole_policy.json") == computed["training"], "Training whole-policy report mismatch")
        baseline = computed["holdout_vs_adaptive"]
        accepted = (adoption_verdict(computed["training"])["adopt"]
                    and adoption_verdict(computed["holdout_vs_incumbent"])["adopt"]
                    and baseline["mean_win_gain"] >= -1e-9
                    and (baseline["mean_win_gain"] > 1e-9 or baseline["mean_return_gain"] >= -1e-9))
        require(type(evaluation["adopted"]) is bool and evaluation["adopted"] == accepted, "Adoption verdict violates win-first rule")
        previous_policy = read(folder / "adopted_policy.json")
        require(previous_policy == (candidate if accepted else incumbent), "Adopted policy differs from verdict")
        round_reports.append({"round": index, "scenes": 40, "train_scenes": 30, "holdout_scenes": 10,
                              "measured_branches": len(measured), "max_branches_in_one_scene": max(
                                  Counter(row["scene_hash"] for row in measured).values(), default=0),
                              "freeze": freeze, "adopted": accepted, "checks_passed": True})
    require(len(all_groups) == len(all_seeds) == 120, "Expected 120 distinct scenarios and seeds")
    require(used_outputs == set(by_output), "Unaccounted native jobs in campaign ledger")
    for row in attempts:
        if str(inside(root, row["path"])) not in used_paths:
            require(row["state"] == "retryable", "Unaccounted failed or duplicate completed native attempt")
    logical_slots = sum(by_path[path]["slots"] for path in used_paths)
    require(logical_slots == 360 + sum(item["measured_branches"] for item in round_reports) <= 720,
            "Completed game count differs from bounded campaign design")
    retry_slots = total_slots - logical_slots
    require(0 <= retry_slots <= 60, "Retry episode-slot allowance exceeded")
    return {"kind": "conditional_v3_completion_checks", "status": "complete", "checks_passed": True,
            "campaign_root": str(root), "rounds": round_reports, "independent_scenes": 120,
            "protocol_sha256": digest(protocol), "source_revision_sha256": digest(revision) if revision else None,
            "frozen_source_files_checked": len(sources), "current_frozen_sources_match": True,
            "native_ledger": {"database": str(database), "read_only": True, "attempt_rows": len(attempts),
                              "reserved_episode_slots": total_slots, "episode_slot_cap": limit,
                              "completed_campaign_games": logical_slots, "retry_episode_slots": retry_slots,
                              "running_or_reserved_rows": 0, "original_rows_preserved": True},
            "audit_only_recoveries": recoveries,
            "recovery_interpretation": "Original exit 1 remains in the ledger. A registered JSON-representation audit recovery reuses those completed games; it is not a failed 40-game batch rerun.",
            "trace_audit": {"replayed": False, "trace_bytes_rehashed": False,
                            "basis": "Existing native and counterfactual audits; current small evidence and saved trace-digest cross-links checked."},
            "sc2_process_exit_checked": False,
            "limitations": ["SC2 process exit must be checked separately.",
                            "Current large trace contents are not re-audited or rehashed by this completion check."]}


def write_completion(root, *, project_root=PROJECT_ROOT):
    report = check_campaign(root, project_root=project_root)
    output = Path(root).resolve() / "completion_checks.json"
    if output.exists():
        require(read(output) == report, "Existing completion report differs; it was not overwritten")
    else:
        with output.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
    return output, report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        output, report = write_completion(args.root)
    except (OSError, ValueError, KeyError, TypeError, sqlite3.Error) as error:
        print(json.dumps({"status": "incomplete_or_invalid", "checks_passed": False,
                          "error": str(error), "success_report_written": False}, ensure_ascii=False), file=sys.stderr)
        return 1
    print(json.dumps({"status": report["status"], "output": str(output),
                      "completed_campaign_games": report["native_ledger"]["completed_campaign_games"],
                      "native_episode_slots": report["native_ledger"]["reserved_episode_slots"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
