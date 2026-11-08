"""Bounded three-round native training with scene-separated whole-policy validation."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import sys

from decision_tree.local.runtime import digest
from decision_tree.local.study import save_once
from decision_tree.phase2.policy import REGROUP_POLICY
from decision_tree.scripts.local_campaign_recover import atomic_json
from decision_tree.scripts.local_campaign_validate import single_manifest
from sc2_skill_adapter.bridge.scenarios import generate_manifest
from skills.local_v3.run import check_completion, source_hashes
from .corrections import fit_layer
from .policy import normalize_policy
from .objective import summarize_pairs, adoption_verdict


ROOT = Path(__file__).resolve().parents[2]
PYTHON = Path(sys.executable)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def subset(manifest, groups):
    result = deepcopy(manifest)
    result["episodes"] = [{**s, "index": n} for n, s in enumerate(
        s for s in manifest["episodes"] if s["sha256"] in set(groups))]
    result["sha256"] = digest({k: v for k, v in result.items() if k != "sha256"})
    return result


def prepare(root):
    env_args = read(ROOT/"decision_tree/configs/protoss_10v10_env.json")
    known = set()
    for path in (ROOT/"decision_tree/artifacts").rglob("manifest.json"):
        if root not in path.parents:
            try:
                known.update(s["sha256"] for s in read(path)["episodes"])
            except (KeyError, ValueError):
                continue
    root.mkdir(parents=True, exist_ok=True)
    save_once(root/"initial_policy.json", normalize_policy())
    save_once(root/"adaptive.json", {"mode": "base"})
    rounds = []
    for index in range(1, 4):
        folder = root/f"round_{index}"
        manifest = generate_manifest(env_args, 40, 5100000 + index * 1000)
        groups = [s["sha256"] for s in manifest["episodes"]]
        if len(set(groups)) != 40 or set(groups) & known:
            raise ValueError("Reused scenario")
        known.update(groups)
        random.Random(6100000 + index).shuffle(groups)
        split = {"training_group_ids": groups[:30], "validation_group_ids": groups[30:]}
        save_once(folder/"manifest.json", manifest)
        save_once(folder/"split.json", split)
        for partition, members in (("train", groups[:30]), ("holdout", groups[30:])):
            save_once(folder/f"{partition}_manifest.json", subset(manifest, members))
        rounds.append({"round": index, "manifest_sha256": manifest["sha256"], "split": split})
    protocol = {"kind": "conditional_v3_bounded_3x40", "rounds": rounds,
                "source_hashes": source_hashes(), "max_native_attempts": 780,
                "budget": {"planned_games_max": 720, "scenes_per_round": 40,
                           "train_scenes": 30, "holdout_scenes": 10,
                           "candidate_branches_per_train_scene": 4,
                           "max_steps_per_scene": 4, "max_effects_per_step": 8,
                           "max_effects_per_family_per_step": 2},
                "procedure": "40 adaptive + 40 incumbent; <=120 outcome-blind counterfactual branches on 30 train scenes; freeze a win-first depth<=2 gain/veto layer; 30 train candidate + 10 holdout candidate whole-policy games.",
                "adoption": "Require complete paired validation, candidate win change nonnegative relative to both incumbent and adaptive, and lexicographic improvement over incumbent. Same wins use return. Training full-policy comparison must also favor candidate.",
                "interpretation": "Forced branch keeps incumbent prefix, changes exactly one step, then continues the incumbent conditional policy. Candidate whole-policy runs may execute learned changes repeatedly.",
                "no_auto_resume_previous_campaign": True}
    save_once(root/"protocol.json", protocol)
    return protocol


def launch(root, directory, manifest, policy=None, forced=None):
    from .jobs import run_job
    if policy is None:
        command = [str(PYTHON), "-B", "-m", "skills.local_v2.run", "--manifest", str(manifest),
                   "--selector", str(root/"adaptive.json"), "--implementation", "v1",
                   "--output-dir", str(directory)]
    else:
        command = [str(PYTHON), "-B", "-m", "skills.local_v3.run", "--manifest", str(manifest),
                   "--policy", str(policy), "--output-dir", str(directory)]
        if forced:
            command += ["--force-step", str(forced["step"]), "--force-binding", forced["binding"]]
    result = run_job(command, directory, root)
    path = Path(result["path"])
    if policy is not None:
        metadata = read(path/"run.json")["conditional_skills"]
        check_completion(path, metadata, len(read(manifest)["episodes"]))
    else:
        from skills.local_v2.explore import read_header, read_episodes
        header = read_header(path)
        for _ in read_episodes(path, header):
            pass
    return path


def progress(root, index, stage, **extra):
    record = {"round": index, "stage": stage, "updated_utc": datetime.now(timezone.utc).isoformat(), **extra}
    atomic_json(root/"progress.json", record)
    print(json.dumps(record), flush=True)


def collect(root, index):
    from .planning import make_plan
    from skills.local_v3.counterfactual import audit_counterfactual, read_audited_run
    folder = root/f"round_{index}"
    previous = root/"initial_policy.json" if index == 1 else root/f"round_{index-1}/adopted_policy.json"
    save_once(folder/"incumbent_policy.json", read(previous))
    progress(root, index, "adaptive_40")
    adaptive = launch(root, folder/"adaptive", folder/"manifest.json")
    progress(root, index, "incumbent_40")
    incumbent = launch(root, folder/"incumbent", folder/"manifest.json", folder/"incumbent_policy.json")
    train = read(folder/"split.json")["training_group_ids"]
    plan = make_plan(incumbent, train, index)
    save_once(folder/"branch_plan.json", plan)
    jobs = plan["jobs"]
    if len(jobs) > 120:
        raise ValueError("Round candidate budget exceeded")
    cached = read_audited_run(incumbent)
    manifest = read(folder/"manifest.json")
    measured = []
    for number, job in enumerate(jobs, 1):
        progress(root, index, "counterfactuals", completed=number-1, planned=len(jobs))
        slot = folder/"branches"/f"{number:03d}"
        scene = next(s for s in manifest["episodes"] if s["sha256"] == job["scene_hash"])
        scene_path = folder/"scenes"/f"{job['scene_hash']}.json"
        save_once(scene_path, single_manifest(manifest, scene))
        branch = launch(root, slot, scene_path, folder/"incumbent_policy.json", job)
        pair = audit_counterfactual(cached, branch, job["episode"], job["step"], job["binding"])
        measured.append({**job, "id": f"r{index}_{number}", **pair,
                         "branch": pair["candidate"], "incumbent": pair["reference"]})
        save_once(slot.with_suffix(".pair.json"), measured[-1])
    save_once(folder/"measured.json", measured)
    save_once(folder/"run_paths.json", {"adaptive": str(adaptive), "incumbent": str(incumbent)})
    progress(root, index, "collected", branches=len(measured))


def comparisons(candidate_path, reference_path, groups):
    candidate = {r["scenario_sha256"]: r for r in read(candidate_path/"episodes.json")}
    reference = {r["scenario_sha256"]: r for r in read(reference_path/"episodes.json")}
    rows = []
    for group in groups:
        a, b = candidate[group], reference[group]
        if a["initial_state_sha256"] != b["initial_state_sha256"]:
            raise ValueError("Paired whole-policy initial state changed")
        rows.append({"group_id": group, "pair_id": group, "scope": "whole_policy",
                     "candidate": {"return": a["reward"], "outcome": a["outcome"]},
                     "reference": {"return": b["reward"], "outcome": b["outcome"]}})
    return summarize_pairs(rows, scope="whole_policy", expected_groups=groups)


def evaluate(root, index):
    folder = root/f"round_{index}"
    data = read(folder/"measured.json")
    split = read(folder/"split.json")
    if any(row["scene_hash"] not in split["training_group_ids"] for row in data):
        raise ValueError("Held-out branch leaked into training")
    candidate = fit_layer(read(folder/"incumbent_policy.json"), data, REGROUP_POLICY["model"]["context_hash"])
    save_once(folder/"candidate_policy.json", candidate)
    save_once(folder/"freeze.json", {"candidate_sha256": digest(candidate),
                                    "training_sha256": digest(data), "split_sha256": digest(split)})
    paths = read(folder/"run_paths.json")
    progress(root, index, "candidate_train_30")
    train = launch(root, folder/"candidate_train", folder/"train_manifest.json", folder/"candidate_policy.json")
    training = comparisons(train, Path(paths["incumbent"]), split["training_group_ids"])
    save_once(folder/"training_whole_policy.json", training)
    progress(root, index, "candidate_holdout_10")
    holdout = launch(root, folder/"candidate_holdout", folder/"holdout_manifest.json", folder/"candidate_policy.json")
    vs_old = comparisons(holdout, Path(paths["incumbent"]), split["validation_group_ids"])
    vs_base = comparisons(holdout, Path(paths["adaptive"]), split["validation_group_ids"])
    accepted = (adoption_verdict(training)["adopt"] and adoption_verdict(vs_old)["adopt"]
                and vs_base["mean_win_gain"] >= -1e-9
                and (vs_base["mean_win_gain"] > 1e-9 or vs_base["mean_return_gain"] >= -1e-9))
    report = {"training": training, "holdout_vs_incumbent": vs_old, "holdout_vs_adaptive": vs_base,
              "adopted": accepted, "candidate_train": str(train), "candidate_holdout": str(holdout)}
    save_once(folder/"evaluation.json", report)
    save_once(folder/"adopted_policy.json", candidate if accepted else read(folder/"incumbent_policy.json"))
    progress(root, index, "round_complete", adopted=accepted, holdout_win_gain=vs_old["mean_win_gain"],
             holdout_return_gain=vs_old["mean_return_gain"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "collect", "evaluate"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--round", type=int, choices=(1, 2, 3))
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_relative_to(ROOT/"decision_tree/artifacts"):
        raise ValueError("Campaign must be inside project artifacts")
    if args.stage == "prepare":
        prepare(root)
    else:
        revision = root/"source_revision.json"
        expected_sources = (read(revision)["source_hashes"] if revision.exists()
                            else read(root/"protocol.json")["source_hashes"])
        if source_hashes() != expected_sources:
            raise ValueError("Campaign source changed after freeze")
        {"collect": collect, "evaluate": evaluate}[args.stage](root, args.round)


if __name__ == "__main__":
    main()
