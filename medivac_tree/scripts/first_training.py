"""Serial first-training collection and held-out evaluation; no automatic fitting."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from medivac_tree.learning.dataset import build_dataset
from medivac_tree.scripts.return_study import command, read


def plan(study, output, *, mode="collect", policy=None, repeat_base=False,
         sc2_path=Path("C:/Program Files (x86)/StarCraft II")):
    study, output = Path(study).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    if mode not in ("collect", "evaluate"):
        raise ValueError("Unknown workflow mode")
    if (mode == "evaluate") != (policy is not None):
        raise ValueError("Only evaluate requires a trained --policy")
    if mode == "evaluate" and repeat_base:
        raise ValueError("--repeat-base is a collection noise audit")
    manifest = read(study / "manifest.json")
    if mode == "collect" and manifest.get("split") in ("validation", "test"):
        raise ValueError("Sealed validation/test scenarios cannot be used to collect training labels")
    if mode == "evaluate":
        split = read(policy).get("selection", {}).get("model", {}).get("training", {}).get("split", {})
        training_groups, validation_groups = split.get("training_groups"), split.get("validation_groups")
        if not isinstance(training_groups, list) or not training_groups or not isinstance(validation_groups, list):
            raise ValueError("Held-out evaluation requires the model's recorded training/validation groups")
        used = set(training_groups) | set(validation_groups)
        if used & {row["sha256"] for row in manifest["episodes"]}:
            raise ValueError("Evaluation scenarios overlap the model's training/validation data")
    other = "candidate" if mode == "collect" else "selector"
    arms = ["base", other] + (["base-repeat"] if repeat_base else [])
    calls = []
    for arm in arms:
        selected_policy = policy if arm == "selector" else None
        # Validate every arm before launching the first game.
        command(study, arm, output / arm, sc2_path, selected_policy)
        call = [sys.executable, str(ROOT / "medivac_tree/scripts/return_study.py"), "run",
                "--study-dir", str(study), "--arm", arm, "--output-dir", str(output / arm),
                "--sc2-path", str(Path(sc2_path).resolve())]
        if selected_policy is not None:
            call += ["--policy", str(Path(selected_policy).resolve())]
        calls.append(call)
    return calls


def save(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def check_workspace_inputs(study, policy):
    paths = [Path(study).resolve() / "base.json", Path(study).resolve() / "candidate.json"]
    if policy is not None:
        paths.append(Path(policy).resolve())
    if any(not path.is_relative_to(ROOT.parent) for path in paths):
        raise ValueError("Keep study/model JSON inside the workspace before starting either arm")


def execute(study, output, *, mode="collect", policy=None, repeat_base=False,
            sc2_path=Path("C:/Program Files (x86)/StarCraft II"), dry_run=False):
    calls = plan(study, output, mode=mode, policy=policy, repeat_base=repeat_base, sc2_path=sc2_path)
    episodes = read(Path(study) / "protocol.json")["episodes_per_arm"]
    if dry_run:
        return {"commands": calls, "planned_games": episodes * len(calls), "started_games": 0}
    check_workspace_inputs(study, policy)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    save(output / "workflow.json", {"mode": mode, "study": str(Path(study).resolve()),
                                   "commands": calls, "planned_games": episodes * len(calls)})
    for call in calls:
        subprocess.run(call, cwd=ROOT, check=True)
    other = "candidate" if mode == "collect" else "selector"
    report_call = [sys.executable, "-m", "medivac_tree.learning.report",
                   "--base-run", str(output / "base"), "--other-run", str(output / other),
                   "--output", str(output / "comparison.json")]
    result = {"output": str(output), "mode": mode, "completed_games": episodes * len(calls)}
    if mode == "collect":
        data = build_dataset(output / "base", output / "candidate")
        save(output / "paired.json", data)
        groups = len({row["group_id"] for row in data["rows"]})
        result.update(accepted_rows=len(data["rows"]), accepted_scenarios=groups,
                      rejected_pairs=len(data["rejected"]), trained_model=False)
        report_call += ["--dataset", str(output / "paired.json")]
        if repeat_base:
            noise = build_dataset(output / "base", output / "base-repeat", baseline_noise=True)
            save(output / "noise.json", noise)
            result["noise_pairs"] = len(noise["noise_audit"])
        result["next_step"] = ("Inspect comparison.json and fit on accepted development data"
                               if groups >= 6 else
                               "Collect more independent development scenarios under the same frozen protocol; do not lower leaf coverage")
    # This comparison includes no-opportunity games. It is not a training-label extractor.
    subprocess.run(report_call, cwd=ROOT, check=True)
    save(output / "workflow_result.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("collect", "evaluate"))
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--repeat-base", action="store_true")
    parser.add_argument("--sc2-path", type=Path, default=Path("C:/Program Files (x86)/StarCraft II"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    result = execute(args.study_dir, args.output_dir, mode=args.mode, policy=args.policy,
                     repeat_base=args.repeat_base, sc2_path=args.sc2_path, dry_run=args.dry_run)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
