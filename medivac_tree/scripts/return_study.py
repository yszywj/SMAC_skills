"""Prepare and run one-intervention return studies; preparing never starts SC2."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from medivac_tree import MedivacTree
from medivac_tree.experiment import validate_context
from medivac_tree.selector import validate_policy
from medivac_tree.validate import INTENTS, validate_tree
from sc2_skill_adapter.bridge.config import SMAC_ROOT, map_file
from sc2_skill_adapter.bridge.scenarios import validate_manifest


# These settings are identical in every arm. Later actions can differ because
# the intervention changes the observed game, not because this controller changes.
RUN_FLAGS = [
    "--method", "adaptive-skill", "--skill-version", "current",
    "--race-profile", "terran", "--terran-keep-heal-order",
    "--unit-value-mode", "contextual", "--adaptive-window-fire",
    "--adaptive-range-preserving-move", "--adaptive-target-scoring",
    "--adaptive-coordinated-approach", "--adaptive-safe-reentry",
    "--adaptive-accurate-damage", "--adaptive-critical-escape",
    "--adaptive-planned-reentry", "--adaptive-engagement-commitment",
    "--adaptive-approach-progress-guard", "--pair-mode", "off",
    "--native-teacher-control", "off", "--medivac-policy", "tree",
    "--medical-patient-coordination", "off",
]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, document):
    Path(path).write_text(json.dumps(document, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frozen_sources(manifest):
    sources = list((ROOT / "sc2_skill_adapter/bridge").glob("*.py"))
    sources += list((ROOT / "skills/current").glob("*.py"))
    sources += list((ROOT / "skills/current/skills").glob("*.py"))
    sources += list((ROOT / "medivac_tree").glob("*.py"))
    sources += [Path(__file__), ROOT / "sc2_skill_adapter/run.py"]
    sources += [SMAC_ROOT / "smacv2/env/starcraft2" / name
                for name in ("starcraft2.py", "wrapper.py", "distributions.py")]
    sources += [map_file(manifest["env_args"])]
    return {os.path.relpath(path, ROOT): file_hash(path)
            for path in sorted(set(sources))}


def prepare(output, manifest_path, base_tree, candidate_intent=None, base_intents=None, min_game_loop=0,
            *, profile="bound", event=None):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite study: {output}")
    manifest, tree = read(manifest_path), read(base_tree)
    validate_manifest(manifest)
    validate_tree(tree)
    if manifest["env_args"]["map_name"] != "10gen_terran":
        raise ValueError("Return study requires a Terran manifest")
    if not 1 <= len(manifest["episodes"]) <= 40:
        raise ValueError("Each arm must contain 1..40 frozen episodes")
    if profile not in ("bound", "idle-recovery"):
        raise ValueError("Unknown candidate profile")
    if profile == "idle-recovery":
        if candidate_intent not in (None, "APPROACH") or base_intents not in (None, ["STOP"]):
            raise ValueError("idle-recovery compares BASE STOP against a generated APPROACH")
        candidate_intent, base_intents = "APPROACH", ["STOP"]
    elif event is not None:
        raise ValueError("Event parameters are only supported by idle-recovery")
    if candidate_intent not in INTENTS or not base_intents or set(base_intents) - INTENTS:
        raise ValueError("Declare one candidate intent and at least one base intent")
    if type(min_game_loop) is not int or min_game_loop < 0:
        raise ValueError("min_game_loop must be a nonnegative integer")
    policy = {
        "schema_version": 1, "policy_kind": "paired_return_selector",
        "version": "medivac-return-study-v1", "base_tree": tree,
        "candidate": {"intent": candidate_intent, "base_intents": sorted(set(base_intents)),
                      "min_game_loop": min_game_loop},
        "selection": {"mode": "base"},
    }
    if profile == "idle-recovery":
        policy["candidate"].update(generator="idle_recovery", event=event or {})
    # Validate before creating the persistent output directory.
    metadata = validate_policy(policy)
    policy["candidate"] = metadata["candidate"]
    source_hashes = frozen_sources(manifest)
    protocol = {
        "schema_version": 1, "purpose": "one_intervention_return_learning",
        "manifest_sha256": manifest["sha256"], "episodes_per_arm": len(manifest["episodes"]),
        "max_instances": 1, "max_episodes_per_run": 40,
        "comparison_contract_hash": metadata["comparison_contract_hash"],
        "candidate": policy["candidate"], "candidate_profile": profile,
        "return_metric": "undiscounted_episode_reward",
        "opportunity": "first legal different-action candidate across the episode",
        "continuation": "same frozen base controller after the one decision",
        "run_flags": RUN_FLAGS, "source_hashes": source_hashes,
        "arms": {"base": "base.json", "candidate": "candidate.json", "base-repeat": "base.json"},
        "evidence_kind": "matched_observed_prefix",
        "limitations": ["No recoverable SC2 snapshot or hidden-state equality guarantee",
                        "No-op, unmatched, missing, or failed command evidence is not a zero-return label",
                        "This study compares one configured proposal, not all patient targets or a global skill tree"],
    }
    candidate_policy = copy.deepcopy(policy)
    candidate_policy["selection"]["mode"] = "candidate"
    output.mkdir(parents=True, exist_ok=False)
    write(output / "manifest.json", manifest)
    write(output / "base.json", policy)
    write(output / "candidate.json", candidate_policy)
    protocol["policy_file_hashes"] = {name: file_hash(output / name)
                                      for name in ("base.json", "candidate.json")}
    write(output / "protocol.json", protocol)
    return {"study": str(output), "episodes_per_arm": len(manifest["episodes"]),
            "comparison_contract_hash": metadata["comparison_contract_hash"],
            "started_games": 0}


def command(study, arm, output, sc2_path, policy_path=None):
    """Validate frozen inputs and return a shell-free command; no game is launched."""
    study, output = Path(study).resolve(), Path(output).resolve()
    protocol, manifest = read(study / "protocol.json"), read(study / "manifest.json")
    validate_manifest(manifest)
    if (protocol.get("schema_version") != 1 or protocol.get("max_instances") != 1
            or protocol.get("max_episodes_per_run") != 40
            or not 1 <= len(manifest["episodes"]) <= 40
            or protocol.get("episodes_per_arm") != len(manifest["episodes"])
            or protocol.get("manifest_sha256") != manifest["sha256"]
            or protocol.get("run_flags") != RUN_FLAGS):
        raise ValueError("Study protocol or manifest changed")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite run: {output}")
    if frozen_sources(manifest) != protocol.get("source_hashes"):
        raise ValueError("Frozen runtime sources changed; prepare a new study")
    for name in ("base.json", "candidate.json"):
        if file_hash(study / name) != protocol.get("policy_file_hashes", {}).get(name):
            raise ValueError(f"Frozen policy changed: {name}")
    if arm == "selector":
        if policy_path is None:
            raise ValueError("selector arm requires --policy pointing to a trained policy")
        policy = Path(policy_path).resolve()
    elif arm in ("base", "candidate", "base-repeat"):
        if policy_path is not None:
            raise ValueError("--policy is only valid with --arm selector")
        policy = study / ("candidate.json" if arm == "candidate" else "base.json")
    else:
        raise ValueError("Unknown return-study arm")
    metadata = MedivacTree(policy).metadata
    if metadata.get("comparison_contract_hash") != protocol["comparison_contract_hash"]:
        raise ValueError("Policy does not belong to the frozen comparison contract")
    expected_mode = {"base": "base", "base-repeat": "base", "candidate": "candidate", "selector": "model"}[arm]
    if metadata.get("selection_mode") != expected_mode:
        raise ValueError("Policy selection mode does not match arm")
    if arm == "selector":
        training = read(policy)["selection"]["model"].get("training", {})
        context = validate_context(training.get("experiment_context"), training.get("experiment_context_hash"))
        if context["env_args"] != manifest["env_args"]:
            raise ValueError("Trained model environment differs from this study")
        current = {key.replace("\\", "/"): value for key, value in protocol["source_hashes"].items()}
        if any(current.get(key) != value for key, value in context["source_hashes"].items()):
            raise ValueError("Trained model execution sources differ from this study")
        # The runner also compares dependency versions and all controller flags
        # against this full context, before it starts the first SC2 process.
    return [sys.executable, str(ROOT / "sc2_skill_adapter/run.py"), "run", *RUN_FLAGS,
            "--manifest", str(study / "manifest.json"), "--medivac-tree", str(policy),
            "--sc2-path", str(Path(sc2_path).resolve()), "--output-dir", str(output)]


def ensure_idle_sc2():
    if os.name == "nt":
        result = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command",
                                 "@(Get-Process -Name SC2_x64 -ErrorAction SilentlyContinue).Count"],
                                capture_output=True, text=True, check=True,
                                creationflags=subprocess.CREATE_NO_WINDOW)
        if int(result.stdout.strip()) > 0:
            raise RuntimeError("SC2 is already running; return studies use a single instance")
    else:
        # This repository's launcher is Windows-only; do not guess process names.
        raise RuntimeError("This study launcher currently supports Windows only")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    prepare_parser = sub.add_parser("prepare", help="Freeze policies and protocol without starting SC2")
    prepare_parser.add_argument("--manifest", type=Path, required=True)
    prepare_parser.add_argument("--base-tree", type=Path, default=ROOT / "medivac_tree/tree_v0.json")
    prepare_parser.add_argument("--profile", choices=("bound", "idle-recovery"), default="bound")
    prepare_parser.add_argument("--candidate-intent", choices=sorted(INTENTS))
    prepare_parser.add_argument("--base-intent", choices=sorted(INTENTS), action="append")
    prepare_parser.add_argument("--min-stop-windows", type=int)
    prepare_parser.add_argument("--min-energy", type=float)
    prepare_parser.add_argument("--min-health-ratio", type=float)
    prepare_parser.add_argument("--max-target-distance", type=float)
    prepare_parser.add_argument("--min-progress", type=float)
    prepare_parser.add_argument("--max-extra-exposure", type=int)
    prepare_parser.add_argument("--min-game-loop", type=int, default=0)
    prepare_parser.add_argument("--output-dir", type=Path, required=True)
    run_parser = sub.add_parser("run", help="Run one frozen arm, or show its command with --dry-run")
    run_parser.add_argument("--study-dir", type=Path, required=True)
    run_parser.add_argument("--arm", choices=("base", "base-repeat", "candidate", "selector"), required=True)
    run_parser.add_argument("--policy", type=Path)
    run_parser.add_argument("--output-dir", type=Path, required=True)
    run_parser.add_argument("--sc2-path", type=Path, default=Path("C:/Program Files (x86)/StarCraft II"))
    run_parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.action == "prepare":
        event = {name: getattr(args, name) for name in
                 ("min_stop_windows", "min_energy", "min_health_ratio", "max_target_distance",
                  "min_progress", "max_extra_exposure") if getattr(args, name) is not None}
        result = prepare(args.output_dir, args.manifest, args.base_tree,
                         args.candidate_intent, args.base_intent, args.min_game_loop,
                         profile=args.profile, event=event or None)
    else:
        argv = command(args.study_dir, args.arm, args.output_dir, args.sc2_path, args.policy)
        if args.dry_run:
            result = {"command": argv, "started_games": 0}
        else:
            policy = Path(argv[argv.index("--medivac-tree") + 1])
            if not policy.is_relative_to(ROOT.parent):
                raise ValueError("Keep study/model JSON inside the workspace so source snapshots can preserve it")
            ensure_idle_sc2()
            # Atomic lock also excludes concurrently launched arms before SC2 starts.
            lock = ROOT / "sc2_skill_adapter/.medivac-return-study.lock"
            with lock.open("x", encoding="utf-8") as handle:
                handle.write(json.dumps({"pid": os.getpid(), "study": str(args.study_dir.resolve()),
                                         "arm": args.arm}))
            try:
                subprocess.run(argv, cwd=ROOT / "sc2_skill_adapter", check=True)
                from medivac_tree.scripts.freeze_sources import freeze
                freeze(args.output_dir, args.output_dir / "source_snapshot")
            finally:
                lock.unlink()
            result = {"run": str(args.output_dir.resolve()), "arm": args.arm, "status": "complete"}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
