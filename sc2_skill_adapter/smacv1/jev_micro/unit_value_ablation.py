"""Prepare or execute matched, model-free off/basic/contextual micro experiments.

The default only writes a reviewable manifest. Use --execute after offline tests
pass. Every attempted run, including failures, remains in the report.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from .catalog import map_param_registry, map_source
from .run import (REPO, add_unit_value_arguments, installation_info, positive_int,
                  unit_value_cli_arguments, unit_value_metadata, unit_value_parameters, write_json)
from .suite import result_row


CASES = (("8m", "focus_fire_budget"), ("8m", "adaptive_focus_fire"),
         ("2s3z", "focus_fire_budget"), ("2s3z", "adaptive_focus_fire"),
         ("MMM", "heal_assign"))
MODES = ("off", "basic", "contextual")


def build_manifest(args):
    """Pure preparation apart from reading public map files and client metadata."""
    common = {"policy": "nearest", "planner": "none", "skill_mode": "always",
              "opponent": "base", "episodes": args.episodes, "step_mul": args.step_mul,
              "posture": args.adaptive_posture,
              "window_fire": args.adaptive_window_fire,
              "target_scoring": args.adaptive_target_scoring,
              "range_preserving_move": args.adaptive_range_preserving_move}
    runs = []
    for seed in args.seeds:
        for map_name, skill in CASES:
            source = map_source(map_name)  # Registration and map presence are checked.
            episode_limit = args.episode_limit or map_param_registry[map_name]["limit"]
            for mode in MODES:
                case_name = f"{map_name}-{skill}-seed-{seed}-{mode}"
                case_output = args.output_dir.resolve() / case_name
                value_args = SimpleNamespace(**{**vars(args), "unit_value_mode": mode})
                parameters = unit_value_parameters(value_args)
                command = [sys.executable, "-X", "utf8", "-B", "-m", "jev_micro.run",
                           "--map", map_name, "--policy", "nearest", "--planner", "none",
                           "--opponent", "base", "--skill-mode", "always", "--skill-name", skill,
                           "--episodes", str(args.episodes), "--seed", str(seed),
                           "--step-mul", str(args.step_mul), "--episode-limit", str(episode_limit),
                           "--adaptive-posture", args.adaptive_posture,
                           "--sc2-path", str(args.sc2_path.resolve()), "--output-dir", str(case_output),
                           *unit_value_cli_arguments(value_args)]
                for flag in ("adaptive_window_fire", "adaptive_target_scoring", "adaptive_range_preserving_move"):
                    if getattr(args, flag):
                        command.append("--" + flag.replace("_", "-"))
                runs.append({"case": case_name, "map": map_name, "skill": skill, "seed": seed,
                             "episode_limit": episode_limit, "output_dir": str(case_output),
                             "map_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                             **unit_value_metadata(parameters), "command": command})
    return {"status": "prepared_not_run", "model_calls": 0, "common": common,
            "client": installation_info(args.sc2_path), "sc2_path": str(args.sc2_path.resolve()),
            "planned_runs": len(runs), "planned_episodes": len(runs) * args.episodes,
            "semantics": "Small-sample regression/exploration; estimates are not actual damage or win contribution.",
            "runs": runs}


def collect_result(case, returncode):
    output = Path(case["output_dir"])
    row = result_row(case["map"], "nearest", output, returncode)
    summary_file = output / "summary.json"
    summary = json.loads(summary_file.read_text(encoding="utf-8")) if summary_file.exists() else {}
    episodes = summary.get("episodes", [])
    return {**row, "case": case["case"], "skill": case["skill"], "seed": case["seed"],
            "source_sha256": summary.get("source_sha256", {}),
            "requested_unit_value_mode": case["unit_value_mode"],
            "executed_configuration_matches": summary.get("unit_value_parameters") == case["unit_value_parameters"],
            "game_versions": sorted({episode["game_version"] for episode in episodes if "game_version" in episode}),
            "timeouts": sum(episode["status"] == "timeout" for episode in episodes),
            "final_observed_allied_health": [episode.get("final", {}).get("allied_health") for episode in episodes],
            "final_observed_allied_shield": [episode.get("final", {}).get("allied_shield") for episode in episodes],
            "final_allies_alive": [episode.get("final", {}).get("allies_alive") for episode in episodes]}


def write_report(output, manifest, rows):
    versions = sorted({version for row in rows for version in row["game_versions"]})
    source_snapshots = {json.dumps(row.get("source_sha256", {}), sort_keys=True) for row in rows}
    report = {"manifest": manifest, "attempted_runs": len(rows), "results": rows,
              "all_attempts_included": True, "observed_game_versions": versions,
              "same_observed_game_version": len(versions) == 1,
              "same_source_snapshot": bool(rows) and len(source_snapshots) == 1
                                      and all(row.get("source_sha256") for row in rows),
              "all_configurations_match": all(row["executed_configuration_matches"] for row in rows),
              "actual_model_requests": sum(row["requests"] for row in rows)}
    write_json(output / "comparison.json", report)
    lines = ["# Unit value ablation", "",
             "All attempted runs are included. Predicted overkill is a heuristic budget estimate, not observed damage.", "",
             f"Same source snapshot: {report['same_source_snapshot']}; same observed game version: {report['same_observed_game_version']}.", "",
             "| Case | Results | Survivors | Actual HP | Target switches | Retreat unit steps | Reengagements | Heal switches | Estimated overkill | Value influences | Eval calls / ms | Failure |",
             "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |"]
    for row in rows:
        retreat_steps = row["retreat_unit_steps"] + row["adaptive_retreat_unit_steps"]
        lines.append(f"| {row['case']} | {', '.join(row['statuses'])} | {row['final_allies_alive']} | "
                     f"{row['final_observed_allied_health']} | {row['target_switches']} | {retreat_steps} | "
                     f"{row['adaptive_reengagements']} | {row['heal_target_switches']} | {row['estimated_overkill']} | "
                     f"{row['unit_value_influenced_decisions']} | {row['unit_value_evaluations']} / "
                     f"{row['unit_value_evaluation_ms']:.3f} | {row['failure'] or ''} |")
    lines.extend(["", "This is a regression/exploration sample; it does not establish a win-rate benefit.", ""])
    (output / "comparison.md").write_text("\n".join(lines), encoding="utf-8")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Run the prepared local SC2 experiments")
    parser.add_argument("--episodes", type=positive_int, default=1)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1])
    parser.add_argument("--step-mul", type=positive_int, default=8)
    parser.add_argument("--episode-limit", type=positive_int)
    parser.add_argument("--adaptive-posture", choices=("aggressive", "balanced", "defensive"), default="balanced")
    parser.add_argument("--adaptive-window-fire", action="store_true")
    parser.add_argument("--adaptive-target-scoring", action="store_true")
    parser.add_argument("--adaptive-range-preserving-move", action="store_true")
    parser.add_argument("--sc2-path", type=Path,
                        default=Path(os.environ.get("SC2PATH", r"C:\Program Files (x86)\StarCraft II")))
    parser.add_argument("--output-dir", type=Path,
                        default=REPO / "jev_runs" / datetime.now().strftime("unit-value-%Y%m%d-%H%M%S-%f"))
    add_unit_value_arguments(parser, include_mode=False)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if len(args.seeds) != len(set(args.seeds)):
        parser.error("seeds must be unique")
    manifest = build_manifest(args)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(args.output_dir / "manifest.json", manifest)
    print(f"Prepared {manifest['planned_runs']} matched runs: {args.output_dir / 'manifest.json'}", flush=True)
    if not args.execute:
        return
    rows = []
    for case in manifest["runs"]:
        with (args.output_dir / (case["case"] + ".log")).open("w", encoding="utf-8") as log:
            completed = subprocess.run(case["command"], cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                                       check=False, text=True, encoding="utf-8")
        row = collect_result(case, completed.returncode)
        rows.append(row)
        write_report(args.output_dir, manifest, rows)
        print(f"{case['case']}: {row['statuses']}; failure={row['failure']}", flush=True)
    manifest["status"] = "completed_with_failures" if any(row["failure"] for row in rows) else "completed"
    write_json(args.output_dir / "manifest.json", manifest)
    write_report(args.output_dir, manifest, rows)
    if any(row["failure"] for row in rows):
        parser.exit(1, "Some runs failed; all attempts are preserved in comparison.json.\n")


if __name__ == "__main__":
    main()
