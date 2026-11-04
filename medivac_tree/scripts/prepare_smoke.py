"""Prepare four composition-selected SMAC episodes without starting StarCraft II."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "sc2_skill_adapter"))
from bridge.config import digest
from bridge.scenarios import generate_manifest, validate_manifest


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def prepare(output: Path, config_path: Path, seed: int = 21000, pool_size: int = 256):
    if output.exists():
        raise FileExistsError(f"Refusing to replace existing study: {output}")
    if pool_size < 4 or pool_size > 4096:
        raise ValueError("pool-size must be in [4, 4096]")
    source = read(config_path)
    env_args = source.get("env_args", source)
    old_hashes = set()
    prior_manifests = sorted((ROOT / "sc2_skill_adapter" / "runs").glob("*/manifest.json"))
    for path in prior_manifests:
        for row in read(path).get("episodes", []):
            if row.get("sha256"):
                old_hashes.add(row["sha256"])
    pool = generate_manifest(env_args, pool_size, seed)
    strata = (("no_medivac_control", 0, 1), ("one_medivac", 1, 1),
              ("multiple_medivacs", 2, 2))
    selected, seen = [], set(old_hashes)
    for label, desired, count in strata:
        matches = []
        for row in pool["episodes"]:
            roster = row["episode_config"]["team_gen"]["ally_team"]
            medics = sum(str(unit).lower() == "medivac" for unit in roster)
            fits = medics == desired if desired < 2 else medics >= 2
            if fits and row["sha256"] not in seen:
                value = copy.deepcopy(row)
                value.update(source_index=row["index"], study_group=label,
                             medivac_count=medics)
                matches.append(value)
                seen.add(row["sha256"])
                if len(matches) == count:
                    break
        if len(matches) != count:
            raise ValueError(f"Pool has insufficient unused scenarios for {label}; no output created")
        selected.extend(matches)
    for index, row in enumerate(selected):
        row["index"] = index
    body = {"version": 1, "seed": seed, "env_args": env_args, "episodes": selected,
            "study": "medivac-direct-tree-smoke-v1",
            "sampling": "First unused 0/1/2+ medivac cases in seeded pool; counts 1/1/2; outcome-blind"}
    manifest = dict(body, sha256=digest(body))
    validate_manifest(manifest)
    protocol = {
        "version": 1, "purpose": "Integration smoke only; no performance or transfer claim",
        "manifest_sha256": manifest["sha256"], "episodes_per_arm": 4,
        "max_instances": 1, "max_episodes_per_run": 40, "max_env_steps_per_episode": 150,
        "arms": {
            "legacy_off": {"medivac_policy": "legacy", "medical_patient_coordination": "off"},
            "tree": {"medivac_policy": "tree", "medical_patient_coordination": "off"},
            "legacy_on": {"medivac_policy": "legacy", "medical_patient_coordination": "on", "optional": True},
            "shadow": {"medivac_policy": "tree-shadow", "medical_patient_coordination": "off", "optional": True},
        },
        "primary_comparison": ["legacy_off", "tree"],
        "expected_main_total_episodes": 8, "with_reference_total_episodes": 12,
        "source_config": str(config_path.resolve()),
        "source_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "source_pool_seed": seed, "source_pool_size": pool_size,
        "prior_run_manifests_checked": len(prior_manifests),
        "excluded_scenario_hash_count": len(old_hashes),
        "limitations": ["Composition-enriched sample; pooled win rate is not native-distribution performance",
                        "Matching initial conditions is not an in-memory SC2 fork",
                        "Initial hand-authored tree is not behavior-equivalent to the legacy policy"],
    }
    output.mkdir(parents=True, exist_ok=False)
    for name, value in (("manifest.json", manifest), ("protocol.json", protocol),
                        ("sampling_pool.json", pool)):
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"output": str(output.resolve()), "manifest_sha256": manifest["sha256"],
            "episodes": [{key: row[key] for key in ("index", "sc2_seed", "study_group", "medivac_count", "sha256")}
                         for row in selected]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path,
                        default=ROOT / "sc2_skill_adapter/configs/terran_g4_screen_n40_seed400.json")
    parser.add_argument("--seed", type=int, default=21000)
    parser.add_argument("--pool-size", type=int, default=256)
    args = parser.parse_args()
    print(json.dumps(prepare(args.output_dir, args.config, args.seed, args.pool_size), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
