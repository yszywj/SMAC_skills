"""Freeze development, validation and test scenarios without starting SC2.

Selection uses initial roster membership and scenario hashes only. Outcomes,
decision traces and intervention trigger rates are never consulted.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sc2_skill_adapter.bridge.config import digest, validate_config
from sc2_skill_adapter.bridge.scenarios import generate_manifest, validate_manifest


class InsufficientCurriculumError(ValueError):
    def __init__(self, stats):
        self.stats = stats
        super().__init__(f"Fixed pool has {stats['eligible_unique_scenarios']} eligible unused scenarios; "
                         f"{stats['requested_scenarios']} required. No files written; "
                         "pool size, split sizes and eligibility were not changed.")


def read(path):
    raw = Path(path).read_bytes()
    return json.loads(raw.decode("utf-8-sig")), hashlib.sha256(raw).hexdigest()


def _write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _positive(value, name, maximum):
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [1, {maximum}]")


def previous_scenarios(run_roots):
    """Conservatively exclude every recorded scenario, including unstarted ones."""
    paths = sorted({path.resolve() for root in run_roots if Path(root).exists()
                    for path in Path(root).rglob("manifest.json")})
    hashes, sources = set(), []
    for path in paths:
        manifest, file_hash = read(path)
        validate_manifest(manifest)
        scenario_hashes = {row["sha256"] for row in manifest["episodes"]}
        hashes.update(scenario_hashes)
        sources.append({"path": str(path), "sha256": file_hash,
                        "manifest_sha256": manifest["sha256"], "unique_scenarios": len(scenario_hashes)})
    return hashes, sources


def prepare(output, config_path, *, seed=31000, pool_size=512, dev_episodes=40,
            validation_episodes=20, test_episodes=20, run_roots=None):
    """Prepare one fixed batch per split (at most 40 episodes in each)."""
    output, config_path = Path(output).resolve(), Path(config_path).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite curriculum: {output}")
    for name, value in (("dev_episodes", dev_episodes), ("validation_episodes", validation_episodes),
                        ("test_episodes", test_episodes)):
        _positive(value, name, 40)
    _positive(pool_size, "pool_size", 4096)
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    source, source_hash = read(config_path)
    env_args = source.get("env_args", source)
    validate_config(env_args)
    if env_args["map_name"] != "10gen_terran":
        raise ValueError("Medical return curriculum requires a Terran environment")
    if "episodes" in source:
        validate_manifest(source)
    run_roots = [ROOT / "sc2_skill_adapter/runs"] if run_roots is None else [Path(p).resolve() for p in run_roots]
    old_hashes, exclusion_sources = previous_scenarios(run_roots)
    pool = generate_manifest(env_args, pool_size, seed)
    validate_manifest(pool)
    requested = dev_episodes + validation_episodes + test_episodes
    stats = {"requested_scenarios": requested, "pool_size": len(pool["episodes"]),
             "without_medivac": 0, "previously_recorded": 0, "duplicate_in_pool": 0,
             "eligible_unique_scenarios": 0, "excluded_prior_unique_scenarios": len(old_hashes),
             "prior_manifests_checked": len(exclusion_sources), "started_games": 0}
    seen, eligible = set(), []
    for row in pool["episodes"]:
        group = row["sha256"]
        roster = row["episode_config"]["team_gen"]["ally_team"]
        medivacs = sum(str(name).lower() == "medivac" for name in roster)
        if medivacs == 0:
            stats["without_medivac"] += 1
        elif group in old_hashes:
            stats["previously_recorded"] += 1
        elif group in seen:
            stats["duplicate_in_pool"] += 1
        else:
            selected = copy.deepcopy(row)
            selected.update(source_index=row["index"], medivac_count=medivacs)
            eligible.append(selected)
            seen.add(group)
    stats["eligible_unique_scenarios"] = len(eligible)
    if len(eligible) < requested:
        raise InsufficientCurriculumError(stats)
    counts = {"development": dev_episodes, "validation": validation_episodes, "test": test_episodes}
    purposes = {
        "development": "Collect paired data, diagnose failures and fit the return selector.",
        "validation": "Keep sealed during fitting; evaluate a frozen selector, not used to fit its tree.",
        "test": "Keep sealed until the final design is fixed; use once for final reporting.",
    }
    manifests, splits = {}, {}
    cursor = 0
    for name, count in counts.items():
        records = copy.deepcopy(eligible[cursor:cursor + count])
        cursor += count
        for index, row in enumerate(records):
            row["index"] = index
        body = {"version": 1, "seed": seed, "env_args": copy.deepcopy(env_args), "episodes": records,
                "study": "medivac-return-curriculum-v1", "split": name,
                "sampling": "First eligible unused scenarios in a fixed seeded pool; initial roster and hashes only"}
        manifests[name] = dict(body, sha256=digest(body))
        validate_manifest(manifests[name])
        splits[name] = {"manifest": f"{name}/manifest.json", "manifest_sha256": manifests[name]["sha256"],
                        "episodes": count, "scenario_groups": [r["sha256"] for r in records],
                        "source_indices": [r["source_index"] for r in records],
                        "sc2_seeds": [r["sc2_seed"] for r in records], "purpose": purposes[name],
                        "sealed_holdout": name != "development", "automatic_execution": False}
    declaration = {
        "schema_version": 1, "purpose": "first_medivac_return_learning_curriculum",
        "seed": seed, "pool_size": pool_size, "requested_counts": counts,
        "source_config": {"path": str(config_path), "sha256": source_hash},
        "pool_manifest_sha256": pool["sha256"], "env_args_sha256": digest(env_args),
        "selection": {"order": "fixed_generator_order", "requires_at_least_one_medivac": True,
                      "unique_by": "scenario_sha256", "outcome_blind": True, "trigger_rate_blind": True},
        "exclusion_roots": [str(p.resolve()) for p in run_roots], "exclusion_sources": exclusion_sources,
        "excluded_scenario_hashes": sorted(old_hashes), "splits": splits, "stats": stats,
        "holdout_rules": ["No validation or test games are started by preparation.",
                          "Any holdout used to redesign features, thresholds or candidates becomes development evidence.",
                          "All seeds of a scenario hash remain one independent group.",
                          "Further development batches must exclude this curriculum's sealed holdout scenario hashes."],
        "limitations": ["Medivac-enriched initial composition; not the unfiltered environment distribution.",
                        "Counts are reserved scenarios, not guaranteed accepted counterfactual labels.",
                        "No failure score, observed result, trigger rate or rollout influenced selection.",
                        "No automatic pool expansion, sample-size reduction or tree coverage relaxation."]}
    declaration["sha256"] = digest(declaration)
    output.mkdir(parents=True, exist_ok=False)
    for name, manifest in manifests.items():
        (output / name).mkdir()
        _write(output / name / "manifest.json", manifest)
    _write(output / "sampling_pool.json", pool)
    _write(output / "split.json", declaration)
    return {"output": str(output), "status": "prepared_no_games_started", "stats": stats,
            "split_sha256": declaration["sha256"],
            "manifests": {name: str(output / spec["manifest"]) for name, spec in splits.items()}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path,
                        default=ROOT / "sc2_skill_adapter/configs/terran_g4_screen_n40_seed400.json")
    parser.add_argument("--seed", type=int, default=31000)
    parser.add_argument("--pool-size", type=int, default=512)
    parser.add_argument("--dev-episodes", type=int, default=40)
    parser.add_argument("--validation-episodes", type=int, default=20)
    parser.add_argument("--test-episodes", type=int, default=20)
    parser.add_argument("--exclude-run-root", type=Path, action="append",
                        help="Additional existing run or sealed curriculum root to exclude; repeatable")
    args = parser.parse_args(argv)
    roots = [ROOT / "sc2_skill_adapter/runs", *(args.exclude_run_root or [])]
    try:
        result = prepare(args.output_dir, args.config, seed=args.seed, pool_size=args.pool_size,
                         dev_episodes=args.dev_episodes, validation_episodes=args.validation_episodes,
                         test_episodes=args.test_episodes, run_roots=roots)
    except InsufficientCurriculumError as exc:
        print(json.dumps({"status": "insufficient_fixed_pool", "error": str(exc), "stats": exc.stats}), file=sys.stderr)
        return 2
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
