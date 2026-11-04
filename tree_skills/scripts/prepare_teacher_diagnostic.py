"""Freeze outcome-blind new scenarios for the four-arm teacher diagnostic."""
import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "sc2_skill_adapter"))
from bridge.config import digest
from bridge.scenarios import generate_manifest, validate_manifest


def prepare(destination):
    destination.mkdir(parents=True, exist_ok=False)
    config = json.loads((ROOT / "sc2_skill_adapter/configs/terran_g4_screen_n40_seed400.json").read_text())["env_args"]
    native = generate_manifest(config, 20, 10000)
    pool = generate_manifest(config, 256, 11000)
    selected = []
    for count, name in ((1, "diagnostic_one_medivac"), (2, "diagnostic_two_plus_medivac")):
        matches = [e for e in pool["episodes"]
                   if (e["episode_config"]["team_gen"]["ally_team"].count("medivac") == 1
                       if count == 1 else e["episode_config"]["team_gen"]["ally_team"].count("medivac") >= 2)]
        if len(matches) < 10:
            raise ValueError("Insufficient outcome-blind composition strata")
        for e in matches[:10]:
            e = copy.deepcopy(e)
            e["study_group"] = name
            selected.append(e)
    episodes = copy.deepcopy(native["episodes"])
    for e in episodes:
        e["study_group"] = "native_distribution"
    episodes.extend(selected)
    for i, e in enumerate(episodes):
        e["source_index"] = e["index"]
        e["index"] = i
    body = {"version": 1, "seed": 10000, "env_args": config, "episodes": episodes,
            "study": "frozen-teacher-four-arm-v1",
            "sampling": "20 native draws seed10000; first 10 one-medivac and first 10 two-plus draws from seed11000 pool; no outcome selection"}
    manifest = dict(body, sha256=digest(body))
    validate_manifest(manifest)
    smoke_pool = generate_manifest(config, 128, 9000)
    smoke = []
    for want in (1, 2):
        match = next(e for e in smoke_pool["episodes"]
                     if (e["episode_config"]["team_gen"]["ally_team"].count("medivac") == 1
                         if want == 1 else e["episode_config"]["team_gen"]["ally_team"].count("medivac") >= 2))
        match = copy.deepcopy(match)
        match["index"] = len(smoke)
        smoke.append(match)
    smoke_body = {"version": 1, "seed": 9000, "env_args": config, "episodes": smoke}
    smoke_manifest = dict(smoke_body, sha256=digest(smoke_body))
    validate_manifest(smoke_manifest)
    protocol = {
        "version": 1, "budget_file": "tree_skills/configs/default_budget.json",
        "formal_episodes_per_arm": 40, "formal_total_episodes": 160,
        "smoke_episodes_per_new_arm": 2, "max_instances": 1,
        "arms": {"A": "all rules", "B": "rules plus conditional teacher healing score",
                 "C": "rule ground units, native actor medivacs", "D": "native actor all units; rules not executed"},
        "checkpoint": "marl-gpt-main.pt", "checkpoint_sha256": "c3deaeb67f679657b27e9d3373e42e4104cc9370be6dba60ab5fd0efe7b1ce5a",
        "last_token": "on", "env_indicator": 0, "history_len": 4,
        "selection": "deterministic native masked argmax; no teacher keep extension; no training",
        "official_actor_config": "eval_configs/smac/terran_v2.yaml: model_type=actor, sample_actions=False, last_token=True",
        "native_mask_caveat": "SMAC native mask admits some full-health healing targets. Preserve that mask for C/D and report TooMuchLife rather than silently adding a wound filter.",
        "primary_comparisons": ["C-A", "C-B", "D-A", "D-C"],
        "report_cohorts_separately": ["native_distribution", "diagnostic_one_medivac", "diagnostic_two_plus_medivac"],
        "limitations": ["small mechanism screen, not final qualification", "composition selection does not guarantee simultaneous injured targets",
                        "D tests checkpoint through the verified local H4/native-balance adapter, not an exact official benchmark reproduction",
                        "diagnostic strata must not be pooled into a claimed native population win rate"],
        "manifest_sha256": manifest["sha256"]}
    for name, obj in (("manifest.json", manifest), ("smoke_manifest.json", smoke_manifest),
                      ("protocol.json", protocol), ("diagnostic_sampling_pool.json", pool)):
        (destination / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"destination": str(destination), "manifest_sha256": manifest["sha256"],
                      "native_medivac_counts": [e["episode_config"]["team_gen"]["ally_team"].count("medivac") for e in episodes[:20]],
                      "formal_seeds": [e["sc2_seed"] for e in episodes],
                      "smoke_seeds": [e["sc2_seed"] for e in smoke]}))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    prepare(p.parse_args().output_dir)
