"""Check v3 action legality on recorded BASE observations without starting SC2."""
import argparse
from collections import Counter
import json
from pathlib import Path

from decision_tree.conditional.policy import gate, normalize_policy
from skills.local_v1 import apply_candidate
from skills.local_v2.explore import checksum, require
from .library import catalog, recommended_candidate
from .run import source_hashes


def replay(plan):
    summary = json.loads((plan/"summary.json").read_text(encoding="utf-8"))
    require(checksum(plan/"menus.jsonl") == summary["menus_sha256"], "Changed registered menu")
    counts, decisions, reasons = Counter(), Counter(), Counter()
    policy = normalize_policy()
    menus = iter((plan/"menus.jsonl").read_text(encoding="utf-8").splitlines())
    payload = None
    base = Path(summary["base"])
    require(checksum(base/"events.jsonl") == summary["base_hashes"]["events.jsonl"], "Changed observations")
    with (base/"events.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            event = json.loads(line)
            if event["event"] == "skill_observation":
                payload = event
            elif event["event"] in ("local_decision", "local_v2_decision"):
                require(payload is not None and payload["episode"] == event["episode"] and
                        payload["step"] == event["step"], "Missing matching pre-action observation")
                expected = json.loads(next(menus))
                require((expected["episode"], expected["step"]) == (event["episode"], event["step"])
                        and event["selected_id"] == "BASE", "Unexpected source policy")
                baseline = {"actions": event["baseline_actions"], "roles": event["baseline_roles"],
                            "predicted_damage": event["baseline_predicted_damage"]}
                current = catalog(payload["payload"], baseline)
                require(current["effects"] == expected["effects"], "Exploration effects were removed or changed")
                counts["frames"] += 1
                counts["nonbase_effects"] += len(current["effects"]) - 1
                for row in current["bindings"]:
                    patch = recommended_candidate(payload["payload"], baseline, row)
                    reason = gate(row, patch, policy)
                    reasons[row["family"] + "/" + reason] += 1
                    if patch:
                        result = apply_candidate(payload["payload"], baseline, patch)
                        changed = [i for i, (a, b) in enumerate(zip(result["actions"], baseline["actions"])) if a != b]
                        require(changed == [int(row["actor"][1:])], "Unexpected action ownership")
                    if reason == "eligible":
                        decisions[row["family"]] += 1
                payload = None
    require(next(menus, None) is None, "Unexamined registered frames")
    return {"kind": "offline_observation_checks", "scope": "existing_BASE_observations_only",
            "plan_sha256": checksum(plan/"summary.json"), "source_hashes": source_hashes(),
            "counts": dict(counts), "eligible_bindings_before_scheduler": dict(decisions),
            "condition_counts": dict(reasons), "native_games_started": 0,
            "counterfactual_returns_computed": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = replay(args.plan)
    with args.output.open("x", encoding="utf-8") as out:
        json.dump(result, out, indent=2, ensure_ascii=False, allow_nan=False)
        out.write("\n")
    print(json.dumps(result["counts"]))
