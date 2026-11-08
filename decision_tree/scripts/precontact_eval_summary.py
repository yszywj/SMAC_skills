"""Summarize paired results for the standalone pre-contact regroup skill.

Consumes completed decision_tree.local.evaluate outputs. Event logs are
streamed and bound to the official evaluation's provenance hashes.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from statistics import mean

from decision_tree.local.evaluate import require
from decision_tree.local.runtime import digest, load_policy
from skills.precontact_regroup import POLICY_PATH


def read_json(path, expected_hash=None):
    data = path.read_bytes()
    if expected_hash is not None:
        require(hashlib.sha256(data).hexdigest() == expected_hash, f"Changed file: {path}")
    return json.loads(data)


def proposal_detail(event, candidate):
    require(candidate["family"] == "regroup_step" and len(candidate["actors"]) == 1
            and candidate["duration_steps"] == 1, "Unexpected proposed skill")
    actor, = candidate["actors"]
    slot = int(actor[1:])
    actions = event["final_actions"]
    base = event["baseline_actions"]
    changed = [i for i, (old, new) in enumerate(zip(base, actions)) if new != old]
    veto = event["vetoed"]
    require(len(base) == len(actions) and changed == ([] if veto else [slot])
            and event["selected_id"] == ("BASE" if veto else candidate["id"])
            and set(candidate["actions"]) == {actor}
            and candidate["actions"][actor] != base[slot], "Proposal action scope mismatch")
    require(actions[slot] == (base[slot] if veto else candidate["actions"][actor]),
            "Wrong action for the selected actor")
    features = candidate["features"]
    return {"status": "vetoed" if veto else "executed", "step": event["step"],
            "game_loop": event["game_loop"], "candidate_id": candidate["id"],
            "actor": actor, "base_action": base[slot], "proposed_action": candidate["actions"][actor],
            "submitted_action": actions[slot], "nearest_enemy": features["nearest_enemy_before"],
            "nearest_ally": features["nearest_ally_before"], "alive_allies": features["ally_alive_count"]}


def read_events(path, expected_hash):
    hasher = hashlib.sha256()
    episodes = defaultdict(lambda: {"observations": {}, "steps": {}, "menus": {}, "proposal": None})
    with path.open("rb") as stream:
        for line in stream:
            hasher.update(line)
            event = json.loads(line)
            kind = event["event"]
            if kind not in ("skill_observation", "step", "local_decision"):
                continue
            episode = episodes[event["episode"]]
            step = event["step"]
            if kind == "skill_observation":
                payload = event["payload"]
                payload["state"].pop("episode", None)
                episode["observations"][step] = digest(payload)
            elif kind == "local_decision":
                candidates = event["candidates"]
                episode["menus"][step] = sorted({c["family"] for c in candidates})
                if event["proposed_id"] != "BASE":
                    require(episode["proposal"] is None, "Repeated proposal in one episode")
                    candidate = next(c for c in candidates if c["id"] == event["proposed_id"])
                    episode["proposal"] = proposal_detail(event, candidate)
                else:
                    require(event["selected_id"] == "BASE" and event["baseline_actions"] == event["final_actions"],
                            "Unexpected action outside proposal")
            else:
                decision_step = step - 1
                episode["steps"][decision_step] = digest({k: event[k] for k in
                    ("actions", "reward", "feedback", "game_loop", "terminated")})
                proposal = episode["proposal"]
                if proposal and proposal["step"] == decision_step:
                    actor = proposal["actor"]
                    receipt = event["feedback"]["actors"][actor]
                    action = proposal["submitted_action"]
                    require(event["actions"][int(actor[1:])] == action and event["game_loop"] == proposal["game_loop"]
                            and int(receipt["requested_action_id"]) == action and receipt["submission"] == "accepted",
                            "Missing or mismatched native receipt")
                    proposal["native_submission"] = "accepted"
    require(hasher.hexdigest() == expected_hash, f"Changed event log: {path}")
    return episodes


def arm_records(path, source, policy, manifest):
    hashes = source["hashes"]
    require(Path(source["path"]).resolve() == path.resolve(), "Arm provenance path mismatch")
    metadata = read_json(path/"run.json", hashes["run.json"])
    require(metadata["local_policy"]["policy"] == policy, "Arm policy mismatch")
    require(read_json(path/"manifest.json", hashes["manifest.json"]) == manifest, "Arm manifest mismatch")
    require(read_json(path/"summary.json", hashes["summary.json"])["status"] == "complete", "Incomplete arm")
    records = read_json(path/"episodes.json", hashes["episodes.json"])
    events = read_events(path/"events.jsonl", hashes["events.jsonl"])
    require(len(records) == len(events) == len(manifest["episodes"]), "Arm episode count mismatch")
    return records, events


def enrich(row, scene, base, model, base_log, model_log):
    require(row["sc2_seed"] == scene["sc2_seed"] and row["group_id"] == scene["sha256"]
            and base["initial_state_sha256"] == model["initial_state_sha256"], "Mismatched pair")
    for arm, record, log in (("base", base, base_log), ("model", model, model_log)):
        indices = set(range(record["steps"]))
        require(record["scenario_sha256"] == scene["sha256"] and record["outcome"] == row[arm+"_outcome"]
                and record["reward"] == row[arm+"_return"] and record["steps"] == row[arm+"_steps"]
                and set(log["steps"]) == set(log["observations"]) == set(log["menus"]) == indices,
                "Recorded episode differs from evaluated row")
    proposal = model_log["proposal"]
    status = proposal["status"] if proposal else "no_proposal"
    require(base_log["proposal"] is None and int(bool(proposal)) == row["proposals"]
            and int(status == "vetoed") == row["vetoes"] and int(status == "executed") == row["interventions"],
            "Proposal counts differ from evaluated row")
    require(not proposal or proposal.get("native_submission") == "accepted", "Missing receipt")
    require(row["gain"] == model["reward"] - base["reward"], "Return difference mismatch")

    # Compare the actual histories through the intervention observation.
    # If no patch was executed, compare the entire observable trajectory.
    if status == "executed":
        anchor = proposal["step"]
        prefix_matches = (all(base_log["observations"].get(t) == model_log["observations"][t]
                              for t in range(anchor+1))
                          and all(base_log["steps"].get(t) == model_log["steps"][t] for t in range(anchor)))
    else:
        prefix_matches = (base_log["observations"] == model_log["observations"]
                          and base_log["steps"] == model_log["steps"])
    later = [families for t, families in base_log["menus"].items()
             if proposal and t > proposal["step"] and families]
    return {**row, "status": status, "proposal": proposal, "observed_prefix_matches": prefix_matches,
            "base_regroup_opportunity_steps": sum("regroup_step" in menu for menu in base_log["menus"].values()),
            "base_later_candidate_steps_after_proposal": len(later) if proposal else None,
            "base_later_candidate_steps_by_family": dict(Counter(f for menu in later for f in menu))}


def read_evaluation(folder):
    result = read_json(folder/"result.json")
    manifest = read_json(folder/"manifest.json")
    protocol = read_json(folder/"protocol.json")
    policy = load_policy(folder/"model_policy.json")
    base_policy = load_policy(folder/"base_policy.json")
    require(result["kind"] == "local_patch_deployment_evaluation"
            and result["protocol_sha256"] == digest(protocol)
            and protocol["manifest_sha256"] == digest(manifest)
            and protocol["model_policy_sha256"] == digest(policy)
            and policy == load_policy(POLICY_PATH) and base_policy["selection"]["mode"] == "base"
            and protocol["base_policy_sha256"] == digest(base_policy), "Unexpected evaluation or skill")
    base, be = arm_records(folder/"base", result["provenance"]["BASE"], base_policy, manifest)
    model, me = arm_records(folder/"model", result["provenance"]["MODEL"], policy, manifest)
    require(result["episodes"] == len(base) == len(result["rows"]) and result["completed_games"] == 2*len(base),
            "Incomplete result rows")
    rows = []
    for number, (scene, row) in enumerate(zip(manifest["episodes"], result["rows"]), 1):
        require(row["episode"] == base[number-1]["episode"] == model[number-1]["episode"] == number,
                "Episode number mismatch")
        rows.append(enrich(row, scene, base[number-1], model[number-1], be[number], me[number]))
    return rows


def summary(rows):
    require(rows and len({r["sc2_seed"] for r in rows}) == len({r["group_id"] for r in rows}) == len(rows),
            "Empty or repeated scenes/seeds")
    rescued = [r["sc2_seed"] for r in rows if r["base_outcome"] != "win" and r["model_outcome"] == "win"]
    lost = [r["sc2_seed"] for r in rows if r["base_outcome"] == "win" and r["model_outcome"] != "win"]
    stats = {}
    for arm in ("base", "model"):
        stats[arm] = {"outcomes": dict(Counter(r[arm+"_outcome"] for r in rows)),
                      "mean_return": mean(r[arm+"_return"] for r in rows)}
    by_status = {}
    for status in ("no_proposal", "vetoed", "executed"):
        group = [r for r in rows if r["status"] == status]
        by_status[status] = {"episodes": len(group), "rescued": sum(r["sc2_seed"] in rescued for r in group),
                             "lost": sum(r["sc2_seed"] in lost for r in group),
                             "mean_gain": mean(r["gain"] for r in group) if group else None}
    return {"independent_scene_count": len(rows), "completed_games": 2*len(rows), "arms": stats,
            "rescued_seeds": rescued, "lost_seeds": lost,
            "same_win_status": len(rows)-len(rescued)-len(lost),
            "proposals": sum(r["status"] != "no_proposal" for r in rows),
            "by_proposal": by_status, "mean_gain": mean(r["gain"] for r in rows),
            "improved": sum(r["gain"] > 1e-9 for r in rows),
            "harmed": sum(r["gain"] < -1e-9 for r in rows),
            "unchanged": sum(abs(r["gain"]) <= 1e-9 for r in rows),
            "observed_prefix_mismatch_seeds": [r["sc2_seed"] for r in rows if not r["observed_prefix_matches"]],
            "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evaluations", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, help="save a new full JSON report; stdout shows totals")
    args = parser.parse_args()
    report = summary([row for path in args.evaluations for row in read_evaluation(path)])
    report["evaluations"] = [str(p.resolve()) for p in args.evaluations]
    if args.output:
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
        report = {key: value for key, value in report.items() if key != "rows"}
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
