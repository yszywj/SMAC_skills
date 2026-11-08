"""Reconstruct one-intervention policies from audited BASE prefixes and forced games.

prepare never exports holdout observations or outcomes. compare replays the same
Selector as deployment. A missing forced game remains unknown, even if another
candidate at the same step was measured. These are reconstructed training results;
the held-out complete-policy rollouts are collected separately.
"""
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
import math
from pathlib import Path

from decision_tree.local.runtime import digest
from decision_tree.local.training import load_dataset
from decision_tree.scripts.local_campaign_validate import validate_cohort
from decision_tree.phase2.policy import Selector, accepted_attack, regroup_bucket

EPS = 1e-9


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def file_sha(path):
    with Path(path).open("rb") as stream:
        h = sha256()
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def bound_read(source, filename):
    path = Path(source["path"]) / filename
    raw = path.read_bytes()
    require(sha256(raw).hexdigest() == source["hashes"][filename], f"Source changed: {path}")
    return json.loads(raw)


def stats(rows):
    """Each scene has equal mass; signed counts below count *scenes*."""
    groups = defaultdict(list)
    for row in rows:
        groups[row["group_id"]].append(row["gain"])
    values = [math.fsum(v)/len(v) for v in groups.values()]
    total = math.fsum(values)
    n = len(values)
    return {"rows": len(rows), "groups": n, "scene_equal_mean_gain": total/n if n else None,
            "positive_groups": sum(v > EPS for v in values),
            "negative_groups": sum(v < -EPS for v in values),
            "zero_groups": sum(abs(v) <= EPS for v in values),
            "leave_one_group_out_range": [(total-max(values))/(n-1), (total-min(values))/(n-1)]
            if n > 1 else None}


def training_frames(source, train_ordinals):
    """Stream one hashed source, retaining only training decision menus."""
    path = Path(source["path"]) / "events.jsonl"
    frames = defaultdict(list)
    fired = defaultdict(bool)
    seen_feedback = Counter()
    h = sha256()
    with path.open("rb") as stream:
        for line in stream:
            h.update(line)
            event = json.loads(line)
            number = event.get("episode")
            if number not in train_ordinals:
                continue
            if event["event"] == "local_decision":
                step = event["step"]
                require(step == len(frames[number]) == seen_feedback[number], "Non-sequential BASE prefix")
                require(event["selected_id"] == "BASE", "BASE policy performed an intervention")
                frames[number].append({"step": step, "team_fired_before": fired[number],
                                       "candidates": event["candidates"]})
            elif event["event"] == "step":
                require(event["step"] == len(frames[number]) == seen_feedback[number]+1, "Feedback order mismatch")
                seen_feedback[number] += 1
                fired[number] = fired[number] or accepted_attack(event["feedback"])
    require(h.hexdigest() == source["hashes"]["events.jsonl"], "BASE event hash changed")
    require(set(frames) == set(train_ordinals), "Missing BASE training prefix")
    for number in train_ordinals:
        require(len(frames[number]) == seen_feedback[number], "Incomplete submitted BASE trajectory")
    return frames


def prepare(dataset_path, split_path):
    input_sha = file_sha(dataset_path)
    data = load_dataset(dataset_path)
    require(file_sha(dataset_path) == input_sha and not data["rejected"], "Incomplete or changing evidence")
    base = data["provenance"]["BASE"]
    manifest = bound_read(base, "manifest.json")
    episodes = bound_read(base, "episodes.json")
    split = read(split_path)
    validate_cohort(manifest, split)
    train = set(split["training_group_ids"])
    ordinals = {i: s for i, s in enumerate(manifest["episodes"], 1) if s["sha256"] in train}
    frames = training_frames(base, ordinals)
    labels = {}
    for arm in data["provenance"]["arms"]:
        entry = arm["entry"]
        number = entry["base_episode"]
        if number in ordinals:
            rec = bound_read(arm, "episodes.json")[entry.get("arm_episode", 1)-1]
            key = (ordinals[number]["sha256"], entry["anchor_step"], entry["candidate_id"])
            require(key not in labels and rec["scenario_sha256"] == key[0], "Unbound or duplicate forced episode")
            labels[key] = rec, arm["path"]
    by_group = defaultdict(list)
    for row in data["rows"]:
        group = row["group_id"]
        if group not in train:
            continue
        record, run = labels.pop((group, row["anchor_step"], row["candidate_id"]))
        require(record["reward"] == row["candidate_return"], "Reward source mismatch")
        by_group[group].append({"step": row["anchor_step"], "candidate_id": row["candidate_id"],
                                "return": row["candidate_return"], "outcome": record["outcome"],
                                "source_run": run})
    require(not labels, "Missing training rows for audited runs")
    scenes = []
    for number, scenario in ordinals.items():
        group = scenario["sha256"]
        rec = episodes[number-1]
        require(rec["scenario_sha256"] == group and rec["steps"] == len(frames[number]), "Incomplete BASE episode")
        scenes.append({"group_id": group, "sc2_seed": scenario["sc2_seed"], "episode": number,
                       "base_return": rec["reward"], "base_outcome": rec["outcome"], "frames": frames[number],
                       "measured": by_group[group]})
    return {"kind": "phase2_training_cohort", "dataset": str(dataset_path.resolve()), "dataset_sha": input_sha,
            "split": str(split_path.resolve()), "split_sha": file_sha(split_path),
            "context": data["context_hash"], "game_version": data["game_version"], "scenes": scenes}


def measured_rows(scene):
    old = Selector({"mode":"regroup"})
    rows = []
    measurements = {(x["step"], x["candidate_id"]): x for x in scene["measured"]}
    for frame in scene["frames"]:
        step, fired, menu = frame["step"], frame["team_fired_before"], frame["candidates"]
        decision = old.decide(menu, step, fired)
        for item in menu:
            record = measurements.pop((step, item["id"]), None)
            if record is None:
                continue
            rows.append({"group_id": scene["group_id"], "sc2_seed": scene["sc2_seed"], "step":step,
                         "candidate_id":item["id"], "family":item["family"],
                         "bucket":(regroup_bucket(item, fired) if item["family"] == "regroup_step" else item["family"]),
                         "old_first_proposal":decision["considered_id"] == item["id"],
                         "team_fired_before":fired, "base_outcome":scene["base_outcome"],
                         "outcome":record["outcome"], "base_return":scene["base_return"],
                         "return":record["return"], "gain":record["return"] - scene["base_return"],
                         "features":item["features"], "source_run":record["source_run"]})
    require(not measurements, "Recorded selected candidate absent from BASE menu")
    return rows


def load_cohorts(paths):
    scenes = []
    heldout, group_ids, seeds, contexts = set(), set(), set(), set()
    for path in paths:
        record = read(path)
        expected = record.pop("record_sha256")
        require(record["kind"] == "phase2_training_cohort" and digest(record) == expected, "Changed preparation")
        require(file_sha(record["dataset"]) == record["dataset_sha"] and file_sha(record["split"]) == record["split_sha"],
                "Evidence or split changed since preparation")
        split = read(record["split"])
        local = record["scenes"]
        ids, new_seeds = {s["group_id"] for s in local}, {s["sc2_seed"] for s in local}
        batch_heldout = set(split["validation_group_ids"])
        require(ids == set(split["training_group_ids"]) and len(ids) == len(new_seeds) == len(local),
                "Prepared groups do not match fixed training cohort")
        require(not (ids | batch_heldout) & (group_ids | heldout) and not (seeds & new_seeds) and not ids & batch_heldout,
                "Duplicate or leaked groups/seeds between training sources")
        heldout |= batch_heldout
        group_ids |= ids
        seeds |= new_seeds
        contexts.add((record["context"], record["game_version"]))
        scenes.extend(local)
    require(len(contexts) == 1, "Incompatible BASE evidence")
    return scenes


def reconstruct(scene, policy):
    selector = Selector(policy)
    waits = []
    for frame in scene["frames"]:
        step = frame["step"]
        decision = selector.decide(frame["candidates"], step, frame["team_fired_before"])
        if decision["action"] == "wait":
            waits.append({"step":step,"candidate":decision["considered_id"],"bucket":decision["bucket"]})
        if decision["action"] == "execute":
            cid = decision["selected_id"]
            row = next((r for r in scene["measured"] if r["step"] == step and r["candidate_id"] == cid), None)
            return {"status":"forced_measured" if row else "unmeasured", "step":step, "selected_id":cid,
                    "waits":waits, "bucket":decision["bucket"], "return":row["return"] if row else None,
                    "outcome":row["outcome"] if row else None, "source_run":row["source_run"] if row else None}
        if decision["action"] == "decline":
            return {"status":"decline_base", "step":step, "selected_id":"BASE", "bucket":decision["bucket"],
                    "waits":waits, "return":scene["base_return"], "outcome":scene["base_outcome"]}
    if selector.policy["mode"] == "forced":
        return {"status":"unmeasured", "step":selector.policy["anchor_step"],
                "selected_id":selector.policy["candidate_id"], "waits":waits, "bucket":"forced",
                "return":None, "outcome":None, "source_run":None, "reason":"anchor_not_reached"}
    return {"status":"no_intervention_base","step":None,"selected_id":"BASE","waits":waits,
            "bucket":None,"return":scene["base_return"],"outcome":scene["base_outcome"]}


def paired_statistics(rows, reference):
    """Exclude only unmeasured arms required for this specific comparison."""
    measured, rescues, lost = [], [], []
    for row in rows:
        tested = row["candidate"]
        old_return = row["base_return"] if reference == "base" else row[reference]["return"]
        old_outcome = row["base_outcome"] if reference == "base" else row[reference]["outcome"]
        if tested["return"] is None or old_return is None:
            continue
        measured.append({"group_id": row["group_id"], "gain": tested["return"] - old_return})
        if old_outcome != "win" and tested["outcome"] == "win":
            rescues.append(row["sc2_seed"])
        if old_outcome == "win" and tested["outcome"] != "win":
            lost.append(row["sc2_seed"])
    return {**stats(measured), "eligible_scenes": len(rows), "complete": len(measured) == len(rows),
            "rescues": rescues, "lost": lost}


def compare(scenes, candidate, incumbent):
    results = []
    groups = defaultdict(list)
    for scene in scenes:
        row = {"group_id": scene["group_id"], "sc2_seed": scene["sc2_seed"],
               "base_return": scene["base_return"], "base_outcome": scene["base_outcome"],
               "candidate": reconstruct(scene, candidate), "incumbent": reconstruct(scene, incumbent)}
        results.append(row)
        groups[row["candidate"]["bucket"] or "no_intervention"].append(row)
    differences = {ref: paired_statistics(results, ref) for ref in ("base", "incumbent")}
    bucket_results = {bucket: {ref: paired_statistics(group, ref) for ref in ("base", "incumbent")}
                      for bucket, group in groups.items()}
    return {"kind": "phase2_reconstructed_training_comparison", "candidate_policy": candidate,
            "incumbent_policy": incumbent, "differences": differences, "by_candidate_terminal_bucket": bucket_results,
            "candidate_coverage": dict(Counter(r["candidate"]["status"] for r in results)),
            "incumbent_coverage": dict(Counter(r["incumbent"]["status"] for r in results)), "rows": results}


def describe(scenes):
    rows = [row for scene in scenes for row in measured_rows(scene)]
    scopes = {}
    for scope in ("all_measured_bindings", "old_first_proposal"):
        sample = rows if scope == "all_measured_bindings" else [r for r in rows if r["old_first_proposal"]]
        scopes[scope] = {key: stats([r for r in sample if r["bucket"]==key]) for key in
                         ("unknown","near_pre","near_engaged","far","retarget","retreat_step","retreat_veto")}
    return {"kind":"phase2_training_diagnostics","training_scenes":len(scenes),"scopes":scopes,"rows":rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--split", type=Path, required=True)
    for name in ("describe", "compare"):
        sub = commands.add_parser(name)
        sub.add_argument("--training", type=Path, required=True, action="append")
        if name == "compare":
            for arg in ("candidate","incumbent"):
                sub.add_argument("--"+arg, type=Path, required=True)
    for sub in commands.choices.values():
        sub.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Refusing to overwrite evidence")
    if args.command == "prepare":
        record = prepare(args.dataset, args.split)
        record["record_sha256"] = digest(record)
    else:
        scenes = load_cohorts(args.training)
        record = describe(scenes) if args.command == "describe" else compare(scenes,read(args.candidate),read(args.incumbent))
        record["sources"] = [{ "path":str(p.resolve()), "sha256":file_sha(p)} for p in args.training]
    with args.output.open("x",encoding="utf-8") as stream:
        json.dump(record, stream, separators=(",",":"), ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output":str(args.output),"kind":record["kind"],
                      "rows":len(record.get("rows",record.get("scenes",[])))}, ensure_ascii=False))


if __name__ == "__main__":
    main()
