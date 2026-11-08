"""Outcome-blind, four-family sampling on recorded conditional-policy trajectories.

The recorded full catalog remains the exploration denominator. Each training
scene receives at most one distinct effect from each family and four games in
total. Scene hashes and round numbers rotate early, middle and late anchors.
"""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from decision_tree.local.runtime import digest
from skills.local_v1 import FEATURE_NAMES, generate_candidates
from skills.local_v1.library import _candidate
from skills.local_v2.explore import checksum, require
from skills.local_v2.regroup import analyze as regroup_v2
from skills.local_v3.run import check_completion


FAMILIES = ("regroup_step", "retarget", "retreat_step", "retreat_veto")
PHASES = ("early", "middle", "late")


def _read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _rank(*parts):
    return hashlib.sha256(json.dumps(parts, separators=(",", ":")).encode()).hexdigest()


def check_budget(jobs):
    by_scene = defaultdict(list)
    for job in jobs:
        require(job["family"] in FAMILIES, "Unknown exploration family")
        by_scene[job["scene_hash"]].append(job)
    for scene, rows in by_scene.items():
        require(len(rows) <= 4, "Scene exceeds four measured branches")
        require(len({row["step"] for row in rows}) <= 4, "Scene exceeds four measured times")
        require(len({(row["step"], row["effect"]) for row in rows}) == len(rows), "Repeated measured effect")
        require(max(Counter(row["family"] for row in rows).values()) <= 1, "Scene repeats a family quota")
        for step in {row["step"] for row in rows}:
            selected = [row for row in rows if row["step"] == step]
            require(len(selected) <= 8, "Time exceeds eight measured branches")
            aliases = Counter()
            for row in selected:
                aliases.update(set(row["effect_families"]))
            require(max(aliases.values(), default=0) <= 2, "Time exceeds two branches of a family")


def _select_jobs(opportunities, train_groups, round_index):
    by_scene = defaultdict(lambda: defaultdict(list))
    for row in opportunities:
        require(row["scene_hash"] in train_groups, "Holdout opportunity entered sampling")
        by_scene[row["scene_hash"]][row["family"]].append(row)
    selected, coverage = [], []
    for scene in sorted(train_groups):
        buckets = by_scene[scene]
        offset = int(_rank(scene, round_index)[:8], 16)
        order = FAMILIES[offset % len(FAMILIES):] + FAMILIES[:offset % len(FAMILIES)]
        used = set()
        for index, family in enumerate(order):
            phase = (offset + round_index + index) % len(PHASES)
            candidates = [row for row in buckets[family] if (row["step"], row["effect"]) not in used
                          and _fits_alias_budget(selected, row)]
            if not candidates:
                coverage.append({"scene_hash": scene, "family": family, "status": "no_distinct_alternative"})
                continue
            # A phase is allocated before looking at return labels. Candidate
            # preference only chooses among that phase's legal alternatives.
            chosen = min(candidates, key=lambda row: (
                (PHASES.index(row["phase"]) - phase) % len(PHASES),
                int((row["binding"] == "BASE") != ((offset + round_index + index) % 2 == 0)),
                row["preference"],
                _rank(scene, round_index, family, row["step"], row["binding"])))
            chosen = {**chosen, "requested_phase": PHASES[phase]}
            used.add((chosen["step"], chosen["effect"]))
            selected.append(chosen)
            coverage.append({"scene_hash": scene, "family": family, "status": "selected",
                             "available_bindings": len(buckets[family]), "requested_phase": PHASES[phase],
                             "actual_phase": chosen["phase"]})
    check_budget(selected)
    return selected, coverage


def _fits_alias_budget(selected, candidate):
    aliases = Counter(set(candidate["effect_families"]))
    for row in selected:
        if (row["scene_hash"], row["step"]) == (candidate["scene_hash"], candidate["step"]):
            aliases.update(set(row["effect_families"]))
    return max(aliases.values(), default=0) <= 2


def _opportunities(event, observation, scene, total_steps):
    """Enumerate all distinct alternatives, including blocked and unselected actors."""
    menu = event["catalog"]
    baseline = {"actions": event["baseline_actions"], "roles": event["baseline_roles"],
                "predicted_damage": event["baseline_predicted_damage"]}
    aliases = {alias: effect for effect, row in menu["effects"].items() for alias in row["aliases"]}
    selected_id = event["selected_id"]
    actual_effect = "BASE" if selected_id == "BASE" else aliases[selected_id]
    selected_family = None if selected_id == "BASE" else selected_id.split(":")[0]
    frozen_v1 = {row["id"] for row in generate_candidates(observation, baseline,
                                                          {"max_candidates": 8, "max_per_family": 2})}
    phase = PHASES[min(2, event["step"] * 3 // total_steps)]
    common = {"episode": event["episode"], "scene_hash": scene["sha256"], "seed": scene["sc2_seed"],
              "step": event["step"], "selected_id": selected_id, "selected_effect": actual_effect,
              "phase": phase, "decision_sha256": digest(event)}
    rows = []
    for binding in menu["bindings"]:
        family, actor = binding["family"], binding["actor"]
        require(family in FAMILIES, "Unexpected family in complete catalog")
        v2_action = None
        if family == "regroup_step":
            v2_action = regroup_v2(observation, baseline, actor)["recommended_action"]
        recommended = binding["recommended_action"]
        for option in binding["options"]:
            action = option["action"]
            cid = f"{family}:{actor}:{action}"
            effect = aliases[cid]
            if effect == actual_effect:
                continue
            if effect == "BASE":
                # Cancel the actual incumbent change once, attributed to its
                # selected family. BASE aliases belonging to other actors do
                # not create new cancellation hypotheses.
                if family != selected_family or actor != selected_id.split(":")[1]:
                    continue
                forced = "BASE"
                preference = 0
                selected_action = int(selected_id.split(":")[2])
                selected_role = {"regroup_step": "reposition", "retreat_step": "retreat",
                                 "retreat_veto": "attack"}.get(family, baseline["roles"][actor])
                features = _candidate(observation, baseline, actor, selected_action, family, selected_role)["features"]
            else:
                forced = cid
                version_difference = action != recommended and (cid in frozen_v1 or action == v2_action)
                preference = 0 if version_difference else 1 if action == recommended else 2
                role = {"regroup_step": "reposition", "retreat_step": "retreat", "retreat_veto": "attack"}.get(
                    family, baseline["roles"][actor])
                features = _candidate(observation, baseline, actor, action, family, role)["features"]
            families = sorted({name.split(":")[0] for name in menu["effects"][effect]["aliases"]})
            rows.append({**common, "family": family, "actor": actor, "binding": forced, "effect": effect,
                         "effect_families": families if effect != "BASE" else [family],
                         "label_kind": "veto" if forced == "BASE" else "gain",
                         "preference": preference, "features": features,
                         "recommended_action": recommended, "forced_action": action,
                         "old_binding_in_menu": cid in frozen_v1})
    # One effect can have several equivalent labels within the same family.
    unique = {}
    for row in rows:
        key = row["family"], row["effect"]
        current = unique.get(key)
        if current is None or (row["preference"], row["binding"]) < (current["preference"], current["binding"]):
            unique[key] = row
    return list(unique.values())


def make_plan(incumbent_path, train_groups, round_index):
    """Return a frozen measurement plan; starts no game and never ranks by outcomes."""
    incumbent_path = Path(incumbent_path).resolve()
    train_groups = set(train_groups)
    require(type(round_index) is int and round_index >= 0, "Round index must be a nonnegative integer")
    manifest = _read(incumbent_path / "manifest.json")
    native = _read(incumbent_path / "run.json")
    metadata = native["conditional_skills"]
    require(not metadata.get("forced"), "Exploration needs the incumbent's unforced trajectories")
    scenes = {n: scene for n, scene in enumerate(manifest["episodes"], 1)}
    groups = {scene["sha256"] for scene in scenes.values()}
    require(train_groups and train_groups <= groups, "Training scene IDs absent from incumbent manifest")
    check_completion(incumbent_path, metadata, len(scenes))
    source_hash = checksum(incumbent_path / "events.jsonl")
    # Only length metadata is used. Rewards and outcomes do not influence the plan.
    lengths = {r["episode"]: r["steps"] for r in _read(incumbent_path / "episodes.json")}
    census = {name: Counter() for name in ("train", "holdout")}
    family_census = {name: Counter() for name in census}
    opportunities = []
    observation = None
    with (incumbent_path / "events.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            event = json.loads(line)
            if event["event"] == "skill_observation":
                observation = event
            elif event["event"] == "conditional_local_decision":
                require(observation is not None and (event["episode"], event["step"]) ==
                        (observation["episode"], observation["step"]), "Observation/decision mismatch")
                scene = scenes[event["episode"]]
                partition = "train" if scene["sha256"] in train_groups else "holdout"
                menu = event["catalog"]
                census[partition]["decision_steps"] += 1
                census[partition]["registered_nonbase_effects"] += len(menu["effects"]) - 1
                census[partition]["steps_with_nonbase_effects"] += int(len(menu["effects"]) > 1)
                for binding in menu["bindings"]:
                    family_census[partition][binding["family"]] += len(binding["options"])
                if partition == "train":
                    opportunities.extend(_opportunities(event, observation["payload"], scene, lengths[event["episode"]]))
                observation = None
    require(checksum(incumbent_path / "events.jsonl") == source_hash, "Source changed while planning")
    jobs, coverage = _select_jobs(opportunities, train_groups, round_index)
    return {"kind": "bounded_conditional_counterfactual_plan", "round_index": round_index,
            "incumbent": str(incumbent_path), "incumbent_events_sha256": source_hash,
            "incumbent_manifest_sha256": checksum(incumbent_path / "manifest.json"),
            "incumbent_policy_sha256": metadata["policy_sha256"],
            "source_hashes": metadata["source_hashes"], "feature_names": list(FEATURE_NAMES),
            "label_reference": "Measured whole return after one forced change and the same conditional continuation, minus incumbent whole return",
            "training_groups": sorted(train_groups), "holdout_groups": sorted(groups - train_groups),
            "bounds": {"branches_per_scene": 4, "times_per_scene": 4,
                       "branches_per_time": 8, "family_branches_per_time": 2,
                       "family_branches_per_scene": 1, "max_total_branches": 4 * len(train_groups)},
            "census": {key: dict(value) for key, value in census.items()},
            "family_option_census": {key: dict(value) for key, value in family_census.items()},
            "available_alternative_bindings": len(opportunities), "planned_branch_games": len(jobs),
            "unmeasured_training_effects": census["train"]["registered_nonbase_effects"] - len(jobs),
            "sampling": "Scene/round hash rotates temporal thirds and family order; action-version differences preferred within phase; no outcome ranking",
            "jobs": jobs, "coverage": coverage}
