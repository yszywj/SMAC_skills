"""Join audited, single-step outcomes to pre-action features from one frozen partition.

Regroup has a single v1 centroid choice. For retarget the caller supplies the
original target, so one unit/step can have several distinct v1 comparisons.
An actor/step with no frozen v1 target has one row with old_action=None.
"""
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

from decision_tree.phase2.policy import accepted_attack
from decision_tree.scripts.local_campaign_validate import validate_cohort
from decision_tree.scripts.local_v2_training_report import load_labels, old_actions, read
from skills.local_v2.explore import FILES, SCOPE, checksum, require
from skills.local_v2.run import source_hashes


def _observations(path, ordinals, expected_hash):
    """Retain only current actor observations and previously completed attack receipts."""
    ready = {}
    fired = defaultdict(bool)
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for line in stream:
            h.update(line)
            event = json.loads(line)
            n = event.get("episode")
            if n not in ordinals:
                continue
            if event["event"] == "step":
                fired[n] |= accepted_attack(event["feedback"])
            elif event["event"] == "skill_observation":
                state = event["payload"]["state"]
                visible = [e["position"] for e in state["visible_enemies"].values()
                           if e.get("alive", True) and _has_position(e)]
                before = {}
                for key, unit in state["allies"].items():
                    nearest = None
                    if visible and _has_position(unit):
                        nearest = min(math.dist(unit["position"][:2], pos[:2]) for pos in visible)
                    before[key] = (unit.get("weapon_ready"), nearest)
                index = (n, event["step"])
                require(index not in ready, "Duplicate BASE observation")
                ready[index] = (fired[n], before)
    require(h.hexdigest() == expected_hash, "BASE trajectory changed")
    return ready


def _has_position(unit):
    pos = unit.get("position")
    return isinstance(pos, (tuple, list)) and len(pos) >= 2 and None not in pos[:2]


def _endpoint(aliases, labels, group, step, family, actor, action, base_result):
    if action is None:
        return None, None
    effect = aliases[f"{family}:{actor}:{action}"]
    result = base_result if effect == "BASE" else labels[group, step, effect]
    return effect, result


def _features(binding, step, old_action, seen, unit):
    ready, nearest = unit
    base_action = binding["base_action"]
    new_action = binding["recommended_action"]
    options = {opt["action"]: opt for opt in binding["options"]}
    f = {"step": step, "team_fired": seen, "weapon_ready": ready,
         "nearest_enemy_before": nearest, "base_attack": base_action >= 6,
         "old_is_base": None if old_action is None else old_action == base_action,
         "new_is_base": new_action == base_action}
    if binding["family"] == "regroup_step":
        f.update(known_contact=binding["contact_id"] is not None,
                 support_can_attack=binding["support_basis"] == "can_attack_contact",
                 support_distance=binding["support_distance_before"])
        for prefix, action in (("old", old_action), ("new", new_action)):
            opt = options.get(action, {})
            f[prefix + "_support_gain"] = opt.get("support_gain_vs_base")
            f[prefix + "_contact_cost"] = opt.get("contact_cost_vs_base")
            f[prefix + "_margin"] = opt.get("margin_vs_base")
    else:
        for prefix, action in (("base", base_action), ("old", old_action), ("new", new_action)):
            opt = options.get(action, {})
            damage = opt.get("useful_damage")
            f[prefix + "_kill_possible"] = opt.get("adds_kill_possible")
            f[prefix + "_kill_certain"] = opt.get("adds_kill_certain")
            f[prefix + "_useful_min"] = damage[0] if damage is not None else None
            f[prefix + "_useful_max"] = damage[1] if damage is not None else None
            assigned = opt.get("other_assigned")
            f[prefix + "_other_assigned"] = len(assigned) if assigned is not None else None
        a, b = f["old_useful_min"], f["base_useful_min"]
        f["old_useful_delta"] = a - b if a is not None and b is not None else None
    return f


def load_pairs(campaign, split):
    """Return (rows, metadata), restricted to train or holdout groups in split.json."""
    partition_key = {"train": "training_group_ids", "holdout": "validation_group_ids"}[split]
    campaign = Path(campaign)
    split_path = campaign / "split.json"
    partition = read(split_path)
    summary = read(campaign / "plan" / "summary.json")
    coll = campaign / "collection"
    protocol = read(coll / "protocol.json")
    original = read(campaign / "protocol.json")
    base = Path(summary["base"])
    manifest = read(base / "manifest.json")
    validate_cohort(manifest, partition)
    require(original["split"] == partition and original["manifest_sha256"] == manifest["sha256"],
            "Partitions or frozen manifest changed")
    require(summary["kind"] == "local_v2_exploration_plan" and summary["scope"] == protocol["scope"] == SCOPE
            and source_hashes() == summary["source_hashes"] == protocol["source_hashes"] == original["source_hashes"]
            and summary["menus_sha256"] == protocol["menus_sha256"]
            and summary["base_hashes"] == protocol["base_hashes"]
            and summary["counts"]["nonbase_effects"] == protocol["total"], "Changed collection bindings")
    for filename in FILES[:-1]:
        require(checksum(base / filename) == summary["base_hashes"][filename], "Changed BASE " + filename)
    groups = set(partition[partition_key])
    labels, states = load_labels(coll, groups, protocol)
    episodes = {e["scenario_sha256"]: e for e in read(base / "episodes.json") if e["scenario_sha256"] in groups}
    require(groups == set(episodes), "Missing or mismatched BASE episodes")
    by_n = {n: s for n, s in enumerate(manifest["episodes"], 1) if s["sha256"] in groups}
    outcomes = {g: {"effect": "BASE", "return": ep["reward"], "outcome": ep["outcome"]} for g, ep in episodes.items()}
    observations = _observations(base / "events.jsonl", by_n, summary["base_hashes"]["events.jsonl"])
    expected = set()
    steps = Counter()
    rows = []
    hasher = hashlib.sha256()
    with (campaign / "plan" / "menus.jsonl").open("rb") as stream:
        for line in stream:
            hasher.update(line)
            menu = json.loads(line)
            group = menu["scene_hash"]
            if group not in groups:
                continue
            n, step, seed = menu["episode"], menu["step"], menu["seed"]
            require(by_n[n]["sha256"] == group and seed == by_n[n]["sc2_seed"] and steps[group] == step,
                    "Menu ordinal, seed or scene changed")
            steps[group] += 1
            effects = menu["effects"]
            expected.update((group, step, effect) for effect in effects if effect != "BASE")
            aliases = {alias: effect for effect, entry in effects.items() for alias in entry["aliases"]}
            seen, units = observations[n, step]
            frozen = set(menu["frozen_v1_menu"])
            for binding in menu["bindings"]:
                actor, family = binding["actor"], binding["family"]
                if family not in ("regroup_step", "retarget"):
                    continue
                new_action = binding["recommended_action"]
                new_effect, new_label = _endpoint(aliases, labels, group, step, family, actor, new_action,
                                                  outcomes[group])
                old_options = old_actions(menu, binding)
                require(len(old_options) == len(set(old_options)), "Duplicate frozen old action for actor/step")
                for old_action in old_options or [None]:
                    old_effect, old_label = _endpoint(aliases, labels, group, step, family, actor, old_action,
                                                      outcomes[group])
                    rows.append({
                        "group_id": group, "seed": seed, "step": step, "anchor_sha256": menu["anchor_sha256"],
                        "actor": actor, "family": family, "base_action": binding["base_action"],
                        "old_action": old_action, "new_action": new_action, "old_effect": old_effect,
                        "new_effect": new_effect, "same_effect": old_effect == new_effect,
                        "in_frozen_menu": f"{family}:{actor}:{old_action}" in frozen,
                        "base": outcomes[group], "old": old_label, "new": new_label,
                        "matched": old_label is not None and new_label is not None,
                        "features": _features(binding, step, old_action, seen, units[actor]),
                    })
    require(expected == set(labels) and hasher.hexdigest() == summary["menus_sha256"], "Menu/ledger mismatch")
    require(all(steps[g] == episodes[g]["steps"] for g in groups), "Incomplete BASE menus")
    return rows, {"split": split, "groups": sorted(groups), "states": states, "labels_accepted": states.get("accepted", 0),
                  "steps": dict(steps), "base_by_group": outcomes, "menus_sha256": summary["menus_sha256"],
                  "split_sha256": checksum(split_path), "protocol_sha256": checksum(coll / "protocol.json")}
