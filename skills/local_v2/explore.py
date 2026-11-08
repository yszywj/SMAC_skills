"""Plan the complete single-actor, single-step local menu independently of tree choices."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from decision_tree.local import dataset, study
from decision_tree.local.runtime import digest, execution_context, load_policy
from .library import catalog, resolve_action
from .run import parse_binding, selector_spec, source_hashes


FILES = ("run.json", "summary.json", "manifest.json", "episodes.json", "events.jsonl")
EVENTS = {"episode_started", "skill_observation", "local_decision", "local_v2_decision",
          "step", "episode_completed"}
SCOPE = "Adaptive BASE trajectories; four registered local families; one actor, one step"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def checksum(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    with Path(path).open("x", encoding="utf-8") as output:
        json.dump(value, output, indent=2, allow_nan=False)
        output.write("\n")


def read_header(path):
    raw = {name: (path/name).read_bytes() for name in FILES[:-1]}
    run = {name: json.loads(value.decode("utf-8-sig")) for name, value in raw.items()}
    run["hashes"] = {name: hashlib.sha256(value).hexdigest() for name, value in raw.items()}
    meta, summary, manifest = (run[n] for n in ("run.json", "summary.json", "manifest.json"))
    require(summary["status"] == "complete" and not summary.get("error"), "Incomplete SC2 run")
    require(meta["method"] == "adaptive-skill" and meta["skill_version"] == "current"
            and meta.get("native_teacher_control") == "off" and meta.get("medivac_policy") == "legacy"
            and not meta.get("skill_tree") and summary.get("pair_mode") == "off", "Unexpected controller")
    scenes = manifest["episodes"]
    require(len(scenes) == len(run["episodes.json"]) == summary["completed_episodes"]
            == summary["expected_episodes"] and bool(scenes), "Incomplete scene list")
    require(len({s["sha256"] for s in scenes}) == len(scenes)
            and all(digest(s["episode_config"]) == s["sha256"] and type(s["sc2_seed"]) is int for s in scenes),
            "Invalid scene identity")
    require(manifest["sha256"] == digest({k:v for k,v in manifest.items() if k != "sha256"})
            == meta["manifest_sha256"] == summary["manifest_sha256"]
            and manifest["env_args"] == meta["env_args"], "Manifest binding mismatch")
    policy = load_policy(meta["local_policy"]["policy"])
    require(meta["local_policy"]["context_hash"] == execution_context(meta, policy), "Context mismatch")
    require(meta["source_hashes"] and meta["versions"], "Missing native source bindings")
    hashes = {str((study.ROOT/k).resolve()): v for k,v in meta["source_hashes"].items()}
    require(hashes == study.source_hashes(manifest), "Frozen native source set differs")
    v2 = meta.get("local_v2")
    if v2 is not None:
        kind, selected, carrier = selector_spec(v2["selector"])
        require(v2 == summary.get("local_v2") and v2["source_hashes"] == source_hashes()
                and v2["selector_kind"] == kind and v2["selector"] == selected
                and carrier == policy and v2["selector_sha256"] == digest(selected), "Changed v2 binding")
    return run


def read_episodes(path, run):
    """Validate raw native receipts and release each large episode before reading the next."""
    retained = []
    number = 1
    h = hashlib.sha256()
    v2 = run["run.json"].get("local_v2") is not None
    with (path/"events.jsonl").open("rb") as stream:
        for line in stream:
            h.update(line)
            if not line.strip():
                continue
            event = json.loads(line)
            name = event["event"]
            require(name != "run_failed", "Failed native event")
            if name not in EVENTS:
                continue
            require(event["episode"] == number, "Unexpected or non-contiguous episode ordinal")
            if name in ("local_decision", "local_v2_decision"):
                require((name == "local_v2_decision") == v2, "Decision event/implementation mismatch")
                # The frozen auditor verifies the original menu and the submitted actions.
                event = {**event, "event": "local_decision"}
            retained.append(event)
            if name == "episode_completed":
                require(retained[0]["event"] == "episode_started", "Missing episode start")
                episode = dataset.episode({**run, "events.jsonl": retained}, number)
                decisions = dataset.audit_episode(episode)
                yield number, episode, decisions
                number += 1
                retained = []
    require(not retained and number == len(run["episodes.json"])+1, "Incomplete episode stream")
    run["hashes"]["events.jsonl"] = h.hexdigest()


def anchors(arm, decisions):
    """Bind each precommit observation/baseline and the complete previous observable trajectory."""
    prefix = digest({"initial":arm["start"]["initial_state_sha256"], "game":arm["start"]["game_version"]})
    for step in range(arm["record"]["steps"]):
        decision = decisions[step]
        obs = arm["observations"][step]["payload"]
        baseline = {"actions":decision["baseline_actions"], "roles":decision["baseline_roles"],
                    "predicted_damage":decision["baseline_predicted_damage"]}
        anchor = digest([prefix, dataset.comparable_observation(obs), baseline, decision["candidates"]])
        yield step, decision, obs, baseline, anchor
        submitted = arm["steps"][step+1]
        prefix = digest([anchor, {k:submitted[k] for k in ("actions","game_loop","feedback","reward","terminated")}])


def plan(base, output):
    base, output = Path(base).resolve(), Path(output)
    run = read_header(base)
    meta = run["run.json"]
    require(meta["local_policy"]["policy"]["selection"]["mode"] == "base", "Exploration requires adaptive BASE")
    v2 = meta.get("local_v2")
    require(v2 is None or v2["forced"] is None,
            "Policy-modified trajectories cannot define the BASE denominator")
    output.mkdir(parents=True, exist_ok=False)
    counts, families = Counter(), Counter()
    menus = output/"menus.jsonl"
    with menus.open("x", encoding="utf-8") as stream:
        for number, arm, decisions in read_episodes(base, run):
            scene = run["manifest.json"]["episodes"][number-1]
            for step, event, obs, baseline, anchor in anchors(arm, decisions):
                require(event["selected_id"] == event["proposed_id"] == "BASE"
                        and event["final_actions"] == baseline["actions"], "BASE modified its action")
                menu = catalog(obs, baseline, event["candidates"])
                effects = len(menu["effects"]) - 1
                counts["decision_steps"] += 1
                counts["frozen_v1_candidate_bindings"] += len(event["candidates"])
                counts["steps_with_frozen_v1_candidates"] += int(bool(event["candidates"]))
                counts["steps_with_nonbase_effects"] += int(effects > 0)
                counts["nonbase_effects"] += effects
                counts["aliases_equivalent_to_base"] += len(menu["effects"]["BASE"]["aliases"])
                for row in menu["bindings"]:
                    families[row["family"]] += len(row["options"])
                record = {"episode":number,"scene_hash":scene["sha256"],"seed":scene["sc2_seed"],
                          "step":step,"game_loop":event["game_loop"],"anchor_sha256":anchor, **menu}
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(",", ":"))+"\n")
    summary = {"kind":"local_v2_exploration_plan","scope":SCOPE,"base":str(base),
               "base_hashes":run["hashes"],"context":meta["local_policy"]["context_hash"],
               "source_hashes":source_hashes(),"menus_sha256":checksum(menus),
               "scenes":len(run["episodes.json"]),"counts":dict(counts),"options_by_family":dict(families)}
    save(output/"summary.json", summary)
    return summary


def audit_branch(path, summary, entries):
    run = read_header(path)
    meta = run["run.json"]
    require(meta["local_policy"]["context_hash"] == summary["context"]
            and len(run["episodes.json"]) == 1, "Wrong environment or forced cohort")
    v2 = meta.get("local_v2")
    selection = meta["local_policy"]["policy"]["selection"]
    if v2 is not None:
        require(v2["implementation"] == "v2" and v2["forced"] and selection["mode"] == "base"
                and v2["selector"] == {"mode":"base"}, "Expected explicit forced v2 BASE branch")
        step, binding = v2["forced"]
    else:
        require(selection["mode"] == "forced", "Expected an explicitly forced v1 branch")
        step, binding = selection["anchor_step"], selection["candidate_id"]
    require(type(step) is int and step >= 0, "Invalid forced anchor")
    family, actor, action = parse_binding(binding)
    scene = run["manifest.json"]["episodes"][0]
    entry = entries.get((scene["sha256"], step))
    require(entry is not None and entry["seed"] == scene["sc2_seed"], "Unregistered scene, step or seed")
    key = next((k for k,v in entry["effects"].items() if binding in v["aliases"]), None)
    require(key is not None, "Unregistered action binding")
    forced_seen = False
    for _, arm, decisions in read_episodes(path, run):
        for t, e, obs, baseline, anchor in anchors(arm, decisions):
            if v2 is not None:
                require(e["local_v2"]["forced"] == (t == step), "Forced proposal at wrong time")
            if t != step:
                require(e["selected_id"] == "BASE" and e["final_actions"] == baseline["actions"],
                        "Unexpected second intervention")
                continue
            forced_seen = True
            require(anchor == entry["anchor_sha256"], "Observable prefix differs from BASE")
            patch = resolve_action(obs, baseline, family, actor, action)
            require(key == ("BASE" if patch is None else patch["effect_key"]), "Effect binding differs")
            expected = list(baseline["actions"])
            expected[int(actor[1:])] = action
            require(e["final_actions"] == expected and e["proposed_id"] == binding
                    and e["selected_id"] == (binding if patch is not None else "BASE")
                    and not e["vetoed"], "Submitted effect differs from forced proposal")
            if v2 is not None:
                info = e["local_v2"]
                require((info["family"],info["actor"],info["requested_action"]) == (family,actor,action),
                        "Forced v2 command metadata differs")
    require(forced_seen, "Forced anchor missing")
    return {"scene_hash":scene["sha256"], "seed":scene["sc2_seed"], "step":step, "effect":key,
            "binding":binding, "return":run["episodes.json"][0]["reward"],
            "outcome":run["episodes.json"][0]["outcome"], "run_hashes":run["hashes"]}


def audit(directory, paths, output):
    directory = Path(directory)
    summary = study.read(directory/"summary.json")
    require(summary["kind"] == "local_v2_exploration_plan" and summary["scope"] == SCOPE
            and summary["source_hashes"] == source_hashes(), "Plan source binding changed")
    base = Path(summary["base"])
    require(all(checksum(base/k) == summary["base_hashes"][k] for k in FILES), "BASE source changed")
    require(checksum(directory/"menus.jsonl") == summary["menus_sha256"], "Exploration menu changed")
    entries = {}
    with (directory/"menus.jsonl").open(encoding="utf-8") as menus:
        for line in menus:
            row = json.loads(line)
            key = row["scene_hash"], row["step"]
            require(key not in entries, "Duplicate exploration anchor")
            entries[key] = {"seed":row["seed"], "anchor_sha256":row["anchor_sha256"], "effects":row["effects"]}
    expected = {(scene,step,k) for (scene,step),e in entries.items() for k in e["effects"] if k != "BASE"}
    require(len(entries) == summary["counts"]["decision_steps"] and
            len(expected) == summary["counts"]["nonbase_effects"], "Plan denominator mismatch")
    covered, accepted, failed = {}, [], []
    for path in paths:
        try:
            branch = audit_branch(Path(path), summary, entries)
            key = branch["scene_hash"], branch["step"], branch["effect"]
            if key in covered:
                require(abs(covered[key]["return"]-branch["return"]) < 1e-8
                        and covered[key]["outcome"] == branch["outcome"], "Conflicting duplicate reruns")
            covered[key] = branch
            accepted.append({"path":str(Path(path).resolve()), **branch})
        except (ValueError, OSError, KeyError, TypeError, IndexError) as exc:
            failed.append({"path":str(path),"reason":str(exc)})
    missing = expected-set(covered)
    result = {"kind":"local_v2_coverage_audit","scope":SCOPE,"menus_sha256":summary["menus_sha256"],
              "registered_nonbase":len(expected),"measured_nonbase":len(expected)-len(missing),
              "unmeasured_nonbase":len(missing),"complete":not missing and not failed,
              "missing":[list(x) for x in sorted(missing)],"accepted_attempts":accepted,"failed_or_rejected":failed}
    save(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("plan")
    create.add_argument("--base", type=Path, required=True)
    create.add_argument("--output", type=Path, required=True)
    check = sub.add_parser("audit")
    check.add_argument("--plan", type=Path, required=True)
    check.add_argument("--run", type=Path, action="append", default=[])
    check.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = plan(args.base, args.output) if args.command == "plan" else audit(args.plan, args.run, args.output)
    print(json.dumps({k:v for k,v in result.items() if k not in ("missing","accepted_attempts")}, indent=2))


if __name__ == "__main__":
    main()
