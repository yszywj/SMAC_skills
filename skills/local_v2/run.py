"""Run v1/v2 behind the same frozen selector, or force one declared binding."""
import argparse
import asyncio
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from decision_tree.local import runtime
from decision_tree.local.runtime import LocalController, digest, load_policy
from decision_tree.phase2.policy import normalize_policy
from decision_tree.phase2.run import Phase2Controller, carrier_policy, source_hashes as phase2_hashes
from decision_tree.scripts.study import DEFAULT_FLAGS
from sc2_skill_adapter.bridge import runner
from sc2_skill_adapter.bridge.scenarios import validate_manifest
from skills.local_v1 import apply_candidate
from skills.precontact_regroup import POLICY_PATH
from .library import KERNELS, analyze_binding, resolve_action


PROJECT = Path(__file__).resolve().parents[2]


def source_hashes():
    hashes = phase2_hashes()
    for name in ("__init__.py","run.py","library.py","regroup.py","retreat.py","retarget.py"):
        p = Path(__file__).with_name(name)
        hashes[p.resolve().relative_to(PROJECT).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
    return hashes


def selector_spec(raw):
    if raw.get("kind") == "local_patch_policy":
        policy = load_policy(raw)
        if policy["selection"]["mode"] not in ("model", "base"):
            raise ValueError("Use --force-step/--force-binding for a fixed action")
        return "local", policy, policy
    policy = normalize_policy(raw)
    if policy["mode"] not in ("regroup", "base"):
        raise ValueError("Expected a phase2 deployment policy")
    return "phase2", policy, carrier_policy(policy)


def parse_binding(value):
    try:
        family, actor, action = value.split(":")
        action = int(action)
        if family not in KERNELS or not actor.startswith("u") or not actor[1:].isdigit():
            raise ValueError
        return family, actor, action
    except (AttributeError, ValueError):
        raise ValueError("Binding must be family:uN:action") from None


def attach(manager, selector, *, implementation, context_hash, posture="balanced", forced=None):
    """Install one precommit owner. The old selector sees exactly its old menu."""
    if implementation not in ("v1", "v2"):
        raise ValueError("Unknown action implementation")
    kind, policy, carrier = selector_spec(selector)
    if forced is not None:
        if policy.get("mode") != "base" or implementation != "v2" or type(forced[0]) is not int or forced[0] < 0:
            raise ValueError("A forced v2 intervention requires a BASE carrier and a nonnegative step")
        parse_binding(forced[1])
    if kind == "phase2":
        owner = Phase2Controller(manager, carrier, policy, posture=posture, context_hash=context_hash)
    else:
        owner = LocalController(manager, carrier, posture=posture, context_hash=context_hash)
    original_decision = manager.local_decision_hook

    def decide(payload, baseline):
        original_result = original_decision(payload, baseline)
        event = owner.events[-1]
        event["event"] = "local_v2_decision"
        frozen_id = event["selected_id"]
        info = {"implementation":implementation, "frozen_proposed_id":event["proposed_id"],
                "frozen_selected_id":frozen_id, "frozen_vetoed":event["vetoed"], "forced":False}
        event["local_v2"] = info
        use_force = forced is not None and event["step"] == forced[0]
        if not use_force and (frozen_id == "BASE" or implementation == "v1"):
            return original_result

        family, actor, original_action = parse_binding(forced[1] if use_force else frozen_id)
        row = analyze_binding(payload, baseline, family, actor)
        if row is None:
            raise ValueError("The declared local binding is not available")
        # Retreat-to-attack is an unchanged v1 action, including its target.
        action = original_action if use_force or row.get("uses_frozen_target") else row["recommended_action"]
        patch = resolve_action(payload, baseline, family, actor, action)
        if use_force:
            # The carrier is a phase2 BASE selector. Its next decision reports
            # the consumed opportunity even when the forced action kept BASE.
            owner.selector.consumed = True
            owner.proposals_used = 1
            owner.interventions += int(patch is not None)
        elif patch is None:
            owner.interventions -= 1
        result = baseline if patch is None else apply_candidate(payload, baseline, patch)
        info.update(family=family, actor=actor, forced=use_force, requested_action=action, comparison=row)
        # The original predictions apply to the frozen action only.
        event["selected_id"] = patch["id"] if patch else "BASE"
        event["proposed_id"] = forced[1] if use_force else event["proposed_id"]
        event["final_actions"] = list(result["actions"])
        event["proposals_used"] = owner.proposals_used
        event["interventions"] = owner.interventions
        event["reason"] = "forced_v2" if use_force else "v2_action" if patch else "v2_keep_baseline"
        return result

    manager.local_decision_hook = decide
    return owner


@contextmanager
def installed(selector, implementation, forced, metadata):
    # Scope the factory override to the dedicated native-runner process.
    original, write_json = runtime.LocalController, runner.write_json

    def factory(manager, policy, *, posture, context_hash):
        if policy != selector_spec(selector)[2]:
            raise ValueError("Unexpected native carrier policy")
        return attach(manager, selector, implementation=implementation, posture=posture,
                      context_hash=context_hash, forced=forced)

    def write(path, value):
        if Path(path).name in ("run.json", "summary.json"):
            value = {**value, "local_v2": metadata}
        write_json(path, value)

    try:
        runtime.LocalController, runner.write_json = factory, write
        yield
    finally:
        runtime.LocalController, runner.write_json = original, write_json


def check_completion(output, metadata, forced, scenes):
    summary = json.loads((output/"summary.json").read_text(encoding="utf-8"))
    if summary["status"] != "complete" or summary["completed_episodes"] != scenes or summary["local_v2"] != metadata:
        raise ValueError("Incomplete or unbound v2 run")
    if metadata["source_hashes"] != source_hashes():
        raise ValueError("Skill sources changed during rollout")
    expected_version = load_policy(POLICY_PATH)["model"]["game_version"]
    forced_steps, selected, accepted, starts = [], {}, set(), []
    with (output/"events.jsonl").open(encoding="utf-8") as log:
        for line in log:
            e = json.loads(line)
            if e["event"] == "episode_started":
                starts.append(e["episode"])
                if e["game_version"] != expected_version:
                    raise ValueError("SC2 version differs from frozen execution context")
            if e["event"] == "local_v2_decision":
                changes = [i for i,(a,b) in enumerate(zip(e["baseline_actions"],e["final_actions"])) if a!=b]
                if e["local_v2"]["forced"]:
                    forced_steps.append((e["step"],e["proposed_id"]))
                if e["selected_id"] == "BASE":
                    if changes:
                        raise ValueError("BASE changed a submitted action")
                else:
                    _, actor, action = parse_binding(e["selected_id"])
                    if changes != [int(actor[1:])] or e["final_actions"][changes[0]]!=action or e["episode"] in selected:
                        raise ValueError("Expected exactly one local change")
                    selected[e["episode"]] = (e["step"], actor, action)
            if e["event"] == "step" and e["episode"] in selected:
                step, actor, action = selected[e["episode"]]
                if e["step"] == step+1:
                    receipt = e["feedback"]["actors"][actor]
                    if e["actions"][int(actor[1:])] != action or receipt["submission"] != "accepted" or int(receipt["requested_action_id"]) != action:
                        raise ValueError("Selected action lacks its native receipt")
                    accepted.add(e["episode"])
    if starts != list(range(1,scenes+1)) or accepted != set(selected):
        raise ValueError("Missing episode or native submission for a selected action")
    if forced is not None and forced_steps != [forced]:
        raise ValueError("Forced binding did not reach its exact step")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--selector", type=Path, help="frozen phase2 or local model policy; default: independent regroup")
    parser.add_argument("--implementation", choices=("v1","v2"), default="v2")
    parser.add_argument("--force-step", type=int)
    parser.add_argument("--force-binding", help="one registered family:uN:action, on a one-scene manifest")
    parser.add_argument("--sc2-path", type=Path)
    args = parser.parse_args()
    if (args.force_step is None) != (args.force_binding is None):
        parser.error("Supply both force arguments")
    forced = (args.force_step,args.force_binding) if args.force_step is not None else None
    if forced and (args.selector or args.implementation!="v2" or args.force_step<0):
        parser.error("A forced run independently uses BASE and one v2 binding")
    if forced:
        parse_binding(args.force_binding)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8-sig"))
    validate_manifest(manifest)
    env = manifest["env_args"]
    if env["map_name"]!="10gen_protoss" or env.get("unit_balance")!="native":
        raise ValueError("This experiment requires native Protoss 10v10")
    if forced and len(manifest["episodes"])!=1:
        raise ValueError("A forced run needs exactly one scene")
    raw = json.loads(args.selector.read_text(encoding="utf-8-sig")) if args.selector else {"mode":"base" if forced else "regroup"}
    kind, selector, carrier = selector_spec(raw)
    meta = {"selector_kind":kind,"selector":selector,"selector_sha256":digest(selector),
            "implementation":args.implementation,"forced":list(forced) if forced else None,"source_hashes":source_hashes()}
    with TemporaryDirectory(prefix="smac-action-v2-") as temp:
        path = Path(temp)/"carrier.json"
        path.write_text(json.dumps(carrier), encoding="utf-8")
        argv = ["run",*DEFAULT_FLAGS,"--manifest",str(args.manifest),"--local-policy",str(path),
                "--output-dir",str(args.output_dir)]
        if args.sc2_path:
            argv.extend(["--sc2-path",str(args.sc2_path)])
        native = runner.parser().parse_args(argv)
        with installed(selector,args.implementation,forced,meta):
            asyncio.run(runner.evaluate(native,manifest,native.method,args.output_dir.resolve()))
    check_completion(args.output_dir,meta,forced,len(manifest["episodes"]))


if __name__ == "__main__":
    main()
