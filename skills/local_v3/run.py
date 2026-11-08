"""Run repeated condition-triggered skills through the native adaptive entry point."""
import argparse
import asyncio
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from decision_tree.conditional.policy import normalize_policy
from decision_tree.conditional.runtime import CARRIER, ConditionalController, normalize_forced
from decision_tree.local import runtime
from decision_tree.learning.dataset import episode as recorded_episode, index, require
from decision_tree.phase2.policy import REGROUP_POLICY
from decision_tree.scripts.study import DEFAULT_FLAGS
from sc2_skill_adapter.bridge import runner
from sc2_skill_adapter.bridge.scenarios import validate_manifest
from skills.local_v2.run import source_hashes as v2_hashes


ROOT = Path(__file__).resolve().parents[2]


def source_hashes():
    hashes = v2_hashes()
    for folder in ("skills/local_v3", "decision_tree/conditional"):
        for path in sorted((ROOT/folder).glob("*.py")):
            if not path.name.startswith("test_"):
                hashes[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


@contextmanager
def installed(policy, metadata):
    previous_class, previous_write = runtime.LocalController, runner.write_json

    def factory(manager, carrier, *, posture, context_hash):
        if carrier != runtime.load_policy(CARRIER):
            raise ValueError("Wrong adaptive carrier")
        return ConditionalController(manager, policy, posture=posture, context_hash=context_hash,
                                     forced=metadata.get("forced"))

    def write(path, value):
        if Path(path).name in ("run.json", "summary.json"):
            value = {**value, "conditional_skills": metadata}
        previous_write(path, value)

    try:
        runtime.LocalController, runner.write_json = factory, write
        yield
    finally:
        runtime.LocalController, runner.write_json = previous_class, previous_write


def _audit_actions(observation, decision, submitted):
    state = observation["state"]
    actions = decision["final_actions"]
    actor_ids = {f"u{i}" for i in range(len(actions))}
    require(set(state["allies"]) == actor_ids, "Incomplete action actors")
    require(actions == submitted["actions"], "Final commands differ from submitted actions")
    require(state["game_loop"] == decision["game_loop"] == submitted["game_loop"], "Decision clock mismatch")
    feedback = submitted["feedback"]
    require(feedback.get("response_mapping") == "exact" and not feedback.get("unmatched_observation_errors"),
            "Unmapped native command response")
    require(feedback.get("from_game_loop") == state["game_loop"] and
            feedback.get("to_game_loop") == state["game_loop"] + state["step_mul"], "Receipt clock mismatch")
    receipts = feedback.get("actors", {})
    require(set(receipts) == actor_ids, "Missing native actor receipt")
    command_index = 0
    for slot, action in enumerate(actions):
        actor = f"u{slot}"
        receipt = receipts[actor]
        require(type(action) is int and receipt.get("requested_action_id") == str(action)
                and not receipt.get("observation_errors"), "Native action receipt mismatch")
        if state["allies"][actor]["alive"]:
            require(str(action) in observation["questions"][actor]["criteria"], "Illegal submitted action")
            require(receipt.get("submission") == "accepted" and isinstance(receipt.get("command"), dict)
                    and isinstance(receipt.get("response"), dict) and receipt["response"].get("code") == 1
                    and type(receipt.get("command_index")) is int and receipt["command_index"] == command_index,
                    "Selected action lacks accepted native receipt")
            command_index += 1
        else:
            require(action == 0 and receipt.get("submission") == "dead_noop" and receipt.get("command") is None
                    and receipt.get("response") is None and "command_index" not in receipt, "Invalid dead-unit receipt")
    require(feedback.get("raw_response_results") == [1] * command_index, "Invalid raw native responses")


def _audit_episode(run, number, policy, forced=None):
    arm = recorded_episode(run, number)
    require(arm["start"]["game_version"] == REGROUP_POLICY["model"]["game_version"],
            "SC2 version differs from frozen execution context")
    decisions = index(arm["events"], "conditional_local_decision", "step")
    require(set(decisions) == set(arm["observations"]), "Missing conditional decision")
    # Replay the pure precommit decision using the logged adaptive actions. This
    # rechecks every candidate, condition, cooldown and counter at each step.
    manager = SimpleNamespace(race_profile="off", _spatial_features_enabled=lambda: False)
    replay = ConditionalController(manager, policy, context_hash=REGROUP_POLICY["model"]["context_hash"], forced=forced)
    previous_loop = arm["start"]["game_loop"]
    for step in range(arm["record"]["steps"]):
        decision, submitted = decisions[step], arm["steps"][step + 1]
        observation = arm["observations"][step]["payload"]
        state = observation["state"]
        require(state["game_loop"] == previous_loop and state["step_mul"] == run["manifest.json"]["env_args"]["step_mul"],
                "Non-contiguous native clock")
        require(len(state["allies"]) == run["manifest.json"]["env_args"]["capability_config"]["n_units"],
                "Wrong number of actors")
        require(type(submitted["terminated"]) is bool and submitted["terminated"] == (step == arm["record"]["steps"] - 1)
                and type(submitted["reward"]) in (int, float) and math.isfinite(submitted["reward"]),
                "Invalid terminal step or reward")
        baseline = {"actions": decision["baseline_actions"], "roles": decision["baseline_roles"],
                    "predicted_damage": decision["baseline_predicted_damage"]}
        replay.step_number = step
        replay.events = []
        replay._decide(observation, baseline)
        recomputed = replay.events[0]
        require(all(decision.get(key) == value for key, value in recomputed.items()),
                "Conditional policy, actions, candidates or counters differ from replay")
        _audit_actions(observation, decision, submitted)
        previous_loop += state["step_mul"]
    record = arm["record"]
    actual_forced = [(step, event["proposed_id"]) for step, event in decisions.items() if event.get("forced")]
    expected_forced = [] if forced is None else [(forced["step"], forced["binding"])]
    require(actual_forced == expected_forced, "Forced override did not occur exactly once at its declared step")
    require(record["outcome"] in ("win", "loss", "draw", "timeout"), "Invalid episode outcome")
    require((record["outcome"] == "win") == bool(record["info"].get("battle_won")), "Win outcome contradicts native info")


def check_completion(output, metadata, expected):
    output = Path(output)
    run = {name: json.loads((output/name).read_text(encoding="utf-8"))
           for name in ("run.json", "summary.json", "manifest.json", "episodes.json")}
    native, summary, manifest, records = (run[name] for name in ("run.json", "summary.json", "manifest.json", "episodes.json"))
    validate_manifest(manifest)
    policy = normalize_policy(metadata["policy"])
    forced = normalize_forced(metadata.get("forced"))
    require(forced is None or expected == 1, "Forced overrides require a single-scene run")
    require(policy == metadata["policy"] and runtime.digest(policy) == metadata["policy_sha256"], "Changed conditional policy")
    require(summary.get("status") == "complete" and not summary.get("error")
            and summary["completed_episodes"] == summary["expected_episodes"] == expected == len(records) == len(manifest["episodes"])
            and expected > 0, "Incomplete conditional-skill run")
    require(native.get("conditional_skills") == summary.get("conditional_skills") == metadata
            and source_hashes() == metadata["source_hashes"], "Changed conditional-skill sources")
    require(native["manifest_sha256"] == summary["manifest_sha256"] == manifest["sha256"]
            and native["env_args"] == manifest["env_args"], "Changed scene or environment binding")
    require(native["local_policy"]["policy"] == runtime.load_policy(CARRIER)
            and native["local_policy"]["context_hash"] == REGROUP_POLICY["model"]["context_hash"]
            == runtime.execution_context(native, runtime.load_policy(CARRIER)), "Changed adaptive execution context")
    require(native.get("method") == "adaptive-skill" and native.get("skill_version") == "current"
            and native.get("native_teacher_control") == "off" and native.get("medivac_policy") == "legacy"
            and summary.get("pair_mode") == "off" and not native.get("skill_tree"), "Other control channel active")
    require([record["episode"] for record in records] == list(range(1, expected + 1)), "Invalid episode records")
    wanted_outcomes = {name: sum(record["outcome"] == name for record in records) for name in ("win", "loss", "draw", "timeout")}
    require(summary["outcomes"] == wanted_outcomes and summary["win_rate"] == wanted_outcomes["win"] / expected,
            "Summary outcomes differ from native episode records")
    relevant = {"episode_started", "skill_observation", "conditional_local_decision", "step", "episode_completed"}
    next_name, number, next_step, retained = "episode_started", 1, 0, []
    with (output/"events.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            event = json.loads(line)
            require(event["event"] != "run_failed", "Failed native run")
            if event["event"] not in relevant:
                continue
            require(event.get("episode") == number and event["event"] == next_name, "Incomplete or unordered native events")
            if event["event"] in ("skill_observation", "conditional_local_decision", "step"):
                wanted_step = next_step + int(event["event"] == "step")
                require(type(event["step"]) is int and event["step"] == wanted_step, "Unordered native step number")
                if event["event"] == "step":
                    next_step += 1
            retained.append(event)
            next_name = {"episode_started": "skill_observation", "skill_observation": "conditional_local_decision",
                         "conditional_local_decision": "step", "step": "skill_observation", "episode_completed": "episode_started"}[event["event"]]
            if event["event"] == "step" and event["terminated"]:
                next_name = "episode_completed"
            if event["event"] == "episode_completed":
                _audit_episode({**run, "events.jsonl": retained}, number, policy, forced)
                retained = []
                number += 1
                next_step = 0
    require(not retained and number == expected + 1 and next_name == "episode_started", "Missing terminal native log")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--policy", type=Path, default=Path(__file__).with_name("policy.json"))
    parser.add_argument("--sc2-path", type=Path)
    parser.add_argument("--force-step", type=int)
    parser.add_argument("--force-binding", help="BASE or one family:uN:action override, followed by the same policy")
    args = parser.parse_args()
    if (args.force_step is None) != (args.force_binding is None):
        parser.error("Supply both --force-step and --force-binding")
    forced = normalize_forced({"step": args.force_step, "binding": args.force_binding}
                              if args.force_step is not None else None)
    policy = normalize_policy(json.loads(args.policy.read_text(encoding="utf-8-sig")))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8-sig"))
    validate_manifest(manifest)
    env = manifest["env_args"]
    if env["map_name"] != "10gen_protoss" or env.get("unit_balance") != "native":
        raise ValueError("Conditional v3 requires native Protoss 10v10")
    if forced is not None and len(manifest["episodes"]) != 1:
        raise ValueError("A forced counterfactual requires one scene")
    metadata = {"kind": "conditional_local_v3", "scope": "repeated_one_actor_one_step",
                "policy": policy, "source_hashes": source_hashes(), "forced": forced,
                "policy_sha256": runtime.digest(policy),
                "research_status": "offline_verified_pending_native_whole_policy_validation"}
    with TemporaryDirectory(prefix="smac-conditional-v3-") as temp:
        carrier = Path(temp)/"carrier.json"
        carrier.write_text(json.dumps(CARRIER), encoding="utf-8")
        argv = ["run", *DEFAULT_FLAGS, "--manifest", str(args.manifest),
                "--local-policy", str(carrier), "--output-dir", str(args.output_dir)]
        if args.sc2_path:
            argv.extend(["--sc2-path", str(args.sc2_path)])
        native = runner.parser().parse_args(argv)
        with installed(policy, metadata):
            asyncio.run(runner.evaluate(native, manifest, native.method, args.output_dir.resolve()))
    check_completion(args.output_dir, metadata, len(manifest["episodes"]))


if __name__ == "__main__":
    main()
