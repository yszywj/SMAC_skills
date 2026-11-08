"""Run a frozen phase2 policy through the native SMAC runner in its own process."""
import argparse
import asyncio
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from decision_tree.local import runtime
from decision_tree.scripts.study import DEFAULT_FLAGS
from sc2_skill_adapter.bridge import runner
from sc2_skill_adapter.bridge.scenarios import validate_manifest
from skills.local_v1 import apply_candidate, generate_candidates
from skills.precontact_regroup import POLICY_PATH
from .policy import REGROUP_POLICY, Selector, accepted_attack, normalize_policy


PROJECT = Path(__file__).resolve().parents[2]


def source_hashes():
    """The additional source files used on top of the runner's native source set."""
    files = [Path(__file__).resolve(), Path(__file__).with_name("policy.py"),
             Path(__file__).with_name("__init__.py"), POLICY_PATH.resolve(),
             POLICY_PATH.with_name("__init__.py").resolve(),
             PROJECT / "decision_tree/scripts/study.py"]
    return {p.relative_to(PROJECT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def carrier_policy(policy):
    """Native engine schema; actual decisions and phase2 policy are separately bound."""
    if policy["mode"] == "regroup":
        return deepcopy(REGROUP_POLICY)
    selection = {"mode": policy["mode"], "max_interventions": 1, "min_gain": 0.0}
    if policy["mode"] == "forced":
        selection.update(anchor_step=policy["anchor_step"], candidate_id=policy["candidate_id"])
    return runtime.load_policy({"schema_version": 1, "kind": "local_patch_policy",
                                "selection": selection, "candidates": deepcopy(REGROUP_POLICY["candidates"])})


class Phase2Controller(runtime.LocalController):
    def __init__(self, manager, carrier, phase2_policy, *, posture, context_hash):
        self.phase2_policy = normalize_policy(phase2_policy)
        if carrier != carrier_policy(self.phase2_policy):
            raise ValueError("Native carrier differs from phase2 policy")
        if context_hash != REGROUP_POLICY["model"]["context_hash"]:
            raise ValueError("Execution differs from the frozen Protoss adaptive configuration")
        super().__init__(manager, carrier, posture=posture, context_hash=context_hash)

    def reset(self):
        super().reset()
        self.selector = Selector(self.phase2_policy)
        self.team_fired = False

    def _decide(self, payload, baseline):
        candidates = generate_candidates(payload, baseline, self.policy["candidates"])
        state = payload["state"]
        self.team_fired = self.team_fired or accepted_attack(state["last_action_result"])
        decision = self.selector.decide(candidates, self.step_number, self.team_fired)
        action, selected_id = decision["action"], decision["selected_id"]
        applied = (next(c for c in candidates if c["id"] == selected_id)
                   if selected_id != "BASE" else None)
        result = apply_candidate(payload, baseline, applied) if applied is not None else baseline
        self.proposals_used = int(decision["consumed"])
        self.interventions += int(applied is not None)
        self.events.append({
            "event": "local_decision", "step": self.step_number, "game_loop": state["game_loop"],
            "baseline_actions": list(baseline["actions"]), "baseline_roles": deepcopy(baseline.get("roles", {})),
            "baseline_predicted_damage": deepcopy(baseline.get("predicted_damage") or {}),
            "candidates": candidates, "selected_id": selected_id,
            "proposed_id": decision["considered_id"] if action in ("execute", "decline") else "BASE",
            "vetoed": action == "decline", "reason": "phase2_"+action,
            "predictions": {}, "final_actions": list(result["actions"]),
            "proposals_used": self.proposals_used, "interventions": self.interventions, "phase2": decision,
        })
        return result


@contextmanager
def _native_extensions(policy, metadata):
    # The frozen runner imports its local controller inside evaluate(). Adapt
    # that interface only for this dedicated process. Every native step, log,
    # command acknowledgement and lifecycle operation still uses the same runner.
    original_class, original_write = runtime.LocalController, runner.write_json

    def controller(manager, local_policy, *, posture, context_hash):
        return Phase2Controller(manager, local_policy, policy, posture=posture, context_hash=context_hash)

    def write(path, value):
        if Path(path).name in ("run.json", "summary.json"):
            value = {**value, "phase2": metadata}
        original_write(path, value)

    try:
        runtime.LocalController, runner.write_json = controller, write
        yield
    finally:
        runtime.LocalController, runner.write_json = original_class, original_write


def _check_completion(output, policy, metadata, scenes):
    """Validate actual SC2 version, phase2 binding, forced arrival and completion."""
    summary = json.loads((output/"summary.json").read_text(encoding="utf-8"))
    if summary["status"] != "complete" or summary["completed_episodes"] != scenes or summary["phase2"] != metadata:
        raise ValueError("Incomplete or unbound phase2 rollout")
    started, forced = [], []
    with (output/"events.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            event = json.loads(line)
            if event["event"] == "episode_started":
                if event["game_version"] != REGROUP_POLICY["model"]["game_version"]:
                    raise ValueError("Actual SC2 build differs from the frozen base")
                started.append(event["episode"])
            elif policy["mode"] == "forced" and event["event"] == "local_decision" and event["selected_id"] != "BASE":
                if event["step"] != policy["anchor_step"] or event["selected_id"] != policy["candidate_id"]:
                    raise ValueError("Wrong forced anchor or binding")
                forced.append(event["episode"])
    expected = list(range(1, scenes+1))
    if started != expected or (policy["mode"] == "forced" and forced != expected):
        raise ValueError("Missing SC2 episode or unreachable forced anchor")
    if metadata["source_hashes"] != source_hashes():
        raise ValueError("Phase2 source changed during rollout")


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--policy", type=Path, required=True)
    cli.add_argument("--manifest", type=Path, required=True)
    cli.add_argument("--output-dir", type=Path, required=True)
    cli.add_argument("--sc2-path", type=Path)
    args = cli.parse_args(argv)
    policy = normalize_policy(json.loads(args.policy.read_text(encoding="utf-8-sig")))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8-sig"))
    validate_manifest(manifest)
    if manifest["env_args"]["map_name"] != "10gen_protoss" or manifest["env_args"].get("unit_balance") != "native":
        raise ValueError("Phase2 requires native Protoss 10v10")
    carrier = carrier_policy(policy)
    metadata = {"schema_version": 1, "runtime": "decision_tree.phase2", "policy": policy,
                "policy_sha256": runtime.digest(policy), "source_hashes": source_hashes(),
                "carrier_policy_sha256": runtime.digest(carrier)}
    with TemporaryDirectory(prefix="phase2-native-") as temp:
        transport = Path(temp)/"local_policy.json"
        transport.write_text(json.dumps(carrier), encoding="utf-8")
        parsed = ["run", *DEFAULT_FLAGS, "--manifest", str(args.manifest), "--output-dir", str(args.output_dir),
                  "--local-policy", str(transport)]
        if args.sc2_path:
            parsed += ["--sc2-path", str(args.sc2_path)]
        native = runner.parser().parse_args(parsed)
        with _native_extensions(policy, metadata):
            asyncio.run(runner.evaluate(native, manifest, native.method, args.output_dir.resolve()))
    _check_completion(args.output_dir, policy, metadata, len(manifest["episodes"]))


if __name__ == "__main__":
    main()
