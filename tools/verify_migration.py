"""Verify portable skill migration using saved observations, never a new game.

Run with the adapter's Python environment. Each replay starts a fresh manager
per experiment and preserves the recorded episode/selection lifecycle. This is
deterministic decision replay, not a simulation of the resulting game state.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import inspect
import json
from pathlib import Path
import subprocess
import sys
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
ADAPTER = ROOT / "smacv2_jev_adapter"
ORIGINAL = ROOT / "game_skill/micro/jev_micro"
OUTPUT = ROOT / "skills/verification"
CASES = (
    ("current", "native-5v5-v6-skill-n20-seed150", 5, 20, 1193),
    ("current", "native-10v10-v6-skill-n20-seed150", 10, 20, 1474),
    ("baseline_v2_70pct", "native-10v10-v2-jev-astra-skill-n10-seed100/adaptive-skill", 10, 10, 749),
    ("baseline_v2_70pct", "native-10v10-v2-skill-n20-seed110", 10, 20, 1375),
)
TELEMETRY_KEYS = ("roles", "attack_assignments", "predicted_damage", "all_in", "retreat_reasons")
CURRENT_MODULES = (
    "client", "skill_manager", "skill_policy", "unit_value", "skills",
    "skills.adaptive_focus_fire", "skills.combat_timing", "skills.coordinated_approach",
    "skills.cover_retreat", "skills.damage_model", "skills.engagement_control",
    "skills.focus_fire", "skills.heal_assign", "skills.movement_scoring",
)
BASELINE_MODULES = tuple(name for name in CURRENT_MODULES if name not in (
    "unit_value", "skills.coordinated_approach", "skills.damage_model", "skills.engagement_control"))


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, details):
    if not condition:
        raise AssertionError(details)


def runtime():
    if str(ADAPTER) not in sys.path:
        sys.path.insert(0, str(ADAPTER))
    return importlib.import_module("bridge.skill_runtime")


def module_path_check(namespace, version, module_names):
    source = (ROOT / "skills" / version).resolve()
    modules = {name: importlib.import_module(namespace + "." + name) for name in module_names}
    paths = {}
    for name, module in modules.items():
        path = Path(module.__file__).resolve()
        require(path.is_relative_to(source), ("wrong module location", name, str(path), str(source)))
        paths[name] = str(path)
    manager, policy, client = (modules[name] for name in ("skill_manager", "skill_policy", "client"))
    require(policy.validate_response is client.validate_response, "protocol function identity")
    try:
        policy.validate_response({}, {"questions": {}})
    except client.JevError as error:
        require(type(error) is client.JevError, "protocol exception identity")
    else:
        raise AssertionError("invalid response did not raise same-namespace JevError")
    adaptive = modules["skills.adaptive_focus_fire"]
    require(manager.allocate_adaptive_focus_fire is adaptive.allocate_adaptive_focus_fire,
            "manager allocator function identity")
    require(modules["skills"].allocate_focus_fire is modules["skills.focus_fire"].allocate_focus_fire,
            "skills package public re-export identity")
    # Exercise patching through the actual namespace used by SkillManager.
    case = next(case for case in CASES if case[0] == version)
    config = read_json(ADAPTER / "runs" / case[1] / "run.json")
    manager_instance = make_manager(manager.SkillManager, config)
    payload = None
    with (ADAPTER / "runs" / case[1] / "events.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            event = json.loads(line)
            if event["event"] == "skill_observation":
                payload = event["payload"]
                break
    require(payload is not None, "missing initial observation")
    manager_instance.select("adaptive_focus_fire", posture=config["skill_parameters"]["posture"])
    with mock.patch(namespace + ".skill_manager.allocate_adaptive_focus_fire",
                    wraps=adaptive.allocate_adaptive_focus_fire) as patched:
        with mock.patch.object(client.JevClient, "__init__", side_effect=AssertionError("API client created")):
            manager_instance.step(payload, case[2])
        patched.assert_called_once()
    sentinel = object()
    original = adaptive.select_retreat_action
    with mock.patch(namespace + ".skills.adaptive_focus_fire.select_retreat_action", sentinel):
        require(adaptive.allocate_adaptive_focus_fire.__globals__["select_retreat_action"] is sentinel,
                "allocator global patch did not resolve in same module")
    require(adaptive.select_retreat_action is original, "patch did not restore original function")
    return {"module_paths": paths, "client_exception_identity": True,
            "manager_patch_exercised": True, "allocator_global_patch": True}


def imported_environment_modules():
    return sorted(name for name in sys.modules if name == "pysc2" or name.startswith("pysc2.")
                  or name == "smac_hard" or name.startswith("smac_hard."))


def spec_origin(name):
    spec = importlib.util.find_spec(name)
    return None if spec is None else spec.origin


def run_probe(kind):
    if kind == "adapter":
        loader = runtime()
        before_path, before_pysc2 = list(sys.path), spec_origin("pysc2")
        result, managers = {}, {}
        for version in ("current", "baseline_v2_70pct"):
            Manager = loader.load_manager(version=version)
            namespace = Manager.__module__.rsplit(".", 1)[0]
            result[version] = module_path_check(namespace, version,
                                               CURRENT_MODULES if version == "current" else BASELINE_MODULES)
            files = loader.source_files(version=version)
            expected = set((ROOT / "skills" / version).glob("*.py")) | set((ROOT / "skills" / version / "skills").glob("*.py"))
            require(set(files) == expected, ("source_files does not cover package", version))
            result[version]["source_file_count"] = len(files)
            result[version]["namespace"] = namespace
            managers[version] = Manager
        require(managers["current"] is not managers["baseline_v2_70pct"], "versions share Manager object")
        require(result["current"]["namespace"] != result["baseline_v2_70pct"]["namespace"], "versions share namespace")
        try:
            loader._load_module("unit_value", version="baseline_v2_70pct")
        except ModuleNotFoundError as error:
            require(error.name.endswith(".unit_value"), ("unexpected missing dependency", error.name))
        else:
            raise AssertionError("baseline unexpectedly imported current unit_value")
        for call in (loader.load_manager, loader.source_files):
            try:
                call(version="invalid-migration-test-version")
            except ValueError:
                pass
            else:
                raise AssertionError("invalid version accepted")
        require(sys.path == before_path, "skill loader changed sys.path")
        require(spec_origin("pysc2") == before_pysc2, "skill loader changed pysc2 resolution")
        require(not before_pysc2 or "game_skill" not in before_pysc2, "adapter starts with vendored pysc2 pollution")
        require(not imported_environment_modules(), ("environment imported", imported_environment_modules()))
        return {"probe": kind, "versions": result, "sys_path_unchanged": True,
                "pysc2_origin": before_pysc2, "environment_modules_loaded": []}
    sys.path.insert(0, str(ORIGINAL.parent))
    result = module_path_check("jev_micro", "current", CURRENT_MODULES)
    package = importlib.import_module("jev_micro")
    require(Path(next(iter(package.__path__))).resolve() == (ROOT / "skills/current").resolve(),
            "legacy __path__ does not prioritize current")
    require(not imported_environment_modules(), ("environment imported", imported_environment_modules()))
    return {"probe": kind, **result, "legacy_package_paths": list(package.__path__),
            "environment_modules_loaded": []}


def make_manager(Manager, config):
    parameters = config["skill_parameters"]
    kwargs = {"max_steps": parameters["max_steps"],
              "combat_params": {key: value for key, value in parameters.items() if isinstance(value, bool)}}
    if "unit_value_parameters" in parameters:
        require("unit_value_params" in inspect.signature(Manager).parameters,
                "selected version cannot accept recorded value parameters")
        kwargs["unit_value_params"] = parameters["unit_value_parameters"]
    return Manager(**kwargs)


def verify_source_bytes(loader):
    reports = {}
    for version, run, _, _, _ in (CASES[0], CASES[2]):
        hashes = {key.replace("\\", "/"): value for key, value in
                  read_json(ADAPTER / "runs" / run / "run.json")["source_hashes"].items()}
        source = ROOT / "skills" / version
        checked = {}
        for path in loader.source_files(version=version):
            relative = path.relative_to(source)
            if relative.as_posix() == "__init__.py":
                continue  # New package marker, never an algorithm source.
            old_key = "game_skill/micro/jev_micro/" + relative.as_posix()
            require(old_key in hashes, ("missing historical hash", version, old_key))
            actual = sha256(path)
            require(actual == hashes[old_key], ("historical source changed", version, relative.as_posix()))
            if version == "current":
                require(not (ORIGINAL / relative).exists(), ("duplicate original algorithm", str(ORIGINAL / relative)))
            checked[relative.as_posix()] = actual
        require(len(checked) == (14 if version == "current" else 10), ("wrong source count", version, len(checked)))
        reports[version] = {"historical_reference": run, "byte_identical_count": len(checked), "sha256": checked}
    require(not list((ORIGINAL / "skills").glob("*.py")), "old skills algorithm source remains")
    return reports


def replay(loader, case):
    version, run_name, n_agents, expected_episodes, expected_frames = case
    run = ADAPTER / "runs" / run_name
    config = read_json(run / "run.json")
    Manager = loader.load_manager(version=version)
    manager = make_manager(Manager, config)
    frames = actions_matched = telemetry_matched = episodes = 0
    pending = None
    digest = hashlib.sha256()
    with (run / "events.jsonl").open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            event = json.loads(line)
            kind = event["event"]
            context = (run_name, line_no, event.get("episode"), event.get("step"))
            if kind == "episode_started":
                require(pending is None, (context, "unfinished preceding frame"))
                manager.reset()
                episodes += 1
            elif kind == "skill_observation":
                require(pending is None, (context, "observation before previous action"))
                payload = event["payload"]
                before = json.dumps(payload, sort_keys=True, allow_nan=False)
                if not manager.active:
                    manager.select("adaptive_focus_fire", posture=config["skill_parameters"]["posture"])
                actions, telemetry = manager.step(payload, n_agents)
                require(json.dumps(payload, sort_keys=True, allow_nan=False) == before, (context, "payload mutated"))
                pending = {"episode": event["episode"], "step": event["step"],
                           "actions": actions, "telemetry": telemetry, "telemetry_checked": False}
                frames += 1
            elif kind == "skill_step":
                require(pending and (event["episode"], event["step"]) == (pending["episode"], pending["step"]),
                        (context, "skill telemetry mismatched observation"))
                for key in TELEMETRY_KEYS:
                    require(event.get(key) == pending["telemetry"].get(key),
                            (context, key, "recorded", event.get(key), "replayed", pending["telemetry"].get(key)))
                require(not pending["telemetry_checked"], (context, "duplicate skill telemetry"))
                pending["telemetry_checked"] = True
                telemetry_matched += 1
            elif kind == "step":
                require(pending and (event["episode"], event["step"] - 1) == (pending["episode"], pending["step"]),
                        (context, "action mismatched observation"))
                require(event["actions"] == pending["actions"],
                        (context, "actions", event["actions"], pending["actions"]))
                require(pending["telemetry_checked"], (context, "missing skill telemetry"))
                digest.update(json.dumps([event["episode"], event["step"], event["actions"],
                                          {key: pending["telemetry"].get(key) for key in TELEMETRY_KEYS}],
                                         sort_keys=True, allow_nan=False).encode("utf-8"))
                actions_matched += 1
                pending = None
    require(pending is None, (run_name, "unfinished final frame"))
    require(frames == actions_matched == telemetry_matched == expected_frames,
            (run_name, frames, actions_matched, telemetry_matched, expected_frames))
    require(episodes == expected_episodes, (run_name, "episode count", episodes))
    return {"version": version, "run": run_name, "episodes": episodes, "frames": frames,
            "action_matches": actions_matched, "telemetry_matches": telemetry_matched,
            "telemetry_keys": list(TELEMETRY_KEYS), "input_payload_unchanged": True,
            "decision_digest_sha256": digest.hexdigest(), "events_sha256": sha256(run / "events.jsonl")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", choices=("adapter", "legacy"))
    args = parser.parse_args()
    if args.probe:
        print(json.dumps(run_probe(args.probe), ensure_ascii=False))
        return
    loader = runtime()
    results = {"scope": "Offline replay of saved observations; no SC2, model calls, or new win-rate estimate.",
               "source_bytes": verify_source_bytes(loader), "fresh_process_probes": {}, "replays": []}
    for probe in ("adapter", "legacy"):
        completed = subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()), "--probe", probe],
                                   cwd=ROOT, capture_output=True, text=True, encoding="utf-8", check=True)
        results["fresh_process_probes"][probe] = json.loads(completed.stdout)
        print("PASS fresh process " + probe, flush=True)
    for case in CASES:
        result = replay(loader, case)
        results["replays"].append(result)
        print(f"PASS {result['version']} {result['run']} {result['frames']} frames", flush=True)
    results["total_frames"] = sum(row["frames"] for row in results["replays"])
    require(results["total_frames"] == 4791, "wrong total frame count")
    results["status"] = "passed"
    OUTPUT.mkdir(parents=True, exist_ok=True)
    destination = OUTPUT / "migration.json"
    destination.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Skill migration verification", "", "All checks passed. No new games or model calls were made.", "",
             "| Package | Saved episodes | Frames | Actions, roles and assignments |", "| --- | ---: | ---: | --- |"]
    for version in ("current", "baseline_v2_70pct"):
        rows = [row for row in results["replays"] if row["version"] == version]
        lines.append(f"| {version} | {sum(row['episodes'] for row in rows)} | {sum(row['frames'] for row in rows)} | Exact match |")
    lines.extend(["", "Current: all 14 original source files match the frozen v6 recorded SHA-256 hashes; no algorithm copies remain at the old locations.",
                  "Baseline: all 10 original source files match the historical v2 recorded SHA-256 hashes.",
                  "", "Fresh processes verified module paths, version isolation, same-namespace protocol exception identity and monkeypatch behavior. Adapter loading does not change sys.path or import PySC2/SMAC-Hard. The legacy jev_micro namespace resolves the moved files through its package path.",
                  "", "Replay also matches predicted damage, All-in decisions and retreat reasons, and verifies that observation payloads stay unchanged.",
                  "", "This verifies migration and historical decision reproduction only. It does not establish a future 70% win rate or compare policies on new game states.",
                  "", "Machine-readable evidence: migration.json. Run with the adapter virtual environment: skills/tools/verify_migration.py.", ""])
    (OUTPUT / "MIGRATION_VERIFICATION.md").write_text("\n".join(lines), encoding="utf-8")
    print("PASS all 4791 frames; report: " + str(destination), flush=True)


if __name__ == "__main__":
    main()
