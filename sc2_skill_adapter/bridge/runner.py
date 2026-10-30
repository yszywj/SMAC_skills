"""Manifest generation, preflight, full episodes and paired evaluation."""
import argparse
import asyncio
import copy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import statistics
import sys
import time

from .config import GAME_ROOT, MAP_FILE, ROOT, SMAC_ROOT, digest, load_config, public_brief, map_file

SPATIAL_SKILLS = ("regroup", "cohesive_search")


def selected_spatial(skill):
    return skill in SPATIAL_SKILLS


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def positive(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def default_sc2():
    return os.environ.get("SC2PATH", r"C:\Program Files (x86)\StarCraft II")


def check(args):
    from .config import source_path
    source_path()
    import pysc2
    import smacv2
    from .client import load_api_key, JevError
    from .planner import find_codex, PlannerError
    from .scenarios import generate_manifest
    key_available = True
    try:
        load_api_key(args.config_file)
    except JevError:
        key_available = False
    try:
        codex = str(find_codex(args.codex_path))
    except PlannerError:
        codex = None
    report = {
        "python": sys.version, "python_executable": sys.executable,
        "pysc2": pysc2.__file__, "smacv2": smacv2.__file__,
        "sc2_path": str(args.sc2_path), "sc2_exists": args.sc2_path.is_dir(),
        "map_exists": MAP_FILE.is_file(), "jev_key_available": key_available,
        "codex_executable": codex, "skills": "off",
        "sampling_reproducible": all(
            generate_manifest(load_config(n), 2, 7) == generate_manifest(load_config(n), 2, 7)
            for n in (5, 10)),
        "versions": {name: importlib.metadata.version(name)
                     for name in ("pysc2", "numpy", "protobuf", "httpx", "PyYAML")},
    }
    if "game_skill" in pysc2.__file__:
        raise RuntimeError("Vendored SMAC-Hard PySC2 contaminated this environment")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.output:
        write_json(args.output, report)


def create_manifest(args):
    from .scenarios import generate_manifest
    if args.output.exists():
        raise FileExistsError(f"Will not overwrite manifest: {args.output}")
    config = load_config(args.size, getattr(args, "race", "protoss"))
    config["unit_balance"] = args.unit_balance
    manifest = generate_manifest(config, args.episodes, args.seed)
    write_json(args.output, manifest)
    print(f"Created {len(manifest['episodes'])} scenarios: {args.output}")


def outcome(info, env):
    if info.get("battle_won"):
        return "win"
    if info.get("dead_allies") == env.n_agents and info.get("dead_enemies") == env.n_enemies:
        return "draw"
    if env._episode_steps >= env.episode_limit and info.get("dead_allies", 0) < env.n_agents:
        return "timeout"
    return "loss"


def assert_clock(env, expected):
    actual = env._controller.observe().observation.game_loop
    if actual != expected:
        raise RuntimeError(f"Game advanced during inference: {expected} -> {actual}")


def initial_state_digest(env):
    """Audit only. Never feed hidden initial state to either model."""
    return digest({
        side: sorted((u.unit_type, round(u.pos.x, 4), round(u.pos.y, 4),
                      round(u.health, 3), round(u.shield, 3)) for u in units.values())
        for side, units in (("allies", env.agents), ("enemies", env.enemies))
    })


async def evaluate(args, manifest, method, output):
    if method in ("jev", "astra-jev") and manifest["env_args"]["map_name"] != "10gen_protoss":
        raise ValueError("Cross-race support is validated for local policies only; model planning remains Protoss-specific")
    from absl import flags
    if not flags.FLAGS.is_parsed():
        flags.FLAGS(["smacv2-jev"])
    from .client import JevClient, JevError, load_api_key
    from .environment import ReplayableEnv, audit_unit_balance
    from .feedback import action_feedback
    from .planner import MapPlanner, attach_plan
    from .policy import build_request, batch_requests, decode_actions, nearest_actions
    from .skill_runtime import (load_manager, source_files as skill_source_files,
                               manager_configuration)
    from .keep_order import validate_keep_heal_order

    is_skill = method == "adaptive-skill"
    skill_config = manager_configuration(args) if is_skill else None
    race_profile = getattr(args, "race_profile", "off")
    if race_profile != "off" and not is_skill:
        raise ValueError("--race-profile requires --method adaptive-skill")
    if race_profile != "off" and manifest["env_args"]["map_name"] != "10gen_" + race_profile:
        raise ValueError("--race-profile must match the manifest's race")
    skill_version = skill_config["version"] if is_skill else None
    value_metadata = skill_config["value_metadata"] if is_skill else {}
    combat_params = skill_config["manager_kwargs"]["combat_params"] if is_skill else {}
    action_extensions = validate_keep_heal_order(args, method, manifest['env_args'])

    output.mkdir(parents=True, exist_ok=False)
    os.environ["SC2PATH"] = str(args.sc2_path.resolve())
    env, client, planner = None, None, None
    skill_manager = None
    records, latencies = [], []
    requests, usage = 0, {}
    started = time.monotonic()
    events = (output / "events.jsonl").open("w", encoding="utf-8")

    def emit(event, **fields):
        events.write(json.dumps({"event": event, "skill_version": skill_version,
                                **fields}, ensure_ascii=False) + "\n")
        events.flush()

    sources = list((ROOT / "bridge").glob("*.py"))
    sources += [SMAC_ROOT / "smacv2/env/starcraft2" / name
                for name in ("starcraft2.py", "wrapper.py", "distributions.py")]
    sources.append(map_file(manifest["env_args"]))
    selected_skill_sources = skill_source_files(skill_version) if is_skill else []
    sources.extend(selected_skill_sources)
    skill_hashes = {p.relative_to(ROOT.parent).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in selected_skill_sources}
    metadata = {
        "method": method, "skills": "adaptive_focus_fire" if is_skill else "off",
        "skill_version": skill_version,
        "skill_source": skill_config["source"] if is_skill else None,
        "skill_source_hashes": skill_hashes,
        **value_metadata,
        "observation_protocol": "smacv2-visible-rows-v2",
        "skill_parameters": {
            "posture": args.adaptive_posture, "max_steps": args.skill_max_steps,
            **combat_params,
            "spatial_params": skill_config["manager_kwargs"].get("spatial_params", {}),
            **({"race_profile": race_profile} if race_profile != "off" else {}),
            **value_metadata,
        } if is_skill else None,
        "manifest_sha256": manifest["sha256"],
        "env_args": manifest["env_args"], "scenario_count": len(manifest["episodes"]),
        "model": args.model if method in ("jev", "astra-jev") else None,
        "planner_model": args.planner_model if method == "astra-jev" else None,
        "planner_effort": args.planner_effort if method == "astra-jev" else None,
        "planning_scope": "distribution-level; no episode snapshot; reused within run",
        "information": "centralized union of SMACv2 observations; not decentralized MARL",
        "batch_concurrency": args.batch_concurrency,
        "request_timeout": args.request_timeout, "max_requests": args.max_requests,
        "source_hashes": {os.path.relpath(p, ROOT.parent): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sources},
        "versions": {n: importlib.metadata.version(n)
                     for n in ("pysc2", "numpy", "protobuf", "s2clientprotocol", "httpx")},
        **({'action_extensions': action_extensions} if action_extensions else {}),
    }
    write_json(output / "run.json", metadata)
    write_json(output / "manifest.json", manifest)
    if is_skill:
        emit("skill_configuration", skill_source=metadata["skill_source"],
             skill_source_hashes=skill_hashes, skill_parameters=metadata["skill_parameters"])
        emit("unit_value_configuration", **value_metadata)
    if action_extensions:
        emit('action_extension_configuration', action_extensions=action_extensions)
    failure = None
    try:
        if is_skill:
            skill_manager = load_manager(skill_version)(**skill_config["manager_kwargs"])
        if method in ("jev", "astra-jev"):
            client = JevClient(load_api_key(args.config_file), timeout=args.request_timeout)
        if method == "astra-jev":
            planner = MapPlanner(output / "planner", emit, model=args.planner_model,
                                 effort=args.planner_effort, timeout=args.planner_timeout,
                                 executable=args.codex_path)
        for episode, scenario in enumerate(manifest["episodes"], 1):
            if skill_manager is not None:
                skill_manager.reset()
            episode_started = time.monotonic()
            config = copy.deepcopy(manifest["env_args"])
            config.update(seed=scenario["sc2_seed"], replay_dir=str(output / "replays"))
            # Fresh process per episode gives every method the same SC2 seed,
            # independent of how many steps a previous method survived.
            random.seed(scenario["sc2_seed"])
            import numpy as np
            np.random.seed(scenario["sc2_seed"])
            rng = random.Random(scenario["sc2_seed"])
            env = ReplayableEnv(config, **({'terran_keep_heal_order': True} if action_extensions else {}))
            if action_extensions and env.keep_heal_order_action_id != action_extensions['terran_keep_heal_order']['action_id']:
                raise RuntimeError('KeepHealOrder action ID differs from recorded extension contract')
            env.reset_record(scenario)
            env.env.scenario_brief = public_brief(config)
            data = env._controller.data()
            if config.get("unit_balance") == "native":
                audit = audit_unit_balance(env, data)
                write_json(output / "balance" / f"episode-{episode:04d}.json", audit)
                emit("unit_balance_verified", episode=episode, audit=audit)
            initial_digest = initial_state_digest(env)
            emit("episode_started", episode=episode, scenario_sha256=scenario["sha256"],
                 initial_state_sha256=initial_digest,
                 game_loop=env._obs.observation.game_loop,
                 game_version=env._controller.ping().game_version)
            health, shields, previous_actions, feedback = {}, {}, {}, None
            total_reward, steps = 0.0, 0
            while True:
                payload = build_request(env, data, args.model, episode,
                                        health, previous_actions, shields, feedback)
                loop = env._obs.observation.game_loop
                if planner is not None:
                    active = await planner.get_plan(payload, env)
                    assert_clock(env, loop)
                    payload = attach_plan(payload, active)
                if skill_manager is not None:
                    emit("skill_observation", episode=episode, step=steps, payload=payload)
                    spatial_choice = None
                    if getattr(args, "auto_spatial_dispatch", False):
                        # Local rule dispatcher (no model): split -> regroup,
                        # free & able -> search, contact -> combat controller.
                        spatial_choice = skill_manager.spatial_dispatch(payload)
                    if not skill_manager.active:
                        selected = spatial_choice or "adaptive_focus_fire"
                        skill_manager.select(selected, posture=args.adaptive_posture)
                        emit("skill_started", episode=episode, step=steps,
                             skill=selected,
                             **({"source": "auto_spatial_dispatch"}
                                if spatial_choice else {}))
                    step_result = skill_manager.step(payload, env.n_agents)
                    if isinstance(step_result[0], dict) and selected_spatial(
                            skill_manager.current_skill):
                        # Spatial skills return partial proposals; the declared
                        # nearest baseline fills every other unit. An
                        # interrupted empty proposal hands the SAME decision
                        # boundary to the combat controller instead of merging
                        # into the movement baseline.
                        partial, telemetry = step_result
                        if telemetry.get("status") == "interrupted" and not partial:
                            skill_manager.select("adaptive_focus_fire",
                                                 posture=args.adaptive_posture)
                            emit("skill_started", episode=episode, step=steps,
                                 skill="adaptive_focus_fire",
                                 source="spatial_interrupt_handoff")
                            actions, telemetry = skill_manager.step(payload, env.n_agents)
                        else:
                            actions = nearest_actions(payload, env.n_agents)
                            for index, action in (partial or {}).items():
                                actions[index] = action
                            telemetry["base_policy"] = "nearest"
                            telemetry["merged_units"] = sorted(partial or {})
                    else:
                        actions, telemetry = step_result
                    emit("skill_step", episode=episode, step=steps, game_loop=loop, **telemetry)
                    if telemetry.get("termination_reason"):
                        emit("skill_terminated", episode=episode, step=steps,
                             reason=telemetry["termination_reason"])
                elif client is not None:
                    semaphore = asyncio.Semaphore(args.batch_concurrency)

                    async def choose(batch):
                        nonlocal requests
                        async with semaphore:
                            for attempt in range(args.api_retries + 1):
                                if requests >= args.max_requests:
                                    raise RuntimeError("JEV request budget exhausted; no fallback policy used")
                                requests += 1
                                request_id = requests
                                emit("jev_request", episode=episode, request_id=request_id,
                                     attempt=attempt, payload=batch)
                                request_started = time.monotonic()
                                try:
                                    result = await client.choose(batch)
                                except JevError as exc:
                                    emit("jev_error", episode=episode, request_id=request_id, error=str(exc))
                                    if attempt == args.api_retries:
                                        raise
                                    continue
                                emit("jev_response", episode=episode, request_id=request_id,
                                     latency_ms=1000 * (time.monotonic() - request_started), **result)
                                for key, value in result["usage"].items():
                                    usage[key] = usage.get(key, 0) + value
                                return result["answers"]

                    decision_started = time.monotonic()
                    results = await asyncio.gather(
                        *(choose(batch) for batch in batch_requests(payload)), return_exceptions=True)
                    for result in results:
                        if isinstance(result, BaseException):
                            raise result
                    answers = {}
                    for result in results:
                        if set(answers) & set(result):
                            raise RuntimeError("Duplicate batch actor")
                        answers.update(result)
                    actions = decode_actions(payload, answers, env.n_agents)
                    latencies.append(1000 * (time.monotonic() - decision_started))
                    assert_clock(env, loop)
                elif method == "nearest":
                    emit("policy_observation", episode=episode, step=steps, payload=payload)
                    actions = nearest_actions(payload, env.n_agents)
                else:
                    actions = decode_actions(payload, {
                        actor: {"choice": rng.choice(list(question["criteria"]))}
                        for actor, question in payload["questions"].items()}, env.n_agents)
                if not isinstance(actions, list) or len(actions) != env.n_agents:
                    raise RuntimeError("Final actions must be a complete per-unit list.")
                for i, action in enumerate(actions):
                    mask = env.get_avail_agent_actions(i)
                    if not isinstance(action, int) or isinstance(action, bool) or not 0 <= action < len(mask) or not mask[action]:
                        raise RuntimeError(f"Illegal action for u{i}: {action}")
                if skill_manager is not None and hasattr(skill_manager, "commit_final_actions"):
                    skill_manager.commit_final_actions(actions)
                health = {u.tag: u.health for u in env.agents.values()}
                shields = {u.tag: u.shield for u in env.agents.values()}
                previous_actions = {f"u{i}": str(action) for i, action in enumerate(actions)}
                reward, terminated, info = env.step(actions)
                steps += 1
                total_reward += float(reward)
                feedback = action_feedback(env, loop)
                delta = env._obs.observation.game_loop - loop
                if delta != env._step_mul:
                    raise RuntimeError(f"Unexpected simulation step: {delta}")
                emit("step", episode=episode, step=steps, game_loop=loop,
                     actions=actions, reward=float(reward), terminated=terminated, feedback=feedback)
                if terminated:
                    break
            replay_dir = output / "replays"
            replay_dir.mkdir(exist_ok=True)
            replay = replay_dir / f"episode-{episode:04d}.SC2Replay"
            replay.write_bytes(env._controller.save_replay())
            record = {
                "episode": episode, "scenario_sha256": scenario["sha256"],
                "initial_state_sha256": initial_digest,
                "outcome": outcome(info, env), "steps": steps, "reward": total_reward,
                "enemy_elimination_fraction": info["dead_enemies"] / env.n_enemies,
                "allied_survival_fraction": 1 - info["dead_allies"] / env.n_agents,
                "wall_seconds": time.monotonic() - episode_started,
                "replay": str(replay), "info": info,
            }
            records.append(record)
            emit("episode_completed", **record)
            write_json(output / "episodes.json", records)
            print(f"{method}: episode {episode}/{len(manifest['episodes'])}: "
                  f"{record['outcome']} ({steps} steps)", flush=True)
            env.close()
            env = None
    except BaseException as exc:
        failure = f"{type(exc).__name__}: {exc}"
        emit("run_failed", error=failure, completed_episodes=len(records))
        raise
    finally:
        if env is not None:
            env.close()
        if client is not None:
            await client.close()
        if planner is not None:
            await planner.close()
        summary = {
            "method": method, "status": "failed" if failure else "complete", "error": failure,
            "skill_version": skill_version,
            "skill_source": metadata["skill_source"],
            "skill_source_hashes": skill_hashes,
            **value_metadata,
            "skill_parameters": metadata["skill_parameters"],
            "manifest_sha256": manifest["sha256"], "completed_episodes": len(records),
            "expected_episodes": len(manifest["episodes"]),
            "outcomes": {name: sum(r["outcome"] == name for r in records)
                         for name in ("win", "loss", "draw", "timeout")},
            "win_rate": sum(r["outcome"] == "win" for r in records) / len(records) if records else None,
            "mean_enemy_elimination": statistics.mean(r["enemy_elimination_fraction"] for r in records) if records else None,
            "wall_seconds": time.monotonic() - started, "jev_requests": requests, "jev_usage": usage,
            "median_team_wait_ms": statistics.median(latencies) if latencies else None,
            "planner": planner.stats if planner else None,
            "skill": skill_manager.stats if skill_manager else None,
            "monetary_cost": None, "cost_note": "Usage recorded; price not assumed.",
            **({'action_extensions': action_extensions} if action_extensions else {}),
        }
        write_json(output / "summary.json", summary)
        events.close()
    return summary


def parser():
    from .skill_runtime import (add_spatial_arguments, add_unit_value_arguments,
                                SOURCES)
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    check_p = sub.add_parser("check", help="Offline preflight; no model calls or SC2 launch")
    check_p.add_argument("--output", type=Path)
    make = sub.add_parser("scenarios", help="Create a reusable paired scenario manifest")
    make.add_argument("--size", type=int, choices=(5, 10), required=True)
    make.add_argument("--race", choices=("protoss", "terran", "zerg"), default="protoss")
    make.add_argument("--episodes", type=positive, default=3)
    make.add_argument("--seed", type=int, default=1)
    make.add_argument("--unit-balance", choices=("original", "native"), default="original")
    make.add_argument("--output", type=Path, required=True)
    run = sub.add_parser("run", help="Run one method on a manifest")
    run.add_argument("--method", choices=("random", "nearest", "jev", "astra-jev", "adaptive-skill"), default="nearest")
    suite = sub.add_parser("suite", help="Paired methods on one immutable scenario manifest")
    suite.add_argument("--methods", nargs="+", choices=("random", "nearest", "jev", "astra-jev", "adaptive-skill"),
                       default=["nearest", "jev", "astra-jev"])
    for command in (check_p, run, suite):
        command.add_argument("--sc2-path", type=Path, default=Path(default_sc2()))
        command.add_argument("--config-file", type=Path, default=GAME_ROOT / "config.md")
        command.add_argument("--codex-path", type=Path)
    for command in (run, suite):
        command.add_argument("--manifest", type=Path, required=True)
        command.add_argument("--output-dir", type=Path, required=True)
        command.add_argument("--model", default="jev-1.13.0")
        command.add_argument("--planner-model", default="gpt-6-astra")
        command.add_argument("--planner-effort", choices=("low", "medium", "high", "xhigh", "max"), default="medium")
        command.add_argument("--planner-timeout", type=positive, default=180)
        command.add_argument("--request-timeout", type=positive, default=15)
        command.add_argument("--max-requests", type=positive, default=10000)
        command.add_argument("--batch-concurrency", type=positive, default=4)
        command.add_argument("--api-retries", type=int, choices=range(4), default=2)
        command.add_argument("--skill-mode", choices=("off", "always"), default=None,
                             help="Optional consistency check; adaptive-skill uses always, other methods off")
        command.add_argument("--skill-version", choices=tuple(SOURCES), default="current",
                             help="Skill implementation; baseline_v2_70pct requires its original fixed preset")
        command.add_argument("--race-profile", choices=("off", "terran", "zerg"), default="off",
                             help="Opt-in race-specific adaptive combat; current skills only")
        command.add_argument('--terran-keep-heal-order', action='store_true',
                             help='Native current Terran only: append an action that submits no command for an eligible observed heal order')
        command.add_argument("--skill-max-steps", type=positive, default=4)
        command.add_argument("--adaptive-posture", choices=("aggressive", "balanced", "defensive"), default="balanced")
        command.add_argument("--adaptive-window-fire", action="store_true")
        command.add_argument("--adaptive-range-preserving-move", action="store_true")
        command.add_argument("--adaptive-target-scoring", action="store_true")
        command.add_argument("--adaptive-coordinated-approach", action="store_true")
        command.add_argument("--adaptive-safe-reentry", action="store_true")
        command.add_argument("--adaptive-accurate-damage", action="store_true")
        command.add_argument("--adaptive-critical-escape", action="store_true")
        command.add_argument("--adaptive-planned-reentry", action="store_true")
        command.add_argument("--adaptive-engagement-commitment", action="store_true")
        command.add_argument("--adaptive-approach-progress-guard", action="store_true",
                             help="Detect own approach loops across enemy motion and target switches")
        add_spatial_arguments(command)
        add_unit_value_arguments(command)
    return p


def main():
    args = parser().parse_args()
    if args.command in ("run", "suite"):
        methods = [args.method] if args.command == "run" else args.methods
        expected = "always" if getattr(args, "method", None) == "adaptive-skill" else "off"
        if args.skill_mode is not None and args.skill_mode != expected:
            raise SystemExit(f"--skill-mode must be {expected} for the selected method")
        if args.skill_version != "current" and "adaptive-skill" not in methods:
            raise SystemExit(f"--skill-version {args.skill_version} requires the adaptive-skill method")
        if args.race_profile != "off" and any(m != "adaptive-skill" for m in methods):
            raise SystemExit("--race-profile requires adaptive-skill only")
        if "adaptive-skill" in methods:
            from .skill_runtime import manager_configuration
            try:
                manager_configuration(args)
            except ValueError as exc:
                raise SystemExit(str(exc)) from exc
    if args.command == "check":
        check(args)
    elif args.command == "scenarios":
        create_manifest(args)
    else:
        from .scenarios import validate_manifest
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        validate_manifest(manifest)
        args.output_dir = args.output_dir.resolve()
        if args.command == "run":
            asyncio.run(evaluate(args, manifest, args.method, args.output_dir))
        else:
            args.output_dir.mkdir(parents=True, exist_ok=False)
            summaries = []
            for method in args.methods:
                summaries.append(asyncio.run(evaluate(args, manifest, method, args.output_dir / method)))
                write_json(args.output_dir / "comparison.json", summaries)
