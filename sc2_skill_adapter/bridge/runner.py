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

from .config import GAME_ROOT, ROOT, SMAC_ROOT, digest, load_config, public_brief, map_file, race_for_map

SPATIAL_SKILLS = ("regroup", "cohesive_search")
EXECUTION_MODES = ("legacy8", "micro8-off", "micro8-hydra", "micro8-all")


def selected_spatial(skill):
    return skill in SPATIAL_SKILLS


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def score_snapshot(env):
    obs = env._obs.observation
    if not obs.HasField('score') or not obs.score.HasField('score_details'):
        raise RuntimeError('SC2 observation missing requested score_details')
    details = obs.score.score_details
    values = {}
    for field in ('total_damage_dealt', 'total_damage_taken', 'total_healed'):
        if not details.HasField(field):
            raise RuntimeError(f'SC2 score_details missing {field}')
        vital = getattr(details, field)
        values[field] = {k: getattr(vital, k) if vital.HasField(k) else None
                         for k in ('life', 'shields', 'energy')}
    return {'game_loop': obs.game_loop, 'player_id': obs.player_common.player_id,
            'score_details': values}


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
        "map_exists": map_file({'map_name': '10gen_protoss'}).is_file(), "jev_key_available": key_available,
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
    if any(k.startswith(('ZERG_SQUAD_', 'ZERG_RELAY_')) for k in os.environ):
        if str(ROOT.parent) not in sys.path:
            sys.path.insert(0, str(ROOT.parent))
        if any(k.startswith('ZERG_RELAY_') for k in os.environ):
            from .zerg_relay import check_options
        else:
            from .zerg_squad import check_options
        check_options(os.environ, method)
    if 'ZERG_SWARM_GATE' in os.environ or 'ZERG_SWARM_GATE_SHA256' in os.environ:
        required = ('ZERG_SWARM_GATE', 'ZERG_SWARM_GATE_SHA256', 'ZERG_SWARM_SHA256',
                    'ZERG_DISTILL_MODEL', 'ZERG_DISTILL_MODEL_SHA256', 'ZERG_DISTILL_OPENING_SHA256',
                    'ZERG_DISTILL_SELECTOR_SHA256', 'ZERG_DISTILL_TRIAL_ID')
        if (method != 'adaptive-skill' or os.environ.get('ZERG_DISTILL_VERSION') != 'v2'
                or not all(os.environ.get(k) for k in required) or 'ZERG_ROUTE_IDENTITY' in os.environ):
            raise ValueError('online Swarm routing requires a frozen full v2 parent and gate')
    if method == 'radical-swarm':
        if (manifest['env_args']['map_name'] != '10gen_zerg' or args.execution_mode != 'micro8-off'
                or args.native_teacher_control != 'off' or args.pair_mode != 'off'
                or any(k.startswith('ZERG_') and k not in ('ZERG_SWARM_SHA256', 'ZERG_SWARM_TRIAL_ID')
                       for k in os.environ)):
            raise ValueError('radical-swarm requires 10gen_zerg, micro8-off and no other controller')
        return await _evaluate(args, manifest, method, output)
    versions = ('v2','alternative','route') if 'ZERG_ROUTE_IDENTITY' in os.environ else ('v1','v2','v3')
    if 'ZERG_DISTILL_VERSION' in os.environ and ('ZERG_DISTILL_MODEL' not in os.environ or
            os.environ['ZERG_DISTILL_VERSION'] not in versions):
        raise ValueError(f'Explicit tree version requires one of {versions} and a distill model')
    if 'ZERG_DISTILL_MODEL' in os.environ:
        if (not all(os.environ.get(k, '').strip() for k in ('ZERG_DISTILL_MODEL', 'ZERG_SHADOW_DIR', 'ZERG_SHADOW_ROSTER'))
                or any(k in os.environ for k in ('ZERG_FORK_SPEC', 'ZERG_WHOLE_AC_ARM'))
                or getattr(args, 'auto_spatial_dispatch', False) or args.execution_mode != 'micro8-all'
                or args.adaptive_posture != 'balanced'):
            raise ValueError('Distillation requires a model, explicit Shadow dir/roster and the balanced micro8-all parent only')
        return await _evaluate(args, manifest, method, output, distill_path=os.environ['ZERG_DISTILL_MODEL'])
    if 'ZERG_WHOLE_AC_ARM' in os.environ:
        arm = os.environ['ZERG_WHOLE_AC_ARM']
        if arm not in ('A', 'C') or 'ZERG_FORK_SPEC' in os.environ:
            raise ValueError('ZERG_WHOLE_AC_ARM must be A or C and is mutually exclusive with ZERG_FORK_SPEC')
        if not all(os.environ.get(k, '').strip() for k in ('ZERG_SHADOW_DIR', 'ZERG_SHADOW_ROSTER')):
            raise ValueError('Whole-game A/C requires explicit ZERG_SHADOW_DIR and ZERG_SHADOW_ROSTER')
        if getattr(args, 'auto_spatial_dispatch', False):
            raise ValueError('Whole-game A/C requires the frozen tactical parent, not spatial dispatch')
        return await _evaluate(args, manifest, method, output, whole_arm=arm)
    if 'ZERG_FORK_SPEC' not in os.environ:
        return await _evaluate(args, manifest, method, output)
    from .zerg_fork import Fork
    output.mkdir(parents=True, exist_ok=False)
    fork = Fork(output / 'fork_receipt.json')
    try:
        fork.load(os.environ['ZERG_FORK_SPEC'], manifest)
        summary = await _evaluate(args, manifest, method, output, fork)
        fork.finish()
        return summary
    except BaseException as exc:
        fork.save(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise


async def _evaluate(args, manifest, method, output, fork=None, whole_arm=None, distill_path=None):
    if method in ("jev", "astra-jev") and manifest["env_args"]["map_name"] != "10gen_protoss":
        raise ValueError("Cross-race support is validated for local policies only; model planning remains Protoss-specific")
    execution_mode = getattr(args, "execution_mode", "legacy8")
    if execution_mode not in EXECUTION_MODES:
        raise ValueError(f"Unknown execution mode: {execution_mode!r}")
    if execution_mode == "micro8-all" and (method != "adaptive-skill"
            or getattr(args, "skill_version", None) != "tactical_v1"):
        raise ValueError("micro8-all requires adaptive-skill with tactical_v1")
    if execution_mode != "legacy8" and manifest["env_args"].get("step_mul") != 8:
        raise ValueError(
            f"--execution-mode {execution_mode} supports step_mul=8 manifests only; "
            f"this manifest declares step_mul={manifest['env_args'].get('step_mul')!r}")
    micro_execution = execution_mode != "legacy8"
    all_attackers = execution_mode == "micro8-all"
    from absl import flags
    if not flags.FLAGS.is_parsed():
        flags.FLAGS(["smacv2-jev"])
    from .client import JevClient, JevError, load_api_key
    from .environment import ReplayableEnv, audit_unit_balance
    from .feedback import action_feedback
    from .micro_trigger import MicroAttackWindow
    from .planner import MapPlanner, attach_plan
    from .policy import build_request, batch_requests, decode_actions, nearest_actions
    from .skill_runtime import (load_manager, source_files as skill_source_files,
                               manager_configuration)
    from .keep_order import validate_keep_heal_order

    is_skill = method == "adaptive-skill"
    if getattr(args, 'skill_tree', None) is not None and not is_skill:
        raise ValueError('--skill-tree requires --method adaptive-skill')
    skill_config = manager_configuration(args) if is_skill else None
    skill_tree_policy = skill_config.get('skill_tree_policy') if is_skill else None
    local_policy = None
    if getattr(args, 'local_policy', None) is not None:
        if str(ROOT.parent) not in sys.path:
            sys.path.insert(0, str(ROOT.parent))
        from decision_tree.local.runtime import load_policy, validate_options, execution_context
        validate_options(args, method, manifest['env_args'])
        local_policy = load_policy(args.local_policy)
    race_profile = getattr(args, "race_profile", "off")
    if race_profile != "off" and not is_skill:
        raise ValueError("--race-profile requires --method adaptive-skill")
    if race_profile != "off" and manifest["env_args"]["map_name"] != "10gen_" + race_profile:
        raise ValueError("--race-profile must match the manifest's race")
    skill_version = skill_config["version"] if is_skill else None
    value_metadata = skill_config["value_metadata"] if is_skill else {}
    combat_params = skill_config["manager_kwargs"].get("combat_params", {}) if is_skill else {}
    action_extensions = validate_keep_heal_order(args, method, manifest['env_args'])
    native_control = getattr(args, "native_teacher_control", "off")
    if native_control != "off":
        if str(ROOT.parent) not in sys.path:
            sys.path.insert(0, str(ROOT.parent))
        from tree_skills.integration.native_teacher import NativeTeacherRuntime, validate_control
        validate_control(args, method, manifest["env_args"])

    if fork is None:
        output.mkdir(parents=True, exist_ok=False)
    shadow = None
    skill_manager = None
    if 'ZERG_SHADOW_DIR' in os.environ or 'ZERG_SHADOW_ROSTER' in os.environ:
        from .zerg_shadow import start_shadow
        shadow, skill_manager = start_shadow(
            os.environ.get('ZERG_SHADOW_DIR'), os.environ.get('ZERG_SHADOW_ROSTER'),
            manifest, method, args, skill_config['manager_kwargs'] if is_skill else {})
    os.environ["SC2PATH"] = str(args.sc2_path.resolve())
    env, client, planner = None, None, None
    skill_tree_controller = None
    local_controller = None
    pair_runtime = None
    native_runtime = None
    records, latencies = [], []
    requests, usage = 0, {}
    started = time.monotonic()
    events = (output / "events.jsonl").open("w", encoding="utf-8")

    def emit(event, **fields):
        events.write(json.dumps({"event": event, "skill_version": skill_version,
                                **fields}, ensure_ascii=False) + "\n")
        events.flush()

    sources = list((ROOT / "bridge").glob("*.py"))
    route = None
    squad = None
    relay = None
    if 'ZERG_RELAY_MODE' in os.environ:
        from .zerg_relay import SOURCE as RELAY_SOURCE, RelayRuntime
        sources.append(ROOT.parent / RELAY_SOURCE)
    if 'ZERG_SQUAD_MODE' in os.environ:
        from .zerg_squad import CROSS_MODES, CROSS_SOURCE, SOURCE, SquadRuntime
        sources.append(ROOT.parent / SOURCE)
        if os.environ['ZERG_SQUAD_MODE'] in CROSS_MODES:
            sources.append(ROOT.parent / CROSS_SOURCE)
    if 'ZERG_SWARM_GATE' in os.environ:
        sources.extend(ROOT.parent / name for name in ('decision_tree/zerg/radical_swarm.py',
            'decision_tree/zerg/policy_route_gate.py', 'decision_tree/scripts/linux/fit_swarm_explorer.py',
            'decision_tree/scripts/linux/fit_swarm_gate_324.py'))
    if method == 'radical-swarm':
        if str(ROOT.parent) not in sys.path:
            sys.path.insert(0, str(ROOT.parent))
        from decision_tree.zerg import radical_swarm
        swarm_source = Path(radical_swarm.__file__).resolve()
        if swarm_source != ROOT.parent / 'decision_tree/zerg/radical_swarm.py':
            raise ValueError('unexpected swarm controller source')
        sources.append(swarm_source)
    if distill_path is not None:
        sources.append(ROOT.parent / 'decision_tree/zerg/policy_distill.py')
        if 'ZERG_ROUTE_IDENTITY' in os.environ:
            sources.extend(ROOT.parent / f'decision_tree/zerg/{name}.py' for name in ('policy_route_alt','policy_route_gate'))
        if os.environ.get('ZERG_DISTILL_VERSION') in ('v2', 'v3','alternative','route'):
            sources.append(ROOT.parent / 'decision_tree/zerg/policy_repair_v2.py')
            if os.environ.get('ZERG_DISTILL_OPENING_SHA256'):
                sources.append(ROOT.parent / 'decision_tree/zerg/opening_support_v2.py')
            if os.environ['ZERG_DISTILL_VERSION'] == 'v3':
                sources.append(ROOT.parent / 'decision_tree/zerg/policy_repair_v3.py')
    sources += [SMAC_ROOT / "smacv2/env/starcraft2" / name
                for name in ("starcraft2.py", "wrapper.py", "distributions.py")]
    selected_map = map_file(manifest['env_args'])
    sources.append(selected_map)
    damage_enabled = bool(os.environ.get('SMAC_DAMAGE_MAP_DIR'))
    if damage_enabled:
        from scripts.make_damage_map import RACE_TYPES, begin_resource_damage, read_resource_damage
        sources.append(ROOT / 'scripts' / 'make_damage_map.py')
    selected_skill_sources = skill_source_files(skill_version) if is_skill else []
    if shadow is not None:
        resolved_source = Path(shadow.bundle_provenance['module_file']).parent
        skill_config['source'] = str(resolved_source)
        selected_skill_sources = [p for p in sorted(resolved_source.rglob('*.py')) if 'tests' not in p.parts]
    medivac_mode = skill_config.get("medivac_policy", "legacy") if is_skill else "legacy"
    medivac_runtime = skill_config["manager_kwargs"].get("medivac_tree") if is_skill else None
    if medivac_runtime is not None:
        sources.extend(sorted((ROOT.parent / "medivac_tree").glob("*.py")))
        sources.append(Path(medivac_runtime.metadata["tree_path"]))
    sources.extend(selected_skill_sources)
    if skill_tree_policy is not None:
        sources.extend(sorted((ROOT.parent / 'decision_tree').glob('*.py')))
    if local_policy is not None:
        sources.extend(sorted((ROOT.parent / 'decision_tree/local').glob('*.py')))
        sources.extend(sorted((ROOT.parent / 'skills/local_v1').glob('*.py')))
    skill_hashes = {p.relative_to(ROOT.parent).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in selected_skill_sources}
    metadata = {
        "method": method, "skills": "adaptive_focus_fire" if is_skill else "off",
        "execution_mode": execution_mode,
        "macro_loops": 8 if micro_execution else int(manifest["env_args"].get("step_mul", 8)),
        "engine_step_loops": 1 if micro_execution else int(manifest["env_args"].get("step_mul", 8)),
        "native_teacher_control": native_control,
        "medivac_policy": medivac_mode,
        "medical_patient_coordination": skill_config.get("medical_patient_coordination") if is_skill else None,
        "medivac_tree_metadata": medivac_runtime.metadata if medivac_runtime is not None else None,
        "rule_controller_executed": is_skill and native_control != "all",
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
        "map_file": str(selected_map.resolve()),
        "damage_by_type_enabled": damage_enabled,
        "env_args": manifest["env_args"], "scenario_count": len(manifest["episodes"]),
        "model": args.model if method in ("jev", "astra-jev") else None,
        "planner_model": args.planner_model if method == "astra-jev" else None,
        "planner_effort": args.planner_effort if method == "astra-jev" else None,
        "planning_scope": "distribution-level; no episode snapshot; reused within run",
        "information": ("per-agent native observations and histories" if native_control == "all"
                        else "centralized union of SMACv2 observations for rules; teacher uses per-agent native observations"),
        "batch_concurrency": args.batch_concurrency,
        "request_timeout": args.request_timeout, "max_requests": args.max_requests,
        "source_hashes": {os.path.relpath(p, ROOT.parent): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sources},
        "versions": {n: importlib.metadata.version(n)
                     for n in ("pysc2", "numpy", "protobuf", "s2clientprotocol", "httpx")},
        **({'action_extensions': action_extensions} if action_extensions else {}),
    }
    if method == 'radical-swarm':
        swarm_hash = metadata['source_hashes']['decision_tree/zerg/radical_swarm.py']
        if os.environ.get('ZERG_SWARM_SHA256', swarm_hash) != swarm_hash:
            raise ValueError('swarm controller changed after freeze')
        metadata['swarm'] = dict(controller_sha256=swarm_hash, trial_id=os.environ.get('ZERG_SWARM_TRIAL_ID'),
                                 protocol='macro-only; one decision per 8 loops; no interior orders')
        metadata['information'] = 'centralized union of visible SMACv2 observations and per-unit action masks'
    if shadow is not None:
        metadata['zerg_shadow'] = {
            'run_id': shadow.run_id, 'scene_config_id': shadow.config['scene_config']['scene_config_id'],
            'records_path': shadow.config['records_path'], 'events_path': shadow.config['events_path'],
              'execution_context_hash': shadow.execution_context_hash, **shadow.bundle_provenance}
    whole = None
    if distill_path is not None:
        from .zerg_distill import Distill, check_protocol
        check_protocol(metadata)
        if 'ZERG_SQUAD_MODE' in os.environ:
            if 'ZERG_RELAY_MODE' in os.environ:
                relay = squad = whole = RelayRuntime(output, distill_path, metadata['zerg_shadow'])
                metadata['combat_relay'] = relay.configure_relay(os.environ, metadata['source_hashes'])
            else:
                squad = whole = SquadRuntime(output, distill_path, metadata['zerg_shadow'])
            metadata['squad_rendezvous'] = squad.configure(os.environ, metadata['source_hashes'])
        else:
            whole = Distill(output, distill_path, metadata['zerg_shadow'])
        metadata['zerg_distill'] = whole.contract
        if 'ZERG_SWARM_GATE' in os.environ:
            from .zerg_swarm_route import OnlineRoute
            route = OnlineRoute(os.environ, whole.contract, metadata['source_hashes']['decision_tree/zerg/radical_swarm.py'])
            metadata['swarm_route'] = route.contract
    if whole_arm is not None:
        from .zerg_whole_ac import WholeAC
        whole = WholeAC(output, whole_arm, metadata['zerg_shadow'])
        metadata['whole_ac'] = whole.contract
    if skill_tree_policy is not None:
        from decision_tree.experiment import policy_contract, validate_deployment
        from decision_tree.schema import policy_hash
        metadata['skill_tree'] = {
            'policy': skill_tree_policy, 'policy_contract': policy_contract(skill_tree_policy),
            'policy_hash': policy_hash(skill_tree_policy),
            'evidence_kind': 'matched_observed_prefix', 'intervention_unit': 'skill_call'}
        validate_deployment(metadata, skill_tree_policy)
    if local_policy is not None:
        from skills.local_v1 import FEATURE_NAMES
        metadata['local_policy'] = {
            'policy': local_policy, 'feature_names': list(FEATURE_NAMES),
            'context_hash': execution_context(metadata, local_policy),
            'evidence_kind': 'matched_observed_prefix_not_internal_snapshot',
            'intervention_unit': 'single_actor_single_decision_step'}
    write_json(output / "run.json", metadata)
    write_json(output / "manifest.json", manifest)
    if is_skill:
        emit("skill_configuration", skill_source=metadata["skill_source"],
             skill_source_hashes=skill_hashes, skill_parameters=metadata["skill_parameters"])
        emit("unit_value_configuration", **value_metadata)
    if action_extensions:
        emit('action_extension_configuration', action_extensions=action_extensions)
    if medivac_runtime is not None:
        emit('medivac_tree_configuration', mode=medivac_mode,
             medical_patient_coordination=skill_config['medical_patient_coordination'],
             metadata=medivac_runtime.metadata)
    failure = None
    try:
        if fork is not None:
            fork.check_context(metadata, shadow)
        if medivac_runtime is not None and medivac_runtime.metadata.get('selection_mode') == 'model':
            from medivac_tree.experiment import validate_deployment
            audit = validate_deployment(metadata, medivac_runtime.selection['model'])
            emit('medivac_return_deployment', **audit)
        if native_control != "off":
            native_runtime = NativeTeacherRuntime(native_control, args.native_teacher_checkpoint,
                                                  args.native_teacher_last_token)
            metadata["native_teacher_metadata"] = native_runtime.describe()
            write_json(output / "run.json", metadata)
            emit("native_teacher_configuration", **native_runtime.describe())
        if is_skill and getattr(args, "pair_mode", "off") != "off":
            import sys as _sys, os as _os
            _root = _os.path.abspath(_os.path.join(
                _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), ".."))
            if _root not in _sys.path:
                _sys.path.insert(0, _root)
            from tree_skills.integration import pair_runtime as _pair_runtime_mod
            pair_runtime = _pair_runtime_mod.PairRuntime.from_args(
                mode=args.pair_mode, model_dir=args.pair_model_dir,
                checkpoint=args.pair_teacher_checkpoint,
                last_token=args.pair_last_token, failure=args.pair_failure,
                shadow_of=args.pair_shadow_of, collect_dir=args.pair_collect,
                run_label=os.path.basename(output))
            skill_config["manager_kwargs"].update(
                pair_scorer=pair_runtime.scorer,
                pair_scoring=("shadow" if args.pair_mode == "shadow" else "on"),
                pair_histories=pair_runtime.histories,
                pair_strict=(args.pair_failure == "strict"))
            pair_metadata = pair_runtime.describe(
                checkpoint=args.pair_teacher_checkpoint,
                model_dir=args.pair_model_dir, last_token=args.pair_last_token)
            emit("pair_configuration", **pair_metadata)
        if is_skill and native_control != "all":
            if shadow is None:
                skill_manager = load_manager(skill_version)(**skill_config["manager_kwargs"])
            # The Shadow wrapper forwards reads/methods but not attribute assignments.
            acting_manager = skill_manager if shadow is None else skill_manager.manager
            acting_manager.micro_all_attackers = all_attackers
            if skill_tree_policy is not None:
                from decision_tree.integration import SkillTreeController
                skill_tree_controller = SkillTreeController(
                    skill_manager, skill_tree_policy, race=race_for_map(manifest['env_args']['map_name']),
                    posture=args.adaptive_posture)
        if local_policy is not None:
            from decision_tree.local.runtime import LocalController
            local_controller = LocalController(skill_manager, local_policy,
                posture=args.adaptive_posture, context_hash=metadata['local_policy']['context_hash'])
        if method in ("jev", "astra-jev"):
            client = JevClient(load_api_key(args.config_file), timeout=args.request_timeout)
        if method == "astra-jev":
            planner = MapPlanner(output / "planner", emit, model=args.planner_model,
                                 effort=args.planner_effort, timeout=args.planner_timeout,
                                 executable=args.codex_path)
        for episode, scenario in enumerate(manifest["episodes"], 1):
            if route is not None:
                route.reset()
            swarm = radical_swarm.RadicalSwarm() if method == 'radical-swarm' else None
            if skill_manager is not None:
                skill_manager.reset()
                acting_manager.micro_all_attackers = all_attackers
            if shadow is not None:
                shadow.episode_seed = scenario['sc2_seed']
            if skill_tree_controller is not None:
                skill_tree_controller.reset()
            if local_controller is not None:
                local_controller.reset()
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
            initial_score = score_snapshot(env)
            env.env.scenario_brief = public_brief(config)
            data = env._controller.data()
            if damage_enabled:
                initial_rosters = {str(p): {str(u.tag): data.units[u.unit_type] for u in units.values()}
                                   for p, units in ((1, env.agents), (2, env.enemies))}
                begin_resource_damage(env._controller)
            if config.get("unit_balance") == "native":
                audit = audit_unit_balance(env, data)
                write_json(output / "balance" / f"episode-{episode:04d}.json", audit)
                emit("unit_balance_verified", episode=episode, audit=audit)
            initial_digest = initial_state_digest(env)
            if skill_tree_policy is not None and skill_tree_policy['selection']['mode'] == 'model':
                expected_version = skill_tree_policy['selection']['model'].get('game_version')
                if not expected_version or env._controller.ping().game_version != expected_version:
                    raise ValueError('Skill return model SC2 build differs from the fitted data')
            if local_policy is not None and local_policy['selection']['mode'] == 'model':
                expected = local_policy['model'].get('game_version')
                if not expected or env._controller.ping().game_version != expected:
                    raise ValueError('Local model requires matching recorded SC2 game version')
            game_version = env._controller.ping().game_version
            if fork is not None:
                fork.start(scenario, initial_digest, game_version, env._obs.observation.game_loop)
            if whole is not None:
                whole.start(episode, scenario, initial_digest, game_version)
            emit("episode_started", episode=episode, scenario_sha256=scenario["sha256"],
                 initial_state_sha256=initial_digest,
                 game_loop=env._obs.observation.game_loop,
                 game_version=game_version)
            health, shields, previous_actions, feedback = {}, {}, {}, None
            total_reward, steps = 0.0, 0
            micro_records, micro_windows = [], 0
            while True:
                payload = build_request(env, data, args.model, episode,
                                        health, previous_actions, shields, feedback)
                loop = env._obs.observation.game_loop
                if planner is not None:
                    active = await planner.get_plan(payload, env)
                    assert_clock(env, loop)
                    payload = attach_plan(payload, active)
                if pair_runtime is not None:
                    pair_runtime.begin_step(f"ep{episode:04d}", int(loop), env,
                                            reset=(steps == 0))
                if route is not None and route.choice == 'swarm':
                    emit('skill_observation', episode=episode, step=steps, payload=payload)
                    actions = None  # Generate the live Swarm action below, once per beat.
                elif swarm is not None:
                    emit('policy_observation', episode=episode, step=steps, payload=payload)
                    masks = [list(map(int, env.get_avail_agent_actions(i))) for i in range(env.n_agents)]
                    decision = swarm.choose(payload, masks)
                    actions = decision['actions']
                    emit('swarm_decision', episode=episode, step=steps, game_loop=loop,
                         avail_actions=masks, decision=decision)
                elif native_control == "all":
                    emit("policy_observation", episode=episode, step=steps, payload=payload)
                    actions = None  # full actor chooses below, without executing any rules
                elif local_controller is not None:
                    emit('skill_observation', episode=episode, step=steps, payload=payload)
                    actions, telemetry, local_events = local_controller.step(payload, env.n_agents)
                    for row in local_events:
                        row = dict(row)
                        name = row.pop('event')
                        emit(name, episode=episode, **row)
                    emit('skill_step', episode=episode, step=steps, game_loop=loop, **telemetry)
                elif skill_tree_controller is not None:
                    emit('skill_observation', episode=episode, step=steps, payload=payload)
                    actions, telemetry, tree_events = skill_tree_controller.step(payload, env.n_agents, nearest_actions)
                    for row in tree_events:
                        row = dict(row)
                        event_name = row.pop('event')
                        emit(event_name, episode=episode, step=steps, **row)
                    emit('skill_step', episode=episode, step=steps, game_loop=loop, **telemetry)
                    if telemetry.get('termination_reason'):
                        emit('skill_terminated', episode=episode, step=steps, reason=telemetry['termination_reason'])
                elif skill_manager is not None:
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
                    hook = fork or whole
                    step_result = (hook.step(steps, loop, skill_manager, payload, env) if hook is not None
                                   else skill_manager.step(payload, env.n_agents))
                    if shadow is not None:
                        expected = whole.receipt['steps'] + int(squad.handoff) if squad is not None else steps + 1
                        if relay is not None:
                            expected = whole.receipt['steps'] + int(relay.native)
                        shadow.check_step(episode, expected)
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
                    if squad is not None:
                        emit('squad_rendezvous_decision', episode=episode, step=steps, game_loop=loop, **squad.row)
                    skill_event = 'squad_v2_proposal' if squad is not None and not squad.handoff else 'skill_step'
                    if relay is not None:
                        emit('combat_relay_decision', episode=episode, step=steps, game_loop=loop, **relay.relay_row)
                        if relay.owner == 'custom':
                            skill_event = 'relay_r6_proposal'
                    emit(skill_event, episode=episode, step=steps, game_loop=loop, **telemetry)
                    if pair_runtime is not None:
                        pair_runtime.collect_step(episode, steps, int(loop),
                                                  telemetry, actions)
                    if (squad is None or squad.handoff) and (relay is None or relay.native) and telemetry.get("termination_reason"):
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
                if route is not None:
                    masks = [list(map(int, env.get_avail_agent_actions(i))) for i in range(env.n_agents)]
                    row = route.decide(payload, masks, actions)
                    actions = row['actions']
                    if row['reason'] == 'first_macro_difference' and route.choice == 'swarm':
                        # Distill prepared a real pending plan. It is NOT a commit or an engine order.
                        row['uncommitted_v2'] = whole.discard_for_swarm(shadow)
                    emit('swarm_route_decision', episode=episode, step=steps, game_loop=loop, **row)
                use_native = (route is None or route.choice != 'swarm') and (squad is None or squad.handoff)
                if relay is not None:
                    use_native = relay.native
                if native_runtime is not None:
                    actions, native_telemetry = native_runtime.choose(
                        f"ep{episode:04d}", int(loop), env, actions)
                    emit("native_teacher_step", episode=episode, step=steps,
                         game_loop=loop, **native_telemetry)
                    assert_clock(env, loop)
                if not isinstance(actions, list) or len(actions) != env.n_agents:
                    raise RuntimeError("Final actions must be a complete per-unit list.")
                for i, action in enumerate(actions):
                    mask = env.get_avail_agent_actions(i)
                    if not isinstance(action, int) or isinstance(action, bool) or not 0 <= action < len(mask) or not mask[action]:
                        raise RuntimeError(f"Illegal action for u{i}: {action}")
                if use_native and skill_manager is not None and hasattr(skill_manager, "commit_final_actions"):
                    if whole is not None:
                        whole.before_commit(acting_manager, actions)
                    skill_manager.commit_final_actions(actions)
                    if whole is not None:
                        whole.committed(acting_manager, actions)
                    if fork is not None and steps == fork.anchor:
                        fork.committed(acting_manager, actions)
                if medivac_runtime is not None:
                    tree_trace = telemetry.get('race_telemetry', {}).get('medivac_tree')
                    if tree_trace is None:
                        raise RuntimeError('medical tree enabled but no decision trace was produced')
                    decision_rows = []
                    for actor, decision in tree_trace['decisions'].items():
                        row = copy.deepcopy(decision)
                        proposed = row.get('proposed_action', row['action'])
                        final = actions[int(actor[1:])]
                        if medivac_mode == 'tree' and proposed != final:
                            raise RuntimeError(f'direct medical tree was silently overridden for {actor}')
                        row.update(actor=actor,
                                   decision_id=f'{output.name}/{episode}/{loop}/{actor}',
                                   leaf_id=row.get('leaf_id', row.get('leaf')),
                                   proposed_action=proposed, final_action=final,
                                   executed=medivac_mode == 'tree',
                                   override_reason=None if medivac_mode == 'tree' else 'shadow_not_applied')
                        decision_rows.append(row)
                    emit('medivac_tree_step', episode=episode, step=steps, game_loop=loop,
                         mode=medivac_mode, candidate_metadata=medivac_runtime.metadata,
                         decisions=decision_rows)
                health = {u.tag: u.health for u in env.agents.values()}
                shields = {u.tag: u.shield for u in env.agents.values()}
                previous_actions = {f"u{i}": str(action) for i, action in enumerate(actions)}
                if micro_execution:
                    # One macro step == one 8-loop window. The window interior
                    # only re-checks the policy's registered authorizations;
                    # the next command still comes from the next decision.
                    intents = []
                    allow_micro = use_native or (relay is not None and relay.owner == 'custom')
                    if allow_micro and skill_manager is not None and hasattr(skill_manager, "last_micro_intents"):
                        intents = skill_manager.last_micro_intents
                    window = MicroAttackWindow(
                        env, intents, macro_step=steps, issued_loop=int(loop),
                        macro_loops=8, execute=(allow_micro and execution_mode != "micro8-off"),
                        emit=emit, episode=episode, all_attackers=allow_micro and all_attackers, static_data=data)
                    reward, terminated, info = env.step_micro_window(actions, window, macro_loops=8)
                    window.close(env._obs.observation.game_loop)
                    micro_windows += 1
                    micro_records.extend(window.records)
                else:
                    reward, terminated, info = env.step(actions)
                steps += 1
                total_reward += float(reward)
                feedback = action_feedback(env, loop)
                if micro_execution:
                    # Keep macro receipts intact; a later micro order is part of the
                    # next observation's command history, not a second tactical step.
                    submitted = [r for r in window.records if r['status'] == 'submitted']
                    feedback['micro_submissions'] = submitted
                    for r in submitted:
                        if r.get('sc2_response') == [1]:
                            actor = r['actor']
                            cmd = r['submitted_action_id']
                            previous_actions[actor] = str(cmd)
                            if hasattr(skill_manager, 'record_micro_action'):
                                skill_manager.record_micro_action(actor, cmd)
                delta = env._obs.observation.game_loop - loop
                if delta != env._step_mul:
                    raise RuntimeError(f"Unexpected simulation step: {delta}")
                emit("step", episode=episode, step=steps, game_loop=loop,
                     actions=actions, reward=float(reward), terminated=terminated, feedback=feedback)
                if fork is not None and steps == fork.anchor + 1:
                    fork.check_engine(feedback)
                if relay is not None:
                    emit('combat_relay_receipt', episode=episode, step=steps-1, game_loop=loop,
                         **relay.after_engine(acting_manager, feedback, steps-1, loop))
                elif whole is not None and use_native:
                    whole.engine(feedback)
                elif squad is not None:
                    squad.acknowledge_squad(acting_manager, feedback)
                if route is not None and route.after_engine(feedback['micro_submissions']):
                    emit('swarm_route_micro_lock', episode=episode, step=steps-1, game_loop=loop,
                         choice='v2', reason='v2_micro_only', submissions=feedback['micro_submissions'])
                if terminated:
                    break
            if fork is not None:
                fork.check_reached()
            if (local_controller is not None and local_policy['selection']['mode'] == 'forced'
                    and local_controller.interventions != 1):
                raise RuntimeError('Forced local intervention was never reached')
            replay_dir = output / "replays"
            replay_dir.mkdir(exist_ok=True)
            replay = replay_dir / f"episode-{episode:04d}.SC2Replay"
            final_score = score_snapshot(env)
            if damage_enabled:
                # Freeze/read uses SC2 steps; never include those in policy steps or score.
                ticks = read_resource_damage(env._controller)
                if len(ticks) != 8 or any(type(n) is not int or n < 0 for n in ticks):
                    raise ValueError(f'Invalid resource damage ticks: {ticks!r}')
                names = (*RACE_TYPES[race_for_map(config['map_name'])], 'other')
                players = {}
                for i in range(2):
                    counts = dict(zip(names, ticks[4*i:4*i+4]))
                    players[str(i+1)] = {
                        'initial_roster_by_api_tag': initial_rosters[str(i+1)],
                        'ticks_by_source_type': counts,
                        'damage_dealt_by_source_type': {k: v / 4096 for k, v in counts.items()},
                    }
                write_json(output / 'damage_by_type' / f'episode-{episode:04d}.json', {
                    'metric': 'damage dealt to opposing player by damage event source type',
                    'ticks_per_damage': 4096, 'players': players,
                    'episode': episode, 'method': method, 'scenario_sha256': scenario['sha256'],
                    'manifest_sha256': manifest['sha256'], 'game_version': game_version,
                    'map_file': metadata['map_file'],
                    'map_sha256': metadata['source_hashes'][os.path.relpath(selected_map, ROOT.parent)],
                    'begin_request_loop': initial_score['game_loop'],
                    'terminal_loop': final_score['game_loop'],
                    'freeze_loop_range': [final_score['game_loop'], final_score['game_loop'] + 2],
                })
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
            if micro_execution:
                # Additive accounting only: macro steps, rewards and damage
                # above are untouched by the micro window interior.
                states = {}
                for row in micro_records:
                    states[row["status"]] = states.get(row["status"], 0) + 1
                record["micro_trigger"] = {
                    "mode": execution_mode, "macro_loops": 8, "engine_step_loops": 1,
                    "windows": micro_windows, "authorizations": len(micro_records),
                    "states": states}
                write_json(output / 'micro_triggers' / f'episode-{episode:04d}.json', {
                    'episode': episode, 'mode': execution_mode,
                    'macro_loops': 8, 'engine_step_loops': 1,
                    'authorizations': micro_records})
            write_json(output / 'score_details' / f'episode-{episode:04d}.json', {
                'scope': 'controlled player cumulative engine counters, not per unit type',
                'episode': episode, 'method': method, 'map': config['map_name'],
                'sc2_seed': scenario['sc2_seed'], 'scenario_sha256': scenario['sha256'],
                'manifest_sha256': manifest['sha256'], 'game_version': game_version,
                'steps': steps, 'outcome': record['outcome'],
                'initial': initial_score, 'final': final_score,
            })
            v2_steps = whole.receipt['steps'] if route is not None or squad is not None else steps
            if squad is not None:
                record['squad_rendezvous'] = dict(v2_steps=v2_steps, squad_steps=squad.squad_steps)
            if relay is not None:
                record['squad_rendezvous']['v2_steps'] = relay.counts['r6_steps']
                record['combat_relay'] = dict(relay.counts)
            if route is not None:
                record['swarm_route'] = dict(choice=route.choice, v2_steps=v2_steps, swarm_steps=steps-v2_steps)
            if shadow is not None:
                record['zerg_shadow_decisions'] = shadow.finish(dict(record, steps=v2_steps), initial_score, final_score)
            if whole is not None:
                if route is not None and route.choice == 'swarm':
                    whole.finish_swarm_prefix(v2_steps)
                else:
                    whole.finish(steps)
                if distill_path is not None:
                    record['zerg_distill'] = whole.receipt['stats']
            records.append(record)
            emit("episode_completed", **record)
            write_json(output / "episodes.json", records)
            print(f"{method}: episode {episode}/{len(manifest['episodes'])}: "
                  f"{record['outcome']} ({steps} steps)", flush=True)
            env.close()
            env = None
    except BaseException as exc:
        failure = f"{type(exc).__name__}: {exc}"
        if whole is not None:
            whole.fail(failure)
        emit("run_failed", error=failure, completed_episodes=len(records))
        raise
    finally:
        if native_runtime is not None:
            native_runtime.close()
        if pair_runtime is not None:
            pair_runtime.close()
            if pair_runtime.collector is not None:
                pair_runtime.collector.close()
        if env is not None:
            env.close()
        if client is not None:
            await client.close()
        if planner is not None:
            await planner.close()
        micro_totals = None
        if micro_execution:
            states, windows, authorizations = {}, 0, 0
            for r in records:
                block = r.get("micro_trigger") or {}
                windows += block.get("windows", 0)
                authorizations += block.get("authorizations", 0)
                for state, count in (block.get("states") or {}).items():
                    states[state] = states.get(state, 0) + count
            micro_totals = {"mode": execution_mode, "macro_loops": 8,
                            "engine_step_loops": 1, "windows": windows,
                            "authorizations": authorizations, "states": states}
        summary = {
            "method": method, "status": "failed" if failure else "complete", "error": failure,
            "execution_mode": execution_mode, "micro_trigger_totals": micro_totals,
            "native_teacher_control": native_control,
            "native_teacher_metadata": native_runtime.describe() if native_runtime is not None else None,
            "pair_mode": getattr(args, "pair_mode", "off"),
            "pair_failure_policy": getattr(args, "pair_failure", "strict"),
            "pair_metadata": (pair_runtime.describe(
                checkpoint=getattr(args, "pair_teacher_checkpoint", None),
                model_dir=getattr(args, "pair_model_dir", None),
                last_token=getattr(args, "pair_last_token", "off"))
                if pair_runtime is not None else None),
            "skill_version": skill_version,
            "skill_source": metadata["skill_source"],
            "skill_source_hashes": skill_hashes,
            **value_metadata,
            "skill_parameters": metadata["skill_parameters"],
            "manifest_sha256": manifest["sha256"], "completed_episodes": len(records),
            "expected_episodes": len(manifest["episodes"]),
            **({'zerg_shadow': {**metadata['zerg_shadow'], 'stats': shadow.stats}} if shadow is not None else {}),
            **({'zerg_distill': {**whole.contract, 'stats': whole.totals}} if distill_path is not None else {}),
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
    run.add_argument("--method", choices=("random", "nearest", "jev", "astra-jev", "adaptive-skill", "radical-swarm"), default="nearest")
    run.add_argument("--execution-mode", choices=EXECUTION_MODES,
                     default="legacy8",
                     help="legacy8 (default): one env.step per tactical decision, unchanged behavior. "
                          "micro8-*: one tactical decision per 8 game loops. off only records hydra "
                          "triggers; hydra or all may send one authorized Attack per eligible actor. "
                          "all also checks native reach for non-hydralisk attackers. Requires adaptive-skill, "
                          "tactical_v1 and a step_mul=8 manifest.")
    suite = sub.add_parser("suite", help="Paired methods on one immutable scenario manifest")
    suite.add_argument("--methods", nargs="+", choices=("random", "nearest", "jev", "astra-jev", "adaptive-skill"),
                       default=["nearest", "jev", "astra-jev"])
    suite.add_argument("--execution-mode", choices=EXECUTION_MODES,
                       default="legacy8",
                       help="Same contract as run --execution-mode; applies to every method.")
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
                             help="Skill implementation; baseline_v2_70pct requires its original fixed preset; "
                                  "tactical_v1 is the joint 7-generic+3-specialized Adaptive controller")
        command.add_argument("--tactical-config", type=Path, default=None,
                             help="tactical_v1 only: strict-JSON override of skills/tactical_v1/defaults.json "
                                  "(per-skill ablation switches, budgets, scoring weights)")
        command.add_argument("--race-profile", choices=("off", "terran", "zerg"), default="off",
                             help="Opt-in race-specific adaptive combat; current skills only")
        command.add_argument('--local-policy', type=Path, default=None,
                             help='Local single-actor patch gain/veto policy; Protoss current baseline')
        command.add_argument('--skill-tree', type=Path, default=None,
                             help='Generic skill-call policy JSON; current local skills only')
        command.add_argument('--terran-keep-heal-order', action='store_true',
                             help='Native current Terran only: append an action that submits no command for an eligible observed heal order')
        command.add_argument('--medivac-policy', choices=('legacy', 'tree-shadow', 'tree'), default='legacy',
                             help='Shared direct-action medical tree; legacy preserves the original controller')
        command.add_argument('--medivac-tree', type=Path, default=None,
                             help='Validated action tree or paired-return policy JSON (defaults to medivac_tree/tree_v0.json)')
        command.add_argument('--medical-patient-coordination', choices=('on', 'off'), default=None,
                             help='Medical-driven ground movement: defaults on for legacy, off for active tree')
        command.add_argument('--pair-mode', choices=('off','a-plug','shadow','teacher-score','tree-score'), default='off',
                             help='Default-off medivac pair-preference channel (tree_skills); current+terran adaptive only')
        command.add_argument('--native-teacher-control', choices=('off', 'medivac', 'all'), default='off',
                             help='Frozen native-action teacher diagnostic; bypass healing gates for controlled units')
        command.add_argument('--native-teacher-checkpoint', default=None)
        command.add_argument('--native-teacher-last-token', choices=('on', 'off'), default='on')
        command.add_argument('--pair-model-dir', default=None,
                             help='Exported/native tree model dir for --pair-mode tree-score/shadow')
        command.add_argument('--pair-teacher-checkpoint', default=None,
                             help='Teacher .pt for --pair-mode teacher-score/shadow (default: first audited checkpoint)')
        command.add_argument('--pair-last-token', choices=('on','off'), default='off',
                             help='Teacher input variant; must match the chosen checkpoint convention')
        command.add_argument('--pair-failure', choices=('strict','tolerant'), default='strict',
                             help='strict: scorer failure marks the episode invalid; tolerant: fall back with per-step accounting')
        command.add_argument('--pair-shadow-of', choices=('teacher-score','tree-score'), default=None,
                             help='Which model to shadow when --pair-mode shadow')
        command.add_argument('--pair-collect', default=None,
                             help='Optional directory for real per-step pair collection JSONL (never overwrites)')
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
        if methods == ['radical-swarm']:
            if args.execution_mode != 'micro8-off':
                raise SystemExit('radical-swarm requires --execution-mode micro8-off (no interior orders)')
        elif getattr(args, "execution_mode", "legacy8") != "legacy8":
            if any(m != "adaptive-skill" for m in methods):
                raise SystemExit("--execution-mode micro8-* requires --method adaptive-skill only")
            if args.skill_version != "tactical_v1":
                raise SystemExit("--execution-mode micro8-* requires --skill-version tactical_v1")
        if args.local_policy is not None and any(m != 'adaptive-skill' for m in methods):
            raise SystemExit('--local-policy requires adaptive-skill only')
        if args.skill_tree is not None and any(m != 'adaptive-skill' for m in methods):
            raise SystemExit('--skill-tree requires adaptive-skill only')
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
