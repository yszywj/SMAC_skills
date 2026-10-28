"""SMACv2 backend worker: JSONL stdio protocol, real env + real SkillManager.

Standalone script (run with the ORIGINAL backend interpreter that has
pysc2/smacv2/numpy/yaml). It never imports the smac_langchain package, so it
cannot accidentally depend on LangChain or Pydantic 2.

Protocol: one JSON object per line on stdin/stdout (UTF-8). stdout carries
protocol responses only; diagnostics go to stderr. Lifecycle ops (start/close)
are driven by the host, never by the model.
"""
from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path

# Locate the workspace: this file lives at langchain_agent/workers/.
PKG_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = PKG_ROOT.parent
sys.path.insert(0, str(WORKSPACE))

from sc2_skill_adapter.bridge import config as bridge_config  # noqa: E402
from sc2_skill_adapter.bridge.environment import ReplayableEnv  # noqa: E402
from sc2_skill_adapter.bridge.feedback import action_feedback  # noqa: E402
from sc2_skill_adapter.bridge.policy import build_request, nearest_actions  # noqa: E402
from sc2_skill_adapter.bridge import skill_runtime  # noqa: E402

SKILLS_PACKAGE = "_smac_langchain_skills"
_manager = None
_policy = None
_env = None
_data = None
_episode_id = None
_skill_version = "current"
_episode_steps = 0
_active_skill = None
_last_termination = None


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _skill_policy():
    global _policy
    if _policy is None:
        _policy = skill_runtime._load_module("skill_policy", version=_skill_version)
    return _policy


def _respond(request_id, ok, payload=None, code=None, message=None):
    response = {"id": request_id, "ok": ok}
    if ok:
        response["result"] = payload or {}
    else:
        response["error"] = {"code": code or "error", "message": message or ""}
    sys.stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _spatial_enabled():
    if _manager is None:
        return []
    params = getattr(_manager, "spatial_params", {}) or {}
    return [name for name in ("regroup", "cohesive_search")
            if params.get("enable_" + name)]


def _snapshot_dict():
    payload = build_request(_env, _data, "codex-agent", _episode_id,
                            _previous_health, _previous_actions, _previous_shields,
                            _last_feedback)
    state = payload["state"]
    visible = state["visible_enemies"]
    policy = _skill_policy()
    spatial_modes = tuple(_spatial_enabled())
    criteria, priorities, modes = policy.skill_choices(
        visible, healers_present=False, spatial_modes=spatial_modes)
    # Model-visible summary: reuse the audited team-level request builder.
    skill_state = policy.skill_request(
        payload, "codex-agent", _manager.max_steps if _manager else 4,
        spatial_modes=spatial_modes)["state"]
    from smacv2.env.starcraft2.starcraft2 import StarCraft2Env  # noqa: F401
    done = _env.env._episode_steps >= _env.env.episode_limit \
        or _env.env.death_tracker_ally is not None and (
            _env.env.death_tracker_ally.sum() == _env.env.n_agents
            or _env.env.death_tracker_enemy.sum() == _env.env.n_enemies)
    return {"run_hint": "smacv2", "episode_id": _episode_id,
            "game_loop": int(state["game_loop"]), "n_agents": _env.env.n_agents,
            "done": bool(done), "state": skill_state,
            "candidates": {cid: {"skill": modes[cid], "priority": priorities[cid],
                                 "text": criteria[cid]} for cid in criteria},
            "spatial_available": list(spatial_modes),
            "active_skill": _manager.current_skill if _manager else None,
            "last_termination": _last_termination,
            "full_state": state,  # host-side filtering decides visibility
            "simulated": False}


def op_start(request):
    global _env, _data, _manager, _episode_id, _skill_version
    global _previous_health, _previous_actions, _previous_shields, _last_feedback
    global _episode_steps, _active_skill, _last_termination
    _skill_version = request.get("skill_version", "current")
    size = int(request.get("size", 5))
    race = request.get("race", "protoss")
    seed = int(request.get("seed", 1))
    manager_kwargs = request["manager_kwargs"]
    env_args = bridge_config.load_config(size, race)
    env_args.update(seed=seed)
    os.environ["SC2PATH"] = request.get(
        "sc2_path", os.environ.get("SC2PATH", r"C:\Program Files (x86)\StarCraft II"))
    _manager = skill_runtime.load_manager(_skill_version)(**manager_kwargs)
    _env = ReplayableEnv(copy.deepcopy(env_args))
    record = {"capability_config": env_args["capability_config"], "sc2_seed": seed,
              "distribution_config": None}
    # Reset with the sampled seed (manifest-free learning mode).
    _env.reset_record({"capability_config": env_args["capability_config"],
                       "sc2_seed": seed})
    _env.env.scenario_brief = bridge_config.public_brief(env_args)
    _data = _env._controller.data()
    _episode_id = int(request.get("episode_id", 1))
    _manager.reset()
    _previous_health, _previous_actions, _previous_shields = {}, {}, {}
    _last_feedback = None
    _episode_steps = 0
    _active_skill, _last_termination = None, None
    return {"n_agents": _env.env.n_agents, "game_loop": 0,
            "episode_id": _episode_id, "map": env_args["map_name"]}


def op_execute(request):
    global _previous_health, _previous_actions, _previous_shields, _last_feedback
    global _episode_steps, _active_skill, _last_termination
    if _env is None:
        raise RuntimeError("not_started")
    choice_id = str(request["choice_id"])
    decision_steps = int(request["decision_steps"])
    posture = request.get("posture")
    snapshot = _snapshot_dict()
    if snapshot["done"]:
        return {"accepted": False, "outcome": "episode_over", "steps_executed": 0,
                "start_game_loop": snapshot["game_loop"],
                "end_game_loop": snapshot["game_loop"],
                "stop_reason": "episode_already_over",
                "state": snapshot["state"], "candidates": snapshot["candidates"],
                "done": True, "simulated": False}
    candidate = snapshot["candidates"].get(choice_id)
    if candidate is None:
        raise RuntimeError(f"unknown_choice:{choice_id}")
    skill, priority = candidate["skill"], candidate["priority"]
    if skill == "direct_control":
        return {"accepted": False, "outcome": "rejected", "steps_executed": 0,
                "start_game_loop": snapshot["game_loop"],
                "end_game_loop": snapshot["game_loop"],
                "stop_reason": "direct_control_not_exposed",
                "state": snapshot["state"], "candidates": snapshot["candidates"],
                "done": False, "simulated": False}
    if skill in ("regroup", "cohesive_search") \
            and skill not in snapshot["spatial_available"]:
        return {"accepted": False, "outcome": "rejected", "steps_executed": 0,
                "start_game_loop": snapshot["game_loop"],
                "end_game_loop": snapshot["game_loop"],
                "stop_reason": skill + "_not_enabled",
                "state": snapshot["state"], "candidates": snapshot["candidates"],
                "done": False, "simulated": False}
    same_mode = _manager.current_skill == skill
    posture_change = posture and skill == "adaptive_focus_fire" \
        and posture != _manager.posture
    if not same_mode or posture_change or not _manager.active:
        _manager.select(skill,
                        priority_target=None if skill == "adaptive_focus_fire" else priority,
                        posture=posture)
        selection_source = "new_selection"
    else:
        selection_source = "continued_mode"
    executed, outcome, stop_reason, handoff = 0, "accepted", None, None
    start_loop = snapshot["game_loop"]
    last_telemetry = {}
    while executed < decision_steps:
        payload = build_request(_env, _data, "codex-agent", _episode_id,
                                _previous_health, _previous_actions,
                                _previous_shields, _last_feedback)
        loop = payload["state"]["game_loop"]
        step_result = _manager.step(payload, _env.env.n_agents)
        if isinstance(step_result[0], dict):
            partial, telemetry = step_result
            if telemetry.get("status") == "interrupted" and not partial:
                _manager.select("adaptive_focus_fire", posture=posture or "balanced")
                handoff = "spatial_interrupt_handoff"
                actions, telemetry = _manager.step(payload, _env.env.n_agents)
            else:
                actions = nearest_actions(payload, _env.env.n_agents)
                for index, action in (partial or {}).items():
                    actions[int(index)] = action
                telemetry["base_policy"] = "nearest"
                telemetry["merged_units"] = sorted(partial or {})
        else:
            actions, telemetry = step_result
        if not isinstance(actions, list) or len(actions) != _env.env.n_agents:
            raise RuntimeError("action_shape_invalid")
        for i, action in enumerate(actions):
            if not _env.env.get_avail_agent_actions(i)[int(action)]:
                raise RuntimeError(f"illegal_action u{i}:{action}")
        _manager.commit_final_actions(actions)
        _previous_health = {u.tag: u.health for u in _env.env.agents.values()}
        _previous_shields = {u.tag: u.shield for u in _env.env.agents.values()}
        _previous_actions = {f"u{i}": str(a) for i, a in enumerate(actions)}
        reward, terminated, info = _env.env.step(actions)
        _episode_steps += 1
        executed += 1
        _last_feedback = action_feedback(_env, loop)
        delta = _env.env._obs.observation.game_loop - loop
        if delta != _env.env._step_mul:
            raise RuntimeError(f"unexpected_simulation_step:{delta}")
        last_telemetry = telemetry
        _active_skill = _manager.current_skill
        if telemetry.get("termination_reason"):
            _last_termination = telemetry["termination_reason"]
            if telemetry["termination_reason"] == "max_steps":
                # routine skill-window end: renew the same mode (adaptive
                # history survives) instead of ending the agent window
                _manager.select(skill,
                                priority_target=None
                                if skill == "adaptive_focus_fire" else priority,
                                posture=posture)
            else:
                outcome = "skill_complete"
                stop_reason = _last_termination
                break
        if terminated:
            outcome = "episode_over"
            stop_reason = "win" if info.get("battle_won") else (
                "loss" if info.get("dead_allies") == _env.env.n_agents else "timeout")
            break
    else:
        stop_reason = stop_reason or "decision_window_exhausted"
    fresh = _snapshot_dict()
    return {"accepted": True, "outcome": outcome, "steps_executed": executed,
            "start_game_loop": start_loop, "end_game_loop": fresh["game_loop"],
            "stop_reason": stop_reason, "handoff": handoff,
            "selection_source": selection_source,
            "telemetry": {"skill": _manager.current_skill,
                          "last_step": last_telemetry},
            "state": fresh["state"], "candidates": fresh["candidates"],
            "done": fresh["done"], "simulated": False}


def op_episode_status(_request):
    if _env is None:
        raise RuntimeError("not_started")
    info = {"battle_won": False}
    done = _env.env._episode_steps >= _env.env.episode_limit
    return {"done": done, "steps": _episode_steps,
            "game_loop": int(_env.env._obs.observation.game_loop),
            "manager_stats": copy.deepcopy(_manager.stats),
            "simulated": False}


def op_close(_request):
    global _env
    if _env is not None:
        try:
            _env.close()
        finally:
            _env = None
    return {}


OPS = {"ping": lambda _r: {"ready": True, "pid": os.getpid()},
       "start": op_start, "snapshot": lambda r: _snapshot_dict(),
       "execute": op_execute, "episode_status": op_episode_status,
       "close": op_close}


def main() -> int:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except ValueError:
            _respond("malformed", False, code="bad_json", message=line[:200])
            continue
        handler = OPS.get(request.get("op"))
        if handler is None:
            _respond(request.get("id"), False, code="unknown_op",
                     message=str(request.get("op")))
            continue
        try:
            _respond(request.get("id"), True, handler(request))
        except Exception as exc:  # noqa: BLE001 - protocol boundary
            log(f"worker error op={request.get('op')}: {type(exc).__name__}: {exc}")
            _respond(request.get("id"), False, code="worker_error", message=str(exc))
    return 0


if __name__ == "__main__":
    sys.exit(main())
