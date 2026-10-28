"""Offline mock backend: synthetic battle driven by the REAL SkillManager.

This exercises the genuine skill lifecycle (select/step/commit/termination) and
the real candidate tables from skills/current, against a tiny simulated
engagement so no SC2 process and no model are needed. Every snapshot it
produces is labelled simulated=true and is never battle evidence.

Loads the skill package the same way sc2_skill_adapter/bridge/skill_runtime.py
does (portable package, stdlib-only), without importing PySC2.
"""
from __future__ import annotations

import copy
import sys
import types
import importlib
from pathlib import Path
from typing import Any

from ..config import resolve_path
from .protocol import Backend, BackendError


def load_skill_package(skills_dir: str | Path):
    skills_dir = Path(skills_dir)
    name = "_smac_langchain_skills"
    if name not in sys.modules:
        package = types.ModuleType(name)
        package.__path__ = [str(skills_dir)]
        sys.modules[name] = package
    return name


def _skill_module(skills_dir, module: str):
    name = load_skill_package(skills_dir)
    return importlib.import_module(f"{name}.{module}")


class MockBattle:
    """Minimal but internally consistent 3v3 engagement state."""

    STEP_MUL = 8
    EPISODE_LIMIT = 200

    def __init__(self, seed: int = 1, n_agents: int = 3):
        self.game_loop = 0
        self.steps = 0
        self.n_agents = n_agents
        rng = _DeterministicRandom(seed)
        self.allies = {f"u{i}": self._unit(rng, 80.0, (10.0 + i, 10.0), "stalker")
                       for i in range(n_agents)}
        self.enemies = {f"e{j}": self._unit(rng, 70.0, (13.0, 13.0), "zergling")
                        for j in range(n_agents)}
        self.previous_health = {a: u["health"] for a, u in self.allies.items()}
        self.previous_shields = {a: u["shield"] for a, u in self.allies.items()}

    @staticmethod
    def _unit(rng, health, position, type_name):
        unit_type_id = {"stalker": 74, "zergling": 105, "marine": 48}.get(type_name, 74)
        return {"type": type_name, "unit_type_id": unit_type_id, "alive": True,
                "health": health,
                "max_health": health, "shield": 0.0, "max_shield": 0.0,
                "effective_health": health, "position": list(position),
                "radius": 0.5, "is_flying": False, "energy": 200.0,
                "max_energy": 200.0, "weapon_cooldown_loops": 0.0,
                "weapon_ready": True, "orders": [],
                "health_lost_since_previous_decision": 0.0,
                "shield_lost_since_previous_decision": 0.0}

    def visible_enemies(self):
        # mock: full sight within 12 world units, mirroring team vision idea
        return {e: u for e, u in self.enemies.items()
                if u["alive"] and _dist(u["position"], (11.0, 11.0)) < 20}

    def avail_actions(self, actor_index: int) -> list[int]:
        mask = [1] * (6 + len(self.enemies))
        for j, enemy in self.enemies.items():
            if not enemy["alive"]:
                mask[6 + int(j[1:])] = 0
        return mask

    def state_payload(self, episode_id: int) -> dict[str, Any]:
        visible = self.visible_enemies()
        allies = {a: dict(u) for a, u in self.allies.items()}
        for a, u in allies.items():
            u["health_lost_since_previous_decision"] = round(
                max(0.0, self.previous_health[a] - u["health"]), 2)
            u["shield_lost_since_previous_decision"] = round(
                max(0.0, self.previous_shields[a] - u["shield"]), 2)
        enemies = {e: {"type": u["type"], "unit_type_id": u["unit_type_id"],
                       "is_flying": False, "alive": u["alive"],
                       "health": u["health"], "shield": u["shield"],
                       "effective_health": round(u["health"] + u["shield"], 2),
                       "position": list(u["position"]), "alive": u["alive"]}
                   for e, u in visible.items()}
        relations = {}
        for a, u in self.allies.items():
            row = {}
            for e, enemy in visible.items():
                d = _dist(u["position"], enemy["position"])
                row[e] = {"distance": round(d, 2), "in_weapon_range": d <= 6.0,
                          "weapon_range_status": "in_range" if d <= 6 else "out_of_range",
                          "native_weapon_range": 6.0,
                          "in_enemy_weapon_range": d <= 3.0,
                          "enemy_weapon_range_status": "in_range" if d <= 3 else "out_of_range",
                          "smac_attack_available": enemy["alive"] and d <= 6.0,
                          "smac_attack_distance_limit": 6,
                          "smac_center_distance_allowed": d <= 6.0,
                          "approach_needed": d > 6.0}
            relations[a] = row
        return {
            "objective": "Eliminate all opposing units before the step limit (SIMULATED mock battle).",
            "map": "mock_arena", "episode": episode_id, "game_loop": self.game_loop,
            "step_mul": self.STEP_MUL,
            "steps_remaining": self.EPISODE_LIMIT - self.steps,
            "allies": allies, "visible_enemies": enemies,
            "relations": relations, "movement_candidates": {}, "heal_candidates": {},
            "unit_type_profiles": {
                "74": {"known": True, "armor": 1,
                       "weapons": [{"target_type": "any", "range": 6,
                                    "base_damage": 13, "hits_per_attack": 1}]},
                "105": {"known": True, "armor": 0,
                        "weapons": [{"target_type": "ground", "range": 3,
                                     "base_damage": 5, "hits_per_attack": 1}]},
            },
            "search_location": [16.0, 16.0],
        }

    def step(self, actions: list[int]):
        for j, enemy in self.enemies.items():
            if not enemy["alive"]:
                continue
            # enemy fire: nearest living ally takes damage (keeps retreat logic exercised)
            living = [(a, u) for a, u in self.allies.items() if u["alive"]]
            if living:
                a, u = min(living, key=lambda item: _dist(item[1]["position"], enemy["position"]))
                u["health"] = round(max(0.0, u["health"] - 4.0), 2)
                if u["health"] <= 0:
                    u["alive"] = False
        for i, action in enumerate(actions):
            actor, unit = f"u{i}", self.allies[f"u{i}"]
            if not unit["alive"]:
                if action != 0:
                    raise BackendError("mock_illegal_action_for_dead_unit", f"u{i}:{action}")
                continue
            if action == 1:
                unit["orders"] = []
            elif action in (2, 3, 4, 5):
                dx, dy = {2: (0, 1), 3: (0, -1), 4: (1, 0), 5: (-1, 0)}[action]
                unit["position"] = [round(unit["position"][0] + dx * 0.5, 3),
                                    round(unit["position"][1] + dy * 0.5, 3)]
            elif action >= 6:
                target = self.enemies.get(f"e{action - 6}")
                if target is None or not target["alive"]:
                    raise BackendError("mock_illegal_attack", str(action))
                visible = self.visible_enemies()
                relation_dist = _dist(unit["position"], target["position"])
                if f"e{action-6}" not in visible or relation_dist > 6.0:
                    raise BackendError("mock_attack_out_of_range", str(action))
                target["health"] = round(max(0.0, target["health"] - 9.0), 2)
                if target["health"] <= 0:
                    target["alive"] = False
        self.previous_health = {a: u["health"] for a, u in self.allies.items()}
        self.previous_shields = {a: u["shield"] for a, u in self.allies.items()}
        self.game_loop += self.STEP_MUL
        self.steps += 1
        return self.done()

    def done(self):
        allies_alive = any(u["alive"] for u in self.allies.values())
        enemies_alive = any(u["alive"] for u in self.enemies.values())
        if not enemies_alive:
            return True, "win"
        if not allies_alive:
            return True, "loss"
        if self.steps >= self.EPISODE_LIMIT:
            return True, "timeout"
        return False, None


def _dist(a, b):
    import math
    return math.hypot(a[0] - b[0], a[1] - b[1])


class _DeterministicRandom:
    def __init__(self, seed):
        self.state = seed or 1

    def random(self):
        self.state = (1103515245 * self.state + 12345) % (2 ** 31)
        return self.state / 2 ** 31


class MockBackend(Backend):
    backend_name = "mock"

    def __init__(self, skills_dir: str | Path | None = None):
        self.skills_dir = Path(skills_dir) if skills_dir else resolve_path("skills/current")
        policy = _skill_module(self.skills_dir, "skill_policy")
        manager_mod = _skill_module(self.skills_dir, "skill_manager")
        self.skill_policy = policy
        self.SkillManager = manager_mod.SkillManager
        self.manager = None
        self.battle: MockBattle | None = None
        self.episode_id = None
        self.active_skill = None
        self.last_termination = None
        self.spatial_enabled = {"regroup": False, "cohesive_search": False}
        self.manager_kwargs = {}

    def start(self, episode_id, seed, manager_kwargs, skill_version="current"):
        if skill_version != "current":
            raise BackendError("mock_unsupported_skill_version", skill_version)
        self.manager_kwargs = dict(manager_kwargs)
        spatial = dict(manager_kwargs.get("spatial_params") or {})
        self.spatial_enabled = {
            "regroup": bool(spatial.get("enable_regroup")),
            "cohesive_search": bool(spatial.get("enable_cohesive_search"))}
        kwargs = {k: v for k, v in manager_kwargs.items() if k != "spatial_params"}
        self.manager = self.SkillManager(**kwargs)
        self.battle = MockBattle(seed=seed)
        self.episode_id = episode_id
        self.active_skill = None
        self.last_termination = None
        return {"n_agents": self.battle.n_agents, "game_loop": 0,
                "episode_id": episode_id, "simulated": True}

    def _payload(self):
        return {"model": "mock", "state": self.battle.state_payload(self.episode_id)}

    def snapshot(self):
        if self.battle is None:
            raise BackendError("mock_not_started", "")
        payload = self._payload()
        visible = payload["state"]["visible_enemies"]
        spatial_modes = tuple(m for m, on in self.spatial_enabled.items() if on)
        criteria, priorities, modes = self.skill_policy.skill_choices(
            visible, healers_present=False, spatial_modes=spatial_modes)
        candidates = {cid: {"skill": modes[cid], "priority": priorities[cid],
                            "text": criteria[cid]} for cid in criteria}
        done, outcome = self.battle.done()
        return {"run_hint": "mock", "episode_id": self.episode_id,
                "game_loop": self.battle.game_loop,
                "n_agents": self.battle.n_agents, "done": done,
                "state": payload["state"], "candidates": candidates,
                "spatial_available": [m for m, on in self.spatial_enabled.items() if on],
                "active_skill": self.manager.current_skill if self.manager else None,
                "last_termination": self.last_termination, "simulated": True}

    def execute(self, choice_id, decision_steps, posture=None):
        if self.battle is None:
            raise BackendError("mock_not_started", "")
        snap = self.snapshot()
        if snap["done"]:
            return self._execution(snap, accepted=False, outcome="episode_over",
                                   stop_reason="episode_already_over")
        candidate = snap["candidates"].get(str(choice_id))
        if candidate is None:
            raise BackendError("mock_unknown_choice", str(choice_id))
        skill, priority = candidate["skill"], candidate["priority"]
        if skill == "direct_control":
            return self._execution(snap, accepted=False, outcome="rejected",
                                   stop_reason="direct_control_not_exposed")
        if skill in ("regroup", "cohesive_search") and skill not in snap["spatial_available"]:
            return self._execution(snap, accepted=False, outcome="rejected",
                                   stop_reason=f"{skill}_not_enabled")
        # Selection policy: reselect on mode/posture change or when inactive.
        same_mode = self.manager.current_skill == skill
        posture_change = posture and skill == "adaptive_focus_fire" \
            and posture != self.manager.posture
        if not same_mode or posture_change or not self.manager.active:
            self.manager.select(skill, priority_target=None
                                if skill == "adaptive_focus_fire" else priority,
                                posture=posture)
            selection_source = "new_selection"
        else:
            selection_source = "continued_mode"
        executed, outcome, stop_reason, handoff = 0, "accepted", None, None
        start_loop = self.battle.game_loop
        while executed < decision_steps:
            payload = self._payload()
            step_result = self.manager.step(payload, self.battle.n_agents)
            partial = None
            if isinstance(step_result[0], dict):
                partial, telemetry = step_result
                if telemetry.get("status") == "interrupted" and not partial:
                    # same-snapshot emergency handoff to adaptive (local rule)
                    self.manager.select("adaptive_focus_fire", posture=posture or "balanced")
                    handoff = "spatial_interrupt_handoff"
                    actions, telemetry = self.manager.step(payload, self.battle.n_agents)
                else:
                    actions = [1] * self.battle.n_agents  # declared base policy: stop
                    for index, action in partial.items():
                        actions[index] = action
                    telemetry["base_policy"] = "stop"
            else:
                actions, telemetry = step_result
            self._validate(actions)
            self.manager.commit_final_actions(actions)
            done, ep_outcome = self.battle.step(actions)
            executed += 1
            self.active_skill = self.manager.current_skill
            if telemetry.get("termination_reason"):
                self.last_termination = telemetry["termination_reason"]
                if telemetry["termination_reason"] == "max_steps":
                    # routine skill-window end: renew the same mode (keeps
                    # adaptive history) instead of ending the agent window
                    self.manager.select(skill, priority_target=None
                                        if skill == "adaptive_focus_fire" else priority,
                                        posture=posture)
                else:
                    outcome = "skill_complete"
                    stop_reason = self.last_termination
                    break
            if done:
                outcome = "episode_over"
                stop_reason = ep_outcome
                break
        else:
            stop_reason = stop_reason or "decision_window_exhausted"
        fresh = self.snapshot()
        return {"accepted": True, "outcome": outcome, "steps_executed": executed,
                "start_game_loop": start_loop, "end_game_loop": self.battle.game_loop,
                "stop_reason": stop_reason, "handoff": handoff,
                "selection_source": selection_source,
                "telemetry": {"skill": self.manager.current_skill},
                "state": fresh["state"], "candidates": fresh["candidates"],
                "done": fresh["done"], "simulated": True}

    def _validate(self, actions):
        if not isinstance(actions, list) or len(actions) != self.battle.n_agents:
            raise BackendError("mock_action_shape", str(actions))
        for i, action in enumerate(actions):
            if not self.battle.avail_actions(i)[int(action)]:
                raise BackendError("mock_illegal_action", f"u{i}:{action}")

    def _execution(self, snap, **fields):
        return {"accepted": fields.get("accepted", False),
                "outcome": fields.get("outcome", "rejected"),
                "steps_executed": 0, "start_game_loop": snap["game_loop"],
                "end_game_loop": snap["game_loop"], "stop_reason": fields.get("stop_reason"),
                "state": snap["state"], "candidates": snap["candidates"],
                "done": snap["done"], "simulated": True}

    def episode_status(self):
        if self.battle is None:
            raise BackendError("mock_not_started", "")
        done, outcome = self.battle.done()
        return {"done": done, "outcome": outcome, "steps": self.battle.steps,
                "game_loop": self.battle.game_loop,
                "manager_stats": copy.deepcopy(self.manager.stats) if self.manager else None,
                "simulated": True}

    def close(self):
        self.battle = None
        self.manager = None
