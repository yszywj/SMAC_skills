"""BattleCoordinator: validation, budgets, single-writer, idempotency.

Everything that writes to the environment goes through here, serially. The
agent tools are thin wrappers around this class. Responsibilities:

- freeze snapshot-bound candidate maps (DecisionContext) locally;
- reject stale snapshots, cross-episode results, changed candidates;
- locally generated command ids keyed to tool_call_id with idempotent replay
  (same id + same params -> cached report; same id + different params -> reject);
- enforce per-episode budgets (execute commands, steps, read tools);
- translate backend results into typed ExecutionReports, distinguishing
  accepted / generated / submitted / interrupted / timeout / skill_complete /
  episode_over / failed / unknown, and recording fallbacks.
"""
from __future__ import annotations

import hashlib
import json
import time
import uuid
from typing import Any

from ..schemas import (BattleSnapshot, DecisionContext, ExecutionReport,
                       SkillCandidate, SkillExecutionRequest)
from ..backends.protocol import Backend, BackendError
from .registry import candidate_enabled, display_name, posture_for


class CoordinatorError(RuntimeError):
    def __init__(self, code: str, message: str = ""):
        super().__init__(f"{code}: {message}")
        self.code = code


class Budget:
    def __init__(self, limits: dict[str, Any]):
        self.limits = dict(limits)
        self.spent = {key: 0 for key in limits}

    def charge(self, key: str, amount: int = 1) -> None:
        if key not in self.spent:
            self.spent[key] = 0
        self.spent[key] += amount
        limit = self.limits.get(key)
        if limit is not None and self.spent[key] > limit:
            raise CoordinatorError(f"budget_exceeded:{key}",
                                   f"{key}={self.spent[key]} > {limit}")

    def check(self, key: str, amount: int = 1) -> None:
        limit = self.limits.get(key)
        if limit is not None and self.spent.get(key, 0) + amount > limit:
            raise CoordinatorError(f"budget_exceeded:{key}",
                                   f"{key} would exceed {limit}")


def _candidates_version(candidates: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(candidates, sort_keys=True,
                                     ensure_ascii=False).encode()).hexdigest()[:16]


class BattleCoordinator:
    def __init__(self, run_id: str, episode_id: int, backend: Backend,
                 config: dict[str, Any], emit=None, codex_stats: dict | None = None,
                 clock_started: float | None = None):
        self.run_id = run_id
        self.episode_id = episode_id
        self.backend = backend
        self.emit = emit or (lambda *_a, **_k: None)
        self.codex_stats = codex_stats if codex_stats is not None else {}
        agent_cfg = config.get("agent", {})
        skill_cfg = config.get("skills", {})
        self.enabled_skills = list(skill_cfg.get("enabled", []))
        self.max_steps_per_execution = int(skill_cfg.get("max_steps_per_execution",
                                          agent_cfg.get("max_steps_per_execution", 24)))
        self.budget = Budget({
            "execute_commands": agent_cfg.get("max_execute_commands_per_episode", 40),
            "read_tools": agent_cfg.get("max_read_tool_calls_per_episode", 60),
            "execution_steps": None,  # charged per execution below
        })
        self.wall_clock_limit = float(agent_cfg.get("wall_clock_timeout_seconds", 3600))
        self.clock_started = clock_started if clock_started is not None else time.monotonic()
        self.snapshot: BattleSnapshot | None = None
        self.decision_context: DecisionContext | None = None
        self.unavailable: dict[str, str] = {}
        self.commands: dict[str, dict[str, Any]] = {}
        self.last_report: ExecutionReport | None = None
        self.episode_stopped = False
        self.fallback_used = False
        self.manager_kwargs = self._manager_kwargs(config, skill_cfg)

    @staticmethod
    def _manager_kwargs(config: dict, skill_cfg: dict) -> dict[str, Any]:
        """Trusted manager configuration derived from config only (never model input)."""
        kwargs: dict[str, Any] = {
            "max_steps": int(skill_cfg.get("skill_max_steps", 4)),
        }
        race_profile = skill_cfg.get("race_profile", "off")
        if race_profile and race_profile != "off":
            kwargs["race_profile"] = race_profile
        spatial = {key: bool(skill_cfg.get(key, False))
                   for key in ("enable_regroup", "enable_cohesive_search")}
        if any(spatial.values()):
            kwargs["spatial_params"] = spatial
        return kwargs

    # -- clock -----------------------------------------------------------------

    def _check_clock(self) -> None:
        elapsed = time.monotonic() - self.clock_started
        if elapsed > self.wall_clock_limit:
            raise CoordinatorError("wall_clock_exceeded", f"{elapsed:.0f}s")

    # -- snapshots ---------------------------------------------------------------

    def refresh_snapshot(self) -> BattleSnapshot:
        if self.episode_stopped:
            raise CoordinatorError("episode_stopped",
                                   "episode halted (unknown state or failure)")
        self._check_clock()
        raw = self.backend.snapshot()
        self._guard_episode(raw)
        counter = getattr(self, "_snap_counter", 0) + 1
        self._snap_counter = counter
        snapshot_id = f"snap-{raw['episode_id']}-{raw['game_loop']}-{counter:04d}"
        summary = self._model_visible_summary(raw)
        self.snapshot = BattleSnapshot(
            run_id=self.run_id, episode_id=raw["episode_id"], game_loop=raw["game_loop"],
            snapshot_id=snapshot_id, summary=summary,
            controller_state={"active_skill": raw.get("active_skill"),
                              "last_termination": raw.get("last_termination"),
                              "done": raw.get("done", False)},
            simulated=bool(raw.get("simulated", False)))
        self._freeze_candidates(raw)
        self.emit("snapshot", snapshot_id=snapshot_id, game_loop=raw["game_loop"],
                  candidates=len(self.decision_context.candidates))
        return self.snapshot

    def _guard_episode(self, raw: dict[str, Any]) -> None:
        if raw.get("episode_id") != self.episode_id:
            self.stop_episode("episode_mismatch")
            raise CoordinatorError("episode_mismatch",
                                   f"backend episode {raw.get('episode_id')}")

    def _model_visible_summary(self, raw: dict[str, Any]) -> dict[str, Any]:
        state = raw.get("state") or {}
        return {"objective": state.get("objective"),
                "map": state.get("map"), "game_loop": raw.get("game_loop"),
                "steps_remaining": state.get("steps_remaining"),
                "live_allies": state.get("live_allies"),
                "team_risk_summary": state.get("team_risk_summary"),
                "allies_core": state.get("allies_core"),
                "visible_enemy_coverage": state.get("visible_enemy_coverage"),
                "scenario_brief": state.get("scenario_brief"),
                "active_skill": raw.get("active_skill"),
                "last_skill_termination": raw.get("last_termination"),
                "simulated": bool(raw.get("simulated", False))}

    def _freeze_candidates(self, raw: dict[str, Any]) -> None:
        candidates, unavailable = [], {}
        for choice_id, raw_candidate in sorted(raw["candidates"].items(),
                                               key=lambda item: int(item[0])):
            enabled, reason = candidate_enabled(self.enabled_skills, raw_candidate,
                                                raw.get("spatial_available", []))
            if not enabled:
                unavailable[display_name(raw_candidate)] = reason or "not available"
                continue
            skill = raw_candidate["skill"]
            candidates.append(SkillCandidate(
                choice_id=str(choice_id), skill_name=display_name(raw_candidate),
                parameters={"posture": posture_for(raw_candidate)} if skill == "adaptive_focus_fire"
                else ({"priority_target": raw_candidate["priority"]}
                      if raw_candidate.get("priority") else {}),
                description=raw_candidate.get("text", ""),
                composite=skill in ("adaptive_focus_fire", "cover_retreat"),
                action_control=("partial" if skill in ("heal_assign", "regroup",
                                                       "cohesive_search") else "full")))
        if not candidates:
            raise CoordinatorError("no_candidates_available", "")
        self.unavailable = unavailable
        self.decision_context = DecisionContext(
            run_id=self.run_id, episode_id=self.episode_id,
            snapshot_id=self.snapshot.snapshot_id,
            candidates_version=_candidates_version(raw["candidates"]),
            candidates=candidates)

    # -- read helpers --------------------------------------------------------------

    def get_battle_state(self) -> dict[str, Any]:
        self.budget.charge("read_tools")
        snapshot = self.snapshot or self.refresh_snapshot()
        return {"snapshot_id": snapshot.snapshot_id,
                "game_loop": snapshot.game_loop,
                "episode_id": snapshot.episode_id,
                "done": snapshot.controller_state.get("done", False),
                "battle": snapshot.summary,
                "note": "simulated mock data" if snapshot.simulated else "real environment facts",
                "unavailable_skills": self.unavailable}

    def list_available_skills(self, snapshot_id: str) -> dict[str, Any]:
        self.budget.charge("read_tools")
        if self.snapshot is None or snapshot_id != self.snapshot.snapshot_id:
            raise CoordinatorError("stale_snapshot",
                                   f"expected {self.snapshot.snapshot_id if self.snapshot else None}")
        return {"snapshot_id": snapshot_id,
                "candidates_version": self.decision_context.candidates_version,
                "candidates": [candidate.model_dump() for candidate in
                               self.decision_context.candidates],
                "unavailable": self.unavailable}

    def get_last_execution_report(self) -> dict[str, Any]:
        self.budget.charge("read_tools")
        if self.last_report is None:
            return {"note": "no execution yet"}
        return self.last_report.model_dump()

    # -- execution ------------------------------------------------------------------

    def execute_skill_window(self, tool_call_id: str, choice_id: str,
                             expected_snapshot_id: str, decision_steps: int) -> dict[str, Any]:
        if self.episode_stopped:
            raise CoordinatorError("episode_stopped", "episode halted")
        self._check_clock()
        command_id = f"cmd-{self.run_id}-{self.episode_id}-{tool_call_id}"
        steps = max(1, min(int(decision_steps), self.max_steps_per_execution))
        params = {"choice_id": str(choice_id), "expected_snapshot_id": expected_snapshot_id,
                  "decision_steps": int(decision_steps)}
        # idempotency: same command id + same params replays the stored report
        prior = self.commands.get(command_id)
        if prior is not None:
            if prior["params"] != params:
                raise CoordinatorError("command_id_conflict",
                                       "same tool_call_id with different parameters")
            if prior["status"] == "completed":
                return prior["report"].model_dump()
            if prior["status"] == "executing":
                raise CoordinatorError("command_in_progress", command_id)
            if prior["status"] == "unknown":
                self.stop_episode("previous_command_unknown")
                raise CoordinatorError("previous_command_unknown", command_id)
        if self.snapshot is None or expected_snapshot_id != self.snapshot.snapshot_id:
            raise CoordinatorError("stale_snapshot",
                                   f"{expected_snapshot_id} is not the current frozen snapshot")
        candidate = next((c for c in self.decision_context.candidates
                          if c.choice_id == str(choice_id)), None)
        if candidate is None:
            raise CoordinatorError("unknown_choice",
                                   f"choice {choice_id} is not in the frozen candidate map")
        self.budget.charge("execute_commands")
        request = SkillExecutionRequest(
            command_id=command_id, tool_call_id=tool_call_id, choice_id=str(choice_id),
            expected_snapshot_id=expected_snapshot_id, decision_steps=steps,
            candidates_version=self.decision_context.candidates_version)
        self.commands[command_id] = {"params": params, "status": "executing",
                                     "request": request, "report": None}
        posture = candidate.parameters.get("posture")
        started = time.monotonic()
        try:
            raw = self.backend.execute(str(choice_id), steps, posture=posture)
        except BackendError as exc:
            self.commands[command_id]["status"] = "unknown"
            self.stop_episode(f"backend_error:{exc.code}")
            report = ExecutionReport(
                request_accepted=False, outcome="unknown", error=str(exc),
                start_snapshot_id=expected_snapshot_id, simulated=False)
            self.last_report = report
            self.emit("execute_unknown", command_id=command_id, error=str(exc))
            return report.model_dump()
        report = self._report_from_backend(raw, expected_snapshot_id, command_id)
        self.commands[command_id].update(status="completed", report=report)
        self.last_report = report
        self.snapshot = None  # old snapshot loses execution permission
        self.decision_context = None
        self.emit("execute_completed", command_id=command_id,
                  outcome=report.outcome, steps=report.steps_executed,
                  stop_reason=report.stop_reason, latency_ms=(time.monotonic() - started) * 1000,
                  codex_cli_starts=self.codex_stats.get("cli_starts", 0))
        return report.model_dump()

    def _report_from_backend(self, raw: dict[str, Any], expected_snapshot_id: str,
                             command_id: str) -> ExecutionReport:
        outcome = raw.get("outcome", "failed")
        if outcome not in ("accepted", "rejected", "generated", "submitted",
                           "interrupted", "timeout", "skill_complete", "episode_over",
                           "failed", "unknown", "fallback"):
            outcome = "failed"
        end_snapshot_id = None
        if raw.get("state") is not None:
            end_snapshot_id = None  # assigned on next refresh; report carries loops
        return ExecutionReport(
            request_accepted=bool(raw.get("accepted")),
            outcome=outcome,
            actual_mode=raw.get("telemetry", {}).get("skill"),
            steps_executed=int(raw.get("steps_executed", 0)),
            start_snapshot_id=expected_snapshot_id,
            end_snapshot_id=end_snapshot_id,
            end_game_loop=raw.get("end_game_loop"),
            stop_reason=raw.get("stop_reason"),
            observation_summary=self._model_visible_summary(raw),
            diagnostics={"handoff": raw.get("handoff"),
                         "selection_source": raw.get("selection_source"),
                         "start_game_loop": raw.get("start_game_loop"),
                         "last_skill_telemetry_keys":
                             sorted((raw.get("telemetry") or {}).keys())},
            fallback_used=bool(self.fallback_used),
            simulated=bool(raw.get("simulated", False)))

    # -- failure handling ---------------------------------------------------------

    def stop_episode(self, reason: str) -> None:
        self.episode_stopped = True
        self.emit("episode_stopped", reason=reason)

    def mark_fallback(self, reason: str) -> None:
        """Degraded run: adaptive baseline keeps history; never shown as pure-agent."""
        self.fallback_used = True
        self.emit("fallback", reason=reason)

    def episode_summary(self) -> dict[str, Any]:
        status = self.backend.episode_status()
        return {"run_id": self.run_id, "episode_id": self.episode_id,
                "budget_spent": dict(self.budget.spent),
                "codex_cli_starts": self.codex_stats.get("cli_starts", 0),
                "fallback_used": self.fallback_used,
                "episode_stopped": self.episode_stopped,
                "backend": status}


def new_run_id() -> str:
    return f"run-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
