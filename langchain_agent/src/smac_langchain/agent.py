"""Agent assembly and episode loop (create_agent based).

The LangChain agent owns the tool loop. Codex (or the offline mock model) is
only the per-turn decision backend via CodexCLIChatModel; all battle tools run
in THIS process through the BattleCoordinator.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

from .backends.mock import MockBackend
from .backends.protocol import Backend, BackendError
from .llm.codex_chat_model import CodexCLIChatModel, MockTacticalChatModel
from .middleware import BudgetGuardMiddleware, TrimHistoryMiddleware
from .runtime.coordinator import BattleCoordinator, CoordinatorError, new_run_id
from .telemetry import Telemetry
from .tools import build_battle_tools

SYSTEM_PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompts" / "tactical_system.md"


def load_system_prompt() -> str:
    return SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")


def build_model(config: dict[str, Any], model_name: str,
                request_root: Path | None = None, emit=None) -> Any:
    if model_name == "mock":
        return MockTacticalChatModel()
    codex_cfg = config.get("codex", {})
    from .llm.codex_client import CodexCLI
    client = CodexCLI(
        model=model_name or codex_cfg.get("model", "gpt-5.1-codex"),
        reasoning_effort=codex_cfg.get("reasoning_effort", "medium"),
        timeout_seconds=codex_cfg.get("timeout_seconds", 180),
        max_retries=codex_cfg.get("max_retries", 1),
        max_output_bytes=codex_cfg.get("max_output_bytes", 2_000_000),
        executable=codex_cfg.get("codex_path") or None,
        request_root=request_root or ".codex_requests",
        logger=emit)
    return CodexCLIChatModel(client=client)


def build_agent(model: Any, coordinator: BattleCoordinator, config: dict[str, Any],
                emit=None) -> Any:
    tools = build_battle_tools(coordinator)
    agent_cfg = config.get("agent", {})
    middleware = [
        BudgetGuardMiddleware(
            {"max_model_calls_per_episode": agent_cfg.get("max_model_calls_per_episode", 40),
             "max_tool_calls_per_episode": agent_cfg.get("max_tool_calls_per_episode", 120),
             "wall_clock_timeout_seconds": agent_cfg.get("wall_clock_timeout_seconds", 3600)},
            emit=emit),
        TrimHistoryMiddleware(keep_pairs=agent_cfg.get("history_keep_pairs", 6)),
    ]
    return create_agent(
        model, tools,
        system_prompt=load_system_prompt(),
        middleware=middleware,
        checkpointer=InMemorySaver(),  # short-term state demo; NOT SC2 recovery
    )


def run_agent_episode(backend: Backend, config: dict[str, Any], run_dir: Path,
                      episode_id: int, seed: int, model_name: str = "mock",
                      live: bool = False) -> dict[str, Any]:
    """One episode: agent decision rounds until done/budget/unknown-state."""
    telemetry = Telemetry(run_dir)
    run_id = new_run_id()
    telemetry("run_started", run_id=run_id, episode_id=episode_id,
              backend=backend.backend_name, model=model_name, live=live)
    telemetry.provenance(config, extra_files=[
        Path(config.get("_pkg_root", ".")) / ".." / "skills" / "current" / "skill_policy.py",
        Path(config.get("_pkg_root", ".")) / ".." / "skills" / "current" / "skill_manager.py"])
    model = build_model(config, model_name,
                        request_root=run_dir / "codex_requests", emit=telemetry)
    codex_stats = getattr(model, "client", None)
    coordinator = BattleCoordinator(
        run_id, episode_id, backend, config, emit=telemetry,
        codex_stats=codex_stats.stats if codex_stats else {})
    agent = build_agent(model, coordinator, config, emit=telemetry)
    config_thread = {"configurable": {"thread_id": f"{run_id}-ep{episode_id}"}}
    backend.start(episode_id, seed, coordinator.manager_kwargs)
    rounds, degraded = 0, False
    outcome = {"done": False, "outcome": None}
    failure = None
    try:
        coordinator.refresh_snapshot()
        while True:
            status = backend.episode_status()
            if status.get("done"):
                outcome = status
                break
            rounds += 1
            try:
                result = agent.invoke(
                    {"messages": [{"role": "user", "content":
                                   "New tactical round. Inspect the battle state, then "
                                   "choose and execute one skill window (one new execute "
                                   "command per round)."}]},
                    config=config_thread)
            except Exception as exc:  # noqa: BLE001 - model/infra failure path
                failure = f"{type(exc).__name__}: {exc}"
                commands = coordinator.commands
                executed_after_failure = any(
                    c["status"] in ("completed", "unknown") for c in commands.values())
                if not executed_after_failure and any(
                        c["status"] == "completed" for c in commands.values()):
                    pass
                if not commands or all(c["status"] == "completed" for c in commands.values()):
                    # Confirmed: nothing executed for THIS round yet -> degrade
                    # to the preserved-history adaptive baseline, clearly marked.
                    degraded = True
                    coordinator.mark_fallback(failure)
                    telemetry("fallback_round", round=rounds, reason=failure)
                    report = backend.execute(_default_adaptive_choice(coordinator), 4)
                    telemetry("fallback_execute", report={k: report[k] for k in
                                                         ("outcome", "steps_executed")})
                    continue
                coordinator.stop_episode("model_failure_after_execution")
                telemetry("episode_stopped_unknown", reason=failure)
                outcome = {"done": True, "outcome": "unknown", "stopped": True}
                break
            # host decides the next round from actual worker facts
            last = coordinator.last_report
            if last is None or last.outcome in ("episode_over",):
                status = backend.episode_status()
                outcome = status
                if status.get("done"):
                    break
    finally:
        summary = coordinator.episode_summary()
        summary.update(rounds=rounds, degraded=degraded, failure=failure,
                       final_outcome=outcome.get("outcome"),
                       wall_seconds=time.monotonic() - coordinator.clock_started)
        telemetry("run_finished", **summary)
        telemetry.close()
        try:
            backend.close()
        except Exception:  # noqa: BLE001
            pass
    return summary


def _default_adaptive_choice(coordinator: BattleCoordinator) -> str:
    """The preserved adaptive baseline candidate id for fallback rounds."""
    if coordinator.decision_context is not None:
        for candidate in coordinator.decision_context.candidates:
            if candidate.skill_name.startswith("adaptive_focus_fire"):
                return candidate.choice_id
    snapshot = coordinator.refresh_snapshot()
    for candidate in coordinator.decision_context.candidates:  # type: ignore[union-attr]
        if candidate.skill_name.startswith("adaptive_focus_fire"):
            return candidate.choice_id
    raise CoordinatorError("no_adaptive_candidate", "fallback needs adaptive_focus_fire")


def run_pure_skill_episode(backend: Backend, config: dict[str, Any], run_dir: Path,
                           episode_id: int, seed: int) -> dict[str, Any]:
    """No-model comparison path: adaptive baseline only, zero Codex calls."""
    telemetry = Telemetry(run_dir)
    run_id = new_run_id()
    telemetry("pure_skill_started", run_id=run_id, episode_id=episode_id,
              backend=backend.backend_name)
    coordinator = BattleCoordinator(run_id, episode_id, backend, config, emit=telemetry)
    backend.start(episode_id, seed, coordinator.manager_kwargs)
    windows = 0
    outcome = None
    try:
        while True:
            status = backend.episode_status()
            if status.get("done"):
                outcome = status.get("outcome")
                break
            coordinator.refresh_snapshot()
            choice = _default_adaptive_choice(coordinator)
            report = backend.execute(choice, config.get("skills", {}).get(
                "pure_skill_window_steps", 8))
            windows += 1
            telemetry("pure_skill_window", window=windows, choice=choice,
                      outcome=report.get("outcome"), steps=report.get("steps_executed"))
    finally:
        summary = coordinator.episode_summary()
        summary.update(windows=windows, outcome=outcome, codex_cli_starts=0)
        telemetry("pure_skill_finished", **summary)
        telemetry.close()
        try:
            backend.close()
        except Exception:  # noqa: BLE001
            pass
    return summary
