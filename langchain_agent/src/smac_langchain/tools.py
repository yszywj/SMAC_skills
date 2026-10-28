"""Battle tools for the LangChain agent.

The coordinator, session ids and budgets are injected via closures (the
LangChain tool-runtime pattern for this version): the model only ever fills in
genuine business fields, never runtime dependencies, and can never call
env.step, shell or Python directly.
"""
from __future__ import annotations

import json
from typing import Annotated, Any

from pydantic import BaseModel, Field
from langchain_core.tools import tool

from .runtime.coordinator import BattleCoordinator


class ListSkillsArgs(BaseModel):
    snapshot_id: str = Field(description="snapshot_id from get_battle_state")


class ExecuteArgs(BaseModel):
    choice_id: str = Field(description="choice id from list_available_skills")
    expected_snapshot_id: str = Field(description="snapshot the candidates were frozen for")
    decision_steps: int = Field(ge=1, le=64,
                                description="local steps to run under this decision")


def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def build_battle_tools(coordinator: BattleCoordinator):
    """Return the four battle tools bound to one episode coordinator."""

    @tool
    def get_battle_state() -> str:
        """Read the current team-visible battle state and snapshot_id.

        Does NOT advance the game. Mock-backend data is labelled simulated.
        """
        return _dumps(coordinator.get_battle_state())

    @tool(args_schema=ListSkillsArgs)
    def list_available_skills(snapshot_id: str) -> str:
        """List the skill candidates frozen for this exact snapshot_id.

        Each candidate has a choice_id, parameters and constraints. Does NOT
        advance the game.
        """
        return _dumps(coordinator.list_available_skills(snapshot_id))

    @tool(args_schema=ExecuteArgs)
    def execute_skill_window(choice_id: str, expected_snapshot_id: str,
                             decision_steps: int) -> str:
        """Execute a bounded local window of the chosen skill.

        Validates snapshot/candidate freshness first, then runs up to
        decision_steps real local steps (each step: fresh observation, manager
        step, mask validation, commit, env step). Returns an ExecutionReport
        with the new observation summary; get_battle_state afterwards for the
        next snapshot_id.
        """
        return _dumps(coordinator.execute_skill_window(
            tool_call_id=_current_call_id.get() or "untracked",
            choice_id=choice_id, expected_snapshot_id=expected_snapshot_id,
            decision_steps=decision_steps))

    @tool
    def get_last_execution_report() -> str:
        """Read-only view of the most recent ExecutionReport."""
        return _dumps(coordinator.get_last_execution_report())

    return [get_battle_state, list_available_skills, execute_skill_window,
            get_last_execution_report]


import contextvars  # noqa: E402

# The tool_call_id arrives from the agent runtime; without runtime injection in
# this LangChain version we track it via middleware-provided contextvar.
_current_call_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "current_tool_call_id", default=None)


def set_current_call_id(call_id: str | None) -> None:
    _current_call_id.set(call_id)
