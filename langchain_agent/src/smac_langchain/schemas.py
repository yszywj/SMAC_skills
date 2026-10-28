"""Typed data contracts between agent, coordinator and backends.

All model-visible facts are JSON-safe and derived from team-visible state only.
Hidden evaluation data never enters these schemas.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

CommandStatus = Literal["pending", "executing", "completed", "failed", "unknown"]
Outcome = Literal[
    "accepted", "rejected", "generated", "submitted", "interrupted",
    "timeout", "skill_complete", "episode_over", "failed", "unknown", "fallback"]


class BattleSnapshot(BaseModel):
    """Frozen view of what the model may know at one decision point."""
    run_id: str
    episode_id: int
    game_loop: int
    snapshot_id: str
    summary: dict[str, Any] = Field(default_factory=dict)
    controller_state: dict[str, Any] = Field(default_factory=dict)
    simulated: bool = False  # mock backend data must be labelled


class SkillCandidate(BaseModel):
    choice_id: str
    skill_name: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    description: str = ""
    composite: bool = False            # adaptive = composed controller
    action_control: str = "full"       # "full" | "partial"


class DecisionContext(BaseModel):
    """Snapshot-bound frozen candidate map, generated locally."""
    run_id: str
    episode_id: int
    snapshot_id: str
    candidates_version: str
    candidates: list[SkillCandidate]


class SkillExecutionRequest(BaseModel):
    """What the model may ask for. No arbitrary code or manager kwargs."""
    command_id: str                    # locally generated id, not model-supplied
    tool_call_id: str
    choice_id: str
    expected_snapshot_id: str
    decision_steps: int = Field(ge=1, le=64)
    candidates_version: str


class ExecutionReport(BaseModel):
    request_accepted: bool
    outcome: Outcome
    actual_mode: str | None = None
    steps_executed: int = 0
    start_snapshot_id: str | None = None
    end_snapshot_id: str | None = None
    end_game_loop: int | None = None
    stop_reason: str | None = None
    observation_summary: dict[str, Any] = Field(default_factory=dict)
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    fallback_used: bool = False
    simulated: bool = False
    error: str | None = None


# --- Codex bridge envelope (custom protocol, see docs/codex_bridge.md) ------
# NOTE: kept a flat JSON Schema on purpose — the Codex CLI/API rejects
# oneOf/anyOf/allOf at the top level of --output-schema. The tool_call/final
# mutual exclusivity is enforced by CodexCLI._validate_envelope in code.

ENVELOPE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "kind": {"type": "string", "enum": ["tool_call", "final"]},
        "tool_name": {"type": ["string", "null"],
                      "description": "tool name when kind=tool_call, else null"},
        "arguments_json": {"type": ["string", "null"],
                           "description": "JSON-serialized object of tool arguments "
                                           "when kind=tool_call, else null"},
        "final_text": {"type": ["string", "null"],
                       "description": "short summary when kind=final, else null"},
    },
    "required": ["kind", "tool_name", "arguments_json", "final_text"],
}
