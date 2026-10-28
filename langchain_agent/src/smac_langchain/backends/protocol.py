"""Host<->backend protocol and the in-process Backend interface.

The wire protocol is plain standard-library JSON (one JSON object per line,
UTF-8) so the SMACv2 worker never needs to import this package, Pydantic or
LangChain. This module documents the messages and defines the host-side
Backend ABC both the mock backend and the RPC client implement.

Request  : {"id": <str>, "op": <str>, ...op-specific fields}
Response : {"id": <str>, "ok": true, "result": {...}} |
           {"id": <str>, "ok": false, "error": {"code": <str>, "message": <str>}}

Ops:
  start  {episode_id, seed, manager_kwargs, skill_version, sc2_path?,
          env_args?}                        -> {n_agents, game_loop, episode_id}
  snapshot {}                               -> SnapshotData (see below)
  execute {choice_id, decision_steps, posture?} -> ExecutionData
  episode_status {}                         -> {done, outcome, steps, game_loop}
  close {}                                  -> {}

SnapshotData (model-visible facts only; mock data carries simulated=true):
  {run_hint, episode_id, game_loop, n_agents, done,
   state: {…team-visible summary…},
   candidates: {choice_id: {"skill": name, "priority": value, "text": desc}},
   spatial_available: [names], active_skill, last_termination}

ExecutionData:
  {accepted, outcome, steps_executed, start_game_loop, end_game_loop,
   stop_reason, handoff, telemetry: {…skill diagnostics…},
   feedback: {…last step feedback…}, state (fresh snapshot state),
   candidates, done}
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BackendError(RuntimeError):
    def __init__(self, code: str, message: str = ""):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


class Backend(ABC):
    """Host-side backend contract. All writes are serialized by the coordinator."""

    backend_name: str = "abstract"

    @abstractmethod
    def start(self, episode_id: int, seed: int, manager_kwargs: dict[str, Any],
              skill_version: str = "current") -> dict[str, Any]: ...

    @abstractmethod
    def snapshot(self) -> dict[str, Any]: ...

    @abstractmethod
    def execute(self, choice_id: str, decision_steps: int,
                posture: str | None = None) -> dict[str, Any]: ...

    @abstractmethod
    def episode_status(self) -> dict[str, Any]: ...

    @abstractmethod
    def close(self) -> None: ...
