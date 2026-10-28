"""JSONL stdio RPC client for a backend worker subprocess.

stdout of the worker is protocol-only; worker logs go to stderr/file. The
worker never imports this package (plain stdlib JSON on both ends).
"""
from __future__ import annotations

import json
import os
import subprocess
import threading
import uuid
from pathlib import Path
from typing import Any

from .protocol import Backend, BackendError


class RPCClientBackend(Backend):
    backend_name = "rpc"

    def __init__(self, command: list[str], cwd: str | Path | None = None,
                 timeout_seconds: float = 600.0, startup_timeout: float = 120.0,
                 on_line=None):
        self.command = list(command)
        self.timeout = float(timeout_seconds)
        creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        self.process = subprocess.Popen(
            self.command, cwd=str(cwd) if cwd else None,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None,
            text=True, encoding="utf-8", errors="replace", creationflags=creationflags)
        self._lock = threading.Lock()
        self._on_line = on_line
        self._wait_ready(startup_timeout)

    def _wait_ready(self, timeout: float) -> None:
        response = self._call({"id": "ready", "op": "ping"}, timeout=timeout)
        if not response.get("ok"):
            raise BackendError("rpc_worker_not_ready", str(response.get("error")))

    def _call(self, request: dict[str, Any], timeout: float | None = None) -> dict[str, Any]:
        with self._lock:
            if self.process.poll() is not None:
                raise BackendError("rpc_worker_exited", f"code {self.process.returncode}")
            line = json.dumps(request, ensure_ascii=False) + "\n"
            try:
                self.process.stdin.write(line)
                self.process.stdin.flush()
            except (BrokenPipeError, OSError) as exc:
                raise BackendError("rpc_worker_unreachable", str(exc)) from None
            # read exactly one response line for this request id
            while True:
                response_line = self.process.stdout.readline()
                if not response_line:
                    raise BackendError("rpc_worker_lost", "stdout closed")
                try:
                    response = json.loads(response_line)
                except ValueError as exc:
                    raise BackendError("rpc_protocol_violation",
                                       f"non-JSON stdout: {response_line[:200]}") from exc
                if self._on_line:
                    self._on_line(response)
                if response.get("id") == request["id"]:
                    return response

    def _invoke(self, op: str, payload: dict[str, Any] | None = None,
                timeout: float | None = None) -> dict[str, Any]:
        request = {"id": uuid.uuid4().hex[:12], "op": op, **(payload or {})}
        response = self._call(request, timeout=timeout or self.timeout)
        if not response.get("ok"):
            error = response.get("error") or {}
            raise BackendError(error.get("code", "rpc_error"), error.get("message", ""))
        return response.get("result") or {}

    # -- Backend API ------------------------------------------------------------

    def start(self, episode_id, seed, manager_kwargs, skill_version="current"):
        return self._invoke("start", {"episode_id": episode_id, "seed": seed,
                                      "manager_kwargs": manager_kwargs,
                                      "skill_version": skill_version})

    def snapshot(self):
        return self._invoke("snapshot")

    def execute(self, choice_id, decision_steps, posture=None):
        return self._invoke("execute", {"choice_id": str(choice_id),
                                        "decision_steps": int(decision_steps),
                                        **({"posture": posture} if posture else {})})

    def episode_status(self):
        return self._invoke("episode_status")

    def close(self):
        try:
            self._invoke("close", timeout=30)
        except BackendError:
            pass
        finally:
            try:
                self.process.stdin.close()
            except OSError:
                pass
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.kill()
