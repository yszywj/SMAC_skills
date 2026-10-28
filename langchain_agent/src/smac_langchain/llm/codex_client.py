"""Restricted non-interactive Codex CLI process client.

One `codex exec` subprocess per request, own request directory and result file,
read-only sandbox, argv list (never shell=True), UTF-8 stdin prompt, JSONL event
stream on stdout parsed line by line. Mirrors the safety pattern of the legacy
bridge/planner.py but with per-request isolation, configurable model and strict
JSON Schema validation of the final message.

The CLI keeps its own login; we never read, copy or print credentials.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from ..schemas import ENVELOPE_SCHEMA


class CodexError(RuntimeError):
    pass


FORBIDDEN_ITEM_TYPES = ("command_execution", "file_change", "file_diff",
                        "mcp_tool_call", "web_search", "patch_apply")


def find_codex(path: str | os.PathLike | None = None) -> Path:
    """Locate the native codex executable. Windows: resolve npm shims to codex.exe."""
    candidate = Path(path) if path else None
    if candidate is None:
        found = shutil.which("codex")
        candidate = Path(found) if found else None
    if os.name == "nt" and (candidate is None
                            or candidate.suffix.lower() in (".cmd", ".ps1", ".bat")):
        npm = candidate.parent if candidate else Path(
            os.environ.get("APPDATA", "")) / "npm"
        matches = sorted((npm / "node_modules" / "@openai" / "codex").glob("**/codex.exe"))
        if matches:
            candidate = matches[0]
    if candidate is None or not candidate.is_file() \
            or candidate.suffix.lower() in (".cmd", ".bat", ".ps1"):
        raise CodexError("codex_native_not_found")
    return candidate.resolve()


class CodexRequestResult:
    def __init__(self, **kwargs: Any):
        self.__dict__.update(kwargs)

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"CodexRequestResult({self.__dict__})"


class CodexCLI:
    """Short-lived `codex exec` invocations with strict validation."""

    def __init__(self, model: str, reasoning_effort: str = "medium",
                 timeout_seconds: float = 180.0, max_retries: int = 1,
                 max_output_bytes: int = 2_000_000,
                 executable: str | os.PathLike | None = None,
                 request_root: str | os.PathLike = ".codex_requests",
                 schema: dict[str, Any] | None = None,
                 logger=None, command_factory=None):
        """command_factory: test hook to substitute the argv (e.g. a Python
        stand-in script). Production use never sets it; all event-stream,
        result-file, timeout and retry validation still applies."""
        self.model = model
        self.effort = reasoning_effort
        self.timeout = float(timeout_seconds)
        self.max_retries = int(max_retries)
        self.max_output_bytes = int(max_output_bytes)
        self.executable = find_codex(executable)
        self.request_root = Path(request_root)
        self.request_root.mkdir(parents=True, exist_ok=True)
        self.schema = schema if schema is not None else ENVELOPE_SCHEMA
        self.logger = logger
        self.command_factory = command_factory
        self.stats = {"cli_starts": 0, "retries": 0, "failures": 0, "usage": {},
                      "latency_ms": []}

    # -- process plumbing ---------------------------------------------------

    def command(self, schema_path: Path, output_path: Path, workdir: Path) -> list[str]:
        if self.command_factory is not None:
            return self.command_factory(schema_path, output_path, workdir)
        # model "default" (or "") uses the account's CLI default model; we
        # never hardcode an account-specific model name.
        model_args = [] if self.model in ("", "default") else ["--model", self.model]
        return [str(self.executable), "exec", "--ignore-user-config", "--ephemeral",
                "--skip-git-repo-check", "--sandbox", "read-only",
                *model_args,
                "--disable", "apps", "--disable", "shell_tool", "--disable", "hooks",
                "--disable", "browser_use", "--disable", "computer_use",
                "-c", 'web_search="disabled"', "-c", "project_doc_max_bytes=0",
                "-c", f'model_reasoning_effort="{self.effort}"',
                "--output-schema", str(schema_path.resolve()),
                "--output-last-message", str(output_path.resolve()),
                "--json", "--color", "never", "-"]

    def _run_once(self, prompt: str, request_dir: Path) -> dict[str, Any]:
        """One subprocess attempt. Returns dict with stdout, result_text, error."""
        schema_path = request_dir / "envelope.schema.json"
        schema_path.write_text(json.dumps(self.schema), encoding="utf-8")
        output_path = request_dir / "last_message.json"
        workdir = request_dir / "workdir"
        workdir.mkdir(exist_ok=True)
        argv = self.command(schema_path, output_path, workdir)
        creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        process = subprocess.Popen(
            argv, cwd=str(workdir), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace",
            creationflags=creationflags)
        self.stats["cli_starts"] += 1
        try:
            stdout, _stderr = process.communicate(prompt, timeout=self.timeout)
        except subprocess.TimeoutExpired:
            self._kill_tree(process)
            return {"stdout": stdout_bytes_to_text(process), "result_text": "",
                    "error": "codex_timeout"}
        if len(stdout.encode("utf-8", "replace")) > self.max_output_bytes:
            return {"stdout": "", "result_text": "", "error": "codex_output_too_long"}
        if process.returncode != 0:
            return {"stdout": stdout, "result_text": "",
                    "error": f"codex_exit_{process.returncode}"}
        if not output_path.is_file():
            return {"stdout": stdout, "result_text": "",
                    "error": "codex_missing_result_file"}
        result_text = output_path.read_text(encoding="utf-8")
        return {"stdout": stdout, "result_text": result_text, "error": None}

    @staticmethod
    def _kill_tree(process: subprocess.Popen) -> None:
        """Kill only processes we started ourselves."""
        try:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                               capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)
            else:
                process.kill()
        except OSError:
            pass

    # -- event-stream validation ---------------------------------------------

    def _validate_events(self, stdout: str) -> tuple[bool, dict[str, Any], str | None]:
        """Parse the JSONL event stream line by line (never one big JSON)."""
        completed, usage, error = False, {}, None
        for line in stdout.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except ValueError:
                continue  # non-JSON log lines are ignored, not fatal
            if not isinstance(event, dict):
                continue
            if event.get("type") == "turn.completed":
                completed = True
                usage = event.get("usage") or {}
            if event.get("type") in ("error", "turn.failed"):
                error = error or "codex_turn_failed"
            item = event.get("item") or {}
            if isinstance(item, dict) and item.get("type") in FORBIDDEN_ITEM_TYPES:
                error = "codex_forbidden_builtin_tool_use"
        return completed, usage, error

    def _validate_envelope(self, result_text: str) -> dict[str, Any]:
        try:
            value = json.loads(result_text)
        except ValueError:
            raise CodexError("codex_invalid_json") from None
        if not isinstance(value, dict) or value.get("kind") not in ("tool_call", "final"):
            raise CodexError("codex_schema_invalid")
        kind = value["kind"]
        if kind == "tool_call":
            if not isinstance(value.get("tool_name"), str) \
                    or not isinstance(value.get("arguments_json"), str) \
                    or value.get("final_text") is not None:
                raise CodexError("codex_schema_invalid")
            try:
                arguments = json.loads(value["arguments_json"])
            except ValueError:
                raise CodexError("codex_arguments_not_json_object") from None
            if not isinstance(arguments, dict):
                raise CodexError("codex_arguments_not_json_object")
            return {"kind": "tool_call", "tool_name": value["tool_name"],
                    "arguments": arguments, "usage": {}}
        if value.get("tool_name") is not None or value.get("arguments_json") is not None \
                or not isinstance(value.get("final_text"), str):
            raise CodexError("codex_schema_invalid")
        return {"kind": "final", "final_text": value["final_text"], "usage": {}}

    # -- public API -----------------------------------------------------------

    def request(self, prompt: str) -> dict[str, Any]:
        """Run one validated request with retries. Returns an envelope dict with usage.

        Envelope: {"kind": "tool_call", "tool_name", "arguments"} or
                  {"kind": "final", "final_text"}; usage only when the CLI reported it.
        """
        last_error = "codex_no_attempt"
        for attempt in range(self.max_retries + 1):
            if attempt:
                self.stats["retries"] += 1
            request_dir = self.request_root / f"req-{uuid.uuid4().hex[:12]}"
            request_dir.mkdir(parents=True, exist_ok=False)
            started = time.monotonic()
            raw = self._run_once(prompt, request_dir)
            latency = (time.monotonic() - started) * 1000
            self.stats["latency_ms"].append(latency)
            completed, usage, stream_error = self._validate_events(raw["stdout"])
            error = raw["error"] or stream_error
            if not error and not completed:
                error = "codex_turn_not_completed"
            if error is None:
                try:
                    envelope = self._validate_envelope(raw["result_text"])
                except CodexError as exc:
                    error = str(exc)
            else:
                envelope = None
            if self.logger:
                self.logger("codex_request", attempt=attempt, latency_ms=latency,
                            error=error, usage=usage,
                            request_dir=str(request_dir))
            if error is None:
                envelope["usage"] = usage  # only what the CLI actually reported
                envelope["latency_ms"] = latency
                for key, value in usage.items():
                    if isinstance(value, (int, float)):
                        self.stats["usage"][key] = self.stats["usage"].get(key, 0) + value
                return envelope
            last_error = error
        self.stats["failures"] += 1
        raise CodexError(last_error)


def stdout_bytes_to_text(process: subprocess.Popen) -> str:
    """Best-effort drain after a kill; avoids Windows pipe deadlock diagnostics."""
    try:
        out, _ = process.communicate(timeout=5)
        return out or ""
    except Exception:
        return ""


class CodexCapabilityProbe:
    """doctor-only checks: version/help inspection, no model calls."""

    @staticmethod
    def version(executable: Path) -> str:
        result = subprocess.run([str(executable), "--version"], capture_output=True,
                                text=True, encoding="utf-8", errors="replace",
                                timeout=30,
                                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        if result.returncode != 0:
            raise CodexError("codex_version_probe_failed")
        return (result.stdout or result.stderr).strip()

    @staticmethod
    def exec_help(executable: Path) -> str:
        result = subprocess.run([str(executable), "exec", "--help"], capture_output=True,
                                text=True, encoding="utf-8", errors="replace", timeout=30,
                                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        return (result.stdout or "") + (result.stderr or "")


# Shared cancellation flag (threading-based; tests may set it to abort waits).
class CancelFlag:
    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()
