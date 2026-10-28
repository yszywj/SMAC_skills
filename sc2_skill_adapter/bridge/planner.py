# Adapted from sibling game_skill/micro/jev_micro; see SOURCE_MANIFEST.json.
"""One Codex-authenticated Astra plan per map, reused by fixed-step JEV."""

import asyncio
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time

from .semantics import planning_input
from .policy import TEAM_CORE_FIELDS, ORDER_FIELDS, ENEMY_FIELDS, BATCH_RELATION_FIELDS


HIERARCHY_SCHEMA_VERSION = 7
TEXT_FIELDS = ("objective", "opening", "target_selection", "movement_and_cooldowns")
LIST_FIELDS = ("priorities", "contingencies", "assumptions")
PLAN_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "map": {"type": "string"},
        **{key: {"type": "string"} for key in TEXT_FIELDS},
        **{key: {"type": "array", "items": {"type": "string"}} for key in LIST_FIELDS},
        "unit_roles": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "properties": {"unit_type": {"type": "string"}, "guidance": {"type": "string"}},
            "required": ["unit_type", "guidance"]}},
    },
}
PLAN_SCHEMA["required"] = list(PLAN_SCHEMA["properties"])

from .semantics import PLANNER_INSTRUCTIONS as INSTRUCTIONS


class PlannerError(RuntimeError):
    pass


def validate_plan(value, map_name):
    if not isinstance(value, dict) or set(value) != set(PLAN_SCHEMA["required"]):
        raise PlannerError("planner_invalid_fields")
    if value["map"] != map_name:
        raise PlannerError("planner_wrong_map")
    def valid_text(text, limit):
        return isinstance(text, str) and 0 < len(text.strip()) <= limit
    if any(not valid_text(value[key], 1000) for key in TEXT_FIELDS):
        raise PlannerError("planner_invalid_text")
    for key in LIST_FIELDS:
        if (not isinstance(value[key], list) or len(value[key]) > 6
                or any(not valid_text(item, 500) for item in value[key])):
            raise PlannerError("planner_invalid_" + key)
    roles = value["unit_roles"]
    if not isinstance(roles, list) or not 1 <= len(roles) <= 12:
        raise PlannerError("planner_invalid_roles")
    for role in roles:
        if (not isinstance(role, dict) or set(role) != {"unit_type", "guidance"}
                or not valid_text(role["unit_type"], 80) or not valid_text(role["guidance"], 500)):
            raise PlannerError("planner_invalid_role")
    if len(json.dumps(value, ensure_ascii=False).encode("utf-8")) > 6000:
        raise PlannerError("planner_plan_too_long")
    return copy.deepcopy(value)



def attach_plan(payload, active):
    """Preserve every legal candidate; JEV alone chooses the concrete actions."""
    if payload["state"]["map"] != active["plan"]["map"]:
        raise PlannerError("planner_wrong_map")
    result = copy.deepcopy(payload)
    result["state"].update(schema_version=HIERARCHY_SCHEMA_VERSION, realtime=False,
                           strategic_plan=copy.deepcopy(active))
    for question in result["questions"].values():
        question["instructions"] += (
            " Use state.strategic_plan.plan as the map-level tactical plan. "
            "Apply its conditional priorities to the CURRENT observation and this unit's role. "
            "Choose only from this question's legal criteria; current facts override obsolete assumptions.")
    return result


def find_codex(path=None):
    candidate = Path(path) if path else None
    if candidate is None:
        found = shutil.which("codex")
        candidate = Path(found) if found else None
    if os.name == "nt" and (candidate is None or candidate.suffix.lower() in (".cmd", ".ps1", ".bat")):
        npm = candidate.parent if candidate else Path(os.environ.get("APPDATA", "")) / "npm"
        matches = list((npm / "node_modules" / "@openai" / "codex").glob("**/codex.exe"))
        if matches:
            candidate = matches[0]
    if candidate is None or not candidate.is_file() or candidate.suffix.lower() in (".cmd", ".bat", ".ps1"):
        raise PlannerError("planner_native_codex_not_found")
    return candidate.resolve()


class MapPlanner:
    def __init__(self, directory, emit, model="gpt-6-astra", effort="medium", timeout=120, executable=None):
        self.executable = find_codex(executable)
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=False)
        self.workdir = self.directory / "workspace"
        self.workdir.mkdir()
        self.schema_path = self.directory / "plan.schema.json"
        self.schema_path.write_text(json.dumps(PLAN_SCHEMA), encoding="utf-8")
        self.emit, self.model, self.effort, self.timeout = emit, model, effort, timeout
        self.active = None
        self.stats = {"model": model, "effort": effort, "requests": 0, "successful_responses": 0,
                      "errors": 0, "usage": {}, "scope": "one distribution-level plan per scenario configuration, reused across episodes"}
        self._process, self._closed = None, False
        self._lock = threading.Lock()

    def command(self, output):
        return [str(self.executable), "exec", "--ignore-user-config", "--ephemeral",
                "--skip-git-repo-check", "--sandbox", "read-only", "--model", self.model,
                "--disable", "apps", "--disable", "shell_tool", "--disable", "hooks",
                "--disable", "browser_use", "--disable", "computer_use",
                "--enable", "respect_system_proxy", "--disable", "unbounded_connection_retries",
                "-c", 'web_search="disabled"', "-c", "project_doc_max_bytes=0",
                "-c", f'model_reasoning_effort="{self.effort}"',
                "--output-schema", str(self.schema_path), "--output-last-message", str(output),
                "--json", "--color", "never", "-"]

    def _request(self, prompt):
        result_path = self.directory / "plan.json"
        if result_path.exists():
            raise PlannerError("planner_output_already_exists")
        try:
            with self._lock:
                if self._closed:
                    raise PlannerError("planner_closed")
                process = subprocess.Popen(self.command(result_path), cwd=self.workdir,
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    text=True, encoding="utf-8",
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                self._process = process
            try:
                stdout, _stderr = process.communicate(prompt, timeout=self.timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, _stderr = process.communicate()
                return {"stdout": stdout, "plan_text": "", "error": "planner_timeout"}
            return {"stdout": stdout,
                    "plan_text": result_path.read_text(encoding="utf-8") if result_path.exists() else "",
                    "error": f"planner_codex_exit_{process.returncode}" if process.returncode else None}
        except OSError:
            raise PlannerError("planner_io_error") from None
        finally:
            with self._lock:
                self._process = None

    async def get_plan(self, payload, env):
        scenario_key = hashlib.sha256(json.dumps(planning_input(payload, env),
            sort_keys=True).encode('utf-8')).hexdigest()
        if self.active is not None and self.active['scenario_sha256'] != scenario_key:
            raise PlannerError('planner_wrong_scenario_configuration')
        if self.active is not None:
            if self.active["plan"]["map"] != env.map_name:
                raise PlannerError("planner_wrong_map")
            return copy.deepcopy(self.active)
        data = planning_input(payload, env)
        prompt = INSTRUCTIONS + "\nINPUT JSON:\n" + json.dumps(data, ensure_ascii=False)
        self.stats["requests"] += 1
        self.emit("planner_request", request_id=self.stats["requests"], model=self.model,
                  effort=self.effort, prompt=prompt, output_schema=PLAN_SCHEMA)
        started = time.monotonic()
        try:
            raw = await asyncio.to_thread(self._request, prompt)
            latency = (time.monotonic() - started) * 1000
            self.emit("planner_response", request_id=self.stats["requests"], model=self.model,
                      latency_ms=latency, **raw)
            completed, usage, failed = False, {}, raw["error"]
            for line in raw["stdout"].splitlines():
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if event.get("type") == "turn.completed":
                    completed, usage = True, event.get("usage", {})
                if event.get("type") in ("error", "turn.failed"):
                    failed = failed or "planner_turn_failed"
                if event.get("item", {}).get("type") in ("command_execution", "file_change", "mcp_tool_call", "web_search"):
                    failed = "planner_unexpected_tool_use"
            self.stats.update(usage=usage, latency_ms=latency)
            if failed or not completed or not raw["plan_text"]:
                raise PlannerError(failed or "planner_missing_result")
            try:
                value = json.loads(raw["plan_text"])
            except ValueError:
                raise PlannerError("planner_invalid_json") from None
            plan = validate_plan(value, env.map_name)
            digest = hashlib.sha256(json.dumps(plan, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
            self.active = {"model": self.model, "plan_sha256": digest, "plan": plan,
                           "scenario_sha256": scenario_key}
            self.stats.update(successful_responses=1, plan_sha256=digest,
                              plan_file=str(self.directory / "plan.json"))
            self.emit("plan_accepted", **copy.deepcopy(self.active))
            return copy.deepcopy(self.active)
        except PlannerError as exc:
            self.stats["errors"] += 1
            self.emit("planner_error", error=str(exc))
            raise

    async def close(self):
        with self._lock:
            self._closed = True
            if self._process is not None and self._process.poll() is None:
                self._process.kill()


def add_planner_arguments(parser):
    def timeout(value):
        result = float(value)
        if not math.isfinite(result) or result <= 0:
            raise argparse.ArgumentTypeError("planner timeout must be finite and positive")
        return result
    parser.add_argument("--planner", choices=("none", "codex"), default="none")
    parser.add_argument("--planner-model", default="gpt-6-astra")
    parser.add_argument("--planner-effort", choices=("low", "medium", "high", "xhigh", "max"), default="medium")
    parser.add_argument("--planner-timeout", type=timeout, default=120)
    parser.add_argument("--codex-path", type=Path)
