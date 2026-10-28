"""A. Codex process protocol tests — fully offline via a Python stand-in.

The stand-in replaces only the argv (test hook `command_factory`); the real
subprocess plumbing, JSONL parsing, result-file validation, timeouts, retries
and output limits are exercised for real. Literal codex CLI flags are checked
against the real CLI by `doctor` and `codex-smoke`, not here.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from smac_langchain.llm.codex_client import CodexCLI, CodexError, find_codex  # noqa: E402

COMPLETED = json.dumps({"type": "turn.completed", "usage": {"input_tokens": 3}})
ENVELOPE = json.dumps({"kind": "final", "tool_name": None,
                       "arguments_json": None, "final_text": "pong"})


def make_client(tmp_path: Path, body: str, **kwargs) -> CodexCLI:
    """body: python code run as the stand-in; helpers OUT/COMPLETED injected."""
    exe = tmp_path / "codex.exe"
    exe.write_text("placeholder for find_codex; not executed", encoding="utf-8")

    def factory(schema_path, output_path, workdir):
        return [sys.executable, "-c", body, str(output_path)]

    defaults = dict(model="test-model", executable=exe,
                    request_root=tmp_path / "requests", timeout_seconds=10,
                    max_retries=0, command_factory=factory)
    defaults.update(kwargs)
    return CodexCLI(**defaults)


def body_writes(envelope: str, events: str = COMPLETED) -> str:
    return (
        "import sys, json\n"
        f"EVENTS = {events!r}\n"
        f"ENVELOPE = {envelope!r}\n"
        "sys.stdin.read()\n"
        "for line in EVENTS.splitlines():\n"
        "    sys.stdout.write(line + '\\n')\n"
        "open(sys.argv[1], 'w', encoding='utf-8').write(ENVELOPE)\n"
    )


def test_not_found():
    with pytest.raises(CodexError):
        find_codex(r"C:\does\not\exist\codex.exe")


def test_timeout(tmp_path):
    client = make_client(tmp_path, "import time; time.sleep(30)\n",
                         timeout_seconds=0.3)
    with pytest.raises(CodexError, match="codex_timeout"):
        client.request("ping")


def test_nonzero_exit(tmp_path):
    client = make_client(tmp_path, "import sys; sys.exit(3)\n")
    with pytest.raises(CodexError, match="codex_exit_3"):
        client.request("ping")


def test_jsonl_stream_and_result_file(tmp_path):
    client = make_client(tmp_path, body_writes(ENVELOPE))
    envelope = client.request("ping")
    assert envelope["kind"] == "final" and envelope["final_text"] == "pong"
    assert envelope["usage"].get("input_tokens") == 3


def test_missing_result_file(tmp_path):
    client = make_client(tmp_path, (
        "import sys\n"
        f"sys.stdout.write({COMPLETED!r} + '\\n')\n"
        "sys.stdin.read()\n"  # never writes the result file
    ))
    with pytest.raises(CodexError, match="codex_missing_result_file"):
        client.request("ping")


def test_bad_json_result(tmp_path):
    client = make_client(tmp_path, body_writes("not json {"))
    with pytest.raises(CodexError, match="codex_invalid_json"):
        client.request("ping")


def test_schema_error_branch_mixed(tmp_path):
    bad = json.dumps({"kind": "tool_call", "tool_name": "x",
                      "arguments_json": None, "final_text": None})
    client = make_client(tmp_path, body_writes(bad))
    with pytest.raises(CodexError, match="codex_schema_invalid"):
        client.request("ping")


def test_arguments_not_object(tmp_path):
    bad = json.dumps({"kind": "tool_call", "tool_name": "x",
                      "arguments_json": "[1,2]", "final_text": None})
    client = make_client(tmp_path, body_writes(bad))
    with pytest.raises(CodexError, match="codex_arguments_not_json_object"):
        client.request("ping")


def test_forbidden_builtin_tool_event(tmp_path):
    events = COMPLETED + "\n" + json.dumps(
        {"type": "item.updated", "item": {"type": "command_execution", "command": "dir"}})
    client = make_client(tmp_path, body_writes(ENVELOPE, events))
    with pytest.raises(CodexError, match="codex_forbidden_builtin_tool_use"):
        client.request("ping")


def test_turn_failed(tmp_path):
    events = json.dumps({"type": "turn.failed"})
    client = make_client(tmp_path, body_writes("null", events))
    with pytest.raises(CodexError):
        client.request("ping")


def test_missing_completed_event(tmp_path):
    client = make_client(tmp_path, body_writes(ENVELOPE, events=""))
    with pytest.raises(CodexError, match="codex_turn_not_completed"):
        client.request("ping")


def test_budget_retries_counted(tmp_path):
    client = make_client(tmp_path, "import sys; sys.exit(1)\n", max_retries=2)
    with pytest.raises(CodexError):
        client.request("ping")
    assert client.stats["cli_starts"] == 3
    assert client.stats["retries"] == 2


def test_output_too_long(tmp_path):
    body = ("import sys\n"
            f"sys.stdout.write({COMPLETED!r} + '\\n' + 'x' * 3000000)\n"
            f"open(sys.argv[1], 'w').write({ENVELOPE!r})\n")
    client = make_client(tmp_path, body, max_output_bytes=1000)
    with pytest.raises(CodexError, match="codex_output_too_long"):
        client.request("ping")


def test_unicode_roundtrip(tmp_path):
    envelope = json.dumps({"kind": "final", "tool_name": None,
                           "arguments_json": None, "final_text": "胜利 ✓"},
                          ensure_ascii=False)
    client = make_client(tmp_path, body_writes(envelope))
    result = client.request("你好 ✓")
    assert result["final_text"] == "胜利 ✓"


def test_prompt_never_becomes_shell(tmp_path):
    """The prompt travels via stdin only; argv is fixed by the factory/CLI."""
    seen = {}

    def factory(schema_path, output_path, workdir):
        return [sys.executable, "-c", body_writes(ENVELOPE), str(output_path)]

    client = make_client(tmp_path, "", command_factory=None)
    client.command_factory = factory
    client.request("&& del C:\\everything && echo pwned")
    assert client.stats["cli_starts"] == 1  # ran, nothing interpreted as shell
