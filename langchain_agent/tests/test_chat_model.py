"""B. LangChain bridge tests: tool binding, validation, create_agent loop."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage  # noqa: E402

from smac_langchain.llm.codex_client import CodexCLI, CodexError  # noqa: E402
from smac_langchain.llm.codex_chat_model import (  # noqa: E402
    CodexCLIChatModel, MockTacticalChatModel, serialize_messages, validate_tool_call)


def make_model(tmp_path, envelope, tools):
    exe = tmp_path / "codex.exe"
    exe.write_text("placeholder", encoding="utf-8")
    body = (
        "import sys, json\n"
        "sys.stdin.read()\n"
        "sys.stdout.write(json.dumps({'type': 'turn.completed'}) + '\\n')\n"
        f"open(sys.argv[1], 'w', encoding='utf-8').write({envelope!r})\n"
    )

    def factory(schema_path, output_path, workdir):
        return [sys.executable, "-c", body, str(output_path)]

    client = CodexCLI(model="m", executable=exe, request_root=tmp_path / "req",
                      max_retries=0, command_factory=factory)
    model = CodexCLIChatModel(client=client)
    return model.bind_tools(tools)


def dummy_tools():
    from langchain_core.tools import tool

    @tool
    def get_battle_state() -> str:
        """Read battle state."""
        return "{}"

    @tool
    def execute_skill_window(choice_id: str, expected_snapshot_id: str,
                             decision_steps: int) -> str:
        """Execute a skill window."""
        return "{}"

    return [get_battle_state, execute_skill_window]


def test_plain_answer(tmp_path):
    envelope = json.dumps({"kind": "final", "tool_name": None,
                           "arguments_json": None, "final_text": "hello"})
    model = make_model(tmp_path, envelope, dummy_tools())
    result = model.invoke([HumanMessage("hi")])
    assert result.content == "hello" and not result.tool_calls
    assert model._llm_type if hasattr(model, "_llm_type") else True


def test_legal_tool_call(tmp_path):
    envelope = json.dumps({"kind": "tool_call", "tool_name": "execute_skill_window",
                           "arguments_json": json.dumps(
                               {"choice_id": "9", "expected_snapshot_id": "s1",
                                "decision_steps": 4}),
                           "final_text": None})
    model = make_model(tmp_path, envelope, dummy_tools())
    result = model.invoke([HumanMessage("go")])
    assert len(result.tool_calls) == 1
    call = result.tool_calls[0]
    assert call["name"] == "execute_skill_window"
    assert call["args"]["choice_id"] == "9"


def test_unknown_tool_rejected(tmp_path):
    envelope = json.dumps({"kind": "tool_call", "tool_name": "rm_rf",
                           "arguments_json": "{}", "final_text": None})
    model = make_model(tmp_path, envelope, dummy_tools())
    with pytest.raises(CodexError, match="codex_unknown_tool"):
        model.invoke([HumanMessage("go")])


def test_illegal_arguments_rejected(tmp_path):
    for bad_args in ('{"choice_id": "1"}',  # missing required
                     '{"choice_id":"1","expected_snapshot_id":"s","decision_steps":4,"shell":"dir"}'):  # unknown arg
        envelope = json.dumps({"kind": "tool_call", "tool_name": "execute_skill_window",
                               "arguments_json": bad_args, "final_text": None})
        model = make_model(tmp_path, envelope, dummy_tools())
        with pytest.raises(CodexError):
            model.invoke([HumanMessage("go")])


def test_tool_result_correlation_serialized():
    messages = [HumanMessage("go"), AIMessage("", tool_calls=[
        {"id": "call_1", "name": "execute_skill_window",
         "args": {"choice_id": "1", "expected_snapshot_id": "s", "decision_steps": 4}}]),
        ToolMessage("{}", tool_call_id="call_1")]
    serialized = serialize_messages(messages)
    assert serialized[1]["tool_calls"][0]["id"] == "call_1"
    assert serialized[2]["tool_call_id"] == "call_1"
    assert serialized[2]["role"] == "tool"


def test_bindings_isolated(tmp_path):
    """Same model bound to different tool sets must not cross-contaminate."""
    call_envelope = json.dumps({"kind": "tool_call", "tool_name": "get_battle_state",
                                "arguments_json": "{}", "final_text": None})
    tools = dummy_tools()
    model = make_model(tmp_path, call_envelope, tools)
    assert model.invoke([HumanMessage("a")]).tool_calls[0]["name"] == "get_battle_state"
    # different binding with only execute tool: get_battle_state must now be rejected
    envelope2 = json.dumps({"kind": "tool_call", "tool_name": "get_battle_state",
                            "arguments_json": "{}", "final_text": None})
    client = CodexCLI(model="m", executable=tmp_path / "codex.exe",
                      request_root=tmp_path / "req2", max_retries=0)
    body = (
        "import sys, json\nsys.stdin.read()\n"
        "sys.stdout.write(json.dumps({'type': 'turn.completed'}) + '\\n')\n"
        f"open(sys.argv[1], 'w', encoding='utf-8').write({envelope2!r})\n"
    )
    client.command_factory = (lambda schema_path, output_path, workdir:
                              [sys.executable, "-c", body, str(output_path)])
    model2 = CodexCLIChatModel(client=client).bind_tools([tools[1]])
    with pytest.raises(CodexError, match="codex_unknown_tool"):
        model2.invoke([HumanMessage("b")])


def test_with_structured_output_compatible(tmp_path):
    """with_structured_output(function_calling) works through this adapter."""
    from pydantic import BaseModel

    class Verdict(BaseModel):
        action: str
        confidence: float

    envelope = json.dumps({"kind": "tool_call", "tool_name": "Verdict",
                           "arguments_json": json.dumps(
                               {"action": "attack", "confidence": 0.9}),
                           "final_text": None})
    model = make_model(tmp_path, envelope, dummy_tools())
    structured = model.with_structured_output(Verdict, method="function_calling")
    verdict = structured.invoke([HumanMessage("decide")])
    assert isinstance(verdict, Verdict) and verdict.action == "attack"


def test_create_agent_full_loop(tmp_path):
    """Real create_agent tool loop with the mock model (offline)."""
    from langchain.agents import create_agent

    model = MockTacticalChatModel()
    tools = dummy_tools()
    agent = create_agent(model, tools, system_prompt="test")
    result = agent.invoke({"messages": [HumanMessage("start")]},
                          config={"recursion_limit": 30})
    final = result["messages"][-1]
    assert isinstance(final, AIMessage)
    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert tool_messages, "the loop must actually execute tools"
    # request/result pairing intact
    for message in tool_messages:
        assert message.tool_call_id
