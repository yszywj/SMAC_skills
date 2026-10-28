"""Codex CLI as a LangChain BaseChatModel — custom protocol bridge.

IMPORTANT: this is a *custom bridge* from Codex CLI constrained JSON output to
LangChain tool calls. It is NOT an official Codex/LangChain integration, and
Codex does NOT natively invoke our Python tools: the outer LangChain agent
runtime executes every tool.

Protocol per model turn (single tool call at most, first version):
  {"kind": "tool_call", "tool_name": str, "arguments_json": "<json object>"}
  {"kind": "final",     "tool_name": null, "arguments_json": null,
   "final_text": str}
Streaming is not implemented (no fake token streams).
"""
from __future__ import annotations

import json
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.messages.ai import UsageMetadata
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import BaseModel, Field

from .codex_client import CodexCLI, CodexError


def serialize_messages(messages: list[BaseMessage]) -> list[dict[str, Any]]:
    """Keep roles, tool call ids, tool arguments and results correlated."""
    serialized = []
    for message in messages:
        entry: dict[str, Any] = {"role": message.type, "content": message.content}
        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            entry["tool_calls"] = [
                {"id": call["id"], "name": call["name"], "arguments": call["args"]}
                for call in tool_calls]
        tool_call_id = getattr(message, "tool_call_id", None)
        if tool_call_id:
            entry["tool_call_id"] = tool_call_id
        name = getattr(message, "name", None)
        if name:
            entry["name"] = name
        serialized.append(entry)
    return serialized


def _tool_spec(tool: Any) -> dict[str, Any]:
    if isinstance(tool, dict):  # already an OpenAI-style tool/function spec
        function = tool.get("function", tool)
        return {"name": function["name"],
                "description": function.get("description", ""),
                "parameters": function.get("parameters", {})}
    if isinstance(tool, type) and issubclass(tool, BaseModel):
        # raw pydantic schema class (e.g. from with_structured_output)
        schema = tool.model_json_schema()
        return {"name": tool.__name__,
                "description": (tool.__doc__ or "").strip(),
                "parameters": {"type": "object",
                               "properties": schema.get("properties", {}),
                               "required": schema.get("required", [])}}
    return {"name": tool.name, "description": tool.description,
            "parameters": {"type": "object", "properties": tool.args,
                           "required": list(tool.args)}}


def validate_tool_call(specs: dict[str, dict], name: str, arguments: dict) -> None:
    """Validate name and arguments against the actually bound tool schemas."""
    if name not in specs:
        raise CodexError(f"codex_unknown_tool:{name}")
    parameters = specs[name]["parameters"]
    for key in parameters.get("required", []):
        if key not in arguments:
            raise CodexError(f"codex_missing_argument:{name}.{key}")
    unknown = set(arguments) - set(parameters.get("properties", {}))
    if unknown:
        raise CodexError(f"codex_unknown_argument:{name}:{sorted(unknown)}")
    for key, value in arguments.items():
        declared = parameters.get("properties", {}).get(key, {})
        declared_type = declared.get("type")
        if declared_type and not _matches_type(value, declared_type):
            raise CodexError(f"codex_argument_type:{name}.{key}")
    for key in ("expected_snapshot_id", "choice_id", "tool_call_id", "command_id"):
        if key in arguments and not isinstance(arguments[key], str):
            raise CodexError(f"codex_argument_type:{name}.{key}")


def _matches_type(value: Any, declared: str) -> bool:
    return {"string": str, "integer": int, "number": (int, float),
            "boolean": bool, "object": dict, "array": list}.get(
        declared, object) and isinstance(value, {
            "string": str, "integer": int, "number": (int, float),
            "boolean": bool, "object": dict, "array": list}.get(declared, object))


class CodexCLIChatModel(BaseChatModel):
    """One Codex subprocess per model turn; structured envelope output."""

    client: CodexCLI = Field(exclude=True)
    prompt_suffix: str = ""
    simulate: bool = False  # never set in production; mock routing aid

    @property
    def _llm_type(self) -> str:
        return "codex-cli-chat"

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        """Accept the actually-bound tool definitions for THIS binding only."""
        specs = {_tool_spec(tool)["name"]: _tool_spec(tool) for tool in tools}
        return self.bind(_bound_tool_specs=specs, **kwargs)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        specs = kwargs.get("_bound_tool_specs") or {}
        prompt = self._build_prompt(messages, specs)
        envelope = self.client.request(prompt)
        usage = envelope.get("usage") or {}
        if envelope["kind"] == "final":
            message = AIMessage(content=envelope["final_text"])
        else:
            name, arguments = envelope["tool_name"], envelope["arguments"]
            validate_tool_call(specs, name, arguments)
            message = AIMessage(
                content="",
                tool_calls=[{"id": _synthetic_call_id(name), "name": name,
                             "args": arguments}],
            )
        if usage:
            message.usage_metadata = UsageMetadata(
                input_tokens=usage.get("input_tokens", 0),
                output_tokens=usage.get("output_tokens", 0),
                total_tokens=usage.get("total_tokens", 0))
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _build_prompt(self, messages: list[BaseMessage],
                      specs: dict[str, dict]) -> str:
        payload = {"messages": serialize_messages(messages),
                   "available_tools": list(specs.values()),
                   "protocol": [
                       "Reply with EXACTLY ONE JSON object matching the schema.",
                       'To request a tool: {"kind":"tool_call","tool_name":<name>,'
                       '"arguments_json":"<serialized JSON object of arguments>",'
                       '"final_text":null}.',
                       'To finish: {"kind":"final","tool_name":null,'
                       '"arguments_json":null,"final_text":<short summary>}.',
                       "At most ONE tool call per reply. Only use listed tools "
                       "and declared argument names.",
                   ]}
        return json.dumps(payload, ensure_ascii=False) + self.prompt_suffix

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model": self.client.model, "effort": self.client.effort}


def _synthetic_call_id(name: str) -> str:
    import uuid
    return f"call_{name}_{uuid.uuid4().hex[:8]}"


class MockTacticalChatModel(BaseChatModel):
    """Offline stand-in (model="mock"): simple deterministic heuristics.

    Not a real model channel. Used only for the offline demo and tests; it
    never launches Codex.
    """

    heuristic: str = "aggressive"
    state: dict[str, Any] = Field(default_factory=dict, exclude=True)

    @property
    def _state(self) -> dict[str, Any]:
        return self.state

    @property
    def _llm_type(self) -> str:
        return "mock-tactical-chat"

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        specs = {_tool_spec(tool)["name"]: _tool_spec(tool) for tool in tools}
        return self.bind(_bound_tool_specs=specs, **kwargs)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        specs = kwargs.get("_bound_tool_specs") or {}
        memory = self._state
        # scan newest tool results for snapshot/candidate/done hints
        for message in reversed(messages):
            if message.type != "tool":
                continue
            text = message.content if isinstance(message.content, str) else ""
            try:
                data = json.loads(text)
            except ValueError:
                continue
            if data.get("done") or data.get("outcome") == "episode_over" or (
                    isinstance(data.get("observation_summary"), dict)
                    and data["observation_summary"].get("done")):
                message = AIMessage(content="mock model: episode finished")
                return ChatResult(generations=[ChatGeneration(message=message)])
            if "snapshot_id" in data and "snapshot_id" not in memory:
                memory["snapshot_id"] = data["snapshot_id"]
            if data.get("candidates"):
                memory["choice_id"] = data["candidates"][0].get("choice_id",
                                                                memory.get("choice_id", "0"))
            break
        tool_name = memory.pop("next_tool", "get_battle_state")
        memory["next_tool"] = {"get_battle_state": "list_available_skills",
                               "list_available_skills": "execute_skill_window",
                               "execute_skill_window": "get_battle_state",
                               "get_last_execution_report": "get_battle_state"}.get(
            tool_name, "get_battle_state")
        import uuid
        call_id = f"call_mock_{uuid.uuid4().hex[:8]}"
        if tool_name not in specs:
            message = AIMessage(content="mock model finished")
            return ChatResult(generations=[ChatGeneration(message=message)])
        if tool_name == "execute_skill_window":
            # heuristic args are patched by the coordinator tests via messages;
            # here we take the latest snapshot/candidate hints from tool results
            snapshot_id, choice_id = memory.get("snapshot_id", "snap-unknown"), \
                memory.get("choice_id", "0")
            for message in reversed(messages):
                text = message.content if isinstance(message.content, str) else ""
                if message.type == "tool":
                    try:
                        data = json.loads(text)
                    except ValueError:
                        continue
                    snapshot_id = data.get("snapshot_id", snapshot_id)
                    candidates = data.get("candidates") or []
                    if candidates:
                        choice_id = candidates[0].get("choice_id", choice_id)
                    break
            args = {"choice_id": str(choice_id), "expected_snapshot_id": snapshot_id,
                    "decision_steps": 4}
        elif tool_name == "list_available_skills":
            args = {"snapshot_id": memory.get("snapshot_id", "snap-unknown")}
            for message in reversed(messages):
                if message.type == "tool":
                    try:
                        data = json.loads(message.content)
                    except ValueError:
                        break
                    args["snapshot_id"] = data.get("snapshot_id", args["snapshot_id"])
                    break
        else:
            args = {}
        message = AIMessage(content="", tool_calls=[
            {"id": call_id, "name": tool_name, "args": args}])
        return ChatResult(generations=[ChatGeneration(message=message)])

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"heuristic": self.heuristic}
