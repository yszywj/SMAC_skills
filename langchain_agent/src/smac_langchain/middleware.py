"""Agent middleware: budgets, structured errors, history trimming.

- BudgetGuardMiddleware: per-episode model/tool caps, wall clock, and tool-call
  id tracking (never retries environment writes).
- ToolCallIdMiddleware: records the current tool_call_id for the coordinator's
  locally generated command ids.
- TrimHistoryMiddleware: keeps tool request/result pairs intact while bounding
  history length, without extra LLM summarization calls.
"""
from __future__ import annotations

import time
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, BaseMessage, RemoveMessage

from . import tools as battle_tools


class BudgetGuardMiddleware(AgentMiddleware):
    """Fails closed when a budget is exhausted; never auto-retries writes."""

    def __init__(self, limits: dict[str, Any], emit=None):
        super().__init__()
        self.limits = limits
        self.spent = {"model_calls": 0, "tool_calls": 0}
        self.emit = emit or (lambda *_a, **_k: None)
        self.clock_started = time.monotonic()

    def before_model(self, state, runtime):
        self.spent["model_calls"] += 1
        limit = self.limits.get("max_model_calls_per_episode")
        if limit is not None and self.spent["model_calls"] > limit:
            raise RuntimeError(f"budget_exceeded:model_calls:{self.spent['model_calls']}")
        wall = self.limits.get("wall_clock_timeout_seconds")
        if wall is not None and time.monotonic() - self.clock_started > wall:
            raise RuntimeError("budget_exceeded:wall_clock")
        self.emit("model_call", **self.spent)
        return None

    def wrap_tool_call(self, request, handler):
        self.spent["tool_calls"] += 1
        limit = self.limits.get("max_tool_calls_per_episode")
        if limit is not None and self.spent["tool_calls"] > limit:
            raise RuntimeError(f"budget_exceeded:tool_calls:{self.spent['tool_calls']}")
        battle_tools.set_current_call_id(request.tool_call.get("id"))
        try:
            return handler(request)
        except Exception as exc:  # structured error back to the model
            self.emit("tool_error", tool=request.tool_call.get("name", "?"),
                      error=str(exc)[:300])
            return f"ERROR: {type(exc).__name__}: {exc}"
        finally:
            battle_tools.set_current_call_id(None)


class TrimHistoryMiddleware(AgentMiddleware):
    """Drop oldest complete (tool_call -> tool result) pairs beyond `keep_pairs`."""

    def __init__(self, keep_pairs: int = 6):
        super().__init__()
        self.keep_pairs = keep_pairs

    def after_model(self, state, runtime):
        messages: list[BaseMessage] = state.get("messages", [])
        tool_calls = sum(bool(getattr(m, "tool_calls", None)) for m in messages
                         if isinstance(m, AIMessage))
        if tool_calls <= self.keep_pairs:
            return None
        # remove the oldest pairs while never orphaning a tool result
        removals: list[BaseMessage] = []
        pending_call_ids: set[str] = set()
        removed_pairs = 0
        for message in messages:
            if removed_pairs >= tool_calls - self.keep_pairs:
                break
            if isinstance(message, AIMessage) and getattr(message, "tool_calls", None):
                if message.id:
                    pending_call_ids.add(message.id)
                removals.append(message)
                removed_pairs += 1
            elif pending_call_ids and getattr(message, "tool_call_id", None) in pending_call_ids:
                removals.append(message)
                pending_call_ids.discard(message.tool_call_id)
        if removals:
            return {"messages": [RemoveMessage(id=m.id) for m in removals if m.id]}
        return None
