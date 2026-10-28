# LangChain 学习路径（对照本项目代码）

按 Models/Messages → Tools → create_agent → Runtime Context → Structured
Output → Middleware → State/Tracing 排列，每步给出对应源码和一个可运行小实验。

## 1. Models & Messages
- 代码：`src/smac_langchain/llm/codex_chat_model.py` 的 `CodexCLIChatModel`
  （继承 `langchain_core.language_models.chat_models.BaseChatModel`，实现
  `_generate` 与 `_llm_type`）；`serialize_messages()` 展示 role/tool_call_id/
  arguments/result 的对应保存。
- 概念：消息列表就是模型输入；AIMessage.tool_calls 是结构化输出，ToolMessage
  必须回填对应 tool_call_id。
- 实验：`pytest tests/test_chat_model.py::test_plain_answer -q`，再读
  `test_tool_result_correlation_serialized`。

## 2. Tools
- 代码：`src/smac_langchain/tools.py`。`@tool` 装饰器 + pydantic args_schema；
  执行依赖（coordinator、会话、预算）通过闭包注入，不暴露为模型参数。
- 概念：工具的参数 schema 就是模型的行动空间；未知工具/未知参数在
  `validate_tool_call`（codex_chat_model.py）被拒绝。
- 实验：`pytest tests/test_chat_model.py::test_illegal_arguments_rejected -q`。

## 3. create_agent（工具循环）
- 代码：`src/smac_langchain/agent.py::build_agent`。
- 概念：create_agent = 模型 + 工具 + 状态图；agent.invoke 内部完成
  “模型→工具→结果→模型”循环直到 AIMessage 无 tool_calls。
- 实验：`python -m smac_langchain demo`（离线全循环），读
  `runs/demo-*/events.jsonl` 里的 model_call/execute_completed 事件顺序。

## 4. Runtime Context（依赖注入）
- 代码：`tools.py::build_battle_tools(coordinator)` —— 每局构建一次工具集，
  coordinator/budget/snapshot 绑定在闭包里；`middleware.py` 用 contextvar
  把当前 tool_call_id 递给 coordinator 生成本地 command_id。
- 概念：模型永远填业务字段；身份、写入预算由本地运行时控制。
- 实验：`pytest tests/test_coordinator.py::TestCandidates -q`。

## 5. Structured Output
- 代码：`schemas.py::ENVELOPE_SCHEMA` + `codex_client._validate_envelope` +
  `_tool_spec` 对 pydantic class 的归一化。
- 概念：`with_structured_output(X, method="function_calling")` 只是
  bind_tools + 输出解析；我们的桥接支持它（tests/test_chat_model.py::
  test_with_structured_output_compatible）。不支持 provider-native
  structured output，不伪装。
- 实验：改 `configs/local.toml` 的 codex 模型后
  `python -m smac_langchain codex-smoke --live-llm`。

## 6. Middleware
- 代码：`middleware.py` —— `BudgetGuardMiddleware`（before_model 计数、
  wrap_tool_call 捕获异常转结构化错误，绝不自动重试环境写），
  `TrimHistoryMiddleware`（保留完整工具请求/结果配对地裁剪历史，无额外
  LLM 总结）。
- 概念：中间件是环绕模型调用/工具调用的钩子链，适合预算、审计、改写。
- 实验：`pytest tests/test_coordinator.py::TestFailures -q`。

## 7. State & Tracing
- 代码：`agent.py` 的 InMemorySaver（thread_id = run_id+episode；仅短期
  状态，重启后不能拿来操控新环境）；`telemetry.py` 的 JSONL 事件（含
  源码 SHA-256、依赖版本、CLI 启动数、延迟、usage）。
- 概念：checkpointer 保存消息图状态；tracing 记录调用链。LangSmith 预留
  可选接入（config `telemetry.langsmith`），默认关闭，无账号也能全测。
- 实验：跑一次 demo 后逐行读 `runs/demo-*/events.jsonl`。
