# Codex CLI ⇄ LangChain 协议桥接

**性质声明**：这是“Codex CLI 受限 JSON 输出 → LangChain tool call”的**自定义
协议桥接**。它不是官方的 Codex/LangChain 集成，也不是 Codex 原生调用了我们的
Python 工具——工具永远由外层 LangChain create_agent 在主进程执行。
Codex 在本系统中只是一个结构化决策后端。

## 两套协议如何转换

LangChain 侧（`llm/codex_chat_model.py`）：
1. `serialize_messages()` 把 System/Human/AI/Tool 消息序列化（保留 role、
   tool_calls、tool_call_id）。
2. `_build_prompt()` 生成单个 JSON 文本：消息 + 实际绑定的工具定义 +
   协议说明（`bind_tools()` 只影响本次绑定的工具集，不同绑定互不污染）。

CLI 侧（`llm/codex_client.py::CodexCLI.request`）：
1. 每次请求独立目录：`envelope.schema.json`（--output-schema）、`workdir/`、
   `last_message.json`（--output-last-message）。
2. argv 列表调用（绝不 shell=True），prompt 经 UTF-8 stdin。
3. stdout 是 **JSONL 事件流**，逐行 `json.loads`：要求 `turn.completed`；
   `error`/`turn.failed` 失败；任何 `command_execution / file_change /
   mcp_tool_call / web_search / patch_apply` item → 整个结果作废
   （不是靠提示词“请不要用工具”，而是事件级拒绝）。
4. 结果文件按 envelope Schema 校验；`tool_call` 分支再把 `arguments_json`
   解析为 JSON 对象并二次校验工具名/参数（未知名/缺参/类型不符 → 报错）。
   CLI 本身拒绝顶层 allOf/enum 组合，因此互斥性（tool_call vs final）由
   `_validate_envelope` 在代码里强制。
5. envelope → `AIMessage`（final_text）或带 `tool_calls` 的 AIMessage
   （每轮最多一个工具调用）。usage 仅记录 CLI 实际上报的数值。

## 鉴权与权限边界

- CLI 使用它自己的既有登录；本项目不读取、不复制、不打印令牌，不自动登录，
  不修改用户全局 Codex 配置。
- 每个子进程：`--ignore-user-config --ephemeral --skip-git-repo-check
  --sandbox read-only`，禁用 apps/shell_tool/hooks/browser_use/
  computer_use，`web_search=disabled`，`project_doc_max_bytes=0`。
  允许非仓库目录的 CLI 选项不是 Git 操作，不为此初始化仓库。
  限制只作用于本次子进程；不使用跳过 sandbox 的选项。
- 超时/取消只清理自己启动的进程树（Windows `taskkill /T`）；输出大小有上限。
- 模型/推理强度/超时/重试次数在 `configs/*.toml` 配置；`model="default"`
  使用 CLI 账户默认模型，不硬编码可能不可用的模型名。

## 不支持的功能（明确声明）

- token 流式输出：不支持，也不伪造（可上报真实的节点/工具进度事件）。
- 每个模型响应多个工具调用（并行工具请求）：首版不做。
- provider-native structured output / 官方 LangChain 集成：不伪装；
  `with_structured_output(method="function_calling")` 兼容已验证。
- Codex 直接运行游戏、调用 shell、编辑项目、递归启动自己：被协议与事件
  校验双重排除。
