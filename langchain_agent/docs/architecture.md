# 架构：从 Agent 到环境

```
LangChain create_agent (src/smac_langchain/agent.py)
  │  system prompt: prompts/tactical_system.md
  │  middleware: BudgetGuard + TrimHistory (middleware.py)
  ▼
CodexCLIChatModel (llm/codex_chat_model.py)   ← 每个模型轮一个 codex exec 子进程
  │  受限 JSON envelope → AIMessage / tool_calls（每轮最多 1 个工具调用）
  ▼
LangChain Tools (tools.py)                    ← 4 个战斗工具，闭包注入依赖
  get_battle_state / list_available_skills / execute_skill_window / get_last_execution_report
  ▼
BattleCoordinator (runtime/coordinator.py)    ← 校验、预算、幂等、单写入者
  │  冻结快照与候选映射；command_id 去重；stale 拒绝；backend 错误→unknown→停局
  ▼
Backend (backends/)
  ├─ MockBackend (mock.py)：合成战斗 + 真实 SkillManager（离线闭环）
  └─ RPCClientBackend (rpc_client.py) ⇄ workers/smacv2_worker.py（JSONL stdio）
        └─ 原 sc2_skill_adapter.bridge：build_request / nearest_actions /
           action_feedback / ReplayableEnv + skill_runtime.load_manager
  ▼
SkillManager（skills/current，未改动）
  └─ adaptive_focus_fire / focus_fire_budget / cover_retreat /（受开关限制的空间技能）
  ▼
动作掩码校验 → commit_final_actions → env.step → 新观察 → ExecutionReport
  → ToolMessage → Agent 下一轮决策
```

## 三种状态的归属

- **Agent state**：LangChain 消息与近期决策摘要（InMemorySaver 仅为短期状态演示，
  不是 SC2 进程恢复机制）。
- **SkillManager state**：姿态、All-in、撤退历史、接敌记忆——活在 backend 进程内，
  每局 reset，窗口续期不清空。Agent 通过“选模式 + 有限时段”影响它，不重写其规则。
- **环境 state**：真实单位与时钟，只在 backend/worker 内接触。

## adaptive 的特殊位置

adaptive 是有内部编排与局内记忆的组合控制器。Agent 每次执行命令选择
模式（含姿态）并给出步数上限；worker/mock 在窗口内：
每步取新鲜观察 → `manager.step` →（空间技能部分提案按 nearest 基线合并，
interrupted 空提案同快照立即交回 adaptive，记录 `spatial_interrupt_handoff`）
→ 掩码校验 → `commit_final_actions` → `env.step`。
`max_steps` 例行终止时同模式续期（不清历史）；其余终止原因上报为
`skill_complete` 等，不冒充成功。每步的推测性数值（predicted damage 等）
均为启发式估计，报告中原样标注。

## 控制权与安全

- 所有环境写入串行走 coordinator；Agent 模式下 worker 不运行 auto 调度。
- 模型可见数据 = 团队可见事实 + 公开规则 + 筛选后的历史摘要；评测隐藏数据
  只写入 `hidden_events.jsonl`，永不进入 prompt/ToolMessage。
- Codex 子进程：只读 sandbox、禁用 shell/apps/hooks/browser/computer_use、
  web_search 关闭、每请求独立目录与结果文件、逐行解析 JSONL 事件流、
  出现任何内置工具使用事件即拒绝该结果。战斗工具只在主进程执行。
