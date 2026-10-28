# 本次实际执行的验证（2026-10-28）

区分三种验证级别：**Mock**（离线模型替身/mock 后端）、**真实 CLI**（本机
Codex 真实调用）、**真实游戏**（SC2 + SMACv2）。结论只与实际证据一致。

## 1. 离线单元/集成测试（pytest，无真实模型、无 SC2）

命令：`.venv\Scripts\python -m pytest tests`（默认排除 live_llm/live_sc2 标记）

**结果：42 passed, 0 failed**，覆盖：

- A Codex 进程协议（tests/test_codex_client.py，14 项）：未找到原生程序、
  超时、非零退出、JSONL 事件流 + 结果文件、缺失结果文件、坏 JSON、Schema
  互斥违规、arguments 非对象、内置工具事件拒绝、turn.failed、缺
  turn.completed、重试预算计数（3 次启动）、输出超长、Unicode 往返、
  prompt 不进 shell。注：这些测试用 `command_factory` 钩子把 argv 换成
  Python 替身（Windows 不能直接执行文本 exe）；子进程管道、JSONL 解析、
  结果文件、超时、重试均为真实执行。字面 codex CLI 旗标由 doctor 与
  codex-smoke 对真实 CLI 验证。
- B LangChain 桥接（tests/test_chat_model.py，8 项）：普通回答、合法工具
  调用、未知工具、非法参数（缺参/未知参）、消息序列化保持调用 ID 对应、
  不同工具绑定隔离、with_structured_output(function_calling) 兼容、
  真正经 create_agent 的完整工具循环。
- C 候选与状态（test_coordinator.py::TestCandidates，8 项）：快照冻结候选、
  stale 快照拒绝、禁用技能带原因隐藏、候选变化后未知 choice 拒绝、重复
  请求幂等（同 ID 同参返回缓存报告且不重复执行）、同 ID 异参拒绝、未知
  目标数量、decision_steps 被配置上限截断。
- D 执行生命周期（TestLifecycle，5 项）：同模式续期不清历史、每局 reset、
  执行窗口跨多个本地 skill 窗口（max_steps 例行终止→续期）、outcome 语义
  区分、局终后拒绝进一步写入。
- E 故障（TestFailures，3 项）：backend 错误→outcome=unknown→停局；预算
  耗尽；stale snapshot 拒绝。
- F 回归（TestRegression，3 项）：pure-skill 路径 codex 启动数=0；离线
  agent 局完成；运行目录不覆盖。

## 2. 离线 CLI 演示（真实 SkillManager，无真实模型/SC2）

- `python -m smac_langchain demo`：**通过**。mock 模型 + create_agent +
  4 工具 + MockBackend（真实 skills/current SkillManager 执行 adaptive
  微操）。结果：rounds=1（内含 8 次模型调用）、2 次 execute_completed、
  final_outcome=win、degraded=false。所有数据标注 simulated=true。
- `python -m smac_langchain run --backend mock --policy pure-skill`：
  **通过**。codex_cli_starts=0，outcome=win。
- `python -m smac_langchain doctor`：**通过**。识别真实 codex.exe（npm shim
  解析）、版本 0.160.1、检查 --output-schema/--output-last-message/--json/
  --sandbox/--ephemeral/--disable 支持、mock 后端自检 ok。

## 3. 真实 Codex CLI（--live-llm，本机登录）

- `codex-smoke --live-llm`：**通过**。1 次 CLI 启动，envelope 校验通过，
  latency ≈ 9.4 s。
- `agent-smoke --backend mock --live-llm`：**通过**。真实 Codex 驱动完整
  LangChain 工具循环（8 次 codex_request 全部 error=null，usage 由 CLI 实报），
  2 次 execute_completed，final_outcome=win，degraded=false，wall ≈ 95 s。
  期间发现并修复了两个真实环境问题：账户不支持硬编码模型名（改为
  model="default"），以及相对路径 --output-schema 在 workdir 下解析失败
  （改为绝对路径）。

## 4. 真实游戏（SC2 + SMACv2）

**未执行。** `run --backend smacv2` 的 worker（workers/smacv2_worker.py，
复用 sc2_skill_adapter.bridge 与 skills/current）已实现但未启动过 SC2：
默认验证策略不自动启动游戏，且本次未申请真实对局预算。阻塞项/前提：
- backend 解释器需可导入 pysc2/smacv2/numpy/yaml（config `paths.backend_python`）；
- SC2 安装与 `10gen_<race>.SC2Map`（本机均存在，路径见 doctor 输出）。
- 系统解释器直接跑 `skills/current/tests`：41/43 通过；2 个模块因该解释器
  缺 httpx 无法导入（**既有环境问题**，与本次改动无关——本次未修改任何
  既有文件）。

## 5. 未验证/受限项清单

- SMACv2 worker 的真实对局（含 replay、episode 统计）——代码就绪，未跑。
- SMACv1：明确 unsupported，未实现也不伪装。
- token 流式输出、并行工具调用、LangSmith 上报：按范围声明不支持/默认关闭。
- 回退路径（模型失败→adaptive 基线）在单元测试中验证；真实 CLI 失败触发
  的回退未专门演练（agent-smoke 全程无失败）。
