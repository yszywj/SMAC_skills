# BLOCKERS（更新于 2026-10-31）

## 1. BLOCKED_TEACHER_WEIGHTS —— 教师权重未下载、未审计

- 原因：HF 入口已核验存在 `marl-gpt-main.pt` / `marl-gpt-wo-last-token.pt`（约 331 MB，MIT），但按资源规范不自动启动大下载；且下载后还需 strict 加载审计（结构、动作头、历史长度）才能判断是否兼容人族 10v10 协议。
- 连带阻塞：真实教师标签 → B 对照 → 真实蒸馏训练 → C/D 真实对局评测 → 学生分布补标 → 最终 A/B/D 对照。这些阶段**不造结果**。
- 已完成：TEACHER_MANIFEST 状态机、preflight 哈希/登记入口、资格流程与统计准则（EXPERIMENT_PROTOCOL.md）。
- 继续入口：`python tree_skills/teacher/preflight.py --checkpoint-dir <含 .pt 的目录>`；随后按 TEACHER_CARD.md §资格流程 2-3 步执行。

## 2. BLOCKED_TEACHER_INFRA —— 教师推理环境未建立

- 需要：官方 marl-gpt 实现 + torch 的独立常驻进程批量推理（观测编号、错误、超时、进程清理契约见任务书 §2）。适配器 venv 明确不装 torch。
- 已完成：`make_pair_scorer("teacher-score")` 在教师未 ready 时显式抛 BLOCKED_TEACHER_WEIGHTS，绝不静默回退冒充学生成绩。

## 3. 未阻塞且已完成的阶段

代码接入、默认关闭等价、A-plug 等价、影子模式、固定参与重匹配、特征 schema、标签构造、树训练器、if/else 导出与数值等价、合成闭环冒烟、既有回归测试 —— 全部真实执行并通过（RESULTS.md）。

## 4. 游戏对局评测（依赖 SC2）

A-plug/影子模式的合成验证已完成；真游戏 A/B/C/D 对照依赖教师（B/C/D 标签或模型），当前全部被 1/2 阻塞。A 与 A-plug 的真游戏等价实验可先行，但需要用户侧 SC2 环境（未在本轮自动启动）。


---

## 2026-10-31 更新：状态总览

- **G1 已完成**：两检查点结构审计通过（CHECKPOINT_AUDIT.md）。
- **G2 真实快照比对**：管道级已过（INFERENCE_VALIDATION.md）；真实轨迹上的官方参考逐字段对照见 teacher/REFERENCE_CHECK_run.json（随 G3.5 采集运行产出）。
- **G3 已完成**：真实对局 A/A-plug 逐步动作+订单+初始哈希等价（compare_g3_runs.py）。
- **G4 资格未完成**：唯一待办。非技术阻塞，而是预算（40 局筛选 + 200 局资格的 SC2 时间）。命令见 docs/G4_COMMANDS.md（PowerShell 可直接复制）。
- `ready` 在真实输入比对与动作映射验收全部通过后授予；`qualified` 只在 G4 资格成立后授予（当前均未授予）。


## 2026-10-31 G4 筛选结果后的状态

- G4 筛选已完成（G4_REPORT.md）：B-main 与 A 胜率完全相同、零胜负改变 → 未通过预注册门槛，200 局资格验证不启动，G5/G6 性能提升路线停止。
- 当前阻塞性质：**研究协议层**。第一版问题（"已决定治疗时治疗谁"）在原规则锁定语义下可影响空间≈0（多候选机会 6.4%，推荐改配 20 次/40 局，且 7/7 冒烟推荐被 keep 层覆盖）。
- 可选后续（需用户决策后重新预注册，不得自行变更）：
  1. 放宽 keep/救援锁定对教师建议的覆盖（改变原语义，需新增对照）；
  2. 使用多医疗机占比更高的阵容分布（偏离 native 0.10 权重）；
  3. 用已保存的 962 组真实标签做"仅测试蒸馏机制"的研究性训练（明确不宣传胜率）。
