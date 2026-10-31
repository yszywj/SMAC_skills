# BLOCKERS（截至 2026-10-30）

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
