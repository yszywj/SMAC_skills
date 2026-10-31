# RESULTS（截至 2026-10-30）

按任务书 §14 的分类如实列出。没有任何真实对局评测被 pretended。

| 阶段 | 状态 |
|---|---|
| 代码已实现 | ✅ 接入通道（默认关闭/影子/激活）、特征 schema、标签构造、树训练器、导出器、规则运行时、教师 preflight、CLI/脚本骨架 |
| 离线测试已完成 | ✅ 全部通过，见下 |
| 教师已验证 | ❌ BLOCKED_TEACHER_WEIGHTS（未下载未审计） |
| 真实训练已完成 | ❌（无真实教师标签；合成冒烟 ≠ 训练，见下） |
| 真实对局已完成 | ❌（依赖教师/SC2，未启动） |
| 是否存在提升证据 | ❌ 无任何胜率结论；本轮不含任何真实对局 |

## 测试记录（全部真实运行）

1. `tree_skills/tests/test_pair_scorer_channel.py` — 9 用例通过（SMACv2 venv, Py3.10）：
   - 默认 off 与原结果逐位一致；
   - **A-plug 等价：60 个随机种子下 OriginalPairScorer 经新通道完整复现原匹配、原动作**；
   - `terran_combat` 级 A-plug 全动作序列表一致；
   - 影子模式不改动作；激活模式只改参与集合内部配对且一对一合法；全负分仍分配；scorer 异常/非有限分回退原配对并记录；不重开非法/满血目标。
2. `tree_skills/tests/test_features.py` — 6 用例通过（SMACv2 venv）：固定长度 33、schema 版本/名单不符拒绝加载、缺历史显式标记（≠0 掉血）、**单位重编号不变性、候选顺序不变性**、候选增多只增行不增维。
3. `tree_skills/tests/test_rule_model.py` — 5 用例通过（训练 Python 3.13, sklearn 1.7.0）：2000 随机行、约 400 个分裂阈值 × 每特征 × {阈值、float32 阈值、双侧 float32 邻近位、双侧 nextafter} 边界、极值行，**原模型 vs 导出规则最大绝对误差 ≤1e-6**；500 组联合分配一致；NaN 语义确定性。
4. 合成闭环冒烟 `tree_skills/scripts/demo_synthetic_loop.py` — 通过：合成标签 shard → 按局划分验证 → 组权重归一训练 GBDT → 导出 → 在 SMACv2 venv 中经 TreePairScorer 注入 `assign_healing` 运行 10 个随机场景。**仅为机制证明，标签来自合成替身，不构成蒸馏/胜率证据**。
5. 既有回归：`skills/current/tests` 172 用例 + 2 个相对导入模块 13 用例，全部通过（注入默认关闭零破坏）。两个模块在 `discover` 方式下的 ImportError 为既有相对导入与发现方式不兼容，按包方式运行即通过，与本次改动无关。

## 已知边界（如实声明）

- `_forced_matching` 穷举上限 8 个参与医疗机（与原 `_exact_matching` 上限一致）；超过抛错不静默。
- A-plug 等价测试覆盖 2 医疗机 × 3 候选随机场景与固定 4 单位 terran_combat 场景；并列分数在两套枚举中均按 (分数降序, 目标编号) 确定性处理，60 种子未见分歧。
- 特征 v1 为初始设计（33 维），不承诺最佳。
