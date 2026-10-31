# RESULTS（更新于 2026-10-31）

按任务书阶段如实分列。所有"通过"均来自本机实际运行，命令与产物可复查。

| 阶段 | 状态 |
|---|---|
| 权重审计（G1） | ✅ 两个检查点 strict 加载、7,232,256 参数、动作头 20、全 logits 有限且确定（`teacher/CHECKPOINT_AUDIT.md`、`AUDIT_*.json`） |
| 观察/动作映射（G2 管道级） | ✅ 映射模块+测试；观察构建器等价复现官方流水线；常驻服务端到端验证（两种 last-token 变体，`INFERENCE_VALIDATION.md`，synthetic_only） |
| 真实接线（G3） | ✅ **A vs A-plug 真实 SC2 对局 91/91 步全单位最终动作一致**（种子 300–301，`runs/terran-pair-g3-{A,Aplug}-n2-seed300`，`scripts/compare_g3_runs.py` 判 PASS） |
| 教师 B 通道真实冒烟 | ✅ `runs/terran-pair-g3-teacher-n1-seed300`：1 局 strict 全程真实教师评分，48/48 步调用常驻服务、0 失败、0 回退；本局模型影响率 0（1 医疗机对局，教师排序与原规则一致——属正常覆盖情况，非故障）。远端结论仅此，无胜率声称 |
| G4 教师资格（40 局筛选→200 局验证） | ❌ 未开始（见 BLOCKERS） |
| 真实轨迹采集/标注/树训练/导出 | ❌ 依赖 G4；训练器/导出器/指标已就绪并经合成冒烟验证 |
| C/D 最终对照 | ❌ 未开始 |

## 本轮新增测试（全部真实运行通过）

1. `tree_skills/tests/test_action_mapping.py` — 6 用例（原生布局、双向映射、keep 扩展/越界/填充槽拒绝、头宽不足拒绝、6+j 双语义）。
2. `tree_skills/tests/test_p1_fixes.py` — 6 用例（**pair_distance 按配对医疗机取值修复**、**威胁方向修复**（in_enemy_weapon_range vs 自身无武器）、注入层历史传递与缺失标记、匹配参考实现 300 随机一致性、**>8 参与医疗机严格拒绝**、不可行返回 None）。
3. `teacher/validate_inference.py` — 常驻服务端到端（last-token on/off 两变体）：批量 logits、有限性、跨服务重启确定性、重复步/越界防护。
4. 回归：tree_skills 27 用例 + skills 172+13 用例 + **adapter 全套 65 用例**全部通过（runner/skill_runtime 改动默认关闭零破坏）。

## P1-4 修复清单（已全部完成并有测试）

- `pair_distance` 改为当前 healer-target 配对距离（H0/H1 不同距离测试）
- 威胁方向：只用敌方对己方射程信息（`enemy_weapon_range_status`/`in_enemy_weapon_range`）；医疗机自身无武器不再被当作"敌方无威胁"；未知保持 missing
- 历史特征：注入层新增 `pair_histories` 通道（runner 的 HealthHistories 为训练/部署唯一生成器）；不足时显式 missing 标记，不用伪值
- 训练入口统一 `python -m tree_skills.training.train_tree`（`train-export.ps1` 已改）
- 验证指标新增 **joint_assignment_agreement**（按 episode/game_loop 聚合全部参与医疗机、用与部署一致的 forced matching 比较）与单候选率
- 匹配参考实现 `medivac/matching.py`（>8 严格报错，与注入层实现 300 随机输入逐位一致）
- 测试/脚本去除硬编码绝对路径（相对项目根定位）

## 运行产物

- `sc2_skill_adapter/runs/terran-pair-g3-A-n2-seed300`（A：1胜1负）
- `sc2_skill_adapter/runs/terran-pair-g3-Aplug-n2-seed300`（A-plug：1胜1负，逐步动作与 A 完全一致）
- `tree_skills/teacher/INFERENCE_VALIDATION_run.json`（synthetic_only 标注）
- `sc2_skill_adapter/runs/terran-pair-g3-teacher-n1-seed300`（teacher-score strict，summary 含 pair_mode/pair_failure_policy）
