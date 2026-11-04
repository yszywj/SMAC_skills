# RESULTS（最新更新于 2026-11-01）

> 最新四组诊断（2026-11-01）：40个新场景各跑A/B/C/D，160局完成；混合筛选集胜局为14/14/12/10，原始分布20局为7/7/5/6。教师尚无优于规则的证据，qualified=false，未训练学生树。分组结果、执行审计和改进顺序见 [TEACHER_CONTROL_DIAGNOSTIC_REPORT.md](TEACHER_CONTROL_DIAGNOSTIC_REPORT.md)。

> 此前token修复复验：完成962组旧轨迹重标、A/B各40局复测和1005组新标签生成。旧场景A=17/40、B=18/40，唯一第37局负转胜；该微小增益未在新四组实验的A/B对照中重复出现。历史结果见 [TOKENFIX_REPORT.md](TOKENFIX_REPORT.md)。以下保留旧实验记录；旧G2参考复制了同一错误，旧G4不能排除教师输入/标签问题。

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


---

## 2026-10-31 第二轮：P0 修复 + G3.5 + G2 真实快照对照（全部真实运行）

### P0 修复（§1）
- **1.1 健康历史决策步语义**：`HealthHistories.observe` 改为每个新 `(episode, game_loop)` 追加一次（含不变血量），同 loop 幂等；死亡=0.0、无效最大生命保持前值、跨局清空；`begin_step` 统一以 game_loop 为时钟。7 个新单测（含 0.8,0.8,0.8,0.6 四步保留四值、net_change_3 语义、同帧幂等）全部通过。
- **1.2 官方语义**：`service.actor_raw_logits` 显式官方 Actor raw-logits 接口（不再依赖 `__mro__`）；服务严格校验 mask 长度（16/17，keep 扩展被识别且永不入教师槽）、二值性、loop 倒退。**last_token 约定确定**：官方 terran 评测 `last_token=True, history_len=6`，但 6×162+1=973 > block_size 700——出具容量计算，取 H=4（649≤700）；G4 候选 B-main=`main.pt+last_token on`、B-alt=`wo-last-token.pt+off`。**真实快照对照（REFERENCE_CHECK_run.json）：24/24 组真实观察，本地构建器 vs 官方函数参考（官方 stack_dict_tensor/tokenizer/race 追加/reshape_obs）逐元素零差异**（切片 6 行为文档化复现，SC2Decomposer 需起游戏无法实例化）。10 条真实多候选医疗机决策快照核对表（REAL_SNAPSHOT_TABLE.json）：20 个候选槽位=6+友军编号、原生 mask 位全为 1，零问题。manifest → **state=ready**（接口级；qualified=false）。
- **1.3 服务生命周期**：`_LineReader` 读线程+队列实现 Windows 可用的真实超时（启动/请求均受约束），request_id 匹配、管道断裂/进程死亡/坏 JSON 结构化报错；整个 run 权重只加载一次。
- **1.4 文档清理**：TEACHER_MANIFEST 移除陈旧 reject/blocked 字段、填入真实审计字段与状态机；BLOCKERS.md 重写（修复路径转义损坏）。
- **1.5 审计与采集**：runner 每局记录 pair_metadata（checkpoint SHA、last_token、schema 版本、tree_skills 源码哈希）；`compare_g3_runs.py` 扩展为 manifest+初始状态哈希+分配/撤退/集火/keep/角色全表面比较；`data/collect_real.py`+`--pair-collect` 真实采集入口（33 维特征、教师原始 20 维 logits+mask+raw obs、原/建议分配、最终动作）；`build_shard.py` 离线三层指标+训练 shard（教师组内中心化标签）；`paired_report.py` 配对胜率+bootstrap CI+医疗机分组。

### G3.5 门槛测试
- 离线回归：tree_skills 34 + 规则导出 5 + skills 172+13 + adapter 65 全部通过。
- **A/A-plug 等价（新清单种子 310–313，172 步）：manifest/初始哈希/最终动作/分配/撤退/集火/keep/角色 0 差异 PASS**。
- **B-main 严格冒烟（4 局真实教师全程评分）**：172/172 步零失败零回退；99 个评分组、10 个多候选组、7 次推荐改配；真实教师标签 shard 99 组成功构建。
- **关键诊断（如实报告）**：尽管教师推荐了 7 次配对改变，**整局 A 与 B 的最终提交动作零差异（0/172 步）**——所有推荐改配在提交前被 keep-heal-order/订单锁定/自保层覆盖。按任务书 §3.2 诊断顺序，这属于"订单使提议被覆盖"；按 §5 关键约束不擅自放开锁定。含义：当前协议下教师的真实行为影响率≈0，40 局筛选预期 B≈A（筛选照常执行以取得完整证据）。

### G4 筛选（已完成，详见 G4_REPORT.md）
- A 与 B-main 各 40 局（种子 400–439）真实对局：**胜率完全相同（17/40，所有医疗机分组一致），配对差 0 [95%CI 0,0]，0 无效局**。
- 三层指标：962 评分组；多候选机会仅 6.4%；推荐改配 20 次；11/40 局实际动作分歧（242/1856 步，含级联）；**胜负改变 0**。
- 预注册判定：**未通过筛选，不进入 200 局资格**；qualified=false。失败模式=有效机会稀少+订单/keep 覆盖（结构性），非教师标签错误。G5/G6 性能路线停止；962 组真实教师标签 shard 已保存（仅机制研究用途）。
