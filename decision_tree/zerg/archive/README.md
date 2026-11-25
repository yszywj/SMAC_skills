# 历史归档：虫族分层决策树训练框架

本目录保存旧说明和当时的测试记录。以下训练、部署和采纳建议已停用，不是现行流程的准备步骤。返回[正式训练入口](../README.md)。

基于现有 `decision_tree` 框架，为最新虫族 `tactical_v1` 建立
Skill 内部评分树 + Skill 间选择树的训练与运行框架。
**只搭框架与接口**：轨迹采集、数据准备、实际训练、大规模对局由项目负责人执行。
不使用 `tree_skills`（已弃用），不使用 MARL-GPT，不重新实现 tactical_v1。

规范文档：`ARCHITECTURE.md`（源码审计）· `SOURCE_RESOLUTION.md` ·
`FEATURE_SCHEMA.md` · `DATA_SCHEMA.md` · `MODEL_INTERFACE.md` ·
`RUNBOOK.md`（轨迹采集与训练手册）。

## 新增文件说明

| 文件 | 职责 |
|---|---|
| `source.py` | base+overlay 确定性合并、staging 物化、source_manifest.json、整体指纹、变更检测 |
| `features.py` | 固定版本化 `ZergFeatureSchema`（unit/team/skill/candidate 四层）+ 只读提取器 |
| `candidates.py` | `ZergCandidate` 统一结构；从真实 `CandidatePlan` / step summary 适配 A/B 两类候选；结构合法性预检 |
| `scorer.py` | `score_candidates` 统一入口、`ScorerModel`（树/森林、分族、缺失插补）、`rank_by_scores`、四个评分器槽位 |
| `selector.py` | `select_skill_plan`：在 Coordinator 的完整合法联合计划中做偏好重排 |
| `objectives.py` | 四类教师信号：rule imitation / candidate ranking / counterfactual gain / episode outcome（缺失收益绝不伪造为 0） |
| `schema.py` | 决策记录、模型文件、训练配置三类 JSON 契约与校验；NaN→null 序列化 |
| `dataset.py` | 决策记录构建、追加式去重 JSONL、场景级切分、组级权重归一、训练行转换 |
| `fingerprints.py` | 源码指纹、运行时策略契约、执行上下文哈希、模型指纹、部署强校验 |
| `training.py` | `ZergTrainingBackend`（imitation/gain/correction/ranking）+ `ZergCandidateAdapter` + 同契约 CART/森林拟合 |
| `runtime.py` | off/shadow/tree_score 三模式运行时与 manager 包装器（不改动作提交契约） |
| `configs/runtime.json` | 运行时配置，默认 `mode="off"` |
| `models/` | 训练产物存放处（交付时为空，见 README.md 说明） |
| `tests/test_minimal.py` | 5+1 项最小验收测试 |
| `scripts/linux/*.py` | inspect_zerg / resolve_source / validate_zerg_config / export_zerg_model |

## 复用模块说明（未复制算法，未改动现有文件）

- `decision_tree.learning.train`：`group_split`（sha256 场景确定性切分）、
  `group_weights`（场景等权）、`finite/names/positive_int` 校验助手 —— 直接 import。
- `decision_tree.experiment`：`digest`（canonical JSON SHA256）、
  `validate_deployment` 契约模式 —— 直接 import / 同构复用。
- `decision_tree.learning.vector_train` / `local.models`：加权 SSE 贪心阈值树、
  中点阈值、每侧最少行数+最少独立场景、masked 候选独立估计、
  "缺失即缺失"语义 —— 其入口把 Protoss 特征 schema 硬绑在校验层，
  无法直接传入虫族特征；`training.fit_threshold_forest` 是**同一算法契约的
  schema 泛化移植**（含 forest 容量扩展），并在模型 record 中显式声明算法来源。
- `decision_tree.local.training.correction_rows` 模式：correction 任务的
  veto 标签 = 冻结旧模型赢家增益取负（`ZergTrainingBackend.correction_rows`）。
- `decision_tree.conditional`：触发/冷却/有界干预/修正层的**协议语义**
  被 runtime 配置与 objectives 的声明引用；其代码绑定 Protoss local_v1 依赖，
  未被 import。注意：单步干预协议 ≠ 多单位连续策略收益评估
  （`training.report.limitations` 中声明）。
- `local_v1` 及其它现有模块零改动；`tree_skills` 零引用。

## Linux 命令示例

```bash
cd /home/amax/ry/SMAC_skill

# 1) 源码解析与审计
python decision_tree/scripts/linux/resolve_source.py
python decision_tree/scripts/linux/resolve_source.py --check-only   # 变更探测
python decision_tree/scripts/linux/inspect_zerg.py --json /tmp/zerg_audit.json

# 2) 配置与模型校验
python decision_tree/scripts/linux/validate_zerg_config.py
python decision_tree/scripts/linux/validate_zerg_config.py --training-config <cfg.json>

# 3) 最小测试（不启动 SC2）
python -m unittest decision_tree.zerg.tests.test_minimal -v

# 4) 训练（真实数据由 RUNBOOK 的采集步骤产生后）
python - <<'PY'
from decision_tree.zerg.training import ZergTrainingBackend, load_scorer_model
from decision_tree.zerg.dataset import read_records
backend = ZergTrainingBackend()
model, report = backend.train(
    "imitation", read_records("decision_tree/zerg/artifacts/records/shadow.jsonl"),
    scorer="hydralisk_scorer", execution_context_hash="<见 RUNBOOK>")
PY

# 5) 导出可部署模型（绑定当前执行上下文）
python decision_tree/scripts/linux/export_zerg_model.py \
    --model <trained.json> --scorer hydralisk_scorer \
    --out decision_tree/zerg/models/hydralisk_scorer.json

# 6) Shadow 采集（在 runner 中包一层；不改变任何动作）
python - <<'PY'
from decision_tree.zerg.runtime import ZergTreeRuntime
runtime = ZergTreeRuntime(config_path="decision_tree/zerg/configs/runtime.json")
controller = runtime.wrap(skill_manager)   # step/commit_final_actions 签名不变
PY
```

## 训练依赖与运行依赖分离

- 本框架**纯标准库**（与现有 decision_tree 一致），无 numpy/sklearn/torch；
- SMAC/SC2 运行时仅在现场采集时需要（走现有
  `sc2_skill_adapter/run.py` 流程）；训练/校验/导出全部离线完成。

## 最小测试结果（2026-11-23）

```
python -m unittest decision_tree.zerg.tests.test_minimal -v
Ran 7 tests ... OK
  1 源码合并：136 文件、overlay 覆盖、独有保留、指纹确定、原目录未动
  2 特征 schema：真实 Snapshot 提取、确定性、缺失=NaN≠0
  3 候选转换：真实 propose/summary → ZergCandidate（合法、schema 完整）
  4 Shadow Mode：动作逐位一致、JSONL 记录、重复观察去重、off 零包装
  5 原有入口：SkillManager 导入/step/commit/reset 幂等不受影响
  + 训练回路：imitation/gain → 模型校验 → 加载 → 评分回退
回归：skills.tactical_v1.tests（抽样 33 tests）OK
```

## 尚未完成的接口与限制

1. `retreat_quality` / `candidate_support_gain` 两个特征位为预留（恒 MISSING），
   等 Skill 侧提供 retreat 路由质量信号后填充；
2. 爆虫（baneling）专用评分器仅保留槽位，未绑定训练目标；
3. 单位级特征尚未进入 34 维评分行向量（保留在决策记录中，
   单位级评分器升级时再启用）；
4. A 类（Skill 内部）候选的在线 shadow 记录需要项目负责人在 runner/step
   钩子中把 `skill.propose` 输出接入 `collect_step_candidates`（接口已备，
   runtime 默认只做 summary 级 B 类记录）；
5. `tree_score` 换选只在"Coordinator 已记录的 top_alternates"范围内进行；
   不做任何计划合成；
6. 收益标签（gain/ranking/correction）依赖真实配对反事实回执，本框架只提供
   契约与审计，不生成任何无证据标签；
7. 192 棵树（`number_of_trees=192`）为可选容量，未做默认，也未做超参搜索。

---

# V1.1 新增文件与变更（2026-11-23）

新增：`factory.py`（统一 SkillManager 工厂，resolved 树绑定）、
`proposal_collector.py`（A 类候选采集，不重复调 propose）、
`experiment.py` + `configs/exploration.json`（有界干预预算/冷却/族预算）、
`experiment_manifest.py`（实验冻结清单与进度）、
`TRAINING_PROTOCOL.md`、`LIMITATIONS.md`、`tests/test_v11.py`（16 项 V1.1 验收）。

关键修复：
- 特征训练/推理不一致 → 唯一拼接 `schema.build_model_features`，模型
  feature_names 强制同序；整层缺失拒绝评分，单点缺失才可插补并记录比例；
- 数据身份 → run_id/episode_id/scene_config_id/seed/map_id，
  record_key 含 run_id；切分按独立 Episode（跨配置按 scene_config_id 留出），
  分组清单落盘；
- 反事实配对 → `zerg_counterfactual_pair_v1` + 审计（gain 恒等于
  candidate_return - baseline_return，不完整配对不进训练）；
- ranking → 真 Pairwise Ranking（`zerg_pairwise_ranking_model_v1`，
  a./b. 成对布局 + 锦标赛推理）；correction → 冻结旧模型赢家重放
  （veto_target = -赢家真实 gain），不再用实际最优替代；
- tree_score → `apply_selected_plan` 完整计划替换（Validator+repair+
  失败整体回退 BASE），无计划对象即拒绝；默认仍 off；
- resolved 目录加入 .gitignore，manifest 相对路径，可随时重建。

最小测试：`python -m unittest decision_tree.zerg.tests.test_minimal
decision_tree.zerg.tests.test_v11` → 23 tests OK。

---

# V1.2 修复（定向修复，不重写架构）

## Phase 1 实验身份
- `run_id` 由 `generate_run_id()`（时间戳+pid+进程内序号）在 runtime 构造时
  生成一次，或从 config 显式传入；`reset()`/`new_run()` 语义分离；所有决策
  记录、事件、配对记录都携带。
- `episode_id` 在 run 内唯一：`prepare_decision` 在 manager.step **之前**
  解析身份（finalize 边界 + snapshot/loop 回卷双信号），重复局号得到
  `_r<n>` 后缀；换局时 collector 缓存同步清理（先于 propose 采集）。
- `scene_config_id`：experiment manifest 显式提供，否则 canonical hash
  （地图+step_mul+兵种构成+敌我数量+search_location；不含随步变化的位置/
  血量/剩余步数），按 Episode 冻结，来源写入 `scene_config_source`。
- 切分：episode 分组（局部模型）/scene_config 留出（跨配置）；反事实配对
  按 anchor 分组（`split_pairs`）；三集清单+不重叠断言。

## Phase 2 恒缺失特征
- `feature_availability` / `select_active_features`：统计每特征有限值数/
  缺失比例/有效 Episode 数；恒缺失列进入 `inactive_features`（原因
  `constant_missing_in_training`），不进 `active_feature_names`，绝不填 0；
  原始记录保留完整 schema。
- 中位数插补只从训练集的 active 列计算；模型声明
  `active_feature_names`，推理缺失报告只针对 active 列。
- `FEATURE_FORBIDDEN` 补齐 candidate_id/proposal_id/instance_id/task_key。

## Phase 3 A 类链路
- `dataset.expand_skill_proposals`：A 类候选 → 训练行（同唯一拼接；
  `features_internal` = 单位聚合布局）；`export_internal_training_rows`
  按 Skill / 单位类型导出。
- `decision_group_id = run:episode:snapshot:skill:proposal_group_id`
  （proposal_group_id 取 task_key→instance_id→controlled_actors）；
  组内才可比较，跨组不伪造互斥。
- 内部教师标签：`matched_final_actions`（proposed 动作原样进入
  final_actions）→ 唯一匹配者 `teacher=1`；`entered_final_plan` /
  `executed`（需 committed）/ `final_action_kept`；未选中 ≠ 规则错误。
- `schema.build_internal_model_features`：team+skill+candidate+unit 聚合
  （mean/min/max/count/type_fraction，86 维固定顺序，tag 不入特征）；
  `training.train_internal` 用该布局训练 A 类评分器（imitation/gain/
  ranking/correction 全支持，按 candidate_skill/unit_type 过滤）。
- `scorer.score_routed`：按 `route_candidates`（单一兵种→类型评分器，
  混合→joint，未知→joint 安全回退）逐槽评分，runtime 统计各槽候选数。

## Phase 4 B 类完整计划
- `candidates.CompletePlanCandidate`：完整分支对象（plans/actions/claims/
  dependencies/duties/rule_score/features/legal/source_plan_ids）；BASE
  由 `_pending_commit` 真实计划构建；summary 级备选标记
  `plan_objects_available=False`——可评分记录但永不触发替换；不允许用
  JSON 动作伪造 CandidatePlan。
- `selector.apply_complete_plan`：整组替换（不局部拼接），Validator 完整
  重跑 + 有界 repair，失败整体回退 BASE。
- `runtime._tree_score_ready`：模型存在/目标可排序/指纹一致/schema 一致/
  完整计划对象存在/Validator 可用/Manager 状态可更新——任一不满足拒绝。
  默认仍 `mode=off`。

## Phase 5 工厂
- `runtime.wrap` 接受 `ZergManagerBundle`：校验实际模块文件位于 resolved
  树、指纹一致、并使用 bundle 的 effective_config 重建执行上下文
  （含 action_space_version / observation_protocol / model_algorithm）。
- `load_resolved_package`：模块命名空间带源码指纹后缀；同指纹仅在
  `__file__` 匹配时复用缓存；不同指纹加载到新隔离命名空间，绝不驱逐
  存活模块。

## Phase 6 Pairwise Ranking
- 训练样本双向生成（反向 margin 取负），消除固定位置偏差。
- 推理：`PairwiseRankModel.score_candidates` 全轮询锦标赛，双向比较，
  记录 wins/losses；并列时调用方以原规则名次 tie-break；不支持→规则回退。
- 评估：pairwise_accuracy 要求正反两方向同时正确（另报 forward/reverse
  单向准确率）；NDCG 只用候选级真实收益完整且 provenance=paired_gain 的
  决策点。

## Phase 7 执行反馈
- 模型候选替换失败 → `selected_candidate` 记录实际回退的 BASE。
- `committed` 事件区分：`commit_call_completed` + `memory_commit ∈
  {new_commit, idempotent, partial_or_arbitrated, none}`；committed 仅表示
  原 Skill 记忆提交条件通过，不是 SC2 引擎确认。

---

# V1.3 端到端链路（最终框架版）

五条链路全部离线验收通过（`tests/test_v13.py`）：

**A. A 类内部链路**：collector 被动记录 → `observe_step` 用
`(run_id, episode_id, snapshot_id, game_loop)` 校验身份后取当前候选 →
`score_routed` 按槽位评分（A/B 类分别记录）→ 每候选 shadow 日志
（decision_group_id/predicted_rank/teacher/final_action_match/execution_status）
→ `expand_skill_proposals` → `train_internal` → `load_zerg_model` → 在线评分。
propose 调用数不变，动作逐位一致。

**B. B 类完整分支链路**：关键事实——Coordinator 的 `evaluate_once` 只缓存
通过 Validator 的分支，且 `coordinator._cache` 在 decide 后仍持有**当前步
真实 CandidatePlan 对象**。`candidates.build_branch_candidates` 只读导出：
BASE 取 `_pending_commit`（authoritative），备选取缓存（排除 BASE、按动作
去重）——零战术改动、零伪造、零重复执行。`selector.select_complete_plan`
评分（gain/imitation 用回归估计；ranking 用锦标赛），仅**严格优于 BASE**
才换选（并列归规则名次）；`apply_complete_plan` 整组替换 + Validator +
有界 repair + 失败整体回退；`_update_manager_plan_state` 归一化 dict→u 列表
并刷新 `_pending_commit`/`_cache`/summary；Memory 提交仍由原 Manager
surviving-plan 逻辑独占。fixture 实测 2 个真实分支（非 blocked）。

**C. 反事实链路**（新 `pair_dataset.py`）：
`join_counterfactual_pairs`（decision_key/anchor 关联 + gain 推导）→
`audit_counterfactual_pairs`（起点/前缀/上下文/源码/干预记录/完整性/算术
十项审计 + 可训练比例）→ `build_gain_dataset`（A 类与 B 类收益**分开**
回填 `candidate_return`/`baseline_return`，输出独立 record 列表 +
dataset_hash + 来源清单）→ `backend.train("gain")/train_internal("gain")`。
未审计数据进不了训练；缺失收益保留 None。

**D. 模型部署**：`scorer.load_zerg_model` 按 kind 分流
（zerg_scorer_model→ScorerModel；zerg_pairwise_ranking_model_v1→
PairwiseRankModel，互不混用）；统一评分字段
score/supported/confidence/rank/model_kind/fallback_reason；
correction 只有 veto 语义（`correction_veto_scores`，预测>0 表示冻结旧模型
赢家为负收益，绝不作为偏好排序）；模型记录新增
`model_id`/`parent_model_id`；导出支持 `--out-dir <base>/<model_id>/`；
上下文不匹配加载即拒。

**E. 可复现性**：scene_config 在 episode 开局快照含出生位置并整局冻结
（来源标注 `canonical_hash+birth_positions` / manifest）；`write_run_manifest()`
输出 run 级可复现清单（身份/场景/指纹/上下文/config digest）。

Tree Score Gate（V1.3）十项检查：模型存在/目标可排序(gain|ranking|imitation，
correction 不可用于换选)/kind 受支持/源码指纹一致/特征 schema 一致/全部
分支携带完整计划对象/分支属于当前 Snapshot/BASE 在列/Validator 可用/
Manager 状态可更新——任一失败回退 BASE。默认 `mode=off` 不变。

---

# V1.3 部署与审计修复（本轮）

1. **统一模型加载**：`runtime._load_models` 改走 `scorer.load_zerg_model`
   （按 kind 分流）；ranking-kind ↔ target='ranking' 双向强校验；
   correction 模型禁止占用排序槽位（veto 专用）；source fingerprint 与
   execution context 逐项校验，任何不匹配拒绝加载。
2. **Ranking 在线路由**：`score_routed` 对 PairwiseRankModel 分流——joint
   槽走锦标赛（`pairwise_uniform_scores`，tie 用规则名次）；A 类单位槽
   明确回退（reason=`internal_ranking_online_unsupported_rule_fallback`），
   绝不产生错误评分。
3. **反事实审计硬化**：pair record 新增 `source_fingerprint`；join 保留
   `pair_source_fingerprint` + `record_source_fingerprint` 并要求**严格
   相等**；decision record 缺 source hash 或 execution context hash 时拒绝
   严格收益训练；新增 `experiment_manifest.make_anchor_manifest` +
   `validate_anchor_evidence`（冻结的 per-anchor 起点/前缀/上下文/源码/
   baseline_id 证据，pair JSONL 无法自证有效）；baseline_id 必须等于
   manifest 声明的 BASE。
4. **B 类分支特征修复**：`branch_scoring_features(branches, snapshot)` 用
   `snapshot.previous_action_ids` 计算真实动作变化；目标类型取 plans 众数
   goal_kind 编码；progress 取声明通道均值；risk 用 snapshot relations 的
   敌人射程内比例；`candidate_support_gain` 无估计器保持 MISSING（NaN），
   绝不填 0；训练与推理共用同一函数与 schema 拼接。

---

# Gain 数据审计修复（本轮）

1. **强制 Anchor Manifest**：`build_gain_dataset(require_audit=True)` 必须提供
   冻结 anchor manifest（kind/experiment_id/manifest_sha256 完整性/
   source_fingerprint/execution_context_hash 全校验）与执行回执；
   `require_audit=False` 仅调试且输出标记 unaudited。
2. **执行回执**：`zerg_execution_receipt_v1` 记录两分支的 started/completed/
   episode_finished/error/return 与独立 receipt_id/receipt_source；
   completion_status 不再自证实验完成；跨 anchor、回执 return 与 pair
   return 冲突即拒绝。
3. **训练入口强制审计**：`train()/train_internal()` 的
   gain/ranking/correction 任务必须携带 `audit_envelope`（audited + manifest
   hash + records_hash 逐字节绑定 + 上下文/源码一致），否则拒绝；
   审计结论写入模型 `gain_audit` 元数据。Shadow 缺失收益永远不会被解释为 0。
4. **资源冲突特征**：`candidate_resource_conflict` 无估计器 = MISSING
   （branch 与 summary 两条路径一致），训练侧由 active-feature 筛选剔除该列。
5. resolved 目录维持本地生成产物（.gitignore 已含），`resolve_source.py`
   可随时重建，Factory 导入不受影响。

---

# 执行回执多候选关联修复（本轮）

1. `build_receipt_index` 索引键扩为四元
   `(experiment_id, anchor_id, branch_id, candidate_identity)`：同一
   Anchor 的多个 candidate 回执并存不冲突；BASE 用 baseline 臂身份键控；
   同键异容拒绝、完全重复去重。
2. `audit_execution_receipts` 对每个 Pair 精确关联自己的 BASE 与 candidate
   回执，并完成 12 字段身份/证据比对 + 完成性三标志 + 严格收益检查
   （存在、有限、与 pair 声明一致）——兄弟候选的回执永远无法替代。
3. 新增多候选测试（test_v15.Test06）：BASE + A/B/C 三候选共享一份 BASE
   回执、各自绑定自己的回执、无索引冲突、跨候选替代/收益缺失/收益不一致/
   horizon 不一致全部拒绝、三个完整 Pair 全部进入 audited Gain Dataset。
