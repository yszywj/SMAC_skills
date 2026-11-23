# decision_tree/zerg — 虫族分层决策树训练框架

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
