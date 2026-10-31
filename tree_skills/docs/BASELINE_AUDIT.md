# 基线审计（Baseline A，冻结于 2026-10-30）

工作区：`C:\Users\ry\skill\SMAC_agent`。本文件记录人族 adaptive 基线的真实本地状态，是 A/A-plug/B/C/D 全部对照实验的基准。源码未移动、未覆盖；除下文"注入点"小节列出的默认关闭参数外，基线行为逐位不变（183 个既有测试全部通过，见 RESULTS.md）。

## 1. 基线标识与实验参数（以本地 run.json 为准，非 README 历史默认）

最新三个人族 run：`sc2_skill_adapter/runs/terran-{heal-escort-v5, coordination-v6, lifecycle-v7}-20261030-n40-seed170/`，三者 env_args 与 skill 参数一致：

- 地图 `10gen_terran`，10v10（marine/marauder/medivac 权重 0.45/0.45/0.10）
- `unit_balance=native`（每局附 balance/episode-*.json 审计）
- `step_mul=8`，difficulty `7`，move_amount 2，map 32×32
- `action_mask=true`，`conic_fov=false`，`heuristic_ai=false`，`prob_obs_enemy=1.0`
- 动作扩展：`terran-keep-heal-order-v1`，动作 ID 16（native_n_actions=16 → 17），提交 `retained_order`
- skill_version `current`，race_profile `terran`，posture balanced，max_steps 4，全部 combat 开关开启
- unit_value：contextual / unit-value-v2
- 种子 170–209 共 40 局；三 run 均为 16胜/24负（0.40）
- 每局记录源码 SHA-256（run.json: skill_source_hashes / source_hashes）

与任务书假设的差异：任务书猜测 step_mul=8/难度7"待审计确认"——本地确认一致，无需调整协议。

## 2. 每步决策链（观察 → 治疗动作提交）

1. `bridge/policy.py:135 build_request()` — 组装 state（allies/visible_enemies/relations/heal_candidates/movement_candidates），heal_candidates 仅含掉血友军（fog-safe）
2. `bridge/runner.py:260 skill_manager.step(payload, n_agents)`
3. `skills/current/skill_manager.py:283 step()` → `_step_adaptive` → 人族分支调用 `terran_combat.allocate_terran_combat`
4. `terran_combat.py:266 allocate_adaptive_focus_fire` — 基础战斗计划
5. `terran_combat.py:280 _reserve_observed_heal_orders` — 观测治疗订单锁定；满足条件（稳定 3 窗冷却 + margin 0.25）开放救援切换；重写 heal_candidates 排除占用目标
6. `terran_combat.py:281-283 assign_healing(healing_state, ...)` — 联合匹配（≤8 医疗机确定性穷举 `_exact_matching`，否则 scipy LSA；空闲列=0，非法边 -1e9，只接受正分边）
7. `terran_combat.py:286-296 select_keep_orders` — keep-heal-order 决策（治疗证据/跟随/停滞/未确认）
8. `terran_combat.py:297-301` — 救援请求记录与 order_states 冷却
9. `terran_combat.py:308-334` — 医疗机动作覆写（keep > 联合治疗 > 支援/自保）
10. `terran_combat.py:370 coordinate_v6` — v6/v7 协同；`:424-436` 逐单位合法性回退
11. `terran_combat.py:444-457 observe_heal_continuity` + race_memory 回存
12. `runner.py:338-341` 掩码校验 → `:342-343 commit_final_actions` → `:347 env.step(actions)`（keep 动作在 `environment.py:104-112` 转为 retained_order）

**树评分器接入位置**：第 6 步内部，`assign_healing` 完成原始匹配之后、任何动作/订单副作用之前。参与集合 H 固定为原始匹配已分配目标的医疗机；候选边即原始合法边（环境门控 + 订单预留过滤之后）；替换通过注入的 pair scorer 重解"每机必选一目标"的固定参与匹配。keep、自保、救援、订单锁定逻辑全部原样在通道外运行。

## 3. 注入点（对基线源码的全部修改）

- `skills/current/skills/heal_assign.py`：`assign_healing` 新增 `pair_scorer=None, pair_scoring="off"` 参数；新增 `_forced_matching`、`_apply_pair_scorer`（仅在显式传入 scorer 时执行）。默认 off 路径不增加任何计算或字段。
- `skills/current/skills/terran_combat.py`：`allocate_terran_combat` 在 deepcopy 前剥离并透传上述两参数。

## 4. 环境与依赖

- 运行环境：`sc2_skill_adapter/.venv`（Python 3.10.22；pysc2 4.0.0, numpy 1.26.4, scipy 1.15.3, protobuf 3.20.3, PyYAML 6.0.2, absl-py 2.5.1, httpx 0.28.1）。无 scikit-learn/torch —— 树部署路径不得引入。
- 训练环境：系统 Python 3.13（scikit-learn 1.7.0）用于训练/导出；导出后的规则模型仅依赖标准库。
- 测试：`PYTHONPATH=. python -m unittest discover -s skills/current/tests`（172 用例）+ 两个相对导入模块单独运行（13 用例）。

## 5. 回归集与数据集边界

- 种子 170–209（runs 已用）只作回归集。新数据集场景编号从 210 起规划（EXPERIMENT_PROTOCOL.md），按初始化状态哈希查重，不只看种子。
- 基线源码 SHA-256 以各 run.json 的 skill_source_hashes 为准；本审计未改动任何基线文件内容（除第 3 节默认关闭参数，其哈希变化记录于 RESULTS.md）。
