# v8 群体空间技能：设计与接口说明

三个能力及其定位（详见 `current/README.md`）：

| 能力 | 文件 | 顶层候选 | 开关 |
| --- | --- | --- | --- |
| Regroup | `skills/regroup.py` | 是（`regroup`） | `enable_regroup` / `--enable-regroup` |
| CohesiveSearch | `skills/cohesive_search.py` | 是（`cohesive_search`） | `enable_cohesive_search` / `--enable-cohesive-search` |
| CombatPositioning | `skills/combat_positioning.py` | 否（战斗模式内子技能） | `enable_combat_positioning` / `--enable-combat-positioning` |
| 本地规则调度 | `skill_manager.SkillManager.spatial_dispatch()` | — | `auto_spatial_dispatch` / `--auto-spatial-dispatch` |

## 部分动作契约

`allocate_regroup(state, actor_ids, memory, parameters, previous_moves)` 与
`allocate_cohesive_search(...)` 返回：

```python
{
  "actions": {actor_id: legal_action_id},   # 仅受控单位；动作只能来自该单位当前 movement_candidates（Stop=1 也是合法候选）
  "controlled_units": [...],                 # 与 actions 键一致；未输出动作的单位不视为接管
  "status": "running|success|interrupted|blocked|timeout|unavailable",
  "reason": "...",
  "state_updates": {...},                    # 由管理器合并进局内记忆
  "diagnostics": {...},
}
```

`SkillManager._step_regroup / _step_cohesive_search` 把它转换成
`{index: action}` 部分动作（与 heal_assign 的合并先例一致），由调用层在
n_agents 长度的基线策略数组上合并，保证每单位每步只有一个动作来源。
`combat_positioning.allocate_combat_positioning(state, actors, histories,
fixed_actions, parameters)` 沿用 coordinate_approaches 的
`(actions, diagnostics, updates)` 契约；`fixed_actions` 中的单位只参与空间
预留，动作绝不被改写。

## 共享空间工具（group_geometry.py）

- 连通分组：欧氏距离 ≤ `link_radius` 的并查集分量；**长链连通≠集中**，集中需同时满足主要群占比与分位半径。
- 主要群：最大分量（平局按最小 ID）。集中度报告含 `main_share`、`span`、`median/quantile_radius`、`outliers`、`isolated`。
- 角色识别：`unit_role()` 依公开类型（Medivac→healer）、relation 的 native_weapon_range 或 unit_type_profiles 判 melee/ranged；未知类型按保守 ranged 处理；无敌人、无 relations 时仍可识别。
- 无地形数据时所有距离只是几何近似，不宣称真实可达或可支援。

## 阶段切换与动作归属

- 规则调度（`spatial_dispatch`）：接触（可见敌人或合法攻击边或新受伤）→ None（既有战斗控制器）；无接触且严重脱节（分组>1 且 main_share < `dispatch_enter_share` 或离群数≥主体数）→ regroup；否则→ cohesive_search。含最短持续期 `dispatch_min_windows`、退出阈值 `dispatch_exit_share`、失败冷却 `dispatch_cooldown_windows`；接触与受伤不受最短持续期约束立即打断。
- Regroup 完成条件：`main_share ≥ regroup_target_share` 且主要群分位半径 ≤ `regroup_target_radius`；成功时报告剩余离群单位，不悄悄移出输入集合。
- CohesiveSearch 严重脱节返回 `status="blocked", reason="needs_regroup"`，由调度器切换到 Regroup，不复制集结逻辑。
- CombatPositioning 在 `allocate_adaptive_focus_fire` 的 `coordinated_approach` 分支内组合调用（同一单位不会同时执行旧接敌与新站位）；只接收 reposition 角色且无合法攻击的单位。

## 共享记忆与预算（v8.1 修订）

- 观察入口唯一：`SkillManager._observe()` 在每个唯一决策快照执行一次（任意技能或直接控制路径均经过），维护唯一的 `_last_seen_enemies` 账本；搜索调用前由账本构造带位置/观察loop/年龄/来源/代次的派生 `search_hint`，运行器自带提示优先。功能全关时该入口为 no-op，v7 语义逐字节保留。
- 冷却按绝对 `game_loop` 到期（`cooldown_until = loop + windows × step_mul`），不依赖空间技能是否执行；预算以决策步计，同一快照重复调用命中缓存只计一次。
- 执行历史只记最终动作：`commit_final_actions(final_actions)` 在运行器合并与环境掩码校验后调用；被覆盖/被拒的提案不进入 `_last_move` 或集结的 last_final_moves。未接线路径的下一次快照会以提案作为最可用记录回退（已文档化）。
- 编队使用记忆持有的 `march_center`（每窗向航点前进一格、不超前实际队伍 `link_radius`），成员槽位为共享航向参考系下的 (前向,侧向) 偏移；单单位零偏移。
- `reset()`、局标识变化与时间回退清理全部空间状态：预算、航点、失败方向、位置分配、缓存、冷却。

### 复杂度（更正）

两两分组、跨度计算与碰撞比较含 O(n²) 项：成本随人数**二次**增长（此前"线性"的说法不准确）。10 单位实测每步 mean ≤1.2ms、p95 ≤1.6ms。

## 参数（初值与依据）

`skill_manager.SPATIAL_DEFAULTS` 集中配置：`link_radius=6`（约为 SMAC 中程射程外缘，保证互相能在一两个窗口内支援的几何近似）；`regroup_target_share=0.8 / target_radius=5.0`（主体基本同处一屏内）；`regroup_max_windows=24`（约为 10 人在 1 格/步下跨越 ~20 格再留裕量）；`regroup_max_wait_windows=3`（等待约等于一个撤退小节的量级）；`regroup_no_progress_limit=4`、`regroup_retry_cooldown_windows=6`；`search_arrive_radius=3`、`search_waypoint_ttl_windows=6`、`search_max_windows=40`、`search_leader_wait_limit=2`、`search_memory_ttl_loops=1500`（默认 step_mul=8 下约 3 分钟量级的提示有效期）。硬约束：合法动作、单位集合、角色；软评分：距离、拥挤、支援、危险、换向惩罚，均按窗口尺度归一化。

## 协议与候选

`skill_policy.skill_choices(..., spatial_modes=())` 把 `regroup`、`cohesive_search` 追加在既有全部候选之后，仅在显式开启时出现；`spatial_modes` 由运行器一份快照绑定元组同时传给 `skill_request` 与 `decode_skill`，响应时不能重新编号。`combat_positioning` 不进入互斥顶层候选。禁用时旧候选内容、顺序和解码与 v7 完全一致（有回归测试保证）。

## 兼容与退化

- 全部关闭：不新增候选、不改 `SkillManager.step` 行为、`combat_params` 不含 positioning 键（有测试断言）。
- 缺 `map` 边界 / 搜索点 / 视野：搜索走局部退化（合法移动候选形成的短程方向 + 访问记忆 + 停滞轮换），明确不保证全图覆盖；缺失字段不伪造。
- 外部参考：Sharpy `group_combat_manager.py` / `DefaultMicroMethods.handle_groups()` 与 Ares 小队/群体行为教程的分组、重组判断与行为仲裁思想被重新实现并适配到 SMAC 离散动作；未复制其 SC2 坐标指令、默认距离阈值或代码。许可说明见 `licenses/` 与 `THIRD_PARTY_NOTICES.md`（Sharpy: MIT；Ares: GPL-3.0，仅借鉴思想，未包含其代码）。
