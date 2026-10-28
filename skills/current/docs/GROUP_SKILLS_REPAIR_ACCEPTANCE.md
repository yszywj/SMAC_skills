# v8.1 群体空间技能修复验收报告（2026-10-25，R01–R12）

本轮依据对 v8（`459eadd3` 基准）的复核，在实际工作区完成修复、回归、接入核验与文档更新。测试全部使用真实项目函数（Manager、分配器、policy 助手），无替身实现。

## 完成层级

1. **纯技能修复 + 离线回归**：完成（见命令与结果）。
2. **运行器接入**：两个真实入口均已接线并冒烟——SMAC-Hard `game_skill/micro/jev_micro/run.py` 与 SMACv2 `smacv2_jev_adapter/bridge/runner.py`（后者是 v8 轮未接的缺口，R12）。
3. **真实对局验证（任务12 A0–A3）**：**未执行**（本轮未启动 SC2 进程；行为正确性与策略增益分开报告，不以离线结果冒充）。

## 问题—修复—测试对应表

| 缺陷 | 修复 | 回归测试 |
| --- | --- | --- |
| R01 搜索未使用共享敌忆 | `_observe()` 统一观察入口维护唯一 `_last_seen_enemies`；搜索调用前从账本构造带位置/时间/年龄/来源/代次的派生 `search_hint`（不改原始输入）；TTL、负年龄、非有限坐标、过期提示全部拒绝；重新观察产生新代次可重选 | `test_ledger_hint_injected_into_search`、`test_visited_expiry_and_new_generation_reopen`、`test_expired_and_invalid_hints_rejected`、`test_observation_entry_is_noop_when_features_off` |
| R02 反向搜索/环形偏移 | 编队参考系改为共享航向的 (前向,侧向) 槽位；单单位零偏移直奔航点；`march_center`（记忆持有、每窗前进、不超前队伍 link_radius）替代自引用锚点，消除"到达 centroid+1 即停"死锁 | `test_single_unit_advances_toward_each_direction`（东西南北+斜向）、`test_multi_unit_team_makes_net_progress_toward_waypoint`（2/5/10人）、`test_different_speeds_still_advance` |
| R03 假换向/航点不轮换 | 候选先过滤有效性/访问状态/失败冷却再排序；访问记录带时间戳+代次，真过期（loops）；退化模式短程目标固定不滑动；失败方向按方向点积匹配施加有限期惩罚（单一路线不封死）；`direction_changed` 仅在实际动作改变时记录 | `test_waypoint_rotation_first_visited_second_chosen`、`test_blocked_direction_actually_changes`、`test_single_route_is_never_sealed`、`test_all_directions_blocked_returns_blocked_status`、`test_no_waypoint_degenerates_with_fixed_short_range_goal` |
| R04 追逐自身站位 | 槽位锚定小队参考（队伍锚点+真实前沿+所选敌群），不再按自身位置"前进2后退spacing"；`front_back` 相对真实前沿，无近战时仍为接敌弧线；间距只改侧向展开；进展检查（连续无改进→按单位回退协同接敌）；诊断记录实际选中敌群 | `test_spread_front_back_single_unit_makes_engagement_progress`（20→<17，修复前 20→22.36）、`test_spacing_changes_lateral_slot_geometry`、`test_multi_direction_enemies_engage_nearest_cluster`、`test_no_slot_progress_falls_back_per_unit` |
| R05 站位不独立启用 | adaptive 分支改为 `coordinated_approach or combat_positioning`，positioning 优先、旧路径为显式回退（回退也过滤 fixed_actions）；`approach_progress_guard` 在站位路径同样执行（复用 guard_approach，写入 approach 兼容历史） | `test_enabled_independently_inside_adaptive`（True/False 组合 + 双开单提案）、`test_legal_attack_and_fixed_actions_are_never_overridden` |
| R06 冷却只在执行时递减 | 冷却改为绝对 `game_loop` 到期（`cooldown_until` = loop + windows×step_mul），不执行空间技能也会随观察时钟到期 | `test_cooldown_expires_by_observation_clock` |
| R07 部分控制误成功 | 完成判断改用 `controlled_cohesion`（观察仍用全队）；半径 0 有效（不再 `or 1e9`）；所有路径 `controlled_units == actions.keys()`，输入成员单列 `requested_units`；无进展统计只对"前一最终动作为移动且实际位移<0.2"计数，合法等待不算失败 | `test_partial_control_does_not_succeed_via_others`、`test_zero_radius_is_valid_not_missing`、`test_contract_controlled_units_equal_action_keys_everywhere`、`test_arrived_waiting_unit_is_not_a_failing_straggler`、`test_single_duplicate_dead_and_empty_inputs` |
| R08 无合法Stop/占位 | `legal_actions()` 显式适配器契约（`state['legal_actions']` 优先，否则 movement_candidates+活单位 Stop）；Stop 永不进 `predicted_move()`；`base_reservations()` 全体活友方默认占位，已决定动作覆盖为预测终点；空地分层；出生重叠向外脱离不罚；深度重合（gap<30%间距）重罚；到达目标 Stop 优先后不振荡 | `test_stop_wins_ties_at_goal_and_no_oscillation`、`test_legal_actions_contract`、`test_uncontrolled_ally_occupancy_costs_and_avoided`、`test_spawn_overlap_escape_not_sealed`、`test_flying_layer_separate_from_ground` |
| R09 候选错译 | `build_choice_context()` 由本地代码生成并保存请求绑定映射；`decode_skill` 优先用该上下文；无上下文且非默认 `max_explicit_targets`（请求中记录为 `explicit_target_limit`）或含空间候选时明确报错，不静默猜表；旧默认请求完全兼容 | `test_non_default_limit_decodes_via_bound_context_only`（5敌/无治疗/limit=1：id4=激进 adaptive，不再误译为 e1 集火）、`test_default_requests_stay_backward_compatible`、`test_limit_overrides_and_edge_values`（0/1/4/9）、`test_spatial_candidates_and_rejections`、`test_duplicate_and_missing_answers`、`test_request_records_explicit_limit` |
| R10 生命周期 | 同快照幂等（缓存区分局/快照/技能/控制集合/配置，返回副本）；`commit_final_actions()` 只记录最终动作；预算跨窗口累计；reset/新局/时间回退清理全部空间状态（预算、航点、失败方向、冷却、缓存）；死亡/控制集合变化由 requested/controlled 过滤 | `test_same_snapshot_three_calls_count_once`、`test_final_action_not_proposal_is_recorded`、`test_budget_survives_window_end_and_reselection`、`test_time_backwards_clears_filled_spatial_state`、`test_new_episode_clears_spatial_state`、`test_reset_clears_everything_spatial` |
| R11 测试不足 | 旧测试重写（输入顺序改为三种插入序+受控排列对比；时间回退预填记忆后断言清理；间距配置断言槽位几何；预算真到窗口上限再 select；部分动作含固定攻击基线合并），新增上述全部回归组 | 全文件 61 项 |
| R12 接入 | SMACv2 真实入口 `bridge/runner.py`+`skill_runtime.py` 接线：CLI 开关、manager_kwargs、auto 调度、部分动作与 nearest 基线合并、`commit_final_actions`、中断时空提案不再合并到移动基线而是同边界交给战斗控制器 | 解析器冒烟 + 适配器 36 项回归 |

调度（任务10）：进入/退出同时看占比与紧凑度（长链连通但不紧凑→集结；退出集结需 share≥exit 且 radius≤target）；`needs_regroup` 判定实际影响下一次调度；两技能冷却或禁用时显式退化（`regroup_unavailable_or_cooling`），不循环选择不可用模式；配置严格校验（字符串 `"false"` 报错、比例 (0,1]、整数、enter<exit）。见 `DispatchTests` 六项。

## 真实执行的命令与结果

解释器 `C:\Users\ry\skill\smacv2_jev_adapter\.venv\Scripts\python.exe`：

| 命令 | 结果 |
| --- | --- |
| `-m unittest skills.current.tests.test_group_skills`（cwd=C:\Users\ry\skill） | **61/61 通过**（重写后） |
| `-m unittest skills.current.tests.test_approach_progress` | 11/11 通过 |
| `-m unittest discover -s tests`（cwd=game_skill\micro） | 407/407 通过 |
| `-m unittest discover -s tests`（cwd=smacv2_jev_adapter） | 36/36 通过 |
| `bridge.runner.parser()` 解析 `--enable-regroup --auto-spatial-dispatch` + `manager_configuration` | 通过，kwargs 含 `spatial_params` |
| `jev_micro.run.build_parser()` 解析 v8 开关（上轮已接） | 通过 |
| 10人30步行军仿真 | 从 x∈[0,9] 推进至 [20,24] 抵达航点 |
| 性能（10单位/步，500样本，含冷启动样本） | regroup mean 0.79ms / p95 1.15ms；search mean 1.17ms / p95 1.59ms；positioning mean 0.57ms / p95 0.73ms |

跳过：0。真实对局：0 局（未执行）。

## 修正的旧结论（v8 验收报告中）

- "线性复杂度"**不准确**：两两分组、跨度计算与碰撞比较含 O(n²) 项，整体随人数呈二次增长（10人实测 <2.5ms，SMAC 规模无碍），已在 GROUP_SKILLS.md 更正。
- "SMACv2 接入"此前仅接线了 SMAC-Hard 入口；本轮补齐 `smacv2_jev_adapter/bridge`。
- 旧测试"输入顺序无关/时间回退/间距配置/跨窗口预算/部分动作合并"断言不足，已按任务11重写，不保留旧的弱断言。

## 剩余限制与未执行项

1. 任务12 的 5v5/10v10 冒烟与 A0–A3 配对消融未执行（无 SC2 运行）；命令模板已具备：`smacv2_jev_adapter/.venv/Scripts/python.exe run.py run --method adaptive-skill --manifest <m> --output-dir <d> --enable-regroup --enable-cohesive-search --enable-combat-positioning --auto-spatial-dispatch`。
2. `approach_progress_guard` 在站位路径的 A-B-A 保护通过 guard_approach 复用与单测分支验证，尚未有站位专用的多步往返夹具。
3. 语义限制不变：距离为几何近似、访问航点≠区域清空、旧敌位仅是搜索线索。
