# 本地接口审计（2026-10-28）

审计对象为 `SMAC_agent` 当前工作区源码。版本标识使用文件 SHA-256（前 16 位十六进制），
不使用任何 Git 元数据。审计期间未修改任何被审计文件。

| 文件 | SHA-256（前16位） | 行数 |
|---|---|---|
| skills/current/README.md | 3d715f72aae8ea6f | — |
| skills/current/skill_policy.py | 90c56af8ff4bc7c6 | 251 |
| skills/current/skill_manager.py | 71464d9b191594e9 | 1047 |
| skills/current/skills/adaptive_focus_fire.py | 26d39cfaa538d743 | 571 |
| sc2_skill_adapter/adapter.py | fb7033a511c9a7e1 | 64 |
| sc2_skill_adapter/bridge/config.py | 4786c7f0679a90f3 | 113 |
| sc2_skill_adapter/bridge/runner.py | a3e377f3952e11e0 | 496 |
| sc2_skill_adapter/bridge/environment.py | 880f241cc29d6249 | 159 |
| sc2_skill_adapter/bridge/policy.py | 9dac7e3956acc863 | 422 |
| sc2_skill_adapter/bridge/feedback.py | a0810dc39ff6de43 | 77 |
| sc2_skill_adapter/bridge/skill_runtime.py | 0076a8273dc04e27 | 171 |
| sc2_skill_adapter/bridge/planner.py | 3a6f1ac66443e55d | 225 |

## 1. 快照绑定契约（skill_policy.py）

- `skill_choices(visible_enemies, healers_present, max_explicit_targets=4, spatial_modes=())`
  生成动态候选表，返回 `(criteria, priorities, modes)` 三个以候选 id 字符串为键的映射。
  候选顺序：`0` direct、`1` focus_fire_budget(auto)、`2` cover_retreat、（有治疗需求时）`3`
  heal_assign、随后最多 `max_explicit_targets` 个显式集火目标（键为 e 前缀敌方 ID 排序后）、
  再追加 adaptive_focus_fire 三姿态（aggressive/balanced/defensive，`priorities` 中携带姿态
  字符串而非目标 ID）、最后仅当显式启用时追加 regroup / cohesive_search。
- `build_choice_context(...)` 用与请求完全相同的参数在本地重建并保存该候选映射；runner
  将其传给 `decode_skill(payload, answers, choice_context=...)`，此映射是权威解码表，
  模型永远不提供该映射。非默认 `max_explicit_targets` 或启用空间候选且无 choice_context
  时 decode_skill 直接抛错（skill_policy.py:235-242）。
- `skill_request(payload, model, skill_max_steps, ...)` 从 payload["state"] 构造团队级摘要，
  只使用团队可见事实（allies、visible_enemies、relations、heal_candidates 等），并记录
  `explicit_target_limit` 以支持后续解码校验。
- 单位 ID 格式为 `u{i}` / `e{i}`（i 为环境索引），不要写死 a0/e0。

## 2. SkillManager 生命周期（skill_manager.py）

- 构造参数：`max_steps=4, adaptive_params, combat_params, unit_value_params,
  spatial_params, race_profile∈{off,terran,zerg}`。`reset()` 每局开始必须调用；
  姿态、All-in、撤退历史（unit_states/_last_move/_last_seen_enemies）跨 skill 窗口存活
  但不跨局。
- `select(skill, priority_target=None, posture=None)`：设置 `mode_remaining = max_steps`，
  `active` 属性要求 `termination_reason is None`。`step(payload, n_agents)` 每步必须给新鲜
  快照；同一快照的 `_observe()` 幂等（观测时钟 `game_loop` 相同则跳过，倒退则清空空间记忆）。
- `commit_final_actions(final_actions)`：仅空间技能产生 `_pending_spatial_moves` 时有意义，
  从**最终**合并后的动作数组记录移动历史；提案被覆盖则不进入执行历史。同快照幂等。
- 终止原因区分：`max_steps`、`safe_streak`、allocator `terminated`、
  `interrupted_<reason>`、`regroup_complete` 等；interrupted/timeout 从不算成功。

## 3. 部分动作合并与空间技能中断交接（runner.py:253-276）

- 空间技能 `step()` 返回 `(partial_dict, telemetry)`（index→action 的**部分提案**）。
  interrupted 且空提案时，立即在**同一决策边界** `select("adaptive_focus_fire", posture=...)`
  并重新 step，事件标记 `source=spatial_interrupt_handoff`；否则用声明基线
  `nearest_actions(payload, n_agents)` 填满其余单位，再按 `partial` 覆盖，telemetry 记录
  `base_policy="nearest"` 与 `merged_units`。
- heal_assign 同样返回部分动作（仅治疗位），由调用方合并基线。
- 每步最终动作提交前校验：数组长度等于 n_agents、逐单位合法掩码
  `env.get_avail_agent_actions(i)[action]`（runner.py:329-333），然后
  `commit_final_actions(actions)`，再 `env.step(actions)`。

## 4. adaptive 的配置语义

- 姿态在 `select()` 时传入并可续期；All-in（`all_in/all_in_reason`）、stall_steps、
  contacted、per-unit role history 均为局内持续状态，窗口到期重选不清除。
- 撤退预算由 `adaptive_params`（ADAPTIVE_PARAMS 默认 + 覆盖）与 `combat_params` 合并控制；
  race_profile=terran/zerg 切换 race allocator 并启用 `_race_memory`。
- `skill_runtime.manager_configuration(args)` 决定可信配置（combat 开关、unit_value、
  spatial、max_steps、posture、race_profile）；模型不得改这些参数。

## 5. 观察/动作契约（bridge/policy.py）

- 单位动作为整数索引：0=noop（仅死亡）、1=stop、2-5=N/S/E/W、`6+j`=攻击可见 `e{j}`
  （Medivac 则为治疗 `u{j}`）。合法掩码逐单位、逐步获取；敌人只在共享视野内可见，
  离开视野不等于死亡。
- `build_request()` 构造完整 payload（allies/visible_enemies/relations/
  movement_candidates/heal_candidates/unit_type_profiles/…），隐藏敌军与全局 state 不进入。

## 6. 旧 Codex planner 可复用部分（bridge/planner.py）

- `find_codex()`：`shutil.which("codex")` 后，Windows 上若命中 `.cmd/.ps1/.bat` npm shim
  则到 `%APPDATA%/npm/node_modules/@openai/codex` 下找原生 `codex.exe`；否则报
  `planner_native_codex_not_found`。可复用。
- 安全调用模式：参数列表（无 shell=True）、`codex exec --ignore-user-config --ephemeral
  --skip-git-repo-check --sandbox read-only --disable apps/shell_tool/hooks/browser_use/
  computer_use -c web_search=disabled -c project_doc_max_bytes=0
  --model … --output-schema … --output-last-message … --json --color never -`，
  prompt 走 UTF-8 stdin。可复用该模式。
- 结果校验：逐行解析 stdout JSONL 事件流，要求 `turn.completed`、拒绝
  `error/turn.failed` 及任何 `command_execution/file_change/mcp_tool_call/web_search`
  item（视为失败），并读取 `--output-last-message` 结果文件做 Schema 校验。可复用。
- **不可复用**：一次性 `plan.json` 固定文件与“已存在即报错”的单次缓存、固定地图计划
  Schema、默认模型名 `gpt-6-astra`——新客户端每次请求用独立目录/结果文件，模型可配置。

## 7. 外部依赖现状

- Codex CLI：本机 `codex --version` = `codex-cli 0.160.1`，npm shim 位于
  `C:\Users\ry\AppData\Roaming\npm\codex`（需解析到原生 exe）。
- SC2：`C:\Program Files (x86)\StarCraft II` 存在。
- SMACv2 源码：`C:\Users\ry\skill\smacv2`（含 `10gen_{protoss,terran,zerg}.SC2Map`）。
- `game_skill`（SMAC-Hard / SMACv1 路径）：`C:\Users\ry\skill\game_skill` 存在；
  本期 Agent 后端不接入 SMACv1，未实现时明确报 unsupported。
- Python 3.13.14（系统）；langchain_agent 使用独立 venv。

## 8. 调用顺序总结（agent 模式必须遵守）

1. 每步 `build_request(...)` 取新鲜快照（等待模型时游戏时钟不得推进，`assert_clock`）。
2. 需要选择时：`skill_request(...)` + `build_choice_context(...)`（同参数）→ 决策 →
   `decode_skill(..., choice_context=...)` → `manager.select(...)`。
3. `manager.step(payload, n_agents)`（active 时不再 select，除非新决策或中断交接）。
4. 合并/校验最终动作 → `commit_final_actions` → `env.step` → `action_feedback`。
