# 当前微操技能（v8.1）

新增人族/虫族初版组合控制器：`SkillManager(race_profile="terran"|"zerg")`，默认`off`保持原策略。SMACv2入口使用`--race-profile`显式启用。实现范围、迁移来源、使用方式及验证边界见 [种族技能说明](docs/RACE_SKILLS.md)。

2026-10-25 二次复核修复：集结完成不再忽略孤立单位；可见敌人立即中断集结；无提示搜索持有固定短程航点并在到点后轮换；SMAC-Hard JEV 路径合并空间部分动作，所有运行路径在合法性检查后记录最终动作。新增多步与真实运行器离线回归，见 `tests/test_spatial_review_regressions.py` 及两个运行器的空间接入测试。

SMACv2 两个 PowerShell 脚本现在支持 `-EnableRegroup`、`-EnableCohesiveSearch`、`-EnableCombatPositioning`、`-AutoSpatialDispatch`、`-PositioningSpacing` 和 `-PositioningLayout`。脚本开启集结或搜索时自动开启必要的规则调度；Python CLI 仍需显式传 `--auto-spatial-dispatch`。历史版本拒绝空间开关。

本目录是当前唯一维护的技能源码位置，文件从原 `game_skill/micro/jev_micro` 迁入。v7在v6基础上增加可选的接敌进展保护；v8增加三个默认关闭的群体空间能力（Regroup、CohesiveSearch、CombatPositioning）；v8.1完成 R01–R12 修复轮（统一观察时钟与最终动作提交、共享敌忆注入搜索、编队参考系与航点轮换、部分控制契约、合法 Stop 与全体友方占位、站位几何与独立启用、候选映射固定、SMACv2 入口接线）。迁移时的原始版本已另行冻结。

| 文件 | 作用 |
| --- | --- |
| `skill_manager.py` | 技能选择、执行窗口、跨窗口状态、空间技能调度与终止条件、遥测 |
| `skill_policy.py` | 技能名称、模型选择协议、参数与响应校验 |
| `unit_value.py` | 单位核心度/战术价值评估，为集火、保护和治疗提供可选评分 |
| `skills/adaptive_focus_fire.py` | 按局势组合集火、撤退和重新参战 |
| `skills/focus_fire.py` | 按目标优先级及伤害预算分配攻击 |
| `skills/cover_retreat.py` | 判断危险、选择撤退方向、安排掩护攻击 |
| `skills/heal_assign.py` | 为治疗单位匹配合法治疗目标 |
| `skills/combat_timing.py`、`skills/damage_model.py` | 开火窗口与伤害预算估计 |
| `skills/movement_scoring.py` | 比较原地与合法移动的收益 |
| `skills/coordinated_approach.py` | 协同接敌、目标保持与局部避让 |
| `skills/approach_progress.py` | 纯接敌时识别本人往返和连续受阻，跨目标保留短期移动记忆，短时避返 |
| `skills/engagement_control.py` | 跨角色移动历史、撤退预算与重新参战承诺 |
| `skills/group_geometry.py` | v8共享空间工具：友军连通分组、主要群、紧凑度、孤立度、角色识别、候选位评分 |
| `skills/regroup.py` | v8 Regroup：有预算的集结重组（独立顶层技能） |
| `skills/cohesive_search.py` | v8 CohesiveSearch：无接触时共同搜索且不脱节（独立顶层技能） |
| `skills/combat_positioning.py` | v8 CombatPositioning：战斗模式内部的站位子技能（非顶层模式） |
| `client.py` | 原协议依赖，提供 `skill_policy` 所需的响应验证；同时保留原模型客户端定义 |

纯 Skill 不实例化模型客户端，不调用 JEV 或 ASTRA。`client.py` 原样迁入以保持验证函数和异常类型兼容；使用项目现有虚拟环境即可满足依赖。

## v8 群体空间技能

三项能力互相独立、均可单独消融，全部默认关闭；关闭时旧行为、候选表顺序与解码保持逐字节一致。

- **Regroup（`--enable-regroup`）**：把明显脱节的队伍有预算地集结到主要群附近。集合锚点短期稳定；主体单位有等待预算，不因单个长期受阻的掉队者无限停留；出现接触或受伤立即中断并交还战斗控制。`interrupted`/`blocked`/`timeout` 不会报告为成功。
- **CohesiveSearch（`--enable-cohesive-search`）**：无当前敌人位置时以共享航点共同推进。目标优先级：有效最后可见位置 → 运行器搜索提示 → 基于合法移动候选的保守局部退化。到点未见敌即消耗航点；轻微脱节在搜索内部修正，严重脱节返回 `needs_regroup` 由调度器调用 Regroup。访问记忆只记录 `visited_waypoints`，绝不伪称区域已清空。
- **CombatPositioning（`--enable-combat-positioning`，可选 `--positioning-spacing compact|normal|spread`、`--positioning-layout free|front_back`）**：战斗控制器内部的站位子技能，只接管未被攻击/撤退/治疗/恢复任务占用、且没有合法攻击的单位；受保护动作作为固定动作参与空间预留但绝不被改写。多方向受敌时朝最近敌群展开，不冲几何中心。不可用或收益不足时退回原协同接敌。
- **本地规则调度（`--auto-spatial-dispatch`）**：无模型调用的规则入口——无接触且严重脱节→集结；无接触且可推进→搜索；出现接触→既有战斗控制器。带进入/退出阈值差、最短持续期与失败冷却，接触和受伤可立即打断。

边界与信息限制：空间技能只输出"部分动作提案"（仅受控单位），由调度层合并出最终 n_agents 长度动作数组，每单位每步只有一个动作来源；不利用隐藏敌情、地图边界缺失时只作几何近似；距离仅作评分，不宣称真实可达。v8.1 起每个唯一快照经过统一观察入口 `_observe()`（共享敌忆、冷却按 game_loop 到期、同快照幂等），执行历史只记录 `commit_final_actions()` 提交的最终动作；编队使用记忆持有的 march_center 参考系。详细接口、记忆与预算语义见 [docs/GROUP_SKILLS.md](docs/GROUP_SKILLS.md)，修复轮验收见 [docs/GROUP_SKILLS_REPAIR_ACCEPTANCE.md](docs/GROUP_SKILLS_REPAIR_ACCEPTANCE.md)。

新机制是否生效由运行参数决定；当前源码不等于全部开关默认开启。`run_improved_skill.ps1` 默认仍选择v6配置，以保持既有命令行为；运行v7必须显式传 `-PolicyVersion v7`，它在全部v6参数上加 `--adaptive-approach-progress-guard`。v8开关（`--enable-regroup` 等）同样必须显式传入；SMAC-Hard（`jev_micro.run`）与 SMACv2（`smacv2_jev_adapter` 的 `run.py run/suite --method adaptive-skill`）两个入口均已接入。此前机制见 [IMPROVED_SKILL.md](../../smacv2_jev_adapter/IMPROVED_SKILL.md)，v7评测方案见 [PROTOCOL.md](../../smacv2_jev_adapter/analysis/skill_v7_20261024/PROTOCOL.md)。

保护只作用于没有合法攻击的接敌移动。按本人连续观测位置识别A-B-A往返，不把敌人靠近当作本人推进；切换目标也不丢失此记忆。触发后最多两个决策窗口避返，并用局部碰撞、危险和预计接触进展选择路线。只有一条可走路线时不会把它封死；进入攻击、撤退、恢复等角色后清理记忆。日志同时保存自身推进、敌方移动贡献、候选评分及最后实际动作。它不是完整寻路器，也不改变合法攻击掩码。

此目录只负责依据结构化观察生成动作，环境、观察与合法动作掩码由运行器提供。SMACv2 默认加载本目录，SMAC-Hard 原有的技能模块导入也解析到这里；`policy`、`run`、`environment` 等环境模块仍在原包。今后修改当前技能请编辑本目录；历史对照保存在相邻 `baseline_v2_70pct`。

v7的5v5、10v10各40局已经完成，完整结果见 [v7测试报告](../../smacv2_jev_adapter/analysis/skill_v7_20261024/REPORT.md)。目录结构、版本选择及迁移验证见 [技能库说明](../README.md)。
