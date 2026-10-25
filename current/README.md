# 当前微操技能（v7）

本目录是当前唯一维护的技能源码位置，文件从原 `game_skill/micro/jev_micro` 迁入。v7在v6基础上增加可选的接敌进展保护；迁移时的原始版本已另行冻结。

| 文件 | 作用 |
| --- | --- |
| `skill_manager.py` | 技能选择、执行窗口、跨窗口状态、终止条件和遥测 |
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
| `client.py` | 原协议依赖，提供 `skill_policy` 所需的响应验证；同时保留原模型客户端定义 |

纯 Skill 不实例化模型客户端，不调用 JEV 或 ASTRA。`client.py` 原样迁入以保持验证函数和异常类型兼容；使用项目现有虚拟环境即可满足依赖。

新机制是否生效由运行参数决定；当前源码不等于全部开关默认开启。`run_improved_skill.ps1` 默认仍选择v6配置，以保持既有命令行为；运行v7必须显式传 `-PolicyVersion v7`，它在全部v6参数上加 `--adaptive-approach-progress-guard`。原环境CLI/suite也已接入此开关。此前机制见 [IMPROVED_SKILL.md](../../smacv2_jev_adapter/IMPROVED_SKILL.md)，v7评测方案见 [PROTOCOL.md](../../smacv2_jev_adapter/analysis/skill_v7_20261024/PROTOCOL.md)。

保护只作用于没有合法攻击的接敌移动。按本人连续观测位置识别A-B-A往返，不把敌人靠近当作本人推进；切换目标也不丢失此记忆。触发后最多两个决策窗口避返，并用局部碰撞、危险和预计接触进展选择路线。只有一条可走路线时不会把它封死；进入攻击、撤退、恢复等角色后清理记忆。日志同时保存自身推进、敌方移动贡献、候选评分及最后实际动作。它不是完整寻路器，也不改变合法攻击掩码。

此目录只负责依据结构化观察生成动作，环境、观察与合法动作掩码由运行器提供。SMACv2 默认加载本目录，SMAC-Hard 原有的技能模块导入也解析到这里；`policy`、`run`、`environment` 等环境模块仍在原包。今后修改当前技能请编辑本目录；历史对照保存在相邻 `baseline_v2_70pct`。

v7的5v5、10v10各40局已经完成，完整结果见 [v7测试报告](../../smacv2_jev_adapter/analysis/skill_v7_20261024/REPORT.md)。目录结构、版本选择及迁移验证见 [技能库说明](../README.md)。
