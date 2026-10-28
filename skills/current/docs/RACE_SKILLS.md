# 人族与虫族技能

## 2026-10-28 修复版（v2）

- 人族治疗以当前观察到的订单为依据。合法、未受危的持续治疗先保留；其他运输机不得争抢已被实际订单占用的目标。即使原治疗机本步取消/撤离，也须等下一帧确认订单释放后才能交接，避免同时交换目标被拒却误记为成功。计划、观察订单分开记录。能量低于1时不新发治疗命令（基于本项目实测反馈的保守门槛）。
- 无分配且有远处伤员时，运输机保留HealAssign的接近建议，并检查已知对空威胁；受危运输机保留自保建议。
- 人族支援撤退要求当前确有运输机合法分配给该伤员，不再仅凭远处有运输机就改方向。种族撤退覆盖统一检查：威胁数量和可见基础齐射压力不增加、预计重新开火不延后、避免反向/返回上一位置。仍保留每单位每局最多两次支援重定向。
- 刺蛇朝跳虫掩护的撤退重定向还要求比原撤退有明确的威胁数量或基础齐射压力改善；仅仅更靠近跳虫不够。
- 普通虫族单位面对较远毒爆时保留当前可行开火窗口；可合法发射、原生射程内且预计单次攻击足以击杀的毒爆优先拦截。非贴身紧急情况连续避让最多两步，之后交还普通战斗决策一窗口；贴身紧急情况允许继续避让。避让记录连续计数及例外原因，非紧急避让不得直接反向。
- 1.5距离、贴身几何阈值及两步预算均是启发式，不是爆炸半径或命中保证。共享底层默认off策略不变，毒爆聚群目标算法本轮不调整。

验证与真实40+40局结果见 `sc2_skill_adapter/analysis/race_skills_repaired_20261028/`（相对SMAC_agent根目录）。以下保留初版迁移来源与接口说明；行为差异以上述v2规则为准。

本版增加两个独立组合控制器，复用现有基础技能，不修改冻结历史版本。默认 `race_profile="off"` 仍执行原 AdaptiveFocusFire。人族与虫族策略需要显式启用，仅适用于当前三兵种场景。

## 已实现内容

| 控制器 | 迁移来源 | 初版行为 |
|---|---|---|
| `terran_combat.allocate_terran_combat` | AdaptiveFocusFire、HealAssign、移动评分与群体几何 | 输出单位沿用原集火/撤退；运输机接入联合治疗分配，跨窗口保持分配记忆；空能量/闲置运输机跟随友军主要群体；按确认可对空的武器判断威胁；明确不兼容目标被排除；已决定撤退的单位可有限地向治疗支援调整方向。 |
| `zerg_combat.allocate_zerg_combat` | AdaptiveFocusFire、集火预算、接敌与移动评分 | 跳虫分配局部目标并在连续受阻后侧移；刺蛇撤退偏向近战掩护；毒爆独立分配可见目标群并接近，排除普通伤害预算/低血撤退；逼近的敌毒爆可触发打断攻击的分散避让，协调候选落点。 |

人族支援撤退只修改原策略已经选择撤退的单位，且不得增加已知威胁覆盖，每单位每局最多调整两次方向。它不是额外获得两次撤退时间，也不会重置原撤退预算。

治疗仍采用现有一对一匹配。治疗动作会实际发给引擎，不把Stop当成“保持治疗订单”；`AlreadyTargeted`等原生反馈照实记录，不记作新增治疗量。未知武器兼容性不被断言为确认威胁。

毒爆的目标聚集距离、警戒距离是几何启发式，**不是已校准的爆炸半径/伤害**。没有用毒爆消失来推断命中；普通单位的最终攻击预算重新计算，不计毒爆未知爆发伤害。毒爆从原All-in兵力比例的分母和分子中排除，避免正常消耗直接触发该比例条件。

## 使用方式

Python接口（从 `SMAC_agent` 根目录导入）：

```python
from skills.current.skill_manager import SkillManager

manager = SkillManager(race_profile="terran")  # 或 "zerg"；默认 "off"
manager.select("adaptive_focus_fire", posture="balanced")
actions, telemetry = manager.step(payload, n_agents=10)
# 每局开头调用 manager.reset()，每步将最终动作交给现有运行器。
```

`race_profile` 只改变组合战斗模式；显式选用其他独立技能仍执行该技能原来的行为。两个新模块的纯函数接口与原分配器兼容，并增加 `race_memory` 输入和返回值。Manager负责跨窗口记忆、换局清理、同一观察重复调用缓存和被覆盖动作的旧接敌历史清理。

SMACv2 CLI（在 `SMAC_agent/sc2_skill_adapter` 执行）：

```powershell
.\.venv\Scripts\python.exe -B run.py scenarios --race terran --size 10 --episodes 10 --seed 3000 --unit-balance native --output configs/terran-initial-n10.json
.\.venv\Scripts\python.exe -B run.py run --method adaptive-skill --skill-version current --race-profile terran --manifest configs/terran-initial-n10.json --output-dir runs/terran-initial-n10
```

虫族将两处 `terran` 替换为 `zerg`，并使用独立输出目录。新参数不自动开启原有 `--adaptive-*` 优化或空间技能；如需与此前v7组合比较，应在两组中一致传入此前的优化开关和单位价值参数。测试输出目录必须未使用。

历史 `--skill-version baseline_*` 拒绝种族策略。CLI也拒绝与清单种族不匹配或配合非 `adaptive-skill` 方法使用。SMACv1 CLI暂未新增种族开关；通用Python Manager接口可用，但该环境的种族策略尚未做真实验证。

## 日志与验证

- `run.json` 的 `skill_parameters.race_profile` 记录显式种族配置，新源码纳入哈希。
- 每步 `race_telemetry` 记录治疗、动作替换、毒爆目标、几何参数和已清理的历史；普通攻击预测以最终动作重算。
- `skills/tools/verify_race_profiles.py` 用历史真实观察检查新动作合法性、输入不变、重复调用不重复消耗预算，并比较默认off与冻结v7动作是否相同。
- 该离线检查不能产生新胜率：后续观察由旧策略产生，并非新动作造成。

## 当前未完成的优化

本次是可运行的初版，尚未完成全部种族战术：人族对空火力预留、治疗净收益预测和完整射程阵型；虫族严格同步接敌、精确爆炸效果建模、反毒爆中原地击杀与逃离的权衡；两族集结/搜索的专门参数校准。现有Regroup/CohesiveSearch/CombatPositioning开关仍独立，未声称已针对两族调优。

只输出当前接口支持的移动、Stop、攻击/治疗动作；未添加兴奋剂、装载/卸载、潜地或手动自爆。没有新胜率保证，应先检查实际对局，再以独立种子做对照评测。
