# 人族与虫族初版实现验证

## 实现及范围

当前源码新增 `current/skills/terran_combat.py` 与 `zerg_combat.py`，通过 `SkillManager(race_profile=...)` 和SMACv2的 `--race-profile` 接入。默认off；历史版本、错误种族清单及非adaptive-skill方法均拒绝显式种族策略。详细行为及未实现部分见 [种族技能说明](../current/docs/RACE_SKILLS.md)。

迁移后依赖定位同时修正：当前 `SMAC_agent/skills` 和 `sc2_skill_adapter` 使用原外部 `smacv2`、`game_skill` 环境，不改旧junction。两个环境入口帮助与源码/依赖隔离测试通过。

## 离线检查

- 原有技能回归76项、人族新增10项、虫族新增10项，共96项通过。
- 适配器全套57项通过，其中新增6项种族入口及Manager生命周期检查。
- 合计153项检查通过。
- 历史真实观察离线重放：人族40局2,026帧，虫族40局1,171帧，共3,197帧。
- 所有新策略动作均在对应帧合法候选内；死亡单位保持动作0；未修改输入观察；同帧重复调用不重复消耗技能窗口或种族记忆预算。
- 在同样历史参数下，当前默认off与冻结v7的每帧动作一致。
- 新策略相对默认策略改变人族1,360条、虫族550条单位动作。这仅证明策略分支生效，不证明动作更好。
- 机器结果和源码哈希：[race_profiles_initial.json](race_profiles_initial.json)。

复核命令（从SMAC_agent目录）：

```powershell
.\sc2_skill_adapter\.venv\Scripts\python.exe -B -m unittest skills.current.tests.test_approach_progress skills.current.tests.test_group_skills skills.current.tests.test_spatial_review_regressions skills.current.tests.test_terran_combat skills.current.tests.test_zerg_combat -q
.\sc2_skill_adapter\.venv\Scripts\python.exe -B skills/tools/verify_race_profiles.py --output skills/verification/race_profiles_initial.json
```

适配器检查在sc2_skill_adapter目录运行 `.\.venv\Scripts\python.exe -B -m unittest discover -s tests -q`。

## 真实仿真功能验证

预先按是否包含对应支援/爆发兵种选择诊断阵容，不按胜负选种子：人族seed2800含医疗运输机，虫族seed2801含毒爆（seed2800不含毒爆，因此取下一种子）。每族仅1局10v10，使用原v7优化开关、原生属性、难度7及显式种族策略。

| 种族 | 结果 | 决策步 | 原生反馈 | 观察期动作错误 |
|---|---|---:|---|---:|
| 人族 | 败 | 57 | 治疗Success 13次；AlreadyTargeted 29次 | 0 |
| 虫族 | 败 | 42 | 无拒绝反馈 | 0 |

两局均正常结束，保存完整日志与回放。人族确实执行支援/治疗分支，虫族日志记录毒爆接近与攻击角色。AlreadyTargeted保留为原生非成功反馈，不合并成成功或当作新的对局失败。

原始结果位于 `sc2_skill_adapter/runs/race-skills-initial-terran-smoke` 和 `race-skills-initial-zerg-smoke`；配置及机器汇总位于 `sc2_skill_adapter/analysis/race_skills_initial`。

## 结论边界

本轮验证初版可以加载、调度、生成合法动作并完成真实仿真；没有开展新策略胜率评测。两局均败，不能以功能验证证明效果改善。历史观察离线重放也不能估计新策略的胜率，因为后续观察由旧策略动作产生。

毒爆仍采用明确标注的几何启发式，未校准爆炸半径、伤害及效果归因；分散可能牺牲输出。完整人族阵型、对空火力预留、治疗净收益模型和虫族严格同步接敌等仍属后续优化范围。
