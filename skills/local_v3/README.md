# 条件触发的局部技能 v3

Adaptive 每个决策步生成全队动作，本层检查合法候选，最多改变一个单位的一步动作，再由原管理器提交。下一步重新观察判断；默认同一单位至少间隔 2 个决策步，其他单位可以在间隔步修改。默认不限制整局修改次数，可在 [policy.json](policy.json) 中设置 `max_interventions`。

只有实际改变提交动作才计入干预次数。等待、条件不满足或沿用 Adaptive 时保留后续机会；每局重置冷却和次数，相同时钟的重复请求使用缓存。

## 技能与选择

完整合法目录包括靠拢、换目标、撤退和撤退改攻击。默认启发式优先换目标，其次靠拢；同类优先较久未被修改的单位。

- 靠拢：至少 6 个单位存活，执行者正在接敌、没有合法攻击、最近队友距离大于 6，已知最近敌距不超过 12；选择向队友几何中心靠近的合法一步。
- 换目标：武器就绪且原目标伤害信息可用，避免放弃当前确定击杀或重复覆盖已有充足火力的目标；按有效伤害边界及确定击杀比较合法备选。
- 学习修正：策略的 `corrections` 可从四类全部合法动作中选择替代或撤销当前选择，受合法性、单步单单位、冷却和次数预算约束。

## 运行

从仓库根目录执行；以下 `native` 命令会启动游戏。兼容入口统一审计日志与重放中的 tuple/list 表示，使用同一 v3 执行器。

```powershell
$py = '.\sc2_skill_adapter\.venv\Scripts\python.exe'
& $py -B -m decision_tree.scripts.conditional_compat native --manifest <神族10v10场景清单.json> --output-dir <新的结果目录>
```

`--policy <策略.json>` 替换默认配置。日志保留合法菜单、等待原因、最终动作和干预计数；是否被引擎接受由原生回执核验。训练与整局验证见 [decision_tree](../../decision_tree/README.md)，采集分支上限与运行时干预次数分别配置。

```powershell
python -B -m unittest discover -s skills/local_v3/tests
python -B -m unittest decision_tree.conditional.test_objective
```
