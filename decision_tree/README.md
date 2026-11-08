# 技能选择与反事实训练

基础 Adaptive 负责全队控制。本模块从公开观察生成合法候选，选择或撤销局部修改，再交给同一管理器更新动作与记忆。技能实现见 [skills](../skills/README.md)，环境入口见 [sc2_skill_adapter](../sc2_skill_adapter/README.md)。

| 目录 | 用途 |
| --- | --- |
| `conditional/` | 当前条件触发策略、受限采集、胜负优先目标与收益/撤销修正层 |
| [local](local/README.md) | 单次局部干预的收益树、纠错树、审计与评估 |
| `phase2/` | 冻结局部规则及后续训练接口 |
| `configs/` | 示例调用策略与基础环境配置 |
| `scripts/` | 训练包装、恢复、统计和兼容入口 |
| 顶层模块 | 较早的调用级候选、标量/多输出收益树接口 |

## 条件触发运行

从仓库根目录执行，需先安装适配器环境。以下命令启动真实游戏：

```powershell
$py = '.\sc2_skill_adapter\.venv\Scripts\python.exe'
& $py -B -m decision_tree.scripts.conditional_compat native --manifest <神族10v10场景清单.json> --output-dir <新的结果目录>
```

默认使用 `skills/local_v3/policy.json`，可传入 `--policy`。每个决策步最多修改一个单位；同单位默认间隔 2 步。整局次数预算与采集分支预算独立，具体机制见 [local_v3](../skills/local_v3/README.md)。

## 有界训练流程

`campaign` 提供固定种子协议的三轮训练：每轮 40 个场景，预先分成 30 个训练场景和 10 个留出场景。完整登记靠拢、换目标、撤退和撤退改攻击的合法目录；每个训练场景最多测 4 个额外反事实分支，按预定规则抽样。未测候选不产生收益标签。

每轮先执行 Adaptive 与当轮参考策略，再收集同前缀、单时点替换后的终局差异。按场景等权拟合浅收益树与撤销树，先比较胜负变化，再比较回报；叶子至少需要 4 个独立训练场景。冻结候选后进行训练集及留出集整局验证，通过采用条件才进入下一轮。单次干预收益不能直接等同于连续策略收益。

```powershell
# 只准备固定协议、场景清单和划分，不启动游戏。
& $py -B -m decision_tree.conditional.campaign prepare --root decision_tree/artifacts/my_campaign
# 采集与验证会启动真实游戏；依次完成第 1、2、3 轮。
& $py -B -m decision_tree.scripts.conditional_compat collect --root decision_tree/artifacts/my_campaign --round 1
& $py -B -m decision_tree.scripts.conditional_compat evaluate --root decision_tree/artifacts/my_campaign --round 1
```

`prepare` 从基础环境配置创建清单并检查本地已有场景，使用新的输出目录；重复场景会被拒绝。默认协议计划最多 720 局，加重试总上限 780 局。先检查生成的 `protocol.json`，再启动采集。日志、标签、模型和结果写入本地 `artifacts/`，不随源码发布。

```powershell
python -B -m unittest decision_tree.conditional.test_objective
python -B -m unittest discover -s skills/local_v3/tests
```

当前冻结策略带有源码与环境指纹门禁，要求匹配原 Windows 执行环境：SMACv2 位于仓库同级目录、SC2 5.0.17.98310，以及锁定的依赖和源码。更换平台、布局或版本后需重新采集/拟合并验证，不能绕过上下文检查。
