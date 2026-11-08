# 接敌前的单兵靠拢

本技能在 Adaptive 生成全队动作后，让一个脱节单位向队友中心移动一步。默认托管方式每局最多提出一次修改，之后回到基础控制器。连续条件触发版本见 [local_v3](../local_v3/README.md)。

## 条件与接口

执行者存活、正在接敌、无合法可见攻击；全队至少 6 个单位存活，执行者距最近队友大于 6。选择能靠近其他存活队友几何中心的合法移动，排除与基础动作相同的修改。候选上限和排序由 [policy.json](policy.json) 固定。

选择靠拢候选后，最近可见敌距大于 12 时撤销；该次撤销消耗托管选择机会。未知敌距按本版本策略保留靠拢，不能与 v3 的等待规则混用。

`propose(payload, baseline)` 返回 `(candidate, vetoed)`，不修改输入或推进环境。`candidate is None` 表示继续等待；候选被撤销时保持基础动作。未撤销候选可通过 `skills.local_v1.apply_candidate` 合并，交由管理器统一提交。`attach(manager, context_hash=...)` 使用托管控制器，调用方负责 step/commit/reset 生命周期。

## 运行

从仓库根目录执行；第二条命令启动游戏：

```powershell
$py = '.\sc2_skill_adapter\.venv\Scripts\python.exe'
& $py sc2_skill_adapter/run.py scenarios --race protoss --size 10 --episodes 2 --seed 4821000 --unit-balance native --output decision_tree/artifacts/precontact_manifest.json
& $py -B -m skills.precontact_regroup --manifest decision_tree/artifacts/precontact_manifest.json --output-dir decision_tree/artifacts/precontact_example
```

现有 Adaptive 运行命令也可通过 `--local-policy skills/precontact_regroup/policy.json` 接入；需保持策略要求的执行上下文与参数。
