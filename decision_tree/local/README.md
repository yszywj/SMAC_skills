# 单次局部收益树与纠错树

此接口把完整 Adaptive 作为基础策略，在其生成动作后、提交历史前，选择一个单位的一步修改。局部技能来自 [local_v1](../../skills/local_v1/README.md)，包括换目标、撤退改攻击、撤退和靠拢。连续条件触发入口见 [上级说明](../README.md)。

默认每局最多提出一次正收益修改；撤销提议也消耗该次机会。候选按最终动作去重，支持不足或预测不佳时保留 BASE。收益标签为候选终局回报减 BASE，纠错标签为撤销旧模型选择后的增量回报。两类数据须保留场景、原始日志和冻结执行上下文；验证按独立场景划分。

`training` 从数据搜索浅树分裂，`authored` 接受外部提出的条件与拓扑，并核对叶子训练收益和场景支持。模型选择与纠错合并仍需完整策略运行验证。

从仓库根目录查看与执行：

```powershell
$py = '.\sc2_skill_adapter\.venv\Scripts\python.exe'
# 仅检查计划，不启动游戏。
& $py -B -m decision_tree.local.study collect --manifest <场景清单.json> --policy decision_tree/local/base.json --output decision_tree/artifacts/local_example --anchors 2 --max-candidate-games 80 --dry-run
# 对已完成且可重新审计的数据离线拟合。
& $py -B -m decision_tree.local.training gain --dataset <paired.json> --output decision_tree/artifacts/local_gain
& $py -B -m decision_tree.local.training veto --dataset <新的paired.json> --old-model decision_tree/artifacts/local_gain/model.json --output decision_tree/artifacts/local_veto
& $py -B -m decision_tree.local.evaluate --help
```

去掉采集命令中的 `--dry-run` 会启动游戏；`--plan-only` 仍会运行 BASE。候选预算是显式上限，超限时拒绝执行。旧数据的采集方式和执行上下文不同，不能直接混作当前连续策略的收益证据。
