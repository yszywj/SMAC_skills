# 局部技能动作 v2

复用局部调用接口，提供靠拢、撤退和换目标的独立动作实现。调用选择与动作实现可以分别比较：`--selector` 指定调用策略，`--implementation v1|v2` 指定内部动作版本。

`enumerate_bindings` 枚举合法绑定，`analyze_binding` 生成诊断特征，`resolve_action` 根据选中的绑定生成动作。相同最终效果合并，候选记录不等于已测收益。

从仓库根目录运行，清单需由适配器生成：

```powershell
$py = '.\sc2_skill_adapter\.venv\Scripts\python.exe'
& $py -B -m skills.local_v2.run --manifest <场景清单.json> --implementation v2 --output-dir <新的结果目录>
& $py -B -m skills.local_v2.explore --help
```

`run` 启动真实对局；默认选择器为独立单兵靠拢。可传入 `--selector <策略.json>`。强制动作实验需同时提供 `--force-step` 和 `--force-binding`，并使用单场景清单；它会分别执行 BASE 和候选。

```powershell
python -B -m unittest discover -s skills/local_v2/tests
```

连续条件触发与单位冷却见 [local_v3](../local_v3/README.md)。
