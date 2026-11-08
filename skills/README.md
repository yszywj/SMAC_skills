# 微操技能库

环境观察和动作转换由 [sc2_skill_adapter](../sc2_skill_adapter/README.md) 提供。本目录负责技能逻辑与管理器，训练和选择由 [decision_tree](../decision_tree/README.md) 提供。请从整个仓库根目录运行 Python 模块，保持 `skills.*` 导入路径。

| 目录 | 用途 |
| --- | --- |
| [current](current/README.md) | 当前 Adaptive 控制器、集火、撤退、治疗、移动和状态管理 |
| [local_v1](local_v1/README.md) | 单单位、单步的候选生成和动作合并接口 |
| [precontact_regroup](precontact_regroup/README.md) | 可独立调用的接敌前单兵靠拢 |
| [local_v2](local_v2/README.md) | 靠拢、换目标和撤退的动作实现及探索目录 |
| [local_v3](local_v3/README.md) | 每步条件触发、单位冷却和可学习修正层 |
| [baseline_v2_70pct](baseline_v2_70pct/README.md) | 历史 v2 对照源码 |
| [baseline_v6_20261024](baseline_v6_20261024/README.md) | 冻结的 v6 对照源码 |

旧 `baseline_v7_20261024` 入口保留在适配器的 `analysis/skill_v7_20261024/source_snapshot/skills/current` 路径中，只包含运行所需的冻结源码与哈希清单，不包含历史对局和分析结果。

纯技能依赖可用 `python -m pip install -r skills/requirements.txt` 安装。真实对局还需适配器和游戏环境，见仓库根 README。

```powershell
python -B -m unittest skills.local_v1.tests.test_library
python -B -m unittest discover -s skills/local_v2/tests
python -B -m unittest discover -s skills/local_v3/tests
```

Adaptive 先生成基础动作，局部技能返回候选，管理器统一处理最终动作、目标分配和状态提交。候选生成不推进环境；局部执行预算与反事实训练分支预算分别配置。模块版本代表实现差异，不代表效果排序。

第三方来源和许可证见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) 与 [licenses](licenses)。
