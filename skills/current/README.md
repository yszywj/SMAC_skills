# 当前 Adaptive 技能

`skill_manager.py` 管理技能选择、执行窗口、终止、跨步记忆和动作提交。`skill_policy.py` 定义技能调用与参数协议。环境入口见 [适配器](../../sc2_skill_adapter/README.md)。

| 组件 | 职责 |
| --- | --- |
| `skills/adaptive_focus_fire.py` | 集火、撤退与重新接敌的组合控制 |
| `skills/focus_fire.py` | 攻击目标与火力预算分配 |
| `skills/cover_retreat.py` | 危险判断、撤退与掩护 |
| `skills/heal_assign.py` | 合法治疗分配，可选偏好评分通道 |
| `skills/coordinated_approach.py` | 协同接敌和移动目标保持 |
| `skills/combat_timing.py`、`skills/damage_model.py` | 武器时机与伤害估计 |
| `skills/movement_scoring.py` | 合法移动评分 |
| `skills/regroup.py`、`skills/cohesive_search.py`、`skills/combat_positioning.py` | 可选的集结、搜索和空间站位 |

管理器的 `local_decision_hook` 在 Adaptive 产生动作后、历史提交前接入局部修改。技能选择、动作执行与局部修改使用同一最终提交路径。每局应重置管理器和相关策略状态。

从仓库根目录运行离线测试：

```powershell
python -B -m unittest discover -s skills/current/tests
```

通过适配器 `--skill-version current` 选择本源码版本；具体参数以运行入口 `--help` 为准。历史对照源码见 [技能库目录](../README.md)。
