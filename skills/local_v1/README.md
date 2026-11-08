# 单步局部技能 v1

本模块在 Adaptive 已生成的基础动作上提出局部修改，不持有环境或推进时间。每个候选只修改一个执行者的一个决策步，其余动作由基础控制器负责。

| 类别 | 修改 |
| --- | --- |
| `retarget` | 将攻击改为另一个合法可见目标 |
| `retreat_veto` | 撤退或恢复中的单位改为合法攻击 |
| `retreat_step` | 受威胁单位采用一步合法撤退移动 |
| `regroup_step` | 无合法攻击、与队伍脱节的单位向队友中心靠拢 |

`generate_candidates(payload, baseline_result, config=None)` 返回带绑定、特征和效果标识的候选。默认最多 24 项、每类最多 8 项；按最终动作去重，别名不能重复计作训练证据。医疗单位不属于此接口的局部修改对象。

`apply_candidate(payload, baseline_result, candidate)` 复核合法性，返回深拷贝结果，重建最终攻击分配并标记被覆盖的状态增量。管理器再统一提交动作及历史；生成候选本身不修改输入。模型训练见 [局部收益与纠错流程](../../decision_tree/local/README.md)。

```powershell
# 从仓库根目录运行；不启动游戏。
python -B -m unittest skills.local_v1.tests.test_library
```
