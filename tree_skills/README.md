# 医疗目标偏好评分与树蒸馏

可选模块：只替换原医疗技能已决定治疗时，对合法“医疗单位—伤员”配对的偏好评分。参与集合、动作合法性、订单、联合分配和执行仍由原技能负责。该通道默认关闭，与 `decision_tree` 的局部控制训练独立。

| 目录 | 用途 |
| --- | --- |
| `medivac/` | 状态特征与 original/tree/teacher 配对评分接口 |
| `data/`、`training/` | 教师标签、按对局分组的树训练 |
| `export/` | 导出 JSON 规则和纯标准库推理 |
| `integration/` | off/shadow/teacher-score/tree-score 接入 |
| `teacher/` | 外部权重检查和教师接口 |
| `models/` | 部署模型目录；权重与实验输出不包含在源码中 |

入口为 `skills.current.skills.heal_assign.assign_healing(..., pair_scorer=None, pair_scoring="off")`。合法候选内的负偏好分有意义，不应取绝对值或截断。导出模型加载时检查特征 schema；未就绪教师不能作为有效监督源。

从仓库根目录执行离线接口检查：

```powershell
python -B -m unittest tree_skills.tests.test_pair_scorer_channel tree_skills.tests.test_features
```

训练使用独立的含 scikit-learn/joblib 环境，教师推理另需对应模型与 PyTorch；导出 JSON 推理不需要这些训练依赖。训练参数见 `python -m tree_skills.training.train_tree --help`。第三方来源与许可见 [技能库声明](../skills/THIRD_PARTY_NOTICES.md)。
