# HEAL_NEW 收益树实验

状态：**开发采集、模型拟合和独立验证均已完成，2026-11-04**。40 个开发场景运行 80 局，形成 31 条有效配对标签（16 正、13 负、2 零）；7 对因没有候选机会拒收，2 对因多医疗机动作差异拒收。已拟合 5 节点收益树，再用独立 20 场景运行 BASE／MODEL 共 40 局。本轮合计 120 局。内部验证表现下降，实际验证两臂均 9 胜 11 负，模型平均回报略低，**尚未观察到模型改善**。保留为实验版本，没有替换默认策略；最终 20 个测试场景仍未使用。

本轮没有修改运行源码，只新增候选配置、实验数据和模型产物。

## 冻结配置与一次干预

实验计划见 [plan.json](../artifacts/heal_switch_20261104_v1/plan.json)。本轮使用 `bound` 候选配置：

```json
{
  "intent": "HEAL_NEW",
  "base_intents": ["HEAL_CURRENT", "KEEP_HEAL"],
  "min_game_loop": 0
}
```

原有手写动作树的 **17 个节点保持冻结**。当原树选择继续治疗当前患者时，由原有 `bindings` 确定合法的新患者及对应治疗动作；本轮学习的是是否采用这一候选。每局在所有医疗机中合计只取**首次合法且动作不同的机会**：BASE 保留原树选择，CANDIDATE 改选新患者，之后都继续使用同一冻结基础控制器。

本配置没有 idle 实验的连续 STOP 三窗、能量至少 5、距离上限 12 等附加条件。原有绑定规则仍然生效，包括能量至少 1、合法且受伤的地面患者、有效治疗动作及患者预约约束。候选不会绕过合法动作掩码或占用检查。

## 场景与结果目录

两批使用原开发清单中的 40 个场景，互不重复，每个场景分别运行 BASE 和 CANDIDATE，合计已完成 **80 局**。

| 批次 | 原清单场景编号 | 场景数／完成局数 | 运行与审计输出目录 |
| --- | --- | --- | --- |
| pilot | 1、2、3、4、11、18 | 6／12 | `sc2_skill_adapter/runs/heal-switch-pilot-v1` |
| remaining | 原 40 场景中其余编号 | 34／68 | `sc2_skill_adapter/runs/heal-switch-remaining-v1` |

pilot 包含先前观察到协调风险的场景，用于检查协议；没有按本轮候选收益正负筛选。正式批次不重复先导场景。旧 idle 实验的数据不作为本轮训练标签复用。

两批策略和清单分别冻结在实验根目录的 `pilot/study` 与 `remaining/study`。独立的 20 个验证场景位于 `validation/study`，当前真实评估输出目录为 `sc2_skill_adapter/runs/heal-switch-validation-v1`；另有 **20 个最终测试场景尚未使用**。

## 配对标签与整体表现

标签为同一场景中 `CANDIDATE 整局累计 reward − BASE 整局累计 reward`，正、零、负差值均保留。审计要求运行配置、源码、种子和初始状态一致，机会前完整可见观察与动作前缀一致，机会及候选上下文吻合，且该步仅一名医疗机的实际动作不同。最终动作和引擎反馈也必须符合所选方案。

这是独立重跑后的 `matched_observed_prefix` 证据，不等同于恢复 SC2 内部状态的快照分叉。基线重复当前治疗而得到 `AlreadyTargeted` 时，只在既有观察订单、目标和实际命令完全吻合的受限条件下记为保持既有订单，原始拒绝反馈保留；不据此声称发生了治疗。

预约变化可能连带改变另一名医疗机的动作，此类配对拒绝成为单次干预训练标签。没有机会也不能补成零收益标签。不过，这些完整合法对局仍计入整体策略成绩；真实命令失败也保留在整体表现中，避免仅统计成功触发案例。

开发结果见 [development_result.json](../artifacts/heal_switch_20261104_v1/development_result.json)，两例协调连带变化见 [coordination_audit.json](../../sc2_skill_adapter/runs/heal-switch-pilot-v1/coordination_audit.json)。全部 40 对完整对局中，原树胜 **13／40（32.5%）**，首次机会固定采用候选的策略胜 **12／40（30%）**；后者相对原树的整局平均 reward 差为 **−0.301668250962374**。这些是 BASE 与强制候选的成绩，**不是训练后收益树的成绩**。

## 已完成的训练

两批 `paired.json` 的 31 条有效标签已经合并并按场景组划分为 **23 条训练、8 条内部验证**。输入为 46 项固定状态／候选特征，学习目标是候选相对基础选择的终局收益差，无需教师网络。

训练采用最大深度 2 的浅回归树，每叶至少 4 条样本且覆盖至少 4 个独立场景。场景内多条记录不增加该场景的总训练权重；默认按场景组划分 25% 内部验证集，不能把同一场景拆到训练和验证两侧。有效数据不足时停止训练，不降低覆盖门槛。

本次在 `C:\Users\ry\skill\SMAC_agent` 下使用的训练命令：

```powershell
.\sc2_skill_adapter\.venv\Scripts\python.exe -m medivac_tree.learning.train `
  --dataset sc2_skill_adapter/runs/heal-switch-pilot-v1/paired.json `
  --dataset sc2_skill_adapter/runs/heal-switch-remaining-v1/paired.json `
  --policy medivac_tree/artifacts/heal_switch_20261104_v1/pilot/study/base.json `
  --output-dir medivac_tree/artifacts/heal_switch_model_v1 `
  --max-depth 2 --min-samples-leaf 4 --min-leaf-groups 4 `
  --validation-fraction 0.25 --split-seed 0
```

模型产物为 [policy.json](../artifacts/heal_switch_model_v1/policy.json)、[model.json](../artifacts/heal_switch_model_v1/model.json) 和 [report.json](../artifacts/heal_switch_model_v1/report.json)。该输出目录现已存在；重新训练必须使用新目录。模型预测增益大于零时选择候选，否则保留基础动作；仍然遵守每局一次机会的协议。

实际收益树有 **2 个判断节点、3 个叶子，共 5 个节点**：

| 叶子路径 | 预测候选增益 | 训练样本数 | 选择 |
| --- | ---: | ---: | --- |
| `candidate_target_loss_ratio <= 0.06` | −1.0413985 | 6 | 保留 BASE |
| 上述比例 `> 0.06`，且 `energy <= 47.855` | −0.0685466 | 11 | 保留 BASE |
| 上述比例 `> 0.06`，且 `energy > 47.855` | +0.8791157 | 6 | 采用候选 |

`candidate_target_loss_ratio` 表示候选患者**自上一决策观察以来的生命损失／最大生命值**，不是患者当前已损失的总血量比例。表中阈值和增益是本轮数据拟合结果，不是已证实可迁移的治疗规则。

离线复用已有配对结果时，模型选择相对 BASE 的平均增益在训练集为 **+0.2293345**，在内部验证集为 **−1.2670782**。内部验证中，BASE 胜率为 **37.5%**，模型选择对应的胜率为 **25%**。这些指标没有新运行模型策略对局；内部验证下降说明目前不能宣称该模型取得改进。

## 已完成的独立实际验证

冻结模型后，在原先未参与候选设计或拟合的 20 个验证场景上完成 BASE 和 MODEL 两臂，共 40 局。结果见 [实际对局比较](../../sc2_skill_adapter/runs/heal-switch-validation-v1/comparison.json) 和 [模型评估记录](../artifacts/heal_switch_model_v1/evaluation.json)。

| 指标 | BASE | MODEL |
| --- | ---: | ---: |
| 胜／负 | 9／11 | 9／11 |
| 胜率 | 45% | 45% |
| 平均整局回报 | 13.5631850531 | 13.5596025083 |
| 存在候选的局数 | 17 | 17 |
| 实际采用换患者候选的局数 | 0 | 2 |

MODEL 在 17 个机会中保留原选择 15 次、改选 2 次；另有 3 局没有机会。全部 20 对均进入评估。平均回报差为 **−0.0035825448**，19 对回报相同、1 对降低，胜负逐局相同。

两次真正改选发生在验证第 14、20 局，分别得到 0 和 −0.0716508964 的整局回报差，命令均通过实际反馈检查，且当步只有目标医疗机的动作改变，详见 [实际改选审计](../../sc2_skill_adapter/runs/heal-switch-validation-v1/model_interventions.json)。这些 MODEL 轨迹只用于本次评估，没有当成强制候选采集臂生成训练标签。

这次实测没有观察到改进。训练集中的正增益未转化为可靠的验证收益，因此模型继续作为实验产物保留，不替换默认原树，也不据此启用每步反复改选。下一轮应先分析有害改选与当前特征的不足，再决定是否修改候选或补充数据。

[拟合审核](../artifacts/heal_switch_model_v1/fit_audit.json) 已复核输入哈希、31 个场景分组、叶子覆盖和模型加载。原 `report.json` 保留拟合时的 `trained_not_game_validated` 状态，随后完成的真实评估单独记录在 `evaluation.json`，未改写冻结模型。最终 20 个测试场景继续保留；若依据本次独立验证结果修改方案，该验证集以后应视为开发证据。
