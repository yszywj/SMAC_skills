# 改进后的纯 Skill：v6 使用与机制

本次实现日期：2026-10-23。底层沿用修复可见敌人读取后的 SMACv2，策略为本地 `AdaptiveFocusFire`，不调用 JEV 或 ASTRA。

技能源码现已迁入同级 `skills/current`，原运行命令和 v6 配置保持不变。历史 7/10 胜的 v2 原始技能单独保存在 `skills/baseline_v2_70pct`；目录、加载方式及迁移验证见 [技能库说明](../skills/README.md)。下文各批次成绩仍为迁移前的历史实测。

2026-10-24新增同场景对照：保留原有v6算法与参数，在新种子170–189上与恢复的v2分别进行5v5、10v10各20局，共80局。当前v6两规模均为15/20胜，旧v2分别14/20、12/20。完整数据、统计限制和双向轨迹案例见 [本轮报告](analysis/skill_v2_v6_paired_20261024/REPORT.md)；后续使用 `scripts/run_skill_version.ps1` 可明确选择实际源码版本。

v6 在 v5 配置上增加独立开关 `engagement_commitment`（CLI：`--adaptive-engagement-commitment`），协调危急脱离、恢复和重新参战：跨角色保存移动历史与个人撤退预算，只将本人观察到的冷却重启作为开火进展代理；重新接敌后给予最多 3 窗开火承诺，保留有限紧急脱离与超时退出。该开关从两个环境入口传给 `SkillManager`，有效配置进入 `run.json`、`summary.json`，新辅助模块 `engagement_control.py` 进入源码哈希。CLI 默认关闭；包装脚本默认 v6，显式开启。

## 已完成的 v6 实测

冻结同一份代码，复用种子 150–169，5v5、10v10 各 20 局；40 局初始状态逐局匹配 v4、v5，环境和攻击掩码一致。

| 场景 | v5 胜场/总局数 | v6 胜场/总局数 | v6 敌方平均消灭比例 | v6 平均步数 |
|---|---:|---:|---:|---:|
| 5v5 | 12/20 | 11/20 | 78% | 59.65 |
| 10v10 | 9/20 | 10/20 | 86% | 73.70 |

10v10 为 10 胜、9 负、1 平，平局不计胜场。其最长连续反向移动串从 v5 的 24 步降到 4 步，但结果仍低于 v4 的 12/20 胜、89% 消灭比例；5v5 则少 1 胜。v6 修复了已知行为问题，不能据此称为最佳胜率版本。406 项技能测试、29 项适配器测试通过；关闭新开关的历史 5,419 帧动作完全兼容。完整结果、逐局比较和证据见 [v6 报告](analysis/skill_repairs_v6_20261023/REPORT.md)。

v5 在 v4 基础上增加三个独立开关：`accurate_damage` 修正直接命中伤害预算中的属性加成与护盾处理；`critical_escape` 为极危单位判断脱离危险的机会；`planned_reentry` 根据每窗观察重新规划接敌，保留恢复步数上限。它们仍是基于当前可见状态的估计，不保证命中、存活或最优路线。CLI 默认全部关闭；包装脚本的 v5、v6 配置均显式开启这三个开关。

## 已完成的 v5 实测

已完成同种子 150–169 的 5v5、10v10 各 20 局。40 局初始状态逐局匹配 v4，环境与攻击掩码保持一致。

| 场景 | v4 胜场/总局数 | v5 胜场/总局数 | v4→v5 平均敌方消灭比例 | v5 平均步数 |
|---|---:|---:|---:|---:|
| 5v5 | 11/20 | 12/20 | 79%→78% | 59.45 |
| 10v10 | 12/20 | 9/20 | 89%→83.5% | 74.05 |

三项修正确实进入实战，但组合效果并非整体提升，尤其 10v10 表现下降；不要把 v5 当成已经证明胜率更高的版本。394 项技能测试、27 项适配器测试通过；所有对局的属性审计、回放、合法动作及推进时钟均已核对。报告、逐局结果和使用命令见 [v5 报告](analysis/skill_repairs_v5_20261023/REPORT.md)，机制触发与败局证据见 [独立轨迹审计](analysis/skill_repairs_v5_20261023/INDEPENDENT_AUDIT.md)。

v5 已知问题：10v10 第 17 局出现残血追猎者连续 23 步在危急撤退和重新接敌间交替、没有攻击分配的循环。v6 已补充跨角色保护，并用同种子重新实测；历史 v5 日志与源码快照继续保留。

## 已实现

1. **完整接线与可复核日志**：单位价值配置从 CLI 传给 `SkillManager`，在完整的己方共享可见阵容上每步只计算一次。`run.json`、`summary.json` 和事件保存模式、版本、有效参数；源码哈希包含 `unit_value.py` 与新接敌模块。
2. **去除排序混杂**：保持原来 `target_scoring` 开关对应的排序层级，单位价值只修改软评分。将四个消费参数 `focus_value_weight`、`focus_switch_margin`、`retreat_value_margin`、`heal_value_weight` 同时置零，可恢复无价值模块的动作。`ordered_before_value` 记录原无价值基线。
3. **分开敌我价值**：`unit-value-v2` 在可选空间情境中提供敌方 `target_priority_value` 和己方 `preservation_value`。前者考虑当前可见射程覆盖及巨像目标簇集潜力；后者考虑当前有效攻击机会、邻近支援及巨像可攻击目标的簇集。原 `normalized_importance` 保留，缺测证据标记 partial。簇集只是启发式代理，并不计算真实溅射伤害。
4. **协同接敌**：对没有合法攻击目标的单位，按预计接近窗口选择目标，保持仍然合适的目标；优先为近战分配接近方向，考虑友军占位和预测落点。若上一步移动实际位移不足，会重新考虑侧向通道。健康远程可在接敌阶段最多等待一窗，让近战跟上；已有合法攻击的单位继续走集火分配。
5. **撤退与重入**：比较站立和各合法移动。若即将能开火，避免撤出自己的攻击范围却没有减少敌方覆盖；确实减少覆盖和极危情况仍允许撤退。日志保留候选覆盖、预计重新进入射程所需窗口及被拒绝原因。

补修：检测连续接敌时 A→B→A 的位置复现，且接近同目标没有净进展时，最多两窗避免立即返回刚离开的位置。唯一合法移动不会被禁用；目标变化、中断接敌或记忆到期时释放限制。该规则用于打破已观察到的两点循环，不保证解决所有全局绕行问题。

导航历史跨四步技能窗口保留，在每局重置。所有新机制默认关闭，旧的三个增强开关仍独立。v5 三个开关从两个环境入口传给 `SkillManager`，有效配置写入运行元数据，执行日志保留 `damage_model`、`critical_escape_diagnostics`、`planned_reentry_diagnostics`。

## 运行

在 `C:\Users\ry\skill\smacv2_jev_adapter` 中执行：

```powershell
.\scripts\run_improved_skill.ps1 -Size 5 -Episodes 20 -Seed 150 -OutputName repeat-v6-5v5
.\scripts\run_improved_skill.ps1 -Size 10 -Episodes 20 -Seed 150 -OutputName repeat-v6-10v10
```

脚本默认 `-PolicyVersion v6 -Seed 150`，每种规模默认运行 20 局。优先复用已有清单并核对规模、局数、种子与 native 属性设置；没有清单才生成。v6 与既有 v4、v5 复用同一份 150–169 清单，结果写入新目录，不覆盖旧对局。启用以下参数：

```text
--method adaptive-skill
--adaptive-posture balanced --skill-max-steps 4
--adaptive-window-fire --adaptive-range-preserving-move --adaptive-target-scoring
--adaptive-coordinated-approach --adaptive-safe-reentry
--adaptive-accurate-damage --adaptive-critical-escape --adaptive-planned-reentry
--adaptive-engagement-commitment
--unit-value-mode contextual --unit-value-spatial-context
```

默认输出为 `runs/native-5v5-v6-skill-n20-seed150/` 和 `runs/native-10v10-v6-skill-n20-seed150/`。再次运行时用 `-OutputName replay-5v5-v6-seed150` 等新名称，避免覆盖已有结果。脚本拒绝覆盖任何既有对局目录。

两种规模均使用 `10gen_protoss` 的随机神族阵容/出生位置，评测种子 150–169。每局双方使用同一份抽样兵种组成，同兵种原生属性统一；出生位置按反射或包围分布生成。敌方为难度 7 内置 AI；固定步长 8 loops。观测为己方共享视野，不读取隐藏敌人信息。

### 复现 v4 / v5 配置

```powershell
.\scripts\run_improved_skill.ps1 -Size 5 -Episodes 20 -Seed 150 -PolicyVersion v4 -OutputName replay-5v5-v4-seed150
.\scripts\run_improved_skill.ps1 -Size 10 -Episodes 20 -Seed 150 -PolicyVersion v4 -OutputName replay-10v10-v4-seed150
.\scripts\run_improved_skill.ps1 -Size 5 -Episodes 20 -Seed 150 -PolicyVersion v5 -OutputName replay-5v5-v5-seed150
.\scripts\run_improved_skill.ps1 -Size 10 -Episodes 20 -Seed 150 -PolicyVersion v5 -OutputName replay-10v10-v5-seed150
```

`v4` 保留原 v4 全部开关与单位价值参数，不传 v5 三个开关或 v6 开关；`v5` 保留原 v5 参数，不传 v6 开关。在当前源码上复现旧配置时，应同时核对输出中的源码哈希和关闭开关的动作兼容检查；已有 v4/v5 的原始结果与冻结源码继续保留，不能仅凭相同种子宣称实现版本完全相同。

## 验证与消融

运行已有观测上的离线检查：

```powershell
.\.venv\Scripts\python.exe scripts\check_skill_improvements_offline.py
```

该脚本使用旧 20 局的 1,375 帧观测，分别检查旧版、零消费权重、仅价值、仅接敌、仅安全重入及全部开启。旧版和零权重均完整复现历史动作；全部配置通过输入不变、动作合法及有限 JSON 日志检查。**这是固定输入回放，不能把其差异解释为新胜率。**

真正的同种子对照可在同一 manifest 上另建输出目录，分别开启这些开关：

| 配置 | 单位价值 | 协同接敌 | 安全重入 | v5 三个开关 |
|---|---|---|---|---|
| 旧版基线 | off | 关 | 关 | 关 |
| 仅价值 | contextual + spatial-context | 关 | 关 | 关 |
| 仅接敌 | off | 开 | 关 | 关 |
| 仅重入 | off | 关 | 开 | 关 |
| v4 | contextual + spatial-context | 开 | 开 | 关 |
| v5 | contextual + spatial-context | 开 | 开 | 开 |

各组均保留原先三个增强开关。本轮 v5 已完成 5v5 与 10v10 各 20 局，以已有 v4 的同清单结果为配对基线；结果见本页开头和 v5 报告。单项消融的实战对局尚未进行，不能把本轮胜负变化单独归因到某项机制。其他种子的旧版结果只能作非配对参考。

对局审计与结果见 [REPORT.md](analysis/skill_improvements_20261023/REPORT.md)，机器可读结果见同目录 `verified_results.json`，测试输出为 `skill-tests.txt` 和 `adapter-tests.txt`。最终运行源文件保存在同目录的 `source_snapshot_final/`，原始轨迹和回放在 `runs/native-5v5-v4-skill-n20-seed150/` 与 `runs/native-10v10-v4-skill-n20-seed150/`。

以上历史 v4 冻结前验证：370 项技能测试、24 项适配器测试全部通过；不代表 v5 的最终验证计数。

首批 v3 的 130–149 种子用于暴露两点往返问题，保留于 `pilot_seed130/` 报告及原始 v3 对局目录。加入短周期检测后才运行最终 v4 批次；两个版本不混算胜率。

当前接敌仍是四方向短程估计，不是全局寻路；v5 同时参考敌方覆盖数量与基础齐射压力代理，没有预测敌方未来位置、意图或精确伤害交换。这些限制需通过后续同种子单项消融继续验证。
