# 配对回报学习首版验证（2026-11-03）

实现了单候选、每局一次干预的 BASE/CANDIDATE 数据采集和审计、浅收益回归树训练、模型导出与同协议部署入口。原 `tree_v0.json` 没有修改；未新增全队技能总树，也未使用教师。

## 自动化验证

共 **335 项通过**：医疗树 87 项、适配器 65 项、当前技能 183 项。新增覆盖候选不污染记忆、患者预留、多医疗机与跨局门控、完整 BASE 保留、收益正负/平局、模型协议与环境漂移、匹配前缀、真实命令反馈、同场景分组及训练导出。

解释器：`sc2_skill_adapter/.venv/Scripts/python.exe`。

- 医疗树：仓库根目录 `python -m unittest discover -s medivac_tree/tests -t . -q`。
- 适配器：`sc2_skill_adapter` 目录 `python -m unittest discover -s tests -q`。
- 当前技能：按 `skills.current.tests.<文件名>` 的完整模块名加载所有 `test_*.py`，避免该历史测试目录的相对导入问题。

## 真实对局冒烟

复用已公开结果的四局开发清单 `artifacts/smoke_20261101_n4/manifest.json`；没有把它重新当成未见测试集。SC2 报告版本 `5.0.17.98310`，三臂各 4 局，共 12 局完成，单实例串行运行。

候选协议：原树选择 `HEAL_CURRENT` 或 `KEEP_HEAL` 时，在全局首次存在不同合法动作的机会尝试 `HEAL_NEW`，之后继续用原树。三臂分别为 BASE、CANDIDATE、BASE 重复。

| 场景序号 | BASE 回报 | CANDIDATE 回报 | 回报差 | 数据处理 |
|---|---:|---:|---:|---|
| 1 | 11.676768 | 11.676768 | 0.000000 | 无医疗机机会，拒收，不标成零收益样本 |
| 2 | 18.272141 | 19.395678 | +1.123537 | 前缀与执行审计通过 |
| 3 | 15.301414 | 15.515923 | +0.214508 | 前缀与执行审计通过 |
| 4 | 9.119914 | 10.653121 | +1.533207 | 前缀与执行审计通过 |

三臂均为 2 胜 2 负。BASE 重复的全部四局，观察/提交动作/反馈序列与第一遍一致、长度一致、回报差为零。这只是本批重复性证据，不保证所有 SC2 重跑完全确定。

三个干预机会的 BASE 均为重发已有患者的治疗命令，引擎返回 `AlreadyTargeted`。审计核对了订单、ability、目标和 tag，将其记为原订单保持；CANDIDATE 的三个换目标命令均返回 `Success`。没有把命令接受当成治疗效果。

第一次严格审计将三条 `AlreadyTargeted` 基线都拒收，原结果保存在 `paired.json`；修正该具体回执语义并通过回归测试后，重新生成 `paired_audited.json`，没有覆盖旧记录。

## 证据位置

- [冻结协议](../artifacts/return_smoke_20261103_v2/protocol.json)
- [通过审计的配对数据](../artifacts/return_smoke_20261103_v2/paired_audited.json)
- [BASE/BASE 重复检查](../artifacts/return_smoke_20261103_v2/noise.json)
- [BASE 运行](../../sc2_skill_adapter/runs/return-smoke-20261103-v2-base-retry/summary.json)
- [CANDIDATE 运行](../../sc2_skill_adapter/runs/return-smoke-20261103-v2-candidate/summary.json)
- [BASE 重复运行](../../sc2_skill_adapter/runs/return-smoke-20261103-v2-base-repeat/summary.json)

每个已完成运行还保存 `source_snapshot/`、完整事件、逐局结果与回放。最初沙箱内 SC2 启动失败的目录保留其失败记录，未计入 12 局、胜率或收益样本；随后在沙箱外正常完成。

## 当前结论

执行、采集、审计和离线训练接口已接通。真实数据仅有 3 个合格独立场景，训练器按默认分组/叶子覆盖规则正确拒绝训练，没有生成生产模型、降低样本要求或默认更换原树。训练器的拟合与导出行为通过合成测试验证，未把测试数据当成真实游戏经验。

这批正回报差不构成稳定提升或泛化证明；它们来自已知开发场景，胜负未改变，样本也不足。下一轮需要在冻结协议下积累更多开发配对，保留遇到的正、零、负结果，并在独立场景验证训练后的选择策略。
