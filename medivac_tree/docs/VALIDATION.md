# 初始版本验证记录

## 已准备的场景

- 清单：`../artifacts/smoke_20261101_n4/manifest.json`。
- SHA256：`5062e594cf5391b989698be70c8d114544ddb42221321bda2d29b08c59d25432`。
- 场景顺序：21005（无医疗机）、21001（单医疗机）、21000（两台）、21003（两台）。
- 从 seed21000 的 256 场固定生成池按组成依次选择，未访问胜负；排除准备时已记录的场景 hash。
- 主比较：`legacy_off` 与 `tree`，两组患者协调均关闭；`legacy_on` 是原完整策略的额外参考。

## 工具检查

`prepare_smoke.py` 已在现有适配器 Python 环境生成并校验清单；没有通过该脚本启动 SC2。

`analyze_smoke.py` 的合成日志检查确认：四对初态可正确配对；树决策计数正确；提案记录的最终动作与真实 step 不符会报错；已启动但缺失完成事件的局记为 invalid。合成数据仅验证分析器，不是游戏成绩。

## 自动检查与真实对局结果（2026-11-01）

新模块 41 项、已有技能 183 项、适配器 65 项测试通过，共 **289 项**。真实 SC2 四臂各完成 4 局，共 **16 局**，四臂均无运行中无效局或未启动局。机器分析位于 [analysis_final/analysis.json](../artifacts/smoke_20261101_n4/analysis_final/analysis.json)，逐局表位于 [episodes.csv](../artifacts/smoke_20261101_n4/analysis_final/episodes.csv)。

第一次沙箱内 `legacy_off/` 启动失败，完成 0 局；失败目录保留，随后在允许启动本机 SC2 的环境中使用全新 `legacy_off_native/` 目录运行。失败启动不记成游戏败局，也没有覆盖原目录。

| 策略 | 患者协调 | 完成 | 胜局 | 决策步数 | 运行耗时 |
|---|---|---:|---:|---|---:|
| legacy_off | off | 4/4 | 3 | 33/42/52/112 | 48.58 秒 |
| tree | off | 4/4 | 2 | 33/46/40/64 | 44.89 秒 |
| legacy_on（额外参考） | on | 4/4 | 3 | 33/43/52/91 | 45.13 秒 |
| shadow | off | 4/4 | 3 | 33/42/52/112 | 47.99 秒 |

| seed / 医疗机数量 | legacy_off | tree | legacy_on | shadow |
|---|---|---|---|---|
| 21005 / 0 | 负 | 负 | 负 | 负 |
| 21001 / 1 | 胜 | 胜 | 胜 | 胜 |
| 21000 / 2 | 胜 | 胜 | 胜 | 胜 |
| 21003 / 2 | 胜 | 负 | 胜 | 胜 |

**初始树已能完成真实对局，但本批没有优于基线。** 主比较少赢一局；这四个组成筛选场景仅用于冒烟，不能推断总体胜率差，也没有迁移检验。没有依据本次结果修改初始树后重报同一批成绩。

### 接线与动作审计

- 四臂每局的场景 hash、初始状态 hash 全部匹配；主对照共同运行源码 hash 一致。
- 无医疗机对照四臂完整动作序列一致。
- shadow 的四局完整动作序列均与 legacy_off 一致。共记录 370 条树提案，259 条恰与基线动作相同、111 条不同，**应用到最终提交的树决策为 0**。
- tree 的 **254 条**医疗机决策全部与最终提交动作一致，0 个未解释覆盖，0 个树患者预留冲突；预留所有者和目标占用回执检查通过。
- tree 中 `executed=true` 仅指应用到最终提交列表，不能解释为获引擎接受或实际治疗成功。
- 各臂医疗机的引擎反馈映射精确，观察错误为 0。

### 树叶覆盖与引擎反馈

| 实际 tree 意图 | 次数 |
|---|---:|
| STOP | 23 |
| HEAL_NEW | 21 |
| HEAL_CURRENT | 135 |
| HEAL_RESCUE | 5 |
| SUPPORT | 17 |
| APPROACH | 27 |
| KEEP_HEAL | 26 |

这批真实对局没有覆盖 RETREAT 和死亡 NOOP 叶子，不能据此声称所有分支都已实战验证。

| 医疗机提交反馈 | legacy_off | tree | legacy_on | shadow |
|---|---:|---:|---:|---:|
| Success | 168 | 93 | 127 | 168 |
| AlreadyTargeted | 150 | 135 | 153 | 150 |
| retained_order（未发新命令） | 51 | 26 | 49 | 51 |
| NotEnoughEnergy | 1 | 0 | 0 | 1 |

Success 是命令接受，AlreadyTargeted 是引擎对重复目标新请求的反馈，retained_order 是主动保持已有订单；这些数字不能直接相加成为成功治疗次数，也不能当独立样本评价收益。

### 失败轨迹的有限诊断

seed21003 在 loop34 首次出现动作分歧，两臂此前动作相同、当前完整观察 hash 相同。只有医疗机 u8 从基线动作 4 改为树动作 3；树叶为 APPROACH，绑定伤员 u4。此后整条轨迹分叉，因此不能把整局胜负归因于这一单步。

该树轨迹后期，u8 在 loop346–506 连续 21 个决策步选择 STOP，能量由 33.78 增至 39.40，没有耗尽；当时无可治疗候选、没有可绑定 APPROACH，SUPPORT 绑定也选择停止。它提示支援/接近条件可能在该状态下过于保守，**只是后续可检验的假设**，尚未做单项干预来确认因果作用。原始证据及日志 hash 见 [episode4_case.json](../artifacts/smoke_20261101_n4/analysis_primary/episode4_case.json)。初始版本保持冻结。

### 源码和候选封存

`source_snapshots/legacy_off_native`、`source_snapshots/legacy_on` 各保留 41 个运行源文件；`source_snapshots/tree`、`source_snapshots/shadow` 各保留 49 个，额外包含直接树模块和原始 `tree_v0.json`。每个文件先按对应 `run.json.source_hashes` 验证，再保存被验证的字节；没有 hash 差异。快照还包含 run/manifest 和 `snapshot.json` 收据。

实际树为 `medivac-direct-v0`，17 节点；规范化树 hash 为 `3461a9fe9525db056aec012821f614d0209f3dcdae5af9a07a11eb62bfe3cd4f`。原始 JSON 的文件 hash 另在 source_hashes 中记录。上述快照是源码证据，不是可恢复的 SC2 游戏进程。
