# 医疗机直接决策树

本目录将医疗机决策写成可审查的阈值树：合法观察和少量显式记忆生成事实，树叶选择行为意图，确定性绑定器将意图转换为原生动作。现在还支持**配对回报学习**：冻结原动作树，实测一次候选干预相对原选择的回报差，用浅回归树学习何时采用候选。两条执行路径均不需要教师网络或动作 logits。

新流程与命令见 [配对回报学习](docs/RETURN_LEARNING.md)。普通 `tree_v0.json` 仍可直接使用；新策略文件把原树、候选条件、收益树封装在一起，通过同一个 `--medivac-tree` 参数加载。没有实测数据时不会生成或默认启用一个假训练模型。

最新实验（2026-11-04）：已完成“保留当前治疗／切换到合法新患者”的 40 个开发场景、80 局配对采集，用 31 条合格样本拟合出 **5 节点收益树**。随后在独立 20 个验证场景运行 40 局：原树和模型均 9 胜 11 负，模型实际改选 2 次，平均回报差 −0.00358，未观察到改善。模型作为实验版本保存，默认策略没有替换，最终测试集尚未使用。配置、模型规则、产物与完整结果见 [HEAL_SWITCH_TRAINING.md](docs/HEAL_SWITCH_TRAINING.md)。

2026-11-04 新增针对连续 STOP 的接近伤员候选，以及开发/验证/测试场景封存、串行采集和整局评估入口。流程命令见 [停滞恢复收益树：第一次训练](docs/FIRST_TRAINING.md)。正式首批 40 个开发场景已完成两臂共 80 局，两臂均为 13 胜 27 负；候选机会为 0，训练器因没有合格样本拒绝拟合，尚无模型。详见 [首批采集与拟合结果](docs/FIRST_TRAINING_RESULT_20261104.md)。验证与最终测试场景仍封存。

此前停滞恢复实现的 376 项检查通过，最终 12 局真实冒烟跑通采集和审计，得到一条负收益样本；该冒烟没有训练模型，也未证明胜率提升。详见 [停滞恢复验证记录](docs/IDLE_VALIDATION.md)。

`tree_v0.json` 是**手写的简化初始策略**，不保证与原医疗规则等价。后续可以根据开发集轨迹改特征、分支、阈值与行为绑定，并用冻结版本的配对对局决定是否保留；这些改动不能靠更换树名称冒充训练或迁移成功。当前 n4 实验用于检查接口、实际执行及回退/冲突情况，不能证明胜率提升。

初始版本是**一棵共享的 17 节点树，依次为每台医疗机决策**，不是每台医疗机分别学习一棵树。它依次检查存活、自保撤退、紧急抢救、保持治疗、继续治疗当前目标、选择新伤员、接近伤员、支援移动，最后停止；具体目标和动作必须通过合法绑定。

## 模块与控制边界

- `facts.py`：从允许的观察和动作可用性提取事实。
- `bindings.py`：合法候选筛选和公开的词典序行为/目标绑定。
- `memory.py`：显式订单、冷却和接近进展记忆及跨局/跨版本重置。
- `validate.py`：树 schema、节点和特征白名单校验。
- `tree_v0.json`：版本化初始树。
- `runtime.py`：沿路径解释树、输出动作与决策追踪。
- `integration.py`：连接既有技能管理器；由适配器记录实际提交动作。
- `scripts/prepare_smoke.py`：不启动 SC2，冻结四个场景及实验协议。
- `scripts/run-smoke-arm.ps1`：按协议启动一臂实验；最多一个 SC2 实例，不覆盖旧输出。
- `scripts/analyze_smoke.py`：只读运行日志，核查初态配对、树提案、最终提交和引擎反馈。
- `scripts/freeze_sources.py`：按运行日志中的 hash 验证并保存源码、地图和树原始 JSON；源文件已变化时拒绝伪造历史快照。

运行模式为 `--medivac-policy legacy|tree-shadow|tree`。`legacy` 使用原医疗控制器；`tree-shadow` 记录树建议，最终仍执行原规则；`tree` 以直接树负责医疗机。运行时默认读取 `tree_v0.json`，可用 `--medivac-tree` 指定其他文件；本目录的冒烟启动脚本要求显式树路径，以便冻结实验输入。

本阶段接入范围仅为 `--skill-version current`、人族 `--race-profile terran` 和 `--method adaptive-skill`；没有完成其他技能版本、其他种族或其他控制入口的接入。

主要比较使用 `--medical-patient-coordination off`：`legacy_off` 与 `tree` 保持相同的地面部队和患者协调设置，分别使用旧医疗决策与新树。`legacy_on` 保留原完整患者协调作为额外参考；不能将它与主比较混为单变量消融。这里关闭的是患者协调通道，具体范围以接入元数据和源码为准，并不改变环境物理规则。旧 `--pair-mode` 与 `--native-teacher-control` 均为 `off`。

## 本机命令

在 PowerShell 中切到 `C:\Users\ry\skill\SMAC_agent`。现有环境解释器为 `C:\Users\ry\skill\SMAC_agent\sc2_skill_adapter\.venv\Scripts\python.exe`。以下命令使用新输出目录；已有目录会拒绝覆盖。

```powershell
Set-Location 'C:\Users\ry\skill\SMAC_agent'
$treePython = Join-Path $PWD 'sc2_skill_adapter\.venv\Scripts\python.exe'
& $treePython -m medivac_tree.validate medivac_tree/tree_v0.json
& $treePython -m unittest discover -s medivac_tree/tests -t . -v

# 只准备场景，不启动游戏；首次执行时使用全新目录。
& $treePython medivac_tree/scripts/prepare_smoke.py `
  --output-dir medivac_tree/artifacts/smoke_new_n4 --seed 21000

# 串行运行主对照；前一臂正常结束后才启动后一臂。
.\medivac_tree\scripts\run-smoke-arm.ps1 -Arm legacy_off `
  -StudyDir medivac_tree/artifacts/smoke_new_n4 -OutputDir runs/medivac-smoke-new-legacy-off
.\medivac_tree\scripts\run-smoke-arm.ps1 -Arm tree `
  -StudyDir medivac_tree/artifacts/smoke_new_n4 -OutputDir runs/medivac-smoke-new-tree `
  -Tree medivac_tree/tree_v0.json

# 可选：完整原规则参考，或验证shadow不改变最终动作。
.\medivac_tree\scripts\run-smoke-arm.ps1 -Arm legacy_on `
  -StudyDir medivac_tree/artifacts/smoke_new_n4 -OutputDir runs/medivac-smoke-new-legacy-on
.\medivac_tree\scripts\run-smoke-arm.ps1 -Arm shadow `
  -StudyDir medivac_tree/artifacts/smoke_new_n4 -OutputDir runs/medivac-smoke-new-shadow `
  -Tree medivac_tree/tree_v0.json

& $treePython medivac_tree/scripts/analyze_smoke.py `
  --run legacy_off=sc2_skill_adapter/runs/medivac-smoke-new-legacy-off `
  --run tree=sc2_skill_adapter/runs/medivac-smoke-new-tree `
  --output-dir medivac_tree/artifacts/smoke_new_n4/analysis
```

分析额外参考时追加 `--run legacy_on=...` 或 `--run shadow=...`，并选择新的分析输出目录。`run-smoke-arm.ps1` 的相对 `OutputDir` 基于 `sc2_skill_adapter`；其他示例路径基于 `SMAC_agent`。

运行后及时保存源码证据：

```powershell
& $treePython medivac_tree/scripts/freeze_sources.py `
  --run sc2_skill_adapter/runs/medivac-smoke-new-tree `
  --output-dir medivac_tree/artifacts/smoke_new_n4/source_snapshots/tree
```

首次已经准备的清单为 `artifacts/smoke_20261101_n4/`，包含 seed 21005（无医疗机）、21001（单医疗机）、21000/21003（各两台医疗机）。准备时从固定池按组成选择，未依据结果筛选，并排除了当时 `sc2_skill_adapter/runs/*/manifest.json` 已出现的场景 hash。再次生成相同 seed 时，若这些场景已运行，可能会选择池中后续未使用的场景；**复现原实验应直接复用已冻结清单**。

## 输出与判读

场景目录包含 `manifest.json`、`protocol.json`、`sampling_pool.json`。主对照共 8 局，加完整原规则参考为 12 局；shadow 是额外 4 局。每臂 4 局，单实例，单运行上限 40 局。四局按组成富集，不能将其混合胜率称为原始分布胜率。

每个运行目录沿用 `run.json`、`manifest.json`、`events.jsonl`、`episodes.json`、`summary.json` 和 `replays/`。`medivac_tree_step` 按 `episode/game_loop` 关联同一步实际 `step.actions` 和 SC2 feedback；`decision_id` 含 run 标识，不能只凭 `ep0001` 跨运行拼接。

分析产物：

- `analysis.json`：逐臂状态、逐局胜负与有效性、叶子/意图覆盖、提案与提交差异、患者预留冲突、引擎反馈、初始状态 hash 配对和首次动作分歧。
- `episodes.csv`：适合逐局查看的扁平表；未完成而已启动的局记为 `invalid`，未启动记为 `not_started`，两者均不记成输局。

分析器不会把 shadow 建议记成实际树控制。`executed=true` 仅表示树动作应用到最终提交列表，实际是否获引擎接受仍以随后的 `step.feedback` 为准；不能把此字段解释为成功治疗。已应用树提案若被修改，必须有明确覆盖原因。缺失记录、初态不配对、引擎回执无法精确关联等会使审计失败。无医疗机对照的动作序列应保持相同；shadow 应与 `legacy_off` 的动作序列相同。原始引擎 `Success` 仅表示命令接受，不证明已经产生治疗；`AlreadyTargeted` 也不等同于治疗无效。

这里没有实现可恢复的 SC2 中途快照或同内存分叉。比较采用相同冻结场景与 seed，从新游戏进程开始运行；离线重放观察可检查规则，但不能推算新策略的真实胜负。

实验记录见 [docs/VALIDATION.md](docs/VALIDATION.md)，迭代约定见 [docs/ITERATION_PROTOCOL.md](docs/ITERATION_PROTOCOL.md)。

2026-11-01 首轮验证：289 项测试通过，四臂共 16 局真实对局完成；主对照 legacy_off 为 3/4 胜，tree 为 2/4 胜。254 条实际树决策与最终提交一致，shadow 四局动作与基线完全相同。接线冒烟通过，初始树在这批小样本中没有优于基线；完整证据和未覆盖分支见验证记录。
