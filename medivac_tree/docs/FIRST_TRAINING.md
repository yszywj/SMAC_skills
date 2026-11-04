# 停滞恢复收益树：第一次训练

**执行状态（2026-11-04）：本页的首批采集已运行，40 个开发场景、两臂共 80 局全部完成。候选机会与合格训练样本均为 0，实际拟合被拒绝，未生成模型。** 详见 [首批结果](FIRST_TRAINING_RESULT_20261104.md)。下方命令保留为执行流程记录，原采集输出目录已存在；复现实验必须另取新输出目录。当前这批空数据不能进入模型验证，下一轮应先解决候选覆盖问题。

本轮学习一个具体问题：**医疗机连续停止、仍有能量时，继续停止好，还是向一个未被其他医疗机占用的伤员移动一步更好？**

手写 `tree_v0` 继续提供 BASE。新增 `idle_recovery` 生成器提供候选，收益树学习是否采用；不使用教师网络。候选选中后提交一个原生移动命令，随后继续使用原树；不假定这个命令已经到达患者或产生治疗。

## 已实现的触发条件

默认条件已冻结到准备好的策略 JSON：

- 原树本次选择 STOP，之前两个连续决策也确实提交了 STOP；本次是第三个窗口。
- 当前没有治疗订单，能量至少 5，自身血量比例至少 0.6。
- 存在距离不超过 12 的、存活受伤的地面队友，而且没有被其他医疗机预留，也不是其他有能量医疗机当前可合法直接治疗的伤员。
- 其他活着医疗机的能量或合法治疗信息缺失时，不生成接近候选，避免误抢患者。
- 存在合法移动方向，端点相对患者的距离至少缩短 0.1。
- 移动端点比当前位置最多多暴露于 1 个可见威胁的射程中。

这些数值是明确的开发假设，不是训练所得，也不保证路径全程安全。允许有限暴露增长，是为了检验旧规则“任何暴露增加都拒绝”是否过于保守。需要同时保留负收益样本来学习何时不应移动。

只有出现合法、不同动作的候选时才消耗本局唯一一次干预机会。缺伤员、患者被占用、低能量、低血量、无合法方向或风险超限等原因都写入日志。候选生成不改变原树绑定，只在选择后提交记忆和患者预留。

特征版本已升级为 **2，共 46 列**，增加停止窗口、未占用伤员数量/距离、当前位置暴露、候选端点距离/进展/暴露变化。旧版本的 3 条换目标样本和模型不能混入这轮训练。

## 已准备好的目录

`medivac_tree/artifacts/idle_curriculum_20261104_v1/` 已生成并封存：

| 目录 | 场景数 | 用途 |
|---|---:|---|
| `development/study_v2` | 40 | 首轮采集与训练；训练器在合格数据内按场景再留出 25% 做离线验证 |
| `validation/study_v2` | 20 | 模型冻结后的真实对局验证 |
| `test/study_v2` | 20 | 方法最后确定后才运行的最终测试 |

使用表中的 `study_v2`；同目录下的旧 `study` 保留了补充医疗机协调约束之前的协议，已不适用于当前源码。场景清单没有重新选择。正式开发对局已经完成，验证与最终测试场景仍未使用。

场景从固定 seed 的池中按顺序选择，只看初始阵容是否有医疗机和场景 hash，排除既往运行中的场景并去重，不看胜负或候选触发率。三个集合的场景组不交叉。这是医疗机阵容筛选后的分布，不代表未筛选的全体场景。

**40 个开发场景不等于 40 条训练样本。** 没有出现候选机会或没通过配对审计的场景，保留整体对局结果，但不给它编造收益标签。

## 1. 采集开发数据

在 PowerShell 中执行。以下实验与模型输出目录必须尚不存在；已准备好的 study 目录直接复用。

```powershell
Set-Location 'C:\Users\ry\skill\SMAC_agent'
$treePython = '.\sc2_skill_adapter\.venv\Scripts\python.exe'
$studyRoot = 'medivac_tree/artifacts/idle_curriculum_20261104_v1'
$runRoot = 'sc2_skill_adapter/runs/idle-first-dev-v1'

& $treePython medivac_tree/scripts/first_training.py collect `
  --study-dir "$studyRoot/development/study_v2" `
  --output-dir $runRoot
if ($LASTEXITCODE -ne 0) { throw '采集未完成，请先查看运行错误；不要开始训练。' }
```

该命令按顺序运行 BASE 40 局和 CANDIDATE 40 局，共 **80 局**，每次只启动一个 SC2 实例。可先加 `--dry-run` 核查命令而不启动游戏。若需要本批额外的重跑噪声检查，加 `--repeat-base`，会再运行 40 局 BASE。

完成后自动生成：

- `paired.json`：通过观察前缀、动作和反馈审计的收益样本，及所有拒收原因。
- `comparison.json`：全部合格完成对局的胜率、回报差、候选覆盖和未触发原因；无机会局及真实命令拒绝不会从整体表现中删除。
- `workflow_result.json`：接受样本数、独立场景数、下一步提示。
- 各臂的完整日志、回放和源码快照；可选 `noise.json`。

SC2 是独立重跑并核对可见前缀，仍没有实现可恢复的中途内部快照。零机会不能被标为零收益。采集步骤不会自动训练或替换运行策略。

## 2. 检查样本，再训练

先看 `workflow_result.json` 与 `comparison.json`。默认至少要 **6 个通过审计的独立场景** 才能划分训练/离线验证；每个训练叶子仍至少覆盖 4 个场景，形成有效分支通常需要更多。

```powershell
& $treePython -m medivac_tree.learning.train `
  --dataset "$runRoot/paired.json" `
  --policy "$studyRoot/development/study_v2/base.json" `
  --output-dir medivac_tree/artifacts/idle_model_v1
if ($LASTEXITCODE -ne 0) { throw '模型未生成，请按训练错误补充数据或修正输入。' }
```

训练器按实际 `candidate_return - base_return` 拟合浅回归树，最大深度默认 2。同一场景不同 seed 属于同一组，不会一半进训练、一半进验证；重复输入同一场景/seed 会被拒绝。

输出：

- `policy.json`：可直接执行的原树＋收益选择器。
- `model.json`：条件节点与叶子预测收益。
- `report.json`：始终 BASE、始终候选、收益树选择的离线比较，包含可用时的胜负指标。

如果所有数据均为同一种收益符号，训练结果可能是一片常量叶子；不要人为强迫它分裂。训练报告的 `trained_not_game_validated` 表示模型已拟合，尚未完成新的实际对局验证。

## 3. 在封存验证集上实际对局

模型训练成功后才运行：

```powershell
& $treePython medivac_tree/scripts/first_training.py evaluate `
  --study-dir "$studyRoot/validation/study_v2" `
  --policy medivac_tree/artifacts/idle_model_v1/policy.json `
  --output-dir sc2_skill_adapter/runs/idle-first-validation-v1
if ($LASTEXITCODE -ne 0) { throw '验证未完成，请查看运行错误。' }
```

它运行 BASE 20 局和训练选择器 20 局，共 40 局，再输出整体比较。不会给 MODEL 运行生成候选训练标签，也不会边测试边改树。入口会拒绝评估场景与模型已使用训练/离线验证场景重叠。

先看胜负和整体回报，再看候选触发率、动作接受以及哪些状态发生改选；不能仅凭训练误差下降或少数样本 reward 增加宣布胜率改善。原 `legacy_off` 仍应作为后续外部基线；当前收益实验的 BASE 始终是手写初始树。

如果验证结果用于修改方案，该验证集就变成开发材料。`test/study_v2` 继续封存，待最后策略冻结后再用同样的 evaluate 命令做最终测试。

## 样本不足时怎么补

保持候选条件、特征、运行时和环境配置不变，用另一个预先声明的 seed 准备新开发批；必须排除已有课程目录，避免撞入封存验证/测试组。例如：

```powershell
& $treePython -m medivac_tree.scripts.prepare_curriculum `
  --seed 32000 --pool-size 512 `
  --exclude-run-root $studyRoot `
  --output-dir medivac_tree/artifacts/idle_curriculum_dev2

& $treePython medivac_tree/scripts/return_study.py prepare `
  --manifest medivac_tree/artifacts/idle_curriculum_dev2/development/manifest.json `
  --profile idle-recovery `
  --output-dir medivac_tree/artifacts/idle_curriculum_dev2/development/study
```

随后按第 1 步采集新开发批，训练时重复传入 `--dataset` 合并批次，使用新的模型输出目录。不要混入本次用于发现问题的四局冒烟数据，也不要为了让训练通过而降低独立场景覆盖要求。

如果长期没有候选机会，应根据 `candidate_audit` 原因检查假设与生成器。若修改触发阈值、候选行为或特征，必须冻结新协议并重新采集；旧协议数据不能直接混训。准备好的 study 发现源码变化会拒绝继续运行，以免两臂实际使用不同策略。

本轮实现与真实冒烟结果见 [IDLE_VALIDATION.md](IDLE_VALIDATION.md)。最终冒烟三臂各 4 局，均为 2 胜 2 负；只有一条合格的负收益样本，不足以训练，也未证明候选有效。
