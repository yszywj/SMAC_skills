# 医疗机决策树的配对回报学习

本轮实现周报方法的第一个可执行闭环：**冻结原树 → 声明一个候选 → 从相同场景重跑两臂 → 校验干预前的观察与动作 → 计算回报差 → 拟合浅收益树 → 接回执行器**。原动作树负责提供参考选择，收益树决定是否替换本次选择。

2026-11-04 更新：增加“连续 STOP → 有限暴露的一步接近”候选生成器，收益特征升级为版本 2、46 列。首次训练的已准备目录、采集与验证命令见 [FIRST_TRAINING.md](FIRST_TRAINING.md)。以下保留最初换目标实验的通用流程；旧特征版本的数据不可混入新训练。

这里没有增加全队技能总树。`medivac_tree` 已负责医疗机的治疗、保持命令、接近、撤退、支援、停止等局部行为；地面部队继续使用冻结的控制器。原树和原绑定候选继续使用 `bindings.py` 的确定性规则；新增停滞候选由独立生成器提出，不更改原树的绑定。收益树选择接受还是拒绝一个已绑定的候选，并不任意选择所有患者。

## 一次决策的含义

1. 原树得到 `BASE`：完整的意图、动作和目标。`BASE` 可能是撤退或接近，不等于 `KEEP_HEAL`。
2. 候选必须符合声明的原意图条件、时间条件，存在合法绑定，而且原生动作必须不同。
3. **每局只使用第一个满足条件的机会，所有医疗机合计一次**。BASE 臂在这里仍用原选择；CANDIDATE 臂在这里换用候选。两臂都消耗这个机会。
4. 从下一个决策继续使用同一原树和控制算法，但读取各自实际执行后的状态与记忆。不会强行重放相同的后续动作序列。
5. 训练后的 MODEL 在同一机会预测 `gain`；只有 `gain > 0` 才采用候选，否则保留 BASE。即便保留 BASE，也不会在后续机会再次尝试。

候选提案不写记忆、不占用患者；只有被选中的动作进入现有记忆与患者预留提交逻辑。模型版本或运行时源码变化会隔离缓存与策略记忆。

收益标签为 `candidate_return - base_return`，使用适配器已经记录的整局环境 reward 之和。由于干预前缀必须一致，这也对应已观测到的共同前缀之后的回报差。它不是教师分数、治疗量、命令 `Success`，也不是逐个中间节点的贡献。正常完成的正收益、负收益、零收益和超时结果都会保留；失败或缺失对局不补零。

## 可比较条件与限制

审计器要求同一场景清单和 seed、初态 hash、SC2 版本、环境参数、地面策略、运行时源码及动作协议。它进一步比较干预前全部观察、提交动作、反馈，以及同一医疗机在同一 loop 的候选、特征、记忆与患者预留指纹，并核查干预动作确实提交、反馈可关联。

不满足条件的记录写入 `rejected`，不会成为标签。没有候选机会、候选等同原动作、影子模式、动作被覆盖、反馈失败、前缀不匹配、重复干预等都不会被误记成零收益。首版还保守要求干预当步只有该医疗机的提交动作不同；如果患者预留连带改变了另一台医疗机本步的动作，该对局会被拒收。这会减少多医疗机样本，属于当前实验范围的限制。

一种需单独解释的原树反馈是 `AlreadyTargeted`：仅限 BASE 的 `HEAL_CURRENT`，且观察中的原治疗订单、提交命令的 ability、患者与 tag 全部相同，审计才接受它为“已有订单保持”。原始 `rejected/206` 仍保留，不改写成 `Success`，也不宣称发生了治疗。候选改选返回同样结果则仍拒收。

SC2 仍未提供可恢复的中途内部快照。这些数据标为 `matched_observed_prefix`，表示独立重跑的可见前缀匹配，不能证明隐藏状态与随机数发生器完全相同。BASE/BASE 重复臂用于检查重跑噪声，**不生成训练标签**；若重复原策略也出现明显回报波动，应先调查或增加重复实验，再解释候选效果。

跨批次也应固定同一 SC2 安装版本。本轮核对配对对局的实际游戏版本，并冻结运行参数与 Python 代码；没有对 SC2 游戏可执行文件做二进制哈希。

## 文件职责

| 文件 | 用途 |
|---|---|
| `candidates.py` | 原绑定候选或停滞恢复候选、46 个动作前可观测特征 |
| `idle_recovery.py` | STOP 窗口、独立接近候选及未触发原因；只读旧轨迹诊断 |
| `selector.py` | 策略与模型格式校验、收益树解释执行、源码契约 |
| `runtime.py` | 原树提案 → 候选选择 → 仅提交选中结果，每局一次门控 |
| `experiment.py` | 地面控制器、环境、依赖和源码的实验上下文校验 |
| `learning/dataset.py` | 读取真实运行日志，生成一局最多一行收益数据与拒收原因 |
| `learning/train.py` | 按独立场景分组训练、验证和导出浅回归树 |
| `scripts/return_study.py` | 冻结 BASE/CANDIDATE 策略与协议，单实例启动实验 |
| `scripts/prepare_curriculum.py` | 按初始阵容与场景 hash 预先划分开发/验证/测试 |
| `scripts/first_training.py` | 串行采集和模型对局评估，不自动拟合或部署 |
| `learning/report.py` | 包括无机会局在内的整局胜率、回报和候选覆盖比较 |

## 准备和运行

以下命令在 `C:\Users\ry\skill\SMAC_agent` 下执行。所有输出目录使用新名字，实验与模型文件放在工作区内。`prepare` 和 `--dry-run` 不启动 SC2。

```powershell
$treePython = '.\sc2_skill_adapter\.venv\Scripts\python.exe'

# 演示候选：原树准备继续治疗当前患者或保持命令时，尝试另一个合法患者。
# 这是待验证的实验假设，并不表示换目标已被证明更好。
# manifest 可替换为新的开发集；以前看过结果的场景不能再充当最终测试集。
& $treePython medivac_tree/scripts/return_study.py prepare `
  --manifest medivac_tree/artifacts/smoke_20261101_n4/manifest.json `
  --candidate-intent HEAL_NEW `
  --base-intent HEAL_CURRENT --base-intent KEEP_HEAL `
  --output-dir medivac_tree/artifacts/return_study_dev_v1

& $treePython medivac_tree/scripts/return_study.py run `
  --study-dir medivac_tree/artifacts/return_study_dev_v1 --arm base `
  --output-dir sc2_skill_adapter/runs/return-dev-v1-base --dry-run

# 去掉 --dry-run 才运行。各臂必须串行结束后再开始下一臂。
& $treePython medivac_tree/scripts/return_study.py run `
  --study-dir medivac_tree/artifacts/return_study_dev_v1 --arm base `
  --output-dir sc2_skill_adapter/runs/return-dev-v1-base
& $treePython medivac_tree/scripts/return_study.py run `
  --study-dir medivac_tree/artifacts/return_study_dev_v1 --arm candidate `
  --output-dir sc2_skill_adapter/runs/return-dev-v1-candidate
& $treePython medivac_tree/scripts/return_study.py run `
  --study-dir medivac_tree/artifacts/return_study_dev_v1 --arm base-repeat `
  --output-dir sc2_skill_adapter/runs/return-dev-v1-base-repeat
```

每臂最多 40 局，启动器拒绝覆盖旧目录、并发 SC2 或已改动的冻结源文件。正常完成后保存运行时源码和策略快照。进程被强制结束可能留下 `sc2_skill_adapter/.medivac-return-study.lock`；确认没有对应运行后才能清理该锁。

这里的主对照是 `tree_v0` 对“`tree_v0` 加一次候选”。原 `legacy_off` 仍可作外部基线，但不能混入这个收益标签。

## 生成数据与训练

```powershell
& $treePython -m medivac_tree.learning.dataset `
  --base-run sc2_skill_adapter/runs/return-dev-v1-base `
  --candidate-run sc2_skill_adapter/runs/return-dev-v1-candidate `
  --output medivac_tree/artifacts/return_study_dev_v1/paired.json

& $treePython -m medivac_tree.learning.dataset `
  --base-run sc2_skill_adapter/runs/return-dev-v1-base `
  --candidate-run sc2_skill_adapter/runs/return-dev-v1-base-repeat `
  --baseline-noise `
  --output medivac_tree/artifacts/return_study_dev_v1/noise.json

# 积累足够独立场景后再执行；n4 冒烟不能满足默认训练/验证要求。
# 多批相同协议的数据可重复传入 --dataset。
& $treePython -m medivac_tree.learning.train `
  --dataset medivac_tree/artifacts/return_study_dev_v1/paired.json `
  --policy medivac_tree/artifacts/return_study_dev_v1/base.json `
  --output-dir medivac_tree/artifacts/return_model_v1
```

训练器不需要额外机器学习库。它用平方误差选择分裂特征与阈值，叶子输出候选相对 BASE 的平均收益。默认最大深度 2、每叶至少 4 行且覆盖 4 个独立场景；同一场景的多个 seed 不能算成多个场景，每个场景训练总权重相同。负收益和零收益参与拟合。

默认按完整场景 hash 划分 25% 验证组，同一场景所有 seed 都在同侧；也可用 `--validation-dataset` 指定组不交叉的验证文件。默认至少需要 6 个**通过审计且有机会**的独立场景才能分组训练，形成分支通常需要更多。不会为了让小样本跑通而自动放宽覆盖要求。实验上下文不同的数据不能合并训练。

输出 `model.json`、可执行的 `policy.json`、`report.json`。报告比较“始终 BASE”“始终候选”“收益树选择”，这些数值复用已观测的配对结果，状态明确是 `trained_not_game_validated`。它们不是新模型实际对局的胜率。

## 部署验证与下一轮纠错

冻结模型后，在未用来改特征或调树的新场景上准备同协议 study，运行 BASE 和 SELECTOR：

```powershell
& $treePython medivac_tree/scripts/return_study.py run `
  --study-dir medivac_tree/artifacts/return_study_validation_v1 --arm selector `
  --policy medivac_tree/artifacts/return_model_v1/policy.json `
  --output-dir sc2_skill_adapter/runs/return-validation-v1-selector
```

验证 study 需要先用新清单执行同样的 `prepare`。它的候选条件、原树、绑定器、运行时与环境控制参数必须与训练实验一致；只替换场景组。原生 `--medivac-policy tree --medivac-tree .../policy.json` 入口也支持模型，并在启动游戏前校验实验上下文。

这次实现的是“一个候选相对原树”的收益选择器。周报里进一步串联的第二棵“撤销旧选择”纠错树、多候选收益向量、任意中途状态恢复、多次在线干预都还没有实现。先用当前闭环获得可靠的正负样本，再根据失败轨迹决定是否增加这些机制。不能直接把本轮一次干预学出的收益树改成每步调用：那会改变训练所比较的策略。

早期换目标实验见 [验证记录](RETURN_VALIDATION.md)；2026-11-04 停滞恢复实验见 [停滞恢复验证](IDLE_VALIDATION.md)。
