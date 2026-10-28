# SC2 Skill Adapter：统一环境适配层

本目录统一维护 SMACv1（现有实现基于 SMAC-Hard）与 SMACv2 的适配代码。
技能算法仍在同级 `skills/current`，底层环境与地图仍在 `game_skill/micro` 和 `smacv2`。

| 路径 | 职责 |
| --- | --- |
| `adapter.py` | 统一启动入口；按后端启动独立 Python 子进程 |
| `smacv1/jev_micro/` | SMACv1 / SMAC-Hard 观察、动作、环境封装及对局运行源码 |
| `bridge/` | SMACv2 观察、动作、环境封装及对局运行源码 |
| `scripts/`、`tests/` | SMACv2 实验脚本和离线测试；SMACv1 测试仍在 `game_skill/micro/tests` |
| `configs/`、`runs/`、`analysis/` | 已有配置、对局输出及分析资料 |

在 `C:\Users\ry\skill\SMAC_agent\sc2_skill_adapter` 中运行：

```powershell
.\.venv\Scripts\python.exe adapter.py --help
.\.venv\Scripts\python.exe adapter.py smacv1 --help
.\.venv\Scripts\python.exe adapter.py smacv1 suite --help
.\.venv\Scripts\python.exe adapter.py smacv2 --help
```

SMACv1 使用 `game_skill/.venvs/micro` 与自带 PySC2；SMACv2 使用本目录 `.venv` 与其独立依赖。
统一入口不在同一进程中混载两个后端。SMACv1 相对输出路径仍以 `game_skill/micro` 为基准，
SMACv2 以本目录为基准；建议显式传入绝对输出路径。

旧目录 `smacv2_jev_adapter` 是指向本目录的 Windows junction；
`game_skill/micro/jev_micro` 是指向 `smacv1/jev_micro` 的 junction。
它们不是第二份源码，旧启动命令可继续使用。复制到其他机器时需保留或重新建立这些兼容链接，
尤其是旧 `game_skill/jev_star.py` 入口。历史日志和冻结快照保留原内容与原路径记录。

以下为 SMACv2 后端的使用说明。

## 2026-10-27 跨种族纯 Skill 测试

本地 `adaptive-skill`、`nearest` 和 `random` 入口现支持神族、人族、虫族5v5/10v10。
`scenarios --race protoss|terran|zerg` 选择种族，默认仍为神族；旧PowerShell包装脚本仍创建神族清单。
JEV/ASTRA规划尚未扩展至其他种族，非神族模型方法会明确拒绝，避免套用神族计划。

本轮人族、虫族各40局使用 `baseline_v7_20261024`，直接隔离加载历史v7冻结快照，
不改 `skills/current` 或已有基线。参数与神族30/40胜实验相同，新空间技能关闭。
固定方案和运行入口见 [跨种族测试目录](analysis/cross_race_20261027/PROTOCOL.md)；
完成并核验后的汇总写入该目录的 `REPORT.md`、`RESULTS.json`。

```powershell
.\.venv\Scripts\python.exe run.py scenarios --race terran --size 10 --episodes 40 --seed 170 --unit-balance native --output configs\my-terran-40.json
```

`--race zerg` 创建虫族清单。实际运行必须显式选择版本及完整开关；本轮精确参数保存于
`analysis/cross_race_20261027/commands.json`，重跑时应换用新的输出目录，不能覆盖历史结果。

此目录独立连接同级 `game_skill` 和 `smacv2`。只迁移原论文的微操方法：

```text
SMACv2 合法观察 → ASTRA 通用战术计划（每种配置一次）
               → JEV 逐存活单位 Choice → 合法性复核 → SMACv2 step
```

论文复现的 `nearest`、`jev`、`astra-jev` 方法将技能扩展固定为 **off**，不启用
FocusFireBudget、CoverRetreat、HealAssign、AdaptiveFocusFire 或学习型底层策略。
独立的 `adaptive-skill` 方法见下文“纯 Adaptive Skill”。ASTRA 计划作为上下文，不裁剪合法动作。
不需要训练 QMIX/MAPPO，也不需要 WandB。

## 双方属性统一的实验模式

默认 `original` 保留原 SMACv2 的“我方 `_RL` 单位、敌方原生单位”。
若要统一属性，使用 `native`：双方同兵种均由同一个原生单位定义创建，
因此生命/护盾上限、武器、速度、护甲等共享同一套定义。
每局还核对实际最大属性、升级等级与初始 buff，写入 `balance/`。
巨像双方均为当前 SC2 版本的原生巨像，不再存在 200/150 与 250/100 的差异。

```powershell
# 生成同一批 10 个场景并顺序运行三种方法，各 10 局
.\scripts\experiment.ps1 -Size 5 -Episodes 10 -Seed 100 -UnitBalance native -Method suite
```

本次指定实验的报告路径：
`runs/native-5v5-n10-seed100/REPORT.md`（以报告实际存在及审计完成为准）。

Python CLI 可显式创建、运行及审计：

```powershell
.\.venv\Scripts\python.exe run.py scenarios --size 5 --episodes 10 --seed 100 --unit-balance native --output configs\my-native-10.json
.\.venv\Scripts\python.exe run.py suite --manifest configs\my-native-10.json --output-dir runs\my-native-10
.\.venv\Scripts\python.exe scripts\report_paired.py runs\my-native-10
```

**协议区别：**原生单位也恢复我方的原生自动索敌行为，所以这是自定义统一属性协议，
不能与旧 `_RL` 实验混算为同一基准。三种我方方法共享此设置，敌方仍是难度 7
内置 AI。属性一致不等于双方的观察权限、控制频率和决策算法完全对称。
审计脚本要求各方法代码/依赖/清单一致，并逐局核对实际初始化状态哈希，
不只检查随机种子相同。

## 安装和检查

在 PowerShell 中执行：

```powershell
Set-Location C:\Users\ry\skill\smacv2_jev_adapter
.\scripts\setup.ps1
.\scripts\test.ps1
.\.venv\Scripts\python.exe run.py check
```

脚本默认借用同级 `game_skill\.venvs\micro` 的 **Python 3.10 解释器**创建独立
`.venv`，不复用其包或修改版 PySC2。也可指定：

```powershell
.\scripts\setup.ps1 -PythonExe 'C:\path\to\python310.exe'
```

依赖使用 PyPI PySC2 4.0.0；SMACv2 从同级源码加载，不改动两个原项目。
`requirements-lock.txt` 记录本机验证时的完整依赖版本。
地图直接从同级 `smacv2` 读取，无需复制到游戏安装目录。

默认 SC2 路径为 `C:\Program Files (x86)\StarCraft II`；Python CLI 可通过
`SC2PATH` 或 `--sc2-path` 覆盖，PowerShell 脚本使用 `-Sc2Path`。

JEV 从 `TYPESAFE_API_KEY` 或同级 `game_skill/config.md` 的 `api:` 字段读取密钥。
预检只报告是否存在，不输出密钥。ASTRA 使用已登录的原生 Codex CLI；
Python CLI 支持 `--codex-path`。默认沿用原仓库的模型名称
`jev-1.13.0`、`gpt-6-astra`，实际可用性由服务与账户决定。

## 按顺序运行

先跑免费本地策略，确认游戏接口，再进行模型调用。后两步会使用对应服务额度。

```powershell
# 1. 本地 nearest；每种规模 3 局
.\scripts\experiment.ps1 -Size 5 -Method nearest
.\scripts\experiment.ps1 -Size 10 -Method nearest

# 2. JEV-only
.\scripts\experiment.ps1 -Size 5 -Method jev
.\scripts\experiment.ps1 -Size 10 -Method jev

# 3. ASTRA + JEV
.\scripts\experiment.ps1 -Size 5 -Method astra-jev
.\scripts\experiment.ps1 -Size 10 -Method astra-jev
```

相同 Size/Episodes/Seed 的脚本默认复用同一清单，结果写入新的时间戳目录。
给定 `-Manifest` 时以清单内容为准，Size/Episodes/Seed 不会覆盖已有清单。

批量配对评测：

```powershell
.\scripts\experiment.ps1 -Size 5 -Episodes 32 -Seed 100 -Method suite
.\scripts\experiment.ps1 -Size 10 -Episodes 32 -Seed 100 -Method suite
```

`suite` 顺序执行 nearest、JEV-only、ASTRA+JEV，统一场景清单与环境参数。
32 局是起步规模，不是充分的统计结论；正式实验应使用更多独立场景、报告区间。

精细控制示例：

```powershell
.\.venv\Scripts\python.exe run.py scenarios --size 5 --episodes 3 --seed 1 --output configs\my-5v5.json
.\.venv\Scripts\python.exe run.py run --method astra-jev --manifest configs\my-5v5.json --output-dir runs\my-astra-5v5 --max-requests 2000 --request-timeout 15 --planner-timeout 180 --skill-mode off
```

清单和结果目录都拒绝覆盖。失败保留日志；重新运行请换结果目录。
模型错误、请求预算耗尽、SC2 重启都使实验失败，不会静默切换 nearest 或计成败局。
默认 `max-requests=10000`，按实际 HTTP 尝试计数，包含重试。

## 实验协议

- **地图与规模：**两种规模均使用 `10gen_protoss`。基础配置读自
  `smacv2/smacv2/examples/configs/sc2_gen_protoss.yaml`，只改 `n_units/n_enemies`。
- **随机场景：**复用上游分布采样算法，显式设置嵌套 NumPy RNG 与 Python RNG。
  清单保存阵容、位置、SC2 seed、配置及哈希。每局启动新游戏进程以隔离前一局
  的随机状态。控制器永远不接收清单中的隐藏敌军位置。
- **观察：**从 `get_obs_agent()` 的敌军特征块确定可见性，再构建结构化信息。
  使用所有存活友军的视野并集，不直接向模型提供 `get_state()`、完整敌军字典。
  友军冷却、原生武器数据、动作反馈等沿用 JEV-Star 的增强结构化接口。
  这是集中式控制器，不等同于标准分散执行 MARL 的信息条件。
- **动作：**0 仅死亡单位；1 Stop；2–5 四方向移动；6+j 攻击敌军 j。
  合法候选基于环境掩码并要求目标在队伍视野中。标准配置下狂热者、追猎者、
  巨像的候选中心距离阈值为 2、6、7，与原生武器射程分开记录。
- **时序：**固定步进，标准 `step_mul=8`，所有 JEV 批次使用同一快照。
  每次模型等待后检查游戏时钟未前进；每次环境 step 检查推进 8 loops。
- **ASTRA：**每次运行的一种场景配置只生成一份通用条件式计划，所有局复用。
  输入只有公开生成规则和动作协议，不含第一局观察；这是针对随机场景对
  原论文“每图一份计划”的明确迁移。没有每局重规划或中途重规划。
- **反馈：**捕获原生 ResponseAction 与 observation errors，映射到实际提交动作。
  “命令接受”不当作“已经开火/造成伤害”。
- **支持范围：**标准 Protoss 5v5/10v10、非 EPO、非锥形视野、单位射程开启。
  不支持修改这些条件后仍使用原提示词；不支持其他种族或宏观玩法。

清单锁定阵容和请求出生位置，不意味着不同机器/SC2 版本的战斗轨迹逐帧一致。
必须保留游戏版本、依赖版本与源码哈希。更换模型或服务版本也会影响结果。

## 文件与输出

| 文件 | 用途 |
|---|---|
| `bridge/environment.py` | SMACv2 包装、观察过滤、原生命令记录 |
| `bridge/scenarios.py` | 上游采样器的可复现封装与清单 |
| `bridge/policy.py` | 原微操结构化观察、Choice、分批和解码的适配副本 |
| `bridge/client.py` | 原 JEV HTTP 客户端副本 |
| `bridge/planner.py`、`semantics.py` | ASTRA 调用与 SMACv2 规划语义 |
| `bridge/runner.py` | 完整对局、配对评测、日志和错误处理 |
| `scripts/vendor_controller.py` | 从相邻项目重建四个适配副本；源代码锚点变化时失败 |
| `SOURCE_MANIFEST.json` | 原文件位置和导入时哈希 |
| `tests/test_bridge.py` | 不启动游戏、不调用模型的语义测试 |

单次运行保存：

- `run.json`：协议、环境配置、模型、依赖版本、源码/地图哈希。
- `manifest.json`：实际场景清单。
- `events.jsonl`：观察请求、响应、时钟、动作和原生反馈；不含 API 密钥。
- `episodes.json`、`summary.json`：胜负/平局/超时、消灭比例、步数、耗时、调用和 usage。
- `planner/`：ASTRA schema 和实际计划（有规划时）；完整输入保存在 `events.jsonl`。
- `replays/*.SC2Replay`：每局完整回放。

`suite` 额外保存 `comparison.json`。失败运行的胜率只描述已完成局，
不得纳入完整方法比较。金额不硬编码；应按记录的 usage 与实际账单计算。
真实验证结果见 `VALIDATION.md`。

已有独立运行可这样汇总（脚本拒绝混合清单或纳入失败运行）：

```powershell
.\.venv\Scripts\python.exe scripts\summarize.py runs\nearest-5-smoke runs\jev-5-smoke runs\astra-jev-5-smoke
```

## 纯 Adaptive Skill（更新后的执行器）

适配层通过 `bridge/skill_runtime.py` 直接加载同级 `skills/current` 的技能源码，
不复制技能、不将原项目的 vendored PySC2 加入导入路径。每次新运行记录实际技能源码哈希。
当前技能已从原仓库迁出；历史 70% 批次的 v2 源码保存在 `skills/baseline_v2_70pct`，
目录和版本说明见 [技能库](../skills/README.md)。既有运行器默认仍使用 `current`。
`adaptive-skill` 连续选择 AdaptiveFocusFire，默认 balanced、每个窗口最多 4 步；
每步重新感知，窗口间保留局内角色与移动历史，每局重置。该方法不调用 JEV/ASTRA。

复用此前双方同兵种属性统一的 10 个 5v5 场景：

```powershell
.\.venv\Scripts\python.exe run.py run --method adaptive-skill --manifest configs\protoss_5_vs_5_native_n10_seed100.json --output-dir runs\my-adaptive-5v5 --adaptive-window-fire --adaptive-range-preserving-move --adaptive-target-scoring
.\.venv\Scripts\python.exe scripts\report_skill.py runs\my-adaptive-5v5 runs\native-5v5-n10-seed100
```

输出目录必须尚不存在。三个增强开关默认关闭；以上命令显式全部开启。
`--skill-mode` 可省略；如果指定，纯技能模式须为 `always`，其他方法为 `off`。

`skill_observation` 保存完整决策输入，`skill_step` 保存开关、即时/窗口伤害预算、
移动候选分数、原地分数、建议动作和撤退价值判断；`step` 保存实际动作和游戏反馈。
TacticalReposition 仅在移动评分严格优于原地时建议移动，否则交回攻击分配器；
EmergencyRetreat 仍选择最佳合法撤退方向。评分和伤害预算均为估计，不能当作实测伤害。

首次正式结果位于 `runs/native-5v5-adaptive-skill-n10-seed100/REPORT.md`。
报告复用历史三种基线，逐局核对实际初始化哈希，并核对环境、观察、地图和依赖未变化。

## 观察与轨迹问题修复后的配对运行

相邻 `smacv2` 的 `get_obs_agent` 已修正：不可见敌人只保留自己的空行，
不会清空其他可见敌人的数据；死亡敌人不再注册观察标记。EPO 的逐敌屏蔽和
`fully_observable` 语义保留。新运行在 `run.json` 记录
`observation_protocol: smacv2-visible-rows-v2`，不与旧观察版本混算成绩。

技能由同级 `skills/current` 直接加载。开启原有三个增强开关即可使用改进版：
按控制窗口预测移动、结合己方动作候选距离与敌方武器射程评分、最多两步冷却恢复、
残血击杀优先级与集火连续性。伤害进展在新角色选择前记账；短期失视记忆只引导搜索，
不将记忆中的敌人添加到可见目标或攻击候选。完整诊断增加 `reentry_decisions`、
`search_hint`、`observed_team_damage_progress`；nearest 也保存每步 `policy_observation`。

```powershell
.\.venv\Scripts\python.exe run.py suite --methods nearest adaptive-skill --manifest configs\protoss_5_vs_5_native_n10_seed100.json --output-dir runs\my-corrected-pair --adaptive-window-fire --adaptive-range-preserving-move --adaptive-target-scoring
.\.venv\Scripts\python.exe scripts\report_corrected_pair.py runs\my-corrected-pair
```

报告脚本核对两种方法使用相同修正源码、地图、依赖、初始状态、原生属性和步进时钟，
并核验真实动作合法性、观察/决策/反馈完整性。第一次运行目录：
`runs/native-5v5-observation-v2-skill-v2-n10-seed100/`。
旧 `report_skill.py` 用于旧实验，故意拒绝把本次环境变更与旧基线直接混合。

## 来源与许可证

此项目保留并适配 JEV-Star 的 `client.py`、`feedback.py`、`policy.py`、
`planner.py`，不把原项目重新声明为其他许可证。上游归属和第三方条款见
同级 `game_skill/THIRD_PARTY_NOTICES.md`、`game_skill/licenses/` 和
`smacv2/LICENSE`。研究中应引用 JEV-Star 与 SMACv2，并称为迁移复现，
不能把此处成绩当作原 SMAC-Hard 结果。

## 人族/虫族策略初版

当前技能包位于同级 `../skills/current`。`run.py run --method adaptive-skill --skill-version current --race-profile terran` 或 `--race-profile zerg` 可显式开启种族组合策略；必须使用对应种族的manifest。默认`off`保持原控制器，历史版本不接受此开关。详见 [种族技能说明](../skills/current/docs/RACE_SKILLS.md)。

迁移到 `SMAC_agent` 后，外部 `smacv2`、`game_skill` 依赖可位于本目录上一级或上两级工作区；加载时优先使用同级依赖，不需要修改旧兼容链接。

