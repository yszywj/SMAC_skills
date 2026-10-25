# 微操技能库

## 独立仓库使用说明

本仓库保存技能源码及历史版本。纯技能模块的外部 Python 依赖为 httpx，可运行 python -m pip install -r requirements.txt 安装。

真实对局、完整环境测试及历史迁移回放仍依赖同级的 smacv2_jev_adapter、game_skill 和相关数据；它们未包含在本仓库中。下文指向这些同级目录的链接适用于原本地工作区，在 GitHub 上不可用。现有测试使用 skills.current 导入路径，克隆时建议将目录命名为 skills，并从其父目录运行测试。


当前技能已从 `game_skill/micro/jev_micro` 实际迁出，后续统一在本目录维护。迁移保留全部算法与参数；SMACv2 和 SMAC-Hard 的既有运行入口仍可使用。

| 目录 | 用途 |
| --- | --- |
| [current](current/README.md) | 当前v7：技能、调度器、单位核心度评估和必要协议依赖；后续开发修改这里 |
| [baseline_v2_70pct](baseline_v2_70pct/README.md) | 历史 10v10 达到 7 胜 / 10 局的 v2 原始技能；保留作对照基线 |
| [baseline_v6_20261024](baseline_v6_20261024/README.md) | v7开发前冻结的v6；上一轮5v5、10v10各15/20胜 |
| [tools/verify_migration.py](tools/verify_migration.py) | 源码校验、导入检查与历史观察离线回放 |
| [verification](verification/MIGRATION_VERIFICATION.md) | 迁移验证报告、逐项证据和测试输出 |
| [MIGRATION.json](MIGRATION.json) | 当前版各文件的旧路径、新路径与 SHA-256 |

## 源码与环境的分工

`current` 中原有14个Python文件从原位置剪切迁入，迁移当时内容逐字节保持不变，之后的v7改进只修改current。原仓库通过 `jev_micro/__init__.py` 将已迁移技能模块的原有导入路由到这里；SMACv2 通过 `bridge/skill_runtime.py` 从这里隔离加载，避免混用两个环境的PySC2。`MIGRATION.json`及迁移报告保持历史记录，不代表current后续开发仍须字节不变。

两个环境的启动程序、观察转换、地图、测试和历史实验仍在原目录。历史源码快照与轨迹也继续保留，用于追溯结果。运行时的 `skill_source` 与源码哈希记录新位置。

重新审核旧实验时，应核对该实验配套的冻结源码快照。旧日志记录的是当时的路径与哈希，不能用当前文件替代；历史报告工具 `report_skill_improved.py` 直接检查当前工作区，因此需要恢复对应快照的目录结构后才适用于旧实验。

旧版的 10 个原始文件按 Git 历史及当时运行的 SHA-256 恢复，来源见其 `provenance.json`。**70% 指种子 100–109 的历史 7/10 结果**；该批平均敌方消灭比例为 91%，平均 74.9 个决策步。它不是新测试成绩，也不保证后续仍有 70% 胜率。

## 使用当前v7

在适配器目录执行以下命令，每种规模各40局，输出目录须未使用：

```powershell
.\scripts\run_improved_skill.ps1 -Size 5 -Episodes 40 -Seed 170 -PolicyVersion v7 -OutputName my-v7-5v5
.\scripts\run_improved_skill.ps1 -Size 10 -Episodes 40 -Seed 170 -PolicyVersion v7 -OutputName my-v7-10v10
```

v7只增加接敌进展保护开关，详见 [当前技能说明](current/README.md)。冻结v6可通过 `run_skill_version.ps1 -SkillVersion baseline_v6_20261024` 单独加载。旧包装脚本不指定 `-PolicyVersion` 时仍默认v6。

v7的80局已完成：5v5为25/40胜、85.00%平均消灭、61.80步；10v10为30/40胜、94.75%平均消灭、73.33步。每规模分成20局同场景回归和20局新场景留出；同场景相对v6分别少一胜、多一胜，尚无稳定胜率提升证据。完整结果与案例见 [v7测试报告](../smacv2_jev_adapter/analysis/skill_v7_20261024/REPORT.md)。

## 使用原v6配置

现有 SMACv2 命令保持不变。例如在 PowerShell 中执行：

```powershell
Set-Location C:\Users\ry\skill\smacv2_jev_adapter
.\scripts\run_improved_skill.ps1 -Size 5 -Episodes 20 -Seed 150 -PolicyVersion v6 -OutputName my-current-5v5
.\scripts\run_improved_skill.ps1 -Size 10 -Episodes 20 -Seed 150 -PolicyVersion v6 -OutputName my-current-10v10
```

输出目录名称须未被使用。这些命令会启动新的真实对局；本次迁移未执行它们。配置与机制说明见 [IMPROVED_SKILL.md](../smacv2_jev_adapter/IMPROVED_SKILL.md)。

旧版可在适配器目录的 Python 环境中单独加载：

```python
from bridge.skill_runtime import load_manager

CurrentManager = load_manager()
BaselineManager = load_manager("baseline_v2_70pct")
```

两版本的模块缓存相互独立。旧版参数和生命周期用法见 [旧版说明](baseline_v2_70pct/README.md)。既有对局 CLI 默认仍加载 `current`，原包装脚本的 `-PolicyVersion` 选择当前源码上的 v4/v5/v6 配置。

2026-10-24 新增了实际切换源码版本的 `--skill-version` 与批测入口。以下命令使用旧版原始源码和原始三个开关，运行新的 20 局 10v10：

```powershell
Set-Location C:\Users\ry\skill\smacv2_jev_adapter
.\scripts\run_skill_version.ps1 -Size 10 -SkillVersion baseline_v2_70pct -Episodes 20 -Seed 170 -OutputName my-baseline-10v10
```

改为 `-SkillVersion current` 使用current源码的v6配置，新接敌保护开关关闭；固定历史v6请选 `baseline_v6_20261024`。改为 `-Size 5` 即为5v5。该入口拒绝覆盖结果；旧版拒绝接收新增技能参数。每次运行记录实际版本、源路径及哈希。新旧版本的同场景评测方案见 [对照方案](../smacv2_jev_adapter/analysis/skill_v2_v6_paired_20261024/PROTOCOL.md)。

本轮80局已完成：当前版在5v5与10v10均为15/20胜，旧版分别14/20与12/20。结果、轨迹分析及改进建议见 [同场景测试报告](../smacv2_jev_adapter/analysis/skill_v2_v6_paired_20261024/REPORT.md)。这是本批样本成绩，不是稳定胜率承诺。

## 迁移验证

407 项技能及原环境测试、32 项适配器测试通过。当前版 40 局历史观察的 2667 帧、旧版 30 局历史观察的 2124 帧，合计 **4791 帧**的动作与关键诊断均与原记录一致；当前 14 个文件和旧版 10 个文件的历史哈希全部匹配。

以下是迁移当时使用的离线检查命令。它校验迁移时的源码哈希，重验应使用该次冻结快照；已开发到v7的current不再与迁移时逐字节相同。当前版本验收见 [v7离线报告](../smacv2_jev_adapter/analysis/skill_v7_20261024/OFFLINE_REGRESSION.md)。

```powershell
Set-Location C:\Users\ry\skill
.\smacv2_jev_adapter\.venv\Scripts\python.exe -B skills\tools\verify_migration.py
```

这验证迁移后的决策一致性，不产生新的胜率。测试日志见 `verification/skill-tests.txt`、`verification/adapter-tests.txt`。

上游来源与许可证继续适用，参见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) 和 [licenses](licenses)。
