# 旧版纯微操 Skill 基线（v2，历史 70% 批次）

本目录保存产生历史 10v10 **7 胜 / 10 局（70%）**结果的旧技能源码，种子为 **100–109**，敌方平均消灭比例91%，平均74.9个决策步。70%是这10局的样本结果，不是保证胜率，也不是与当前版本在相同种子上的比较。后续同旧配置另一批20局为13胜、6负、1平；不要把不同批次混作配对实验。

这里的10个原始Python文件按记录中的Git版本恢复，必要时仅恢复记录的CRLF换行；每个文件都与原测试记录的SHA-256逐字节相同。没有用当前算法文件替换旧实现，也没有修改原导入。新增的本包 `__init__.py` 只用于Python包标识，不属于原测试的10文件指纹。

## 内容与边界

- `skill_manager.py`：激活、持续执行、退出与遥测。
- `skill_policy.py`：模式常量、团队选择数据结构与验证。
- `skills/`：原版集火、撤退、治疗、Adaptive组合、开火时机与移动评分。
- `client.py`：保留的原协议依赖，供 `skill_policy.py` 导入响应验证函数。它也定义旧模型客户端，但本目录的纯Skill用法不会实例化客户端、发网络请求或读取密钥；文件不含密钥。运行时需要现有环境的 `httpx`，大型治疗匹配可选择使用NumPy/SciPy，否则有纯Python回退。
- `provenance.json`：每个原文件的Git提交、路径、换行恢复方式、原始及导出哈希。
- `provenance/v2_seed100/`：原始run、manifest、summary、episodes文件，保持原字节，用于核对配置和结果。

环境、SC2、SMACv2、观察转换和模型配置继续由项目的适配器/原仓库负责，没有复制到本包。未复制或读取任何秘密 `config.md`。这里也不是完整独立游戏安装包。

## 纯Skill用法

已接入 SMACv2 对局 CLI。运行新一批旧版 20 局 10v10：

```powershell
Set-Location C:\Users\ry\skill\smacv2_jev_adapter
.\scripts\run_skill_version.ps1 -Size 10 -SkillVersion baseline_v2_70pct -Episodes 20 -Seed 170 -OutputName my-baseline-10v10
```

`-Size 5` 切换到 5v5；输出名称必须未被使用。脚本固定启用原三个增强开关，使用 balanced 和 max_steps=4；底层 `--skill-version baseline_v2_70pct` 拒绝新版技能开关及单位价值参数，运行记录包含实际源码哈希。

在工作区根目录的现有Python环境中导入隔离的旧包：

```python
from skills.baseline_v2_70pct.skill_manager import SkillManager
from skills.baseline_v2_70pct.skill_policy import ADAPTIVE_FOCUS_FIRE

manager = SkillManager(
    max_steps=4,
    combat_params={
        "window_fire": True,
        "range_preserving_move": True,
        "target_scoring": True,
    },
)
manager.reset()  # 每局开始时调用
manager.select(ADAPTIVE_FOCUS_FIRE, posture="balanced")
actions, telemetry = manager.step(payload, n_agents)
```

`payload`必须来自适配器的合法、共享可见观察；按现有运行器的生命周期重新选择已结束的Skill。本包不自行启动游戏。旧版没有核心度、协同接敌、安全重入、准确伤害、危急逃离、计划重入或开火承诺等后续开关；不要把新配置混入本基线。

原测试为SMACv2 10gen_protoss、10v10随机神族阵容、双方同兵种原生属性、难度7内置AI、每决策8 loops；无模型调用。精确配置和场景分布以保留的原manifest/run为准。
