# 历史 v2 技能基线

本目录保存历史 v2 的技能源码，用于与当前实现对照。目录名是历史标识，不表示当前场景下的保证胜率。原始文件来源与哈希见 `provenance.json`。

从仓库根目录启动对局：

```powershell
.\sc2_skill_adapter\scripts\run_skill_version.ps1 -Size 10 -SkillVersion baseline_v2_70pct -Episodes 2 -Seed 170 -OutputName baseline-v2-example
```

需要先安装适配器环境和 StarCraft II；输出名称须未使用。此入口固定使用旧版本支持的配置，不接受新增技能参数。

也可通过 `from skills.baseline_v2_70pct.skill_manager import SkillManager` 导入。调用者负责提供合法观察，按管理器生命周期选择技能，并在每局开始时调用 `reset()`。日常开发使用 [current](../current/README.md)。
