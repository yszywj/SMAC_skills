# 历史 v6 技能基线

本目录是冻结的 v6 对照源码，来源和哈希见 `provenance.json`。日常开发使用 [current](../current/README.md)。

从仓库根目录启动对局：

```powershell
.\sc2_skill_adapter\scripts\run_skill_version.ps1 -Size 10 -SkillVersion baseline_v6_20261024 -Episodes 2 -Seed 170 -OutputName baseline-v6-example
```

需要适配器环境和 StarCraft II，输出名称须未使用。适配器通过隔离加载区分不同源码版本；本版本不包含后续新增的接敌进展保护逻辑。
