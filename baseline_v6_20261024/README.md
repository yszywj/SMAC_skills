# 2026-10-24 冻结的 v6

从上一轮已核验的 `analysis/skill_v2_v6_paired_20261024/source_snapshot/skills/current` 恢复，共15个 Python 文件；逐项 SHA-256 见 `provenance.json`。该源码在种子170–189的5v5、10v10各20局中均取得15胜，不能把它解释为稳定75%胜率。

此目录只作历史对照，开发请修改 `skills/current`。通过适配器 `load_manager("baseline_v6_20261024")` 或 `scripts/run_skill_version.ps1 -SkillVersion baseline_v6_20261024` 加载，不包含 v7 的 `approach_progress_guard`。

完整旧结果见 [v2/v6配对报告](../../smacv2_jev_adapter/analysis/skill_v2_v6_paired_20261024/REPORT.md)。
