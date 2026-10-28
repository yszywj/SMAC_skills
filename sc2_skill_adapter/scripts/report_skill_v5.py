"""Audit v5's 40 games and compare each initial state with the frozen v4 run."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics

from report_skill_improved import ROOT, audit, read


def aggregate(episodes):
    return {"wins": sum(e["outcome"] == "win" for e in episodes),
            "total": len(episodes),
            "mean_enemy_elimination": statistics.mean(e["enemy_elimination_fraction"] for e in episodes),
            "mean_steps": statistics.mean(e["steps"] for e in episodes)}


def new_diagnostics(path):
    counts, damage_status, reasons = Counter(), Counter(), Counter()
    for event in map(json.loads, (path / "events.jsonl").open(encoding="utf-8")):
        if event["event"] == "step":
            feedback = event["feedback"]
            for row in feedback["actors"].values():
                assert row["submission"] in ("accepted", "dead_noop"), row
                counts["actions_" + row["submission"]] += 1
                counts["observation_errors"] += len(row.get("observation_errors", []))
            counts["unmatched_observation_errors"] += len(feedback.get("unmatched_observation_errors", []))
        if event["event"] != "skill_step":
            continue
        for key in ("accurate_damage", "critical_escape", "planned_reentry"):
            assert event[key + "_enabled"]
        for key in ("critical_escape", "planned_reentry"):
            for row in event[key + "_diagnostics"].values():
                counts[key + "_evaluations"] += 1
                counts[key + "_applied"] += bool(row.get("applied"))
                reasons[key + ":" + str(row.get("reason", "unspecified"))] += 1
        model = event["damage_model"]
        for pairs in model.get("pairs", {}).values():
            for estimate in pairs.values():
                damage_status[estimate["status"]] += 1
                if estimate["estimate"] is not None:
                    assert 0 <= estimate["lower_bound"] <= estimate["estimate"] <= estimate["upper_bound"]
    summary = read(path / "summary.json")
    for key in ("critical_escape", "planned_reentry"):
        for suffix in ("_evaluations", "_applied"):
            assert counts[key + suffix] == summary["skill"][key + suffix]
    return {"counts": dict(counts), "damage_status": dict(damage_status), "reasons": dict(reasons)}


def main(output):
    rows, metas = [], []
    validation = read(output / "validation.json")
    assert validation["skill-tests.txt"]["passed"] and validation["adapter-tests.txt"]["passed"]
    previous_snapshot = ROOT / "analysis/skill_improvements_20261023/source_snapshot_final"
    for size in (5, 10):
        current = ROOT / f"runs/native-{size}v{size}-v5-skill-n20-seed150"
        previous = ROOT / f"runs/native-{size}v{size}-v4-skill-n20-seed150"
        row, meta = audit(current, size, 150)
        assert all(meta["skill_parameters"][key] for key in (
            "accurate_damage", "critical_escape", "planned_reentry"))
        old = read(previous / "episodes.json")
        old_meta = read(previous / "run.json")
        assert read(current / "manifest.json") == read(previous / "manifest.json")
        assert meta["versions"] == old_meta["versions"]
        for name, sha in old_meta["source_hashes"].items():
            assert hashlib.sha256((previous_snapshot / name).read_bytes()).hexdigest() == sha
            if name.startswith("smacv2\\") or name.endswith("bridge\\environment.py"):
                assert meta["source_hashes"][name] == sha, "Environment changed: " + name
        pairs, transitions = [], Counter()
        for a, b in zip(old, row["episodes"]):
            assert a["episode"] == b["episode"]
            assert a["initial_state_sha256"] == b["initial_state_sha256"]
            assert a["scenario_sha256"] == b["scenario_sha256"]
            transitions[a["outcome"] + "->" + b["outcome"]] += 1
            pairs.append({"episode": a["episode"], "seed": 149 + a["episode"],
                          "before": a["outcome"], "after": b["outcome"],
                          "elimination_before": a["enemy_elimination_fraction"],
                          "elimination_after": b["enemy_elimination_fraction"],
                          "steps_before": a["steps"], "steps_after": b["steps"]})
        row.update(previous=aggregate(old), paired_episodes=pairs,
                   transitions=dict(transitions), diagnostics=new_diagnostics(current),
                   all_initial_states_match=True)
        rows.append(row)
        metas.append(meta)
    for field in ("source_hashes", "skill_parameters", "unit_value_parameters", "versions"):
        assert metas[0][field] == metas[1][field], field
    frozen = read(output / "source_snapshot/snapshot.json")
    assert frozen["source_hashes"] == metas[0]["source_hashes"]
    for name, sha in frozen["source_hashes"].items():
        assert hashlib.sha256((output / "source_snapshot" / name).read_bytes()).hexdigest() == sha
    data = {"runs": rows, "same_initial_states": True, "same_environment": True,
            "validation": validation,
            "source_hashes": metas[0]["source_hashes"],
            "skill_parameters": metas[0]["skill_parameters"],
            "scope": "One frozen v5 policy, 40 new games; v4 historical paired baseline. No per-feature causal ablation."}
    (output / "verified_results.json").write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# Skill v5：修正后 5v5、10v10 各 20 局", "",
             "2026-10-23；纯本地 AdaptiveFocusFire，未调用模型。SMACv2 10gen_protoss，"
             "双方同兵种原生属性，敌方难度 7 内置 AI，每步推进 8 loops，使用己方共享可见信息。"
             "两种规模均复用 v4 的种子 150–169 和清单；40 局初始状态哈希逐局一致，"
             "环境源码、地图、攻击合法性掩码未修改。", "",
             "| 场景 | 版本 | 敌方平均消灭比例 | 胜率（胜场／总局数） | 平均步数 |",
             "|---|---|---:|---:|---:|"]
    for row in rows:
        for label, value in (("v4（历史同种子）", row["previous"]), ("v5（本轮）", row)):
            lines.append(f"| {row['size']}v{row['size']} | {label} | {value['mean_enemy_elimination']:.1%} | "
                         f"{value['wins']/value['total']:.0%}（{value['wins']}／{value['total']}） | {value['mean_steps']:.2f} |")
    lines += ["", "消灭比例、平均步数均包含全部 20 局；胜率分母也是全部 20 局。", "",
              "本轮结果没有体现整体提升：5v5 增加 1 胜，但平均消灭比例下降 1 个百分点；"
              "10v10 减少 3 胜，平均消灭比例下降 5.5 个百分点。功能修正确已接入，不能据此称 v5 为胜率更优版本。"
              "v4 的结果与冻结源码继续保留，后续应以同种子单项消融定位取舍，不能仅由本批三项同时改变来判定原因。", "",
              "**本轮暴露的已知问题：**10v10 第 17 局，u5 追猎者在 step65–87 连续 23 个决策中北撤、南返交替，"
              "没有攻击分配；step66 起冷却为零，有合法目标的窗口触发危急撤退，随后又重入。"
              "保命与重入组合仍可能造成不开火的循环，本轮尚未修复。该行为可由轨迹确认，但不能证明它单独造成整局失败。"
              "详细证据见 [独立轨迹审计](INDEPENDENT_AUDIT.md)。", "",
              "## 修改内容", "",
              "- 伤害预算保留单位属性和武器对属性加成，按每击护甲计算；此前无盾追猎者对追猎者的 12 点估计修为 17 点。"
              "有盾跨层使用明确的保守上下界，未知属性、未知升级不假报精确伤害，也不虚构巨像溅射收益。",
              "- 危急保命：保留本窗口可击杀目标的优先级；低生存值且存在降低敌方覆盖的可执行移动时，不再一律要求上一窗口损失 20% 最大生命盾值。",
              "- 撤退与重入共同考虑移动后返回合法攻击范围所需窗口、武器冷却和敌方覆盖，用于避免没有攻击机会的恢复绕行。",
              "- 三项增强默认关闭；本轮显式开启，原单位价值、协同接敌和安全重入仍启用。每步事件保存启用状态、候选比较和实际采用结果。", "",
              "## 验收与解释", "",
              "两组使用同一源码快照和参数；40 局的属性一致性审计、回放、逐步动作合法性及实际 8 loops 推进全部通过。"
              "完整源码快照、测试输出、离线历史状态检查、逐局比较见本目录。",
              f"冻结前 {validation['skill-tests.txt']['tests']} 项技能测试、"
              f"{validation['adapter-tests.txt']['tests']} 项适配器测试通过；关闭新开关在历史 v4 的 "
              f"{validation['recorded_v4_steps']:,} 帧中完整复现原动作。更早的 1,375 帧中旧配置和零消费权重配置也与历史动作一致。",
              "相同初始状态使这次比较比不同种子更有解释力，但每种规模仅 20 局，而且三项机制同时变化；"
              "不能把胜率变化归因到某一项，也不能将样本胜率视为稳定真实胜率。移动后的敌人位置仍是短期几何估计，敌方意图未知。", ""]
    for row in rows:
        counts = row["diagnostics"]["counts"]
        lines += [f"### {row['size']}v{row['size']} 配对变化", "",
                  f"逐局胜负转移：`{row['transitions']}`。",
                  f"危急保命实际采用 {counts.get('critical_escape_applied', 0)} 个单位步；"
                  f"重入计划实际采用 {counts.get('planned_reentry_applied', 0)} 个单位步。"
                  "这些是机制触发次数，不等于被救活数或额外胜场。", "",
                  "| 局 | 种子 | v4 结果 | v5 结果 | v4 消灭比例 | v5 消灭比例 | v4 步数 | v5 步数 |",
                  "|---:|---:|---|---|---:|---:|---:|---:|"]
        for pair in row["paired_episodes"]:
            lines.append(f"| {pair['episode']} | {pair['seed']} | {pair['before']} | {pair['after']} | "
                         f"{pair['elimination_before']:.0%} | {pair['elimination_after']:.0%} | "
                         f"{pair['steps_before']} | {pair['steps_after']} |")
        lines.append("")
    lines += ["## 使用方法", "", "在 `C:\\Users\\ry\\skill\\smacv2_jev_adapter` 执行：", "", "```powershell",
              ".\\scripts\\run_improved_skill.ps1 -Size 5 -PolicyVersion v5 -Episodes 20 -Seed 150 -OutputName my-v5-5v5",
              ".\\scripts\\run_improved_skill.ps1 -Size 10 -PolicyVersion v5 -Episodes 20 -Seed 150 -OutputName my-v5-10v10",
              "```", "", "指定新的输出目录名；脚本拒绝覆盖已有结果。运行需要本机 StarCraft II 与适配器虚拟环境。", ""]
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps([{k: v for k, v in row.items() if k not in ("episodes", "paired_episodes")}
                      for row in rows], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "analysis/skill_repairs_v5_20261023")
    main(parser.parse_args().output.resolve())
