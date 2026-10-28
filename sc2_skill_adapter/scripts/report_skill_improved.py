"""Audit and report the frozen improved Skill runs at two team sizes."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics

from report_corrected_pair import behavior
from report_paired import wilson

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def audit(path, size, seed=150):
    summary, meta, manifest, episodes = [read(path / name) for name in (
        "summary.json", "run.json", "manifest.json", "episodes.json")]
    assert summary["status"] == "complete" and summary["error"] is None
    assert summary["completed_episodes"] == len(episodes) == len(manifest["episodes"]) == 20
    assert summary["jev_requests"] == 0 and summary["planner"] is None
    assert [e["sc2_seed"] for e in manifest["episodes"]] == list(range(seed, seed + 20))
    config = manifest["env_args"]
    assert config["unit_balance"] == "native" and config["difficulty"] == "7"
    assert config["capability_config"]["n_units"] == size
    assert config["capability_config"]["n_enemies"] == size
    assert meta["observation_protocol"] == "smacv2-visible-rows-v2"
    assert meta["unit_value_mode"] == "contextual"
    assert meta["unit_value_parameters"]["spatial_context"]
    assert all(meta["skill_parameters"][key] for key in (
        "window_fire", "range_preserving_move", "target_scoring",
        "coordinated_approach", "safe_reentry"))
    assert any(Path(name).name == "unit_value.py" for name in meta["source_hashes"])
    assert any(Path(name).name == "coordinated_approach.py" for name in meta["source_hashes"])
    for name, sha in meta["source_hashes"].items():
        assert hashlib.sha256((ROOT.parent / name).read_bytes()).hexdigest() == sha, name
    checked = behavior(path)
    counts = Counter()
    approach_reasons, reentry_reasons = Counter(), Counter()
    for event in map(json.loads, (path / "events.jsonl").open(encoding="utf-8")):
        if event["event"] != "skill_step":
            continue
        counts["decision_steps"] += 1
        assert event["coordinated_approach_enabled"] and event["safe_reentry_enabled"]
        assert event["unit_value_mode"] == "contextual"
        assert event["unit_value_evaluation_calls"] == 1
        estimate = event["unit_value_estimate"]
        assert estimate["mode"] == "contextual"
        for side, field in (("allies", "preservation_value"), ("visible_enemies", "target_priority_value")):
            for unit in estimate[side].values():
                if field in unit:
                    counts["spatial_unit_estimates"] += 1
                    assert 0 <= unit[field] <= 1
        counts["value_influence_witnesses"] += event.get("unit_value_influenced_decisions", 0)
        for diagnostic in event["approach_diagnostics"].values():
            counts["approach_unit_steps"] += 1
            counts["approach_cycles_detected"] += bool(diagnostic.get("cycle_detected"))
            approach_reasons[diagnostic.get("reason", "unspecified")] += 1
        for diagnostic in event["safe_reentry_diagnostics"].values():
            counts["safe_reentry_evaluations"] += 1
            counts["safe_reentry_critical_exceptions"] += bool(diagnostic.get("critical_exception"))
        for diagnostic in event["movement_diagnostics"].values():
            if "safe_reentry" in diagnostic:
                reentry_reasons.update(diagnostic.get("rejected_candidates", {}).values())
        json.dumps(event, allow_nan=False)
    assert counts["decision_steps"] == checked["steps"] == sum(e["steps"] for e in episodes)
    assert summary["skill"]["unit_value_evaluations"] == counts["decision_steps"]
    assert counts["spatial_unit_estimates"] > 0 and counts["approach_unit_steps"] > 0
    for i, episode in enumerate(episodes, 1):
        balance = read(path / f"balance/episode-{i:04d}.json")
        assert balance["passed"] and balance["mode"] == "native"
        for profile in balance["profiles"].values():
            attributes = profile["shared_attributes"]
            expected = [attributes["health_max"], attributes["shield_max"]]
            assert all(hp == expected for units in profile["initial_health_shields"].values() for hp in units)
        assert (path / f"replays/episode-{i:04d}.SC2Replay").stat().st_size > 0
        assert episode["scenario_sha256"] == manifest["episodes"][i - 1]["sha256"]
        assert episode["outcome"] in ("win", "loss", "draw", "timeout")
    wins = sum(e["outcome"] == "win" for e in episodes)
    return {"size": size, "run_directory": str(path), "outcomes": summary["outcomes"],
            "wins": wins, "total": len(episodes), "win_rate": wins / len(episodes),
            "win_rate_wilson_95": wilson(wins, len(episodes)),
            "mean_enemy_elimination": statistics.mean(e["enemy_elimination_fraction"] for e in episodes),
            "mean_steps": statistics.mean(e["steps"] for e in episodes),
            "balance_audits_and_replays": len(episodes), "model_calls": 0,
            "behavior_checks": checked, "new_mechanisms": dict(counts),
            "approach_reasons": dict(approach_reasons), "safe_reentry_reasons": dict(reentry_reasons),
            "episodes": episodes}, meta


def main(five, ten, output, seed=150):
    a, am = audit(five, 5, seed)
    b, bm = audit(ten, 10, seed)
    assert am["source_hashes"] == bm["source_hashes"]
    assert am["skill_parameters"] == bm["skill_parameters"]
    assert am["unit_value_parameters"] == bm["unit_value_parameters"]
    assert am["versions"] == bm["versions"]
    report = {"runs": [a, b], "seed_start": seed, "same_source_snapshot": True,
              "source_hashes": am["source_hashes"], "skill_parameters": am["skill_parameters"],
              "unit_value_parameters": am["unit_value_parameters"],
              "comparison_limit": "New seeds, changed policy. Historical results are not a paired baseline."}
    output.mkdir(parents=True, exist_ok=True)
    (output / "verified_results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 改进 Skill：SMACv2 5v5 与 10v10 各 20 局", "",
             f"测试日期：2026-10-23。两种规模各用种子 {seed}–{seed + 19}；使用同一份冻结代码与参数。",
             "SMACv2 的 10gen_protoss 场景，随机神族阵容与出生位置；同兵种双方统一为原生属性，敌方为难度 7 内置 AI。"
             "每决策步推进 8 loops，使用己方共享可见信息。双方阵容构成可能不同，属性统一不等于阵容完全镜像。", "",
             "纯本地 AdaptiveFocusFire，balanced 姿态、4 步激活窗口，无模型调用。"
             "保留原三个增强开关，并启用空间情境价值、协同接敌、安全重入；价值参数使用验证后的默认权重。", "",
             "| 场景 | 方法 | 敌方平均消灭比例 | 胜率（胜场／总局数） | 平均步数 |",
             "|---|---|---:|---:|---:|"]
    for row in (a, b):
        lines.append(f"| {row['size']}v{row['size']} | 改进后的纯 Skill | "
                     f"{row['mean_enemy_elimination']:.1%} | {row['win_rate']:.1%}"
                     f"（{row['wins']}／{row['total']}） | {row['mean_steps']:.2f} |")
    lines += ["", "敌方平均消灭比例取各局最终消灭数／敌方初始数的平均；平局和超时均不计胜场。平均步数包含全部 20 局。", "",
              "## 已实现的改进", "",
              "- SMACv2 的 CLI、执行器、事件日志及源码哈希完整接入单位价值模块。",
              "- 价值排序沿用原有排序层级；零权重基线不再因额外排序规则改变动作。",
              "- 空间情境分别提供敌方优先消灭价值与己方保存价值，并记录可解释的局部信息。",
              "- 接敌保持短期目标，考虑拥堵、预测位置预留及观察到的短周期往返；撤退比较站立和下一次开火机会。", "",
              "## 验收与限制", "",
              "40 局均保留属性审计、逐步观测、决策诊断和 SC2Replay；逐步动作合法性与 8 loops 推进通过核对。"
              "两组源码快照一致，核心度每个决策步实际计算一次，协同接敌实际参与动作选择。",
              "冻结前 370 项技能测试、24 项适配器测试通过；旧 1,375 帧中的关闭配置及零消费权重配置均与历史动作完全一致。",
              "此次衡量的是改进版的实战表现。旧版使用其他种子，不能仅按历史胜率差值认定改进收益；"
              "固定旧观测的离线消融只检验行为和兼容性，不是新对局或胜率。核心度仍是启发式估计，簇集只是范围伤害潜力的代理。", ""]
    if seed == 150:
        lines += ["开发验证记录：首批 v3 使用种子 130–149，各 20 局：5v5 为 12/20 胜、消灭 86%、67.4 步；"
                  "10v10 为 10/20 胜、消灭 87.5%、70.4 步。复查发现局部接敌可能在两点间往返，"
                  "补充短周期检测后，以 v4 和新种子 150–169 完成本报告的最终评测。"
                  "首批完整结果见 pilot_seed130/REPORT.md；两种版本成绩不合并，未按成绩挑选对局。", ""]
    for row in (a, b):
        counts = row["new_mechanisms"]
        low, high = row["win_rate_wilson_95"]
        lines += [f"### {row['size']}v{row['size']}", "",
                  f"结果：{row['outcomes']}。胜率 95% Wilson 区间：{low:.1%}–{high:.1%}。",
                  f"核心度计算 {counts['decision_steps']} 次；协同接敌 {counts['approach_unit_steps']} 个单位步；"
                  f"安全重入评估 {counts.get('safe_reentry_evaluations', 0)} 次。"
                  f"检测同目标两点循环 {counts.get('approach_cycles_detected', 0)} 次，"
                  f"有限时禁回影响 {row['approach_reasons'].get('bounded_cycle_escape', 0)} 个单位步。", "",
                  "| 局数 | 种子 | 结果 | 消灭比例 | 步数 |", "|---:|---:|---|---:|---:|"]
        for i, episode in enumerate(row["episodes"]):
            lines.append(f"| {i + 1} | {seed + i} | {episode['outcome']} | "
                         f"{episode['enemy_elimination_fraction']:.0%} | {episode['steps']} |")
        lines.append("")
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps([{k: v for k, v in row.items() if k != "episodes"} for row in (a, b)],
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--five", type=Path, required=True)
    parser.add_argument("--ten", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "analysis/skill_improvements_20261023")
    parser.add_argument("--seed", type=int, default=150)
    args = parser.parse_args()
    main(args.five.resolve(), args.ten.resolve(), args.output.resolve(), args.seed)
