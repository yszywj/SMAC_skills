"""Audit a skill run against the historical paired native-unit suite."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from report_paired import load, wilson


def report(skill, baseline):
    paths = {name: baseline / name for name in ("nearest", "jev", "astra-jev")}
    paths["adaptive-skill"] = skill
    summaries = {name: load(path / "summary.json") for name, path in paths.items()}
    runs = {name: load(path / "run.json") for name, path in paths.items()}
    episodes = {name: load(path / "episodes.json") for name, path in paths.items()}
    manifest = load(skill / "manifest.json")
    n = len(manifest["episodes"])
    assert n == 10 and manifest["env_args"]["unit_balance"] == "native"
    source = runs["nearest"]["source_hashes"]
    # Controller/runner intentionally changed; physics, visibility, observations,
    # command feedback, map and scenario generation must remain identical.
    critical = [key for key in source if key.replace("\\", "/").endswith((
        "/environment.py", "/config.py", "/scenarios.py", "/policy.py", "/feedback.py",
        "/starcraft2.py", "/wrapper.py", "/distributions.py", ".SC2Map"))]
    assert len(critical) >= 8
    for name, path in paths.items():
        assert load(path / "manifest.json") == manifest
        assert runs[name]["versions"] == runs["nearest"]["versions"]
        assert all(runs[name]["source_hashes"][key] == source[key] for key in critical)
        assert summaries[name]["status"] == "complete"
        assert summaries[name]["completed_episodes"] == n == len(episodes[name])
    pairs = []
    for i in range(n):
        rows = {name: records[i] for name, records in episodes.items()}
        assert len({row["initial_state_sha256"] for row in rows.values()}) == 1
        assert len({row["scenario_sha256"] for row in rows.values()}) == 1
        audit = load(skill / "balance" / f"episode-{i+1:04d}.json")
        assert audit["passed"] and audit["mode"] == "native"
        for profile in audit["profiles"].values():
            attr = profile["shared_attributes"]
            expected = [attr["health_max"], attr["shield_max"]]
            assert all(value == expected for values in
                       profile["initial_health_shields"].values() for value in values)
        assert (skill / "replays" / f"episode-{i+1:04d}.SC2Replay").stat().st_size > 0
        pairs.append({"episode": i + 1, "seed": manifest["episodes"][i]["sc2_seed"],
                      **{name: row["outcome"] for name, row in rows.items()}})
    counts, reasons, versions = Counter(), Counter(), set()
    observed, decisions, steps = {}, {}, {}
    for name, path in paths.items():
        with (path / "events.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                event = json.loads(line)
                kind = event["event"]
                if kind == "episode_started":
                    versions.add(event["game_version"])
                if name != "adaptive-skill":
                    continue
                counts[kind] += 1
                if kind == "skill_observation":
                    observed[event["episode"], event["step"]] = event["payload"]
                if kind == "skill_step":
                    decisions[event["episode"], event["step"]] = event
                    assert all(event[key] for key in ("window_fire_enabled",
                               "range_preserving_move_enabled", "target_scoring_enabled"))
                    assert all(key in event for key in
                               ("immediate_damage", "window_only_damage", "movement_diagnostics"))
                    for row in event["movement_diagnostics"].values():
                        reasons[row["reason"]] += 1
                        if row["reason"] in ("scored_move", "keep_attack_position"):
                            assert "standing_score" in row and "movement_candidate_scores" in row
                if kind == "step":
                    steps[event["episode"], event["step"] - 1] = event
                    feedback = event["feedback"]
                    assert feedback["response_mapping"] == "exact"
                    assert feedback["to_game_loop"] - feedback["from_game_loop"] == 8
    assert len(versions) == 1
    assert observed.keys() == decisions.keys() == steps.keys()
    assert len(steps) == sum(row["steps"] for row in episodes["adaptive-skill"])
    assert summaries["adaptive-skill"]["jev_requests"] == 0
    assert summaries["adaptive-skill"]["planner"] is None
    verified = {"paired_initial_states_match": True, "native_attribute_audits": n,
                "unchanged_environment_observation_sources": critical,
                "runtime_versions_match": True, "sc2_version": next(iter(versions)),
                "skill_steps": len(steps), "complete_observations_and_diagnostics": True,
                "movement_reasons": dict(reasons),
                "summaries": summaries, "paired_results": pairs,
                "baseline_results_reused": str(baseline)}
    (skill / "verified_results.json").write_text(
        json.dumps(verified, ensure_ascii=False, indent=2), encoding="utf-8")
    with (skill / "paired_results.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pairs[0]))
        writer.writeheader()
        writer.writerows(pairs)
    lines = ["# 更新后的纯 Skill：Protoss 5v5，10 局", "",
             "本次仅新跑 AdaptiveFocusFire（balanced，4 步窗口，三个新开关全部启用）。"
             "nearest、JEV-only、ASTRA＋JEV 复用此前已完成的 10 局数据。",
             "使用相同 native 属性协议和 seed 100–109；10 组实际初始状态哈希完全一致。", "",
             "| 方法 | 胜 | 负 | 平/超时 | 胜率 | 95% Wilson 区间 |",
             "|---|---:|---:|---:|---:|---|"]
    for name, summary in summaries.items():
        o = summary["outcomes"]
        lo, hi = wilson(o["win"], n)
        lines.append(f"| {name} | {o['win']} | {o['loss']} | {o['draw']}/{o['timeout']} | "
                     f"{summary['win_rate']:.0%} | {lo:.1%}–{hi:.1%} |")
    lines += ["", "只有 10 个配对场景，不能据此断言稳定胜率优势。"
              "这是双方同兵种原生属性统一的自定义协议；敌方仍为难度 7 内置 AI。",
              "", f"核验：{len(steps)} 个 skill 步均保存观察、诊断、动作和反馈；"
              "全部推进 8 loops。10 份属性审计通过、10 份回放保存，模型调用为 0。",
              f"SC2 版本：{next(iter(versions))}。环境、观察、地图等关键源码及依赖版本与旧实验一致；"
              "执行器及技能源码变更单独记录于 run.json。",
              "", "移动决策原因计数（单位级评估，包含被后续撤退价值判断拒绝的建议）：",
              "", "```json", json.dumps(dict(reasons), ensure_ascii=False, indent=2), "```",
              "", "standing_score 与 movement_candidate_scores 是启发式位置评分；"
              "immediate_damage/window_only_damage 是预算预测，不是实测伤害。",
              "", "| 局 | seed | nearest | JEV-only | ASTRA＋JEV | Adaptive Skill |",
              "|---:|---:|---|---|---|---|"]
    for row in pairs:
        lines.append("| " + " | ".join(str(v) for v in row.values()) + " |")
    (skill / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({name: s["outcomes"] for name, s in summaries.items()}, indent=2))
    print(f"Audit passed: {n} paired games; {len(steps)} skill steps; {dict(reasons)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("skill", type=Path)
    parser.add_argument("baseline", type=Path)
    args = parser.parse_args()
    report(args.skill.resolve(), args.baseline.resolve())
