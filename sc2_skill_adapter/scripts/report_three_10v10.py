"""Audit fresh JEV, ASTRA+JEV and skill runs on the corrected SMACv2 environment."""
import argparse
import csv
import json
from pathlib import Path

from report_paired import load, wilson
from report_corrected_pair import behavior


def report(root):
    names = ("jev", "astra-jev", "adaptive-skill")
    labels = {"jev": "JEV-only", "astra-jev": "ASTRA＋JEV", "adaptive-skill": "纯 Skill"}
    summaries = {n: load(root / n / "summary.json") for n in names}
    metadata = {n: load(root / n / "run.json") for n in names}
    episodes = {n: load(root / n / "episodes.json") for n in names}
    manifest = load(root / names[0] / "manifest.json")
    n_games = len(manifest["episodes"])
    assert n_games == 10 and manifest["env_args"]["unit_balance"] == "native"
    common_sources = metadata["jev"]["source_hashes"]
    versions, steps, requests, audits = set(), {}, {}, 0
    for name in names:
        directory = root / name
        summary = summaries[name]
        assert summary["status"] == "complete"
        assert summary["completed_episodes"] == n_games == len(episodes[name])
        assert load(directory / "manifest.json") == manifest
        assert metadata[name]["observation_protocol"] == "smacv2-visible-rows-v2"
        assert metadata[name]["versions"] == metadata["jev"]["versions"]
        assert all(metadata[name]["source_hashes"][p] == h for p, h in common_sources.items())
        steps[name], requests[name] = 0, 0
        with (directory / "events.jsonl").open(encoding="utf-8") as handle:
            for event in map(json.loads, handle):
                if event["event"] == "episode_started":
                    versions.add(event["game_version"])
                if event["event"] == "jev_request":
                    requests[name] += 1
                if event["event"] == "step":
                    steps[name] += 1
                    f = event["feedback"]
                    assert f["response_mapping"] == "exact"
                    assert f["to_game_loop"] - f["from_game_loop"] == 8
        assert steps[name] == sum(row["steps"] for row in episodes[name])
        assert requests[name] == summary["jev_requests"]
        for i in range(1, n_games + 1):
            audit = load(directory / "balance" / f"episode-{i:04d}.json")
            assert audit["passed"] and audit["mode"] == "native"
            for profile in audit["profiles"].values():
                attrs = profile["shared_attributes"]
                expected = [attrs["health_max"], attrs["shield_max"]]
                assert all(hp == expected for values in profile["initial_health_shields"].values() for hp in values)
            assert (directory / "replays" / f"episode-{i:04d}.SC2Replay").stat().st_size > 0
            audits += 1
    assert len(versions) == 1
    assert metadata["jev"]["skills"] == metadata["astra-jev"]["skills"] == "off"
    assert summaries["astra-jev"]["planner"]["successful_responses"] == 1
    assert summaries["adaptive-skill"]["jev_requests"] == 0
    assert summaries["adaptive-skill"]["planner"] is None
    skill_checks = behavior(root / "adaptive-skill")
    paired = []
    for i in range(n_games):
        rows = [episodes[name][i] for name in names]
        assert len({r["initial_state_sha256"] for r in rows}) == 1
        assert len({r["scenario_sha256"] for r in rows} | {manifest["episodes"][i]["sha256"]}) == 1
        paired.append({"episode": i + 1, "seed": manifest["episodes"][i]["sc2_seed"],
                       **{name: episodes[name][i]["outcome"] for name in names}})
    result = {"observation_protocol": "smacv2-visible-rows-v2",
              "manifest_sha256": manifest["sha256"],
              "paired_initial_states_match": True,
              "shared_source_hashes_and_dependencies_match": True,
              "game_version": next(iter(versions)), "attribute_audits_passed": audits,
              "audited_steps": steps, "skill_checks": skill_checks,
              "all_three_methods_freshly_run": True,
              "summaries": summaries, "paired": paired}
    (root / "verified_results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with (root / "paired_results.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired[0]))
        writer.writeheader()
        writer.writerows(paired)
    lines = ["# SMACv2 Protoss 10v10：JEV、ASTRA＋JEV、纯 Skill 各10局", "",
             "三种方法均为本轮新跑，各10局，共30局；纯Skill没有复用上一轮结果。"
             "相同Protoss 10v10清单，seed100–109，双方同兵种原生属性统一，敌方难度7。",
             "观察修复版本为 smacv2-visible-rows-v2；三种方法的实际初始状态、"
             "共有源码、地图和依赖一致。旧观察版本的成绩不混入本表。",
             "", "| 方法 | 胜 | 负 | 平 | 超时 | 胜率 | 95% Wilson区间 |",
             "|---|---:|---:|---:|---:|---:|---|"]
    for name in names:
        s = summaries[name]
        o = s["outcomes"]
        lo, hi = wilson(o["win"], n_games)
        lines.append(f"| {labels[name]} | {o['win']} | {o['loss']} | {o['draw']} | "
                     f"{o['timeout']} | {s['win_rate']:.0%} | {lo:.1%}–{hi:.1%} |")
    lines += ["", "JEV-only与ASTRA＋JEV关闭技能扩展。ASTRA按此前协议生成一份公开场景分布级计划，"
              "10局复用；没有读取第一局快照来规划，也没有逐局重规划。"
              "纯Skill为改进版AdaptiveFocusFire，balanced、4步窗口、三个增强开关全部开启。",
              "", f"SC2版本 {next(iter(versions))}；{audits}份属性审计和30份回放完整，"
              f"共{sum(steps.values())}个环境步，均推进8 loops并精确映射原生命令反馈。",
              "10局样本较小，胜率差不能单独证明稳定优劣；本实验是同兵种属性统一的自定义协议。",
              "", "## 逐局结果", "",
              "| 局 | seed | JEV-only | ASTRA＋JEV | 纯Skill |",
              "|---:|---:|---|---|---|"]
    for row in paired:
        lines.append("| " + " | ".join(str(v) for v in row.values()) + " |")
    lines += ["", "## 调用记录", "",
              "| 方法 | 平均敌方消灭比例 | 平均步数 | JEV请求（含重试） | ASTRA请求 | 总耗时秒 |",
              "|---|---:|---:|---:|---:|---:|"]
    for name in names:
        s = summaries[name]
        lines.append(f"| {labels[name]} | {s['mean_enemy_elimination']:.0%} | "
                     f"{steps[name] / n_games:.1f} | {s['jev_requests']} | "
                     f"{s['planner']['requests'] if s['planner'] else 0} | {s['wall_seconds']:.1f} |")
    lines += ["", "usage保存在各方法summary.json，未按假定价格换算金额。",
              "完整证据：verified_results.json、paired_results.csv、各方法events.jsonl、run.json及replays/。",
              ""]
    (root / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({n: summaries[n]["outcomes"] for n in names}, indent=2))
    print(f"Audit passed: {audits} games, {sum(steps.values())} steps.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    report(p.parse_args().root.resolve())
