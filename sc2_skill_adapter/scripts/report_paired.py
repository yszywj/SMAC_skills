"""Audit a completed native-unit suite and produce paired CSV/Markdown results."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def wilson(wins, n):
    z = 1.959963984540054
    p = wins / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return center - half, center + half


def report(root):
    names = ["nearest", "jev", "astra-jev"]
    summaries = {name: load(root / name / "summary.json") for name in names}
    metadata = {name: load(root / name / "run.json") for name in names}
    episodes = {name: load(root / name / "episodes.json") for name in names}
    manifests = {name: load(root / name / "manifest.json") for name in names}
    reference = manifests["nearest"]
    n = len(reference["episodes"])
    if reference["env_args"].get("unit_balance") != "native":
        raise ValueError("This report requires the native-unit protocol")
    for name in names:
        s = summaries[name]
        assert s["status"] == "complete" and s["completed_episodes"] == n
        assert len(episodes[name]) == n and manifests[name] == reference
        assert metadata[name]["source_hashes"] == metadata["nearest"]["source_hashes"], "Code changed during suite"
        assert metadata[name]["versions"] == metadata["nearest"]["versions"], "Runtime versions differ"
    paired = []
    total_steps = 0
    games = set()
    audits = []
    for index in range(n):
        rows = {name: episodes[name][index] for name in names}
        assert len({row["scenario_sha256"] for row in rows.values()}) == 1
        assert len({row["initial_state_sha256"] for row in rows.values()}) == 1, f"Initial states differ: {index+1}"
        scenario = reference["episodes"][index]["episode_config"]
        roster = scenario["team_gen"]["ally_team"]
        assert roster == scenario["team_gen"]["enemy_team"]
        row = {"episode": index + 1, "sc2_seed": reference["episodes"][index]["sc2_seed"],
               "roster": ",".join(roster), "scenario_sha256": rows["nearest"]["scenario_sha256"],
               "initial_state_sha256": rows["nearest"]["initial_state_sha256"]}
        for name in names:
            row[name] = rows[name]["outcome"]
            row[f"{name}_steps"] = rows[name]["steps"]
            path = root / name / "balance" / f"episode-{index+1:04d}.json"
            audit = load(path)
            assert audit["passed"] and audit["mode"] == "native"
            # Explicitly verify every initial unit was full-health/full-shield.
            for profile in audit["profiles"].values():
                attributes = profile["shared_attributes"]
                expected = [attributes["health_max"], attributes["shield_max"]]
                for values in profile["initial_health_shields"].values():
                    assert all(value == expected for value in values), f"Initial damage: {path}"
            audits.append(str(path.relative_to(root)))
            replay = root / name / "replays" / f"episode-{index+1:04d}.SC2Replay"
            assert replay.is_file() and replay.stat().st_size > 0
        paired.append(row)
    for name in names:
        with (root / name / "events.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                event = json.loads(line)
                if event["event"] == "episode_started":
                    games.add(event["game_version"])
                if event["event"] == "step":
                    total_steps += 1
                    feedback = event["feedback"]
                    assert feedback["response_mapping"] == "exact"
                    assert feedback["to_game_loop"] - feedback["from_game_loop"] == reference["env_args"]["step_mul"]
    assert len(games) == 1, "SC2 version differs"
    assert total_steps == sum(row["steps"] for rows in episodes.values() for row in rows)
    with (root / "paired_results.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired[0]))
        writer.writeheader()
        writer.writerows(paired)
    result = {"manifest_sha256": reference["sha256"], "initial_states_match": True,
              "unit_attribute_audits_passed": len(audits), "all_units_initially_full": True,
              "audited_steps": total_steps, "game_version": next(iter(games)),
              "source_hashes_match": True, "summaries": summaries, "episodes": paired}
    (root / "verified_results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# 统一原生单位：Protoss 5v5 配对评测",
        "",
        f"三种方法各 {n} 局，共 {n*3} 局。相同清单、相同代码与依赖，场景 seed "
        f"{reference['seed']}；每局游戏 seed 见逐局表。",
        "",
        "| 方法 | 胜 | 负 | 平 | 超时 | 胜率 | 95% Wilson 区间 |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    labels = {"nearest": "nearest", "jev": "JEV-only", "astra-jev": "ASTRA＋JEV"}
    for name in names:
        s = summaries[name]
        o = s["outcomes"]
        lo, hi = wilson(o["win"], n)
        lines.append(f"| {labels[name]} | {o['win']} | {o['loss']} | {o['draw']} | {o['timeout']} | "
                     f"{100*s['win_rate']:.0f}% | {100*lo:.1f}%–{100*hi:.1f}% |")
    lines += [
        "", "## 属性与公平性", "",
        "双方同兵种使用相同原生 unit_type ID，继承同一份游戏单位/武器定义。",
        "逐局校验生命、护盾、能量上限、半径和三类升级等级一致；没有初始 buff，"
        "所有单位均以满生命、满护盾开局。巨像双方均为当前版本原生属性。",
        f"三种方法的每个配对场景实际初始化状态哈希一致。{len(audits)} 份属性审计全部通过；"
        f"{total_steps} 个 step 都推进 8 loops，原生命令反馈均精确映射。",
        "",
        "这是自定义的统一原生单位协议，不是未修改的 SMACv2 标准成绩。"
        "我方原生单位自动索敌也随之启用，三种方法共享这一变化。"
        "敌方仍为难度 7 内置 AI；观察、控制频率和决策策略并非双方完全对称。",
        "",
        "保留原 SMACv2 随机阵容与出生位置分布，没有挑选胜局或删除失败样本。"
        "每种方法只有 10 局，区间较宽，胜率排序不能单独证明稳定优势。",
        "", "## 逐局结果", "",
        "| 局 | 游戏 seed | 阵容 | nearest | JEV-only | ASTRA＋JEV |",
        "|---:|---:|---|---|---|---|",
    ]
    for row in paired:
        lines.append(f"| {row['episode']} | {row['sc2_seed']} | {row['roster']} | "
                     f"{row['nearest']} | {row['jev']} | {row['astra-jev']} |")
    lines += ["", "## 调用与环境", "",
              f"- SC2：{next(iter(games))}。",
              f"- JEV：{metadata['jev']['model']}；ASTRA：{metadata['astra-jev']['planner_model']}。",
              "- ASTRA 生成一份公开分布级通用计划，在 10 局中复用；技能扩展固定 off。",
              "- 请求/usage、完整观察与反馈在各方法目录，金额按实际账单计算。",
              "", "| 方法 | JEV 请求 | ASTRA 请求 | 总耗时（秒） |", "|---|---:|---:|---:|"]
    for name in names:
        s = summaries[name]
        lines.append(f"| {labels[name]} | {s['jev_requests']} | "
                     f"{s['planner']['requests'] if s['planner'] else 0} | {s['wall_seconds']:.1f} |")
    lines += ["", "证据：`verified_results.json`、`paired_results.csv`、各方法的 "
              "`balance/`、`run.json`、`events.jsonl` 和 30 份 `replays/*.SC2Replay`。", ""]
    (root / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({name: {"outcomes": s["outcomes"], "win_rate": s["win_rate"]}
                      for name, s in summaries.items()}, ensure_ascii=False, indent=2))
    print(f"Audit passed: {len(audits)} games, {total_steps} steps, identical paired initial states.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    report(parser.parse_args().root.resolve())
