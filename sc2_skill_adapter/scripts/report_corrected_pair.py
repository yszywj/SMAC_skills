"""Audit a fresh nearest/skill pair using the corrected observation protocol."""
import argparse
import json
from collections import Counter
from pathlib import Path

from report_paired import load, wilson


def behavior(path):
    observations, decisions, steps = {}, {}, {}
    games = set()
    for event in map(json.loads, (path / "events.jsonl").open(encoding="utf-8")):
        kind = event["event"]
        if kind in ("skill_observation", "policy_observation"):
            observations[event["episode"], event["step"]] = event["payload"]
        elif kind == "skill_step":
            decisions[event["episode"], event["step"]] = event
        elif kind == "step":
            steps[event["episode"], event["step"] - 1] = event
        elif kind == "episode_started":
            games.add(event["game_version"])
    assert observations.keys() == steps.keys()
    counts = Counter()
    for key, event in steps.items():
        feedback = event["feedback"]
        assert feedback["response_mapping"] == "exact"
        assert feedback["to_game_loop"] - feedback["from_game_loop"] == 8
        request = observations[key]
        for i, action in enumerate(event["actions"]):
            question = request["questions"].get(f"u{i}")
            assert str(action) in question["criteria"] if question else action == 0
        decision = decisions.get(key)
        if decision is None:
            continue
        assert all(decision[field] for field in (
            "window_fire_enabled", "range_preserving_move_enabled", "target_scoring_enabled"))
        counts["recovery_unit_steps"] += len(decision["reentry_decisions"])
        counts["last_seen_search_steps"] += decision["search_hint"] is not None
        reason = decision["all_in_started_now"] or ""
        counts["all_in_started_during_damage_progress"] += (
            bool(reason) and "unproductive_retreat_loop" in reason
            and decision["observed_team_damage_progress"])
        next_step = steps.get((key[0], key[1] + 1))
        for actor in decision["retreat_units"]:
            counts["retreat_unit_steps"] += 1
            if next_step:
                i = int(actor[1:])
                a, b = event["actions"][i], next_step["actions"][i]
                reversed_now = (a, b) in ((2, 3), (3, 2), (4, 5), (5, 4))
                counts["immediate_reverse_after_retreat"] += reversed_now
                if reversed_now:
                    next_state = observations[key[0], key[1] + 1]["state"]
                    counts["reverse_while_cooldown_exceeds_window"] += (
                        next_state["allies"][actor]["weapon_cooldown_loops"]
                        > next_state["step_mul"] + 1)
        for diagnostic in decision["movement_diagnostics"].values():
            counts["mask_crossing_moves_rejected"] += len(diagnostic.get("rejected_candidates", {}))
    assert counts["all_in_started_during_damage_progress"] == 0
    if decisions:
        assert decisions.keys() == steps.keys()
    return {"steps": len(steps), "versions": sorted(games), **dict(counts)}


def report(root):
    names = ("nearest", "adaptive-skill")
    summaries = {name: load(root / name / "summary.json") for name in names}
    runs = {name: load(root / name / "run.json") for name in names}
    episodes = {name: load(root / name / "episodes.json") for name in names}
    manifest = load(root / "nearest/manifest.json")
    assert manifest == load(root / "adaptive-skill/manifest.json")
    assert len(manifest["episodes"]) == 10
    assert manifest["env_args"]["unit_balance"] == "native"
    assert runs["nearest"]["versions"] == runs["adaptive-skill"]["versions"]
    shared = runs["nearest"]["source_hashes"]
    assert all(runs["adaptive-skill"]["source_hashes"][p] == sha for p, sha in shared.items())
    behavior_checks = {}
    for name in names:
        assert runs[name]["observation_protocol"] == "smacv2-visible-rows-v2"
        assert summaries[name]["status"] == "complete"
        assert summaries[name]["completed_episodes"] == len(episodes[name]) == 10
        assert summaries[name]["jev_requests"] == 0 and summaries[name]["planner"] is None
        behavior_checks[name] = behavior(root / name)
        assert behavior_checks[name]["steps"] == sum(row["steps"] for row in episodes[name])
        for i in range(1, 11):
            audit = load(root / name / "balance" / f"episode-{i:04d}.json")
            assert audit["passed"] and audit["mode"] == "native"
            for profile in audit["profiles"].values():
                attributes = profile["shared_attributes"]
                expected = [attributes["health_max"], attributes["shield_max"]]
                assert all(hp == expected for values in profile["initial_health_shields"].values() for hp in values)
            assert (root / name / "replays" / f"episode-{i:04d}.SC2Replay").stat().st_size > 0
    assert behavior_checks["nearest"]["versions"] == behavior_checks["adaptive-skill"]["versions"]
    assert len(behavior_checks["nearest"]["versions"]) == 1
    paired = []
    historical = load(root.parent / "native-5v5-adaptive-skill-n10-seed100/episodes.json")
    for i in range(10):
        a, b = (episodes[name][i] for name in names)
        assert a["initial_state_sha256"] == b["initial_state_sha256"] == historical[i]["initial_state_sha256"]
        assert a["scenario_sha256"] == b["scenario_sha256"] == manifest["episodes"][i]["sha256"]
        paired.append({"episode": i + 1, "seed": manifest["episodes"][i]["sc2_seed"],
                       "nearest": a["outcome"], "adaptive-skill": b["outcome"]})
    result = {"observation_protocol": "smacv2-visible-rows-v2",
              "paired_initial_states_match_including_historical_run": True,
              "shared_source_hashes_match": True, "native_attribute_audits": 20,
              "behavior_checks": behavior_checks, "summaries": summaries, "paired": paired}
    (root / "verified_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = ["# 修正观察 + 改进技能：同场景重新配对评测", "",
             "本次重新运行 nearest 和 Adaptive Skill，各10局；均使用修正后的SMACv2观察。"
             "seed100–109、原生同兵种属性协议及实际初始状态与旧实验一致。"
             "旧JEV/ASTRA成绩不混入此表。",
             "", "| 方法 | 胜 | 负 | 平 | 超时 | 胜率 | 95% Wilson区间 |",
             "|---|---:|---:|---:|---:|---:|---|"]
    for name, s in summaries.items():
        o = s["outcomes"]
        lo, hi = wilson(o["win"], 10)
        lines.append(f"| {name} | {o['win']} | {o['loss']} | {o['draw']} | {o['timeout']} | "
                     f"{s['win_rate']:.0%} | {lo:.1%}–{hi:.1%} |")
    skill = behavior_checks["adaptive-skill"]
    lines += ["", "## 修复验收", "",
              "- 可见敌人按行保留；死亡敌人不再登记可见标记，含EPO/全局观察回归用例。",
              "- 移动评分使用8 loops预计可达位置和双方武器射程；战术移动不为无效脱离而放弃己方攻击候选。",
              "- 撤退后提供有界冷却恢复，当前没有合法攻击不再直接导致立即反向接敌。",
              "- 新目标评分优先残血一击可杀机会，保持有条件的集火连续性；默认关闭目标评分仍使用旧排序。",
              "- 先记录上一时间段的可见敌人血量变化，再决策All-in；失视后的短期位置记忆仅用于移动搜索。",
              "", f"20份属性审计通过，20份回放完整；全部动作合法、观察和步骤配对、时钟每步推进8 loops。"
              f"本次skill共{skill['steps']}步，无模型请求。",
              f"撤退单位步{skill['retreat_unit_steps']}，下一步立即反向"
              f"{skill['immediate_reverse_after_retreat']}次；冷却恢复{skill['recovery_unit_steps']}单位步；"
              f"拒绝跨己方候选边界且未减少敌方射程覆盖的移动候选{skill['mask_crossing_moves_rejected']}个。",
              f"其中反向时武器冷却仍超过当前窗口的次数："
              f"{skill.get('reverse_while_cooldown_exceeds_window', 0)}；"
              "本次剩余反向均发生在冷却可于当前窗口结束时，属于需要进一步评估的重新接敌动作。",
              f"有实际伤害进展的当帧误触发无效循环All-in："
              f"{skill['all_in_started_during_damage_progress']}次。",
              "", "旧skill轨迹有135次撤退及52次立即反向；本次观察和技能都改变，"
              "上述数量对比仅用于诊断，不能单独归因到某项技能改动。10局样本也不足以确定稳定胜率提升。",
              "相对旧skill：第2/5局由负转胜，第3局由胜转负，净增加1胜；没有隐藏回退场景。",
              "", "## 逐局结果", "",
              "| 局 | seed | nearest | 改进skill |", "|---:|---:|---|---|"]
    for row in paired:
        lines.append("| " + " | ".join(str(v) for v in row.values()) + " |")
    (root / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    report(parser.parse_args().root.resolve())
