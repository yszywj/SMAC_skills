"""Audit 40 frozen v6 games against both historical paired baselines."""
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path

from report_skill_improved import ROOT, audit, read
from report_skill_v5 import aggregate, new_diagnostics

OUT = ROOT / "analysis/skill_repairs_v6_20261023"
OLD_ANALYSIS = ROOT / "analysis/skill_repairs_v5_20261023/regression_analysis"


def load_analysis(name):
    spec = importlib.util.spec_from_file_location(name, OLD_ANALYSIS / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def engagement_audit(path):
    counts, reasons, fields = Counter(), Counter(), set()
    pending = None
    for event in map(json.loads, (path / "events.jsonl").open(encoding="utf-8")):
        if event["event"] == "step":
            assert pending is not None
            assert (pending["episode"], pending["step"] + 1) == (event["episode"], event["step"])
            for actor, diag in pending["engagement_diagnostics"].items():
                assert diag["selected_action"] == event["actions"][int(actor[1:])]
                assert diag["selected_role"] == pending["roles"][actor]
            for actor, target in pending["attack_assignments"].items():
                assert event["actions"][int(actor[1:])] == 6 + int(target[1:])
            pending = None
        if event["event"] != "skill_step":
            continue
        pending = event
        assert event["engagement_commitment_enabled"] is True
        counts["decision_steps"] += 1
        for actor, diag in event["engagement_diagnostics"].items():
            assert actor.startswith("u") and isinstance(diag, dict)
            fields.update(diag)
            counts["unit_evaluations"] += 1
            for key, value in diag.items():
                if isinstance(value, bool):
                    counts[key] += value
            reasons[str(diag.get("reason", "unspecified"))] += 1
    assert counts["unit_evaluations"] > 0
    summary = read(path / "summary.json")
    assert summary["skill"]["engagement_evaluations"] == counts["unit_evaluations"]
    for counter, field in (("shot_proxies", "shot_proxy"), ("cycles_detected", "cycle_detected"),
                           ("retreat_blocks", "retreat_blocked"), ("commitments_started", "commitment_started"),
                           ("commitments_fulfilled", "commitment_fulfilled"), ("commitments_timed_out", "commitment_timed_out"),
                           ("danger_overrides", "danger_override"), ("repeated_moves_blocked", "repeated_move_blocked")):
        assert summary["skill"]["engagement_" + counter] == counts[field], counter
    return {"counts": dict(counts), "reasons": dict(reasons), "diagnostic_fields": sorted(fields),
            "summary": {k: v for k, v in summary["skill"].items() if k.startswith("engagement_")}}


def write_report(data):
    rows = data["runs"]
    lines = ["# Skill v6：跨角色撤退与开火承诺修复及 40 局测试", "",
             "2026-10-23。纯本地 AdaptiveFocusFire；5v5、10v10 各 20 局，均使用种子 150–169。"
             "SMACv2 `10gen_protoss`，同兵种双方采用相同原生属性，敌方为难度 7 内置 AI，"
             "每个决策推进 8 loops，策略仅使用己方共享可见信息。本轮没有模型调用。", "",
             "40 局初始状态哈希逐局匹配历史 v4、v5；地图、环境代码和攻击掩码一致。"
             "本轮只使用一份冻结的 v6 代码，没有在对局中途调整参数。", "",
             "| 场景 | 版本 | 敌方平均消灭比例 | 胜率（胜场／总局数） | 平均步数 |",
             "|---|---|---:|---:|---:|"]
    for row in rows:
        for version, value in [(v, row["previous"][v]) for v in ("v4", "v5")] + [("v6（本轮）", row)]:
            lines.append(f"| {row['size']}v{row['size']} | {version} | {value['mean_enemy_elimination']:.1%} | "
                         f"{value['wins']/value['total']:.0%}（{value['wins']}／{value['total']}） | {value['mean_steps']:.2f} |")
    lines += ["", "平均消灭比例和平均步数均包含每组全部 20 局；没有把失败对局剔除。"
              "10v10 的结果为 10 胜、9 负、1 平；第 11 局双方同时全灭，计为平局，不计胜场。", "",
              "本轮改善了 v5 的长循环问题，但未实现整体胜率提升：5v5 比 v5 少 1 胜，消灭比例不变；"
              "10v10 增加 1 胜、1 平，消灭比例提高 2.5 个百分点，仍低于 v4 的 12/20 胜、89%。"
              "因此 v6 是完成本轮机制修复的实验版本，不能直接称为当前最佳策略。", "",
              "## 修改", "",
              "- 撤退、恢复和重新接敌共享位置历史，检查实际 A→B→A 往返；后置改向同步到导航历史，避免下步错误地惩罚未执行方向。",
              "- 个人撤退额度跨角色累计，队友造成伤害不再替本人清账。攻击指令只代表意图；本人冷却重新启动才作为开火进展的观测代理。",
              "- 返回合法攻击范围后给予最多 3 个决策窗口的开火承诺，仍通过原集火分配器攻击；新增的实际受击或威胁允许有限紧急脱离。",
              "- 承诺超时后退出强制约束并重新选路；失去目标、连续失视、死亡或新一局清理相应状态。单个单位的循环不再单独触发全队强攻。",
              "- 增加独立开关 `--adaptive-engagement-commitment`，CLI 默认关闭，v6 包装配置显式开启；v5 的三项机制继续启用。", "",
              "## 实际轨迹核对", "",
              "下表统计冷却为 0、同时满足原生射程和 SMAC 合法攻击掩码时，最终仍选择移动的单位步。"
              "它衡量开火机会被移动占用的频率，不是实际丢失的命中数；紧急脱离也会计入。", "",
              "| 场景 | 版本 | 仍移动／可开火机会 | 比例 |", "|---|---|---:|---:|"]
    for row in rows:
        for version in ("v4", "v5", "v6"):
            c = row["action_behavior"][version]["counts"]
            numerator, denominator = c.get("moving_ready_now_and_legal", 0), c["ready_now_and_legal"]
            lines.append(f"| {row['size']}v{row['size']} | {version} | {numerator}／{denominator} | {numerator/denominator:.2%} |")
    lines += ["", "10v10 的移动循环：相邻反向串指连续至少 4 个移动命令，且每一步都与上一步方向相反；"
              "位置往返指当前位置与两步前相距不超过 0.2、与上一步超过 0.4，前两条命令均为移动。", "",
              "| 版本 | 观察到的 A→B→A 次数 | 至少 4 窗反向串 | 至少 8 窗反向串 | 最长反向串 |",
              "|---|---:|---:|---:|---:|"]
    for version, x in data["movement_10v10"].items():
        runs = x["opposite_move_runs_at_least4"]
        lines.append(f"| {version} | {x['observed_A_B_A_position_roundtrips']} | {len(runs)} | "
                     f"{sum(r['windows'] >= 8 for r in runs)} | {max((r['windows'] for r in runs), default=0)} |")
    lines += ["", "机制触发和预算计数保存在 `verified_results.json` 及各局 `events.jsonl`。"
              "`engagement.applied` 表示局部机制采用，不一定代表相对关闭开关后的最终动作差异；"
              "开火代理也不证明具体命中目标或伤害数。固定旧观测的回归检查不是反事实对局，不提供胜率。", "",
              "重点案例：10v10 第 17 局由失败变为胜利，第 11 局由失败变为双方全灭的平局；"
              "5v5 第 14 局由胜利变为失败。"
              "详细见 [10v10 独立轨迹核对](INDEPENDENT_AUDIT.md)、"
              "[5v5 配对分析](FIVE_V5_V6_ANALYSIS.md) 和 [历史序列回归说明](OFFLINE_REGRESSION.md)。", "",
              "## 验收", "",
              f"冻结前通过 {data['validation']['skill-tests.txt']['tests']} 项技能测试、"
              f"{data['validation']['adapter-tests.txt']['tests']} 项适配器测试；关闭新开关在历史 "
              f"{data['validation']['recorded_off_compatibility_frames']:,} 帧中完全复现旧动作。"
              f"开启新开关对 {data['validation']['recorded_on_legality_frames']:,} 帧检查动作合法、诊断可序列化且不修改输入。",
              "两组使用相同的 42 文件运行快照及参数；40 局全部完成，属性一致性审计、回放、动作合法性、"
              "动作提交反馈和实际 8 loops 推进均通过核对。原有实验文件未覆盖。", "",
              "这是固定 20 个种子的样本比较，可用于检查已知退化案例；它不是独立留出的测试集，"
              "不能由本批胜率宣称稳定泛化收益，也不能把整局胜负归因到某一个机制。", "",
              "## 使用方法", "", "在 `C:\\Users\\ry\\skill\\smacv2_jev_adapter` 执行，另取输出名称避免覆盖：", "",
              "```powershell", ".\\scripts\\run_improved_skill.ps1 -Size 5 -PolicyVersion v6 -Episodes 20 -Seed 150 -OutputName repeat-v6-5v5",
              ".\\scripts\\run_improved_skill.ps1 -Size 10 -PolicyVersion v6 -Episodes 20 -Seed 150 -OutputName repeat-v6-10v10", "```", ""]
    for row in rows:
        lines += [f"## {row['size']}v{row['size']} 逐局结果", "",
                  "| 局 | 种子 | v4 胜负 | v5 胜负 | v6 胜负 | v6 消灭比例 | v6 步数 |",
                  "|---:|---:|---|---|---|---:|---:|"]
        for p in row["paired_episodes"]:
            lines.append(f"| {p['episode']} | {p['seed']} | {p['v4']['outcome']} | {p['v5']['outcome']} | "
                         f"{p['v6']['outcome']} | {p['v6']['enemy_elimination_fraction']:.0%} | {p['v6']['steps']} |")
        lines.append("")
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    validation = read(OUT / "validation.json")
    assert all(validation[key]["passed"] for key in ("skill-tests.txt", "adapter-tests.txt"))
    offline = read(OUT / "offline_checks.json")
    off = [r for r in offline["results"] if r["case"] == "engagement_off"]
    assert len(off) == 4 and all(r["changed_unit_actions"] == 0 for r in off)
    frozen = read(OUT / "source_snapshot/snapshot.json")["source_hashes"]
    for name, sha in frozen.items():
        assert hashlib.sha256((OUT / "source_snapshot" / name).read_bytes()).hexdigest() == sha
    behavior = load_analysis("aggregate_behavior")
    movement = load_analysis("movement_loop_audit")
    rows, metas = [], []
    for size in (5, 10):
        current = ROOT / f"runs/native-{size}v{size}-v6-skill-n20-seed150"
        row, meta = audit(current, size, 150)
        assert meta["source_hashes"] == frozen
        assert all(meta["skill_parameters"][k] for k in (
            "accurate_damage", "critical_escape", "planned_reentry", "engagement_commitment"))
        row["previous"], row["transitions"], row["paired_episodes"] = {}, {}, []
        histories = {}
        for version, snapshot in (
            ("v4", ROOT / "analysis/skill_improvements_20261023/source_snapshot_final"),
            ("v5", ROOT / "analysis/skill_repairs_v5_20261023/source_snapshot"),
        ):
            old_path = ROOT / f"runs/native-{size}v{size}-{version}-skill-n20-seed150"
            old, old_meta = read(old_path / "episodes.json"), read(old_path / "run.json")
            assert read(current / "manifest.json") == read(old_path / "manifest.json")
            assert meta["versions"] == old_meta["versions"]
            for name, sha in old_meta["source_hashes"].items():
                assert hashlib.sha256((snapshot / name).read_bytes()).hexdigest() == sha
                if name.startswith("smacv2\\") or name.endswith("bridge\\environment.py"):
                    assert meta["source_hashes"][name] == sha, "Environment changed: " + name
            row["previous"][version] = aggregate(old)
            histories[version] = old
            changes = Counter()
            for a, b in zip(old, row["episodes"]):
                assert a["episode"] == b["episode"]
                assert a["initial_state_sha256"] == b["initial_state_sha256"]
                assert a["scenario_sha256"] == b["scenario_sha256"]
                changes[a["outcome"] + "->" + b["outcome"]] += 1
            row["transitions"][version] = dict(changes)
        for index, episode in enumerate(row["episodes"]):
            pair = {"episode": episode["episode"], "seed": 150 + index}
            for version, item in [(v, es[index]) for v, es in histories.items()] + [("v6", episode)]:
                pair[version] = {k: item[k] for k in ("outcome", "enemy_elimination_fraction", "steps")}
            row["paired_episodes"].append(pair)
        row["diagnostics"] = new_diagnostics(current)
        assert row["diagnostics"]["counts"].get("observation_errors", 0) == 0
        assert row["diagnostics"]["counts"].get("unmatched_observation_errors", 0) == 0
        row["engagement"] = engagement_audit(current)
        row["action_behavior"] = {v: behavior.analyze(size, v) for v in ("v4", "v5", "v6")}
        row["all_initial_states_match"] = True
        rows.append(row)
        metas.append(meta)
    for key in ("source_hashes", "skill_parameters", "unit_value_parameters", "versions"):
        assert metas[0][key] == metas[1][key], key
    loops = {}
    for version in ("v4", "v5", "v6"):
        result = movement.summarize(movement.load(version))
        result.pop("exact_ready_legal_movement_examples")
        loops[version] = result
    data = {"runs": rows, "same_initial_states": True, "same_environment": True,
            "source_hashes": frozen, "skill_parameters": metas[0]["skill_parameters"],
            "validation": validation, "offline_checks": offline, "movement_10v10": loops,
            "scope": "40 new v6 games, fixed seeds 150-169; historical v4/v5 paired baselines. "
                     "No tuning during evaluation; no per-feature causal win-rate claim."}
    (OUT / "verified_results.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(data)
    for row in rows:
        print(json.dumps({k: row[k] for k in ("size", "wins", "total", "mean_enemy_elimination", "mean_steps", "transitions", "engagement")}, ensure_ascii=False))
    print(json.dumps({v: {"position_roundtrips": x["observed_A_B_A_position_roundtrips"],
                          "opposite_runs_4": len(x["opposite_move_runs_at_least4"]),
                          "longest_opposite_run": max((a["windows"] for a in x["opposite_move_runs_at_least4"]), default=0)}
                      for v, x in loops.items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
