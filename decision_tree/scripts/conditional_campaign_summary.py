"""Summarize completed conditional campaigns from native episode records.

Writes campaign_summary.json and results.md only. Never runs games or changes
policies. Incomplete rounds contribute status only, even with --partial.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics

from .conditional_trace_summary import summarize_run


PARTITIONS = ("train", "holdout", "all")
STRATEGIES = ("adaptive", "incumbent", "candidate")
SCOPE_NOTE = ("Each round evaluates its own frozen candidate against that round's incumbent and adaptive. "
              "Across-round totals mix different sequentially trained policies. Training scenes were used to fit candidates. "
              "The sequential holdouts are not an unbiased 120-game test of one final model; "
              "the final adopted policy has not been replayed across all campaign scenes.")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def outcomes(rows):
    counts = Counter(row["outcome"] for row in rows)
    return {"scenes": len(rows), "wins": counts["win"], "outcomes": dict(counts),
            "win_rate": counts["win"] / len(rows) if rows else None,
            "mean_return": statistics.mean(row["reward"] for row in rows) if rows else None}


def paired(candidate, reference, scenes):
    require(set(candidate) == set(reference), "Pair cohorts differ")
    rows, counts = [], Counter()
    for group in sorted(candidate):
        a, b = candidate[group], reference[group]
        require(a["initial_state_sha256"] == b["initial_state_sha256"], "Paired initial state changed: " + group)
        win_gain = int(a["outcome"] == "win") - int(b["outcome"] == "win")
        category = "rescued" if win_gain > 0 else "lost" if win_gain < 0 else "both_win" if a["outcome"] == "win" else "both_nonwin"
        counts[category] += 1
        rows.append({"scenario_sha256": group, "seed": scenes[group]["sc2_seed"],
                     "category": category, "candidate_outcome": a["outcome"], "reference_outcome": b["outcome"],
                     "candidate_return": a["reward"], "reference_return": b["reward"],
                     "win_gain": win_gain, "return_gain": a["reward"] - b["reward"]})
    return {"scenes": len(rows), "counts": dict(counts),
            "mean_win_gain": statistics.mean(row["win_gain"] for row in rows) if rows else None,
            "mean_return_gain": statistics.mean(row["return_gain"] for row in rows) if rows else None,
            "rescued_seeds": [row["seed"] for row in rows if row["category"] == "rescued"],
            "lost_seeds": [row["seed"] for row in rows if row["category"] == "lost"], "pairs": rows}


def load_run(path, expected_groups, expected_policy_hash=None):
    path = Path(path).resolve()
    require(read(path / "summary.json").get("status") == "complete", "Native run is incomplete: " + str(path))
    rows = read(path / "episodes.json")
    groups = [row["scenario_sha256"] for row in rows]
    require(len(set(groups)) == len(groups) and set(groups) == set(expected_groups), "Native episode cohort mismatch: " + str(path))
    for row in rows:
        require(row["outcome"] in ("win", "loss", "draw", "timeout") and
                type(row["reward"]) in (int, float) and math.isfinite(row["reward"]), "Invalid terminal result")
    header = read(path / "run.json")
    if expected_policy_hash:
        require(header.get("conditional_skills", {}).get("policy_sha256") == expected_policy_hash,
                "Native run used a different policy: " + str(path))
    return {row["scenario_sha256"]: row for row in rows}


def verify_saved_summary(actual, saved):
    require(saved.get("complete") and saved["matched_scenes"] == actual["scenes"], "Saved evaluation cohort mismatch")
    for key in ("mean_win_gain", "mean_return_gain"):
        require(math.isclose(actual[key], saved[key], rel_tol=0, abs_tol=1e-9), "Saved evaluation differs from episodes: " + key)


def completed_round(root, index):
    folder = root / f"round_{index}"
    evaluation = read(folder / "evaluation.json")
    freeze = read(folder / "freeze.json")
    split = read(folder / "split.json")
    manifest = read(folder / "manifest.json")
    scenes = {s["sha256"]: s for s in manifest["episodes"]}
    groups = {"train": set(split["training_group_ids"]), "holdout": set(split["validation_group_ids"]), "all": set(scenes)}
    require(not groups["train"] & groups["holdout"] and groups["train"] | groups["holdout"] == groups["all"], "Invalid scene split")
    require(len(scenes) == len(manifest["episodes"]), "Duplicate manifest scenes")
    policies = {name: read(folder / f"{name}_policy.json") for name in ("incumbent", "candidate", "adopted")}
    hashes = {name: digest(value) for name, value in policies.items()}
    require(freeze["candidate_sha256"] == hashes["candidate"], "Candidate changed since freeze")
    require(hashes["adopted"] == hashes["candidate" if evaluation["adopted"] else "incumbent"], "Adopted policy differs from saved verdict")
    run_paths = read(folder / "run_paths.json")
    paths = {name: Path(run_paths[name]).resolve() for name in ("adaptive", "incumbent")}
    paths.update({name: Path(evaluation[name]).resolve() for name in ("candidate_train", "candidate_holdout")})
    actual = {"adaptive": load_run(paths["adaptive"], groups["all"]),
              "incumbent": load_run(paths["incumbent"], groups["all"], hashes["incumbent"]),
              "candidate": {**load_run(paths["candidate_train"], groups["train"], hashes["candidate"]),
                            **load_run(paths["candidate_holdout"], groups["holdout"], hashes["candidate"])}}
    partitions = {}
    for partition, members in groups.items():
        selected = {name: {g: row for g, row in actual[name].items() if g in members} for name in STRATEGIES}
        comparisons = {name: paired(selected[a], selected[b], scenes) for name, a, b in (
            ("candidate_vs_incumbent", "candidate", "incumbent"),
            ("candidate_vs_adaptive", "candidate", "adaptive"),
            ("incumbent_vs_adaptive", "incumbent", "adaptive"))}
        partitions[partition] = {"strategies": {name: outcomes(list(rows.values())) for name, rows in selected.items()},
                                 "comparisons": comparisons}
    verify_saved_summary(partitions["train"]["comparisons"]["candidate_vs_incumbent"], evaluation["training"])
    verify_saved_summary(partitions["holdout"]["comparisons"]["candidate_vs_incumbent"], evaluation["holdout_vs_incumbent"])
    verify_saved_summary(partitions["holdout"]["comparisons"]["candidate_vs_adaptive"], evaluation["holdout_vs_adaptive"])
    plan, measured = read(folder / "branch_plan.json"), read(folder / "measured.json")
    require(len(plan["jobs"]) == len(measured), "Counterfactual collection is incomplete")
    require(all(row["scene_hash"] in groups["train"] for row in measured), "Holdout label in measured training data")
    require(freeze["training_sha256"] == digest(measured) and freeze["split_sha256"] == digest(split), "Training evidence changed after freeze")
    trace = {name: summarize_run(path) for name, path in paths.items()}
    require(all(not row["counts"].get("anomalies", 0) and not row["counts"].get("missing_changed_command_receipts", 0)
                for row in trace.values()), "Trace statistics found a command/counter anomaly")
    return {"round": index, "adopted": evaluation["adopted"], "partitions": partitions,
            "policy_sha256": hashes, "policy_layers": {name: len(p.get("corrections", [])) for name, p in policies.items()},
            "actual_run_paths": {name: str(path) for name, path in paths.items()},
            "branch_games": len(measured),
            "branch_families": dict(Counter(row["family"] for row in measured)),
            "branch_families_and_labels": dict(Counter(row["family"] + ":" + row["label_kind"] for row in measured)),
            "scene_family_coverage": dict(Counter(row["family"] + ":" + row["status"] for row in plan["coverage"])),
            "completed_native_games": 3 * len(scenes) + len(measured),
            "trace": trace, "evaluation_path": str(folder / "evaluation.json")}


def aggregate(rounds):
    totals = {}
    for partition in PARTITIONS:
        strategies = {}
        for name in STRATEGIES:
            records = [r["partitions"][partition]["strategies"][name] for r in rounds]
            count = sum(r["scenes"] for r in records)
            wins = sum(r["wins"] for r in records)
            strategies[name] = {"scenes": count, "wins": wins, "win_rate": wins / count if count else None,
                                "mean_return": sum(r["mean_return"] * r["scenes"] for r in records) / count if count else None}
        comparisons = {}
        for name in ("candidate_vs_incumbent", "candidate_vs_adaptive", "incumbent_vs_adaptive"):
            records = [r["partitions"][partition]["comparisons"][name] for r in rounds]
            count = sum(r["scenes"] for r in records)
            comparisons[name] = {"scenes": count,
                "mean_win_gain": sum(r["mean_win_gain"] * r["scenes"] for r in records) / count if count else None,
                "mean_return_gain": sum(r["mean_return_gain"] * r["scenes"] for r in records) / count if count else None,
                "rescued_seeds": [s for r in records for s in r["rescued_seeds"]],
                "lost_seeds": [s for r in records for s in r["lost_seeds"]]}
        totals[partition] = {"strategies": strategies, "comparisons": comparisons}
    return {"kind": "descriptive_sequential_policy_totals", "interpretation": SCOPE_NOTE,
            "partitions": totals, "branch_games": sum(r["branch_games"] for r in rounds),
            "completed_native_games": sum(r["completed_native_games"] for r in rounds)}


def build_summary(root, partial=False):
    root = Path(root).resolve()
    protocol = read(root / "protocol.json")
    require(protocol.get("kind") == "conditional_v3_bounded_3x40", "Not a conditional v3 campaign")
    indices = [r["round"] for r in protocol["rounds"]]
    require(len(indices) == len(set(indices)), "Duplicate protocol round IDs")
    completed, unfinished = [], []
    for index in indices:
        folder = root / f"round_{index}"
        if (folder / "evaluation.json").exists() and (folder / "adopted_policy.json").exists():
            completed.append(index)
        else:
            unfinished.append({"round": index, "status": "not_complete", "included_in_results": False,
                               "evaluation_exists": (folder / "evaluation.json").exists(),
                               "adopted_policy_exists": (folder / "adopted_policy.json").exists()})
    require(partial or not unfinished, "Campaign is incomplete; use --partial to summarize completed rounds only")
    rounds = [completed_round(root, index) for index in completed]
    scene_ids = [row["scenario_sha256"] for r in rounds for row in r["partitions"]["all"]["comparisons"]["candidate_vs_incumbent"]["pairs"]]
    require(len(scene_ids) == len(set(scene_ids)), "Scene reused across campaign rounds")
    return {"kind": "conditional_campaign_actual_episode_summary", "created_utc": datetime.now(timezone.utc).isoformat(),
            "root": str(root), "status": "partial" if unfinished else "complete",
            "expected_rounds": indices, "completed_rounds": completed, "unfinished_rounds": unfinished,
            "interpretation": SCOPE_NOTE, "rounds": rounds, "sequential_totals": aggregate(rounds)}


def markdown(summary):
    lines = ["# 条件技能训练结果", "", f"状态：{'已完成' if summary['status'] == 'complete' else '部分完成'}；已纳入轮次："
             + ("、".join(map(str, summary["completed_rounds"])) or "无") + "。", "",
             "以下结果直接来自实际对局记录。每轮候选都先冻结，再进行整局验证；未完成轮次不计入结果。", "",
             "**统计范围：各轮使用不同的顺序训练候选。合计属于训练流程的描述统计，不代表最终单一模型完成了120局无偏测试。训练组参与拟合；每轮10场留出也属于顺序验证。**", "",
             "| 轮次 | 场景组 | adaptive 胜局 / 均回报 | 旧策略 胜局 / 均回报 | 候选 胜局 / 均回报 | 采用候选 |",
             "|---|---|---|---|---|---|"]
    labels = {"train": "训练", "holdout": "留出", "all": "全部"}
    for record in summary["rounds"]:
        for part in PARTITIONS:
            stats = record["partitions"][part]["strategies"]
            values = [f"{stats[k]['wins']}/{stats[k]['scenes']}（{stats[k]['win_rate']:.1%}） / {stats[k]['mean_return']:.4f}" for k in STRATEGIES]
            lines.append(f"| {record['round']} | {labels[part]} | " + " | ".join(values) + f" | {'是' if record['adopted'] else '否'} |")
    for record in summary["rounds"]:
        lines += ["", f"## 第{record['round']}轮", "", f"反事实分支：{record['branch_games']}；按家族："
                  + "，".join(f"{k} {v}" for k, v in sorted(record["branch_families"].items())) + "。",
                  f"纠错层数：旧策略 {record['policy_layers']['incumbent']} → 候选 {record['policy_layers']['candidate']}；最终保留 {record['policy_layers']['adopted']} 层。", ""]
        for part in PARTITIONS:
            for reference, label in (("incumbent", "旧策略"), ("adaptive", "adaptive")):
                p = record["partitions"][part]["comparisons"]["candidate_vs_" + reference]
                seeds = lambda values: ", ".join(map(str, values)) or "无"
                lines.append(f"- {labels[part]}组相对{label}：救回 {len(p['rescued_seeds'])} 场（seed：{seeds(p['rescued_seeds'])}）；"
                             f"丢掉 {len(p['lost_seeds'])} 场（seed：{seeds(p['lost_seeds'])}）；均回报差 {p['mean_return_gain']:+.4f}。")
        lines += ["", "| 实际策略运行 | 干预提交 | 原生接受 | 层内撤销选择 | 多层改写步 | 下一步恢复原动作 ID |", "|---|---:|---:|---:|---:|---:|"]
        for name in ("incumbent", "candidate_train", "candidate_holdout"):
            trace = record["trace"][name]
            c = trace["counts"]
            cancellations = sum(layer.get("cancelled_to_BASE", 0) for layer in trace["correction_layers"].values())
            lines.append(f"| {name} | {c.get('changed_command_submissions', 0)} | {c.get('changed_command_receipt_accepted', 0)} | {cancellations} | "
                         f"{c.get('steps_with_multiple_layer_changes', 0)} | {c.get('next_step_same_actor_restored_prior_baseline_action_id', 0)} |")
        lines += ["", "层内改写只是在选择阶段修改候选，最终最多提交一个改单兵命令；恢复原动作 ID 不自动等于无效干预。"]
    if summary["rounds"]:
        total = summary["sequential_totals"]
        lines += ["", "## 顺序训练汇总", "", f"已完成原生对局 {total['completed_native_games']} 局，其中反事实分支 {total['branch_games']} 局；重试预算以原生任务台账为准。", "",
                  "| 场景组 | adaptive 胜局 / 均回报 | 各轮旧策略 胜局 / 均回报 | 各轮候选 胜局 / 均回报 |", "|---|---|---|---|"]
        for part in PARTITIONS:
            stats = total["partitions"][part]["strategies"]
            values = [f"{stats[k]['wins']}/{stats[k]['scenes']}（{stats[k]['win_rate']:.1%}） / {stats[k]['mean_return']:.4f}" for k in STRATEGIES]
            lines.append(f"| {labels[part]} | " + " | ".join(values) + " |")
    if summary["unfinished_rounds"]:
        lines += ["", "未完成、未纳入结果的轮次：" + "、".join(str(r["round"]) for r in summary["unfinished_rounds"]) + "。"]
    return "\n".join(lines) + "\n"


def write_report(root, summary):
    root = Path(root).resolve()
    for name, content in (("campaign_summary.json", json.dumps(summary, ensure_ascii=False, indent=2) + "\n"),
                          ("results.md", markdown(summary))):
        destination = root / name
        temporary = root / (name + ".tmp")
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--partial", action="store_true", help="Include completed rounds only, explicitly label remaining rounds unfinished.")
    args = parser.parse_args()
    summary = build_summary(args.root, args.partial)
    write_report(args.root, summary)
    print(json.dumps({"status": summary["status"], "completed_rounds": summary["completed_rounds"],
                      "outputs": [str(args.root.resolve() / name) for name in ("campaign_summary.json", "results.md")]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
