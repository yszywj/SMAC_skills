"""Validate and report the frozen five-arm paired experiment; no game execution."""
from collections import Counter, defaultdict
import json
import hashlib
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis/spatial_ablation_20261025"
GROUPS = ("off", "regroup", "search", "positioning", "all")
LABELS = {"off": "新技能全关", "regroup": "仅集结", "search": "仅协同搜索",
          "positioning": "仅战斗站位", "all": "三项全开"}


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def wilson(wins, n):
    z = 1.959963984540054
    p, denominator = wins / n, 1 + z*z/n
    middle = (p + z*z/(2*n)) / denominator
    half = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denominator
    return [middle-half, middle+half]


def paired(base, other):
    gained, lost = [], []
    for b, o in zip(base, other):
        assert b["episode"] == o["episode"]
        assert b["scenario_sha256"] == o["scenario_sha256"]
        assert b["initial_state_sha256"] == o["initial_state_sha256"], "Real initial states differ"
        if b["outcome"] != "win" and o["outcome"] == "win":
            gained.append(o["episode"])
        if b["outcome"] == "win" and o["outcome"] != "win":
            lost.append(o["episode"])
    discordant = len(gained) + len(lost)
    probability = min(1., 2*sum(math.comb(discordant, k) for k in range(min(len(gained), len(lost))+1))
                      / 2**discordant) if discordant else 1.
    return {"nonwin_to_win": gained, "win_to_nonwin": lost, "exact_p": probability,
            "wins_delta": len(gained)-len(lost)}


def mechanisms(path, size):
    counts, statuses = Counter(), Counter()
    activated = defaultdict(set)
    episode_first_attack = {}
    first_observations = {}
    current, telemetry = None, None
    split_steps = 0
    # Import actual frozen-checked geometry without mixing environment packages.
    import sys
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from bridge.skill_runtime import _load_module
    geometry = _load_module("skills.group_geometry")
    with (path/"events.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            e = json.loads(line)
            event = e["event"]
            if event == "skill_observation":
                current, telemetry = e["payload"], None
                first_observations.setdefault(e["episode"], hashlib.sha256(
                    json.dumps(current, sort_keys=True, ensure_ascii=True).encode()).hexdigest())
                report = geometry.cohesion_report(current["state"])
                if len(report["groups"]) > 1:
                    split_steps += 1
            elif event == "skill_step":
                telemetry = e
                skill = e["skill"]
                if skill in ("regroup", "cohesive_search"):
                    counts[skill+"_calls"] += 1
                    statuses[skill+":"+str(e.get("status"))] += 1
                    controlled = e.get("controlled_units", [])
                    counts[skill+"_unit_steps"] += len(controlled)
                    if controlled:
                        activated[skill].add(e["episode"])
                rows = e.get("approach_diagnostics", {})
                positioning = [row for row in rows.values() if "selected_cluster" in row]
                counts["positioning_unit_steps"] += len(positioning)
                if positioning:
                    activated["positioning"].add(e["episode"])
            elif event == "step":
                actions = e["actions"]
                assert isinstance(actions, list) and len(actions) == size
                assert current is not None
                for actor, question in current["questions"].items():
                    index = int(actor[1:])
                    assert str(actions[index]) in question["criteria"], (path, e["episode"], e["step"], actor)
                counts["validated_decisions"] += 1
                if any(action >= 6 for action in actions):
                    episode_first_attack.setdefault(e["episode"], e["step"]-1)
                feedback = e.get("feedback") or {}
                # Keep a count of explicit backend rejections when this schema provides them.
                counts["feedback_records"] += bool(feedback)
                for actor_feedback in feedback.get("actors", {}).values():
                    counts["rejected_commands"] += actor_feedback.get("submission") == "rejected"
                    counts["observation_errors"] += len(actor_feedback.get("observation_errors", []))
                counts["observation_errors"] += len(feedback.get("unmatched_observation_errors", []))
    return {**counts, "split_observation_steps": split_steps,
            "active_episodes": {k: len(activated[k]) for k in ("regroup", "cohesive_search", "positioning")},
            "active_episode_ids": {k: sorted(activated[k]) for k in ("regroup", "cohesive_search", "positioning")},
            "status_counts": dict(statuses), "first_attack_order_steps": episode_first_attack,
            "first_observation_sha256": first_observations,
            "mean_first_attack_order_step": statistics.mean(episode_first_attack.values())
                if episode_first_attack else None}


def action_traces(path):
    traces = defaultdict(list)
    state_hash, skill, positioning, hint, visible_count = None, None, [], None, 0
    with (path/"events.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            e = json.loads(line)
            if e["event"] == "skill_observation":
                state = e["payload"]["state"]
                visible = {key: {actor: {k: unit.get(k) for k in
                    ("alive", "type", "position", "health", "shield", "weapon_cooldown_loops")}
                    for actor, unit in state[key].items()} for key in ("allies", "visible_enemies")}
                state_hash = hashlib.sha256(json.dumps(visible, sort_keys=True).encode()).hexdigest()
                skill, positioning, hint, visible_count = None, [], None, len(state["visible_enemies"])
            elif e["event"] == "skill_step":
                skill = e["skill"]
                hint = e.get("search_hint")
                positioning = [actor for actor, row in e.get("approach_diagnostics", {}).items()
                               if "selected_cluster" in row]
            elif e["event"] == "step":
                traces[e["episode"]].append({"step": e["step"]-1, "actions": e["actions"],
                    "visible_state_sha256": state_hash, "skill": skill, "positioning_units": positioning,
                    "search_hint": hint, "visible_enemy_count": visible_count})
    return traces


def compare_traces(base, other):
    differences = {}
    for episode, rows in base.items():
        compared = other[episode]
        for b, o in zip(rows, compared):
            if b["actions"] != o["actions"]:
                differences[episode] = {"step": b["step"], "baseline_actions": b["actions"],
                    "variant_actions": o["actions"], "variant_skill": o["skill"],
                    "positioning_units": o["positioning_units"],
                    "baseline_search_hint": b["search_hint"], "variant_search_hint": o["search_hint"],
                    "visible_enemy_count": o["visible_enemy_count"],
                    "same_visible_state": b["visible_state_sha256"] == o["visible_state_sha256"]}
                break
        else:
            assert len(rows) == len(compared), "Identical action prefix but different termination"
    return {"identical_action_episodes": sorted(set(base)-set(differences)), "first_differences": differences}


def main():
    frozen = load(OUT/"frozen_hashes.json")
    data = {}
    traces = {}
    for size in (5, 10):
        data[size] = {}
        reference_params = reference_manifest = reference_versions = None
        for group in GROUPS:
            path = ROOT/"runs"/f"spatial-20261025-{group}-{size}v{size}-n20-seed210"
            summary, meta, episodes = load(path/"summary.json"), load(path/"run.json"), load(path/"episodes.json")
            assert summary["status"] == "complete" and summary["completed_episodes"] == len(episodes) == 20
            assert summary["jev_requests"] == 0
            for name, value in meta["source_hashes"].items():
                assert frozen[name.replace("\\", "/")] == value, (group, name)
            params = dict(meta["skill_parameters"])
            spatial = params.pop("spatial_params")
            expected = {"enable_regroup": group in ("regroup", "all"),
                        "enable_cohesive_search": group in ("search", "all"),
                        "enable_combat_positioning": group in ("positioning", "all"),
                        "auto_spatial_dispatch": group in ("regroup", "search", "all")}
            assert all(bool(spatial.get(k, False)) == v for k, v in expected.items()), (group, spatial)
            if reference_params is None:
                reference_params, reference_manifest, reference_versions = params, meta["manifest_sha256"], meta["versions"]
            assert params == reference_params and meta["manifest_sha256"] == reference_manifest
            assert meta["versions"] == reference_versions
            wins = summary["outcomes"]["win"]
            row = {"group": group, "size": size, "wins": wins, "episodes": episodes,
                   "outcomes": summary["outcomes"], "wilson95": wilson(wins, 20),
                   "mean_enemy_elimination": summary["mean_enemy_elimination"],
                   "mean_steps": statistics.mean(e["steps"] for e in episodes),
                   "mean_allied_survival": statistics.mean(e["allied_survival_fraction"] for e in episodes),
                   "mechanisms": mechanisms(path, size), "run_directory": str(path),
                   "manifest_sha256": meta["manifest_sha256"], "spatial_parameters": spatial}
            traces[(size, group)] = action_traces(path)
            if group != "off":
                row["paired"] = paired(data[size]["off"]["episodes"], episodes)
                assert row["mechanisms"]["first_observation_sha256"] == data[size]["off"]["mechanisms"]["first_observation_sha256"]
                row["trace_comparison"] = compare_traces(traces[(size, "off")], traces[(size, group)])
            data[size][group] = row
    (OUT/"results.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 新空间技能：200局配对对照结果", "",
        "每个配置20局5v5、20局10v10，共200局。共同基线为修复后的current上v7战斗配置；",
        "地图10gen_protoss，native属性，难度7，种子210–229。全关仅关闭新增三项空间技能。", "",
        "本批没有出现任何一组相对全关的净胜场提升：集结持平；协同搜索在10v10少2胜；",
        "战斗站位在5v5少2胜、10v10少4胜；三项全开分别少1胜、少4胜。搜索和站位当前配置不适合作为默认增强。",
        "保留全关基线与默认关闭设置；优先分析搜索开局调位和站位对接敌/集火的影响。集结触发样本很少，不能由持平推断普遍有效或无效。",
        "每种规模只有20个配对场景，区间较宽且精确配对检验均未达到0.05；这里的退步是本批观察结果，不是稳定胜率差异的定论。", "",
        "## 结果", "", "| 规模 | 配置 | 胜场 | 胜率95%区间 | 平均消灭 | 平均步数 | 平均己方存活 |",
        "|---|---|---:|---|---:|---:|---:|"]
    for size, groups in data.items():
        for group, row in groups.items():
            lo, hi = row["wilson95"]
            lines.append(f"| {size}v{size} | {LABELS[group]} | {row['wins']}/20 | {lo:.1%}–{hi:.1%} | "
                         f"{row['mean_enemy_elimination']:.1%} | {row['mean_steps']:.2f} | {row['mean_allied_survival']:.1%} |")
    lines += ["", "## 相对全关的逐局变化", "", "| 规模 | 配置 | 非胜→胜（局号） | 胜→非胜（局号） | 净胜场 | 精确配对p值 |",
              "|---|---|---|---|---:|---:|"]
    for size, groups in data.items():
        for group, row in groups.items():
            if group == "off":
                continue
            p = row["paired"]
            lines.append(f"| {size}v{size} | {LABELS[group]} | {p['nonwin_to_win']} | {p['win_to_nonwin']} | "
                         f"{p['wins_delta']:+d} | {p['exact_p']:.4f} |")
    lines += ["", "p值为胜/非胜的双侧精确配对检验，未作多重比较校正；20局样本只反映本批表现，不能证明稳定胜率。", "",
              "## 技能是否实际执行", "",
              "单位步表示一个单位在一个决策步被技能接管；战斗站位含合法等待。调用次数不等同于收益。", "",
              "| 规模 | 配置 | 集结单位步/触发局数 | 搜索单位步/触发局数 | 站位单位步/触发局数 | 队伍分组>1的观察步 | 首次攻击指令平均步 |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for size, groups in data.items():
        for group, row in groups.items():
            m, a = row["mechanisms"], row["mechanisms"]["active_episodes"]
            lines.append(f"| {size}v{size} | {LABELS[group]} | {m.get('regroup_unit_steps',0)}/{a['regroup']} | "
                         f"{m.get('cohesive_search_unit_steps',0)}/{a['cohesive_search']} | "
                         f"{m.get('positioning_unit_steps',0)}/{a['positioning']} | {m['split_observation_steps']} | "
                         f"{m['mean_first_attack_order_step']:.2f} |")
    lines += ["", "首次攻击指令不是命中或实际伤害。分裂步数也受对局时长、死亡和战况影响，不能单独解释因果。", "",
        "开关还有共享状态效应：当前代码开启任一空间能力后，会使用共享敌人记忆的1500-loop有效期；全关时沿用v7的4个决策步（32 loops）。",
        "因此本实验估计的是各开关的整体效果，包含共享记忆通路，不是剥离共享状态后的单个分配算法因果效应。",
        "例如仅集结10v10第6局并未实际集结，但第73步因仍保留40-loop前的敌位线索而改变搜索动作；胜负未改变。", "",
        "## 动作轨迹与胜负变化", "",
        "| 规模 | 配置 | 与全关逐步动作完全相同的局数 | 胜负变化局的首处分歧（从0计步） |",
        "|---|---|---:|---|"]
    for size, groups in data.items():
        for group, row in groups.items():
            if group == "off":
                continue
            t, p = row["trace_comparison"], row["paired"]
            changes = []
            for episode in sorted(p["nonwin_to_win"] + p["win_to_nonwin"]):
                d = t["first_differences"][episode]
                changes.append(f"局{episode}: 步{d['step']}，{d['variant_skill']}，"
                               f"首处分歧前可见战况{'一致' if d['same_visible_state'] else '不同'}")
            lines.append(f"| {size}v{size} | {LABELS[group]} | {len(t['identical_action_episodes'])}/20 | "
                         + ("；".join(changes) or "无胜负变化") + " |")
    lines += ["", "首处分歧只用于定位行为变化；后续位置、集火和伤害相互影响，不能把单步动作视作胜负的唯一原因。", "",
        "## 配对和复现核验", "",
        "- 200局完整完成，逐决策动作长度和活单位候选合法性均核验；无模型API调用。",
        "- 每规模各80对相对全关的场景哈希、真实初态哈希和完整首观察哈希完全匹配（共160对）；同规模清单、原战斗参数、依赖版本一致。",
        "- 十组实际记录的决策源码哈希均与最终冻结版本一致；新增开关与预定五组设置一致。",
        "- 开赛前两次失败均为0局完成：沙箱内SC2启动失败，以及首局站位+v7保护缺字段异常；原始失败现场和第一次冻结快照保留。",
        "- 全关10v10首次执行在第10局第0步出现SC2传输停滞，超过3分钟无事件；核实进程归属后结束该SC2子进程，运行器明确报无效局。",
        "  原9局和异常现场存于`.failed-sc2-03`目录，按预定规则同源码同清单重跑整组；这9局不加入主对照的200局，也不挑选种子或计作败局。",
        "- 修复缺字段后重新冻结，全部200局从头使用同一最终版本；没有依据胜负调参。种子210开局曾用于排错。",
        "- 最终离线回归525项通过：当前技能76、SMAC-Hard410、SMACv2适配器39。",
        f"- 原生指令拒绝总数：{sum(r['mechanisms'].get('rejected_commands',0) for g in data.values() for r in g.values())}；"
        f"观察期动作错误总数：{sum(r['mechanisms'].get('observation_errors',0) for g in data.values() for r in g.values())}。", "",
        "见 [预定方案](PROTOCOL.md)、[机器可读结果](results.json)、[执行记录](execution.json)、[最终源码哈希](frozen_hashes.json)。",
        "每局观察、动作和回放位于对应runs/spatial-20261025-*目录，results.json包含绝对路径。"]
    (OUT/"REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print("Validated all 200 episodes and wrote REPORT.md/results.json")


if __name__ == "__main__":
    main()
