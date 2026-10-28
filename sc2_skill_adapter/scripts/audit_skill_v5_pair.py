"""Independent, read-only event audit. Never imports the environment or policy.

Run after both v5 series finish. --self-check validates accounting on v4 only.
"""
import argparse
import collections
import hashlib
import json
import statistics
from pathlib import Path

ADAPTER = Path(__file__).resolve().parents[1]
MECHANISMS = ("critical_escape", "planned_reentry")


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fraction_counts(counter):
    total = sum(counter.values())
    return {"total": total, "counts": dict(counter),
            "fractions": {key: count / total for key, count in counter.items()} if total else {}}


def damage_category(status):
    if status == "unknown":
        return "unknown"
    if status == "catalog_exact_unshielded":
        return "unshielded_catalog_exact"
    if status == "catalog_shield_only_estimate":
        return "shield_only_prior_estimate"
    if status == "bounded_shield_transition":
        return "bounded_shield_transition"
    return "unrecognized_status:" + str(status)


def unit_brief(unit):
    return {key: unit.get(key) for key in (
        "type", "alive", "health", "shield", "effective_health", "max_health", "max_shield",
        "position", "weapon_cooldown_loops", "attack_upgrade_level", "armor_upgrade_level", "shield_upgrade_level")}


def load_run(path, size, expected):
    summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    episodes = json.loads((path / "episodes.json").read_text(encoding="utf-8"))
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if summary.get("status") != "complete" or len(episodes) != expected:
        raise ValueError(f"Not a complete {expected}-episode run: {path}")
    if len({row['episode'] for row in episodes}) != expected:
        raise ValueError(f"Duplicate episode numbers: {path}")
    step_counts = collections.Counter()
    damage_all, damage_assigned, reasons = collections.Counter(), collections.Counter(), collections.Counter()
    bonus_counts, profile_counts = collections.Counter(), collections.Counter()
    mechanism_counts = {name: collections.Counter() for name in MECHANISMS}
    applied_cases = {name: [] for name in MECHANISMS}
    frames, actions, states_seen, terminal = {}, {}, set(), {}
    damage_examples, damage_example_keys, models = [], set(), set()
    current_state = None
    for line_number, line in enumerate((path / "events.jsonl").open(encoding="utf-8"), 1):
        event = json.loads(line)
        key = (event.get("episode"), event.get("step"))
        kind = event.get("event")
        if kind == "skill_observation":
            current_state = event["payload"]["state"]
            states_seen.add(key)
            frames[key] = {"observation_line": line_number,
                           "allies": {a: unit_brief(u) for a, u in current_state["allies"].items()},
                           "visible_enemies": {a: unit_brief(u) for a, u in current_state["visible_enemies"].items()},
                           "native_inrange_enemies": {a: [e for e, r in rows.items() if r.get("in_weapon_range") is True]
                                                       for a, rows in current_state.get("relations", {}).items()},
                           "legal_targets": {a: [e for e, r in rows.items() if r.get("smac_attack_available")]
                                             for a, rows in current_state.get("relations", {}).items()}}
            for profile in current_state.get("unit_type_profiles", {}).values():
                profile_counts["profile_rows"] += 1
                profile_counts["explicit_attributes"] += isinstance(profile.get("attributes"), list)
                for weapon in profile.get("weapons", []):
                    profile_counts["weapon_rows"] += 1
                    profile_counts["explicit_damage_bonus_list"] += isinstance(weapon.get("damage_bonus"), list)
                    profile_counts["nonempty_damage_bonus_list"] += bool(weapon.get("damage_bonus"))
            for unit in [*current_state["allies"].values(), *current_state["visible_enemies"].values()]:
                if unit.get("alive"):
                    profile_counts["living_unit_rows"] += 1
                    for field in ("attack_upgrade_level", "armor_upgrade_level", "shield_upgrade_level"):
                        profile_counts[field + "_missing"] += unit.get(field) is None
        elif kind == "skill_step":
            if key not in states_seen:
                raise ValueError(f"Skill event without preceding observation at {key}")
            step_counts["skill_frames"] += 1
            frame = frames[key]
            frame.update(skill_step_line=line_number, roles=event.get("roles", {}),
                         assignments=event.get("attack_assignments", {}), phase=event.get("phase"))
            model = event.get("damage_model") or {}
            if model.get("version"):
                models.add(model["version"])
                step_counts["frames_with_damage_model"] += 1
            pairs = model.get("pairs", {})
            for actor, rows in pairs.items():
                for enemy, detail in rows.items():
                    category = damage_category(detail.get("status"))
                    damage_all[category] += 1
                    assigned = event.get("attack_assignments", {}).get(actor) == enemy
                    if assigned:
                        damage_assigned[category] += 1
                    for reason in detail.get("reasons", []):
                        reasons[reason] += 1
                    if (detail.get("matching_attribute_bonus") or 0) > 0:
                        bonus_counts["all_pairs_with_matching_bonus"] += 1
                        bonus_counts["assigned_pairs_with_matching_bonus"] += assigned
                    example_key = (category, current_state["allies"][actor].get("type"),
                                   current_state["visible_enemies"][enemy].get("type"),
                                   detail.get("matching_attribute_bonus"), tuple(detail.get("reasons", [])))
                    if len(damage_examples) < 24 and example_key not in damage_example_keys and assigned:
                        damage_example_keys.add(example_key)
                        attacker = current_state["allies"][actor]
                        target = current_state["visible_enemies"][enemy]
                        profiles = current_state["unit_type_profiles"]
                        damage_examples.append({"episode": key[0], "step_zero_based": key[1],
                            "event_line": line_number, "actor": actor, "enemy": enemy, "detail": detail,
                            "attacker": unit_brief(attacker), "target": unit_brief(target),
                            "attacker_profile": profiles.get(str(attacker["unit_type_id"])),
                            "target_profile": profiles.get(str(target["unit_type_id"]))})
            for actor, enemy in event.get("attack_assignments", {}).items():
                if enemy not in pairs.get(actor, {}):
                    step_counts["assigned_pairs_without_damage_diagnostic"] += 1
            for name in MECHANISMS:
                rows = event.get(name + "_diagnostics", {})
                counts = mechanism_counts[name]
                counts["evaluations"] += len(rows)
                for actor, detail in rows.items():
                    if detail.get("applied"):
                        counts["logged_applied"] += 1
                        applied_cases[name].append({"episode": key[0], "step_zero_based": key[1],
                            "event_line": line_number, "actor": actor,
                            "unit_before": frame["allies"].get(actor),
                            "legal_targets_before": frame["legal_targets"].get(actor, []),
                            "detail": detail, "role": event.get("roles", {}).get(actor)})
        elif kind == "step":
            # runner's transition step is one greater than its observation/skill step.
            transition_key = (event["episode"], event["step"] - 1)
            actions[transition_key] = {"actions": event.get("actions", []),
                                      "feedback": event.get("feedback", {}).get("actors", {}),
                                      "event_line": line_number}
            if event.get("terminated"):
                terminal[event["episode"]] = {"step_event_line": line_number,
                                               "step": event["step"], "terminated": True}
    outcomes = {row["episode"]: row["outcome"] for row in episodes}
    if set(actions) != states_seen or len(frames) != step_counts["skill_frames"]:
        raise ValueError(f"Observation/skill/action transition alignment disagrees: {path}")
    for name, cases in applied_cases.items():
        for case in cases:
            key = (case["episode"], case["step_zero_based"])
            transition = actions.get(key, {})
            index = int(case["actor"][1:])
            choices = transition.get("actions", [])
            actual = choices[index] if index < len(choices) else None
            detail = case["detail"]
            selected = detail.get("selected_final_action", detail.get("proposed_action")
                                  if name == "critical_escape" else detail.get("selected_action"))
            feedback = transition.get("feedback", {}).get(case["actor"], {})
            matched = selected is not None and actual == selected
            accepted = feedback.get("submission") == "accepted"
            case.update(submitted_action=actual, selected_action_matches_submission=matched,
                        command_accepted=accepted, command_submission=feedback.get("submission"),
                        command_observation_errors=feedback.get("observation_errors", []),
                        episode_outcome=outcomes[case["episode"]])
            mechanism_counts[name]["selected_action_matches_submission"] += matched
            mechanism_counts[name]["matching_action_accepted"] += matched and accepted
            case["following_observations"] = [{"step_zero_based": key[1] + offset,
                "unit": frames[(key[0], key[1] + offset)]["allies"].get(case["actor"]),
                "role": frames[(key[0], key[1] + offset)].get("roles", {}).get(case["actor"]),
                "assignment": frames[(key[0], key[1] + offset)].get("assignments", {}).get(case["actor"])}
                for offset in range(1, 4) if (key[0], key[1] + offset) in frames]
    losses = []
    for row in episodes:
        if row["outcome"] == "win":
            continue
        info = row.get("info", {})
        if info.get("dead_allies") == size:
            reason = "all_allies_dead"
        elif row["outcome"] == "timeout" or info.get("episode_limit"):
            reason = "episode_limit"
        else:
            reason = "other_or_unresolved_from_record"
        episode_keys = sorted(key for key in frames if key[0] == row["episode"])
        tail_keys = episode_keys[-20:]
        tail_counts, tail_roles = collections.Counter(), collections.Counter()
        for key in tail_keys:
            frame = frames[key]
            choices = actions[key]["actions"]
            for actor, unit in frame["allies"].items():
                if not unit.get("alive"):
                    continue
                tail_counts["living_unit_steps"] += 1
                tail_roles[str(frame.get("roles", {}).get(actor))] += 1
                action = choices[int(actor[1:])]
                tail_counts["attack_commands"] += action >= 6
                tail_counts["move_commands"] += action in (2, 3, 4, 5)
                tail_counts["stop_commands"] += action == 1
                tail_counts["native_inrange_but_no_legal_attack_unit_steps"] += bool(
                    frame["native_inrange_enemies"].get(actor) and not frame["legal_targets"].get(actor))
        for name in MECHANISMS:
            tail_counts[name + "_logged_applied"] = sum(
                case["episode"] == row["episode"] and (case["episode"], case["step_zero_based"]) in tail_keys
                for case in applied_cases[name])
        final_frame = frames[episode_keys[-1]]
        losses.append({"episode": row["episode"], "outcome": row["outcome"],
                       "enemy_kills": info.get("dead_enemies"),
                       "enemy_elimination_fraction": row["enemy_elimination_fraction"],
                       "steps": row["steps"], "termination_reason_from_record": reason,
                       "info": info, "terminal_transition": terminal.get(row["episode"]),
                       "last_up_to_20_decision_steps": {"first_step": tail_keys[0][1], "last_step": tail_keys[-1][1],
                           "counts": dict(tail_counts), "roles": dict(tail_roles)},
                       "last_preterminal_observation": {"step": episode_keys[-1][1],
                           "allies": {a: u for a, u in final_frame["allies"].items() if u.get("alive")},
                           "visible_enemies": final_frame["visible_enemies"],
                           "scope": "last decision observation, not a complete terminal enemy roster"}})
    wins = sum(row["outcome"] == "win" for row in episodes)
    if wins != summary.get("outcomes", {}).get("win") or step_counts["skill_frames"] != sum(row["steps"] for row in episodes):
        raise ValueError(f"Summary/event accounting disagrees: {path}")
    return {"run": str(path), "sources": {name: sha(path / name) for name in ("summary.json", "episodes.json", "manifest.json")},
            "wins": wins, "episodes": episodes, "total": len(episodes),
            "mean_enemy_elimination": statistics.mean(row["enemy_elimination_fraction"] for row in episodes),
            "mean_steps": statistics.mean(row["steps"] for row in episodes),
            "env_args": manifest["env_args"], "step_counts": dict(step_counts),
            "damage": {"versions": sorted(models), "all_evaluated_pairs": fraction_counts(damage_all),
                       "assigned_pairs": fraction_counts(damage_assigned), "reasons": dict(reasons),
                       "bonus_counts": dict(bonus_counts), "input_presence": dict(profile_counts),
                       "examples": damage_examples},
            "mechanisms": {name: {"counts": dict(mechanism_counts[name]), "applied_cases": cases}
                           for name, cases in applied_cases.items()}, "losses": losses}


def pair_runs(old, new):
    before = {row["episode"]: row for row in old["episodes"]}
    after = {row["episode"]: row for row in new["episodes"]}
    if before.keys() != after.keys() or old["env_args"] != new["env_args"]:
        raise ValueError("Run configurations or episode identifiers differ; paired comparison refused")
    rows, flips = [], collections.Counter()
    for episode in sorted(before):
        a, b = before[episode], after[episode]
        for field in ("scenario_sha256", "initial_state_sha256"):
            if not a.get(field) or a[field] != b.get(field):
                raise ValueError(f"Episode {episode}: {field} mismatch; paired comparison refused")
        transition = a["outcome"] + "_to_" + b["outcome"]
        flips[transition] += 1
        rows.append({"episode": episode, "transition": transition,
                     "old_kills": a.get("info", {}).get("dead_enemies"),
                     "new_kills": b.get("info", {}).get("dead_enemies"),
                     "old_steps": a["steps"], "new_steps": b["steps"],
                     "scenario_and_initial_state_match": True})
    return {"flips": dict(flips), "episodes": rows}


def representative_cases(mechanism, limit=3):
    cases = mechanism["applied_cases"]
    ordered = sorted(cases, key=lambda row: (not (row["selected_action_matches_submission"] and row["command_accepted"]),
                                             row["unit_before"].get("effective_health", 0)))
    picked, episodes = [], set()
    # Include both a return to attacking and a recorded subsequent death;
    # neither is a counterfactual claim of survival benefit or harm.
    selectors = (
        lambda row: len(row["following_observations"]) >= 2 and all(
            item["unit"].get("alive") for item in row["following_observations"])
            and any(item.get("assignment") for item in row["following_observations"]),
        lambda row: any(not item["unit"].get("alive") for item in row["following_observations"]),
    )
    for selector in selectors:
        choice = next((row for row in ordered if row["episode"] not in episodes
                       and row["selected_action_matches_submission"] and row["command_accepted"]
                       and selector(row)), None)
        if choice is not None:
            picked.append(choice)
            episodes.add(choice["episode"])
    for row in ordered:
        if row["episode"] not in episodes:
            picked.append(row)
            episodes.add(row["episode"])
        if len(picked) >= limit:
            break
    return picked


def render(report):
    lines = ["# v5 纯 Skill 配对独立审计", "",
             "只读取已完成日志；未运行 SC2 或模型。逐局场景哈希、初始状态哈希及环境参数均已核对。step 为从 0 开始的决策步。", "",
             "| 场景 | 版本 | 胜场/总局数 | 平均敌方消灭比例 | 平均步数 |", "| --- | --- | --- | --- | --- |"]
    for size, data in report["sizes"].items():
        for version in ("baseline", "current"):
            run = data[version]
            lines.append(f"| {size}v{size} | {run['version']} | {run['wins']}/{run['total']} | {run['mean_enemy_elimination']:.1%} | {run['mean_steps']:.2f} |")
    for size, data in report["sizes"].items():
        new = data["current"]
        lines += ["", f"## {size}v{size}", "", "配对结果：" + "；".join(f"{k}={v}" for k, v in data["paired"]["flips"].items()) + "。", "",
                  "伤害统计的分母是重复的攻击者—目标估计次数，不是实际命中次数。已分配目标也可能仍在冷却。", "",
                  "| 口径 | 总数 | 无盾 catalog 精确 | 纯盾层先验估计 | 跨盾区间 | unknown |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
        for name, label in (("all_evaluated_pairs", "全部候选 pair"), ("assigned_pairs", "实际分配 pair")):
            stats = new["damage"][name]
            cells = []
            for status in ("unshielded_catalog_exact", "shield_only_prior_estimate", "bounded_shield_transition", "unknown"):
                count = stats["counts"].get(status, 0)
                cells.append(f"{count} ({count / stats['total']:.1%})" if stats["total"] else "无记录")
            lines.append(f"| {label} | {stats['total']} | " + " | ".join(cells) + " |")
        bonus = new["damage"]["bonus_counts"].get("assigned_pairs_with_matching_bonus", 0)
        lines += ["", f"实际分配中匹配到非零原生属性加成：{bonus} 次。模型版本：{', '.join(new['damage']['versions']) or '未记录'}。",
                  f"未携带伤害诊断的攻击分配：{new['step_counts'].get('assigned_pairs_without_damage_diagnostic', 0)} 次。", ""]
        for mechanism in MECHANISMS:
            item = new["mechanisms"][mechanism]
            counts = item["counts"]
            lines.append(f"- {mechanism}：评估 {counts.get('evaluations', 0)}，日志 applied {counts.get('logged_applied', 0)}，所选动作与提交动作一致且指令 accepted {counts.get('matching_action_accepted', 0)}。")
            for case in representative_cases(item):
                detail = case["detail"]
                after = ", ".join(f"s{r['step_zero_based']}:{r['unit'].get('effective_health')}HP/{r.get('role')}" for r in case["following_observations"])
                lines.append(f"  - ep{case['episode']} step{case['step_zero_based']} {case['actor']} {case['unit_before'].get('type')}，{case['unit_before'].get('effective_health')}HP，提交动作 {case['submitted_action']}，日志行 {case['event_line']}；{detail.get('reason') or detail.get('phase')}；随后 {after or '已到终局，无后续决策观测'}。")
        lines += ["", "败局及终局事实：", "", "| 局 | 敌方消灭数 | 步数 | 终止依据 |", "| --- | --- | --- | --- |"]
        for row in new["losses"]:
            lines.append(f"| {row['episode']} | {row['enemy_kills']}/{size} | {row['steps']} | {row['termination_reason_from_record']} |")
        if not new["losses"]:
            lines.append("| — | 无败局 | — | — |")
    lines += ["", "## 解释范围", "",
              "这是同场景、同初始状态的两套完整策略比较；各机制同时改变，不能将胜负翻转单独归因于伤害修正、危急撤退或重入规划。applied 表示同一观察下与关闭该局部机制的选择存在差异，不是反事实整局胜率。指令 accepted 不保证命中、伤害或安全；后续存活也不证明该动作救活了单位。", "",
              "planned_reentry 的 applied 还可能表示否决旧移动；审计优先使用 selected_final_action 核对最终提交，避免把空 selected_action 误报为未执行。全部候选 pair 包括没有被分配或不可即时开火的组合。", "",
              "纯盾层使用显式原生 Protoss 零级盾甲先验；跨盾采用保守下界。无盾 catalog 精确仅指已建模的零升级单次武器公式，不包括未建模 buff、弹道、实际攻击时机或溅射。unknown 不记为零伤害真值，也不计入伤害预算。每组 20 局不能证明泛化胜率提高。", "",
              "完整逐局翻转、每类原因、输入字段完整性及所有实际触发片段见 INDEPENDENT_AUDIT.json。"]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-version", default="v4")
    parser.add_argument("--current-version", default="v5")
    parser.add_argument("--seed", type=int, default=150)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=ADAPTER / "analysis/skill_repairs_v5_20261023")
    args = parser.parse_args()
    report = {"audit_version": 1, "sizes": {}}
    for size in (5, 10):
        old = load_run(ADAPTER / f"runs/native-{size}v{size}-{args.baseline_version}-skill-n{args.episodes}-seed{args.seed}", size, args.episodes)
        old["version"] = args.baseline_version
        if args.self_check:
            paired = pair_runs(old, old)
            assert sum(v for k, v in paired["flips"].items() if k not in ("win_to_win", "loss_to_loss", "draw_to_draw", "timeout_to_timeout")) == 0
            print(f"{size}v{size} self-check: {old['wins']}/{old['total']}, {old['step_counts']['skill_frames']} frames; zero self flips")
            continue
        new = load_run(ADAPTER / f"runs/native-{size}v{size}-{args.current_version}-skill-n{args.episodes}-seed{args.seed}", size, args.episodes)
        new["version"] = args.current_version
        report["sizes"][str(size)] = {"baseline": old, "current": new, "paired": pair_runs(old, new)}
    if not args.self_check:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "INDEPENDENT_AUDIT.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        (args.output_dir / "INDEPENDENT_AUDIT.md").write_text(render(report), encoding="utf-8")
        print(args.output_dir / "INDEPENDENT_AUDIT.md")


if __name__ == "__main__":
    main()
