"""Read-only, observation-safe diagnostics for the recorded Adaptive Skill run.

Cooldown resets are firing-cycle proxies, never confirmed damage attribution.
An enemy disappearing from team vision is not counted as a kill.
"""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import statistics


def distance(a, b):
    return math.dist(a, b)


def health(unit):
    return unit["health"] + unit["shield"]


def median(values):
    return round(statistics.median(values), 3) if values else None


def main(root):
    observations, decisions, steps = {}, {}, {}
    for line in (root / "events.jsonl").open(encoding="utf-8"):
        event = json.loads(line)
        kind = event["event"]
        if kind == "skill_observation":
            observations[event["episode"], event["step"]] = event["payload"]["state"]
        elif kind == "skill_step":
            decisions[event["episode"], event["step"]] = event
        elif kind == "step":
            steps[event["episode"], event["step"] - 1] = event
    assert observations.keys() == decisions.keys() == steps.keys()
    episodes = json.loads((root / "episodes.json").read_text(encoding="utf-8"))
    output = root / "trajectory_analysis"
    output.mkdir(exist_ok=True)
    timeline, units, aggregate = [], [], Counter()
    per_episode, fire_proxy = defaultdict(Counter), defaultdict(Counter)
    retreat_reasons, retreat_modes, types = Counter(), Counter(), defaultdict(Counter)
    movement, selected_cases = defaultdict(list), defaultdict(list)
    first_deaths, all_in = {}, []
    for (episode, step), state in sorted(observations.items()):
        d, actual = decisions[episode, step], steps[episode, step]
        next_state = observations.get((episode, step + 1))
        previous = steps.get((episode, step - 1))
        previous_state = observations.get((episode, step - 1))
        allies = state["allies"]
        visible = state["visible_enemies"]
        tally = per_episode[episode]
        tally["team_steps"] += 1
        if d["all_in"]:
            tally["all_in_steps"] += 1
        if d["all_in_started_now"]:
            all_in.append({"episode": episode, "step": step, "game_loop": state["game_loop"],
                           "reason": d["all_in_started_now"],
                           "ally_alive": sum(u["alive"] for u in allies.values()),
                           "visible_enemies": len(visible),
                           "ally_hp": sum(health(u) for u in allies.values() if u["alive"]),
                           "enemy_visible_hp": sum(health(u) for u in visible.values())})
        for actor, diag in d["movement_diagnostics"].items():
            retreat_reasons[diag["retreat_value_reason"]] += 1
        damage_visible = None
        if next_state:
            overlap = visible.keys() & next_state["visible_enemies"].keys()
            damage_visible = sum(max(0, health(visible[e]) -
                                     health(next_state["visible_enemies"][e])) for e in overlap)
        row = {"episode": episode, "step": step, "game_loop": state["game_loop"],
               "ally_alive": sum(u["alive"] for u in allies.values()),
               "ally_hp": sum(health(u) for u in allies.values() if u["alive"]),
               "visible_enemy_count": len(visible),
               "visible_enemy_hp": sum(health(u) for u in visible.values()),
               "enemy_hp_only_complete": sum(health(u) for u in visible.values()) if len(visible) == 5 else None,
               "observed_overlap_enemy_loss_next_step": damage_visible,
               "phase": d["phase"], "attack_assignments": d["attack_assignments"],
               "retreat_units": d["retreat_units"]}
        if not visible:
            tally["no_visible_enemies_steps"] += 1
            if next_state:
                allied_loss = sum(max(0, health(u) - health(next_state["allies"][a]))
                                  for a, u in allies.items() if u["alive"])
                if allied_loss > 0:
                    tally["no_visible_enemies_but_next_window_takes_damage"] += 1
                    selected_cases["unseen_damage"].append(
                        {"episode": episode, "step": step, "game_loop": state["game_loop"],
                         "observed_ally_loss": round(allied_loss, 2)})
        timeline.append(row)
        for actor, unit in allies.items():
            if not unit["alive"]:
                continue
            index = int(actor[1:])
            action = actual["actions"][index]
            rels = state["relations"].get(actor, {})
            legal = {e: r for e, r in rels.items() if r["smac_attack_available"]}
            ready = unit["weapon_cooldown_loops"] <= 9
            in_range = {e: r for e, r in legal.items() if r["in_weapon_range"] is True}
            old_action = previous["actions"][index] if previous else None
            kind = "attack" if action >= 6 else "move" if action in (2, 3, 4, 5) else "stop"
            tally[f"{kind}_actions"] += 1
            types[unit["type"]][f"{kind}_actions"] += 1
            entry = {"episode": episode, "step": step, "game_loop": state["game_loop"],
                     "actor": actor, "type": unit["type"], "action": action,
                     "hp": health(unit), "cooldown": unit["weapon_cooldown_loops"],
                     "legal_targets": len(legal), "native_in_range_targets": len(in_range),
                     "role": d["roles"].get(actor), "old_action": old_action}
            if legal:
                tally["legal_attack_unit_steps"] += 1
            if kind == "move" and legal:
                tally["moves_with_legal_attack"] += 1
                if in_range and ready:
                    tally["moves_with_window_fire"] += 1
                    selected_cases["moves_with_window_fire"].append(entry.copy())
            next_unit = next_state["allies"].get(actor) if next_state else None
            if next_unit and not next_unit["alive"]:
                tally["observed_deaths"] += 1
                first_deaths.setdefault(episode, {"step": step + 1, "game_loop": next_state["game_loop"],
                                                  "actor": actor, "allies_before": row["ally_alive"],
                                                  "visible_enemies_before": len(visible)})
                selected_cases["deaths"].append(entry.copy())
            if kind == "attack":
                target = f"e{action-6}"
                relation = rels[target]
                entry.update(target=target, target_hp=health(visible[target]),
                             target_distance=relation["distance"],
                             target_native_in_range=relation["in_weapon_range"])
                tally["attacks_outside_native_range"] += relation["in_weapon_range"] is False
                types[unit["type"]]["attacks_outside_native_range"] += relation["in_weapon_range"] is False
                if old_action is not None and old_action >= 6:
                    tally["consecutive_attack_pairs"] += 1
                switched = old_action is not None and old_action >= 6 and old_action != action
                category = "changed_target" if switched else "same_target" if old_action == action else "new_attack"
                if switched:
                    tally["target_switches"] += 1
                    old_target = f"e{old_action-6}"
                    still_legal = old_target in legal
                    tally["switch_old_target_still_legal"] += still_legal
                    entry.update(old_target_still_legal=still_legal,
                                 old_target_hp=health(visible[old_target]) if old_target in visible else None)
                    selected_cases["switches"].append(entry.copy())
                nearest = min(legal, key=lambda e: legal[e]["distance"])
                if legal[nearest]["distance"] + 0.5 < relation["distance"]:
                    tally["attacks_farther_than_nearest_by_0_5"] += 1
                if relation["in_weapon_range"] is False and in_range:
                    tally["out_of_range_choice_with_in_range_alternative"] += 1
                    selected_cases["out_of_range_choice"].append(entry.copy())
                if next_unit and next_unit["alive"] and relation["in_weapon_range"] is True and ready:
                    key = f"{unit['type']}:{category}"
                    fire_proxy[key]["windows"] += 1
                    # Cooldown above simple 8-loop countdown is evidence of reset,
                    # but not a measurement of which victim took damage.
                    reset = next_unit["weapon_cooldown_loops"] > max(0, unit["weapon_cooldown_loops"] - 8) + 0.1
                    fire_proxy[key]["cooldown_resets"] += reset
                    entry["cooldown_reset_proxy"] = reset
            if kind == "move" and next_unit and next_unit["alive"]:
                actual_distance = distance(unit["position"], next_unit["position"])
                predicted_distance = distance(unit["position"],
                                              state["movement_candidates"][actor][str(action)]["destination"])
                movement[unit["type"]].append(actual_distance)
                entry.update(actual_displacement=round(actual_distance, 3),
                             nominal_displacement=round(predicted_distance, 3))
                tally["observed_move_transitions"] += 1
                tally["move_displacement_under_0_2"] += actual_distance < 0.2
            if actor in d["retreat_units"]:
                tally["retreat_unit_steps"] += 1
                retreat_modes[f"{unit['type']}:{d['movement_diagnostics'][actor]['movement_mode']}"] += 1
            if kind == "move" and old_action in (2, 3, 4, 5):
                tally["consecutive_move_pairs"] += 1
                reversal = (old_action, action) in ((2, 3), (3, 2), (4, 5), (5, 4))
                tally["reversed_move_direction"] += reversal
                if reversal:
                    tally["reversed_move_with_no_visible_enemies"] += not visible
                    tally["reversed_move_after_retreat"] += (
                        actor in decisions.get((episode, step - 1), {}).get("retreat_units", []))
            if next_unit and next_unit["alive"] and kind == "attack":
                # In-range idle windows, useful only as a timing diagnostic.
                entry["next_cooldown"] = next_unit["weapon_cooldown_loops"]
                entry["actual_displacement"] = round(distance(unit["position"], next_unit["position"]), 3)
            units.append(entry)
    for tally in per_episode.values():
        aggregate.update(tally)
    result = {"scope": "Recorded partial observations and actions; no causal counterfactual games.",
              "all_episodes": dict(aggregate),
              "per_episode": [{"episode": e["episode"], "outcome": e["outcome"],
                               "enemy_elimination_fraction": e["enemy_elimination_fraction"],
                               "steps": e["steps"], "first_ally_death": first_deaths.get(e["episode"]),
                               **dict(per_episode[e["episode"]])} for e in episodes],
              "by_unit_type": dict(types), "retreat_value_reasons": dict(retreat_reasons),
              "accepted_retreat_modes": dict(retreat_modes), "all_in_triggers": all_in,
              "movement_displacement": {kind: {"count": len(values), "median": median(values),
                                              "mean": round(statistics.mean(values), 3),
                                              "nominal": 2} for kind, values in movement.items()},
              "cooldown_reset_proxy": dict(fire_proxy)}
    for name, value in (("metrics.json", result), ("team_timeline.json", timeline),
                        ("unit_timeline.json", units), ("case_candidates.json", selected_cases)):
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    main(p.parse_args().root.resolve())
