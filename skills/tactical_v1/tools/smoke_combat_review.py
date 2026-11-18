"""Minimal total viability check for the combat-review changes.

Three small synthetic scenes (zerg with a second consecutive snapshot,
terran, protoss) through the REAL SkillManager: load defaults, step(),
commit_final_actions(). Checks only that config/module loading works,
action counts match and every action is in that snapshot's legal menu,
and commit + short telemetry succeed. NO game, NO episode, NO win-rate or
tactical-quality claim: passing this proves none of the combat logic.
Optional local output goes to runs/tactical_combat_review/.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from skills.tactical_v1.skill_manager import SkillManager  # noqa: E402
from skills.tactical_v1.tests.fixtures import (  # noqa: E402
    ally, enemy, scene, with_questions)


def legal_menu(payload):
    return {actor: {int(a) for a in row["criteria"]}
            for actor, row in payload["questions"].items()}


def run_scene(manager, payload, label, failures):
    actions, summary = manager.step(payload, len(payload["state"]["allies"]))
    menu = legal_menu(payload)
    if len(actions) != len(payload["state"]["allies"]):
        failures.append(f"{label}: wrong action count {len(actions)}")
    for i, action in enumerate(actions):
        actor = f"u{i}"
        if action not in menu.get(actor, set()):
            failures.append(f"{label}: {actor} action {action} not legal")
    manager.commit_final_actions(actions)
    print(f"{label}: {len(actions)} actions, joint_score="
          f"{summary.get('joint_score')}, commit ok")
    return actions, summary


def main():
    failures = []

    # Zerg: zerglings + hydralisks + banelings vs an approaching enemy
    # baneling and hydralisk; two CONSECUTIVE snapshots so the new motion
    # history and memory paths are exercised at least once.
    zerg1 = scene(
        [((10., 10.), "zergling"), ((10.5, 10.5), "zergling"),
         ((11., 9.), "hydralisk"), ((11.5, 9.5), "hydralisk"),
         ((12., 11.), "baneling"), ((12.5, 11.5), "baneling")],
        enemies={"e0": enemy((6., 10.), "baneling", health=30., max_health=30.),
                 "e1": enemy((5., 9.), "hydralisk")})
    for actor, health in (("u4", 30.), ("u5", 30.)):
        zerg1["allies"][actor].update(health=health, max_health=health)
    zerg1 = with_questions(zerg1)
    manager = SkillManager()
    run_scene(manager, zerg1, "zerg#1", failures)
    zerg2 = scene(
        [((10.4, 10.4), "zergling"), ((10.9, 10.9), "zergling"),
         ((11.2, 9.2), "hydralisk"), ((11.7, 9.7), "hydralisk"),
         ((12.2, 11.2), "baneling"), ((12.7, 11.7), "baneling")],
        enemies={"e0": enemy((7., 10.2), "baneling", health=30., max_health=30.),
                 "e1": enemy((5.6, 9.1), "hydralisk")},
        loop=zerg1["state"]["game_loop"] + zerg1["state"]["step_mul"],
        episode=zerg1["state"]["episode"])
    for actor in ("u4", "u5"):
        zerg2["allies"][actor].update(health=30., max_health=30.)
    run_scene(manager, with_questions(zerg2), "zerg#2", failures)

    # Terran: marines + marauder + medivac vs zerglings.
    terran = scene([((10., 10.), "marine"), ((10.5, 10.5), "marine"),
                    ((11., 9.5), "marauder"), ((12., 12.), "medivac")],
                   enemies={"e0": enemy((6., 10.), "zergling"),
                            "e1": enemy((5.5, 9.), "zergling")})
    run_scene(SkillManager(), with_questions(terran), "terran", failures)

    # Protoss: zealot + stalker + colossus vs zerglings/hydralisk.
    protoss = scene([((10., 10.), "zealot"), ((11., 9.5), "stalker"),
                     ((12., 12.), "colossus")],
                    enemies={"e0": enemy((6., 10.), "zergling"),
                             "e1": enemy((5., 9.), "hydralisk")})
    run_scene(SkillManager(), with_questions(protoss), "protoss", failures)

    if failures:
        print("FAILURES:")
        for row in failures:
            print(" -", row)
        print("OVERALL: FAIL")
        return 1
    print("OVERALL: PASS (minimal viability only; no tactical validation)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
