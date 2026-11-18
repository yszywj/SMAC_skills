"""Minimal total viability check for the combat-review changes.

Three small synthetic scenes (zerg with a second consecutive snapshot,
terran, protoss) through the REAL SkillManager: load defaults, step(),
commit_final_actions(). Checks only that config/module loading works,
action counts match and every action is in that snapshot's legal menu,
and commit + short telemetry succeed. NO game, NO episode, NO win-rate or
tactical-quality claim: passing this proves none of the combat logic.
Optional local output goes to runs/tactical_combat_review/.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from skills.tactical_v1.skill_manager import SkillManager  # noqa: E402
from skills.tactical_v1.tests.fixtures import (  # noqa: E402
    ally, enemy, scene, with_questions)


def calibrate_baneling(state):
    """Install the REAL calibration catalog contract for type id 9.

    The generic fixture ships a fake baneling with an ordinary weapon; the
    native profile requires exactly the contract fields below (empty
    weapons, speed 2.5, attributes [3], armor 0, known). Using the contract
    keeps burst.profile() calibrated instead of silently mismatching.
    """
    profile = json.loads((ROOT / "skills/tactical_v1/burst_native_profile.json")
                         .read_text(encoding="utf8"))["catalog_contract"]
    state["unit_type_profiles"]["9"] = dict(profile)
    return state


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

    # Zerg: a REAL calibrated baneling with a legal contact attack whose
    # explosion would cover two nearby ground enemies (positive exchange
    # candidates -> trigger_assignments -> selective-commitment basis is
    # actually executed), plus an approaching enemy baneling and hydralisk.
    # Two consecutive snapshots exercise motion history and memory paths.
    def zerg_allies(shift=(0., 0.)):
        return [((10. + shift[0], 10. + shift[1]), "zergling"),
                ((10.5 + shift[0], 10.5 + shift[1]), "zergling"),
                ((11. + shift[0], 9. + shift[1]), "hydralisk"),
                ((11.5 + shift[0], 9.5 + shift[1]), "hydralisk"),
                ((12. + shift[0], 11. + shift[1]), "baneling"),
                ((12.5 + shift[0], 11.5 + shift[1]), "baneling")]

    def zerg_enemies(shift=(0., 0.)):
        return {"e0": dict(enemy((6. + shift[0], 10. + shift[1]), "baneling",
                                 health=30., max_health=30.), tag="tag-e0"),
                "e1": dict(enemy((5. + shift[0], 9. + shift[1]), "hydralisk"),
                           tag="tag-e1"),
                "e2": dict(enemy((12.8 + shift[0], 11. + shift[1]), "hydralisk"),
                           tag="tag-e2"),
                "e3": dict(enemy((12.3 + shift[0], 11.9 + shift[1]), "hydralisk"),
                           tag="tag-e3")}

    zerg1 = scene(zerg_allies(), enemies=zerg_enemies())
    for actor in ("u4", "u5"):
        zerg1["allies"][actor].update(health=30., max_health=30., tag=f"tag-{actor}")
    zerg1 = with_questions(calibrate_baneling(zerg1))
    manager = SkillManager()
    run_scene(manager, zerg1, "zerg#1", failures)
    zerg2 = scene(zerg_allies((.4, .4)), enemies=zerg_enemies((1., .2)),
                  loop=zerg1["state"]["game_loop"] + zerg1["state"]["step_mul"],
                  episode=zerg1["state"]["episode"])
    for actor in ("u4", "u5"):
        zerg2["allies"][actor].update(health=30., max_health=30., tag=f"tag-{actor}")
    zerg2 = with_questions(calibrate_baneling(zerg2))
    run_scene(manager, zerg2, "zerg#2", failures)
    # The calibrated profile must NOT report a catalog mismatch.
    from skills.tactical_v1.context import Context, build_snapshot
    from skills.tactical_v1.burst import profile as burst_profile
    probe = Context(build_snapshot(zerg2, 99))
    evidence = burst_profile(probe).evidence
    print(f"zerg burst profile evidence: {evidence}")
    if "mismatch" in evidence:
        failures.append(f"zerg: burst profile catalog mismatch: {evidence}")

    # Terran: marines + marauder + medivac vs zerglings.
    terran = scene([((10., 10.), "marine"), ((10.5, 10.5), "marine"),
                    ((11., 9.5), "marauder"), ((12., 12.), "medivac")],
                   enemies={"e0": dict(enemy((6., 10.), "zergling"), tag="tag-e0"),
                            "e1": dict(enemy((5.5, 9.), "zergling"), tag="tag-e1")})
    run_scene(SkillManager(), with_questions(terran), "terran", failures)

    # Protoss: zealot + stalker + colossus vs zerglings/hydralisk.
    protoss = scene([((10., 10.), "zealot"), ((11., 9.5), "stalker"),
                     ((12., 12.), "colossus")],
                    enemies={"e0": dict(enemy((6., 10.), "zergling"), tag="tag-e0"),
                             "e1": dict(enemy((5., 9.), "hydralisk"), tag="tag-e1")})
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
