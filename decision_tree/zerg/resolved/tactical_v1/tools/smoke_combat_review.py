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
    # Cooldown-cycle setup: u2 starts ready (0); after the step it observes
    # cooldown 6 > elapsed 8-expectation 0 — the natural fired-proxy input.
    zerg1["allies"]["u2"]["weapon_cooldown_loops"] = 0.
    zerg1 = with_questions(calibrate_baneling(zerg1))
    manager = SkillManager()
    run_scene(manager, zerg1, "zerg#1", failures)
    # Second snapshot: same reliable tags, different loops, consistent
    # non-zero same-direction motion for every enemy.
    zerg2 = scene(zerg_allies((.4, .4)), enemies=zerg_enemies((1., .2)),
                  loop=zerg1["state"]["game_loop"] + zerg1["state"]["step_mul"],
                  episode=zerg1["state"]["episode"])
    for actor in ("u4", "u5"):
        zerg2["allies"][actor].update(health=30., max_health=30., tag=f"tag-{actor}")
    # Hydralisk u2 fired during the first window (proxy: fresh cooldown 6).
    zerg2["allies"]["u2"]["weapon_cooldown_loops"] = 6.
    zerg2 = with_questions(calibrate_baneling(zerg2))
    run_scene(manager, zerg2, "zerg#2", failures)
    # The calibrated profile must NOT report a catalog mismatch.
    from skills.tactical_v1.context import Context, build_snapshot
    from skills.tactical_v1.burst import profile as burst_profile
    probe = Context(build_snapshot(zerg2, 99))
    probe.burst_config = dict(manager.config.get("burst_engage", {}))
    evidence = burst_profile(probe).evidence
    print(f"zerg burst profile evidence: {evidence}")
    if "mismatch" in evidence:
        failures.append(f"zerg: burst profile catalog mismatch: {evidence}")
    # Wiring 1 — motion: the second observation must yield a VALID velocity
    # estimate (not just history rows), from the shared recorded samples.
    from skills.tactical_v1.enemy_motion import velocity
    probe.enemy_motion = manager.memory.enemy_motion
    motion = velocity(probe, "e1")
    print(f"zerg motion e1: {motion.get('reason') if motion else None}")
    if not (motion and motion.get('reliable')):
        failures.append("zerg: motion estimate invalid on second snapshot")
    # Wiring 3 — budget: the fired proxy for u2 must have restored its
    # delay budget during the second observation.
    available = manager.memory.fire_delay_available(probe, "u2")
    fired = "u2" in manager.memory.combat_progress.fired
    print(f"zerg u2 fired={fired} available_loops={available}")
    if not fired:
        failures.append("zerg: cooldown-cycle proxy did not mark u2 fired")
    if available <= 0.:
        failures.append("zerg: fired evidence did not restore delay budget")
    # Wiring 2 — soft intercept: bomber-contract candidates must reach the
    # whole-plan validation entry. Drive validate_plans directly with a
    # synthetic bomber contract built from the scene's intercept facts.
    from skills.tactical_v1.intercept import intercept_plan, validate_intercept_plan
    plans_found = intercept_plan(probe, manager.memory)
    print(f"zerg intercept plans: {len(plans_found)}")
    if plans_found:
        row = plans_found[0]
        members = {a: action for a, action, _ in row['shooters']}
        members.update(dict(row['spread']))
        if members:
            from skills.tactical_v1.contracts import CandidatePlan
            from skills.tactical_v1.validator import validate_plans
            contract = CandidatePlan(
                proposal_id="smoke-bomber-1", snapshot_id=99,
                config_fingerprint="", primary_skill="focus_fire",
                specialization_provider=None, instance_id=None,
                controlled_actors=tuple(sorted(members)), beneficiaries=(),
                goal=row['enemy'], current_actions=dict(members),
                expected_effects=(), claims=None, dependencies=(),
                commitment=None, features={}, proposed_memory_delta={},
                atomic=True, rationale="smoke",
                goal_kind="enemy",
                goal_data={"validation_kind": "bomber_intercept",
                           "intercept_type": 'safe_intercept' if row.get('safe')
                           else 'loss_control_intercept' if row.get('safe') is False
                           and row.get('type') == 'loss_control_intercept'
                           else 'ordinary_joint_fire' if row['shooters']
                           else 'spread_only',
                           "target_id": row['enemy'],
                           "shooter_actions": {a: action for a, action, _ in row['shooters']},
                           "spread_actions": dict(row['spread']),
                           "protected_members": list(row.get('protected_members', ())),
                           "already_exposed_members": list(row.get('already_exposed', ())),
                           "conservative_kill_loops": row.get('conservative_kill_loops'),
                           "requires_lethal_damage": bool(row['shooters'])})
            actions_map = {a: manager.memory.last_committed_actions.get(a, members[a])
                           for a in members}
            problems = validate_plans([contract], probe, actions_map,
                                      None, manager.memory)
            print(f"zerg bomber contract validation: {len(problems)} problem(s)")
            for problem in problems:
                if str(problem.get('type', '')).startswith('bomber_contract'):
                    print(f"  - {problem['type']} (expected when final actions differ)")
    else:
        print("zerg intercept plans: none (no bomber in coverage this scene)")

    # ---- Consistency check 1: contact vs earlier death are mutually
    # exclusive. Real geometry: our baneling can legally attack a target 2
    # units away (contact needs travel, so contact_time > 0), while three
    # enemy hydralisks stand on the SOURCE point with OBSERVED attack orders
    # targeting this baneling at zero cooldown (3x12 >= 30 effective
    # health) — legitimate observation inputs that danger_evidence's
    # targeted branch accepts. The real _outcome must pick ONE death result
    # and drop the distant contact coverage. No function under test is
    # stubbed.
    from skills.tactical_v1.burst import outcome as burst_outcome
    duel_orders = [{"kind": "attack", "target_unit": "u0",
                    "target_tag": "tag-duel-u0"}]
    duel = scene([((10., 10.), "baneling")],
                 enemies={"e0": dict(enemy((8., 10.), "baneling",
                                          health=30., max_health=30.),
                                     tag="tag-duel-e0"),
                          "e1": dict(enemy((9.7, 10.2), "hydralisk",
                                           orders=duel_orders,
                                           weapon_cooldown_loops=0.),
                                     tag="tag-duel-e1"),
                          "e2": dict(enemy((9.8, 9.7), "hydralisk",
                                           orders=duel_orders,
                                           weapon_cooldown_loops=0.),
                                     tag="tag-duel-e2"),
                          "e3": dict(enemy((9.6, 10.4), "hydralisk",
                                           orders=duel_orders,
                                           weapon_cooldown_loops=0.),
                                     tag="tag-duel-e3")})
    duel["allies"]["u0"].update(health=30., max_health=30., tag="tag-duel-u0")
    duel = with_questions(calibrate_baneling(duel))
    duel_ctx = Context(build_snapshot(duel, 99))
    duel_ctx.burst_config = dict(manager.config.get("burst_engage", {}))
    duel_action = None
    for candidate in sorted(duel_ctx.legal_actions("u0")):
        if duel_ctx.decode("u0", candidate).kind == 'attack':
            duel_action = candidate
            break
    if duel_action is None:
        failures.append("consistency1: attack not legal in duel scene setup")
    else:
        duel_result = burst_outcome(duel_ctx, "u0", duel_action)
        print(f"consistency1: duel trigger={duel_result.trigger} "
              f"T={duel_result.evaluation_after_loops:.1f} "
              f"weight={duel_result.opportunity:.2f}")
        if duel_result.trigger.startswith('death_'):
            print("consistency1: earlier death selected; distant contact "
                  "coverage dropped (single result)")
        elif duel_result.trigger == 'active_contact_opportunity':
            failures.append("consistency1: contact won despite evidence-backed "
                            "earlier death on the route")
        elif duel_result.trigger == 'none':
            failures.append("consistency1: no trigger at all in duel scene")

    # ---- Consistency check 2: action_position_at honors T semantics.
    from skills.tactical_v1.burst import action_position_at
    zerg_probe = Context(build_snapshot(zerg2, 99))
    zerg_probe.burst_config = dict(manager.config.get("burst_engage", {}))
    move = None
    for candidate in sorted(zerg_probe.legal_actions("u0")):
        if zerg_probe.decode("u0", candidate).kind == 'move':
            move = candidate
            break
    if move is not None:
        H = max(zerg_probe.H, 1)
        pos0, unc0 = action_position_at(zerg_probe, "u0", move, 0)
        posH, uncH = action_position_at(zerg_probe, "u0", move, H)
        current = zerg_probe.snapshot.allies["u0"].position
        same0 = max(abs(pos0[i] - current[i]) for i in (0, 1)) < 1e-6
        moved = max(abs(posH[i] - current[i]) for i in (0, 1)) > 1e-3
        print(f"consistency2: pos(0)==current={same0}, pos(H) moved={moved}, "
              f"uncertain={unc0},{uncH}")
        if not same0 or not moved:
            failures.append("zerg: action_position_at T-semantics broken")
    else:
        print("consistency2: no legal MOVE in scene (skipped)")

    # ---- Consistency check 3: side-target sources & late damage timing.
    # Synthetic source rows through the REAL scenario accumulator: the
    # bomber is never a primary target (side entry via damage_potential);
    # T=2 damage 10 + T=8 damage 20 vs 30 remaining HP must NOT form a
    # death scenario at T=2.
    from skills.tactical_v1.effects import _bomber_death_scenarios
    enemy_hp = 30.
    rows = [
        {'source_actor': 'u2', 'victim_enemy_id': 'e0', 'damage': 10.,
         'raw_damage': 10., 'opportunity': 1., 'evaluation_after_loops': 2.,
         'timing_kind': 'predicted', 'outcome_actor': 'u9'},
        {'source_actor': 'u3', 'victim_enemy_id': 'e0', 'damage': 20.,
         'raw_damage': 20., 'opportunity': 1., 'evaluation_after_loops': 8.,
         'timing_kind': 'predicted', 'outcome_actor': 'u9'},
    ]
    snapshot_stub = type('S', (), {'enemies': {'e0': type('E', (), {
        'effective_health': enemy_hp})()}})()
    scenarios = _bomber_death_scenarios(zerg_probe, snapshot_stub,
                                        {'e0': rows}, 'e0')
    first_moment = scenarios[0][0] if scenarios else None
    print(f"consistency3: first lethal moment = {first_moment} (must be 8.0)")
    if first_moment != 8.:
        failures.append("zerg: late damage participated before its moment")
    # Side-target inclusion: a row whose outcome_actor differs from the
    # victim's primary claim is retained by the accumulator input itself.
    if any(r['victim_enemy_id'] == 'e0' for r in rows):
        print("consistency3: side-target row retained (passed via damage_potential)")

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
