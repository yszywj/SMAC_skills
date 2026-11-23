"""Inject the round-2 DEFAULT-OFF hydralisk switches (A2 supported_escape,
B2 guard_support_move) into COPIES of the frozen contact_response snapshot.
The round-1 experiment files (hydra_switches*, run_hydra_switches_ab.py) and
their meaning are untouched; the frozen source, historical runs, the shared
SMACv2 install and the workspace default policy are never modified.

All four experiment groups receive byte-identical switch code differing only
in hydra_supported.json flags.
"""
import json
from pathlib import Path

TEMPLATE = Path(__file__).with_name("hydra_supported_module_template.py")
MODULE = "hydra_supported"
FLAGS = ("supported_escape", "guard_support_move")

SKILL_ANCHOR = (
    "        actions, summary, plans = self.coordinator.decide(\n"
    "            context, self.skills, self.memory, self.telemetry, n_agents)\n")
SKILL_HOOK = SKILL_ANCHOR + (
    "        # round-2 research switches (default off; hydra_supported.json)\n"
    "        from . import hydra_supported as _supported\n"
    "        actions = _supported.apply(self, context, plans, summary, actions)\n")

COMMIT_ANCHOR = "        self.last_micro_intents = intents\n"
COMMIT_HOOK = ("        from . import hydra_supported as _supported\n"
               "        _supported.confirm(self, context, final_actions)\n"
               + COMMIT_ANCHOR)

FORMATION_ROLES_ANCHOR = ('    options, stationary, facts = {}, {}, {}\n')
FORMATION_ROLES_HOOK = (
    '    from ..hydra_supported import ENABLED as _supported_enabled\n'
    '    if _supported_enabled:\n        context.hydra_move_roles = {}\n'
    + FORMATION_ROLES_ANCHOR)
FORMATION_ROLES_BODY_ANCHOR = (
    '            facts[a, cmd] = (end, gap, distance(end, slot), len(model.threats_at(a, end)),\n')
FORMATION_ROLES_BODY_HOOK = (
    '            if _supported_enabled:\n'
    '                escapes = len(model.threats_at(a, end)) < origin_threats\n'
    '                aligns = distance(end, slot) < distance(start, slot) - .05 and gap <= current_gap + .05\n'
    '                role = (\'hold\' if cmd == wait else \'search\' if not target else\n'
    '                        \'support\' if cmd in counter_moves else \'entry\' if advancing else\n'
    '                        \'alignment\' if aligns else \'escape\' if escapes else \'other\')\n'
    '                context.hydra_move_roles[a, cmd] = role\n'
    + FORMATION_ROLES_BODY_ANCHOR)

# A2's one-window permit: the ONLY order allowed to withdraw before its next
# shot is due, without silently relaxing the rule for everyone else.
VALIDATOR_ANCHOR = ('        if action in MOVE_ACTIONS and ready_hydralisk_shots(context, actor) and (\n')
VALIDATOR_HOOK = (
    '        if (context.snapshot.snapshot_id, actor, action) == getattr(context, "_hydra_escape_permit", None):\n'
    '            continue  # ONLY the ready-shot-before-withdrawal check for this one order\n'
    + VALIDATOR_ANCHOR)

# A withdrawal never acquires an approach authorization, even if it happens
# to close on the formation's different, shared focus.
MICRO_ANCHOR = "        elif role == ('coordinated_engage', 'approaching'):\n"
MICRO_HOOK = (
    "            if getattr(ctx, 'hydra_move_roles', {}).get((actor, cmd)) == 'escape':\n"
    "                continue\n" + MICRO_ANCHOR)

RUNNER_INIT_ANCHOR = "            micro_records, micro_windows = [], 0\n"
RUNNER_INIT = RUNNER_INIT_ANCHOR + "            switch_rows = []\n"
RUNNER_COLLECT_ANCHOR = '                    skill_manager.commit_final_actions(actions)\n'
RUNNER_COLLECT = (
    RUNNER_COLLECT_ANCHOR +
    "                    for switch_row in telemetry.get('hydra_supported_decisions', ()):\n"
    "                        switch_row.update(episode=episode, sc2_seed=scenario['sc2_seed'])\n"
    "                        if not switch_row.get('followup'):\n"
    "                            actual = actions[int(switch_row['actor'][1:])]\n"
    "                            switch_row['final_action'] = actual\n"
    "                            if switch_row['changed'] and actual != switch_row['candidate_action']:\n"
    "                                switch_row.update(changed=False, result='not_committed:final_mismatch')\n"
    "                        switch_rows.append(switch_row)\n")
RUNNER_RECEIPT_ANCHOR = "                feedback = action_feedback(env, loop)\n"
RUNNER_RECEIPT_HOOK = RUNNER_RECEIPT_ANCHOR + (
    "                if switch_rows:\n"
    "                    switch_receipts = feedback.get('actors', {})\n"
    "                    for switch_row in reversed(switch_rows):\n"
    "                        if switch_row['game_loop'] != loop:\n"
    "                            break\n"
    "                        if switch_row.get('changed') and switch_row.get('actor') in switch_receipts:\n"
    "                            receipt = switch_receipts[switch_row['actor']]\n"
    "                            response = receipt.get('response') or {}\n"
    "                            switch_row['sc2_receipt'] = dict(\n"
    "                                requested_action_id=receipt.get('requested_action_id'),\n"
    "                                submission=receipt.get('submission'),\n"
    "                                code=response.get('code'), name=response.get('name'))\n")
RECORD_ANCHOR = (
    "                write_json(output / 'micro_triggers' / f'episode-{episode:04d}.json', {\n"
    "                    'episode': episode, 'mode': execution_mode,\n"
    "                    'macro_loops': 8, 'engine_step_loops': 1,\n"
    "                    'authorizations': micro_records})\n")
RECORD_HOOK = RECORD_ANCHOR + (
    "            if switch_rows:\n"
    "                switch_states = {}\n"
    "                for switch_row in switch_rows:\n"
    "                    if switch_row.get('followup'):\n"
    "                        continue\n"
    "                    key = f\"{switch_row['switch']}:{switch_row['result']}\"\n"
    "                    switch_states[key] = switch_states.get(key, 0) + 1\n"
    "                record['hydra_supported'] = {\n"
    "                    'decisions': len(switch_rows), 'states': switch_states}\n"
    "                write_json(output / 'hydra_supported' / f'episode-{episode:04d}.json', {\n"
    "                    'episode': episode, 'decisions': switch_rows})\n")


def _patch(text, anchor, hook, where):
    if text.count(anchor) != 1:
        raise ValueError(f"anchor appears {text.count(anchor)} times in {where}; "
                         "refusing to patch a changed snapshot")
    return text.replace(anchor, hook)


def apply_variant(work, supported_escape=False, guard_support_move=False):
    """Patch one copied label directory (skills/ + sc2_skill_adapter/)."""
    work = Path(work).resolve()
    shared = TEMPLATE.resolve().parents[2]
    if work == shared or 'source_snapshot' in work.parts:
        raise ValueError(f"only inject into a working copy: {work}")
    tactical = work / "skills" / "tactical_v1"
    manager = tactical / "skill_manager.py"
    formation = tactical / 'skills/ranged_fire_formation.py'
    validator = tactical / 'validator.py'
    runner = work / "sc2_skill_adapter" / "bridge" / "runner.py"
    for path in (manager, runner, formation, validator):
        if path.resolve() != path or path.stat().st_nlink != 1:
            raise ValueError(f"not an independent working copy: {path}")
    module_dest = tactical / f"{MODULE}.py"
    if any(p.exists() or p.is_symlink() for p in
           (module_dest, tactical / "hydra_supported.json")):
        raise FileExistsError(f"switches already injected: {module_dest}")
    manager_text = _patch(manager.read_text(encoding="utf-8"),
                          SKILL_ANCHOR, SKILL_HOOK, manager)
    manager_text = _patch(manager_text, COMMIT_ANCHOR, COMMIT_HOOK, manager)
    manager_text = _patch(manager_text, MICRO_ANCHOR, MICRO_HOOK, manager)
    validator.write_text(_patch(validator.read_text(encoding='utf-8'),
                                VALIDATOR_ANCHOR, VALIDATOR_HOOK, validator),
                         encoding='utf-8')
    formation_text = _patch(formation.read_text(encoding='utf-8'),
                            FORMATION_ROLES_ANCHOR, FORMATION_ROLES_HOOK, formation)
    formation_text = _patch(formation_text, FORMATION_ROLES_BODY_ANCHOR,
                            FORMATION_ROLES_BODY_HOOK, formation)
    formation.write_text(formation_text, encoding='utf-8')
    text = runner.read_text(encoding="utf-8")
    text = _patch(text, RUNNER_INIT_ANCHOR, RUNNER_INIT, "runner:switch_rows init")
    text = _patch(text, RUNNER_COLLECT_ANCHOR, RUNNER_COLLECT, "runner:row collect")
    text = _patch(text, RUNNER_RECEIPT_ANCHOR, RUNNER_RECEIPT_HOOK, "runner:receipts")
    text = _patch(text, RECORD_ANCHOR, RECORD_HOOK, "runner:episode record")
    runner.write_text(text, encoding="utf-8")
    module_dest.write_text(TEMPLATE.read_text(encoding="utf-8"), encoding="utf-8")
    (tactical / "hydra_supported.json").write_text(
        json.dumps({"supported_escape": bool(supported_escape),
                    "guard_support_move": bool(guard_support_move)},
                   indent=2) + "\n", encoding="utf-8")
    manager.write_text(manager_text, encoding="utf-8")
    return {"supported_escape": bool(supported_escape),
            "guard_support_move": bool(guard_support_move),
            "hydra_supported_module": str(module_dest)}
