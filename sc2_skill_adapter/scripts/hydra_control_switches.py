"""Inject the two DEFAULT-OFF hydralisk research switches into COPIES of the
frozen contact_response snapshot. The frozen source, historical runs and the
workspace default policy are never touched: apply_variant() only edits a
copied label directory, and all four experiment groups receive byte-identical
switch code differing only in hydra_switches.json flags.

A isolated_escape / B protect_next_shot: see
hydra_switch_module_template.py for the full criteria.
"""
import json
from pathlib import Path

TEMPLATE = Path(__file__).with_name("hydra_switch_module_template.py")

SKILL_ANCHOR = (
    "        actions, summary, plans = self.coordinator.decide(\n"
    "            context, self.skills, self.memory, self.telemetry, n_agents)\n")
SKILL_HOOK = SKILL_ANCHOR + (
    "        # research switches (default off; flags in hydra_switches.json)\n"
    "        from . import hydra_switches as _hydra_switches\n"
    "        actions = _hydra_switches.apply(self, context, plans, summary, actions)\n")

COMMIT_ANCHOR = "        self.last_micro_intents = intents\n"
COMMIT_HOOK = ("        from . import hydra_switches as _hydra_switches\n"
               "        _hydra_switches.confirm(self, context, final_actions)\n"
               + COMMIT_ANCHOR)

FORMATION_ANCHOR = '            facts[a, cmd] = (end, gap, distance(end, slot), len(model.threats_at(a, end)),\n'
FORMATION_HOOK = ('''            if _switch_enabled:
                escapes = len(model.threats_at(a, end)) < origin_threats
                aligns = distance(end, slot) < distance(start, slot) - .05 and gap <= current_gap + .05
                role = ('hold' if cmd == wait else 'search' if not target else
                        'support' if cmd in counter_moves else 'entry' if advancing else
                        'alignment' if aligns else 'escape' if escapes else 'other')
                context.hydra_move_roles[a, cmd] = role
''' + FORMATION_ANCHOR)

RUNNER_INIT_ANCHOR = "            micro_records, micro_windows = [], 0\n"
RUNNER_INIT = RUNNER_INIT_ANCHOR + "            switch_rows = []\n"
RUNNER_COLLECT_ANCHOR = ("                    reward, terminated, info = "
                         "env.step_micro_window(actions, window, macro_loops=8)\n")
RUNNER_COLLECT = (
    "                    if telemetry.get('hydra_switch_decisions'):\n"
    "                        switch_rows.extend(telemetry['hydra_switch_decisions'])\n"
    + RUNNER_COLLECT_ANCHOR)
RECORD_ANCHOR = (
    "                write_json(output / 'micro_triggers' / f'episode-{episode:04d}.json', {\n"
    "                    'episode': episode, 'mode': execution_mode,\n"
    "                    'macro_loops': 8, 'engine_step_loops': 1,\n"
    "                    'authorizations': micro_records})\n")
RECORD_HOOK = RECORD_ANCHOR + (
    "            if switch_rows:\n"
    "                switch_states = {}\n"
    "                for row in switch_rows:\n"
    "                    if row.get('followup'):\n"
    "                        continue\n"
    "                    key = f\"{row['switch']}:{row['result']}\"\n"
    "                    switch_states[key] = switch_states.get(key, 0) + 1\n"
    "                record['hydra_switches'] = {\n"
    "                    'decisions': len(switch_rows), 'states': switch_states}\n"
    "                write_json(output / 'hydra_switches' / f'episode-{episode:04d}.json', {\n"
    "                    'episode': episode, 'decisions': switch_rows})\n")


def _patch(text, anchor, hook, where):
    if text.count(anchor) != 1:
        raise ValueError(f"anchor appears {text.count(anchor)} times in {where}; "
                         "refusing to patch a changed snapshot")
    return text.replace(anchor, hook)


def apply_variant(work, escape=False, shot_guard=False):
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
    module_dest = tactical / "hydra_switches.py"
    if any(p.exists() or p.is_symlink() for p in (module_dest, tactical / "hydra_switches.json")):
        raise FileExistsError(f"switches already injected: {module_dest}")
    manager_text = _patch(manager.read_text(encoding="utf-8"), SKILL_ANCHOR, SKILL_HOOK, manager)
    manager_text = _patch(manager_text, COMMIT_ANCHOR, COMMIT_HOOK, manager)
    # A withdrawal never acquires an approach authorization, even if it
    # happens to get closer to the formation's different, shared focus.
    micro_anchor = "        elif role == ('coordinated_engage', 'approaching'):\n"
    manager_text = _patch(manager_text, micro_anchor,
        "            if getattr(ctx, 'hydra_move_roles', {}).get((actor, cmd)) == 'escape':\n"
        "                continue\n" + micro_anchor, manager)
    validator_text = validator.read_text(encoding='utf-8')
    val_anchor = '        if action in MOVE_ACTIONS and ready_hydralisk_shots(context, actor) and (\n'
    validator_text = _patch(validator_text, val_anchor,
        "        if (context.snapshot.snapshot_id, actor, action) == getattr(context, '_hydra_escape_permit', None):\n"
        "            continue  # ONLY the ready-shot-before-withdrawal check for this one order\n"
        + val_anchor, validator)
    formation_text = _patch(formation.read_text(encoding='utf-8'), '    options, stationary, facts = {}, {}, {}\n',
        '    from ..hydra_switches import ENABLED as _switch_enabled\n'
        '    if _switch_enabled:\n        context.hydra_move_roles = {}\n'
        '    options, stationary, facts = {}, {}, {}\n', formation)
    formation_text = _patch(formation_text, FORMATION_ANCHOR, FORMATION_HOOK, formation)
    text = runner.read_text(encoding="utf-8")
    text = _patch(text, RUNNER_INIT_ANCHOR, RUNNER_INIT, "runner:switch_rows init")
    text = _patch(text, RUNNER_COLLECT_ANCHOR, RUNNER_COLLECT, "runner:row collect")
    text = _patch(text, RECORD_ANCHOR, RECORD_HOOK, "runner:episode record")
    module_dest.write_text(TEMPLATE.read_text(encoding="utf-8"), encoding="utf-8")
    (tactical / "hydra_switches.json").write_text(
        json.dumps({"isolated_escape": bool(escape),
                    "protect_next_shot": bool(shot_guard)}, indent=2) + "\n",
        encoding="utf-8")
    manager.write_text(manager_text, encoding="utf-8")
    formation.write_text(formation_text, encoding='utf-8')
    validator.write_text(validator_text, encoding='utf-8')
    runner.write_text(text, encoding="utf-8")
    return {"isolated_escape": bool(escape), "protect_next_shot": bool(shot_guard),
            "hydra_switches_module": str(module_dest)}
