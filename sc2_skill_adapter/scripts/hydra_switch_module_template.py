"""Two DEFAULT-OFF hydralisk research switches for the frozen contact_response
policy. This file is INJECTED into copies of the frozen tactical_v1 package
only (never the workspace default) and is inert unless hydra_switches.json
in the same directory enables a switch.

A  isolated_escape    One hydralisk per window whose FINAL arbitrated action
                      is an ATTACK may swap that single attack for one legal
                      one-step move, only with ALL three evidence classes:
                      isolated (currently threatened and no second living
                      hydralisk can return fire on that same threat),
                      unfavorable exchange (ling+hydra pincer, or sustained
                      bleeding while clearly behind on duel HP), and a
                      worthwhile move (sheds a threat without adding one, or
                      pulls the pursuit into ally coverage, never into an
                      ally's reserved destination). Keep attacking on an
                      immediate kill, or when the next shot is due and the
                      hydra is not critical. One macro step only; the window
                      is re-evaluated fresh next decision.

B  protect_next_shot  A hydra's FINAL retreat-class move (disengage ESCAPE
                      non-rescue, combat_reposition 'retreat', or an
                      exclusive formation escape, not entry/alignment) is
                      replaced by the preserved-target Attack order when the
                      target is legal NOW, the weapon is ready or ready
                      within the next macro step, and the move neither sheds
                      a threat, nor improves ally support, nor keeps the
                       existing range or a shared firing lane. Urgent rescues, real
                      relief and support-improving moves are never
                      intercepted.

Interplay: A runs first; B never touches an actor A already moved. A cannot
bypass its own worthwhile-move evidence; B cannot re-attack a valid escape.

Every judgement (applied / unchanged:<veto> / exempt:<reason> /
out_of_scope) is logged with actor, tag, loop, provenance, candidate, final
action, target, cooldown, health history and ally support. Destination
outcomes are geometric estimates on the current visible snapshot and are
flagged estimated=True; they are never logged as observed facts.
"""
import json
import math
from collections import deque
from pathlib import Path

from .skills.combat_reposition import CombatRepositionSkill
from .action_duties import rescue_actions
from .hydra_support import support_path_clear
from .joint_combat import action_position
from .validator import validate_actions, validate_plans

CONFIG_PATH = Path(__file__).with_name("hydra_switches.json")

DEFAULTS = {
    "isolated_escape": False,
    "protect_next_shot": False,
    "max_escapes_per_window": 1,
    "critical_health_fraction": 0.35,
    "duel_hp_ratio": 0.7,
    "recent_damage_windows": 2,
    "recent_damage_lookback": 3,
    "ready_within_loops": 8,
    "range_margin": 0.05,
    "followup_windows": 3,
}

UNIT_HYDRA = "hydralisk"


def load_config():
    cfg = dict(DEFAULTS)
    try:
        cfg.update(json.loads(CONFIG_PATH.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        pass  # missing or unreadable file keeps every switch OFF
    return cfg


CONFIG = load_config()
ENABLED = bool(CONFIG.get("isolated_escape")) or bool(CONFIG.get("protect_next_shot"))


def _new_state():
    return {"recent_damage": {}, "tracked": {}}


def _state(manager):
    state = getattr(manager, "_hydra_switch_state", None)
    if state is None:
        state = manager._hydra_switch_state = _new_state()
    return state


def _hydras(context):
    return [a for a, unit in sorted(context.snapshot.allies.items())
            if unit.alive and context.capabilities[a].key == UNIT_HYDRA]


def _support_map(context, actor, threats, mates):
    """Alive mate hydralisks that can return fire on each threat NOW.

    window_shots >= 1 is this codebase's own fire-support standard
    (the same test disengage_reenter._protect_shot uses). Observed snapshot
    fact, not an estimate.
    """
    support = {}
    for threat in threats:
        helpers = []
        for mate in mates:
            if mate == actor:
                continue
            if threat in context.attackable_enemies(mate) and \
                    context.fire_estimate(mate, threat).window_shots >= 1:
                helpers.append(mate)
        support[threat] = helpers
    return support


def _threat_types(context, threats):
    return {t: context.snapshot.enemies[t].type_name
            for t in threats if t in context.snapshot.enemies}


def _pursuer_into_ally_coverage(context, actor, point, threats, mates, actions):
    """Conditional *new* coverage for a melee pursuer following one step.

    A ranged threat standing at firing distance does not chase merely because
    the actor's destination happens to be inside another gun's range.
    """
    origin = context.snapshot.allies[actor].position
    for t in threats:
        enemy = context.snapshot.enemies[t]
        if enemy.type_name != 'zergling' or math.dist(enemy.position, point) <= math.dist(enemy.position, origin):
            continue
        followed = tuple(e + p - o for e, p, o in zip(enemy.position, point, origin))
        for mate in mates:
            if mate == actor:
                continue
            reach = context.effective_attack_reach(mate, t)
            ally = context.snapshot.allies[mate]
            if ally.weapon_cooldown_loops > context.H:
                continue
            spot = action_position(context, mate, actions[ally.index])[0]
            if reach is not None and math.dist(spot, followed) <= reach < math.dist(spot, enemy.position):
                return {'mate': mate, 'threat': t}
    return None


def _covers_ally_threat(context, actor, point, reach_of):
    """Estimate: from `point` this actor could newly cover an ally's current
    threat that it cannot cover from where it stands."""
    for mate, mate_unit in sorted(context.snapshot.allies.items()):
        if mate == actor or not mate_unit.alive:
            continue
        for threat in context.risk_at(mate, mate_unit.position).threats:
            enemy = context.snapshot.enemies.get(threat)
            if enemy is None:
                continue
            now_reach = reach_of(actor, threat)
            if now_reach is not None and now_reach >= 0 and \
                    context.distance(actor, threat) <= now_reach:
                continue
            future = reach_of(actor, threat)
            if future is not None and future >= 0 and \
                    math.hypot(point[0] - enemy.position[0],
                               point[1] - enemy.position[1]) \
                     <= future:
                return f"covers_ally_threat:{mate}:{threat}"
    return None


def _plan_for(plans, actor):
    for plan in plans or ():
        if actor in plan.controlled_actors:
            return plan
    return None


def _base_row(context, actor, action_id, kind, target):
    unit = context.snapshot.allies[actor]
    tag = str((context.snapshot.raw_state.get("allies", {}).get(actor, {})
               or {}).get("tag", ""))
    target_tag = None
    if isinstance(target, str):
        raw = (context.snapshot.raw_state.get("visible_enemies", {}).get(target)
               or {})
        target_tag = str(raw.get("tag", "")) or None
    return {
        "actor": actor, "actor_tag": tag, "game_loop": context.snapshot.game_loop,
        "window": context.snapshot.snapshot_id,
        "original_action": action_id, "original_kind": kind,
        "original_target": target, "target_tag": target_tag,
        "candidate_action": None, "final_action": action_id, "changed": False,
        "weapon_cooldown_loops": unit.weapon_cooldown_loops,
        "weapon_ready": unit.weapon_ready,
        "health": unit.health, "shield": unit.shield,
        "health_fraction": round(unit.health_fraction, 4),
        "health_lost": unit.health_lost, "shield_lost": unit.shield_lost,
        "estimated": True, "result": None, "veto": None,
    }


# ------------------------------------------------------------------ pass A
def _evaluate_A(context, actor, actions, plans, summary, state, cfg, mates):
    unit = context.snapshot.allies[actor]
    action_id = actions[unit.index]
    decoded = context.decode(actor, action_id)
    row = _base_row(context, actor, action_id, decoded.kind, decoded.target)
    plan = _plan_for(plans, actor)
    role = summary.get('units', {}).get(actor, {})
    row['original_skill'] = plan.primary_skill if plan else role.get('skill')
    row['original_phase'] = plan.commitment.phase if plan else role.get('phase')
    row['original_rationale'] = plan.rationale if plan else None
    risk = context.risk_at(actor, unit.position)
    threats = list(risk.threats)
    row["current_threats"] = threats
    row["threat_types"] = _threat_types(context, threats)
    row["switch"] = "A"
    if decoded.kind != "attack" or not context.can_issue_attack(actor, decoded.target) or context.contact_status(actor, decoded.target) != 'in_range':
        row["result"] = "out_of_scope:final_action_not_attack"
        return row, None
    if risk.heuristic:
        row.update(result='unchanged:threat_range_unknown', veto='missing_native_range')
        return row, None
    if not threats:
        row["result"] = "unchanged:no_current_threat"
        row["veto"] = "no_current_threat"
        return row, None
    support = _support_map(context, actor, threats, mates)
    row["ally_support"] = support
    if any(support[t] for t in threats):
        row["result"] = "unchanged:ally_can_return_fire"
        row["veto"] = "ally_fire_support_on_a_threat"
        return row, None
    killable = next((t for t in context.attackable_enemies(actor)
                     if (context.fire_estimate(actor, t).damage_est or 0)
                     >= context.snapshot.enemies[t].effective_health), None)
    if killable is not None:
        row["result"] = "unchanged:immediate_kill_available"
        row["veto"] = "immediate_kill_available"
        return row, None
    history = list(state["recent_damage"].get(actor, ()))
    row['observed_loss_history'] = history
    sustained = len(history) >= cfg['recent_damage_windows'] and all(history[-cfg['recent_damage_windows']:])
    behind = decoded.target in threats and unit.effective_health < cfg['duel_hp_ratio'] * context.snapshot.enemies[decoded.target].effective_health
    bleeding = unit.health_lost + unit.shield_lost > 0
    pincer = {"zergling", UNIT_HYDRA} <= set(row["threat_types"].values())
    if pincer:
        unfavorable = "pincer_ling_and_hydra"
    elif bleeding and sustained and behind:
        unfavorable = "sustained_bleeding_behind_on_hp"
    else:
        unfavorable = None
    row["unfavorable_exchange"] = unfavorable
    row["duel_hp_behind"] = bool(behind)
    row["sustained_bleeding"] = bool(sustained)
    if unfavorable is None:
        row["result"] = "unchanged:exchange_not_unfavorable"
        row["veto"] = ("single_window_loss_only" if bleeding
                       else "no_pincer_and_no_sustained_bleeding")
        return row, None
    critical = unit.health_fraction <= cfg["critical_health_fraction"]
    row["critical"] = critical
    if (unit.weapon_ready or unit.weapon_cooldown_loops <= cfg['ready_within_loops']) and not critical:
        row["result"] = "unchanged:weapon_ready_prefer_fire"
        row["veto"] = "weapon_ready_and_not_critical" if unit.weapon_ready else 'next_shot_due_and_not_critical'
        return row, None
    now = set(threats)
    joint = {other: actions[u.index] for other, u in context.snapshot.allies.items()}
    options = []
    for move in sorted(context.move_destinations(actor)):
        dest = context.predict_position(actor, move).position
        memory = getattr(context, 'deployment_memory', None)
        if context.speed(actor) is None or (memory is not None and memory.movement_blockage(context, actor, move) >= .5):
            continue
        if not support_path_clear(context, actor, dest, joint):
            continue
        mid = tuple((a + b) / 2 for a, b in zip(dest, unit.position))
        left, right = (context.risk_at(actor, p) for p in (mid, dest))
        after = right.threats
        if left.effect_weapon_near or right.effect_weapon_near or (set(left.threats) | set(after)) - now:
            continue                   # no new crossfire
        shed = sorted(now - set(after))
        coverage = _pursuer_into_ally_coverage(context, actor, dest, threats, mates, actions)
        if not shed and coverage is None:
            continue
        options.append((len(shed), -len(after), -move, move, dest, shed, coverage))
    if not options:
        row["result"] = "unchanged:no_worthwhile_move"
        row["veto"] = "every_move_adds_threat_or_keeps_pressure"
        return row, None
    options.sort(reverse=True)
    _, _, _, move, dest, shed, coverage = options[0]
    row["candidate_action"] = move
    row["candidate_kind"] = "move"
    row["candidate_destination"] = [round(dest[0], 3), round(dest[1], 3)]
    row["worthwhile_move"] = {"shed": shed, "pursuer_covered_by": coverage,
                              "estimated": True}
    row["result"] = "candidate_selected"
    return row, move


# ------------------------------------------------------------------ pass B
def _preserved_target(context, actor, plan, role, manager, forced):
    """The shot the policy hierarchy wants kept: plan target, then the
    mandated hydra target, then memory's battle target, then the live
    order's target. Must be visible and mask-legal right now."""
    enemies = context.snapshot.enemies
    candidates = []
    stationary = getattr(context, '_debug_stationary', {}).get(actor)
    if role.get('skill') == 'ranged_fire_formation' and stationary is not None:
        candidates.append(context.decode(actor, stationary).target)
    if plan is not None:
        data = plan.goal_data or {}
        if isinstance(data.get("target_id"), str):
            candidates.append(data["target_id"])
        elif plan.goal_kind == "enemy" and isinstance(plan.goal, str):
            candidates.append(plan.goal)
    if role.get('goal_kind') == 'enemy':
        candidates.append(role.get('goal'))
    tags_to_id = {}
    for enemy_id, raw in (context.snapshot.raw_state.get("visible_enemies",
                                                         {}) or {}).items():
        tag = str(raw.get("tag", ""))
        if tag:
            tags_to_id[tag] = enemy_id
    for order in (context.snapshot.raw_state.get("allies", {}).get(actor, {})
                  or {}).get("orders") or []:
        if order.get('kind') != 'attack':
            continue
        enemy = tags_to_id.get(str(order.get("target_tag", "")))
        if enemy is not None:
            candidates.append(enemy)
    battle = manager.memory.battle_target(context, actor)
    candidates.extend([battle, *forced])
    seen = set()
    for target in candidates:
        if not isinstance(target, str) or target in seen:
            continue
        seen.add(target)
        if target in enemies and (not forced or target in forced) and context.can_issue_attack(actor, target):
            return target
    return None


def _evaluate_B(context, actor, actions, plans, summary, manager, cfg, mates):
    unit = context.snapshot.allies[actor]
    action_id = actions[unit.index]
    decoded = context.decode(actor, action_id)
    row = _base_row(context, actor, action_id, decoded.kind, decoded.target)
    row["switch"] = "B"
    plan = _plan_for(plans, actor)
    per_unit = (summary.get("units", {}) or {}).get(actor, {})
    row["original_skill"] = plan.primary_skill if plan else per_unit.get("skill")
    row["original_phase"] = plan.commitment.phase if plan else per_unit.get("phase")
    row["original_rationale"] = plan.rationale if plan else None
    threats = list(context.risk_at(actor, unit.position).threats)
    row["current_threats"] = threats
    row["threat_types"] = _threat_types(context, threats)
    if decoded.kind != 'move':
        row['result'] = 'out_of_scope:final_action_not_a_move'
        return row, None
    if action_id not in context.move_destinations(actor):
        row["result"] = "out_of_scope:move_without_destination"
        return row, None
    dest = context.predict_position(actor, action_id).position
    after = context.risk_at(actor, dest).threats
    relief = len(after) < len(threats)
    target = _preserved_target(context, actor, plan, per_unit, manager, context.forced_hydra_targets(actor))
    row['current_target'] = target
    if target:
        row['target_tag'] = str(context.snapshot.raw_state['visible_enemies'][target].get('tag') or '') or None
    row['retreat_geometry'] = bool(target and CombatRepositionSkill({})._classify(context, actor, action_id, target, threats) == 'retreat')
    row["move_sheds_threat"] = relief
    objective = ((plan.goal_data or {}).get("objective") or {}) if plan else {}
    reason = str(objective.get("execution_reason", ""))
    duty = summary.get('execution_duties', {}).get('actors', {}).get(actor, {})
    rescue = actor in summary.get('joint_rescue_substitutions', {}) or duty.get('kind') == 'rescue' or bool(objective.get("one_window_rescue")) \
        or "lethal" in reason or "blast" in reason
    row["provenance_rescue"] = rescue
    name, phase = row["original_skill"], row["original_phase"]
    rationale = str(row["original_rationale"] or "")
    if rescue:
        # Emergency relief is out of B's reach even when its provenance is
        # an escape skill; log the exemption, never intercept it.
        row["result"] = "exempt:urgent_rescue"
        return row, None
    formation_role = getattr(context, 'hydra_move_roles', {}).get((actor, action_id))
    row['formation_move_role'] = formation_role
    in_scope = (
        (name == "disengage_reenter" and phase == "ESCAPE")
        or (name == "combat_reposition" and rationale.startswith("retreat"))
        or (name == 'ranged_fire_formation' and phase == 'exclusive_control' and formation_role == 'escape'))
    if not in_scope:
        row["result"] = "out_of_scope:provenance_not_retreat"
        row["veto"] = f"move_from_{name}:{phase}_is_not_a_retreat"
        return row, None
    if action_id in rescue_actions(context, actor, manager.memory):
        row.update(result='exempt:urgent_rescue', veto='frozen_survivable_rescue')
        return row, None
    if duty.get('isolated_shooter_support', {}).get('satisfied'):
        row.update(result='exempt:improves_support', veto='selected_support_assignment')
        return row, None
    if context.speed(actor) is None:
        row.update(result='unchanged:move_reach_unknown', veto='missing_native_speed')
        return row, None
    if relief:
        row["result"] = "exempt:real_relief"
        return row, None
    reach_of = lambda a, t: context.effective_attack_reach(a, t)
    improves = _covers_ally_threat(context, actor, dest, reach_of) \
        or _pursuer_into_ally_coverage(context, actor, dest, threats, mates, actions)
    row["improves_support"] = improves if isinstance(improves, str) \
        else bool(improves)
    if improves:
        row["result"] = "exempt:improves_support"
        row["veto"] = improves if isinstance(improves, str) \
            else "pursuer_enters_ally_coverage"
        return row, None
    if target is None:
        row["result"] = "unchanged:no_preservable_target"
        row["veto"] = "no_legal_target_to_preserve"
        return row, None
    if not (unit.weapon_ready or
            unit.weapon_cooldown_loops <= cfg["ready_within_loops"]):
        row["result"] = "unchanged:weapon_not_ready_in_time"
        row["veto"] = "cooldown_beyond_next_macro_step"
        return row, None
    enemy = context.snapshot.enemies[target]
    reach = context.effective_attack_reach(actor, target)
    in_range_now = context.contact_status(actor, target) == "in_range"
    row["in_range_now"] = in_range_now
    if not in_range_now:
        row["result"] = "unchanged:no_existing_range_to_lose"
        row["veto"] = "target_not_in_range_before_move"
        return row, None
    dest_dist = math.hypot(dest[0] - enemy.position[0],
                           dest[1] - enemy.position[1])
    keeps = reach is None or reach < 0 or dest_dist <= reach + cfg["range_margin"]
    row["destination_distance"] = round(dest_dist, 3)
    row["attack_reach"] = None if reach is None else round(reach, 3)
    row["keeps_range_estimate"] = bool(keeps)
    shared_lost = []
    for other_target in context.attackable_enemies(actor):
        shared_reach = context.effective_attack_reach(actor, other_target)
        if (shared_reach is not None and shared_reach >= 0
                and context.contact_status(actor, other_target) == 'in_range'
                and math.dist(dest, context.snapshot.enemies[other_target].position) > shared_reach + cfg['range_margin']
                and any(other_target in context.attackable_enemies(a) and context.contact_status(a, other_target) == 'in_range'
                        for a in mates if a != actor)):
            shared_lost.append(other_target)
    row['shared_targets_lost_estimate'] = shared_lost
    if keeps and not shared_lost:
        row["result"] = "unchanged:keeps_range"
        row["veto"] = "destination_still_reaches_target"
        return row, None
    shot = context.attack_action(actor, target)
    if shot is None:
        row["result"] = "unchanged:target_action_illegal"
        row["veto"] = "attack_not_on_legal_menu"
        return row, None
    row["candidate_action"] = shot
    row["candidate_kind"] = "attack"
    row["result"] = "candidate_selected"
    return row, shot


# --------------------------------------------------------------- follow-ups
def _followups(state, context, cfg):
    rows = []
    loop = context.snapshot.game_loop
    for key, track in list(state["tracked"].items()):
        actor = track['actor']
        unit = context.snapshot.allies.get(actor)
        alive = unit.alive if unit is not None and str(unit.tag) == track['actor_tag'] else None
        cooldown_restarted = None
        if alive:
            cooldown_restarted = unit.weapon_cooldown_loops > track['last_cd']
            track['last_cd'] = unit.weapon_cooldown_loops
        threat_state = {}
        if alive:
            for threat in track.get("threats", ()):
                raw = context.snapshot.raw_state['visible_enemies'].get(threat, {})
                if threat not in context.snapshot.enemies or str(raw.get('tag')) != track['threat_tags'][threat]:
                    threat_state[threat] = "not_visible"
                    continue
                covered = any(threat in context.attackable_enemies(mate)
                              and context.contact_status(mate, threat) == 'in_range'
                              for mate in _hydras(context) if mate != actor)
                threat_state[threat] = "in_ally_range" if covered else "not_in_ally_range"
        row = {
            "switch": track["switch"], "actor": actor,
            "actor_tag": track["actor_tag"], "followup": True,
            "decision_loop": track["loop"], "game_loop": loop,
            "current_target": track['current_target'], "target_tag": track['target_tag'],
            "threat_tags": track['threat_tags'], "submitted_action": track['action'],
            "windows_since": context.snapshot.snapshot_id - track["window"],
            "alive": alive, "health": unit.health if alive is not None else None,
            "weapon_cooldown_loops": unit.weapon_cooldown_loops if alive else None,
            "weapon_ready": unit.weapon_ready if alive else None,
            "cooldown_restarted": cooldown_restarted,  # proxy; not shot proof
            "threat_state": threat_state,
            "survived_16_loops": alive if loop - track['loop'] == 16 else None,
            "survived_24_loops": alive if loop - track['loop'] == 24 else None,
            "estimated": True,
        }
        keep = alive and (loop - track["loop"]) < 8 * cfg["followup_windows"]
        if alive is False:
            row["final"] = True
            row["outcome"] = "died_observed"
            row["died_observed_loop"] = loop
            del state['tracked'][key]
        elif not keep:
            row["final"] = True
            row["outcome"] = "followup_window_ended"
            del state['tracked'][key]
        rows.append(row)
    return rows


# ------------------------------------------------------------------- entry
def _select(manager, context, plans, actions, changed, row):
    """Validate the complete substitute, including surviving plan contracts."""
    actor = row['actor']
    trial = list(actions)
    trial[context.snapshot.allies[actor].index] = row['candidate_action']
    joint = {f'u{i}': action for i, action in enumerate(trial)}
    released = changed | {actor}
    kept = [p for p in plans if released.isdisjoint(p.controlled_actors)]
    reasons = []
    permit = (context.snapshot.snapshot_id, actor, row['candidate_action']) if row['switch'] == 'A' else None
    previous = getattr(context, '_hydra_escape_permit', None)
    allowed = permit or previous
    for a, cmd in joint.items():
        targets = context.forced_hydra_targets(a)
        if targets and context.decode(a, cmd).target not in targets and (context.snapshot.snapshot_id, a, cmd) != allowed:
            reasons.append('forced_hydra_targets')
    if any(len(p.controlled_actors) > 1 and actor in p.controlled_actors for p in plans):
        reasons.append('joint_plan_contract')
    for a, end in getattr(context, 'formation_reserved_paths', {}).items():
        if a not in released and not support_path_clear(context, a, end, joint):
            reasons.append('reserved_path_obstructed')
    if allowed is not None:
        _, a, cmd = allowed
        if not support_path_clear(context, a, context.predict_position(a, cmd).position, joint):
            reasons.append('escape_path_obstructed')
    reasons.extend(p['type'] for p in validate_actions(trial, context, context.snapshot.n_allies))
    if not reasons:
        effects = manager.evaluator.effects(context, joint)
        context._hydra_escape_permit = allowed
        try:
            reasons.extend(p['type'] for p in validate_plans(kept, context, joint, effects, manager.memory))
        finally:
            context._hydra_escape_permit = previous
    if reasons:
        row.update(result='unchanged:validation_veto', veto=reasons[0], validation_errors=reasons)
        return False
    actions[:] = trial
    changed.add(actor)
    if permit is not None:
        context._hydra_escape_permit = permit
        row['one_window_permit'] = list(permit)
    row.update(result='applied', changed=True, final_action=row['candidate_action'])
    return True


def confirm(manager, context, final_actions):
    """Start follow-up only for the switch orders the runner actually commits."""
    if not ENABLED:
        return
    state = _state(manager)
    for row in state.pop('pending', ()):
        actor = row['actor']
        unit = context.snapshot.allies[actor]
        if final_actions[unit.index] != row['final_action']:
            continue
        key = row['switch'], row['actor_tag'], row['game_loop']
        state['tracked'][key] = dict(actor=actor, actor_tag=row['actor_tag'], switch=row['switch'],
            loop=row['game_loop'], window=row['window'], action=row['final_action'],
            current_target=row.get('current_target', row['original_target']), target_tag=row['target_tag'],
            threats=row['current_threats'], threat_tags=row['threat_tags'], last_cd=unit.weapon_cooldown_loops)


def apply(manager, context, plans, summary, actions):
    """Post-arbitration hook; returns the (possibly) modified action list."""
    if not ENABLED:
        return actions
    cfg = CONFIG
    snap = context.snapshot
    if snap.step_mul != 8 or any(u.type_name not in ('hydralisk', 'zergling') for u in snap.allies.values()):
        return actions
    if snap.snapshot_id == 1:
        manager._hydra_switch_state = _new_state()
    state = _state(manager)
    rows = _followups(state, context, cfg)
    mates = _hydras(context)
    for actor in mates:
        unit = snap.allies[actor]
        history = state['recent_damage'].setdefault(actor, deque(maxlen=cfg['recent_damage_lookback']))
        history.append(unit.health_lost + unit.shield_lost)
    overridden = set()
    if cfg.get("isolated_escape"):
        candidates = []
        for actor in mates:
            row, move = _evaluate_A(context, actor, actions, plans, summary, state,
                                    cfg, mates)
            rows.append(row)
            if move is not None:
                candidates.append((row.get("health_fraction", 1.0),
                                   -len(row["worthwhile_move"]["shed"]),
                                   actor, move, row))
        if candidates:
            candidates.sort()
            applied = 0
            for _, _, actor, move, row in candidates:
                if applied >= cfg.get("max_escapes_per_window", 1):
                    row["result"] = "unchanged:window_cap_reached"
                    row["veto"] = "one_escape_per_window"
                    continue
                applied += _select(manager, context, plans, actions, overridden, row)
    if cfg.get("protect_next_shot"):
        for actor in mates:
            if actor in overridden:
                continue          # A owns this actor this window
            row, shot = _evaluate_B(context, actor, actions, plans, summary,
                                    manager, cfg, mates)
            rows.append(row)
            if shot is None:
                continue
            _select(manager, context, plans, actions, overridden, row)
    state['pending'] = [r for r in rows if r.get('changed')]
    for r in rows:
        if r.get('followup'):
            continue
        a = r['actor']
        r['observed_loss_history'] = list(state['recent_damage'][a])
        r['threat_tags'] = {t: str(snap.raw_state['visible_enemies'][t].get('tag')) for t in r['current_threats']}
        if 'ally_support' not in r:
            r['ally_support'] = _support_map(context, a, r['current_threats'], mates)
        r.setdefault('current_target', r['original_target'])
        r['conditions_met'] = r['candidate_action'] is not None
        r['selected_for_commit'] = r['changed']
        r['final_action'] = actions[snap.allies[a].index]
    if overridden:
        for r in state['pending']:
            a = r['actor']
            kind = 'isolated_escape' if r['switch'] == 'A' else 'protect_next_shot'
            summary.setdefault('actions', {})[a] = r['final_action']
            summary.setdefault('units', {})[a] = dict(action=r['final_action'], skill=kind, phase='one_window',
                goal=r.get('current_target') or '', goal_kind='enemy' if r['switch'] == 'B' else '', provider=None, plan=None)
        removed = {p.proposal_id for p in plans if not overridden.isdisjoint(p.controlled_actors)}
        plans[:] = [p for p in plans if p.proposal_id not in removed]
        summary['chosen_plans'] = [p for p in summary.get('chosen_plans', ()) if p['id'] not in removed]
        summary['coordinator_assessment_scope'] = 'pre_switch; units and actions are post_switch'
    if rows:
        summary["hydra_switch_decisions"] = rows
    return actions
