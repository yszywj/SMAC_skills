"""Two DEFAULT-OFF hydralisk research switches (round 2) for the frozen
contact_response policy. INJECTED into copies of the frozen tactical_v1
package only (as hydra_supported.py); inert unless hydra_supported.json in
the same directory enables a switch. The round-1 experiment (A/B) and its
meaning are untouched.

A2  supported_escape   Replaces "forecast pressure relief" with a named
                       support triple. When the FINAL arbitrated action of a
                       threatened hydralisk is an ATTACK, the row records the
                       threatened actor, the specific threatening enemy, and
                       the specific mate that could support against that
                       enemy. Only when no ally can return fire now AND one
                       legal, conflict-free one-step destination exists that
                       puts a NAMED mate's reach over the landing point does
                       the attack become that one move. Two evidence classes
                       are kept apart: 'now' (the mate's reach already covers
                       the destination) and 'if_pursued' (coverage only forms
                       if the enemy advances to its far engagement envelope -
                       a conditional estimate, never reported as realized
                       support). Moving nearer an ally, or shedding pressure
                       alone, is not support. Triggering needs the post-fire
                       cooldown window, or observed close pressure / a clearly
                       unfavorable exchange; an immediate kill keeps the
                       shot. One actor, one action, one macro step; original
                       emergency escapes are never blocked, and the withdrawal
                       never acquires a micro-step attack authorization.

B2  guard_support_move Only formation moves whose intent the formation itself
                       classified as support/entry/alignment are examined
                       (escape-classified formation moves and unreadable
                       intents are out of scope - the definition of retreat
                       is NOT widened). The row records which enemy the move
                       supports against and which mates already cover that
                       enemy. The move becomes the preserved-target Attack
                       only when the target is still visible and legally
                       nameable, the weapon is ready or ready within this
                       window, the move may break the existing range or a
                       shared firing lane, and it offers no recognizable
                       relief, rescue or new firing-line benefit. When the
                       current target can no longer be legally re-issued the
                       row is recorded as cannot_reissue - no fabricated
                       attacks, no STOP as "kept attack".

A2 runs first; B2 never touches an actor A2 already moved. Every judgement
log carries actor/tag, loop, provenance, candidate, veto reason, health,
cooldown and support facts; destination outcomes are geometric estimates on
the current visible snapshot (estimated=True) and follow-up rows separate
arrival, pressure relief, entering the designated mate's range, and the
mate's own fire proxy. Dying before the next observation is recorded as
arrival_unknown, never as "did not walk".
"""
import json
import math
from collections import deque
from pathlib import Path

from .skills.combat_reposition import CombatRepositionSkill  # initialize frozen skills before hydra_support
from .action_duties import rescue_actions
from .hydra_support import support_path_clear
from .joint_combat import action_position
from .validator import validate_actions, validate_plans

CONFIG_PATH = Path(__file__).with_name("hydra_supported.json")

DEFAULTS = {
    "supported_escape": False,
    "guard_support_move": False,
    "max_escapes_per_window": 1,
    "duel_hp_ratio": 0.7,
    "recent_damage_windows": 2,
    "recent_damage_lookback": 3,
    "ready_within_loops": 8,
    "range_margin": 0.05,
    "followup_windows": 3,
    "arrival_tolerance": 0.35,
}

UNIT_HYDRA = "hydralisk"
# Formation intents B2 may examine: the move's purpose is readable. Escape
# roles stay out by design; hold/search/other carry no readable line intent.
GUARD_ROLES = ("support", "entry", "alignment")


def load_config():
    cfg = dict(DEFAULTS)
    try:
        cfg.update(json.loads(CONFIG_PATH.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        pass  # missing or unreadable file keeps every switch OFF
    return cfg


CONFIG = load_config()
ENABLED = bool(CONFIG.get("supported_escape")) or bool(CONFIG.get("guard_support_move"))


def _new_state():
    return {"recent_damage": {}, "tracked": {}}


def _state(manager):
    state = getattr(manager, "_hydra_supported_state", None)
    if state is None:
        state = manager._hydra_supported_state = _new_state()
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


def _threat_tags(context, threats):
    visible = context.snapshot.raw_state.get("visible_enemies", {}) or {}
    return {t: str(visible.get(t, {}).get("tag", "")) for t in threats}


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
                    math.dist(point, enemy.position) <= future:
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
        "support_threat": None, "support_mate": None, "support_class": None,
    }


# ------------------------------------------------------------------ pass A2
def _support_route(context, actor, dest, threat, mates):
    """Named-mate support evidence for landing exactly on `dest`.

    'now'        : the mate's known reach already covers the landing point -
                   standing area coverage, no enemy choice required.
    'if_pursued' : coverage only forms when the threat advances far enough to
                   engage us at the destination. Conditional estimate.
    Returns (mate, class, margin) or None; never a realized-support claim.
    """
    enemy = context.snapshot.enemies[threat]
    mine = context.snapshot.allies[actor]
    weapon = context._enemy_weapon_range(threat, mine.is_flying)
    engagement = None if weapon is None else weapon + enemy.radius + mine.radius
    distance = math.dist(enemy.position, dest)
    if distance <= math.dist(enemy.position, mine.position):
        return None  # not withdrawing from this particular enemy
    intercept = None
    if engagement is not None and distance > engagement:
        intercept = tuple(d + (e - d) * engagement / distance
                          for d, e in zip(dest, enemy.position))
    best = None
    for mate in mates:
        if mate == actor:
            continue
        mate_unit = context.snapshot.allies[mate]
        reach = context.effective_attack_reach(mate, threat)
        if reach is None or reach < 0:
            continue
        gap = math.dist(mate_unit.position, dest)
        if gap >= math.dist(mate_unit.position, mine.position):
            continue
        if gap <= reach:
            candidate = (0, gap - reach, mate)
        elif intercept is not None and math.dist(mate_unit.position, intercept) <= reach:
            candidate = (1, math.dist(mate_unit.position, intercept) - reach, mate)
        else:
            continue
        if best is None or candidate < best:
            best = candidate
    if best is None:
        return None
    cls, margin, mate = best
    return {"mate": mate, "class": ("now" if cls == 0 else "if_pursued"),
            "margin": round(-margin, 3), "estimated": True,
            "enemy_engagement_reach": engagement, "pursuit_intercept": intercept}


def _evaluate_A2(context, actor, actions, plans, summary, state, cfg, mates):
    unit = context.snapshot.allies[actor]
    action_id = actions[unit.index]
    decoded = context.decode(actor, action_id)
    row = _base_row(context, actor, action_id, decoded.kind, decoded.target)
    row["switch"] = "A2"
    plan = _plan_for(plans, actor)
    role = summary.get('units', {}).get(actor, {})
    row["original_skill"] = plan.primary_skill if plan else role.get('skill')
    row["original_phase"] = plan.commitment.phase if plan else role.get('phase')
    row["original_rationale"] = plan.rationale if plan else None
    duty = summary.get('execution_duties', {}).get('actors', {}).get(actor, {})
    row['original_provenance'] = dict(skill=row['original_skill'], phase=row['original_phase'],
                                      proposal=plan.proposal_id if plan else None,
                                      action=role.get('action'), duty=duty.get('reason'),
                                      forced_targets=list(context.forced_hydra_targets(actor)))
    risk = context.risk_at(actor, unit.position)
    threats = list(risk.threats)
    row["current_threats"] = threats
    row["threat_types"] = _threat_types(context, threats)
    row["threat_tags"] = _threat_tags(context, threats)
    if decoded.kind != "attack" or not context.can_issue_attack(actor, decoded.target) \
            or context.contact_status(actor, decoded.target) != 'in_range':
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
        row["veto"] = "effective_return_fire_exists"
        return row, None
    if plan and (len(plan.controlled_actors) > 1 or
                 any(d.kind == 'fire_window' for d in plan.dependencies)):
        row.update(result='unchanged:fire_commitment', veto='selected_plan_contract')
        return row, None
    killable = next((t for t in context.attackable_enemies(actor)
                     if (context.fire_estimate(actor, t).damage_est or 0)
                     >= context.snapshot.enemies[t].effective_health), None)
    if killable is not None:
        row["result"] = "unchanged:immediate_kill_available"
        row["veto"] = "immediate_kill_available"
        return row, None
    # Timing gate: the post-fire cooldown window normally hosts the move;
    # observed close pressure or a clearly unfavorable exchange may open it
    # before critical health. No future or hindsight condition is used.
    actor_relations = context.snapshot.relations.get(actor, {})
    close_pressure = [t for t in threats
                      if getattr(actor_relations.get(t), 'in_enemy_weapon_range', None) is True]
    history = list(state["recent_damage"].get(actor, ()))
    row["observed_loss_history"] = history
    sustained = len(history) >= cfg['recent_damage_windows'] and \
        all(history[-cfg['recent_damage_windows']:])
    behind = decoded.target in threats and unit.effective_health < \
        cfg['duel_hp_ratio'] * context.snapshot.enemies[decoded.target].effective_health
    bleeding = unit.health_lost + unit.shield_lost > 0
    pincer = {"zergling", UNIT_HYDRA} <= set(row["threat_types"].values())
    if pincer:
        unfavorable = "pincer_ling_and_hydra"
    elif bleeding and sustained and behind:
        unfavorable = "sustained_bleeding_behind_on_hp"
    else:
        unfavorable = None
    row["close_pressure"] = close_pressure
    row["unfavorable_exchange"] = unfavorable
    row["sustained_bleeding"] = bool(sustained)
    row["duel_hp_behind"] = bool(behind)
    if not unfavorable:
        row.update(result='unchanged:exchange_not_unfavorable', veto='no_unfavorable_exchange')
        return row, None
    # Named support route: (threatened actor, threatening enemy, supporting
    # mate) with one legal conflict-free landing spot under the mate's reach.
    memory = getattr(context, 'deployment_memory', None)
    joint = {other: actions[u.index] for other, u in context.snapshot.allies.items()}
    stationary_mates = [a for a in mates if context.decode(a, joint[a]).kind != 'move']
    options = []
    for move in sorted(context.move_destinations(actor)):
        dest = context.predict_position(actor, move).position
        if context.speed(actor) is None or \
                (memory is not None and memory.movement_blockage(context, actor, move) >= .5):
            continue
        if not support_path_clear(context, actor, dest, joint):
            continue
        if any(math.dist(dest, spot) < context.snapshot.allies[actor].radius +
               context.snapshot.allies[other].radius + 0.15
               for other, other_unit in context.snapshot.allies.items()
               if other != actor and other_unit.alive
               for spot in [action_position(context, other, joint[other])[0]]
               if spot is not None):
            continue  # an ally's reserved/planned position
        mid = tuple((a + b) / 2 for a, b in zip(dest, unit.position))
        left, right = (context.risk_at(actor, p) for p in (mid, dest))
        if left.effect_weapon_near or right.effect_weapon_near \
                or (set(left.threats) | set(right.threats)) - set(threats):
            continue                   # no new crossfire
        for threat in threats:
            route = _support_route(context, actor, dest, threat, stationary_mates)
            if route is not None:
                options.append((0 if route['class'] == 'now' else 1, -route['margin'],
                                move, threat, dest, route))
    if not options:
        row["result"] = "unchanged:no_support_route"
        row["veto"] = ("no_named_mate_reach_over_any_legal_landing_spot"
                       if memory is not None else "no_legal_move")
        return row, None
    options.sort()
    _, _, move, threat, dest, route = options[0]
    row['support_threat'] = threat
    row['support_threat_tag'] = row['threat_tags'].get(threat)
    row['support_mate_tag'] = str(context.snapshot.allies[route['mate']].tag)
    row["candidate_action"] = move
    row["candidate_kind"] = "move"
    row["candidate_destination"] = [round(dest[0], 3), round(dest[1], 3)]
    row["support_mate"] = route["mate"]
    row["support_class"] = route["class"]
    row["support_route"] = dict(mate=route["mate"], threat=threat,
                                destination=row["candidate_destination"],
                                mate_reach=context.effective_attack_reach(
                                    route["mate"], threat),
                                distance_to_landing=round(math.dist(
                                    context.snapshot.allies[route["mate"]].position, dest), 3),
                                klass=route['class'], estimated=True,
                                enemy_engagement_reach=route['enemy_engagement_reach'],
                                pursuit_intercept=route['pursuit_intercept'])
    row["result"] = "candidate_selected"
    return row, move


# ------------------------------------------------------------------ pass B2
def _evaluate_B2(context, actor, actions, plans, summary, manager, cfg, mates):
    unit = context.snapshot.allies[actor]
    action_id = actions[unit.index]
    decoded = context.decode(actor, action_id)
    row = _base_row(context, actor, action_id, decoded.kind, decoded.target)
    row["switch"] = "B2"
    role_row = (summary.get("units", {}) or {}).get(actor, {})
    row["original_skill"] = role_row.get("skill")
    row["original_phase"] = role_row.get("phase")
    plan = _plan_for(plans, actor)
    row["original_rationale"] = plan.rationale if plan else None
    if decoded.kind != 'move':
        row['result'] = 'out_of_scope:final_action_not_a_move'
        return row, None
    if action_id not in context.move_destinations(actor):
        row["result"] = "out_of_scope:move_without_destination"
        return row, None
    role = (getattr(context, 'hydra_move_roles', {}) or {}).get((actor, action_id))
    row["formation_move_role"] = role
    if role_row.get("skill") != 'ranged_fire_formation' \
            or role_row.get("phase") != 'exclusive_control':
        row["result"] = "out_of_scope:not_formation_provenance"
        row["veto"] = f"move_from_{role_row.get('skill')}:{role_row.get('phase')}"
        return row, None
    if role not in GUARD_ROLES:
        # Escape-classified formation moves stay out by design; unreadable
        # intents (hold/search/other/missing) are never overridden.
        row["result"] = "out_of_scope:intent_unreadable"
        row["veto"] = f"formation_role_{role or 'missing'}_has_no_guardable_intent"
        return row, None
    support_target = role_row.get("goal")
    if not isinstance(support_target, str) or support_target not in context.snapshot.enemies \
            or role_row.get("goal_kind") != 'enemy':
        row["result"] = "out_of_scope:intent_unreadable"
        row["veto"] = "no_readable_focus_enemy"
        return row, None
    # Intent record: which enemy the move supports against, and which mates
    # already hold fire on that enemy (the line the move joins or extends).
    row["support_target"] = support_target
    row["support_target_tag"] = str(context.snapshot.raw_state['visible_enemies']
                                    .get(support_target, {}).get('tag') or '') or None
    row["supporting_for"] = [mate for mate in mates if mate != actor
                             and context.can_issue_attack(mate, support_target)
                              and context.contact_status(mate, support_target) == 'in_range']
    row['allies_threatened_by_support_target'] = [a for a, u in context.snapshot.allies.items()
            if a != actor and u.alive and support_target in context.risk_at(a, u.position).threats]
    threats = list(context.risk_at(actor, unit.position).threats)
    row["current_threats"] = threats
    row["threat_types"] = _threat_types(context, threats)
    dest = context.predict_position(actor, action_id).position
    after = context.risk_at(actor, dest).threats
    relief = len(after) < len(threats)
    row["move_sheds_threat"] = relief
    objective = ((plan.goal_data or {}).get("objective") or {}) if plan else {}
    reason = str(objective.get("execution_reason", ""))
    duty = summary.get('execution_duties', {}).get('actors', {}).get(actor, {})
    rescue = actor in summary.get('joint_rescue_substitutions', {}) \
        or duty.get('kind') == 'rescue' or bool(objective.get("one_window_rescue")) \
        or "lethal" in reason or "blast" in reason
    if rescue or action_id in rescue_actions(context, actor, manager.memory):
        row["provenance_rescue"] = True
        row["result"] = "exempt:urgent_rescue"
        return row, None
    if relief:
        row["result"] = "exempt:real_relief"
        return row, None
    if (duty.get('isolated_shooter_support') or {}).get('satisfied'):
        row.update(result='exempt:improves_support', veto='isolated_shooter_support')
        return row, None
    reach_of = lambda a, t: context.effective_attack_reach(a, t)
    improves = _covers_ally_threat(context, actor, dest, reach_of)
    row["improves_support"] = improves or False
    if improves:
        row["result"] = "exempt:improves_support"
        row["veto"] = improves
        return row, None
    if context.speed(actor) is None:
        row.update(result='unchanged:move_reach_unknown', veto='missing_native_speed')
        return row, None
    # A formation focus is evidence of movement intent, NOT an existing shot.
    orders = context.snapshot.raw_state.get('allies', {}).get(actor, {}).get('orders') or []
    active = orders[0] if orders else {}
    native = active.get('target_tag') if active.get('kind') == 'attack' else None
    wait = getattr(context, '_debug_stationary', {}).get(actor)
    stationary = context.decode(actor, wait) if wait is not None else None
    if native:
        target = next((t for t, raw in context.snapshot.raw_state['visible_enemies'].items()
                       if str(raw.get('tag')) == str(native)), None)
        source = 'active_attack_order'
    elif stationary and stationary.kind == 'attack':
        target, source = stationary.target, 'formation_stationary_attack'
    else:
        target, source = None, None
    row.update(current_target=target, target_source=source,
               target_tag=str(native) if native else _threat_tags(context, [target]).get(target) or None,
               native_order_target_tag=str(native) if native else None)
    if not target or not context.can_issue_attack(actor, target):
        if source is None:
            row.update(result='out_of_scope:intent_unreadable', veto='no_current_attack_source')
            return row, None
        row["reissue_tier"] = "native_order_only_not_reissuable" if native \
            else "focus_target_not_menu_legal"
        row["result"] = "cannot_reissue:target_not_menu_legal"
        row["veto"] = "attack_target_not_visible_or_outside_command_mask_now"
        return row, None
    row["current_target"] = target
    if not (unit.weapon_ready or
            unit.weapon_cooldown_loops <= cfg["ready_within_loops"]):
        row["result"] = "unchanged:weapon_not_ready_in_time"
        row["veto"] = "cooldown_beyond_this_window"
        return row, None
    enemy = context.snapshot.enemies[target]
    reach = context.effective_attack_reach(actor, target)
    in_range_now = context.contact_status(actor, target) == "in_range"
    row["in_range_now"] = in_range_now
    dest_dist = math.dist(dest, enemy.position)
    row["destination_distance"] = round(dest_dist, 3)
    row["attack_reach"] = None if reach is None else round(reach, 3)
    keeps = reach is None or reach < 0 or dest_dist <= reach + cfg["range_margin"]
    row["keeps_range_estimate"] = bool(keeps)
    if support_target != target:
        support_reach = reach_of(actor, support_target)
        support_enemy = context.snapshot.enemies[support_target]
        if (support_reach is not None and support_reach >= 0 and
                math.dist(dest, support_enemy.position) <= support_reach + cfg['range_margin']):
            row.update(result='exempt:support_line_at_destination', veto='intended_line_still_forms')
            return row, None
        if role == 'support':  # formation computed a one/two-window counter route
            row.update(result='exempt:improves_support', veto='formation_counter_route')
            return row, None
    shared_lost = []
    for other_target in context.attackable_enemies(actor):
        shared_reach = context.effective_attack_reach(actor, other_target)
        if (shared_reach is not None and shared_reach >= 0
                and context.contact_status(actor, other_target) == 'in_range'
                and math.dist(dest, context.snapshot.enemies[other_target].position)
                > shared_reach + cfg['range_margin']
                and any(other_target in context.attackable_enemies(a)
                        and context.contact_status(a, other_target) == 'in_range'
                        for a in mates if a != actor)):
            shared_lost.append(other_target)
    row["shared_targets_lost_estimate"] = shared_lost
    if not in_range_now:
        row["result"] = "unchanged:no_existing_range_to_lose"
        row["veto"] = "focus_target_not_in_native_range_before_move"
        return row, None
    if keeps and not shared_lost:
        row["result"] = "unchanged:keeps_range"
        row["veto"] = "destination_still_reaches_target_and_lanes"
        return row, None
    # Recognizable benefit of completing the move: the intended line still
    # forms at the destination, or the move newly covers some visible enemy.
    if keeps:
        row["result"] = "exempt:support_line_at_destination"
        row["veto"] = "intended_line_still_forms"
        return row, None
    for visible_id, visible in context.snapshot.enemies.items():
        future = reach_of(actor, visible_id)
        if future is not None and future >= 0 \
                and math.dist(dest, visible.position) <= future \
                and not (context.contact_status(actor, visible_id) == 'in_range'):
            row["result"] = "exempt:new_line_at_destination"
            row["veto"] = f"new_coverage:{visible_id}"
            return row, None
    shot = context.attack_action(actor, target)
    if shot is None:
        row["result"] = "cannot_reissue:target_action_illegal"
        row["veto"] = "attack_not_on_legal_menu"
        return row, None
    row["candidate_action"] = shot
    row["candidate_kind"] = "attack"
    row["result"] = "candidate_selected"
    return row, shot


# ------------------------------------------------------------------ follow-ups
def _followups(state, context, cfg):
    rows = []
    loop = context.snapshot.game_loop
    snap = context.snapshot
    for key, track in list(state["tracked"].items()):
        actor = track['actor']
        unit = snap.allies.get(actor)
        alive = unit.alive if unit is not None and str(unit.tag) == track['actor_tag'] else None
        cooldown_restarted = None
        if alive:
            cooldown_restarted = unit.weapon_cooldown_loops > max(0, track['last_cd'] - 8) + 0.02
            track['last_cd'] = unit.weapon_cooldown_loops
        arrival = None
        position = None
        pressing = None
        threat_state = {}
        mate_state = None
        if alive:
            position = [round(unit.position[0], 3), round(unit.position[1], 3)]
            if track['predicted_dest'] is not None:
                arrival = math.dist(unit.position, track['predicted_dest']) <= cfg['arrival_tolerance']
            actor_relations = snap.relations.get(actor, {})
            pressing = [t for t in track['threats'] if
                        str(snap.raw_state['visible_enemies'].get(t, {}).get('tag')) == track['threat_tags'].get(t)
                        and getattr(actor_relations.get(t), 'in_enemy_weapon_range', None) is True]
            for threat in track.get("threats", ()):
                raw = snap.raw_state['visible_enemies'].get(threat, {})
                if threat not in snap.enemies or str(raw.get('tag')) != track['threat_tags'].get(threat):
                    threat_state[threat] = "not_visible"
                    continue
                pressing_flag = getattr(snap.relations.get(actor, {}).get(threat),
                                        'in_enemy_weapon_range', None) is True
                covered = any(threat in context.attackable_enemies(mate)
                              and context.contact_status(mate, threat) == 'in_range'
                              for mate in _hydras(context) if mate != actor)
                designated = ''
                mate = track.get('support_mate')
                if mate and mate != actor:
                    mate_relation = snap.relations.get(mate, {}).get(threat)
                    designated = '_in_designated_mate_range' if (
                         snap.allies.get(mate) and snap.allies[mate].alive
                        and str(snap.allies[mate].tag) == track.get('support_mate_tag')
                        and getattr(mate_relation, 'smac_attack_available', None) is True
                        and getattr(mate_relation, 'in_weapon_range', None) is True) else ''
                threat_state[threat] = ("pressing" if pressing_flag else "visible") + \
                    ("_and_covered" if covered else "_uncovered") + designated
            mate = track.get('support_mate')
            if mate and mate in snap.allies and snap.allies[mate].alive \
                    and str(snap.allies[mate].tag) == track.get('support_mate_tag'):
                previous = track.get('mate_cd')
                if previous is not None:
                    mate_state = dict(
                        mate=mate,
                        weapon_cooldown_loops=snap.allies[mate].weapon_cooldown_loops,
                        fire_proxy=snap.allies[mate].weapon_cooldown_loops
                        > max(0, previous - 8) + 0.02)
                track['mate_cd'] = snap.allies[mate].weapon_cooldown_loops
        if alive and arrival and track.get('first_arrival_loop') is None:
            track['first_arrival_loop'] = loop
        row = {
            "switch": track["switch"], "actor": actor,
            "actor_tag": track["actor_tag"], "followup": True,
            "decision_loop": track["loop"], "game_loop": loop,
            "current_target": track['current_target'], "target_tag": track['target_tag'],
            "threat_tags": track['threat_tags'], "submitted_action": track['action'],
            "support_mate": track.get('support_mate'), "support_threat": track.get('support_threat'),
            "predicted_destination": track.get('predicted_dest'),
            "windows_since": snap.snapshot_id - track["window"],
            "alive": alive, "health": unit.health if alive is not None else None,
            "position": position,
            "weapon_cooldown_loops": unit.weapon_cooldown_loops if alive else None,
            "weapon_ready": unit.weapon_ready if alive else None,
            "cooldown_restarted": cooldown_restarted,  # proxy; not shot proof
            "arrived_at_predicted": arrival,
            "pressing_threats_now": pressing,
            "threat_state": threat_state,
            "designated_mate_fire_proxy": (mate_state or {}).get('fire_proxy'),
            "designated_mate_cooldown": (mate_state or {}).get('weapon_cooldown_loops'),
            "survived_16_loops": alive if loop - track['loop'] == 16 else None,
            "survived_24_loops": alive if loop - track['loop'] == 24 else None,
            "estimated": True,
        }
        keep = alive and (loop - track["loop"]) < 8 * cfg["followup_windows"]
        if alive is False:
            row["final"] = True
            row["outcome"] = "died_observed"
            row["died_observed_loop"] = loop
            row["arrival_status"] = ("arrived" if track.get('first_arrival_loop') is not None
                                     else "arrival_unknown")
            del state['tracked'][key]
        elif not keep:
            row["final"] = True
            row["outcome"] = "followup_window_ended"
            row["arrival_status"] = ("arrived" if track.get('first_arrival_loop') is not None
                                     else 'arrival_unknown' if alive is None else "not_reached_within_24")
            del state['tracked'][key]
        if track['predicted_dest'] is None:
            row['arrival_status'] = None
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
    permit = (context.snapshot.snapshot_id, actor, row['candidate_action']) \
        if row['switch'] == 'A2' else None
    previous = getattr(context, '_hydra_escape_permit', None)
    allowed = permit or previous
    for a, cmd in joint.items():
        targets = context.forced_hydra_targets(a)
        if targets and context.decode(a, cmd).target not in targets:
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
        state['tracked'][key] = dict(
            actor=actor, actor_tag=row['actor_tag'], switch=row['switch'],
            loop=row['game_loop'], window=row['window'], action=row['final_action'],
            current_target=row.get('current_target', row['original_target']),
            target_tag=row.get('target_tag'), threats=row.get('current_threats', []),
            threat_tags=row.get('threat_tags', {}),
            last_cd=unit.weapon_cooldown_loops,
            support_mate=row.get('support_mate'),
            support_mate_tag=str(context.snapshot.allies[row['support_mate']].tag)
            if row.get('support_mate') and row['support_mate'] in context.snapshot.allies else None,
            support_threat=row.get('support_threat'),
            predicted_dest=tuple(row['candidate_destination'])
            if row.get('candidate_destination') else None,
            first_arrival_loop=None)
        if row.get('support_mate') and row['support_mate'] in context.snapshot.allies:
            # Baseline for the designated mate's own fire proxy (cooldown
            # restart beyond ordinary decay), measured from this decision.
            state['tracked'][key]['mate_cd'] = \
                context.snapshot.allies[row['support_mate']].weapon_cooldown_loops


def apply(manager, context, plans, summary, actions):
    """Post-arbitration hook; returns the (possibly) modified action list."""
    if not ENABLED:
        return actions
    cfg = CONFIG
    snap = context.snapshot
    if snap.step_mul != 8 or any(u.type_name not in ('hydralisk', 'zergling')
                                 for u in snap.allies.values()):
        return actions
    if snap.snapshot_id == 1:
        manager._hydra_supported_state = _new_state()
    state = _state(manager)
    rows = _followups(state, context, cfg)
    mates = _hydras(context)
    for actor in mates:
        unit = snap.allies[actor]
        history = state['recent_damage'].setdefault(
            actor, deque(maxlen=cfg['recent_damage_lookback']))
        history.append(unit.health_lost + unit.shield_lost)
    overridden = set()
    if cfg.get("supported_escape"):
        candidates = []
        for actor in mates:
            row, move = _evaluate_A2(context, actor, actions, plans, summary,
                                     state, cfg, mates)
            rows.append(row)
            if move is not None:
                candidates.append((0 if row.get("support_class") == "now" else 1,
                                   row.get("health_fraction", 1.0), actor, move, row))
        if candidates:
            candidates.sort()
            applied = 0
            for _, _, actor, move, row in candidates:
                if applied >= cfg.get("max_escapes_per_window", 1):
                    row["result"] = "unchanged:window_cap_reached"
                    row["veto"] = "one_supported_escape_per_window"
                    continue
                applied += _select(manager, context, plans, actions, overridden, row)
    if cfg.get("guard_support_move"):
        for actor in mates:
            if actor in overridden:
                continue          # A2 owns this actor this window
            row, shot = _evaluate_B2(context, actor, actions, plans, summary,
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
        r.setdefault('threat_tags', _threat_tags(context, r.get('current_threats', [])))
        if 'ally_support' not in r and 'current_threats' in r:
            r['ally_support'] = _support_map(context, a, r['current_threats'], mates)
        r.setdefault('current_target', r['original_target'])
        r['conditions_met'] = r['candidate_action'] is not None
        r['selected_for_commit'] = r['changed']
        r['final_action'] = actions[snap.allies[a].index]
    if overridden:
        for r in state['pending']:
            a = r['actor']
            kind = 'supported_escape' if r['switch'] == 'A2' else 'guard_support_move'
            summary.setdefault('actions', {})[a] = r['final_action']
            summary.setdefault('units', {})[a] = dict(
                action=r['final_action'], skill=kind, phase='one_window',
                goal=r.get('support_target') or r.get('support_threat') or '',
                goal_kind='enemy' if r['switch'] == 'B2' else '',
                provider=None, plan=None)
        removed = {p.proposal_id for p in plans
                   if not overridden.isdisjoint(p.controlled_actors)}
        plans[:] = [p for p in plans if p.proposal_id not in removed]
        summary['chosen_plans'] = [p for p in summary.get('chosen_plans', ())
                                   if p['id'] not in removed]
        summary['coordinator_assessment_scope'] = 'pre_switch; units and actions are post_switch'
    if rows:
        summary["hydra_supported_decisions"] = rows
    return actions
