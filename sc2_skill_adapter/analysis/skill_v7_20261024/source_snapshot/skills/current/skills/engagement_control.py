"""Local, bounded engagement history shared by every Adaptive role.

An attack order is intent. Only a fresh observed cooldown restart is counted
as personal firing progress; this is a proxy, never confirmed hit attribution.
All functions return new dictionaries and leave supplied observations/history
untouched. Fixed limits are deliberately independent of the benchmark result.
"""
import math

from .combat_timing import window_loops
from .movement_scoring import engagement_reach, predicted_move, threat_count, visible_volley_pressure

COMMIT_WINDOWS = 3
BACKOFF_WINDOWS = 2
LOST_VISION_WINDOWS = 3
AVOID_WINDOWS = 2


def _number(value, default=0.0):
    return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else default


def _targets(state, actor):
    return [e for e, r in state.get('relations', {}).get(actor, {}).items()
            if e in state.get('visible_enemies', {}) and r.get('smac_attack_available')
            and r.get('in_weapon_range') is True]


def prepare_engagement(state, histories, parameters):
    """Advance observations once, before role selection; no command earns credit."""
    prepared, diagnostics = {}, {}
    loop, window = state.get('game_loop', 0), max(1, state.get('step_mul', 8))
    for actor, unit in state['allies'].items():
        if not unit.get('alive'):
            continue
        original = histories.get(actor, {})
        h = dict(original.get('engagement', {}))
        if h.get('last') and loop < h['last']['loop']:
            h = {}  # A reset observation timeline cannot inherit a future promise.
        last, older = h.get('last'), h.get('before_last')
        elapsed = loop - last['loop'] if last else 0
        fresh = bool(last and 0 < elapsed <= 2 * window)
        raw_cd = unit.get('weapon_cooldown_loops')
        observed_cd = float(raw_cd) if isinstance(raw_cd, (int, float)) and math.isfinite(raw_cd) else None
        cooldown = max(0.0, _number(observed_cd))
        shot = bool(fresh and observed_cd is not None and last.get('cooldown') is not None
                    and cooldown > max(0.0, last['cooldown'] - elapsed) + .5)
        visible = state.get('visible_enemies', {})
        relations = state.get('relations', {}).get(actor, {})
        targets = _targets(state, actor)
        previous_target = h.get('target')
        candidates = [e for e, r in relations.items() if e in visible
                      and r.get('weapon_range_status') != 'no_compatible_weapon']
        target = (previous_target if previous_target in candidates else
                  min(candidates, key=lambda e: (e not in targets, relations[e].get('distance', 1e9), e), default=None))
        position = list(unit['position'])
        distance = math.dist(position, visible[target]['position']) if target else None
        cycle = bool(fresh and older and last['loop'] - older['loop'] == window
                     and elapsed == window and last.get('action') in (2, 3, 4, 5)
                     and older.get('action') in (2, 3, 4, 5)
                     and math.dist(position, older['position']) <= .2
                     and math.dist(position, last['position']) > .4
                     and target is not None and older.get('target') == target
                     and older.get('target_distance') is not None
                     and older['target_distance'] - distance <= .2 and not shot)
        duplicate = bool(last and elapsed == 0)
        gap = 0 if visible else h.get('visibility_gap', 0) + int(not duplicate)
        reset = 'observed_cooldown_restart' if shot else ('visible_engagement_ended' if gap >= LOST_VISION_WINDOWS else None)
        had_commit = bool(h.get('commit_until', 0))
        fulfilled = bool(had_commit and shot)
        timed_out = bool(had_commit and loop >= h['commit_until'] and not shot)
        exit_reason = None
        if had_commit:
            if shot:
                exit_reason = 'observed_cooldown_restart'
            elif previous_target not in visible:
                exit_reason = 'target_no_longer_visible'
            elif cooldown > window_loops(state, parameters):
                exit_reason = 'weapon_not_ready'
            elif timed_out:
                exit_reason = 'commitment_timeout'
            if exit_reason:
                h['commit_until'] = 0
                h['backoff_until'] = loop + BACKOFF_WINDOWS * window
                if timed_out:
                    h['failed_commit_target'] = previous_target
        if reset:
            h.update(retreat_budget_used=0, retreat_cycles=0, no_fire_steps=0,
                     danger_overrides_used=0, commit_until=0, avoid_until=0, failed_commit_target=None)
        else:
            h['no_fire_steps'] = h.get('no_fire_steps', 0) + int(bool(visible) and fresh)
        h['visibility_gap'] = gap
        if cycle:
            h.update(avoid_position=list(last['position']), avoid_until=loop + AVOID_WINDOWS * window)
        if target is not None and target != h.get('failed_commit_target'):
            h['failed_commit_target'] = None
        failed = target is not None and target == h.get('failed_commit_target')
        ready = bool(targets) and observed_cd is not None and cooldown <= window_loops(state, parameters)
        active = loop < h.get('commit_until', 0)
        start = bool(not active and not exit_reason and not shot and not failed and not duplicate and ready
                     and loop >= h.get('backoff_until', 0)
                     and (h.get('retreat_budget_used', 0) > 0 or cycle))
        if start:
            h['commit_until'] = loop + COMMIT_WINDOWS * window
            active = True
        pressure = visible_volley_pressure(state, actor, position, parameters)
        threats = threat_count(state, actor, position, parameters)
        h.update(target=target, current_loop=loop, current_position=position,
                 current_cooldown=cooldown, current_target_distance=distance,
                 current_observed_cooldown=observed_cd, repeated_observation=duplicate,
                 current_threat_count=threats, current_pressure=pressure,
                 cycle_detected=cycle, shot_proxy=shot)
        prepared[actor] = {**original, 'engagement': h,
                           # A failed promise is not permanent permission to
                           # force attacks. Keep the budget as evidence, but
                           # release its hard guard and reassess routes locally.
                           'retreat_steps': 0 if failed else h.get('retreat_budget_used', 0),
                           'retreat_cycles': h.get('retreat_cycles', 0)}
        diagnostics[actor] = dict(applied=False, reason=exit_reason or 'observe', shot_proxy=shot,
            progress_basis='observed cooldown restart; not attack intent or teammate damage',
            cycle_detected=cycle, personal_no_fire_steps=h.get('no_fire_steps', 0),
            retreat_budget_used=h.get('retreat_budget_used', 0), reset_reason=reset,
            commitment_active=active, commitment_started=start, commitment_fulfilled=fulfilled,
            commitment_timed_out=timed_out, commitment_until_loop=h.get('commit_until', 0),
            commitment_fallback=failed, repeated_observation=duplicate,
            retreat_blocked=False, danger_override=False, repeated_move_blocked=False,
            ready_legal_targets=targets, target=target)
    return prepared, diagnostics


def retreat_override(state, actor, action, history, parameters):
    """None means ordinary policy; otherwise (retreat?, reason), local only."""
    h = history.get('engagement', {})
    failed = h.get('target') is not None and h.get('target') == h.get('failed_commit_target')
    active = state.get('game_loop', 0) < h.get('commit_until', 0) and bool(_targets(state, actor))
    capped = not failed and h.get('retreat_budget_used', 0) >= parameters.get('max_retreat_steps', 2)
    if not active and not capped:
        return None
    unit = state['allies'][actor]
    last = h.get('last') or {}
    hp = unit.get('health', 0) + unit.get('shield', 0)
    maximum = (unit.get('max_health') or 0) + (unit.get('max_shield') or 0) or 1
    loss = unit.get('health_lost_since_previous_decision', 0) + unit.get('shield_lost_since_previous_decision', 0)
    threats = h.get('current_threat_count', 0)
    pressure = h.get('current_pressure', 0)
    # Old low HP alone cannot repeatedly cancel a promise. A *new* observed
    # hit or additional enemy envelope, plus a genuinely safer route, may.
    new_danger = (loss >= max(1.0, .10 * maximum)
                  or threats > last.get('threat_count', threats))
    if (new_danger and pressure >= max(1.0, hp) and h.get('danger_overrides_used', 0) < 1
            and str(action) in state.get('movement_candidates', {}).get(actor, {})):
        destination = predicted_move(state, actor, action)['destination']
        if (threat_count(state, actor, destination, parameters) < threats
                and visible_volley_pressure(state, actor, destination, parameters) < pressure):
            return True, 'engagement_new_danger_escape'
    return False, 'engagement_shot_commitment' if active else 'engagement_retreat_budget'


def finalize_engagement(state, histories, actions, roles, assignments, retreat_actions, retreat_reasons, diagnostics, parameters):
    """Bound repeated non-attack movement, then persist all-role observed history."""
    updates = {}
    loop = state.get('game_loop', 0)
    for actor, row in diagnostics.items():
        h = dict(histories[actor]['engagement'])
        index, role = int(actor[1:]), roles[actor]
        action = actions[index]
        moves = state.get('movement_candidates', {}).get(actor, {})
        target = h.get('target')
        if (role in ('retreat', 'reposition', 'recover') and str(action) in moves and target
                and loop < h.get('avoid_until', 0) and h.get('avoid_position')):
            destination = predicted_move(state, actor, action)['destination']
            returning = math.dist(destination, h['avoid_position']) < .3
            reach = engagement_reach(state, actor, target)
            # A bounded direct return is useful if it creates a legal firing
            # opportunity next window; the local promise protects that shot.
            reentry = (h.get('retreat_budget_used', 0) > 0 and reach is not None
                       and target != h.get('failed_commit_target')
                       and h.get('current_observed_cooldown') is not None
                       and h['current_cooldown'] <= window_loops(state, parameters)
                       and math.dist(destination, state['visible_enemies'][target]['position']) <= reach)
            if returning and not reentry:
                choices = []
                old_threat = threat_count(state, actor, destination, parameters)
                old_pressure = visible_volley_pressure(state, actor, destination, parameters)
                for a in moves:
                    dest = predicted_move(state, actor, a)['destination']
                    threat = threat_count(state, actor, dest, parameters)
                    if (math.dist(dest, h['avoid_position']) >= .3 and threat <= old_threat
                            and visible_volley_pressure(state, actor, dest, parameters) <= old_pressure):
                        choices.append((math.dist(dest, state['visible_enemies'][target]['position']) + threat, int(a)))
                if choices:
                    original_action = action
                    action = actions[index] = min(choices)[1]
                    row.update(applied=True, repeated_move_blocked=True, original_action=original_action,
                               reason='bounded_cross_role_cycle_escape')
                    if role == 'retreat':
                        retreat_actions[actor] = action
            elif returning and reentry:
                row['reason'] = 'cycle_reentry_with_shot_commitment'
        if role == 'retreat' and not h.get('repeated_observation'):
            h['retreat_budget_used'] = h.get('retreat_budget_used', 0) + 1
            if histories[actor].get('role') != 'retreat':
                h['retreat_cycles'] = h.get('retreat_cycles', 0) + 1
            if retreat_reasons.get(actor, '').startswith('engagement_new_danger_escape'):
                h['danger_overrides_used'] = h.get('danger_overrides_used', 0) + 1
                h['commit_until'] = 0
                row.update(danger_override=True, applied=True, reason='engagement_new_danger_escape')
        # Merely assigning an attack never clears budget, cycles or no-fire.
        if not h.get('repeated_observation'):
            h['before_last'] = dict(h['last']) if h.get('last') else None
            recorded_target = assignments.get(actor) or target
            recorded_distance = (math.dist(state['allies'][actor]['position'], state['visible_enemies'][recorded_target]['position'])
                                 if recorded_target in state['visible_enemies'] else None)
            h['last'] = dict(loop=loop, position=list(state['allies'][actor]['position']),
                             cooldown=h['current_observed_cooldown'], action=action, role=role,
                             target=recorded_target, target_distance=recorded_distance,
                             threat_count=h.get('current_threat_count', 0))
        row.update(retreat_budget_used=h.get('retreat_budget_used', 0),
                   selected_action=action, selected_role=role, attack_order_issued=actor in assignments)
        updates[actor] = {'engagement': h, 'retreat_steps': h.get('retreat_budget_used', 0),
                          'retreat_cycles': h.get('retreat_cycles', 0)}
    return updates
