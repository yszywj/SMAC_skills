"""Personal, short-lived progress checks for pure approach movement only.

Observed positions are authoritative. Enemy movement and target selection never
count as the actor's own movement. Candidate destinations/contact times remain
local estimates; this helper is not a pathfinder or an attack-legality oracle.
"""
from copy import deepcopy
import math


MOVES = (2, 3, 4, 5)


def _number(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def _legal_attack(state, actor):
    return any(row.get('smac_attack_available') for row in state.get('relations', {}).get(actor, {}).values())


def guard_approach(state, actor, history, target, candidates, features, original_action,
                   allowed_actions=None):
    """Return a candidate, diagnostic and bounded memory, without mutating inputs."""
    unit = state['allies'][actor]
    origin = list(unit['position'])
    loop, window = state.get('game_loop', 0), state.get('step_mul', 8)
    memory = deepcopy(history.get('approach_progress') or {})
    samples = memory.get('samples', [])[-2:]
    previous = samples[-1] if samples else None
    diagnostic = {
        'enabled': True, 'cycle_detected': False, 'blocked_stall': False,
        'active': False, 'applied': False, 'original_action': int(original_action),
        'guard_selected_action': int(original_action), 'final_action': int(original_action),
        'own_net_displacement': None, 'path_distance': None,
        'own_target_progress': None, 'target_motion_progress': None,
        'target_distance_progress': None, 'target_changed': False,
        'reset_reason': None, 'repeated_observation': False, 'shot_proxy': False,
        'blocked_actions': [], 'single_route_release': False, 'candidates': {},
        'meaning': 'observed personal movement; future destinations/contact are estimates',
    }
    # Keep an intentional bounded wait (for example for a melee screen).
    # Progress recovery may redirect a move, never manufacture one from wait.
    if int(original_action) not in MOVES:
        diagnostic['reset_reason'] = 'intentional_nonmoving_approach'
        return int(original_action), diagnostic, {}
    # Repeating a game loop does not grow history or extend an escape window.
    if previous and previous['loop'] == loop and previous.get('tag') == unit.get('tag') and previous['position'] == origin:
        action = int(previous['action'])
        if action == 1 or str(action) in candidates:
            diagnostic.update(repeated_observation=True, reset_reason='duplicate_observation',
                              guard_selected_action=action, final_action=action)
            return action, diagnostic, memory
    if previous and (previous.get('tag') != unit.get('tag') or loop - previous['loop'] != window):
        diagnostic['reset_reason'] = ('unit_replaced' if previous.get('tag') != unit.get('tag')
                                      else 'nonconsecutive_observation')
        samples, memory, previous = [], {}, None
    if previous and previous.get('action') not in MOVES:
        diagnostic['reset_reason'] = 'previous_approach_did_not_move'
        samples, memory, previous = [], {}, None
    cooldown = unit.get('weapon_cooldown_loops')
    if previous and _number(cooldown) and _number(previous.get('cooldown')):
        diagnostic['shot_proxy'] = cooldown > max(0, previous['cooldown'] - window) + .1
        if diagnostic['shot_proxy']:
            diagnostic['reset_reason'] = 'observed_cooldown_restart'
            samples, memory, previous = [], {}, None
    if previous:
        diagnostic['target_changed'] = previous.get('target') != target
    if len(samples) == 2 and samples[1]['loop'] - samples[0]['loop'] == window:
        older, previous = samples
        first_leg = math.dist(older['position'], previous['position'])
        second_leg = math.dist(previous['position'], origin)
        net = math.dist(older['position'], origin)
        diagnostic.update(own_net_displacement=net, path_distance=first_leg + second_leg,
                          observed_previous_displacement=second_leg)
        old_target_position = older.get('target_position')
        old_target = state['visible_enemies'].get(older.get('target'))
        if old_target_position is not None:
            own_progress = math.dist(older['position'], old_target_position) - math.dist(origin, old_target_position)
            diagnostic['own_target_progress'] = own_progress
            if old_target is not None:
                target_progress = math.dist(origin, old_target_position) - math.dist(origin, old_target['position'])
                diagnostic['target_motion_progress'] = target_progress
                diagnostic['target_distance_progress'] = own_progress + target_progress
        both_moves = older.get('action') in MOVES and previous.get('action') in MOVES
        diagnostic['cycle_detected'] = bool(both_moves and first_leg > .4 and second_leg > .4 and net < .2)
        diagnostic['blocked_stall'] = bool(both_moves and first_leg < .2 and second_leg < .2
                                           and older.get('expected_travel', 0) > .4
                                           and previous.get('expected_travel', 0) > .4)
    triggered = diagnostic['cycle_detected'] or diagnostic['blocked_stall']
    avoid_until = memory.get('avoid_until_loop', 0)
    avoid_positions = memory.get('avoid_positions', [])
    blocked_attempt = memory.get('blocked_attempt_action')
    # An existing escape expires independently of target switches and new
    # detections; observing another cycle cannot extend it indefinitely.
    if loop >= avoid_until:
        avoid_until, avoid_positions, blocked_attempt = 0, [], None
        if triggered:
            avoid_until = loop + 2 * window
            avoid_positions = [list(previous['position'])] if diagnostic['cycle_detected'] else []
            blocked_attempt = previous['action'] if diagnostic['blocked_stall'] else None
            memory.update(escape_origin=origin, escape_target_position=list(state['visible_enemies'][target]['position']))
    elif memory.get('escape_target_position') is not None:
        reference = memory['escape_target_position']
        self_progress = math.dist(memory['escape_origin'], reference) - math.dist(origin, reference)
        diagnostic['escape_own_progress'] = self_progress
        if self_progress > .4 and not triggered:
            avoid_until, avoid_positions, blocked_attempt = 0, [], None
            diagnostic['reset_reason'] = 'observed_self_progress'
    active = loop < avoid_until
    action = int(original_action)
    if active:
        available = [str(a) for a in (allowed_actions or candidates) if str(a) in candidates]
        # Avoid both the cycle's previously occupied cell and immediately
        # returning to the position left during an active escape.
        banned_positions = list(avoid_positions)
        if previous:
            banned_positions.append(previous['position'])
        blocked = [a for a in available if (int(a) == blocked_attempt or
                    any(math.dist(candidates[a], point) < .3 for point in banned_positions))]
        eligible = [a for a in available if a not in blocked]
        diagnostic['blocked_actions'] = [int(a) for a in blocked]
        diagnostic['single_route_release'] = bool(blocked and not eligible)
        if eligible:
            # On a detected failure only, prefer measured forward recovery and
            # short estimated contact while retaining the original occupancy
            # and visible-danger costs. No global target policy is changed.
            travel = max(math.dist(origin, candidates[a]) for a in available)
            target_position = state['visible_enemies'][target]['position']
            for candidate in available:
                feature = features[candidate]
                own_forward = math.dist(origin, target_position) - math.dist(candidates[candidate], target_position)
                lateral_cost = .35 * max(0, 1 - own_forward / max(travel, 1e-6))
                score = feature['base_score'] + lateral_cost
                diagnostic['candidates'][candidate] = {
                    **feature, 'destination': list(candidates[candidate]),
                    'predicted_own_forward_progress': own_forward,
                    'lateral_cost': lateral_cost, 'guard_score': score,
                    'return_blocked': candidate in blocked}
            action = int(min(eligible, key=lambda a: (diagnostic['candidates'][a]['guard_score'], int(a))))
        # With no alternative, retain the legal original action, including a
        # bounded wait. An escape never turns a sole legal route into a no-op.
    destination = origin if action == 1 else candidates[str(action)]
    sample = {'loop': loop, 'position': origin, 'tag': unit.get('tag'), 'action': action,
              'target': target, 'target_position': list(state['visible_enemies'][target]['position']),
              'cooldown': cooldown, 'expected_travel': math.dist(origin, destination)}
    memory.update(samples=[*samples, sample][-2:], avoid_until_loop=avoid_until,
                  avoid_positions=avoid_positions, blocked_attempt_action=blocked_attempt)
    diagnostic.update(active=active, guard_selected_action=action, final_action=action,
                      applied=action != int(original_action), avoid_until_loop=avoid_until,
                      reason=('personal_position_cycle' if diagnostic['cycle_detected'] else
                              'personal_blocked_approach' if diagnostic['blocked_stall'] else
                              'bounded_personal_escape' if active else 'observe'))
    return action, diagnostic, memory


def finalize_progress(state, actions, roles, histories, updates, diagnostics):
    """Synchronize memory after later action filters and release exited actors.

    Only the passed fresh output containers are updated. Input histories/state
    remain untouched. A guard proposal reversed by a later filter is not counted
    as an applied guard decision.
    """
    for actor, unit in state['allies'].items():
        index = int(actor[1:])
        action = actions[index] if index < len(actions) else 0
        row = diagnostics.get(actor, {}).get('progress_guard')
        valid = bool(unit.get('alive') and roles.get(actor) == 'reposition'
                     and not _legal_attack(state, actor) and row is not None)
        if not valid:
            if histories.get(actor, {}).get('approach_progress') or row is not None:
                updates.setdefault(actor, {})['approach_progress'] = {}
            if row is not None:
                row.update(applied=False, final_action=action, final_role=roles.get(actor),
                           reset_reason='exited_pure_approach', memory={})
            continue
        memory = updates[actor]['approach_progress']
        if memory.get('samples'):
            sample = memory['samples'][-1]
            sample['action'] = action
            sample['expected_travel'] = (math.dist(unit['position'], diagnostics[actor]['predicted_destinations'][str(action)])
                                         if str(action) in diagnostics[actor]['predicted_destinations'] else 0.0)
        if action not in MOVES:
            memory = {}
            updates[actor]['approach_progress'] = memory
            row['reset_reason'] = 'final_approach_did_not_move'
        row.update(final_action=action, final_role=roles.get(actor),
                   postfilter_changed=action != row['guard_selected_action'],
                   applied=bool(row['guard_selected_action'] != row['original_action']
                                and action == row['guard_selected_action']),
                   attribution='changed guard proposal survives final action filters', memory=deepcopy(memory))
