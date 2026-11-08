"""Bounded, visible-only approach coordination; not a replacement pathfinder.

Only actors without legal attack candidates use this module. Targets persist
while still competitive, short-window destinations are reserved in actor order,
and a measured stalled move encourages a lateral route. Predictions do not
claim that collision-free movement or firing will actually occur.
"""
import math

from .movement_scoring import engagement_reach, predicted_move, threat_count
from .approach_progress import guard_approach


def _distance_to_segment(point, start, end):
    vector = [b - a for a, b in zip(start, end)]
    length2 = sum(v * v for v in vector)
    fraction = max(0.0, min(1.0, sum((p - a) * v for p, a, v in
                                   zip(point, start, vector)) / length2)) if length2 else 0.0
    return math.dist(point, [a + fraction * v for a, v in zip(start, vector)])


def _melee(state, actor):
    ranges = [r['native_weapon_range'] for r in state.get('relations', {}).get(actor, {}).values()
              if r.get('native_weapon_range') is not None]
    return bool(ranges) and min(ranges) <= 1.5


def coordinate_approaches(state, actors, histories, fixed_actions=None, parameters=None):
    """Return actions, diagnostics, history updates without mutating inputs.

    History is scoped to an episode by the manager. It is useful only when the
    immediately preceding decision was also an approach; stale movement is not
    misclassified as a new obstruction.
    """
    params = parameters or {}
    allies, enemies = state['allies'], state['visible_enemies']
    fixed_actions = fixed_actions or {}
    reservations = {}
    for actor, action in fixed_actions.items():
        if actor not in allies or not allies[actor].get('alive'):
            continue
        reservations[actor] = (predicted_move(state, actor, action)['destination']
                               if str(action) in state.get('movement_candidates', {}).get(actor, {})
                               else allies[actor]['position'])
    actions, diagnostics, updates = {}, {}, {}
    # Melee gets first choice of routes, but never waits for ranged teammates.
    ordered = sorted(actors, key=lambda a: (not _melee(state, a), int(a[1:])))
    for actor in ordered:
        unit = allies[actor]
        moves = state.get('movement_candidates', {}).get(actor, {})
        relations = state.get('relations', {}).get(actor, {})
        if not enemies or not moves or any(r.get('smac_attack_available') for r in relations.values()):
            continue
        history = histories.get(actor, {})
        origin = unit['position']
        travel = max(math.dist(origin, predicted_move(state, actor, a)['destination']) for a in moves)
        if travel <= 1e-6:
            continue
        costs = {}
        for enemy, relation in relations.items():
            if enemy not in enemies or relation.get('weapon_range_status') == 'no_compatible_weapon':
                continue
            reach = engagement_reach(state, actor, enemy)
            if reach is None:
                continue  # Unknown compatibility must not invent a reachable attack.
            costs[enemy] = max(0.0, math.dist(origin, enemies[enemy]['position']) - reach) / travel
        if not costs:
            continue
        target = min(costs, key=lambda e: (costs[e], int(e[1:])))
        previous_target = history.get('approach_target')
        kept = previous_target in costs and costs[previous_target] <= costs[target] + 1.0
        if kept:
            target = previous_target
        recent = history.get('approach_loop') == state.get('game_loop', 0) - state.get('step_mul', 8)
        previous_action = history.get('approach_action') if recent else None
        displacement = (math.dist(origin, history['approach_position'])
                        if recent and 'approach_position' in history else None)
        stalled = (previous_action in (2, 3, 4, 5) and displacement is not None
                   and displacement < 0.2 and history.get('approach_expected_travel', 0) > 0.4)
        target_position = enemies[target]['position']
        target_distance = math.dist(origin, target_position)
        loop, window = state.get('game_loop', 0), state.get('step_mul', 8)
        two_back = history.get('approach_previous_position')
        old_distance = history.get('approach_previous_target_distance')
        same_target = recent and previous_target == target
        net_progress = old_distance - target_distance if same_target and old_distance is not None else None
        # A unit can move its full distance yet alternate between two cells.
        # Detect observed A-B-A with no target-distance progress, then avoid
        # returning to the just-left cell for only this and the next window.
        cycle_detected = bool(same_target and two_back is not None
                              and displacement is not None and displacement > 0.4
                              and math.dist(origin, two_back) < 0.2
                              and net_progress is not None and net_progress <= 0.2)
        avoid_until = history.get('approach_cycle_avoid_until_loop', 0) if same_target else 0
        if cycle_detected:
            avoid_until = loop + 2 * window
        direct = min(moves, key=lambda a: (math.dist(predicted_move(state, actor, a)['destination'],
                                                    target_position), int(a)))
        scores, candidates, progress_features = {}, {}, {}
        for action in moves:
            destination = predicted_move(state, actor, action)['destination']
            reach = engagement_reach(state, actor, target)
            remaining = max(0.0, math.dist(destination, target_position) - reach) / travel
            collision_cost = 0.0
            for other, friend in allies.items():
                if other == actor or not friend.get('alive') or friend.get('is_flying') != unit.get('is_flying'):
                    continue
                other_position = reservations.get(other, friend['position'])
                clearance = unit.get('radius', 0.5) + friend.get('radius', 0.5) + 0.1
                # Escape from existing spawn overlaps is not penalized as if
                # entering a new occupied space.
                old_gap = math.dist(origin, other_position)
                end_gap = math.dist(destination, other_position)
                if end_gap < clearance and end_gap < old_gap + 0.1:
                    collision_cost += 2.0 * (clearance - end_gap) / clearance
                if old_gap >= clearance and _distance_to_segment(other_position, origin, destination) < clearance:
                    collision_cost += 0.6
            danger = threat_count(state, actor, destination, params)
            # One observed blocked direct move triggers a single lateral
            # preference, never a long global path search or permanent ban.
            blocked_penalty = 2.0 if stalled and int(action) == previous_action else 0.0
            score = remaining + collision_cost + 0.12 * danger + blocked_penalty
            scores[str(action)] = round(score, 4)
            candidates[str(action)] = destination
            if params.get('approach_progress_guard'):
                progress_features[str(action)] = {
                    'base_score': score, 'estimated_contact_windows': remaining,
                    'collision_cost': collision_cost, 'visible_danger_count': danger,
                    'blocked_move_penalty': blocked_penalty}
        avoid_return = [a for a in moves if same_target and loop < avoid_until
                        and math.dist(candidates[str(a)], history['approach_position']) < 0.3]
        eligible = [a for a in moves if a not in avoid_return]
        # A single remaining legal route must never be permanently forbidden.
        cycle_filter_applied = bool(avoid_return and eligible)
        action = min(eligible or moves, key=lambda a: (scores[str(a)], int(a)))
        reason = 'stable_visible_approach'
        wait_count = history.get('approach_wait_steps', 0) if recent else 0
        # A healthy ranged actor may wait at most one step for a nearby melee
        # screen, only if currently outside every visible enemy envelope. A
        # legal attack never reaches this branch.
        melee_friends = [a for a, u in allies.items() if a != actor and u.get('alive')
                         and _melee(state, a) and math.dist(origin, u['position']) <= 4.0]
        maximum = unit.get('max_health', 0) + unit.get('max_shield', 0) or 1
        healthy = (unit.get('health', 0) + unit.get('shield', 0)) / maximum >= 0.5
        if (not _melee(state, actor) and healthy and melee_friends and wait_count < 1
                and threat_count(state, actor, origin, params) == 0
                and threat_count(state, actor, candidates[str(action)], params) > 0
                and all(math.dist(allies[a]['position'], target_position) >
                        math.dist(origin, target_position) + 0.5 for a in melee_friends)):
            action, reason = 1, 'bounded_wait_for_frontline'
        else:
            action = int(action)
            if stalled:
                reason = 'observed_stall_route_reassessment'
            if cycle_filter_applied:
                reason = 'bounded_cycle_escape'
        progress_diagnostic, progress_memory = None, None
        if params.get('approach_progress_guard'):
            action, progress_diagnostic, progress_memory = guard_approach(
                state, actor, history, target, candidates, progress_features, action, eligible or moves)
            if progress_diagnostic.get('applied'):
                reason = 'personal_approach_progress_guard'
        destination = origin if action == 1 else candidates[str(action)]
        reservations[actor] = destination
        actions[actor] = action
        updates[actor] = {'approach_target': target, 'approach_position': list(origin),
                          'approach_loop': state.get('game_loop', 0), 'approach_action': action,
                          'approach_expected_travel': math.dist(origin, destination),
                          'approach_wait_steps': wait_count + 1 if action == 1 else wait_count,
                          'approach_target_distance': target_distance,
                          'approach_previous_position': list(history['approach_position']) if same_target else None,
                          'approach_previous_target_distance': history.get('approach_target_distance') if same_target else None,
                          'approach_cycle_avoid_until_loop': avoid_until}
        diagnostics[actor] = {'target': target, 'retained_target': kept,
                              'estimated_target_windows': {e: round(v, 3) for e, v in costs.items()},
                              'observed_previous_displacement': displacement,
                              'observed_stall': stalled, 'direct_action': int(direct),
                              'cycle_detected': cycle_detected,
                              'two_window_target_progress': net_progress,
                              'cycle_avoid_until_loop': avoid_until,
                              'cycle_avoided_actions': [int(a) for a in avoid_return] if cycle_filter_applied else [],
                              'scores': scores, 'predicted_destinations': candidates,
                              'selected_action': action, 'reason': reason,
                              'meaning': 'Visible-only short-window estimates, not confirmed path or firing.'}
        if progress_diagnostic is not None:
            updates[actor]['approach_progress'] = progress_memory
            diagnostics[actor]['progress_guard'] = progress_diagnostic
    return actions, diagnostics, updates
