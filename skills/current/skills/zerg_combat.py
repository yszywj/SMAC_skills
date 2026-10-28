"""First Zerg specialization over the unchanged generic combat allocator.

Banelings have an explicit disposable burst role: their empty public weapon
catalog is never converted into invented damage, range or cooldown facts.
Cluster/avoidance distances below are geometric heuristics, NOT explosion
radii. Only legal snapshot moves, Stop and attacks are emitted. All enemy
reasoning uses visible_enemies. Inputs, including episode memory, are pure.
"""
from collections import Counter
from copy import deepcopy
from math import dist

from .adaptive_focus_fire import allocate_adaptive_focus_fire
from .focus_fire import _countable, expected_damage
from .damage_model import expected_attack_damage_details
from .movement_scoring import predicted_move, threat_count
from .combat_timing import can_fire_within_window
from .race_movement_guard import retreat_override_allowed, fresh_history, reverses_move


PARAMS = {
    'cluster_neighborhood': 2.5,  # geometric proxy, never a splash radius
    'enemy_baneling_alert_distance': 3.5,  # geometric precaution, not blast reach
    'minimum_escape_gain': .25,
    'melee_target_load_weight': 1.25,
    'close_baneling_distance': 1.5,  # conservative proximity heuristic, not blast radius
    'max_noncontact_escape_steps': 2,
}


def _kind(unit):
    name = str(unit.get('type', '')).lower()
    for kind, unit_id in (('baneling', 9), ('zergling', 105), ('hydralisk', 107)):
        if unit.get('unit_type_id') == unit_id or name.startswith(kind):
            return kind
    return 'ordinary'


def _alive(unit):
    return bool(unit.get('alive', unit.get('health', 0) > 0)) and unit.get('health', 1) > 0


def _legal_targets(state, actor):
    return sorted((e for e, r in state.get('relations', {}).get(actor, {}).items()
                   if e in state['visible_enemies'] and _alive(state['visible_enemies'][e])
                   and r.get('smac_attack_available')), key=lambda e: int(e[1:]))


def _moves(state, actor):
    return {int(a): predicted_move(state, actor, a)['destination']
            for a in state.get('movement_candidates', {}).get(actor, {}) if int(a) in (2, 3, 4, 5)}


def _crowding(state, actor, position, reservations):
    unit = state['allies'][actor]
    total = 0.
    for other, destination in reservations.items():
        if other == actor or state['allies'][other].get('is_flying') != unit.get('is_flying'):
            continue
        clearance = unit.get('radius', .5) + state['allies'][other].get('radius', .5) + .25
        total += max(0., clearance - dist(position, destination)) / max(clearance, .1)
    return total


def _approach(state, actor, target_position, reservations, stalled=False):
    moves = _moves(state, actor)
    if not moves:
        return 1
    direct = min(moves, key=lambda a: (dist(moves[a], target_position), a))
    def score(action):
        position = moves[action]
        return (dist(position, target_position) + 2.5 * _crowding(state, actor, position, reservations)
                + (2.0 if stalled and action == direct and len(moves) > 1 else 0.0), action)
    return min(moves, key=score)


def _history(memory, state, actor):
    history = memory.get('actors', {}).get(actor, {})
    unit = state['allies'][actor]
    if history.get('tag') != unit.get('tag'):
        return {}
    loop = state.get('game_loop', 0)
    if history.get('loop', loop) > loop:
        return {}
    return history


def _clear_overridden(result, actors):
    for key in ('retreat_actions', 'retreat_reasons', 'movement_diagnostics',
                'approach_diagnostics', 'approach_state_updates', 'safe_reentry_diagnostics',
                'critical_escape_diagnostics', 'planned_reentry_diagnostics',
                'engagement_diagnostics', 'engagement_state_updates', 'reentry_decisions'):
        if isinstance(result.get(key), dict):
            for actor in actors:
                result[key].pop(actor, None)
    result['reengaged'] = [a for a in result.get('reengaged', []) if a not in actors]
    if actors:
        # Attribution from a discarded generic plan must not survive as an
        # explanation of a specialized action, including aggregate counts.
        result['unit_value_diagnostics'] = {}


def _refresh_damage(state, result, params):
    """Rebuild predictions for final ordinary attacks; never credit burst."""
    predicted, immediate, window_only, pairs = {}, {}, {}, {}
    unknown = []
    for actor, target in result['attack_assignments'].items():
        if _kind(state['allies'][actor]) == 'baneling':
            unknown.append(actor)
            continue
        predicted.setdefault(target, 0.)
        immediate.setdefault(target, 0.)
        window_only.setdefault(target, 0.)
        amount = expected_damage(state, actor, target, params)
        if params.get('accurate_damage'):
            pairs[actor] = {target: expected_attack_damage_details(state, actor, target)}
        if amount is None:
            unknown.append(actor)
        elif _countable(state, actor, target, state['relations'][actor][target], params):
            predicted[target] += amount
            (immediate if state['allies'][actor].get('weapon_ready') else window_only)[target] += amount
    for key, value in (('predicted_damage', predicted), ('immediate_damage', immediate),
                       ('window_only_damage', window_only)):
        result[key] = {e: round(v, 3) for e, v in value.items()}
    result['estimated_overkill'] = {
        e: round(max(0., v - state['visible_enemies'][e]['health']
                     - state['visible_enemies'][e].get('shield', 0)), 3)
        for e, v in predicted.items()}
    result['damage_model'] = {'scope': 'final_ordinary_attacks_only; baneling_effect_unknown',
                              'pairs': pairs, 'unknown_damage_actors': sorted(unknown)}


def allocate_zerg_combat(state, n_agents, race_memory=None, **kwargs):
    """Generic result contract plus race_memory and transparent race_telemetry.

    The manager owns episode resets. This function returns fresh memory and
    identifies actors whose old generic approach/engagement history must be
    removed. Banelings (including dead ones) are excluded from generic force
    ratios, damage budgets and low-health retreat decisions.
    """
    memory = race_memory or {}
    params = kwargs.get('parameters') or {}
    allies = state['allies']
    living = sorted((a for a, u in allies.items() if _alive(u) and int(a[1:]) < n_agents),
                    key=lambda a: int(a[1:]))
    banes = [a for a in living if _kind(allies[a]) == 'baneling']
    ordinary = {a: u for a, u in allies.items() if _kind(u) != 'baneling'}
    restricted = dict(state, allies=ordinary)
    for key in ('relations', 'movement_candidates', 'heal_candidates'):
        restricted[key] = {a: rows for a, rows in state.get(key, {}).items() if a in ordinary}
    baseline_kwargs = dict(kwargs)
    for key in ('unit_states', 'previous_attack_assignments', 'move_history'):
        if baseline_kwargs.get(key):
            baseline_kwargs[key] = {a: v for a, v in baseline_kwargs[key].items() if a in ordinary}
    result = allocate_adaptive_focus_fire(restricted, n_agents, **baseline_kwargs)
    if not any(_alive(u) for u in ordinary.values()):
        result.update(all_in=False, all_in_reason=None)
    overrides = set(banes)
    reservations = {a: list(allies[a]['position']) for a in living}
    details, burst_targets, target_load, melee_load = {}, {}, Counter(), Counter()
    next_actors = {}

    # Coordinate actual next-window destinations, not the nominal move end.
    for actor in living:
        if actor in banes:
            continue
        action = result['actions'][int(actor[1:])]
        reservations[actor] = _moves(state, actor).get(action, list(allies[actor]['position']))

    def replace(actor, action, role, reason):
        index = int(actor[1:])
        if action != result['actions'][index] or role != result['roles'].get(actor):
            overrides.add(actor)
        result['actions'][index] = action
        result['roles'][actor] = role
        reservations[actor] = _moves(state, actor).get(action, list(allies[actor]['position']))
        details.setdefault(actor, {}).update(reason=reason, selected_action=action)

    # Jumpers keep melee contact rather than copying ranged cooldown kiting.
    for actor in (a for a in living if _kind(allies[a]) == 'zergling'):
        unit = allies[actor]
        legal = _legal_targets(state, actor)
        history = _history(memory, state, actor)
        max_health = unit.get('max_health', 1) or 1
        preserve_emergency = (result['roles'].get(actor) == 'retreat'
                              and unit.get('health', 0) / max_health < .2)
        if legal and not preserve_emergency:
            target = min(legal, key=lambda e: (dist(unit['position'], state['visible_enemies'][e]['position'])
                          + PARAMS['melee_target_load_weight'] * melee_load[e], e))
            melee_load[target] += 1
            blocked = (history.get('loop') == state.get('game_loop', 0) - state.get('step_mul', 8)
                       and history.get('target') == target and history.get('action', 0) >= 6
                       and history.get('position') is not None
                       and dist(unit['position'], history['position']) < .15
                       and state['relations'][actor][target].get('in_weapon_range') is False)
            blocked_steps = history.get('blocked_steps', 0) + 1 if blocked else 0
            action = 6 + int(target[1:])
            role, reason = 'attack', 'melee_local_target_allocation'
            if blocked_steps >= 2 and _moves(state, actor):
                action = _approach(state, actor, state['visible_enemies'][target]['position'], reservations, stalled=True)
                role, reason = 'reposition', 'observed_melee_blockage_lateral_retry'
            replace(actor, action, role, reason)
            details[actor].update(target=target, blocked_steps=blocked_steps)
        else:
            details.setdefault(actor, {}).update(reason='preserve_emergency' if preserve_emergency else 'generic_coordinated_approach')

    # Hydralisk retreats prefer a non-worse threat envelope near the melee
    # screen. No hold is introduced when the generic allocator can fire.
    screen = [a for a in living if _kind(allies[a]) == 'zergling']
    for actor in (a for a in living if _kind(allies[a]) == 'hydralisk'):
        if result['roles'].get(actor) != 'retreat' or not screen:
            continue
        moves = _moves(state, actor)
        original = result['actions'][int(actor[1:])]
        if original not in moves:
            continue
        eligible = [a for a in moves if a == original or retreat_override_allowed(
            state, actor, original, a, params, _history(memory, state, actor), require_safer=True)]
        action = min(eligible, key=lambda a: (threat_count(state, actor, moves[a], params),
                     min(dist(moves[a], allies[s]['position']) for s in screen)
                     + _crowding(state, actor, moves[a], reservations), a))
        if action != original:
            replace(actor, action, 'retreat', 'hydralisk_retreat_toward_screen')

    # Banelings use only explicit public roles and visible ground targets.
    # Neighbor count represents grouping opportunity, never estimated damage.
    visible_ground = [e for e, u in state['visible_enemies'].items()
                      if _alive(u) and u.get('is_flying') is False]
    for actor in banes:
        unit = allies[actor]
        previous = _history(memory, state, actor).get('target')
        neighborhoods = {e: [other for other in visible_ground
                             if dist(state['visible_enemies'][e]['position'], state['visible_enemies'][other]['position'])
                             <= PARAMS['cluster_neighborhood']] for e in visible_ground}
        legal = _legal_targets(state, actor)
        if visible_ground:
            def rank(enemy):
                nearby = neighborhoods[enemy]
                used = sum(target_load[e] for e in nearby)
                return (.15 * dist(unit['position'], state['visible_enemies'][enemy]['position'])
                        - .9 * len(nearby) + 1.8 * used - (.2 if enemy == previous else 0), enemy)
            target = min(visible_ground, key=rank)
            target_load[target] += 1
            burst_targets[actor] = target
            if target in legal:
                action, role = 6 + int(target[1:]), 'baneling_strike'
            else:
                action = _approach(state, actor, state['visible_enemies'][target]['position'], reservations)
                role = 'baneling_approach' if action != 1 else 'baneling_hold'
            replace(actor, action, role, 'visible_group_geometry_proxy')
            details[actor].update(target=target, nearby_visible_count=len(neighborhoods[target]),
                                  estimated_explosion_damage=None, explosion_radius=None)
        else:
            # No hidden target is retained. Search uses the public map hint.
            destination = state.get('search_location', unit['position'])
            action = _approach(state, actor, destination, reservations) if _moves(state, actor) else 1
            replace(actor, action, 'baneling_search' if action != 1 else 'baneling_hold', 'no_visible_ground_target')

    # Highest-priority ordinary-unit safety pass can replace legal attacks.
    # Separate destinations reduce shared exposure without asserting a hit.
    enemy_banes = [u for u in state['visible_enemies'].values() if _alive(u) and _kind(u) == 'baneling']
    evaders = [a for a in living if a not in banes and enemy_banes]
    evaders.sort(key=lambda a: (min(dist(allies[a]['position'], e['position']) for e in enemy_banes), a))
    for actor in evaders:
        origin = allies[actor]['position']
        near = [e for e in enemy_banes if dist(origin, e['position']) <= PARAMS['enemy_baneling_alert_distance']]
        if not near:
            continue
        before = min(dist(origin, e['position']) for e in enemy_banes)
        original = result['actions'][int(actor[1:])]
        history = fresh_history(state, actor, _history(memory, state, actor))
        streak = history.get('escape_streak', 0)
        # A currently legal, in-range lethal shot takes priority over running.
        # This is an expected ordinary attack, never guaranteed interception.
        intercepts = []
        for enemy in _legal_targets(state, actor):
            target = state['visible_enemies'][enemy]
            damage = expected_damage(state, actor, enemy, params)
            if (_kind(target) == 'baneling' and state['relations'][actor][enemy].get('in_weapon_range') is True
                    and can_fire_within_window(state, actor, enemy, parameters=params)
                    and damage is not None and damage >= target['health'] + target.get('shield', 0)):
                intercepts.append(enemy)
        if intercepts:
            target = min(intercepts, key=lambda e: (dist(origin, state['visible_enemies'][e]['position']), e))
            replace(actor, 6 + int(target[1:]), 'attack', 'feasible_lethal_baneling_interception')
            details[actor]['avoidance_skipped'] = 'expected_lethal_shot_not_confirmed_hit'
            continue
        target = f'e{original - 6}' if original >= 6 else None
        contact = any(dist(origin, e['position']) <= allies[actor].get('radius', .375)
                      + e.get('radius', .375) + .25 for e in near)
        firing = target is not None and can_fire_within_window(state, actor, target, parameters=params)
        if (firing and before > PARAMS['close_baneling_distance']) or (
                streak >= PARAMS['max_noncontact_escape_steps'] and not contact):
            details.setdefault(actor, {})['avoidance_skipped'] = 'preserve_fire_window' if firing else 'bounded_escape_reentry'
            continue
        moves = _moves(state, actor)
        eligible = [a for a in moves if min(dist(moves[a], e['position']) for e in enemy_banes)
                    >= before + PARAMS['minimum_escape_gain']
                    and (contact or not reverses_move(state, actor, a, history))]
        if not eligible:
            continue
        action = min(eligible, key=lambda a: (-min(dist(moves[a], e['position']) for e in enemy_banes)
                    + 3.0 * _crowding(state, actor, moves[a], reservations)
                    + .2 * threat_count(state, actor, moves[a], params), a))
        replace(actor, action, 'retreat', 'anti_baneling_geometric_spread')
        details[actor].update(overridden_action=original, distance_before=before,
                              predicted_distance_after=min(dist(moves[action], e['position']) for e in enemy_banes),
                              blast_radius_known=False, escape_streak=streak + 1, contact_emergency=contact)

    # Reconcile the final plan; do not publish generic plans we discarded.
    _clear_overridden(result, overrides)
    assignments = {}
    for actor in living:
        action = result['actions'][int(actor[1:])]
        legal = _legal_targets(state, actor)
        if action >= 6 and f'e{action - 6}' in legal:
            assignments[actor] = f'e{action - 6}'
        elif action not in (1, *_moves(state, actor)):
            raise ValueError(f'Illegal specialized action for {actor}: {action}')
        if result['roles'][actor] == 'retreat':
            result['retreat_actions'][actor] = action
            result['retreat_reasons'][actor] = details.get(actor, {}).get('reason', 'generic_retreat')
        target = burst_targets.get(actor) or assignments.get(actor)
        prior = _history(memory, state, actor)
        next_actors[actor] = {'tag': allies[actor].get('tag'), 'loop': state.get('game_loop', 0),
                              'target': target, 'position': list(allies[actor]['position']), 'action': action,
                              'escape_streak': details.get(actor, {}).get('escape_streak', 0),
                              'blocked_steps': details.get(actor, {}).get('blocked_steps', 0)}
        if prior.get('loop') == state.get('game_loop', 0):
            next_actors[actor]['blocked_steps'] = prior.get('blocked_steps', 0)
    result['attack_assignments'] = assignments
    result['attack_units'] = sorted(assignments)
    result['retreat_units'] = sorted(a for a in living if result['roles'][a] == 'retreat')
    result['support_units'] = sorted(a for a in living if result['roles'][a] == 'support')
    result['terminated'] = not living
    result['termination_reason'] = 'all_units_dead' if not living else None
    result['phase'] = 'mixed' if result['retreat_units'] else ('all_in' if result['all_in'] else 'engage')
    result['focus_fire_terminated'] = None if assignments else 'no_final_legal_attacks'
    _refresh_damage(state, result, params)
    result['race_memory'] = {'actors': next_actors}
    result['race_telemetry'] = {
        'race': 'zerg', 'version': 'zerg-specialization-v2', 'actors': details,
        'reset_actor_histories': sorted(overrides), 'baneling_targets': burst_targets,
        'baneling_damage_model': 'unknown; excluded from ordinary single-target budget',
        'geometry_proxy_parameters': dict(PARAMS),
        'scope': 'visible geometry heuristics; no explosion radius/damage or confirmed hit claims',
        'capabilities_not_implemented': ['manual_detonation', 'burrow', 'effect_attribution', 'global_pathfinding'],
    }
    return result
