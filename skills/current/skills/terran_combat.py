"""Initial Terran composition: existing combat plus joint Medivac support.

Only public observations and supplied legal candidates are used. Position
scores are geometric heuristics, not pathing or guaranteed healing. This
module never invents a hold-order/no-op: Stop is a real cancelling command.
Inputs are copied; the caller owns/reset the returned race_memory per episode.
"""
import copy
import math

from .adaptive_focus_fire import allocate_adaptive_focus_fire
from .focus_fire import _compatible_weapon, _countable, expected_damage, fallback_action
from .group_geometry import legal_actions, main_group, centroid
from .heal_assign import assign_healing, is_healer, healer_endangered, HEAL_ORDER_ABILITY
from .movement_scoring import predicted_move
from .race_movement_guard import retreat_override_allowed


def _legal(state, actor):
    unit = state['allies'][actor]
    if not unit['alive']:
        return {0}
    actions = legal_actions(state, actor)
    explicit = (state.get('legal_actions') or {}).get(actor)
    if is_healer(unit):
        candidates = {int(a) for a in state.get('heal_candidates', {}).get(actor, {})}
    else:
        candidates = {6 + int(e[1:]) for e, relation in state.get('relations', {}).get(actor, {}).items()
                      if relation.get('smac_attack_available')}
    return actions | candidates if explicit is None else actions & (candidates | {1} | {
        int(a) for a in state.get('movement_candidates', {}).get(actor, {})})


def _compatible(state, actor, enemy):
    unit = state['allies'][actor]
    relation = state.get('relations', {}).get(actor, {}).get(enemy, {})
    if is_healer(unit) or relation.get('weapon_range_status') == 'no_compatible_weapon':
        return False
    profile = state.get('unit_type_profiles', {}).get(str(unit['unit_type_id']), {})
    if profile.get('known') and profile.get('weapons'):
        return _compatible_weapon(profile, state['visible_enemies'][enemy]) is not None
    return True  # Unknown data does not manufacture a new legal target.


def _threats(state, actor):
    """Known compatible enemies only; unknown compatibility is not asserted."""
    unit = state['allies'][actor]
    rows = {}
    for enemy, relation in state.get('relations', {}).get(actor, {}).items():
        target = state.get('visible_enemies', {}).get(enemy)
        if target is None or relation.get('enemy_weapon_range_status') == 'no_compatible_weapon':
            continue
        profile = state.get('unit_type_profiles', {}).get(str(target['unit_type_id']), {})
        weapon = _compatible_weapon(profile, unit)
        if profile.get('known') and profile.get('weapons') and weapon is None:
            continue  # A catalog-confirmed ground-only weapon is not an air threat.
        confirmed = relation.get('enemy_weapon_range_status') in ('in_range', 'out_of_range')
        if not weapon and not confirmed:
            continue
        reach = ((weapon or {}).get('range') or 0) + unit.get('radius', 0) + target.get('radius', 0)
        # When catalog range is absent, current relation supplies only a
        # conservative local estimate; do not invent an exact native range.
        if not weapon:
            reach = relation.get('distance', 0) if relation.get('in_enemy_weapon_range') else 0
        rows[enemy] = (target['position'], reach)
    return rows


def _pressure(threats, position):
    return sum(math.dist(position, p) <= reach for p, reach in threats.values())


def _support_action(state, actor, threats):
    """Safe idle support follows the largest friendly combat group."""
    friends = [a for a, u in state['allies'].items() if u['alive'] and not is_healer(u)]
    group = main_group(state, friends) if friends else []
    anchor = centroid(state['allies'][a]['position'] for a in group)
    origin = state['allies'][actor]['position']
    choices = []
    allowed = _legal(state, actor)
    if 1 in allowed:
        choices.append((1, origin))  # explicit idle Stop; never healing retention
    choices += [(int(a), predicted_move(state, actor, a)['destination'])
                for a in state.get('movement_candidates', {}).get(actor, {}) if int(a) in allowed]
    if not choices:
        raise ValueError(f'{actor}: no legal support move or Stop')
    def score(row):
        action, pos = row
        gap = max(0.0, math.dist(pos, anchor) - 2.0) if anchor is not None else 0.0
        nearest = min((math.dist(pos, p) for p, _ in threats.values()), default=100.0)
        # Air threats decide danger; incompatible nearby ground units never
        # push a Medivac away from healing coverage.
        return (_pressure(threats, pos), gap, -min(nearest, 8.0), action != 1, action)
    return min(choices, key=score)[0]


def _support_retreat(state, actor, baseline_action, assignments=None, parameters=None, history=None):
    healers = [a for a, u in state['allies'].items() if u['alive'] and is_healer(u)
               and isinstance(u.get('energy'), (int, float)) and u['energy'] >= 1
               and (assignments or {}).get(a) == actor
               and any(c['target'] == actor for c in state.get('heal_candidates', {}).get(a, {}).values())]
    if not healers or str(baseline_action) not in state.get('movement_candidates', {}).get(actor, {}):
        return baseline_action
    origin = state['allies'][actor]['position']
    threats = _threats(state, actor)
    baseline = predicted_move(state, actor, baseline_action)['destination']
    current_gap = min(math.dist(origin, state['allies'][h]['position']) for h in healers)
    base_gap = min(math.dist(baseline, state['allies'][h]['position']) for h in healers)
    candidates = []
    for action in state.get('movement_candidates', {}).get(actor, {}):
        if int(action) not in _legal(state, actor):
            continue
        pos = predicted_move(state, actor, action)['destination']
        gap = min(math.dist(pos, state['allies'][h]['position']) for h in healers)
        if (_pressure(threats, pos) <= min(_pressure(threats, origin), _pressure(threats, baseline))
                and gap + .1 < min(current_gap, base_gap)
                and retreat_override_allowed(state, actor, baseline_action, int(action), parameters, history)):
            candidates.append((gap, int(action)))
    return min(candidates)[1] if candidates else baseline_action


def _reserve_observed_heal_orders(state):
    """No simultaneous swaps: release occupied targets only after observation.

    Even an order being cancelled this step reserves its target until the next
    snapshot. A plan in race_memory is not evidence that an order took effect.
    """
    observed, owners, locked = {}, {}, {}
    for actor, unit in sorted(state['allies'].items()):
        if not unit['alive'] or not is_healer(unit):
            continue
        orders = unit.get('orders', [])
        if orders and orders[0].get('ability_id') == HEAL_ORDER_ABILITY:
            target = orders[0].get('target_unit')
            if target in state['allies'] and state['allies'][target]['alive']:
                observed[actor] = target
                owners.setdefault(target, actor)
    for actor, candidates in state.get('heal_candidates', {}).items():
        own = observed.get(actor)
        valid_own = (own is not None and owners.get(own) == actor
                     and not healer_endangered(state, actor)
                     and any(c['target'] == own for c in candidates.values()))
        state['heal_candidates'][actor] = {
            a: c for a, c in candidates.items()
            if owners.get(c['target'], actor) == actor and (not valid_own or c['target'] == own)}
        if valid_own:
            locked[actor] = own
    return observed, locked


def _clear_actor(result, actor):
    result['attack_assignments'].pop(actor, None)
    for key in ('movement_diagnostics', 'approach_diagnostics', 'approach_state_updates',
                'engagement_diagnostics', 'engagement_state_updates', 'safe_reentry_diagnostics',
                'critical_escape_diagnostics', 'planned_reentry_diagnostics', 'reentry_decisions'):
        result.get(key, {}).pop(actor, None)
    result.get('damage_model', {}).get('pairs', {}).pop(actor, None)
    result['reengaged'] = [a for a in result['reengaged'] if a != actor]


def _refresh_budget(state, result, parameters):
    predicted, immediate, window = {}, {}, {}
    for actor, enemy in result['attack_assignments'].items():
        for table in (predicted, immediate, window):table.setdefault(enemy, 0.0)
        relation = state['relations'][actor][enemy]
        damage = expected_damage(state, actor, enemy, parameters)
        if damage is not None and _countable(state, actor, enemy, relation, parameters):
            predicted[enemy] += damage
            (immediate if state['allies'][actor]['weapon_ready'] else window)[enemy] += damage
    result.update(predicted_damage=predicted, immediate_damage=immediate, window_only_damage=window,
                  estimated_overkill={e: max(0.0, d - state['visible_enemies'][e]['health']
                                             - state['visible_enemies'][e]['shield']) for e, d in predicted.items()})


def allocate_terran_combat(state, n_agents, race_memory=None, **kwargs):
    """Same combat kwargs/result contract, with joint healing and race telemetry."""
    local = copy.deepcopy(state)
    options = copy.deepcopy(kwargs)
    memory = copy.deepcopy(race_memory or {})
    filtered = []
    for actor, row in local.get('relations', {}).items():
        for enemy, relation in row.items():
            if enemy in local['visible_enemies'] and not _compatible(local, actor, enemy):
                if relation.get('smac_attack_available'):filtered.append([actor, enemy])
                relation['smac_attack_available'] = False
                relation['weapon_range_status'] = 'no_compatible_weapon'
                relation['in_weapon_range'] = False
    # Respect any stricter external mask before generating the base plan.
    for actor, moves in local.get('movement_candidates', {}).items():
        explicit = (local.get('legal_actions') or {}).get(actor)
        if explicit is not None:
            legal = {int(a) for a in explicit}
            local['movement_candidates'][actor] = {a:v for a,v in moves.items() if int(a) in legal}
            for enemy, relation in local.get('relations', {}).get(actor, {}).items():
                if 6 + int(enemy[1:]) not in legal:relation['smac_attack_available'] = False
    for actor, unit in local['allies'].items():
        if not is_healer(unit):continue
        allowed = _legal(local, actor)
        energy = unit.get('energy')
        local.setdefault('heal_candidates', {})[actor] = {
            a:c for a,c in local.get('heal_candidates', {}).get(actor, {}).items()
            if int(a) in allowed and not (isinstance(energy, (int,float)) and energy < 1)}
    result = allocate_adaptive_focus_fire(local, n_agents, **options)
    healing_state = copy.deepcopy(local)
    healer_threats = {}
    for actor, unit in healing_state['allies'].items():
        if not unit['alive'] or not is_healer(unit):continue
        threats = _threats(local, actor);healer_threats[actor] = threats
        healing_state['relations'][actor] = {e:r for e,r in local.get('relations', {}).get(actor, {}).items() if e in threats}
        for move in healing_state.get('movement_candidates', {}).get(actor, {}).values():
            move['distances_to_visible_enemies'] = {e:math.dist(move['destination'], p) for e,(p,_) in threats.items()}
        allowed = _legal(local, actor)
        energy = unit.get('energy')
        healing_state.setdefault('heal_candidates', {})[actor] = {
            a:c for a,c in local.get('heal_candidates', {}).get(actor, {}).items()
            if int(a) in allowed and not (isinstance(energy, (int,float)) and energy < 1)}
    observed_orders, locked_orders = _reserve_observed_heal_orders(healing_state)
    healing = assign_healing(healing_state, n_agents,
                             previous_assignments=observed_orders,
                             unit_values=options.get('unit_values'))
    healing.update(observed_orders=observed_orders, locked_orders=locked_orders,
                   handoff_policy='reserve observed target until a later snapshot confirms release')
    changes = {};reset = set();retreat_uses = memory.get('support_retreat_uses', {})
    for actor in healing.get('healers', []):
        index = int(actor[1:])
        if index >= n_agents:continue
        before = result['actions'][index]
        if actor in healing['assignments']:
            action = healing['healer_actions'][actor];reason = 'joint_heal_assignment'
        else:
            action = _support_action(local, actor, healer_threats[actor])
            reason = 'air_threat_avoidance_or_friendly_support'
            proposed = healing['healer_actions'].get(actor)
            if proposed in _legal(local, actor) and proposed in (2, 3, 4, 5):
                threats = healer_threats[actor]
                origin = local['allies'][actor]['position']
                chosen_pos = predicted_move(local, actor, action)['destination'] if action != 1 else origin
                proposed_pos = predicted_move(local, actor, proposed)['destination']
                if actor in healing.get('self_preserving', {}) or (
                    healing.get('approach_targets') and local['allies'][actor].get('energy', 0) >= 1
                    and _pressure(threats, proposed_pos) <= min(_pressure(threats, origin), _pressure(threats, chosen_pos))):
                    action = proposed
                    reason = 'healer_self_preservation' if actor in healing.get('self_preserving', {}) else 'approach_wounded_ally'
        result['actions'][index] = action
        result['roles'][actor] = 'support'
        result['retreat_actions'].pop(actor, None);result['retreat_reasons'].pop(actor, None)
        _clear_actor(result, actor);reset.add(actor)
        if before != action:changes[actor] = {'before':before,'after':action,'reason':reason}
    for actor in list(result['retreat_units']):
        if actor in healer_threats or retreat_uses.get(actor, 0) >= 2:continue
        index = int(actor[1:]);before = result['actions'][index]
        action = _support_retreat(local, actor, before, healing['assignments'], options.get('parameters'),
                                  memory.get('actors', {}).get(actor))
        if action != before:
            result['actions'][index] = action;result['retreat_actions'][actor] = action
            result['retreat_reasons'][actor] = 'bounded_move_toward_healing_support'
            retreat_uses[actor] = retreat_uses.get(actor, 0) + 1
            changes[actor] = {'before':before,'after':action,'reason':'bounded_support_retreat'}
            _clear_actor(result, actor);reset.add(actor)
    for actor, unit in local['allies'].items():
        index = int(actor[1:])
        if index >= n_agents:continue
        allowed = _legal(local, actor);action = result['actions'][index]
        if action not in allowed:
            fallback = fallback_action(local, actor) if not is_healer(unit) else None
            if fallback not in allowed:fallback = 1 if 1 in allowed else min(allowed, default=None)
            if fallback is None:raise ValueError(f'{actor}: no legal action')
            result['actions'][index] = fallback
            result['roles'][actor] = 'support' if is_healer(unit) else 'reposition'
            result['retreat_actions'].pop(actor, None);result['retreat_reasons'].pop(actor, None)
            _clear_actor(result, actor);reset.add(actor)
            changes[actor] = {'before':action,'after':fallback,'reason':'legal_fallback'}
    for key, roles in [('attack_units', ('attack','all_in_attack')),('retreat_units', ('retreat',)),('support_units',('support',))]:
        result[key] = sorted(a for a,r in result['roles'].items() if r in roles)
    result['phase'] = 'all_in' if result['all_in'] else ('mixed' if result['retreat_units'] else 'engage')
    _refresh_budget(local, result, options.get('parameters'))
    if reset:
        # Generic value-attribution witnesses describe the pre-overlay plan.
        result['unit_value_diagnostics'] = healing.get('unit_value_diagnostics', {})
    result['race_memory'] = {'heal_assignments':dict(healing.get('assignments', {})),
                             'observed_heal_orders':observed_orders,
                             'support_retreat_uses':dict(retreat_uses),
                             'actors':{a:dict(tag=u.get('tag'),loop=local.get('game_loop',0),
                                              action=result['actions'][int(a[1:])],position=list(u['position']))
                                       for a,u in local['allies'].items() if u['alive'] and int(a[1:]) < n_agents}}
    result['race_telemetry'] = {'profile':'terran','version':'terran-specialization-v2','healing':healing,'changes':changes,
        'filtered_incompatible_attacks':filtered,'air_threats':{a:sorted(t) for a,t in healer_threats.items()},
        'reset_actor_histories':sorted(reset),'damage_budget':'recomputed_from_final_attack_assignments',
        'support_retreat_limit':'at most two redirected retreat moves per actor per episode',
        'order_semantics':'heal action reissued; Stop only for explicit idle cancellation, never order retention'}
    return result
