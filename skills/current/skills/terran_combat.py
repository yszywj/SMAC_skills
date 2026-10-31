"""Terran composition: v4 combat/healing plus local retreat/therapy coordination.

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
from .heal_assign import (assign_healing, is_healer, healer_endangered,
                          HEAL_ORDER_ABILITY, health_ratio, score_heal_need)
from .movement_scoring import predicted_move
from .race_movement_guard import retreat_override_allowed
from .heal_escort import (PARAMS as ESCORT_PARAMS, select_heal_escort,
                          observe_heal_continuity, V6_PARAMS, V6_LIMITS, V7_PARAMS, keep_action,
                          select_keep_orders, coordinate_v6)
from .race_movement_guard import evaluate_retreat_override
from . import terran_motion


PARAMS = {
    'stable_patient_ratio': .55,
    'stable_recent_loss_ratio': .10,
    'critical_patient_ratio': .25,
    'rapid_loss_ratio': .20,
    'rapid_loss_patient_ratio': .50,
    'rescue_urgency_margin': .25,
    'rescue_switch_cooldown_windows': 3,
}


def _legal(state, actor):
    unit = state['allies'][actor]
    if not unit['alive']:
        return {0}
    actions = legal_actions(state, actor)
    explicit = (state.get('legal_actions') or {}).get(actor)
    if is_healer(unit):
        candidates = {int(a) for a in state.get('heal_candidates', {}).get(actor, {})}
        keep = keep_action(state, actor)
        if keep is not None:candidates.add(keep)
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


def _reserve_observed_heal_orders(state, memory=None):
    """No simultaneous swaps: release occupied targets only after observation.

    Even an order being cancelled this step reserves its target until the next
    snapshot. A plan in race_memory is not evidence that an order took effect.
    """
    observed, owners, locked, rescue_options, order_states, confirmations = {}, {}, {}, {}, {}, {}
    loop = state.get('game_loop', 0)
    step = state.get('step_mul', 8)
    for actor, unit in state['allies'].items():
        if not unit['alive'] or not is_healer(unit):
            continue
        prior = (memory or {}).get('heal_order_state', {}).get(actor, {})
        if prior.get('tag') != unit.get('tag') or prior.get('loop') not in (loop, loop - step):
            prior = {}
        order_states[actor] = {'tag': unit.get('tag'), 'loop': loop,
                               'last_rescue_request_loop': prior.get('last_rescue_request_loop')}
        if prior.get('pending_target'):
            confirmations[actor] = {'requested_target': prior['pending_target'], 'status': 'not_observed'}
    for actor, unit in sorted(state['allies'].items()):
        if not unit['alive'] or not is_healer(unit):
            continue
        orders = unit.get('orders', [])
        if orders and orders[0].get('ability_id') == HEAL_ORDER_ABILITY:
            target = orders[0].get('target_unit')
            if target in state['allies'] and state['allies'][target]['alive']:
                observed[actor] = target
                owners.setdefault(target, actor)
                if actor in confirmations:
                    confirmations[actor]['status'] = ('observed' if confirmations[actor]['requested_target'] == target
                                                       else 'different_order_observed')
        order_states[actor]['observed_target'] = observed.get(actor)
    for actor, candidates in state.get('heal_candidates', {}).items():
        own = observed.get(actor)
        valid_own = (own is not None and owners.get(own) == actor
                     and not healer_endangered(state, actor)
                     and any(c['target'] == own for c in candidates.values()))
        permitted = {own} if valid_own else None
        if valid_own:
            patient = state['allies'][own]
            prior_request = order_states[actor]['last_rescue_request_loop']
            cooldown_ready = (prior_request is None or loop - prior_request >= step * PARAMS['rescue_switch_cooldown_windows'])
            stable = (health_ratio(patient) >= PARAMS['stable_patient_ratio']
                      and patient.get('health_lost_since_previous_decision', 0) / (patient['max_health'] or 1)
                          < PARAMS['stable_recent_loss_ratio'])
            urgent = []
            if stable and cooldown_ready:
                current_need = score_heal_need(state, own)
                for entry in candidates.values():
                    target = entry['target']
                    if target == own or target in owners:
                        continue
                    ally = state['allies'][target]
                    ratio = health_ratio(ally)
                    recent_loss = ally.get('health_lost_since_previous_decision', 0) / (ally['max_health'] or 1)
                    critical = (ratio <= PARAMS['critical_patient_ratio'] or (
                        ratio <= PARAMS['rapid_loss_patient_ratio'] and recent_loss >= PARAMS['rapid_loss_ratio']))
                    if critical and score_heal_need(state, target) >= current_need + PARAMS['rescue_urgency_margin']:
                        urgent.append(target)
            if urgent:
                permitted.update(urgent)
                rescue_options[actor] = {'current_target': own, 'candidates': sorted(urgent),
                                         'reason': 'stable_patient_and_more_urgent_unoccupied_legal_target'}
        state['heal_candidates'][actor] = {
            a: c for a, c in candidates.items()
            if owners.get(c['target'], actor) == actor and (permitted is None or c['target'] in permitted)}
        if valid_own and actor not in rescue_options:
            locked[actor] = own
    return observed, locked, rescue_options, order_states, confirmations


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
    # Optional default-off pair-scorer injection (tree_skills medivac channel):
    # strip it before deepcopy so the base planner contract is unchanged.
    pair_scorer = kwargs.pop('pair_scorer', None)
    pair_scoring = kwargs.pop('pair_scoring', 'off')
    pair_histories = kwargs.pop('pair_histories', None)
    pair_strict = kwargs.pop('pair_strict', False)
    options = copy.deepcopy(kwargs)
    memory = copy.deepcopy(race_memory or {})
    if V6_PARAMS['enabled'] and V6_PARAMS['known_threats']:
        options['parameters']={**(options.get('parameters') or {}),'known_compatible_threats':True}
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
    observed_orders, locked_orders, rescue_options, order_states, confirmations = _reserve_observed_heal_orders(healing_state, memory)
    healing = assign_healing(healing_state, n_agents,
                             previous_assignments=observed_orders,
                             unit_values=options.get('unit_values'),
                             pair_scorer=pair_scorer,
                             pair_scoring=pair_scoring,
                             pair_histories=pair_histories,
                             pair_strict=pair_strict)
    healing['observed_orders']=observed_orders
    keep_decisions={};keep_memory={};keep_outcomes={}
    if V6_PARAMS['enabled'] and V6_PARAMS['keep_order']:
        def safe_keep(healer,action,target):
            prediction=terran_motion.estimate(local,healer,action,memory.get('v6_motion',{}).get(healer),
                                              'keep',target,V6_PARAMS['motion'])
            threats=healer_threats[healer]
            current=_pressure(threats,local['allies'][healer]['position'])
            possible=terran_motion.possible_exposure(threats,local['allies'][healer]['position'],prediction)
            return {'allowed':bool(prediction.get('known') and prediction.get('reliable') and possible<=current),
                    'current_compatible_threats':current,'possible_compatible_threats':possible,
                    'prediction':prediction,'meaning':'uncertainty envelope checked; automatic following is conditional'}
        keep_decisions,keep_memory,keep_outcomes=select_keep_orders(local,healing,memory.get('v6_keep_order',{}),safe_keep,PARAMS)
    rescue_requests = {}
    for healer, target in healing['assignments'].items():
        if healer in rescue_options and target != observed_orders[healer]:
            rescue_requests[healer] = {'from': observed_orders[healer], 'to': target}
            order_states[healer].update(last_rescue_request_loop=local.get('game_loop', 0), pending_target=target)
    healing.update(observed_orders=observed_orders, locked_orders=locked_orders,
                   rescue_options=rescue_options, rescue_requests=rescue_requests, rescue_confirmations=confirmations,
                   handoff_policy='reserve observed target until a later snapshot confirms release')
    changes = {};reset = set();retreat_uses = memory.get('support_retreat_uses', {})
    # Restore v2 support positioning and approach behavior; old coverage memory
    # deliberately has no action-producing path in this rescue-only variant.
    for actor in healing.get('healers', []):
        index = int(actor[1:])
        if index >= n_agents:continue
        before = result['actions'][index]
        if keep_decisions.get(actor,{}).get('selected'):
            action=keep_decisions[actor]['action'];reason='preserve_observed_heal_order'
        elif actor in healing['assignments']:
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
    escort_decisions = {};v6_result={};reversal_used=dict(memory.get('v6_reversal_used',{}))
    if V6_PARAMS['enabled']:
        legacy_actions={}
        for actor in result['retreat_units']:
            if actor in healer_threats:continue
            before=result['actions'][int(actor[1:])]
            legacy_actions[actor]=(_support_retreat(local,actor,before,healing['assignments'],options.get('parameters'),
                                                  memory.get('actors',{}).get(actor))
                                   if retreat_uses.get(actor,0)<2 else before)
        def patient_guard(actor,before,baseline,proposal,healer):
            primary=result.get('engagement_state_updates',{}).get(actor,{}).get('engagement',{}).get('target')
            context={'healer':healer,'primary_target':primary,
                     'allow_reentry_delay':bool(V6_PARAMS['reentry_slack']),
                     'allow_blocked_reversal':bool(V6_PARAMS['blocked_reversal']),
                     'reversal_used':bool(reversal_used.get(actor))}
            reports=[evaluate_retreat_override(local,actor,a,proposal,options.get('parameters'),
                        memory.get('actors',{}).get(actor),coordination=context) for a in sorted({before,baseline})]
            threats=_threats(local,actor)
            proposed=predicted_move(local,actor,proposal)['destination']
            pressure_ok=_pressure(threats,proposed)<=min(_pressure(threats,local['allies'][actor]['position']),
                *[_pressure(threats,predicted_move(local,actor,a)['destination']) for a in {before,baseline}])
            return {'allowed':pressure_ok and all(r['allowed'] for r in reports),'reports':reports,
                    'compatible_pressure_guard':pressure_ok,
                    'exceptions':sorted({e for r in reports for e in r['exceptions']})}
        def healer_guard(healer,baseline,proposal,target):
            if proposal not in _legal(local,healer):return False
            threats=healer_threats[healer];origin=local['allies'][healer]['position']
            old=predicted_move(local,healer,baseline)['destination'] if str(baseline) in local['movement_candidates'].get(healer,{}) else origin
            new=predicted_move(local,healer,proposal)['destination']
            estimate=terran_motion.estimate(local,healer,proposal,memory.get('v6_motion',{}).get(healer),
                                           'move',target,V6_PARAMS['motion'])
            possible=terran_motion.possible_exposure(threats,origin,estimate)
            return (estimate.get('known') and estimate.get('reliable')
                    and possible<=min(_pressure(threats,origin),_pressure(threats,old))
                    and _pressure(threats,new)<=min(_pressure(threats,origin),_pressure(threats,old)))
        v6_result=coordinate_v6(local,result['actions'],result['roles'],healing,legacy_actions,memory,
                                 patient_guard,healer_guard,keep_decisions)
        for actor,unit in local['allies'].items():
            index=int(actor[1:]);before=result['actions'][index];action=v6_result['actions'][index]
            if before==action:continue
            result['actions'][index]=action
            if actor in healer_threats:
                reason='bounded_approach_intent';result['roles'][actor]='support'
            else:
                decision=v6_result['decisions'].get(actor,{})
                assert result['roles'][actor]=='retreat' and retreat_uses.get(actor,0)<2
                reason='bounded_terran_coordination' if decision.get('applied') else 'bounded_support_retreat'
                result['retreat_actions'][actor]=action;result['retreat_reasons'][actor]=reason
                retreat_uses[actor]=retreat_uses.get(actor,0)+1
                if 'critical_heal_blocked_reversal' in decision.get('selected_guard',{}).get('exceptions',[]):
                    reversal_used[actor]=True
            changes[actor]={'before':before,'after':action,'reason':reason}
            _clear_actor(result,actor);reset.add(actor)
    for actor in list(result['retreat_units']):
        if V6_PARAMS['enabled']:break
        if actor in healer_threats:continue
        index = int(actor[1:]);before = result['actions'][index]
        if retreat_uses.get(actor, 0) >= 2:
            if ESCORT_PARAMS['enabled']:
                escort_decisions[actor] = {'actor':actor, 'baseline_action':before,
                    'selected_action':before, 'applied':False, 'reason':'existing_support_retreat_budget_exhausted'}
            continue
        action = _support_retreat(local, actor, before, healing['assignments'], options.get('parameters'),
                                  memory.get('actors', {}).get(actor))
        legacy_action = action
        if ESCORT_PARAMS['enabled']:
            threats = _threats(local, actor)
            origin = local['allies'][actor]['position']
            def pressure_guard(proposal):
                proposed_pos = predicted_move(local, actor, proposal)['destination']
                original_pos = predicted_move(local, actor, before)['destination']
                baseline_pos = predicted_move(local, actor, legacy_action)['destination']
                return _pressure(threats, proposed_pos) <= min(_pressure(threats, origin),
                    _pressure(threats, original_pos), _pressure(threats, baseline_pos))
            decision = select_heal_escort(local, actor, legacy_action, result['roles'][actor],
                healing, result['actions'], options.get('parameters'), memory.get('actors', {}), pressure_guard,
                combat_baseline=before)
            decision['combat_baseline_action'] = before
            escort_decisions[actor] = decision
            action = decision['selected_action']
        if action != before:
            result['actions'][index] = action;result['retreat_actions'][actor] = action
            escort_applied = escort_decisions.get(actor, {}).get('applied', False)
            result['retreat_reasons'][actor] = ('bounded_heal_escort' if escort_applied
                                              else 'bounded_move_toward_healing_support')
            retreat_uses[actor] = retreat_uses.get(actor, 0) + 1
            changes[actor] = {'before':before,'after':action,
                             'reason':'bounded_heal_escort' if escort_applied else 'bounded_support_retreat'}
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
    continuity, escort_pairs = observe_heal_continuity(local, healing, result['actions'],
        memory.get('heal_escort_pairs', {})) if ESCORT_PARAMS['enabled'] else ({}, {})
    result['race_memory'] = {'heal_assignments':dict(healing.get('assignments', {})),
                             'observed_heal_orders':observed_orders,
                             'heal_order_state':order_states,
                             'support_retreat_uses':dict(retreat_uses),
                             'heal_escort_pairs':escort_pairs,
                             'actors':{a:dict(tag=u.get('tag'),loop=local.get('game_loop',0),
                                              action=result['actions'][int(a[1:])],position=list(u['position']))
                                       for a,u in local['allies'].items() if u['alive'] and int(a[1:]) < n_agents}}
    if V6_PARAMS['enabled']:
        result['race_memory'].update(v6_motion=v6_result.get('motion_memory',{}),
            v6_approach_intents=v6_result.get('approach_intents',{}),v6_keep_order=keep_memory,
            v6_reversal_used=reversal_used)
    result['race_telemetry'] = {'profile':'terran','version':('terran-specialization-v7' if V7_PARAMS['enabled'] else 'terran-specialization-v6') if V6_PARAMS['enabled'] else 'terran-specialization-v5','healing':healing,'changes':changes,
        'terran_coordination':{'enabled':bool(V6_PARAMS['enabled']),'parameters':dict(V6_PARAMS),'limits':dict(V6_LIMITS),
            'motion_parameters':dict(terran_motion.PARAMS),'v7_parameters':dict(V7_PARAMS),'v7_motion_parameters':dict(terran_motion.V7_PARAMS),
            'decisions':v6_result.get('decisions',{}),'pair_monitor':v6_result.get('pair_monitor',{}),
            'approach_intents':v6_result.get('approach_intents',{}),
            'approach_decisions':v6_result.get('approach_decisions',{}),
            'approach_outcomes':v6_result.get('approach_outcomes',{}),'keep_order':keep_decisions,'keep_outcomes':keep_outcomes,
            'meaning':'conditional movement estimates and observed orders; approach/keep are not heal assignments'},
        'heal_escort':{'enabled':bool(ESCORT_PARAMS['enabled']), 'decisions':escort_decisions,
                       'continuity':continuity, 'planned_pairs':escort_pairs,
                       'parameters':dict(ESCORT_PARAMS),
                       'meaning':'candidate-range geometry and observed orders only; no healing-amount or saved-unit claim'},
        'coverage':{}, 'coverage_policy':'disabled; v2 friendly support and wounded approach', 'parameters':dict(PARAMS),
        'filtered_incompatible_attacks':filtered,'air_threats':{a:sorted(t) for a,t in healer_threats.items()},
        'reset_actor_histories':sorted(reset),'damage_budget':'recomputed_from_final_attack_assignments',
        'support_retreat_limit':'at most two redirected retreat moves per actor per episode',
        'order_semantics':'heal action reissued; Stop only for explicit idle cancellation, never order retention'}
    return result
