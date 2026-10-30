"""Local retreat/therapy coordination; geometric estimates are not healing proof.

The healer keeps its single selected heal command. Its current position is a
reference, NOT a forecast of automatic following. Only already-retreating
patients' supplied legal moves may be changed; actual orders are checked next
observation. No health delta is interpreted as restored health.
"""
import math

from .heal_assign import HEAL_ORDER_ABILITY
from .movement_scoring import predicted_move
from .race_movement_guard import fresh_history, retreat_override_allowed
from . import terran_motion
from .heal_assign import is_healer, score_heal_need


PARAMS = {'enabled': True, 'healer_away_tolerance': .25}

V6_PARAMS = {'enabled': True, 'known_threats': True, 'motion': True,
             'approach': True, 'keep_order': True, 'coordination': True,
             'reentry_slack': True, 'blocked_reversal': True}
V6_LIMITS = {'approach_max_windows': 2, 'approach_min_progress': .1,
             'coordination_min_gain': .1, 'keep_max_windows': 1}


V7_PARAMS = {'enabled': True, 'stall_windows': 2, 'max_unconfirmed_follow_windows': 4, 'energy_drop': .05, 'approach_windows': 4}


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def public_heal_range(state):
    """Read the public SMAC center-distance limit; never infer it from a mask."""
    sources = [state.get('action_contract', {}).get('candidate_center_distance_by_type', {}),
               state.get('scenario_brief', {}).get('candidate_ranges', {})]
    values = [row['medivac'] for row in sources if 'medivac' in row]
    if not values or any(not _number(v) or v <= 0 for v in values):
        return None
    return float(values[0]) if all(v == values[0] for v in values) else None


def _observed_target(unit):
    orders = unit.get('orders', [])
    return orders[0].get('target_unit') if orders and orders[0].get('ability_id') == HEAL_ORDER_ABILITY else None


def _heal_action(state, healer, target):
    explicit = (state.get('legal_actions') or {}).get(healer)
    for action, entry in state.get('heal_candidates', {}).get(healer, {}).items():
        if entry.get('target') == target and (explicit is None or int(action) in {int(a) for a in explicit}):
            return int(action)
    return None


def _legal_move(state, actor, action):
    explicit = (state.get('legal_actions') or {}).get(actor)
    return (str(action) in state.get('movement_candidates', {}).get(actor, {})
            and (explicit is None or action in {int(a) for a in explicit}))


def select_heal_escort(state, actor, baseline, role, healing, final_actions,
                       parameters=None, histories=None, pressure_guard=None,
                       combat_baseline=None):
    """Return a proposal; the caller commits it once under the existing budget.

    ``baseline`` is the v4 support-retreat result. ``pressure_guard`` retains
    the Terran controller's known-compatible threat check in addition to the
    shared exposure, reversal and re-entry guard.
    """
    row = {'actor': actor, 'baseline_action': baseline, 'selected_action': baseline,
           'applied': False, 'reason': 'disabled', 'candidates': {},
           'geometry_model': 'one-window straight-line patient travel without collisions/pathing; current healer position reference; following unknown'}
    def skip(reason):
        row['reason'] = reason
        return row
    if not PARAMS['enabled']:
        return row
    unit = state['allies'][actor]
    if role != 'retreat' or not unit['alive'] or unit.get('is_flying') is not False or not _legal_move(state, actor, baseline):
        return skip('not_ground_retreat_move')
    healers = [h for h, t in healing.get('assignments', {}).items() if t == actor]
    if len(healers) != 1:
        return skip('no_unique_planned_pair')
    healer = healers[0]
    medical = state['allies'].get(healer, {})
    expected = _heal_action(state, healer, actor)
    row.update(healer=healer, patient=actor,
               pair_status='observed' if _observed_target(medical) == actor else 'pending')
    if (not medical.get('alive') or healer in healing.get('self_preserving', {})
            or not _number(medical.get('energy')) or medical['energy'] < 1):
        return skip('healer_unavailable')
    if expected is None or final_actions[int(healer[1:])] != expected:
        return skip('final_healer_action_not_legal_heal')
    limit = public_heal_range(state)
    if limit is None:
        return skip('unknown_public_heal_range')
    row['candidate_range'] = limit
    speed = state.get('unit_type_profiles', {}).get(str(unit['unit_type_id']), {}).get('movement_speed_native')
    if not _number(speed) or speed <= 0 or not _number(state.get('step_mul')) or state['step_mul'] <= 0:
        return skip('unknown_patient_travel')
    origin = unit['position'];anchor = medical['position']
    if math.dist(origin, anchor) > limit:
        return skip('current_pair_outside_public_range')
    past = fresh_history(state, healer, (histories or {}).get(healer))
    if past.get('position') is not None:
        away = math.dist(anchor, origin) - math.dist(past['position'], origin)
        row['observed_healer_away_displacement'] = round(away, 6)
        if away > PARAMS['healer_away_tolerance']:
            return skip('healer_recently_moving_away')
    old = predicted_move(state, actor, baseline)['destination']
    row['baseline_predicted_clearance'] = round(limit - math.dist(old, anchor), 6)
    if math.dist(old, anchor) <= limit:
        return skip('baseline_retains_range_reference')
    choices = []
    for action in sorted(int(a) for a in state.get('movement_candidates', {}).get(actor, {})):
        if action == baseline or not _legal_move(state, actor, action):
            continue
        pos = predicted_move(state, actor, action)['destination']
        clearance = limit - math.dist(pos, anchor)
        safe = retreat_override_allowed(state, actor, baseline, action, parameters,
                                        (histories or {}).get(actor))
        if combat_baseline is not None:
            safe = safe and retreat_override_allowed(state, actor, combat_baseline, action,
                                                       parameters, (histories or {}).get(actor))
        compatible_safe = pressure_guard(action) if pressure_guard is not None else False
        row['candidates'][str(action)] = {'predicted_clearance': round(clearance, 6),
                                         'shared_guard': bool(safe), 'compatible_pressure_guard': bool(compatible_safe)}
        if clearance >= 0 and safe and compatible_safe:
            choices.append((-clearance, action))
    if not choices:
        return skip('no_safe_range_preserving_move')
    _, selected = min(choices)
    row.update(selected_action=selected, applied=True, reason='preserve_heal_candidate_range',
               selected_predicted_clearance=row['candidates'][str(selected)]['predicted_clearance'])
    return row


def observe_heal_continuity(state, healing, final_actions, previous=None):
    """Audit consecutive planned pairs using observed orders and current legality.

    A legal observed order window is not a claim of heal ticks or survival.
    Range failure means loss of the public candidate envelope, not proven beam
    interruption. Full health, death, energy and target changes stay distinct.
    """
    prior = previous or {};observations = {};memory = {}
    loop = state.get('game_loop', 0);step = state.get('step_mul', 8)
    limit = public_heal_range(state)
    for healer, record in prior.items():
        target = record['target'];h = state['allies'].get(healer, {});p = state['allies'].get(target, {})
        if (record.get('loop') != loop - step or record.get('healer_tag') != h.get('tag')
                or record.get('patient_tag') != p.get('tag')):
            continue
        order = _observed_target(h) == target
        legal = _heal_action(state, healer, target) is not None
        if not h.get('alive'):reason = 'healer_dead'
        elif not p.get('alive'):reason = 'patient_dead'
        elif p.get('health', 0) >= p.get('max_health', 0):reason = 'patient_full_health'
        elif not _number(h.get('energy')) or h['energy'] < 1:reason = 'insufficient_or_unknown_energy'
        elif limit is not None and math.dist(h['position'], p['position']) > limit:reason = 'outside_public_candidate_range'
        elif not legal:reason = 'no_legal_heal_candidate'
        elif not order:reason = 'different_or_missing_observed_order'
        else:reason = 'observed_legal_order'
        streak = record.get('observed_legal_windows', 0) + 1 if reason == 'observed_legal_order' else 0
        observations[healer] = {'patient': target, 'reason': reason, 'order_observed': order,
                                'legal_candidate': legal, 'observed_legal_windows': streak,
                                'previous_pair_status': record.get('pair_status')}
        if h.get('alive') and p.get('alive'):
            observed_gap = math.dist(h['position'], p['position'])
            observations[healer]['actual_center_gap'] = round(observed_gap, 6)
            observations[healer]['previous_predicted_gap'] = record.get('predicted_gap')
            predicted = record.get('patient_predicted_position')
            if predicted is not None:
                observations[healer]['patient_position_prediction_error'] = round(math.dist(p['position'], predicted), 6)
            anchor = record.get('healer_reference_position')
            if anchor is not None:
                observations[healer]['healer_observed_displacement'] = round(math.dist(h['position'], anchor), 6)
    for healer, target in healing.get('assignments', {}).items():
        expected = _heal_action(state, healer, target)
        if expected is None or final_actions[int(healer[1:])] != expected:
            continue
        h = state['allies'][healer];p = state['allies'][target]
        if not h.get('alive') or not p.get('alive') or not _number(h.get('energy')) or h['energy'] < 1:
            continue
        old = observations.get(healer, {})
        streak = old.get('observed_legal_windows', 0) if old.get('patient') == target else 0
        memory[healer] = {'target': target, 'healer_tag': h.get('tag'), 'patient_tag': p.get('tag'),
                          'loop': loop, 'observed_legal_windows': streak,
                          'pair_status': 'observed' if _observed_target(h) == target else 'pending'}
        action = final_actions[int(target[1:])]
        speed = state.get('unit_type_profiles', {}).get(str(p['unit_type_id']), {}).get('movement_speed_native')
        if (_legal_move(state, target, action) and _number(speed) and speed > 0
                and _number(state.get('step_mul')) and state['step_mul'] > 0):
            predicted = predicted_move(state, target, action)['destination']
            memory[healer].update(patient_predicted_position=list(predicted),
                healer_reference_position=list(h['position']),
                predicted_gap=round(math.dist(predicted, h['position']), 6))
    return observations, memory


def keep_action(state, healer):
    """Only the adapter's explicit, enabled no-new-command candidate is usable."""
    contract=state.get('action_contract',{}).get('keep_heal_order',{})
    candidate=state.get('keep_heal_order_candidates',{}).get(healer,{})
    action=contract.get('action_id')
    if (not V6_PARAMS['enabled'] or not V6_PARAMS['keep_order'] or contract.get('enabled') is not True
            or not isinstance(action,int) or isinstance(action,bool) or action < 6
            or candidate.get('action_id') != action):
        return None
    medical=state['allies'].get(healer,{})
    target=candidate.get('target');patient=state['allies'].get(target,{})
    explicit=(state.get('legal_actions') or {}).get(healer)
    if (not medical.get('alive') or _observed_target(medical)!=target
            or not patient.get('alive') or is_healer(patient) or patient.get('is_flying') is not False
            or patient.get('health',0)>=patient.get('max_health',0)
            or not _number(medical.get('energy')) or medical['energy']<1
            or (explicit is not None and action not in {int(a) for a in explicit})):
        return None
    return action


def select_keep_orders(state, healing, previous, safe, rescue_parameters):
    """Request at most one preserved-order window per out-of-range episode.

    This changes no raw command into a heal. Existing observed occupancy stays
    reserved even when another legal urgent patient preempts preservation.
    """
    decisions={};memory={};outcomes={};loop=state.get('game_loop',0);step=state.get('step_mul',8)
    for healer,old in (previous or {}).items():
        if not old.get('selected') or old.get('loop')!=loop-step:continue
        target=old['target'];h=state['allies'].get(healer,{});p=state['allies'].get(target,{})
        same_identity=old.get('healer_tag')==h.get('tag') and old.get('patient_tag')==p.get('tag')
        order_same=_observed_target(h)==target;available=_heal_action(state,healer,target) is not None
        reason=('identity_changed' if not same_identity else 'healer_dead' if not h.get('alive') else
                'patient_dead' if not p.get('alive') else 'patient_full_health' if p['health']>=p['max_health'] else
                'insufficient_or_unknown_energy' if not _number(h.get('energy')) or h['energy']<1 else
                'order_changed_or_missing' if not order_same else 'legal_heal_available' if available else
                'order_observed_candidate_still_unavailable')
        row={'target':target,'reason':reason,'same_identity':same_identity,'order_same':order_same,
             'candidate_available':available,'healer_alive':bool(h.get('alive')),'patient_alive':bool(p.get('alive')),
             'energy':h.get('energy'),'observed_target':_observed_target(h),
             'patient_full_health':p.get('health',0)>=p.get('max_health',1)}
        if same_identity and h.get('alive') and p.get('alive'):
            row['gap_progress']=round(old['gap']-math.dist(h['position'],p['position']),6)
        outcomes[healer]=row
    for healer,target in healing.get('observed_orders',{}).items():
        h=state['allies'][healer];p=state['allies'][target]
        row={'target':target,'selected':False,'reason':'no_explicit_keep_candidate'}
        decisions[healer]=row
        previous_row=(previous or {}).get(healer,{})
        valid=(previous_row.get('loop')==loop-step and previous_row.get('healer_tag')==h.get('tag')
               and previous_row.get('patient_tag')==p.get('tag') and previous_row.get('target')==target)
        old=previous_row if valid else {}
        gap=math.dist(h['position'],p['position'])
        if old:
            row.update(previous_keep_selected=old.get('selected',False),
                       observed_gap_progress=round(old['gap']-gap,6),order_still_observed=True)
        if _heal_action(state,healer,target) is not None:
            row['reason']='legal_heal_available_reissue';continue
        used=old.get('used',0)
        record={'target':target,'healer_tag':h.get('tag'),'patient_tag':p.get('tag'),
                'loop':loop,'gap':gap,'used':used,'selected':False}
        # Retain the consumed token while this same observed order remains
        # outside the legal heal set, even after a rejected cancellation.
        if used:memory[healer]=record
        action=keep_action(state,healer)
        if action is None:continue
        if V7_PARAMS['enabled']:
            energy_drop=(old.get('energy',h['energy'])-h['energy']) if old else 0
            progress=old.get('gap',gap)-gap
            productive=bool(old and energy_drop>V7_PARAMS['energy_drop'])
            following=bool(old and progress>=V6_LIMITS['approach_min_progress'])
            stalled=0 if productive or following else old.get('stalled',0)+int(bool(old))
            unconfirmed=0 if productive else old.get('unconfirmed',0)+1
            record.update(energy=h['energy'],stalled=stalled,unconfirmed=unconfirmed)
            row.update(order_mode='treatment_evidence' if productive else 'following' if following else 'observing',energy_drop=energy_drop,gap_progress=progress,stalled=stalled,unconfirmed=unconfirmed)
            if stalled>=V7_PARAMS['stall_windows'] or unconfirmed>V7_PARAMS['max_unconfirmed_follow_windows']:
                row['reason']='order_without_sufficient_progress';memory[healer]=record;continue
        elif used>=V6_LIMITS['keep_max_windows']:
            row['reason']='preserved_order_window_exhausted';continue
        safety=safe(healer,action,target)
        row['safety']=safety if isinstance(safety,dict) else {'allowed':bool(safety)}
        if healer in healing.get('self_preserving',{}) or not row['safety']['allowed']:
            row['reason']='unsafe_preserved_order_motion';continue
        other=healing.get('assignments',{}).get(healer)
        if other is not None and other!=target:
            u=state['allies'][other];ratio=u['health']/max(1,u['max_health'])
            loss=u.get('health_lost_since_previous_decision',0)/max(1,u['max_health'])
            urgent=(ratio<=rescue_parameters['critical_patient_ratio'] or
                    (ratio<=rescue_parameters['rapid_loss_patient_ratio'] and loss>=rescue_parameters['rapid_loss_ratio']))
            if urgent and score_heal_need(state,other)>=score_heal_need(state,target)+rescue_parameters['rescue_urgency_margin']:
                row.update(reason='necessary_legal_rescue',rescue_target=other);continue
        row.update(selected=True,reason='preserve_productive_or_following_order' if V7_PARAMS['enabled'] else 'preserve_observed_order_one_window',action=action,
                   pair_mode='keep',displaced_legal_assignment=other)
        memory[healer]={**record,'used':used+1,'selected':True}
        healing['assignments'].pop(healer,None)
        healing['healer_actions'][healer]=action
        healing.get('released',{}).pop(healer,None)
        healing.setdefault('retained_orders',{})[healer]=True
        healing.setdefault('preserved_order_requests',{})[healer]=target
        # A displaced matching witness no longer describes the final plan.
        healing.pop('unit_value_diagnostics',None)
    return decisions,memory,outcomes


def coordinate_v6(state, actions, roles, healing, legacy_actions, memory,
                  patient_guard, healer_guard, keep_decisions):
    """Select once from v4 fallback, actual-pair retention and approach intents.

    patient_guard returns an independently checked safety report for the
    original combat and legacy fallback moves. It owns bounded exceptions.
    """
    selected=list(actions);decisions={};intents={};approach_log={};pair_modes={};approach_outcomes={}
    limit=public_heal_range(state);motion_history=memory.get('v6_motion',{})
    def prediction(actor,action,kind=None,target=None):
        enemy=f'e{action-6}'
        if (kind is None and not is_healer(state['allies'][actor]) and action>=6
                and state.get('relations',{}).get(actor,{}).get(enemy,{}).get('smac_attack_available')):
            kind='attack';target=enemy
        return terran_motion.estimate(state,actor,action,motion_history.get(actor),kind,target,V6_PARAMS['motion'])
    def geometry(actor,pa,healer,ha,mode):
        return terran_motion.pair_geometry(prediction(actor,pa),prediction(healer,ha,mode,actor),limit)
    budget=memory.get('support_retreat_uses',{})
    for actor,action in legacy_actions.items():selected[int(actor[1:])]=action
    pairs={h:(p,'heal') for h,p in healing.get('assignments',{}).items()
           if _heal_action(state,h,p)==selected[int(h[1:])]}
    pairs.update({h:(r['target'],'keep') for h,r in keep_decisions.items() if r['selected']})
    for healer,(actor,mode) in pairs.items():
        pair_modes[healer]={'target':actor,'mode':mode}
        before=actions[int(actor[1:])];baseline=selected[int(actor[1:])]
        old=geometry(actor,baseline,healer,selected[int(healer[1:])],mode)
        row={'actor':actor,'healer':healer,'pair_mode':mode,'combat_baseline_action':before,
             'baseline_action':baseline,'selected_action':baseline,'applied':False,
             'baseline_geometry':old,'candidates':{},'reason':'protected_patient_role'}
        decisions[actor]=row
        if V7_PARAMS['enabled'] and mode=='keep':
            row['reason']='native_order_range_not_new_heal_range';continue
        if roles.get(actor)!='retreat' or not _legal_move(state,actor,baseline):continue
        if budget.get(actor,0)>=2:row['reason']='existing_support_retreat_budget_exhausted';continue
        if not V6_PARAMS['coordination']:row['reason']='coordination_component_disabled';continue
        if not old.get('known') or not old.get('reliable'):
            row['reason']='uncertain_pair_prediction';continue
        if old['upper_gap']<=limit:row['reason']='baseline_envelope_within_public_range';continue
        options=[]
        for action in sorted(int(a) for a in state.get('movement_candidates',{}).get(actor,{})):
            if action==baseline or not _legal_move(state,actor,action):continue
            geo=geometry(actor,action,healer,selected[int(healer[1:])],mode)
            guard=patient_guard(actor,before,baseline,action,healer)
            row['candidates'][str(action)]={'geometry':geo,'guard':guard}
            if (geo.get('known') and geo.get('reliable') and guard['allowed'] and geo['upper_gap']<=limit
                    and old['upper_gap']-geo['upper_gap']>=V6_LIMITS['coordination_min_gain']):
                options.append((geo['upper_gap'],action))
        if not options:row['reason']='no_safe_envelope_preserving_move';continue
        _,action=min(options);selected[int(actor[1:])]=action
        row.update(selected_action=action,applied=True,reason='anticipate_pair_range_loss',
                   selected_geometry=row['candidates'][str(action)]['geometry'],
                   selected_guard=row['candidates'][str(action)]['guard'])
    approach_windows=V7_PARAMS['approach_windows'] if V7_PARAMS['enabled'] else V6_LIMITS['approach_max_windows']
    if V6_PARAMS['approach'] and limit is not None:
        occupied=set(healing.get('observed_orders',{}).values())|set(healing.get('assignments',{}).values())
        loop=state.get('game_loop',0);step=state.get('step_mul',8)
        for healer,old in memory.get('v6_approach_intents',{}).items():
            if old.get('loop')!=loop-step:continue
            target=old['target'];h=state['allies'].get(healer,{});p=state['allies'].get(target,{})
            reason=('healer_dead' if not h.get('alive') else 'patient_dead' if not p.get('alive') else
                    'identity_changed' if old.get('healer_tag')!=h.get('tag') or old.get('patient_tag')!=p.get('tag') else
                    'patient_full_health' if p['health']>=p['max_health'] else
                    'legal_heal_available' if _heal_action(state,healer,target) is not None else 'pending_approach')
            outcome={'target':target,'reason':reason,'previous_status':old.get('status','active')}
            if reason not in ('healer_dead','patient_dead','identity_changed'):
                outcome['gap_progress']=round(old['gap']-math.dist(h['position'],p['position']),6)
            approach_outcomes[healer]=outcome
        for healer in sorted(healing.get('healers',[])):
            h=state['allies'][healer];log={'selected':False,'reason':'not_idle_available_healer'};approach_log[healer]=log
            if (healer in pairs or _observed_target(h) is not None or healer in healing.get('self_preserving',{})
                    or not _number(h.get('energy')) or h['energy']<1):continue
            old=memory.get('v6_approach_intents',{}).get(healer,{})
            fresh=(old.get('loop')==loop-step and old.get('healer_tag')==h.get('tag'))
            if not fresh:old={}
            if old.get('status')=='cooldown':
                if loop<old['cooldown_until']:
                    intents[healer]={**old,'loop':loop};log['reason']='approach_intent_cooldown';continue
                old={}
            if old.get('windows',0)>=approach_windows:
                log['reason']='approach_intent_expired'
                intents[healer]={**old,'loop':loop,'cooldown_until':loop+step,'status':'cooldown'};continue
            if old.get('target') in state['allies']:
                target=state['allies'][old['target']]
                if (target['alive'] and old.get('patient_tag')==target.get('tag')
                        and old.get('gap',0)-math.dist(h['position'],target['position'])<V6_LIMITS['approach_min_progress']):
                    log['reason']='approach_intent_no_observed_progress'
                    intents[healer]={**old,'loop':loop,'cooldown_until':loop+step,'status':'cooldown'};continue
            options=[]
            for actor,p in state['allies'].items():
                if (actor in occupied or not p['alive'] or p.get('is_flying') is not False or is_healer(p)
                        or p['health']>=p['max_health'] or _heal_action(state,healer,actor) is not None):continue
                gap=math.dist(p['position'],h['position'])
                if old.get('target')==actor and old.get('patient_tag')==p.get('tag'):
                    if old.get('gap',gap)-gap<V6_LIMITS['approach_min_progress']:
                        log['reason']='approach_intent_no_observed_progress';continue
                baseline=selected[int(actor[1:])];before=actions[int(actor[1:])]
                choices=[baseline]
                if roles.get(actor)=='retreat' and budget.get(actor,0)<2 and _legal_move(state,actor,baseline) and V6_PARAMS['coordination']:
                    choices=sorted(set(choices)|{int(a) for a in state.get('movement_candidates',{}).get(actor,{}) if _legal_move(state,actor,int(a))})
                for ha in sorted(int(a) for a in state.get('movement_candidates',{}).get(healer,{}) if _legal_move(state,healer,int(a))):
                    if not healer_guard(healer,selected[int(healer[1:])],ha,actor):continue
                    hp=prediction(healer,ha,'move',actor)
                    for pa in choices:
                        guard={'allowed':True,'exceptions':[],'reason':'patient_action_unchanged'}
                        if pa!=baseline:
                            guard=patient_guard(actor,before,baseline,pa,healer)
                            if not guard['allowed']:continue
                        pp=prediction(actor,pa);geo=terran_motion.pair_geometry(pp,hp,limit)
                        if (not geo.get('known') or not geo.get('reliable') or gap-geo['center_gap' if V7_PARAMS['enabled'] else 'upper_gap']<V6_LIMITS['approach_min_progress']
                                or geo['upper_gap']>limit+hp['travel_limit']*(approach_windows-1 if V7_PARAMS['enabled'] else 1)):continue
                        eta=1+max(0.,geo['upper_gap']-limit)/max(.01,hp['travel_limit'])
                        loss=p.get('health_lost_since_previous_decision',0)
                        if V7_PARAMS['enabled'] and (eta>approach_windows or (loss>0 and p['health']/loss<=eta)):
                            continue
                        continuity=old.get('target')==actor
                        options.append((not continuity,-score_heal_need(state,actor),geo['upper_gap'],actor,ha,pa,geo,guard))
            if not options:
                if log['reason']=='not_idle_available_healer':log['reason']='no_safe_short_horizon_approach'
                continue
            _,_,_,actor,ha,pa,geo,guard=min(options,key=lambda r:r[:6])
            p=state['allies'][actor];before=actions[int(actor[1:])];baseline=selected[int(actor[1:])]
            selected[int(healer[1:])]=ha;selected[int(actor[1:])]=pa;occupied.add(actor)
            windows=old.get('windows',0)+1 if old.get('target')==actor else 1
            intents[healer]={'target':actor,'healer_tag':h.get('tag'),'patient_tag':p.get('tag'),
                'loop':loop,'windows':windows,'gap':math.dist(p['position'],h['position']),
                'expires_loop':loop+(approach_windows-windows)*step,
                'healer_action':ha,'patient_action':pa,'geometry':geo,'mode':'approach_not_healing','status':'active'}
            log.update(selected=True,reason='approach_unoccupied_wounded_ally',target=actor,
                       healer_action=ha,patient_action=pa,geometry=geo,windows=windows,
                       patient_baseline_action=baseline,patient_combat_baseline_action=before,
                       patient_role=roles.get(actor))
            pair_modes[healer]={'target':actor,'mode':'approach'}
            if pa!=baseline:
                decisions[actor]={'actor':actor,'healer':healer,'pair_mode':'approach','combat_baseline_action':before,
                    'baseline_action':baseline,'selected_action':pa,'applied':True,'reason':'bounded_approach_coordination',
                    'selected_geometry':geo,'selected_guard':guard,'candidates':{}}
    # Monitor every observed or requested pair, irrespective of patient role.
    for healer,target in healing.get('observed_orders',{}).items():
        pair_modes.setdefault(healer,{'target':target,'mode':'observed_order_not_selected'})
    estimates={};monitor={}
    for actor,u in state['allies'].items():
        if not u['alive']:continue
        pair=pair_modes.get(actor,{})
        mode=pair.get('mode') if pair.get('mode') in ('heal','keep') else None
        estimates[actor]=prediction(actor,selected[int(actor[1:])],mode,pair.get('target'))
    for healer,pair in pair_modes.items():
        target=pair['target']
        if healer not in estimates or target not in estimates:continue
        monitor[healer]={**pair,'patient_role':roles.get(target),'patient_action':selected[int(target[1:])],
            'healer_action':selected[int(healer[1:])],'order_observed':_observed_target(state['allies'][healer])==target,
            'controller_heal_candidate':_heal_action(state,healer,target) is not None,
            'patient_prediction':estimates[target],'healer_prediction':estimates[healer],
            'geometry':terran_motion.pair_geometry(estimates[target],estimates[healer],limit)}
    return {'actions':selected,'decisions':decisions,'approach_intents':intents,'approach_decisions':approach_log,
            'approach_outcomes':approach_outcomes,'pair_monitor':monitor,'motion_memory':terran_motion.remember(state,estimates)}
