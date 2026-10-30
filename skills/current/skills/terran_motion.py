"""Conditional one-window Terran motion estimates, never a path guarantee.

Error allowances use only previous observations of the same tagged actor.
They are engineering uncertainty allowances, not calibrated confidence bounds.
An attack/heal command can move its actor; a preserved heal order is not Stop.
"""
import math

from .movement_scoring import predicted_move
from .race_movement_guard import fresh_history
from .heal_assign import HEAL_ORDER_ABILITY


PARAMS = {'base_error': .15, 'turn_error': .25, 'crowding_error': .12,
          'crowding_margin': .5, 'unreliable_error': 1.5}


V7_ENABLED = True
V7_PARAMS = {'startup_fraction': .253, 'acceleration_increment_fraction': .45,
             'moving_error': .25, 'startup_error': .15}


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def observed_heal_target(unit):
    orders=unit.get('orders',[])
    return orders[0].get('target_unit') if orders and orders[0].get('ability_id')==HEAL_ORDER_ABILITY else None


def observed_attack_target(unit):
    orders=unit.get('orders',[])
    return orders[0].get('target_unit') if orders and orders[0].get('ability_id')==23 else None


def estimate(state, actor, action, history=None, kind=None, target=None, enabled=True):
    unit = state['allies'][actor];origin = list(unit['position'])
    move = str(action) in state.get('movement_candidates', {}).get(actor, {})
    kind=kind or ('move' if move else 'stop' if action==1 else
                  'attack' if action>=6 and unit.get('unit_type_id')!=54 else 'unknown')
    speed = state.get('unit_type_profiles', {}).get(str(unit['unit_type_id']), {}).get('movement_speed_native')
    step = state.get('step_mul')
    row = {'actor':actor, 'action':action, 'kind':kind or 'unknown', 'target':target,
           'center':origin, 'radius':None, 'known':False, 'reasons':[],
           'model':'one-window conditional estimate; collision/pathing/following not guaranteed'}
    if not finite(speed) or speed < 0 or not finite(step) or step <= 0:
        row['reasons'].append('unknown_native_speed_or_window');return row
    travel = speed * step / 16.0
    past = fresh_history(state, actor, history)
    velocity = [origin[i]-past['position'][i] for i in range(2)] if past.get('position') is not None else None
    allowance = PARAMS['base_error'] if enabled else 0.0
    phase = kind
    if move:
        row['kind']='move';center=list(predicted_move(state, actor, action)['destination'])
        if enabled and velocity is not None:
            desired=[center[i]-origin[i] for i in range(2)]
            a=math.hypot(*velocity);b=math.hypot(*desired)
            if a > .05 and b > .05 and sum(x*y for x,y in zip(velocity,desired))/(a*b) < .5:
                allowance += PARAMS['turn_error'];row['reasons'].append('turning_from_observed_travel')
        if enabled and V7_ENABLED and unit.get('unit_type_id') == 54:
            desired=[center[i]-origin[i] for i in range(2)];length=math.hypot(*desired)
            previous_speed=math.hypot(*velocity) if velocity is not None else None
            if previous_speed is None:
                phase='unknown_start';distance=travel*.5;allowance=travel*.5
            elif previous_speed < .05:
                phase='startup';distance=travel*V7_PARAMS['startup_fraction'];allowance=V7_PARAMS['startup_error']
            else:
                cosine=sum(x*y for x,y in zip(velocity,desired))/(previous_speed*length) if length else 1
                phase='turn' if cosine<.5 else 'accelerating' if previous_speed<travel*.8 else 'cruise'
                distance=min(travel,previous_speed+travel*V7_PARAMS['acceleration_increment_fraction'])
                allowance=V7_PARAMS['moving_error']+(PARAMS['turn_error'] if phase=='turn' else 0)
            center=[origin[i]+desired[i]*min(length,distance)/length for i in range(2)] if length else origin
            row['reasons'].append('medivac_'+phase)
    elif action == 1:
        center=origin;row['kind']='stop'
    elif not enabled:
        center=origin;row['reasons'].append('motion_component_disabled_current_reference')
    elif (velocity is not None and past.get('target')==target
          and ((kind in ('heal','keep') and past.get('kind') in ('heal','keep')
                and observed_heal_target(unit)==target and past.get('observed_heal_target')==target)
               or (kind=='attack' and past.get('kind')==kind and past.get('action')==action
                   and target is not None and observed_attack_target(unit)==target
                   and past.get('observed_attack_target')==target))):
        length=math.hypot(*velocity);scale=min(1.0,travel/length) if length else 1.0
        center=[origin[i]+velocity[i]*scale for i in range(2)]
        row['reasons'].append('same_order_observed_velocity')
    else:
        center=origin;allowance=max(allowance,travel)
        row['reasons'].append('automatic_motion_not_yet_observed')
    row['phase']=phase
    if enabled and past.get('predicted_center') is not None and (not V7_ENABLED or unit.get('unit_type_id')!=54 or past.get('phase')==phase):
        residual=math.dist(origin,past['predicted_center'])
        row['previous_prediction_error']=round(residual,6)
        allowance=max(allowance,residual)
        if residual > PARAMS['unreliable_error']:
            row['reasons'].append('large_previous_prediction_error')
    if enabled and not unit.get('is_flying'):
        crowded=sum(1 for other,u in state['allies'].items() if other!=actor and u['alive']
                    and not u.get('is_flying') and math.dist(origin,u['position']) <=
                    unit.get('radius',.5)+u.get('radius',.5)+PARAMS['crowding_margin'])
        if crowded:
            allowance += min(crowded,4)*PARAMS['crowding_error']
            row['reasons'].append('nearby_ground_bodies')
        row['nearby_ground_bodies']=crowded
    row.update(center=center,radius=round(allowance,6),known=True,
               reliable='large_previous_prediction_error' not in row['reasons'],travel_limit=round(travel,6))
    return row


def pair_geometry(patient, healer, limit):
    if limit is None or not patient.get('known') or not healer.get('known'):
        return {'known':False,'reason':'unknown_pair_motion_or_public_range'}
    distance=math.dist(patient['center'],healer['center'])
    error=patient['radius']+healer['radius']
    return {'known':True,'center_gap':round(distance,6),
            'lower_gap':round(max(0.0,distance-error),6),'upper_gap':round(distance+error,6),
            'clearance':round(limit-distance-error,6),
            'reliable':patient.get('reliable',True) and healer.get('reliable',True)}


def possible_exposure(threats, origin, prediction):
    """Known weapon envelopes intersecting the estimated swept uncertainty."""
    if not prediction.get('known'):return None
    end=prediction['center'];vector=[end[i]-origin[i] for i in range(2)]
    square=sum(x*x for x in vector);count=0
    for position,reach in threats.values():
        t=max(0.,min(1.,sum((position[i]-origin[i])*vector[i] for i in range(2))/square)) if square else 0.
        closest=[origin[i]+t*vector[i] for i in range(2)]
        count+=math.dist(position,closest)<=reach+prediction['radius']
    return count


def remember(state, estimates):
    return {a:{'tag':state['allies'][a].get('tag'),'loop':state.get('game_loop',0),
               'position':list(state['allies'][a]['position']),'action':r['action'],
               'kind':r['kind'],'phase':r.get('phase'),'target':r.get('target'),'predicted_center':list(r['center']),
               'observed_heal_target':observed_heal_target(state['allies'][a]),
               'observed_attack_target':observed_attack_target(state['allies'][a]),
               'error_allowance':r['radius']}
            for a,r in estimates.items() if state['allies'][a]['alive'] and r.get('known')}
