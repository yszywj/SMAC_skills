"""Conservative overlay guards; static geometry is not a path/hit guarantee."""
from math import dist, isfinite

from .movement_scoring import (predicted_move, threat_count,
                               visible_volley_pressure, estimated_attack_timing)


def fresh_history(state, actor, history):
    history = history or {}
    return history if (history.get('tag') == state['allies'][actor].get('tag')
                       and history.get('loop') == state.get('game_loop', 0) - state.get('step_mul', 8)) else {}


def reverses_move(state, actor, action, history):
    history = fresh_history(state, actor, history)
    if {2: 3, 3: 2, 4: 5, 5: 4}.get(history.get('action')) == action:
        return True
    previous = history.get('position')
    return bool(previous is not None and history.get('action') in (2, 3, 4, 5)
                and dist(predicted_move(state, actor, action)['destination'], previous) < .3)


def retreat_override_allowed(state, actor, baseline, proposal, parameters=None,
                             history=None, require_safer=False, coordination=None):
    """Preserve legacy checks unless bounded critical-heal coordination is supplied."""
    return evaluate_retreat_override(state, actor, baseline, proposal, parameters,
                                     history, require_safer, coordination)['allowed']


def _critical_heal_context(state, actor, coordination):
    """A caller's intent cannot fabricate a live critical patient/healing relation."""
    context = coordination or {}
    unit = state['allies'][actor]
    healer = context.get('healer')
    medical = state['allies'].get(healer, {})
    def number(value):
        return isinstance(value, (float, int)) and not isinstance(value, bool) and isfinite(value)
    hp, maximum, energy = unit.get('health'), unit.get('max_health'), medical.get('energy')
    if (not unit.get('alive') or unit.get('is_flying') is not False
            or not number(hp) or not number(maximum) or maximum <= 0 or not 0 < hp / maximum <= .25
            or not medical.get('alive') or not number(energy) or energy < 1):
        return False
    explicit = (state.get('legal_actions') or {}).get(healer)
    candidate = any(row.get('target') == actor and (explicit is None or int(action) in {int(a) for a in explicit})
                    for action, row in state.get('heal_candidates', {}).get(healer, {}).items())
    orders = medical.get('orders', [])
    observed = bool(orders and orders[0].get('ability_id') == 386 and orders[0].get('target_unit') == actor)
    return candidate or observed


def evaluate_retreat_override(state, actor, baseline, proposal, parameters=None,
                              history=None, require_safer=False, coordination=None):
    """Explain safety and the narrowly bounded Terran healing exceptions.

    Exposure and visible base-volley pressure may never increase. Critical
    patients with a real candidate/order may protect their explicit primary
    target instead of every enemy, with at most one control window of extra
    estimated delay. A reversal additionally needs verified lack of progress,
    a closer healing reference and an unused per-episode reversal allowance.
    These remain static estimates, not future damage or healing guarantees.
    """
    context = coordination or {}
    row = {'allowed': False, 'reasons': [], 'exceptions': [], 'risk': {}, 'reentry': {}}
    moves = state.get('movement_candidates', {}).get(actor, {})
    if str(baseline) not in moves or str(proposal) not in moves:
        row['reasons'].append('not_supplied_move')
        return row
    critical = _critical_heal_context(state, actor, context) if context else False
    origin = state['allies'][actor]['position']
    old = predicted_move(state, actor, baseline)['destination']
    new = predicted_move(state, actor, proposal)['destination']
    reversal = reverses_move(state, actor, proposal, history)
    row['reversal'] = reversal
    if reversal:
        past = fresh_history(state, actor, history)
        anchor = state['allies'].get(context.get('healer'), {}).get('position')
        blocked = (past.get('position') is not None and past.get('action') in (2, 3, 4, 5)
                   and dist(origin, past['position']) < .15)
        closer = anchor is not None and dist(new, anchor) + .1 < dist(old, anchor)
        if (critical and context.get('allow_blocked_reversal') is True
                and context.get('reversal_used') is False and blocked and closer):
            row['exceptions'].append('critical_heal_blocked_reversal')
        else:
            row['reasons'].append('reverses_previous_move')
    def risk(pos):
        return (threat_count(state, actor, pos, parameters),
                visible_volley_pressure(state, actor, pos, parameters))
    old_risk, new_risk, current = risk(old), risk(new), risk(origin)
    row['risk'] = {'origin': current, 'baseline': old_risk, 'proposal': new_risk}
    for name, n, o, c in zip(('threat_count', 'base_volley_pressure'), new_risk, old_risk, current):
        if n > min(o, c):
            row['reasons'].append(name + '_increased')
    if require_safer and new_risk == old_risk:
        row['reasons'].append('no_strict_risk_improvement')
    primary = context.get('primary_target')
    primary_relation = state.get('relations', {}).get(actor, {}).get(primary)
    relaxed = (critical and context.get('allow_reentry_delay') is True
               and primary in state.get('visible_enemies', {}) and primary_relation is not None
               and primary_relation.get('weapon_range_status') != 'no_compatible_weapon'
               and estimated_attack_timing(state, actor, old, primary, True, parameters) is not None)
    step = state.get('step_mul', 8)
    if not isinstance(step, (float, int)) or isinstance(step, bool) or not isfinite(step) or step <= 0:
        relaxed = False
    # The legacy mode protects each known re-entry option. Healing mode protects
    # only a caller's explicit primary target, never a guessed nearest enemy.
    for enemy, relation in state.get('relations', {}).get(actor, {}).items():
        if relation.get('weapon_range_status') == 'no_compatible_weapon':
            continue
        if relaxed and enemy != primary:
            continue
        before = estimated_attack_timing(state, actor, old, enemy, True, parameters)
        after = estimated_attack_timing(state, actor, new, enemy, True, parameters)
        row['reentry'][enemy] = {'baseline': before, 'proposal': after,
                                 'allowed_extra_loops': step if relaxed else 0}
        if before is not None:
            if after is None or after['estimated_shot_loops'] > before['estimated_shot_loops'] + (step if relaxed else 0):
                row['reasons'].append('reentry_delayed:' + enemy)
            elif relaxed and after['estimated_shot_loops'] > before['estimated_shot_loops']:
                row['exceptions'].append('critical_heal_one_window_reentry')
    if relaxed:
        row['protected_primary_target'] = primary
    row['allowed'] = not row['reasons']
    return row
