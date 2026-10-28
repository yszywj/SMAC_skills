"""Conservative overlay guards; static geometry is not a path/hit guarantee."""
from math import dist

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
                             history=None, require_safer=False):
    """Never replace an attack or worsen retreat exposure/re-entry estimates."""
    moves = state.get('movement_candidates', {}).get(actor, {})
    if str(baseline) not in moves or str(proposal) not in moves:
        return False
    if reverses_move(state, actor, proposal, history):
        return False
    origin = state['allies'][actor]['position']
    old = predicted_move(state, actor, baseline)['destination']
    new = predicted_move(state, actor, proposal)['destination']
    def risk(pos):
        return (threat_count(state, actor, pos, parameters),
                visible_volley_pressure(state, actor, pos, parameters))
    old_risk, new_risk, current = risk(old), risk(new), risk(origin)
    if any(n > min(o, c) for n, o, c in zip(new_risk, old_risk, current)):
        return False
    if require_safer and new_risk == old_risk:
        return False
    # Preserve each known re-entry option, not just proximity to a friendly.
    for enemy, relation in state.get('relations', {}).get(actor, {}).items():
        if relation.get('weapon_range_status') == 'no_compatible_weapon':
            continue
        before = estimated_attack_timing(state, actor, old, enemy, True, parameters)
        after = estimated_attack_timing(state, actor, new, enemy, True, parameters)
        if before is not None and (after is None or after['estimated_shot_loops'] > before['estimated_shot_loops']):
            return False
    return True
