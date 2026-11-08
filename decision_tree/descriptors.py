"""Public invocation descriptors for sparse candidate-return regression.

Unit identifiers and observed outcomes are deliberately excluded. Presence bits
distinguish an unspecified inherited/default parameter from an explicit zero.
"""
from .features import FEATURE_NAMES

SKILL_NAMES = ('adaptive_focus_fire', 'focus_fire_budget', 'cover_retreat',
               'heal_assign', 'regroup', 'cohesive_search')
POSTURE_NAMES = ('aggressive', 'balanced', 'defensive')
PARAMETER_NAMES = ('max_retreat_steps', 'max_recover_steps', 'stall_steps',
                   'all_in_team_ratio', 'max_retreat_cycles', 'critical_escape_ratio',
                   'disable_all_in', 'disable_retreat')
DESCRIPTOR_NAMES = tuple(
    ['call_skill_' + name for name in SKILL_NAMES]
    + ['call_posture_' + name for name in POSTURE_NAMES]
    + ['call_has_priority_target', 'call_changes_priority_target',
       'call_has_max_steps', 'call_max_steps']
    + [name for parameter in PARAMETER_NAMES
       for name in ('call_has_' + parameter, 'call_' + parameter)]
    + ['call_spacing_' + name for name in ('compact', 'normal', 'spread')]
    + ['call_layout_' + name for name in ('free', 'front_back')])
MODEL_FEATURE_NAMES = tuple(FEATURE_NAMES) + DESCRIPTOR_NAMES


def call_features(call, base_call=None):
    base = base_call or {}
    posture = call.get('posture', base.get('posture'))
    duration = call.get('max_steps', base.get('max_steps'))
    params = dict(base.get('adaptive_params', {})) if call['skill'] == 'adaptive_focus_fire' else {}
    params.update(call.get('adaptive_params', {}))
    movement = call.get('movement_profile', base.get('movement_profile', {})
                        if call['skill'] == base.get('skill') else {})
    result = {'call_skill_' + name: float(call['skill'] == name) for name in SKILL_NAMES}
    result.update({'call_posture_' + name: float(posture == name) for name in POSTURE_NAMES})
    result.update(call_has_priority_target=float(call.get('priority_target') is not None),
                  call_changes_priority_target=float(call.get('priority_target') != base.get('priority_target')),
                  call_has_max_steps=float(duration is not None), call_max_steps=float(duration or 0))
    for name in PARAMETER_NAMES:
        result['call_has_' + name] = float(name in params)
        result['call_' + name] = float(params.get(name, 0))
    result.update({'call_spacing_' + name: float(movement.get('spacing') == name)
                   for name in ('compact', 'normal', 'spread')})
    result.update({'call_layout_' + name: float(movement.get('layout') == name)
                   for name in ('free', 'front_back')})
    return result


def candidate_features(features, call, base_call=None):
    return {**{name: features[name] for name in FEATURE_NAMES}, **call_features(call, base_call)}
