"""Overlay direct medical decisions onto an untouched ground combat plan.

Only common result-contract cleanup is delegated to the existing allocator.
No legacy healing selector, matching, or teacher runs here. Any optional return
selector has already made its choice in the medical runtime before this overlay.
"""
from copy import deepcopy


def apply_tree_plan(state, result, plan, metadata, clear_actor, refresh_budget, parameters=None):
    changes, reset = {}, []
    decisions = deepcopy(plan['decisions'])
    for actor, action in plan['actions'].items():
        index = int(actor[1:])
        before = result['actions'][index]
        if action not in state['legal_actions'][actor]:
            raise ValueError(f'{actor}: direct tree produced illegal action {action}')
        result['actions'][index] = action
        if state['allies'][actor]['alive']:
            result['roles'][actor] = 'support'
        result['retreat_actions'].pop(actor, None)
        result['retreat_reasons'].pop(actor, None)
        clear_actor(result, actor)
        reset.append(actor)
        decisions[actor].update(proposed_action=action, final_action=action,
                                base_action=before, override_reason=None)
        if action != before:
            changes[actor] = {'before':before, 'after':action, 'reason':'direct_medivac_tree'}
    for key, roles in [('attack_units', ('attack', 'all_in_attack')),
                       ('retreat_units', ('retreat',)), ('support_units', ('support',))]:
        result[key] = sorted(a for a, role in result['roles'].items() if role in roles)
    result['phase'] = 'all_in' if result['all_in'] else ('mixed' if result['retreat_units'] else 'engage')
    refresh_budget(state, result, parameters)
    if reset:
        # Pre-overlay attribution does not describe the final medical policy.
        result['unit_value_diagnostics'] = {}
    result['race_memory'] = {'medivac_tree':deepcopy(plan['memory'])}
    result['race_telemetry'] = {
        'profile':'terran', 'version':metadata['version'],
        'medical_patient_coordination':False,
        'reset_actor_histories':reset, 'changes':changes,
        'healing':{'assignments':deepcopy(plan.get('assignments', {})),
                   'healer_actions':dict(plan['actions']),
                   'scoring':('paired_return_gain' if metadata.get('selection_mode') == 'model' else 'none')},
        'medivac_tree':{'mode':'tree', 'metadata':deepcopy(metadata),
                        'decisions':decisions,
                        'assignments':deepcopy(plan.get('assignments', {})),
                        'observed_orders':deepcopy(plan.get('observed_orders', {}))},
        'order_semantics':'KEEP uses explicit no-new-command contract; STOP cancels',
    }
    return result
