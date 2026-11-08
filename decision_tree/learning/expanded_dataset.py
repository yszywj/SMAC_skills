"""Sparse, isolated-call counterfactual evidence from multiple anchored reruns.

One BASE trajectory may support several comparisons. Every candidate is rerun
independently, changes exactly one invocation, and is audited against the full
observed BASE prefix. Reusing BASE never creates extra independent scenarios.
"""
from copy import deepcopy
from pathlib import Path

from decision_tree.descriptors import MODEL_FEATURE_NAMES, candidate_features
from decision_tree.experiment import digest, execution_context, policy_contract
from decision_tree.learning.dataset import episode, read_run, require

EVIDENCE_KIND = 'matched_observed_prefix_single_call'


def decisions(arm):
    return [event for event in arm['events'] if event['event'] == 'skill_tree_decision']


def comparable_observation(payload):
    """Batch ordinal is administrative; scene/seed identity is checked separately."""
    value = deepcopy(payload)
    value.get('state', {}).pop('episode', None)
    return value


def audit_expanded_pair(base, other, anchor_call_number, candidate_id=None, candidate_rank=None):
    """Audit a single isolated intervention; action-equivalent calls are valid."""
    bd, cd = decisions(base), decisions(other)
    require(all(e['selected_id'] == 'BASE' for e in bd), 'base_has_intervention')
    bg = [e for e in bd if e['call_number'] == anchor_call_number]
    cg = [e for e in cd if e['call_number'] == anchor_call_number]
    require(len(bg) == len(cg) == 1, 'anchor_not_reached')
    b, c = bg[0], cg[0]
    require(b['opportunity'] and c['opportunity'], 'anchor_not_eligible')
    require(sum(e['selected_id'] != 'BASE' for e in cd) == 1, 'not_one_isolated_intervention')
    cid = c['selected_id']
    require(cid != 'BASE' and (candidate_id is None or cid == candidate_id), 'wrong_selected_candidate')
    require(all(e['selected_id'] == 'BASE' for e in cd if e is not c), 'additional_intervention')
    require(all(e['selected_call'] == e['base_call'] for e in bd + cd if e['selected_id'] == 'BASE'),
            'baseline_continuation_changed')
    declared = {row['id']: row['call'] for row in c['candidate_calls']}
    require(len(declared) == len(c['candidate_calls']) and cid in declared, 'undeclared_candidate')
    if candidate_rank is not None:
        require(type(candidate_rank) is int and 0 <= candidate_rank < len(c['candidate_calls']), 'candidate_rank_unavailable')
        require(c['candidate_calls'][candidate_rank]['id'] == cid, 'candidate_rank_mismatch')
    require(c['selected_call'] == declared[cid], 'candidate_call_changed')
    for key in ('step', 'game_loop', 'features', 'base_call', 'candidate_calls', 'base_actions',
                'candidate_actions', 'base_effective_config', 'candidate_effective_configs'):
        require(b[key] == c[key], 'anchor_mismatch:' + key)
    for key in ('initial_state_sha256', 'scenario_sha256', 'game_version'):
        require(base['start'].get(key) and base['start'][key] == other['start'].get(key), 'start_mismatch:' + key)
    require(b.get('invocation_applied') is True and c.get('invocation_applied') is True, 'invocation_not_applied')
    require(b['selected_effective_config'] == b['base_effective_config'], 'base_configuration_mismatch')
    require(c['selected_effective_config'] == c['candidate_effective_configs'][cid], 'applied_configuration_mismatch')
    require(c['selected_effective_config'] != b['base_effective_config'], 'no_configuration_intervention')
    t = b['step']
    prefix = {'observations': [], 'steps': []}
    for step in range(t + 1):
        payload = comparable_observation(base['observations'][step]['payload'])
        require(payload == comparable_observation(other['observations'][step]['payload']), 'observed_prefix_mismatch')
        prefix['observations'].append(payload)
    for step in range(1, t + 1):
        fields = ('actions', 'reward', 'feedback', 'game_loop', 'terminated')
        item = {key: base['steps'][step][key] for key in fields}
        require(item == {key: other['steps'][step][key] for key in fields}, 'executed_prefix_mismatch')
        prefix['steps'].append(item)
    base_action, candidate_action = base['steps'][t + 1]['actions'], other['steps'][t + 1]['actions']
    require(base_action == b['base_actions'] and candidate_action == c['candidate_actions'][cid], 'preview_execution_mismatch')
    ends = [min((e['step'] for e in decisions(arm) if e['step'] > t), default=arm['record']['steps'])
            for arm in (base, other)]
    for arm, end in zip((base, other), ends):
        for step in range(t + 1, end + 1):
            feedback = arm['steps'][step]['feedback']
            require(feedback.get('response_mapping') == 'exact', 'unmapped_command_result')
            require(not feedback.get('unmatched_observation_errors'), 'unmatched_command_error')
            for actor in feedback.get('actors', {}).values():
                require(actor.get('submission') not in ('rejected', 'result_unmapped'), 'command_rejected')
                require(not actor.get('observation_errors'), 'command_observation_error')
    end = min(max(ends), base['record']['steps'], other['record']['steps'])
    changed = [step for step in range(t + 1, end + 1)
               if base['steps'][step]['actions'] != other['steps'][step]['actions']]
    return {'anchor_call_number': anchor_call_number, 'step': t, 'game_loop': b['game_loop'],
            'candidate_id': cid, 'prefix_sha256': digest(prefix), 'invocation_applied': True,
            'configuration_changed': True, 'first_action_changed': base_action != candidate_action,
            'action_changed_during_comparison_window': bool(changed), 'changed_steps': changed,
            'action_comparison_end_step': end, 'base_call_end_step': ends[0],
            'candidate_call_end_step': ends[1],
            'observation_normalization': 'remove_only_administrative_state.episode_batch_ordinal',
            'effect_scope': 'aligned_steps_through_longer_call_capped_at_first_episode_end'}, b, c


def build_expanded_dataset(base_run, arm_runs, plan=None):
    """Read BASE once, stream candidate runs; missing/rejected labels stay absent.

    arm_runs entries: path, base_episode, anchor_call_number, optional arm_episode
    (defaults to 1), anchor_id, candidate_id, candidate_rank. Entire plan is bound
    into provenance, but audit derives labels only from actual executed traces.
    """
    base = read_run(base_run)
    policy = base['run.json']['skill_tree']['policy']
    require(policy['selection']['mode'] == 'base', 'base_mode_required')
    require(policy.get('exploration', {}).get('enabled') is True, 'expanded_protocol_required')
    context = execution_context(base['run.json'])
    rows, rejected, comparisons, provenance, seen = [], [], [], [], set()
    for entry in arm_runs:
        item = deepcopy(entry)
        item['path'] = str(item['path'])
        try:
            arm = read_run(item['path'])
            provenance.append({'path': arm['path'], 'hashes': arm['hashes'], 'comparison': item})
            require(execution_context(arm['run.json']) == context, 'execution_context_mismatch')
            require(arm['game_version'] == base['game_version'], 'game_version_mismatch')
            selection = arm['run.json']['skill_tree']['policy']['selection']
            require(selection['mode'] == 'candidate' and selection['max_interventions_per_episode'] == 1,
                    'isolated_candidate_mode_required')
            require(selection.get('anchor_call_number') == item['anchor_call_number'], 'selection_anchor_mismatch')
            if 'candidate_id' in item:
                require(selection.get('candidate_id') == item['candidate_id'], 'selection_id_mismatch')
            elif 'candidate_rank' in item:
                require(selection.get('candidate_rank') == item['candidate_rank'], 'selection_rank_mismatch')
            n, an = item['base_episode'], item.get('arm_episode', 1)
            bs, cs = base['manifest.json']['episodes'][n - 1], arm['manifest.json']['episodes'][an - 1]
            require(bs['sha256'] == cs['sha256'] and bs['sc2_seed'] == cs['sc2_seed'], 'scenario_or_seed_mismatch')
            b, c = episode(base, n), episode(arm, an)
            compare = {**item, 'group_id': bs['sha256'], 'sc2_seed': bs['sc2_seed'],
                       'base_return': b['record']['reward'], 'candidate_return': c['record']['reward'],
                       'gain': c['record']['reward'] - b['record']['reward'],
                       'base_outcome': b['record']['outcome'], 'candidate_outcome': c['record']['outcome'],
                       'base_steps': b['record']['steps'], 'candidate_steps': c['record']['steps'],
                       'accepted': False}
            comparisons.append(compare)
            audit, bg, cg = audit_expanded_pair(b, c, item['anchor_call_number'],
                                               item.get('candidate_id'), selection.get('candidate_rank'))
            cid = audit['candidate_id']
            key = (bs['sha256'], bs['sc2_seed'], item['anchor_call_number'], cid)
            require(key not in seen, 'duplicate_comparison')
            seen.add(key)
            compare['accepted'] = True
            compare['candidate_id'] = cid
            rows.append({'group_id': bs['sha256'], 'sc2_seed': bs['sc2_seed'],
                         'base_episode': n, 'anchor_id': item.get('anchor_id', str(item['anchor_call_number'])),
                         'anchor_call_number': item['anchor_call_number'], 'candidate_id': cid,
                         'state_features': bg['features'], 'base_call': bg['base_call'],
                         'candidate_call': cg['selected_call'],
                         'features': candidate_features(bg['features'], cg['selected_call'], bg['base_call']),
                         'base_return': compare['base_return'], 'candidate_return': compare['candidate_return'],
                         'gain': compare['gain'], 'base_outcome': compare['base_outcome'],
                         'candidate_outcome': compare['candidate_outcome'],
                         'evidence_kind': EVIDENCE_KIND, 'gate': audit})
        except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
            rejected.append({**item, 'reason': str(exc)})
    return {'schema_version': 2, 'kind': 'skill_candidate_call_pairs', 'game_version': base['game_version'],
            'split': base['manifest.json'].get('split', 'development'), 'context': context,
            'context_hash': digest(context), 'policy_contract': policy_contract(policy),
            'candidate_ids': [c['id'] for c in policy['candidates']], 'feature_names': list(MODEL_FEATURE_NAMES),
            'rows': rows, 'rejected': rejected, 'comparisons': comparisons,
            'protocol': {'intervention_unit': 'one_complete_skill_invocation',
                         'reference_continuation': 'frozen_baseline_controller_after_changed_call',
                         'matching': 'exact_observed_prefix_not_internal_snapshot',
                         'missing_candidate_label': 'absent_never_zero', 'plan': deepcopy(plan or {})},
            'provenance': {'BASE': {'path': base['path'], 'hashes': base['hashes']}, 'arms': provenance}}
