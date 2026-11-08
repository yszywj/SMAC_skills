"""Audited one-step local interventions from real independently rerun episodes.

Matching the complete observed prefix is evidence of comparability, not a claim
that StarCraft's hidden state has been cloned. Missing/rejected arms stay absent.
"""
from collections import defaultdict
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

from decision_tree.learning.dataset import episode as _episode, index, require
from skills.local_v1 import FEATURE_NAMES, generate_candidates
from .runtime import digest, execution_context, load_policy


def episode(run, number):
    arm = _episode(run, number)
    arm['candidate_config'] = deepcopy(run['run.json']['local_policy']['policy']['candidates'])
    return arm


def read_run(path):
    path = Path(path)
    run = {'path': str(path.resolve()), 'hashes': {}}
    for name in ('run.json', 'manifest.json', 'summary.json', 'episodes.json', 'events.jsonl'):
        raw = (path / name).read_bytes()
        run['hashes'][name] = hashlib.sha256(raw).hexdigest()
        run[name] = ([json.loads(line) for line in raw.decode('utf-8-sig').splitlines() if line]
                     if name.endswith('jsonl') else json.loads(raw))
    meta, manifest, summary = run['run.json'], run['manifest.json'], run['summary.json']
    require(summary.get('status') == 'complete' and not summary.get('error'), 'incomplete_run')
    require(not any(e['event'] == 'run_failed' for e in run['events.jsonl']), 'failed_event')
    require(manifest['sha256'] == digest({k: v for k, v in manifest.items() if k != 'sha256'}), 'manifest_hash')
    n = len(manifest['episodes'])
    require(n > 0 and len(run['episodes.json']) == n == summary['completed_episodes'] == summary['expected_episodes'], 'episode_count')
    require(meta['manifest_sha256'] == summary['manifest_sha256'] == manifest['sha256'], 'manifest_binding')
    require(meta['env_args'] == manifest['env_args'], 'environment_binding')
    require(meta.get('method') == 'adaptive-skill' and meta.get('skill_version') == 'current', 'wrong_controller')
    require(meta.get('native_teacher_control') == 'off' and meta.get('medivac_policy') == 'legacy'
            and summary.get('pair_mode') == 'off' and not meta.get('skill_tree'), 'other_intervention')
    local = meta['local_policy']
    policy = load_policy(local['policy'])
    require(policy == local['policy'], 'unnormalized_policy')
    require(meta['source_hashes'] and meta['versions'], 'missing_execution_context')
    require(local['context_hash'] == execution_context(meta, policy), 'context_hash_mismatch')
    require(local['feature_names'] == list(FEATURE_NAMES), 'invalid_feature_schema')
    versions = {e.get('game_version') for e in run['events.jsonl'] if e['event'] == 'episode_started'}
    require(len(versions) == 1 and all(isinstance(v, str) and v for v in versions), 'missing_or_mixed_game_version')
    run['game_version'] = next(iter(versions))
    for scene in manifest['episodes']:
        require(scene['sha256'] == digest(scene['episode_config']), 'scene_hash')
    for number in range(1, n + 1):
        audit_episode(episode(run, number))
    return run


def comparable_observation(payload):
    value = deepcopy(payload)
    value.get('state', {}).pop('episode', None)  # Batch ordinal, never game state.
    return value


def audit_episode(arm, candidate_config=None):
    """Recompute proposals from observations and bind each action to its receipt.

    Baseline roles and damage estimates are logged allocator outputs, not
    independently replayed manager state. Public candidate features and their
    deterministic enumeration are recomputed rather than trusted from the log.
    """
    config = candidate_config if candidate_config is not None else arm['candidate_config']
    decisions = index(arm['events'], 'local_decision', 'step')
    require(set(decisions) == set(arm['observations']), 'incomplete_local_decisions')
    for step, event in decisions.items():
        submitted = arm['steps'][step + 1]
        payload = arm['observations'][step]['payload']
        state = payload['state']
        require(event['game_loop'] == submitted['game_loop'] == state['game_loop'], 'decision_clock_mismatch')
        require(event['final_actions'] == submitted['actions'], 'final_actions_not_submitted')
        require(len(event['baseline_actions']) == len(event['final_actions']), 'action_length_mismatch')
        actor_ids = {f'u{i}' for i in range(len(event['final_actions']))}
        require(set(state['allies']) == actor_ids, 'action_actor_coverage')
        baseline = {'actions': event['baseline_actions'], 'roles': event['baseline_roles'],
                    'predicted_damage': event['baseline_predicted_damage']}
        require(event['candidates'] == generate_candidates(payload, baseline, config),
                'candidate_snapshot_mismatch')
        for actor, action in enumerate(event['final_actions']):
            require(type(action) is int, 'invalid_action_type')
            unit = state['allies'][f'u{actor}']
            if not unit['alive']:
                require(action == 0, 'dead_actor_action')
            else:
                menu = payload['questions'][f'u{actor}']['criteria']
                require(str(action) in menu, 'illegal_submitted_action')
        if event['selected_id'] == 'BASE':
            require(event['final_actions'] == event['baseline_actions'], 'baseline_continuation_changed')
        feedback = submitted['feedback']
        require(feedback.get('response_mapping') == 'exact', 'unmapped_command_result')
        require(not feedback.get('unmatched_observation_errors'), 'unmatched_command_error')
        step_mul = state['step_mul']
        require(type(step_mul) is int and step_mul > 0, 'invalid_step_mul')
        require(feedback.get('from_game_loop') == state['game_loop']
                and feedback.get('to_game_loop') == state['game_loop'] + step_mul,
                'feedback_clock_mismatch')
        receipts = feedback.get('actors', {})
        require(isinstance(receipts, dict) and set(receipts) == actor_ids, 'feedback_actor_coverage')
        command_index = 0
        for slot, action in enumerate(event['final_actions']):
            actor = f'u{slot}'
            receipt = receipts[actor]
            require(receipt.get('requested_action_id') == str(action), 'feedback_action_mismatch')
            require(not receipt.get('observation_errors'), 'command_observation_error')
            if state['allies'][actor]['alive']:
                require(receipt.get('submission') == 'accepted', 'command_rejected')
                response = receipt.get('response')
                require(isinstance(receipt.get('command'), dict)
                        and isinstance(response, dict) and type(response.get('code')) is int
                        and response['code'] == 1
                        and type(receipt.get('command_index')) is int
                        and receipt['command_index'] == command_index, 'invalid_command_receipt')
                command_index += 1
            else:
                require(action == 0 and receipt.get('submission') == 'dead_noop'
                        and receipt.get('command') is None and receipt.get('response') is None
                        and 'command_index' not in receipt, 'invalid_dead_noop_receipt')
        raw_results = feedback.get('raw_response_results')
        require(isinstance(raw_results, list) and all(type(code) is int for code in raw_results)
                and raw_results == [1] * command_index, 'invalid_raw_command_results')
    return decisions


def audit_pair(base, other, anchor_step, candidate_id):
    bd, cd = audit_episode(base), audit_episode(other)
    require(type(anchor_step) is int and anchor_step >= 0, 'invalid_anchor')
    require(anchor_step in bd and anchor_step in cd, 'anchor_not_reached')
    require(all(e['selected_id'] == 'BASE' for e in bd.values()), 'base_has_intervention')
    changed = [e for e in cd.values() if e['selected_id'] != 'BASE']
    require(len(changed) == 1 and changed[0]['step'] == anchor_step, 'not_one_isolated_intervention')
    b, c = bd[anchor_step], cd[anchor_step]
    require(c['selected_id'] == c['proposed_id'] == candidate_id and c.get('vetoed') is False, 'wrong_selected_candidate')
    for key in ('initial_state_sha256', 'scenario_sha256', 'game_version'):
        require(base['start'].get(key) and base['start'][key] == other['start'].get(key), 'start_mismatch:' + key)
    for key in ('step', 'game_loop', 'baseline_actions', 'baseline_roles',
                'baseline_predicted_damage', 'candidates'):
        require(b[key] == c[key], 'anchor_mismatch:' + key)
    candidates = {p['id']: p for p in b['candidates']}
    require(len(candidates) == len(b['candidates']) and candidate_id in candidates, 'undeclared_candidate')
    patch = candidates[candidate_id]
    require(patch.get('duration_steps') == 1 and len(patch['actors']) == 1, 'not_local_one_step')
    require(set(patch['actors']) == set(patch['actions']) == set(patch['roles']), 'invalid_patch_ownership')
    wanted = list(b['baseline_actions'])
    for actor, action in patch['actions'].items():
        require(actor.startswith('u') and actor[1:].isdigit(), 'invalid_actor_id')
        slot = int(actor[1:])
        require(0 <= slot < len(wanted), 'invalid_actor_slot')
        wanted[slot] = action
    require(wanted == c['final_actions'], 'patch_execution_mismatch')
    require(wanted != b['baseline_actions'], 'no_actual_action_intervention')
    prefix = {'observations': [], 'steps': []}
    for step in range(anchor_step + 1):
        obs = comparable_observation(base['observations'][step]['payload'])
        require(obs == comparable_observation(other['observations'][step]['payload']), 'observed_prefix_mismatch')
        prefix['observations'].append(obs)
    for step in range(1, anchor_step + 1):
        fields = ('actions', 'reward', 'feedback', 'game_loop', 'terminated')
        record = {key: base['steps'][step][key] for key in fields}
        require(record == {key: other['steps'][step][key] for key in fields}, 'executed_prefix_mismatch')
        prefix['steps'].append(record)
    return {'anchor_step': anchor_step, 'game_loop': b['game_loop'], 'candidate_id': candidate_id,
            'changed_actors': patch['actors'], 'prefix_sha256': digest(prefix),
            'matching': 'exact_observed_prefix_not_internal_snapshot',
            'intervention_unit': 'one_actor_one_decision_step', 'commands_accepted': True}, patch, b


def build_dataset(base_path, arm_entries, plan=None):
    base = read_run(base_path)
    local = base['run.json']['local_policy']
    require(local['policy']['selection']['mode'] == 'base', 'base_mode_required')
    reference_steps = {}
    reference = (plan or {}).get('reference_policy')
    if reference is not None:
        from .models import select_candidate
        reference = load_policy(reference)
        require(reference['selection']['mode'] == 'model'
                and reference['model']['context_hash'] == local['context_hash']
                and reference['candidates'] == local['policy']['candidates'], 'reference_context_mismatch')
        for number in range(1, len(base['manifest.json']['episodes']) + 1):
            decisions = sorted((e for e in episode(base, number)['events'] if e['event'] == 'local_decision'),
                               key=lambda e: e['step'])
            for event in decisions:
                choice = select_candidate(reference['model'], event['candidates'], reference['selection']['min_gain'])
                if choice['candidate_id'] != 'BASE':
                    reference_steps[number] = event['step']
                    break
    rows, rejected, provenance, seen = [], [], [], set()
    expected = {}
    for entry in arm_entries:
        item = {**deepcopy(entry), 'path': str(entry['path'])}
        key = (item['base_episode'], item['anchor_step'], item['candidate_id'])
        require(key not in seen, 'duplicate_comparison')
        seen.add(key)
        try:
            other = read_run(item['path'])
            meta = other['run.json']['local_policy']
            require(meta['context_hash'] == local['context_hash'], 'execution_context_mismatch')
            require(meta['feature_names'] == local['feature_names'], 'feature_schema_mismatch')
            require(other['game_version'] == base['game_version'], 'game_version_mismatch')
            selection = meta['policy']['selection']
            require(selection['mode'] == 'forced' and selection['anchor_step'] == item['anchor_step']
                    and selection['candidate_id'] == item['candidate_id']
                    and selection['max_interventions'] == 1, 'wrong_forced_protocol')
            n, an = item['base_episode'], item.get('arm_episode', 1)
            bs, cs = base['manifest.json']['episodes'][n-1], other['manifest.json']['episodes'][an-1]
            require(bs['sha256'] == cs['sha256'] and bs['sc2_seed'] == cs['sc2_seed'], 'scenario_or_seed_mismatch')
            b, c = episode(base, n), episode(other, an)
            audit, patch, anchor = audit_pair(b, c, item['anchor_step'], item['candidate_id'])
            features = patch['features']
            require(set(features) == set(local['feature_names']), 'candidate_feature_schema')
            require(all(type(v) in (int, float) and math.isfinite(v) for v in features.values()), 'invalid_features')
            anchor_id = f"{bs['sc2_seed']}:{item['anchor_step']}"
            available = [p['id'] for p in anchor['candidates']]
            expected[(bs['sha256'], bs['sc2_seed'], anchor_id)] = set(available)
            rows.append({'group_id': bs['sha256'], 'sc2_seed': bs['sc2_seed'], 'anchor_id': anchor_id,
                         'anchor_step': item['anchor_step'], 'candidate_id': patch['id'],
                         'reference_first_proposal': reference_steps.get(n) == item['anchor_step'],
                         'candidate_family': patch['family'], 'features': deepcopy(features),
                         'base_return': b['record']['reward'], 'candidate_return': c['record']['reward'],
                         'gain': c['record']['reward'] - b['record']['reward'],
                         'available_candidate_ids': available, 'anchor_complete': False,
                         'effect_key': patch['effect_key'], 'aliases': patch.get('aliases', []),
                         'evidence_kind': 'matched_observed_prefix_single_local_action', 'audit': audit})
            provenance.append({'path': other['path'], 'hashes': other['hashes'], 'entry': item})
        except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
            rejected.append({**item, 'reason': str(exc)})
    collected = defaultdict(set)
    for row in rows:
        collected[(row['group_id'], row['sc2_seed'], row['anchor_id'])].add(row['candidate_id'])
    for row in rows:
        key = (row['group_id'], row['sc2_seed'], row['anchor_id'])
        row['anchor_complete'] = collected[key] == expected[key]
    return {'schema_version': 1, 'kind': 'local_patch_dataset', 'audited': True,
            'context_hash': local['context_hash'], 'feature_names': local['feature_names'],
            'policy': deepcopy(local['policy']), 'game_version': base['game_version'],
            'split': base['manifest.json'].get('split', 'development'), 'rows': rows, 'rejected': rejected,
            'protocol': {'matching': 'observed_prefix_not_internal_snapshot',
                         'reward': 'whole_episode_candidate_return_minus_BASE',
                         'continuation': 'same_frozen_baseline_after_one_local_action',
                         'missing_labels': 'absent_never_zero', 'plan': deepcopy(plan or {})},
            'provenance': {'BASE': {'path': base['path'], 'hashes': base['hashes']}, 'arms': provenance}}
