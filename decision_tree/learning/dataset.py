"""Full-call counterfactual labels from audited independently rerun episodes."""
import argparse
import hashlib
import json
import math
from pathlib import Path

from decision_tree.experiment import digest, execution_context, policy_contract


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


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
    require(meta.get('native_teacher_control') == 'off' and meta.get('medivac_policy') == 'legacy'
            and summary.get('pair_mode') == 'off', 'other_intervention')
    require(meta['skill_tree']['policy_contract'] == policy_contract(meta['skill_tree']['policy']), 'policy_binding')
    require(meta['source_hashes'] and meta['versions'], 'missing_execution_context')
    game_versions = {e.get('game_version') for e in run['events.jsonl'] if e['event'] == 'episode_started'}
    require(len(game_versions) == 1 and all(isinstance(v, str) and v for v in game_versions), 'missing_or_mixed_game_version')
    run['game_version'] = next(iter(game_versions))
    for scene in manifest['episodes']:
        require(scene['sha256'] == digest(scene['episode_config']), 'scene_hash')
    return run


def index(events, name, key):
    result = {}
    for event in events:
        if event['event'] == name:
            value = event[key]
            require(value not in result, 'duplicate_' + name)
            result[value] = event
    return result


def episode(run, number):
    events = [e for e in run['events.jsonl'] if e.get('episode') == number]
    starts = [e for e in events if e['event'] == 'episode_started']
    ends = [e for e in events if e['event'] == 'episode_completed']
    records = [e for e in run['episodes.json'] if e['episode'] == number]
    require(len(starts) == len(ends) == len(records) == 1, 'episode_boundaries')
    record = records[0]
    require(record == {k: ends[0][k] for k in record}, 'episode_record_mismatch')
    require(record['initial_state_sha256'] == starts[0]['initial_state_sha256'], 'initial_binding')
    require(record['scenario_sha256'] == starts[0]['scenario_sha256'] == run['manifest.json']['episodes'][number-1]['sha256'], 'scene_binding')
    require(isinstance(record['reward'], (int, float)) and math.isfinite(record['reward']), 'invalid_return')
    steps = index(events, 'step', 'step')
    observations = index(events, 'skill_observation', 'step')
    require(set(steps) == set(range(1, record['steps'] + 1)), 'incomplete_steps')
    require(set(observations) == set(range(record['steps'])), 'incomplete_observations')
    require(steps[record['steps']]['terminated'], 'missing_terminal')
    require(math.isclose(sum(e['reward'] for e in steps.values()), record['reward'], abs_tol=1e-8), 'return_sum')
    return {'start': starts[0], 'record': record, 'events': events, 'steps': steps, 'observations': observations,
            'gates': [e for e in events if e['event'] == 'skill_tree_decision' and e['opportunity']]}


def audit_pair(base, other, candidate_id):
    require(len(base['gates']) == len(other['gates']) == 1, 'not_exactly_one_opportunity')
    b, c = base['gates'][0], other['gates'][0]
    require(b['selected_id'] == 'BASE' and c['selected_id'] == candidate_id, 'wrong_selected_arm')
    require(b['selected_call'] == b['base_call'], 'base_call_changed')
    declared = {row['id']: row['call'] for row in c['candidate_calls']}
    require(c['selected_call'] == declared[candidate_id], 'candidate_call_changed')
    for arm, gate_event in ((base, b), (other, c)):
        require(all(e['selected_id'] == 'BASE' for e in arm['events']
                    if e['event'] == 'skill_tree_decision' and e is not gate_event), 'additional_intervention')
    for key in ('step', 'game_loop', 'features', 'base_call', 'candidate_calls', 'base_actions', 'candidate_actions'):
        require(b[key] == c[key], 'gate_mismatch:' + key)
    for key in ('initial_state_sha256', 'scenario_sha256', 'game_version'):
        require(base['start'].get(key) and base['start'][key] == other['start'].get(key), 'start_mismatch:' + key)
    t = b['step']
    for step in range(t + 1):
        require(base['observations'][step]['payload'] == other['observations'][step]['payload'], 'observed_prefix_mismatch')
    for step in range(1, t + 1):
        for key in ('actions', 'reward', 'feedback', 'game_loop', 'terminated'):
            require(base['steps'][step][key] == other['steps'][step][key], 'executed_prefix_mismatch:' + key)
    base_action = base['steps'][t+1]['actions']
    candidate_action = other['steps'][t+1]['actions']
    require(base_action == b['base_actions'] and candidate_action == c['candidate_actions'][candidate_id], 'preview_execution_mismatch')
    # Any number of actors may change; the intervention is a complete call.
    ends = [min((e['step'] for e in arm['events'] if e['event'] == 'skill_tree_decision' and e['step'] > t),
                default=arm['record']['steps']) for arm in (base, other)]
    comparison_end = min(max(ends), base['record']['steps'], other['record']['steps'])
    require(any(base['steps'][s]['actions'] != other['steps'][s]['actions']
                for s in range(t+1, comparison_end+1)), 'no_actual_intervention')
    for arm in (base, other):
        end = ends[0] if arm is base else ends[1]
        for step in range(t+1, end+1):
            feedback = arm['steps'][step]['feedback']
            require(feedback.get('response_mapping') == 'exact', 'unmapped_command_result')
            require(not feedback.get('unmatched_observation_errors'), 'unmatched_command_error')
            for actor in feedback.get('actors', {}).values():
                require(actor.get('submission') not in ('rejected', 'result_unmapped'), 'command_rejected')
                require(not actor.get('observation_errors'), 'command_observation_error')
    return {'step': t, 'game_loop': b['game_loop'], 'changed_actors': [i for i, (a, z) in enumerate(zip(base_action, candidate_action)) if a != z],
            'base_call_end_step': ends[0], 'candidate_call_end_step': ends[1]}


def compare_runs(base, other):
    require(base['manifest.json'] == other['manifest.json'], 'comparison_manifest_mismatch')
    require(execution_context(base['run.json']) == execution_context(other['run.json']), 'comparison_context_mismatch')
    rows = []
    for i in range(1, len(base['manifest.json']['episodes']) + 1):
        b, c = episode(base, i), episode(other, i)
        require(b['start']['initial_state_sha256'] == c['start']['initial_state_sha256'] and
                b['start']['game_version'] == c['start']['game_version'], 'comparison_initial_state')
        rows.append({'episode': i, 'base_return': b['record']['reward'], 'other_return': c['record']['reward'],
                     'gain': c['record']['reward'] - b['record']['reward'],
                     'base_outcome': b['record']['outcome'], 'other_outcome': c['record']['outcome'],
                     'opportunities': len(c['gates']),
                     'interventions': sum(g['selected_id'] != 'BASE' for g in c['gates'])})
    return {'episodes': len(rows), 'base_wins': sum(r['base_outcome'] == 'win' for r in rows),
            'other_wins': sum(r['other_outcome'] == 'win' for r in rows),
            'mean_gain': sum(r['gain'] for r in rows)/len(rows), 'rows': rows,
            'scope': 'all paired episodes including no opportunity and rejected commands'}


def build_dataset(base_path, candidate_paths):
    from decision_tree.features import FEATURE_NAMES
    base = read_run(base_path)
    policy = base['run.json']['skill_tree']['policy']
    require(policy['selection']['mode'] == 'base', 'base_mode_required')
    require(policy['selection']['max_interventions_per_episode'] == 1, 'single_call_study_required')
    ids = [c['id'] for c in policy['candidates']]
    require(set(candidate_paths) == set(ids), 'missing_candidate_arms')
    arms = {k: read_run(v) for k, v in candidate_paths.items()}
    context = execution_context(base['run.json'])
    comparisons = {}
    for cid, arm in arms.items():
        selection = arm['run.json']['skill_tree']['policy']['selection']
        require(selection['mode'] == 'candidate' and selection['candidate_id'] == cid and selection['max_interventions_per_episode'] == 1, 'candidate_mode_required')
        comparisons[cid] = compare_runs(base, arm)
    rows, rejected = [], []
    for i, scene in enumerate(base['manifest.json']['episodes'], 1):
        try:
            b = episode(base, i)
            audit, returns = {}, {}
            for cid, arm in arms.items():
                c = episode(arm, i)
                audit[cid] = audit_pair(b, c, cid)
                returns[cid] = c['record']['reward']
            rows.append({'group_id': scene['sha256'], 'sc2_seed': scene['sc2_seed'],
                         'features': b['gates'][0]['features'], 'base_return': b['record']['reward'],
                         'candidate_returns': returns,
                         'gains': {k: v-b['record']['reward'] for k, v in returns.items()},
                         'evidence_kind': 'matched_observed_prefix', 'gate': audit})
        except (ValueError, KeyError) as exc:
            rejected.append({'episode': i, 'group_id': scene['sha256'], 'reason': str(exc)})
    return {'schema_version': 1, 'kind': 'skill_call_pairs', 'game_version': base['game_version'],
            'split': base['manifest.json'].get('split', 'development'),
            'context': context, 'context_hash': digest(context),
            'policy_contract': policy_contract(policy), 'candidate_ids': ids,
            'feature_names': sorted(FEATURE_NAMES), 'rows': rows, 'rejected': rejected,
            'comparisons': comparisons, 'provenance': {k: {'path': r['path'], 'hashes': r['hashes']} for k, r in {'BASE': base, **arms}.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True)
    parser.add_argument('--candidate', action='append', required=True, help='ID=RUN_DIRECTORY')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = build_dataset(args.base, dict(v.split('=', 1) for v in args.candidate))
    with Path(args.output).open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)


if __name__ == '__main__':
    main()
