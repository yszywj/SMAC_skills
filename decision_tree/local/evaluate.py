"""Evaluate a frozen local policy against BASE on new complete episodes.

This measures deployed-policy performance, including episodes with no patch.
It does not create local counterfactual labels. --dry-run audits inputs without
starting StarCraft or creating an output directory.
"""
import argparse
from copy import deepcopy
import importlib.metadata
import json
import math
from pathlib import Path

from . import study
from .dataset import comparable_observation, episode, read_run
from .models import model_hash, predict, select_candidate
from .runtime import digest, load_policy, validate_options


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def _groups(values, label):
    require(isinstance(values, list) and all(isinstance(v, str) and v for v in values)
            and len(values) == len(set(values)), label + ' must be unique scene IDs')
    return set(values)


def excluded_groups(policy, training_reports=()):
    """Bind optional legacy split reports by exact model hash, never file name."""
    models = {'gain': policy['model']}
    if policy.get('veto_model') is not None:
        models['veto'] = policy['veto_model']
    reports, evidence = {}, []
    for path in training_reports:
        path = Path(path).resolve()
        report = study.read(path)
        matches = [name for name, model in models.items() if report.get('model_hash') == model_hash(model)]
        require(len(matches) == 1, 'Training report does not bind an embedded model: ' + str(path))
        name = matches[0]
        require(report.get('schema_version') == 1 and report.get('kind') == name,
                'Unsupported training report')
        require(name not in reports, 'Duplicate report for ' + name)
        require(report.get('context_hash') == models[name]['context_hash'], 'Training report context mismatch')
        train = _groups(report.get('split', {}).get('train_groups'), 'Report training groups')
        validation = _groups(report.get('split', {}).get('validation_groups'), 'Report validation groups')
        require(train == set(models[name]['training_group_ids']), 'Report fitted groups mismatch')
        require(not train & validation, 'Report train/validation overlap')
        reports[name] = validation
        evidence.append({'path': str(path), 'sha256': digest(report), 'model_kind': name})
    excluded, breakdown = set(), {}
    for name, model in models.items():
        train = set(model['training_group_ids'])
        if 'validation_group_ids' in model:
            validation = _groups(model['validation_group_ids'], name + ' validation groups')
            if name in reports:
                require(validation == reports[name], 'Model/report validation groups mismatch')
        else:
            require(name in reports, name + ' model has no recorded validation scene IDs; '
                    'provide its original hash-bound --training-report')
            validation = reports[name]
        require(not train & validation, 'Model train/validation overlap')
        excluded |= train | validation
        breakdown[name] = {'model_hash': model_hash(model), 'training_groups': sorted(train),
                           'validation_groups': sorted(validation)}
    return excluded, breakdown, evidence


def _absolute_sources(meta):
    return {str((study.ROOT / path).resolve()): value for path, value in meta['source_hashes'].items()}


def _validate_frozen_flags(protocol, manifest, policy_path, output, reference):
    """Resolve the original switches without constructing an environment."""
    from sc2_skill_adapter.bridge.runner import parser
    from sc2_skill_adapter.bridge.skill_runtime import manager_configuration
    from sc2_skill_adapter.bridge.keep_order import validate_keep_heal_order
    flags = protocol.get('flags')
    require(isinstance(flags, list) and all(isinstance(f, str) for f in flags), 'Invalid frozen flags')
    reserved = {'--manifest', '--output-dir', '--local-policy', '--skill-tree', '--sc2-path'}
    require(not any(f.split('=')[0] in reserved for f in flags), 'Frozen flags override evaluation inputs')
    cli = parser()
    options = cli._subparsers._group_actions[0].choices['run']._option_string_actions
    require(not any(f.startswith('--') and f.split('=')[0] not in options for f in flags),
            'Frozen flags require full recognized argument names')
    try:
        args = cli.parse_args(['run', *flags, '--manifest', str(output/'manifest.json'),
                               '--local-policy', str(policy_path), '--output-dir', str(output/'base')])
    except SystemExit as exc:
        raise ValueError('Invalid frozen runner flags') from exc
    validate_options(args, args.method, manifest['env_args'])
    configuration = manager_configuration(args)
    require(args.skill_mode in (None, 'always'), 'adaptive-skill requires always skill mode')
    kwargs = configuration['manager_kwargs']
    parameters = {'posture': args.adaptive_posture, 'max_steps': args.skill_max_steps,
                  **kwargs['combat_params'], 'spatial_params': kwargs.get('spatial_params', {}),
                  **configuration['value_metadata']}
    require(parameters == reference['run.json']['skill_parameters'], 'Frozen flags differ from recorded BASE parameters')
    extensions = validate_keep_heal_order(args, args.method, manifest['env_args'])
    require((extensions or {}) == (reference['run.json'].get('action_extensions') or {}),
            'Frozen action extensions differ from recorded BASE')
    versions = reference['run.json']['versions']
    require(all(importlib.metadata.version(name) == version for name, version in versions.items()),
            'Installed dependency versions differ from frozen BASE')
    require(isinstance(protocol.get('python'), str) and Path(protocol['python']).is_file(),
            'Frozen Python interpreter is missing')


def preflight(study_directory, policy_path, manifest_path, output, training_reports=()):
    from sc2_skill_adapter.bridge.scenarios import validate_manifest
    directory, output = Path(study_directory).resolve(), Path(output).resolve()
    protocol = study.read(directory/'protocol.json')
    require(protocol.get('schema_version') == 1 and protocol.get('kind') == 'local_patch_collection',
            'A frozen local_patch_collection protocol is required')
    original_manifest, manifest = study.read(directory/'manifest.json'), study.read(manifest_path)
    validate_manifest(original_manifest)
    validate_manifest(manifest)
    base_policy, policy = load_policy(directory/'base_policy.json'), load_policy(policy_path)
    require(base_policy['selection']['mode'] == 'base' and policy['selection']['mode'] == 'model',
            'Evaluation requires the original BASE policy and an exported model policy')
    require(digest(original_manifest) == protocol.get('manifest_sha256')
            and digest(base_policy) == protocol.get('policy_sha256'), 'Frozen study input hashes changed')
    require(base_policy['candidates'] == policy['candidates'], 'Candidate enumeration bounds changed')
    env = lambda m: {k: v for k, v in m['env_args'].items() if k not in ('seed', 'replay_dir')}
    require(env(manifest) == env(original_manifest), 'Evaluation environment differs from frozen study')
    sources = protocol.get('sources')
    require(isinstance(sources, dict) and sources, 'Missing frozen source hashes')
    require(study.source_hashes(original_manifest) == sources == study.source_hashes(manifest),
            'Source set or contents differ from frozen collection')
    study.check_sources(sources)
    reference = read_run(directory/'base')
    require(reference['manifest.json'] == original_manifest
            and reference['run.json']['local_policy']['policy'] == base_policy, 'Original BASE input binding changed')
    require(_absolute_sources(reference['run.json']) == sources, 'Original BASE source hashes differ from collection')
    local = reference['run.json']['local_policy']
    for name in ('model', 'veto_model'):
        if policy.get(name) is not None:
            require(policy[name]['context_hash'] == local['context_hash'], name + ' execution context mismatch')
            require(policy[name].get('game_version') == reference['game_version'], name + ' game version mismatch')
    _validate_frozen_flags(protocol, manifest, policy_path, output, reference)
    reports = [Path(p).resolve() for p in training_reports]
    sibling_report = Path(policy_path).resolve().parent/'report.json'
    if sibling_report.is_file() and sibling_report not in reports:
        reports.append(sibling_report)
    excluded, breakdown, report_evidence = excluded_groups(policy, reports)
    # Independent final evaluation also excludes unused development scenes from
    # the supplied collection, not only rows selected by the fitted trees.
    development = {e['sha256'] for e in original_manifest['episodes']}
    requested = {e['sha256'] for e in manifest['episodes']}
    overlap = requested & (excluded | development)
    require(not overlap, 'Evaluation scenes overlap training/validation/development: ' + ', '.join(sorted(overlap)))
    identities = [(e['sha256'], e['sc2_seed']) for e in manifest['episodes']]
    require(len(identities) == len(set(identities)), 'Duplicate scene/seed evaluation episode')
    require(all(type(e['sc2_seed']) is int for e in manifest['episodes']), 'Evaluation requires explicit integer SC2 seeds')
    evaluation_protocol = {
        'schema_version': 1, 'kind': 'local_patch_evaluation',
        'scope': 'complete_deployed_policy_episodes_not_local_counterfactual_labels',
        'collection_directory': str(directory), 'collection_protocol_sha256': digest(protocol),
        'reference_base_hashes': reference['hashes'], 'execution': deepcopy(protocol),
        'context_hash': local['context_hash'], 'game_version': reference['game_version'],
        'manifest_sha256': digest(manifest), 'model_policy_sha256': digest(policy),
        'base_policy_sha256': digest(base_policy), 'excluded_model_groups': breakdown,
        'excluded_collection_groups': sorted(development), 'training_reports': report_evidence,
        'planned_games': 2 * len(manifest['episodes']),
    }
    return {'protocol': evaluation_protocol, 'manifest': manifest, 'base_policy': base_policy,
            'policy': policy, 'reference': reference, 'output': output}


def _audit_selection(arm, policy):
    """Check that recorded deployment decisions implement the frozen selector."""
    decisions = sorted((e for e in arm['events'] if e['event'] == 'local_decision'), key=lambda e: e['step'])
    proposals = interventions = vetoes = 0
    for event in decisions:
        selected = proposed = 'BASE'
        vetoed = False
        if policy['selection']['mode'] == 'model' and proposals == 0:
            proposed = select_candidate(policy['model'], event['candidates'],
                                        policy['selection']['min_gain'])['candidate_id']
            selected = proposed
            if proposed != 'BASE' and policy.get('veto_model') is not None:
                candidate = next(c for c in event['candidates'] if c['id'] == proposed)
                prediction = predict(policy['veto_model'], candidate['family'], candidate['features'])
                vetoed = prediction['supported'] and prediction['value'] > 1e-9
                if vetoed:
                    selected = 'BASE'
        require(event['selected_id'] == selected and event['proposed_id'] == proposed
                and event['vetoed'] is vetoed, 'Recorded decision differs from frozen selector')
        wanted = list(event['baseline_actions'])
        if selected != 'BASE':
            candidate = next(c for c in event['candidates'] if c['id'] == selected)
            require(candidate['duration_steps'] == 1 and len(candidate['actors']) == 1, 'Nonlocal deployed patch')
            for actor, action in candidate['actions'].items():
                wanted[int(actor[1:])] = action
            require(wanted != event['baseline_actions'], 'Deployed patch did not change any action')
        require(wanted == event['final_actions'], 'Deployed action differs from selected patch')
        proposals += proposed != 'BASE'
        interventions += selected != 'BASE'
        vetoes += vetoed
        require(event.get('proposals_used') == proposals and event.get('interventions') == interventions,
                'Recorded intervention budget differs from selector')
    return {'decisions': len(decisions), 'candidate_opportunities': sum(bool(e['candidates']) for e in decisions),
            'proposals': proposals, 'interventions': interventions, 'vetoes': vetoes}


def _validate_run_binding(run, manifest, policy, protocol):
    require(run['manifest.json'] == manifest, 'Evaluation run manifest mismatch')
    local = run['run.json']['local_policy']
    require(local['policy'] == policy, 'Evaluation run policy mismatch')
    require(local['context_hash'] == protocol['context_hash'], 'Evaluation run execution context mismatch')
    require(run['game_version'] == protocol['game_version'], 'Evaluation run game version mismatch')
    require(_absolute_sources(run['run.json']) == protocol['execution']['sources'],
            'Evaluation run source hashes mismatch')


def compare_runs(base, model, manifest, base_policy, policy, protocol):
    """Aggregate every completed matched episode, never filtering by action."""
    for run, expected_policy in ((base, base_policy), (model, policy)):
        _validate_run_binding(run, manifest, expected_policy, protocol)
    rows = []
    for number, scene in enumerate(manifest['episodes'], 1):
        b, m = episode(base, number), episode(model, number)
        for key in ('initial_state_sha256', 'scenario_sha256', 'game_version'):
            require(b['start'].get(key) and b['start'][key] == m['start'].get(key),
                    'Evaluation initial state mismatch: ' + key)
        require(comparable_observation(b['observations'][0]['payload'])
                == comparable_observation(m['observations'][0]['payload']), 'Evaluation initial observation mismatch')
        _audit_selection(b, base_policy)
        counts = _audit_selection(m, policy)
        br, mr = b['record'], m['record']
        require(br['outcome'] in ('win', 'loss', 'timeout', 'draw') and mr['outcome'] in ('win', 'loss', 'timeout', 'draw'),
                'Unrecognized complete-episode outcome')
        require(all(type(v) in (int, float) and math.isfinite(v) for v in (br['reward'], mr['reward'])),
                'Invalid complete-episode return')
        rows.append({'episode': number, 'group_id': scene['sha256'], 'sc2_seed': scene['sc2_seed'],
                     'base_return': br['reward'], 'model_return': mr['reward'],
                     'gain': mr['reward'] - br['reward'], 'base_outcome': br['outcome'],
                     'model_outcome': mr['outcome'], 'base_steps': br['steps'], 'model_steps': mr['steps'], **counts})
    count = len(rows)
    require(count > 0, 'No completed evaluation episodes')
    metrics = {}
    for label in ('base', 'model'):
        metrics[label] = {'mean_return': sum(r[label+'_return'] for r in rows) / count,
                          'wins': sum(r[label+'_outcome'] == 'win' for r in rows),
                          'win_rate': sum(r[label+'_outcome'] == 'win' for r in rows) / count,
                          'outcomes': {name: sum(r[label+'_outcome'] == name for r in rows)
                                       for name in ('win', 'loss', 'timeout', 'draw')}}
    return {'schema_version': 1, 'kind': 'local_patch_deployment_evaluation',
            'evidence_kind': 'full_policy_matched_scenario_evaluation',
            'scope': 'all complete episodes including no opportunity, no positive choice, and veto',
            'episodes': count, 'independent_scene_groups': len({r['group_id'] for r in rows}),
            'completed_games': 2 * count, **metrics,
            'mean_gain': sum(r['gain'] for r in rows) / count,
            'win_rate_gain': metrics['model']['win_rate'] - metrics['base']['win_rate'],
            'improved_episodes': sum(r['gain'] > 1e-9 for r in rows),
            'harmed_episodes': sum(r['gain'] < -1e-9 for r in rows),
            'unchanged_episodes': sum(abs(r['gain']) <= 1e-9 for r in rows),
            'episodes_with_intervention': sum(r['interventions'] > 0 for r in rows),
            'episodes_without_intervention': sum(r['interventions'] == 0 for r in rows),
            'rows': rows, 'provenance': {'BASE': {'path': base['path'], 'hashes': base['hashes']},
                                       'MODEL': {'path': model['path'], 'hashes': model['hashes']}},
            'limitations': ['Whole-policy reruns; these returns are not isolated local counterfactual labels.',
                            'Equal initial observations and seeds do not establish identical hidden simulator state.',
                            'Episode means are descriptive; no statistical significance or generalization is claimed.']}


def evaluate(study_directory, policy_path, manifest_path, output, *, training_reports=(), dry_run=False):
    prepared = preflight(study_directory, policy_path, manifest_path, output, training_reports)
    output = prepared['output']
    require(not output.exists(), 'Evaluation output already exists; refusing to overwrite evidence')
    protocol, manifest = prepared['protocol'], prepared['manifest']
    execution = protocol['execution']
    commands = {arm.upper(): study.command(execution, output/'manifest.json', output/(arm+'_policy.json'), output/arm)
                for arm in ('base', 'model')}
    if dry_run:
        return {'started_games': 0, 'planned_games': protocol['planned_games'], 'commands': commands,
                'protocol': protocol, 'note': 'Input and source audit only; no output created and no StarCraft process started.'}
    output.mkdir(parents=True, exist_ok=False)
    for name, value in (('protocol.json', protocol), ('manifest.json', manifest),
                        ('base_policy.json', prepared['base_policy']), ('model_policy.json', prepared['policy'])):
        study.save_once(output/name, value)
    study.snapshot_sources(execution['sources'], output/'source_snapshot')
    for arm in ('base', 'model'):
        study.run_arm(execution, output/'manifest.json', output/(arm+'_policy.json'), output/arm)
        run = read_run(output/arm)
        _validate_run_binding(run, manifest, prepared['base_policy' if arm == 'base' else 'policy'], protocol)
    study.check_sources(execution['sources'])
    base, model = read_run(output/'base'), read_run(output/'model')
    result = compare_runs(base, model, manifest, prepared['base_policy'], prepared['policy'], protocol)
    result['protocol_sha256'] = digest(protocol)
    study.save_once(output/'result.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', required=True, type=Path, help='original frozen local collection directory')
    parser.add_argument('--policy', required=True, type=Path, help='exported gain or gain+veto policy.json')
    parser.add_argument('--manifest', required=True, type=Path, help='new independent complete-episode manifest')
    parser.add_argument('--output', required=True, type=Path, help='new evaluation directory')
    parser.add_argument('--training-report', action='append', default=[], type=Path,
                        help='optional hash-bound split report; required for legacy models without validation scene IDs')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    result = evaluate(args.study, args.policy, args.manifest, args.output,
                      training_reports=args.training_report, dry_run=args.dry_run)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
