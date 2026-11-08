"""Compare frozen policies on a round's ten predeclared internal validation scenes."""
import argparse
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path

from decision_tree.local import evaluate as deployment
from decision_tree.local import study
from decision_tree.local.dataset import comparable_observation, episode
from decision_tree.local.models import model_hash
from decision_tree.local.runtime import digest, load_policy
from decision_tree.scripts import local_campaign_recover as recovery
from sc2_skill_adapter.bridge.scenarios import validate_manifest


require = deployment.require
EPS = 1e-9


def validate_cohort(manifest, split):
    """Return original 1-based BASE ordinals, preserving the recorded order and seeds."""
    require(split.get('kind') == 'local_scene_split' and split.get('schema_version') == 1,
            'Expected predeclared local_scene_split')
    scenes = manifest['episodes']
    ids = [scene['sha256'] for scene in scenes]
    seeds = [scene['sc2_seed'] for scene in scenes]
    train = deployment._groups(split['training_group_ids'], 'training scene groups')
    holdout = deployment._groups(split['validation_group_ids'], 'validation scene groups')
    require(len(ids) == len(set(ids)) == 40 and len(train) == 30 and len(holdout) == 10,
            'Expected 40 independent scenes with 30 training and 10 validation groups')
    require(not train & holdout and train | holdout == set(ids), 'Split must partition the recorded cohort')
    require(len(set(seeds)) == 40 and all(type(seed) is int for seed in seeds), 'Invalid scenario seeds')
    return [(number, scene) for number, scene in enumerate(scenes, 1) if scene['sha256'] in holdout]


def single_manifest(manifest, scene):
    single = {**manifest, 'episodes': [{**deepcopy(scene), 'index': 0}]}
    single['sha256'] = digest({key: value for key, value in single.items() if key != 'sha256'})
    return single


def validate_model_policy(policy, policy_path, reference, holdout):
    base_local = reference['run.json']['local_policy']
    require(policy['selection']['mode'] == 'model', 'Expected a deployed model policy')
    require(policy['candidates'] == base_local['policy']['candidates'], 'Changed candidate configuration')
    for name in ('model', 'veto_model'):
        if policy.get(name) is None:
            continue
        model = policy[name]
        require(model['context_hash'] == base_local['context_hash'], 'Model context mismatch')
        require(model.get('game_version') == reference['game_version'], 'Model game version mismatch')
        require(not set(model['training_group_ids']) & holdout, 'Validation scene was used in model training')
    # Bind the frozen authored structure before running any validation episode.
    folder = Path(policy_path).resolve().parent
    report = study.read(folder/'report.json')
    structure = study.read(folder/'structure.json')
    model = policy.get('veto_model') or policy['model']
    require(report['construction'] == 'LLM_supplied_predicates_and_topology'
            and study.read(folder/'model.json') == model
            and report['model_hash'] == model_hash(model)
            and report['structure_sha256'] == digest(structure)
            and report['structure'] == structure, 'Authored model/structure/report mismatch')
    return {'policy_sha256': digest(policy), 'structure_sha256': digest(structure),
            'model_hash': model_hash(model), 'report_sha256': digest(report)}


def audited_attempts(protocol, manifest_path, policy_path, stem):
    """Resume past environmental failures with at most three new attempts per invocation."""
    manifest, policy = study.read(manifest_path), load_policy(policy_path)
    failures = []
    number, launches = 0, 0
    while True:
        number += 1
        path = stem.parent / (stem.name + f'.attempt-{number:02d}')
        if not path.exists():
            if launches == 3:
                raise RuntimeError('Environment retry limit reached: ' + str(failures))
            # Interrupted launch sidecars reserve the name. Starting another process while a
            # previous one might still be alive needs explicit resolution, not a blind retry.
            require(not any(path.with_name(path.name + suffix).exists()
                            for suffix in ('.console.log', '.command.json', '.process.json')),
                    'Unresolved prior launch: ' + str(path))
            recovery.run_attempt(protocol, manifest_path, policy_path, path, timeout_seconds=120)
            launches += 1
        try:
            return recovery.successful(path, manifest, policy)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            kind = recovery.failure_kind(path)
            failures.append({'path': str(path), 'kind': kind, 'error': str(exc)})
            if kind not in ('environment_transport', 'watchdog_timeout'):
                raise ValueError('Invalid validation attempt: ' + str(failures[-1])) from exc
            if kind == 'watchdog_timeout':
                require(study.read(path/'timeout.json').get('termination') == 'owned process tree terminated',
                        'Previous timed-out validation process needs manual resolution: ' + str(path))


def match_episode(run, original, policy, protocol):
    """Check the whole-policy trajectory and its common initial observation with BASE."""
    require(run['game_version'] == protocol['game_version']
            and run['run.json']['local_policy']['context_hash'] == protocol['context_hash']
            and deployment._absolute_sources(run['run.json']) == protocol['execution']['sources'],
            'Validation run differs from frozen execution context')
    arm = episode(run, 1)
    for key in ('scenario_sha256', 'initial_state_sha256', 'game_version'):
        require(arm['start'].get(key) and arm['start'][key] == original['start'].get(key),
                'Initial episode mismatch: ' + key)
    require(comparable_observation(arm['observations'][0]['payload'])
            == comparable_observation(original['observations'][0]['payload']), 'Initial observation mismatch')
    return episode_result(arm, policy, run, 1)


def episode_result(arm, policy, run, ordinal):
    record = arm['record']
    require(type(record['reward']) in (float, int) and math.isfinite(record['reward']), 'Invalid episode return')
    require(record['outcome'] in ('win', 'loss', 'draw', 'timeout'), 'Unknown complete-episode outcome')
    return {'return': record['reward'], 'outcome': record['outcome'], 'steps': record['steps'],
            **deployment._audit_selection(arm, policy), 'run': run['path'], 'ordinal': ordinal,
            'hashes': run['hashes']}


def summarize(rows):
    require(len(rows) == 10, 'Adoption requires all ten predeclared validation scenes')
    metrics = {}
    for label in ('incumbent', 'candidate'):
        values = [row[label] for row in rows]
        metrics[label] = {'mean_return': math.fsum(v['return'] for v in values)/len(values),
                          'win_rate': sum(v['outcome'] == 'win' for v in values)/len(values),
                          'outcomes': {name: sum(v['outcome'] == name for v in values)
                                       for name in ('win', 'loss', 'timeout', 'draw')}}
    gains = [row['gain'] for row in rows]
    mean_gain = math.fsum(gains)/len(gains)
    old, new = metrics['incumbent']['outcomes'], metrics['candidate']['outcomes']
    checks = {'mean_return_improved': mean_gain > EPS, 'wins_not_reduced': new['win'] >= old['win'],
              'losses_not_increased': new['loss'] <= old['loss']}
    return {**metrics, 'mean_gain': mean_gain, 'positive': sum(g > EPS for g in gains),
            'zero': sum(abs(g) <= EPS for g in gains), 'negative': sum(g < -EPS for g in gains),
            'adoption_checks': checks, 'decision': 'adopt_candidate' if all(checks.values()) else 'retain_incumbent'}


@contextmanager
def validation_lock(output):
    """OS-released lock so an interrupted validator can restart without stale lock deletion."""
    with (output/'.validation.lock').open('a+b') as handle:
        if handle.tell() == 0:
            handle.write(b'0')
            handle.flush()
        handle.seek(0)
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def validate(study_directory, split_path, candidate_policy, output, incumbent_policy=None, dry_run=False):
    directory, output = Path(study_directory).resolve(), Path(output).resolve()
    require(output != directory and directory not in output.parents, 'Use a separate validation output directory')
    execution = study.read(directory/'protocol.json')
    manifest = study.read(directory/'manifest.json')
    split = study.read(split_path)
    validate_manifest(manifest)
    require(split == study.read(directory.parent/'split.json'), 'Split differs from the predeclared round split')
    env = manifest['env_args']
    cap = env['capability_config']
    require(env['map_name'] == '10gen_protoss' and env.get('unit_balance') == 'native'
            and cap['n_units'] == cap['n_enemies'] == 10, 'Expected frozen native Protoss 10v10 scenes')
    cohort = validate_cohort(manifest, split)
    base = load_policy(directory/'base_policy.json')
    require(execution.get('schema_version') == 1 and execution.get('kind') == 'local_patch_collection'
            and execution['manifest_sha256'] == digest(manifest)
            and execution['policy_sha256'] == digest(base) and base['selection']['mode'] == 'base',
            'Frozen study or BASE policy changed')
    require(study.source_hashes(manifest) == execution['sources'], 'Changed frozen sources')
    reference = recovery.successful(directory/'base', manifest, base)
    require(deployment._absolute_sources(reference['run.json']) == execution['sources'], 'Changed BASE source binding')
    deployment._validate_frozen_flags(execution, manifest, candidate_policy, output, reference)
    candidate = load_policy(candidate_policy)
    incumbent = load_policy(incumbent_policy) if incumbent_policy else base
    holdout = set(split['validation_group_ids'])
    binding = {'candidate': validate_model_policy(candidate, candidate_policy, reference, holdout)}
    if incumbent['selection']['mode'] == 'base':
        require(incumbent == base, 'Expected original BASE policy')
        binding['incumbent'] = {'policy_sha256': digest(base), 'mode': 'reuse_original_BASE'}
    else:
        binding['incumbent'] = validate_model_policy(incumbent, incumbent_policy, reference, holdout)
    protocol = {'schema_version': 1, 'kind': 'local_campaign_internal_validation',
                'scope': 'predeclared_internal_holdout_for_version_selection',
                'study': str(directory), 'collection_protocol_sha256': digest(execution), 'execution': execution,
                'base_run_hashes': reference['hashes'], 'context_hash': reference['run.json']['local_policy']['context_hash'],
                'game_version': reference['game_version'], 'manifest_sha256': digest(manifest), 'split': split,
                'heldout_episodes': [{'ordinal': n, 'group_id': scene['sha256'], 'sc2_seed': scene['sc2_seed']}
                                     for n, scene in cohort],
                'bindings': binding, 'timeout_seconds': 120, 'new_attempts_per_scene_per_invocation': 3,
                'tolerance': EPS,
                'validator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'launcher_sha256': hashlib.sha256(Path(recovery.__file__).read_bytes()).hexdigest()}
    if dry_run:
        return {'started_games': 0, 'planned_model_games': len(cohort) * (1 + int(incumbent != base)),
                'protocol': protocol}
    if output.exists() and any(output.iterdir()) and not (output/'protocol.json').is_file():
        raise ValueError('Refusing to use a nonempty directory without a validation protocol')
    output.mkdir(parents=True, exist_ok=True)
    with validation_lock(output):
        study.save_once(output/'protocol.json', protocol)
        study.save_once(output/'candidate_policy.json', candidate)
        study.save_once(output/'incumbent_policy.json', incumbent)
        rows = []
        for number, scene in cohort:
            manifest_path = output/f'scene_{number:03d}.json'
            study.save_once(manifest_path, single_manifest(manifest, scene))
            original = episode(reference, number)
            row = {'base_episode': number, 'group_id': scene['sha256'], 'sc2_seed': scene['sc2_seed'],
                   'base': episode_result(original, base, reference, number)}
            for label, policy in (('incumbent', incumbent), ('candidate', candidate)):
                if policy == base:
                    row[label] = row['base']
                else:
                    run = audited_attempts(execution, manifest_path, output/(label+'_policy.json'),
                                           output/f'scene_{number:03d}_{label}')
                    row[label] = match_episode(run, original, policy, protocol)
            row['gain'] = row['candidate']['return'] - row['incumbent']['return']
            study.save_once(output/f'comparison_{number:03d}.json', row)
            rows.append(row)
            print(f'Validated {len(rows)}/{len(cohort)} complete scene comparisons', flush=True)
        study.check_sources(execution['sources'])
        result = {'schema_version': 1, 'kind': 'internal_validation', 'scope': protocol['scope'],
                  'protocol_sha256': digest(protocol), 'rows': rows, **summarize(rows)}
        study.save_once(output/'result.json', result)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', required=True, type=Path)
    parser.add_argument('--split', required=True, type=Path)
    parser.add_argument('--candidate-policy', required=True, type=Path)
    parser.add_argument('--incumbent-policy', type=Path, help='default: original BASE')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    result = validate(args.study, args.split, args.candidate_policy, args.output, args.incumbent_policy, args.dry_run)
    print(json.dumps(result if args.dry_run else {k: v for k, v in result.items() if k != 'rows'},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
