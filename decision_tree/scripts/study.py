"""Prepare, collect and evaluate frozen skill-call studies. No automatic training."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from decision_tree.experiment import digest, policy_contract
from decision_tree.schema import load_policy

# Same current-code switches as the existing Protoss v7 launcher.
DEFAULT_FLAGS = ['--method', 'adaptive-skill', '--skill-version', 'current',
    '--adaptive-posture', 'balanced', '--skill-max-steps', '4',
    '--adaptive-window-fire', '--adaptive-range-preserving-move', '--adaptive-target-scoring',
    '--adaptive-coordinated-approach', '--adaptive-safe-reentry', '--adaptive-accurate-damage',
    '--adaptive-critical-escape', '--adaptive-planned-reentry', '--adaptive-engagement-commitment',
    '--adaptive-approach-progress-guard', '--unit-value-mode', 'contextual', '--unit-value-spatial-context']


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path, value):
    with Path(path).open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def sources():
    from sc2_skill_adapter.bridge.config import SMAC_ROOT
    paths = list((ROOT/'skills/current').glob('*.py')) + list((ROOT/'skills/current/skills').glob('*.py'))
    paths += list((ROOT/'sc2_skill_adapter/bridge').glob('*.py')) + list((ROOT/'decision_tree').glob('*.py'))
    paths += [SMAC_ROOT/'smacv2/env/starcraft2'/name for name in ('starcraft2.py', 'wrapper.py', 'distributions.py')]
    return {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def prepare(manifest_path, policy_path, output, *, split='development', candidates=None, flags=None):
    from sc2_skill_adapter.bridge.scenarios import validate_manifest
    from sc2_skill_adapter.bridge.config import map_file, race_for_map
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    manifest = read(manifest_path)
    validate_manifest(manifest)
    if manifest.get('split') in ('validation', 'test') and split != manifest['split']:
        raise ValueError('Cannot relabel a sealed validation/test manifest')
    manifest['split'] = split
    manifest['sha256'] = digest({k: v for k, v in manifest.items() if k != 'sha256'})
    policy = load_policy(policy_path)
    if candidates:
        selected = set(candidates)
        if not selected <= {c['id'] for c in policy['candidates']}:
            raise ValueError('Unknown candidate ID')
        policy['candidates'] = [c for c in policy['candidates'] if c['id'] in selected]
    policy['selection'] = {'mode': 'base', 'max_interventions_per_episode': 1, 'min_gain': 0.0}
    policy = load_policy(policy)
    if flags is not None and (not isinstance(flags, list) or not all(isinstance(f, str) for f in flags)):
        raise ValueError('flags-json must contain an array of argument strings')
    flags = list(DEFAULT_FLAGS if flags is None else flags)
    if any(f.split('=')[0] in ('--manifest', '--output-dir', '--skill-tree', '--sc2-path') for f in flags):
        raise ValueError('Study owns manifest, output, policy and SC2 path arguments')
    from sc2_skill_adapter.bridge.runner import parser
    from sc2_skill_adapter.bridge.skill_runtime import manager_configuration
    cli = parser()
    run_parser = cli._subparsers._group_actions[0].choices['run']
    if any(f.startswith('--') and f.split('=')[0] not in run_parser._option_string_actions for f in flags):
        raise ValueError('Use complete recognized argument names; abbreviations are not accepted')
    args = cli.parse_args(['run', *flags, '--manifest', str(manifest_path), '--output-dir', str(output/'preflight')])
    if args.method != 'adaptive-skill' or args.skill_version != 'current':
        raise ValueError('Study requires current adaptive-skill')
    if args.pair_mode != 'off' or args.native_teacher_control != 'off' or args.medivac_policy != 'legacy':
        raise ValueError('Study excludes other learned/teacher channels')
    manager_configuration(args)
    if args.skill_mode not in (None, 'always'):
        raise ValueError('adaptive-skill uses skill-mode always')
    from sc2_skill_adapter.bridge.keep_order import validate_keep_heal_order
    validate_keep_heal_order(args, 'adaptive-skill', manifest['env_args'])
    race = race_for_map(manifest['env_args']['map_name'])
    if args.race_profile != 'off' and args.race_profile != race:
        raise ValueError('Race profile does not match manifest')
    if race not in policy['supported_races']:
        raise ValueError('Study race must be explicitly supported by this policy')
    frozen = sources()
    map_path = map_file(manifest['env_args']).resolve()
    frozen[str(map_path)] = hashlib.sha256(map_path.read_bytes()).hexdigest()
    output.mkdir(parents=True, exist_ok=False)
    save(output/'manifest.json', manifest)
    save(output/'base.json', policy)
    for candidate in policy['candidates']:
        arm = deepcopy(policy)
        arm['selection'].update(mode='candidate', candidate_id=candidate['id'])
        save(output/(candidate['id']+'.json'), load_policy(arm))
    protocol = {'schema_version': 1, 'split': split, 'manifest_sha256': manifest['sha256'],
                'policy_contract': policy_contract(policy), 'flags': flags,
                'candidate_ids': [c['id'] for c in policy['candidates']], 'sources': frozen,
                'episodes_per_arm': len(manifest['episodes']),
                'return_definition': 'whole episode cumulative reward; one call intervention then original scheduler',
                'evidence_kind': 'matched_observed_prefix'}
    save(output/'protocol.json', protocol)
    return {'study_dir': str(output), 'episodes_per_arm': len(manifest['episodes']),
            'candidate_ids': protocol['candidate_ids'], 'started_games': 0}


def plan(study, output, *, mode='collect', policy=None, sc2_path=None):
    study, output = Path(study).resolve(), Path(output).resolve()
    protocol, manifest = read(study/'protocol.json'), read(study/'manifest.json')
    if output.exists():
        raise FileExistsError(output)
    from sc2_skill_adapter.bridge.scenarios import validate_manifest
    validate_manifest(manifest)
    if manifest['sha256'] != protocol['manifest_sha256']:
        raise ValueError('Prepared manifest changed')
    for path, value in protocol['sources'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != value:
            raise ValueError('Frozen source changed; prepare a new study: ' + path)
    if mode == 'collect' and protocol['split'] != 'development':
        raise ValueError('Collection requires development scenarios')
    if (mode == 'evaluate') != (policy is not None):
        raise ValueError('Only evaluate requires --policy')
    if mode == 'evaluate':
        model_policy = load_policy(policy)
        if model_policy['selection']['mode'] != 'model':
            raise ValueError('Evaluation requires a fitted model policy')
        game_version = model_policy['selection']['model'].get('game_version')
        if not isinstance(game_version, str) or not game_version.strip():
            raise ValueError('Missing or invalid model game_version provenance')
        training = model_policy['selection']['model'].get('training', {})
        if not training.get('train_groups') or 'validation_groups' not in training:
            raise ValueError('Missing model group provenance')
        used = set(training['train_groups']) | set(training['validation_groups'])
        if used & {s['sha256'] for s in manifest['episodes']}:
            raise ValueError('Evaluation overlaps fitted/tuning scenarios')
        if policy_contract(model_policy) != protocol['policy_contract']:
            raise ValueError('Model candidate/initial-tree contract differs from study')
        from sc2_skill_adapter.bridge.runner import parser
        from decision_tree.experiment import expected_context
        args = parser().parse_args(['run', *protocol['flags'], '--manifest', str(study/'manifest.json'),
                                    '--output-dir', str(output/'model'), '--skill-tree', str(Path(policy).resolve())])
        expected_context(args, manifest, model_policy)
    arms = ['base', *protocol['candidate_ids']] if mode == 'collect' else ['base', 'model']
    commands = {}
    for arm in arms:
        path = Path(policy).resolve() if arm == 'model' else study/(arm+'.json')
        body = load_policy(path)
        if policy_contract(body) != protocol['policy_contract']:
            raise ValueError('Prepared arm changed')
        if arm != 'model':
            expected = 'base' if arm == 'base' else 'candidate'
            if body['selection']['mode'] != expected or (arm != 'base' and body['selection']['candidate_id'] != arm):
                raise ValueError('Prepared arm selection changed')
        call = [sys.executable, str(ROOT/'sc2_skill_adapter/run.py'), 'run', *protocol['flags'],
                '--manifest', str(study/'manifest.json'), '--skill-tree', str(path), '--output-dir', str(output/arm)]
        if sc2_path is not None:
            call += ['--sc2-path', str(Path(sc2_path).resolve())]
        commands[arm] = call
    return protocol, commands


def execute(study, output, *, mode='collect', policy=None, sc2_path=None, dry_run=False):
    protocol, commands = plan(study, output, mode=mode, policy=policy, sc2_path=sc2_path)
    result = {'mode': mode, 'commands': commands,
              'planned_games': protocol['episodes_per_arm']*len(commands), 'started_games': 0}
    if dry_run:
        return result
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    save(output/'workflow.json', result)
    # Share the existing real-game study lock; never launch two studies together.
    lock = ROOT/'sc2_skill_adapter/.medivac-return-study.lock'
    with lock.open('x', encoding='utf-8') as handle:
        handle.write(str(output))
    try:
        for arm, command in commands.items():
            subprocess.run(command, cwd=ROOT, check=True)
        # Preserve executable evidence, rather than relying only on mutable paths.
        snapshot = output/'source_snapshot'
        snapshot.mkdir()
        mapping = {}
        for i, (path, expected) in enumerate(sorted(protocol['sources'].items())):
            raw = Path(path).read_bytes()
            if hashlib.sha256(raw).hexdigest() != expected:
                raise ValueError('Source changed during study: '+path)
            name = f'{i:04d}_{Path(path).name}'
            (snapshot/name).write_bytes(raw)
            mapping[name] = {'original_path': path, 'sha256': expected}
        save(snapshot/'index.json', mapping)
    finally:
        lock.unlink()
    from decision_tree.learning.dataset import build_dataset, read_run, compare_runs
    if mode == 'collect':
        dataset = build_dataset(output/'base', {cid: output/cid for cid in protocol['candidate_ids']})
        dataset['split'] = protocol['split']
        save(output/'paired.json', dataset)
        save(output/'comparison.json', dataset['comparisons'])
        result.update(accepted_rows=len(dataset['rows']), rejected_pairs=len(dataset['rejected']), trained_model=False)
    else:
        comparison = compare_runs(read_run(output/'base'), read_run(output/'model'))
        save(output/'comparison.json', comparison)
        result['evaluation'] = comparison
    result['completed_games'] = result['planned_games']
    result['started_games'] = result['planned_games']
    save(output/'workflow_result.json', result)
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--manifest', required=True)
    prep.add_argument('--policy', required=True)
    prep.add_argument('--output-dir', required=True)
    prep.add_argument('--split', choices=('development', 'validation', 'test'), default='development')
    prep.add_argument('--candidate', action='append')
    prep.add_argument('--flags-json', help='JSON array replacing the default Protoss v7 flags')
    for name in ('collect', 'evaluate'):
        run = sub.add_parser(name)
        run.add_argument('--study-dir', required=True)
        run.add_argument('--output-dir', required=True)
        run.add_argument('--sc2-path')
        run.add_argument('--dry-run', action='store_true')
        if name == 'evaluate':
            run.add_argument('--policy', required=True)
    args = p.parse_args(argv)
    if args.command == 'prepare':
        result = prepare(args.manifest, args.policy, args.output_dir, split=args.split,
                         candidates=args.candidate, flags=read(args.flags_json) if args.flags_json else None)
    else:
        result = execute(args.study_dir, args.output_dir, mode=args.command,
                         policy=getattr(args, 'policy', None), sc2_path=args.sc2_path, dry_run=args.dry_run)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
