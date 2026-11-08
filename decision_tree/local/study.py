"""Freeze and collect single-actor, single-step paired SMAC reruns.

The dry run starts no game. Actual collection first records BASE, chooses
outcome-blind anchors, saves the complete candidate plan, then enforces budget.
Completed runs can be resumed; incomplete runs are never overwritten or silently
relabelled as losses. No model fitting is performed by collection.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from .dataset import read_run, build_dataset, episode
from .runtime import digest, load_policy, validate_options
from decision_tree.scripts.study import DEFAULT_FLAGS

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save_once(path, value):
    path = Path(path)
    if path.exists():
        if read(path) != value:
            raise ValueError('Refusing to overwrite changed artifact: ' + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def source_hashes(manifest):
    from sc2_skill_adapter.bridge.config import SMAC_ROOT, map_file
    paths = []
    for folder in ('skills/current', 'skills/current/skills', 'skills/local_v1',
                   'decision_tree/local', 'sc2_skill_adapter/bridge'):
        paths.extend((ROOT / folder).glob('*.py'))
    paths += [SMAC_ROOT/'smacv2/env/starcraft2'/name for name in
              ('starcraft2.py', 'wrapper.py', 'distributions.py')]
    paths.append(map_file(manifest['env_args']))
    return {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def check_sources(sources):
    for path, expected in sources.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise ValueError('Source changed during frozen study: ' + path)


def snapshot_sources(sources, directory):
    directory.mkdir(exist_ok=True)
    entries = {}
    for i, (path, expected) in enumerate(sorted(sources.items())):
        name = f'{i:04d}_{Path(path).name}'
        destination = directory / name
        if destination.exists():
            if hashlib.sha256(destination.read_bytes()).hexdigest() != expected:
                raise ValueError('Source snapshot changed: ' + name)
        else:
            destination.write_bytes(Path(path).read_bytes())
        entries[name] = {'original_path': path, 'sha256': expected}
    save_once(directory/'index.json', entries)


def select_anchors(events, count=2, min_gap=4, rule='earliest', required_step=None):
    if type(count) is not int or not 1 <= count <= 3:
        raise ValueError('anchors must be 1..3')
    if type(min_gap) is not int or min_gap < 1:
        raise ValueError('min_gap must be positive')
    if rule not in ('earliest', 'family_coverage'):
        raise ValueError('Unknown anchor rule')
    eligible = sorted((e for e in events if e['event'] == 'local_decision' and e['candidates']),
                      key=lambda e: e['step'])
    selected = [e for e in eligible if e['step'] == required_step] if required_step is not None else []
    if required_step is not None and len(selected) != 1:
        raise ValueError('Required reference opportunity is missing')
    if rule == 'family_coverage':
        covered = {c['family'] for e in selected for c in e['candidates']}
        while len(selected) < count:
            choices = [e for e in eligible if all(abs(e['step'] - s['step']) >= min_gap for s in selected)]
            if not choices:
                break
            def rank(event):
                families = {c['family'] for c in event['candidates']}
                return (-len(families - covered), -len(families), event['step'])
            chosen = min(choices, key=rank)
            selected.append(chosen)
            covered.update(c['family'] for c in chosen['candidates'])
        return sorted(selected, key=lambda event: event['step'])
    for event in eligible:
        if len(selected) == count:
            break
        if all(abs(event['step'] - s['step']) >= min_gap for s in selected):
            selected.append(event)
            if len(selected) == count:
                break
    return sorted(selected, key=lambda event: event['step'])


def make_plan(base_path, output, policy, anchors=2, min_gap=4, anchor_rule='earliest', reference_policy=None):
    base = read_run(base_path)
    if base['run.json']['local_policy']['policy'] != policy:
        raise ValueError('BASE policy changed')
    if reference_policy is not None:
        reference_policy = load_policy(reference_policy)
        if (reference_policy['selection']['mode'] != 'model'
                or reference_policy['candidates'] != policy['candidates']
                or reference_policy['model']['context_hash'] != base['run.json']['local_policy']['context_hash']):
            raise ValueError('Reference policy must match the frozen collection context')
    arms, anchor_records, scenes = [], [], []
    manifest = base['manifest.json']
    for number, scene in enumerate(manifest['episodes'], 1):
        events = episode(base, number)['events']
        reference_step = None
        if reference_policy is not None:
            from .models import select_candidate
            for event in sorted((e for e in events if e['event'] == 'local_decision'), key=lambda e: e['step']):
                choice = select_candidate(reference_policy['model'], event['candidates'],
                                          reference_policy['selection']['min_gain'])
                if choice['candidate_id'] != 'BASE':
                    reference_step = event['step']
                    break
        selected = select_anchors(events, anchors, min_gap, anchor_rule, reference_step)
        scenes.append({'base_episode': number, 'group_id': scene['sha256'],
                       'sc2_seed': scene['sc2_seed'], 'anchors': len(selected)})
        for gate in selected:
            available = gate['candidates']
            ids = [c['id'] for c in available]
            if len(set(ids)) != len(ids):
                raise ValueError('Duplicate canonical candidates')
            anchor_records.append({'base_episode': number, 'anchor_step': gate['step'],
                                   'reference_first_proposal': gate['step'] == reference_step,
                                   'available_candidate_ids': ids})
            for rank, candidate in enumerate(available):
                # Names use indices, never candidate text as an OS path.
                name = f'ep{number:04d}_s{gate["step"]:05d}_c{rank:03d}'
                single = deepcopy(manifest)
                single['episodes'] = [deepcopy(scene)]
                single['episodes'][0]['index'] = 0
                single['sha256'] = digest({k: v for k, v in single.items() if k != 'sha256'})
                arm_policy = deepcopy(policy)
                arm_policy['selection'] = {'mode': 'forced', 'candidate_id': candidate['id'],
                                           'anchor_step': gate['step'], 'max_interventions': 1, 'min_gain': 0.0}
                arm_policy = load_policy(arm_policy)
                mp, pp = output/'inputs'/(name+'_manifest.json'), output/'inputs'/(name+'_policy.json')
                save_once(mp, single); save_once(pp, arm_policy)
                arms.append({'name': name, 'path': str((output/name).resolve()),
                             'base_episode': number, 'arm_episode': 1, 'anchor_step': gate['step'],
                             'reference_first_proposal': gate['step'] == reference_step,
                             'candidate_id': candidate['id'], 'candidate_family': candidate['family'],
                             'manifest': str(mp.resolve()), 'policy': str(pp.resolve()),
                             'manifest_sha256': digest(single), 'policy_sha256': digest(arm_policy)})
    return {'schema_version': 1, 'kind': 'local_patch_study_plan',
            'base_path': str(Path(base_path).resolve()), 'base_hashes': base['hashes'],
            'reference_policy': reference_policy,
            'anchor_rule': {'count': anchors, 'minimum_step_gap': min_gap,
                            'name': anchor_rule,
                            'selection': ('greedy new-family coverage, total families, then earliest; outcomes unused'
                                          if anchor_rule == 'family_coverage' else
                                          'first eligible then earliest sufficiently spaced; outcomes unused')},
            'scenes': scenes, 'anchors': anchor_records, 'arms': arms,
            'candidate_games': len(arms), 'total_games': len(scenes) + len(arms)}


def command(protocol, manifest, policy, output):
    cmd = [protocol['python'], '-B', str(ROOT/'sc2_skill_adapter/run.py'), 'run',
           *protocol['flags'], '--manifest', str(manifest), '--local-policy', str(policy),
           '--output-dir', str(output)]
    if protocol.get('sc2_path'):
        cmd += ['--sc2-path', protocol['sc2_path']]
    return cmd


def _retryable_start_failure(output):
    """Never retry a run once it has emitted any episode-start evidence."""
    try:
        summary = read(output/'summary.json')
        events = [json.loads(line) for line in (output/'events.jsonl').read_text(
            encoding='utf-8-sig').splitlines() if line]
        error = str(summary.get('error', ''))
        return (summary.get('status') == 'failed'
                and type(summary.get('completed_episodes')) is int
                and summary['completed_episodes'] == 0
                and not any(e.get('event') in ('episode_started', 'skill_observation',
                                               'policy_observation', 'step', 'episode_completed')
                            for e in events)
                and error.startswith(('ConnectError:', 'ConnectionError:',
                                      'ConnectionRefusedError:', 'ConnectionResetError:',
                                      'ConnectionAbortedError:')))
    except (OSError, ValueError, TypeError, AttributeError):
        return False


def _archive_start_failure(output, archive_root=None):
    """Keep the run and sibling command/console evidence inside one checked root.

    archive_root is explicit for isolated tests; normal execution is confined
    to this repository's decision_tree/artifacts directory. No files are deleted.
    """
    root = Path(archive_root or ROOT/'decision_tree/artifacts').resolve()
    output = Path(output).absolute()
    parent = output.parent.resolve()
    if not parent.is_relative_to(root) or output.resolve().parent != parent:
        raise ValueError('Failed-start archive is outside its allowed artifact directory')
    for attempt in (1, 2):
        archived = output.with_name(output.name + f'.failed-start-{attempt:03d}')
        pairs = [(output, archived)] + [
            (parent/(output.name+suffix), parent/(archived.name+suffix))
            for suffix in ('.console.log', '.command.json')]
        if any(target.exists() or target.is_symlink() for _, target in pairs):
            continue
        for source, target in pairs:
            if (not source.resolve().is_relative_to(root)
                    or not target.resolve().is_relative_to(root)
                    or source.resolve().parent != parent or target.resolve().parent != parent):
                raise ValueError('Failed-start archive path escaped its artifact directory')
        for source, target in pairs:
            if source.exists():
                source.rename(target)
        return archived
    raise ValueError('SC2 startup retry limit reached; both failed attempts are preserved')


def run_arm(protocol, manifest, policy, output, *, archive_root=None):
    output = Path(output)
    while True:
        check_sources(protocol['sources'])
        try:
            if output.exists():
                existing = read_run(output)
                if (existing['manifest.json'] != read(manifest)
                        or existing['run.json']['local_policy']['policy'] != load_policy(policy)):
                    raise ValueError('Cannot resume a different run: ' + str(output))
                return
            cmd = command(protocol, manifest, policy, output)
            save_once(output.parent/(output.name+'.command.json'), cmd)
            log = output.parent/(output.name+'.console.log')
            with log.open('x', encoding='utf-8') as stream:
                subprocess.run(cmd, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                               env={**os.environ, 'PYTHONUNBUFFERED': '1'}, check=True)
            read_run(output)  # A zero process exit alone is insufficient.
            return
        except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError):
            if not _retryable_start_failure(output):
                raise
            if (read(output/'manifest.json') != read(manifest)
                    or read(output/'run.json').get('local_policy', {}).get('policy') != load_policy(policy)):
                raise ValueError('Cannot retry a failed run with different frozen inputs')
            archived = _archive_start_failure(output, archive_root)
            print(f'SC2 startup failed before episode start; evidence saved: {archived.name}; retrying', flush=True)


def run_candidates(protocol, arms, output, workers=1):
    """Finish other declared arms after a failure; absent rewards remain absent."""
    if type(workers) is not int or workers not in (1, 2):
        raise ValueError('workers must be 1 or 2')
    output = Path(output)
    errors_path = output/'candidate_errors.json'
    history = read(errors_path) if errors_path.exists() else {'schema_version': 1, 'errors': []}
    errors, succeeded = [], 0

    def execute(arm):
        if (digest(read(arm['manifest'])) != arm['manifest_sha256']
                or digest(load_policy(arm['policy'])) != arm['policy_sha256']):
            raise ValueError('Planned input changed')
        run_arm(protocol, arm['manifest'], arm['policy'], arm['path'])

    with ThreadPoolExecutor(max_workers=workers) as executor:
        pending = {executor.submit(execute, arm): arm for arm in arms}
        for completed, future in enumerate(as_completed(pending), 1):
            arm = pending[future]
            try:
                future.result()
                succeeded += 1
                status = 'complete'
            except Exception as exc:
                item = {'name': arm['name'], 'path': arm['path'],
                        'candidate_id': arm['candidate_id'], 'error': type(exc).__name__ + ': ' + str(exc)}
                errors.append(item); history['errors'].append(item)
                temporary = errors_path.with_suffix('.json.tmp')
                temporary.write_text(json.dumps(history, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
                temporary.replace(errors_path)
                status = 'failed (preserved; no zero label)'
            print(f'Local candidate {completed}/{len(arms)} {status}: {arm["name"]}', flush=True)
    return succeeded, errors


def collect(manifest_path, policy_path, output, *, max_candidate_games, flags=None,
            anchors=2, min_gap=4, python=None, sc2_path=None, dry_run=False, plan_only=False,
            anchor_rule='earliest', workers=1, reference_policy=None):
    if type(max_candidate_games) is not int or max_candidate_games < 0:
        raise ValueError('max_candidate_games must be a nonnegative integer')
    if type(workers) is not int or workers not in (1, 2):
        raise ValueError('workers must be 1 or 2')
    select_anchors([], anchors, min_gap, anchor_rule)
    manifest, policy = read(manifest_path), load_policy(policy_path)
    reference = load_policy(reference_policy) if reference_policy is not None else None
    from sc2_skill_adapter.bridge.scenarios import validate_manifest
    from sc2_skill_adapter.bridge.runner import parser
    from sc2_skill_adapter.bridge.skill_runtime import manager_configuration
    validate_manifest(manifest)
    if policy['selection']['mode'] != 'base':
        raise ValueError('Collection input policy must select BASE')
    flags = list(DEFAULT_FLAGS if flags is None else flags)
    if not all(isinstance(f, str) for f in flags):
        raise ValueError('flags-json must contain argument strings')
    reserved = {'--manifest', '--output-dir', '--local-policy', '--skill-tree', '--sc2-path'}
    if any(f.split('=')[0] in reserved for f in flags):
        raise ValueError('Collector owns manifest/output/policy/SC2 arguments')
    cli = parser()
    opts = cli._subparsers._group_actions[0].choices['run']._option_string_actions
    if any(f.startswith('--') and f.split('=')[0] not in opts for f in flags):
        raise ValueError('Use full recognized flags; abbreviations are not accepted')
    output = Path(output).resolve()
    args = cli.parse_args(['run', *flags, '--manifest', str(manifest_path),
                           '--local-policy', str(policy_path), '--output-dir', str(output/'preflight')])
    validate_options(args, args.method, manifest['env_args'])
    manager_configuration(args)
    interpreter = Path(python or ROOT/'sc2_skill_adapter/.venv/Scripts/python.exe').resolve()
    if not interpreter.is_file():
        raise ValueError('Python interpreter not found: ' + str(interpreter))
    protocol = {'schema_version': 1, 'kind': 'local_patch_collection',
                'manifest_sha256': digest(manifest), 'policy_sha256': digest(policy),
                'flags': flags, 'python': str(interpreter),
                'sc2_path': str(Path(sc2_path).resolve()) if sc2_path else None,
                'anchors': anchors, 'min_gap': min_gap, 'anchor_rule': anchor_rule,
                'reference_policy': reference,
                'sources': source_hashes(manifest), 'workers': workers}
    base_cmd = command(protocol, output/'manifest.json', output/'base_policy.json', output/'base')
    if dry_run:
        return {'started_games': 0, 'base_games': len(manifest['episodes']),
                'candidate_game_upper_bound': len(manifest['episodes'])*anchors*policy['candidates']['max_candidates'],
                'max_candidate_games': max_candidate_games, 'base_command': base_cmd, 'workers': workers,
                'note': 'Exact candidate budget is checked after BASE, before any candidate run.'}
    output.mkdir(parents=True, exist_ok=True)
    lock = output/'.collect.lock'
    with lock.open('x', encoding='utf-8') as f:
        f.write(str(os.getpid()))
    try:
        save_once(output/'protocol.json', protocol)
        save_once(output/'manifest.json', manifest)
        save_once(output/'base_policy.json', policy)
        check_sources(protocol['sources'])
        snapshot_sources(protocol['sources'], output/'source_snapshot')
        print(f'Running BASE: {len(manifest["episodes"])} scenarios', flush=True)
        run_arm(protocol, output/'manifest.json', output/'base_policy.json', output/'base')
        plan = make_plan(output/'base', output, policy, anchors, min_gap, anchor_rule, reference)
        save_once(output/'plan.json', plan)
        print(f'BASE complete; {plan["candidate_games"]} candidate games planned; workers={workers}', flush=True)
        if plan['candidate_games'] > max_candidate_games:
            raise ValueError(f"Plan requires {plan['candidate_games']} candidate games; budget is {max_candidate_games}. "
                             'Plan saved; no candidate was started. Resume with a sufficient explicit budget.')
        if plan_only:
            return {'started_candidate_games': 0, 'plan': str(output/'plan.json'), **plan}
        successful, errors = run_candidates(protocol, plan['arms'], output, workers)
        check_sources(protocol['sources'])
        data = build_dataset(output/'base', plan['arms'], plan)
        save_once(output/'paired.json', data)
        completed = len(manifest['episodes'])
        for arm in plan['arms']:
            summary_path = Path(arm['path'])/'summary.json'
            if summary_path.is_file():
                try:
                    count = read(summary_path).get('completed_episodes', 0)
                    if type(count) is int and count >= 0:
                        completed += count
                except (OSError, ValueError, AttributeError):
                    pass  # The dataset preserves this malformed/missing arm as rejected.
        result = {'completed_games': completed, 'planned_games': plan['total_games'],
                  'successful_candidate_runs': successful, 'candidate_errors': len(errors),
                  'accepted_rows': len(data['rows']),
                  'rejected_pairs': len(data['rejected']), 'complete_anchor_rows': sum(r['anchor_complete'] for r in data['rows']),
                  'dataset': str(output/'paired.json'), 'trained_model': False}
        save_once(output/'result.json', result)
        return result
    finally:
        lock.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('collect')
    p.add_argument('--manifest', required=True)
    p.add_argument('--policy', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--flags-json')
    p.add_argument('--anchors', type=int, choices=(1, 2, 3), default=2)
    p.add_argument('--min-gap', type=int, default=4)
    p.add_argument('--anchor-rule', choices=('earliest', 'family_coverage'), default='earliest')
    p.add_argument('--max-candidate-games', type=int, required=True)
    p.add_argument('--python')
    p.add_argument('--sc2-path')
    p.add_argument('--workers', type=int, choices=(1, 2), default=1)
    p.add_argument('--reference-policy', help='frozen deployed selector: include its first actual proposal opportunity')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    result = collect(args.manifest, args.policy, args.output, max_candidate_games=args.max_candidate_games,
                     flags=read(args.flags_json) if args.flags_json else None, anchors=args.anchors,
                     min_gap=args.min_gap, python=args.python, sc2_path=args.sc2_path,
                     dry_run=args.dry_run, plan_only=args.plan_only, anchor_rule=args.anchor_rule,
                     workers=args.workers, reference_policy=args.reference_policy)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
