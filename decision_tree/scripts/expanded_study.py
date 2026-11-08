"""Two-anchor sparse counterfactual exploration; one changed call per replay."""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from decision_tree.experiment import digest, policy_contract
from decision_tree.schema import load_policy
from decision_tree.scripts.study import ROOT, prepare, read, save


ANCHOR_RULE = {
    'first': 'first eligible approach/engage call; otherwise first eligible call',
    'second': 'first later eligible call with >=0.10 effective health ratio loss, fewer live allies, or call_number >= first+3; otherwise second eligible call',
    'candidates': 'two without replacement using SHA256(scenario hash, SC2 seed, anchor, exploration seed); no reward input',
}


def check_sources(protocol):
    for path, expected in protocol['sources'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise ValueError('Frozen source changed: ' + path)


def prepare_expanded(manifest_path, policy_path, output):
    policy = load_policy(policy_path)
    if not policy.get('exploration', {}).get('enabled'):
        raise ValueError('Expanded collection requires exploration.enabled')
    result = prepare(manifest_path, policy_path, output)
    protocol = read(Path(output)/'protocol.json')
    # Freeze analysis/orchestration as well as runtime for this experiment.
    paths = list((ROOT/'decision_tree/learning').glob('*.py')) + list((ROOT/'decision_tree/scripts').glob('*.py'))
    protocol['sources'].update({str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    protocol.update(schema_version=2, kind='expanded_skill_call_study', anchor_rule=ANCHOR_RULE,
                    anchors_per_scene=2, candidates_per_anchor=2,
                    training={'max_depth': 2, 'min_samples_leaf': 4, 'min_leaf_groups': 4,
                              'validation_fraction': .25, 'split_seed': 0},
                    return_definition='candidate whole-episode SMAC dense return minus BASE; exactly one changed call then frozen baseline dispatcher',
                    free_exploration='supported by runtime; excluded from isolated-call labels')
    save(Path(output)/'expanded_protocol.json', protocol)
    return {**result, 'planned_games': len(read(manifest_path)['episodes'])*5}


def select_anchors(gates):
    eligible = [g for g in gates if g.get('opportunity') and g.get('candidate_calls')]
    if not eligible:
        return []
    first = next((g for g in eligible if g['phase'] in ('approach', 'engage')), eligible[0])
    later = [g for g in eligible if g['call_number'] > first['call_number']]
    result = [(first, 'first_suitable')]
    if later:
        f = first['features']
        second = next((g for g in later if
                       g['features']['ally_effective_health_ratio'] <= f['ally_effective_health_ratio'] - .10 or
                       g['features']['ally_count'] < f['ally_count'] or
                       g['call_number'] >= first['call_number'] + 3), None)
        result.append((second or later[0], 'damage_or_later_boundary' if second else 'second_boundary_fallback'))
    return result


def make_plan(base_path, output, protocol):
    from decision_tree.learning.dataset import read_run, episode
    base = read_run(base_path)
    policy = base['run.json']['skill_tree']['policy']
    manifest = base['manifest.json']
    if policy_contract(policy) != protocol['policy_contract']:
        raise ValueError('Baseline contract mismatch')
    arms, scenes = [], []
    inputs = Path(output)/'inputs'
    inputs.mkdir()
    for number, scene in enumerate(manifest['episodes'], 1):
        ep = episode(base, number)
        gates = select_anchors(ep['gates'])
        scenes.append({'base_episode': number, 'group_id': scene['sha256'], 'sc2_seed': scene['sc2_seed'],
                       'anchors': len(gates), 'missing_anchors': 2-len(gates)})
        for index, (gate, reason) in enumerate(gates, 1):
            candidates = list(gate['candidate_calls'])
            # Scenario salt spreads six candidate families across the forty scenes.
            salt = [scene['sha256'], scene['sc2_seed'], index, policy['exploration']['seed']]
            candidates.sort(key=lambda c: digest([salt, c['id']]))
            for rank, candidate in enumerate(candidates[:2]):
                name = f'ep{number:03d}_a{index}_c{rank+1}_{candidate["id"]}'
                single = deepcopy(manifest)
                single['episodes'] = [deepcopy(scene)]
                single['episodes'][0]['index'] = 0
                single['sha256'] = digest({k:v for k,v in single.items() if k != 'sha256'})
                arm = deepcopy(policy)
                arm['selection'] = {'mode': 'candidate', 'candidate_id': candidate['id'],
                                    'anchor_call_number': gate['call_number'],
                                    'max_interventions_per_episode': 1, 'min_gain': 0}
                save(inputs/(name+'_manifest.json'), single)
                save(inputs/(name+'_policy.json'), load_policy(arm))
                arms.append({'name': name, 'path': str((Path(output)/name).resolve()),
                             'base_episode': number, 'arm_episode': 1,
                             'anchor_id': f'a{index}', 'anchor_call_number': gate['call_number'],
                             'anchor_reason': reason, 'anchor_step': gate['step'],
                             'candidate_id': candidate['id'], 'candidate_rank': rank,
                             'candidate_call': candidate['call'],
                             'sampling_probability': min(2, len(candidates))/len(candidates),
                             'manifest_sha256': single['sha256'], 'policy_sha256': digest(load_policy(arm)),
                             'manifest': str((inputs/(name+'_manifest.json')).resolve()),
                             'policy': str((inputs/(name+'_policy.json')).resolve())})
    return {'schema_version': 2, 'kind': 'expanded_anchor_plan', 'anchor_rule': ANCHOR_RULE,
            'baseline_path': str(Path(base_path).resolve()), 'base_hashes': base['hashes'],
            'scene_count': len(scenes), 'scenes': scenes, 'arms': arms,
            'planned_games': len(scenes)+len(arms),
            'protocol_sha256': digest(protocol), 'label_unit': 'single call, independent rerun'}


def run_command(protocol, manifest, policy, output, sc2_path=None):
    call = [sys.executable, str(ROOT/'sc2_skill_adapter/run.py'), 'run', *protocol['flags'],
            '--manifest', str(manifest), '--skill-tree', str(policy), '--output-dir', str(output)]
    if sc2_path:
        call += ['--sc2-path', str(sc2_path)]
    return call


def run_arm(protocol, manifest, policy, output, sc2_path=None):
    check_sources(protocol)
    if policy_contract(load_policy(policy)) != protocol['policy_contract']:
        raise ValueError('Arm policy contract changed')
    output = Path(output)
    if output.exists():
        from decision_tree.learning.dataset import read_run
        existing = read_run(output)
        if existing['manifest.json'] != read(manifest) or existing['run.json']['skill_tree']['policy'] != load_policy(policy):
            raise ValueError('Cannot resume a different arm: '+str(output))
        return
    log = output.parent/(output.name+'.console.log')
    with log.open('x', encoding='utf-8') as stream:
        subprocess.run(run_command(protocol, manifest, policy, output, sc2_path), cwd=ROOT,
                       stdout=stream, stderr=subprocess.STDOUT,
                       env={**os.environ, 'PYTHONUNBUFFERED': '1'}, check=True)


def collect(study, output, sc2_path=None):
    study, output = Path(study).resolve(), Path(output).resolve()
    protocol = read(study/'expanded_protocol.json')
    check_sources(protocol)
    manifest = read(study/'manifest.json')
    from sc2_skill_adapter.bridge.scenarios import validate_manifest
    validate_manifest(manifest)
    if manifest['sha256'] != protocol['manifest_sha256'] or protocol['split'] != 'development':
        raise ValueError('Changed manifest or non-development collection')
    baseline_policy = load_policy(study/'base.json')
    if baseline_policy['selection']['mode'] != 'base' or policy_contract(baseline_policy) != protocol['policy_contract']:
        raise ValueError('Changed baseline selection or policy contract')
    output.mkdir(parents=True, exist_ok=True)
    if (output/'workflow.json').exists():
        if read(output/'workflow.json')['protocol_sha256'] != digest(protocol):
            raise ValueError('Resume protocol mismatch')
    else:
        save(output/'workflow.json', {'protocol_sha256': digest(protocol), 'study': str(study)})
    lock = ROOT/'sc2_skill_adapter/.medivac-return-study.lock'
    with lock.open('x', encoding='utf-8') as handle:
        handle.write(str(output))
    try:
        print('Running BASE reference trajectories', flush=True)
        run_arm(protocol, study/'manifest.json', study/'base.json', output/'base', sc2_path)
        if (output/'anchor_plan.json').exists():
            plan = read(output/'anchor_plan.json')
            if read(output/'anchor_plan_binding.json') != {'sha256': digest(plan)} or plan['protocol_sha256'] != digest(protocol):
                raise ValueError('Anchor plan changed')
            from decision_tree.learning.dataset import read_run
            if plan['base_hashes'] != read_run(output/'base')['hashes']:
                raise ValueError('Baseline changed after anchor planning')
        else:
            plan = make_plan(output/'base', output, protocol)
            save(output/'anchor_plan.json', plan)
            save(output/'anchor_plan_binding.json', {'sha256': digest(plan)})
        print(f'BASE complete; {len(plan["arms"])} independent candidate reruns planned', flush=True)
        for i, arm in enumerate(plan['arms'], 1):
            if read(arm['manifest'])['sha256'] != arm['manifest_sha256'] or digest(load_policy(arm['policy'])) != arm['policy_sha256']:
                raise ValueError('Prepared arm changed: ' + arm['name'])
            run_arm(protocol, arm['manifest'], arm['policy'], arm['path'], sc2_path)
            print(f'Candidate {i}/{len(plan["arms"])} complete: {arm["name"]}', flush=True)
        check_sources(protocol)
        snapshot = output/'source_snapshot'
        if not snapshot.exists():
            snapshot.mkdir()
            mapping = {}
            for i, (path, expected) in enumerate(sorted(protocol['sources'].items())):
                name = f'{i:04d}_{Path(path).name}'
                (snapshot/name).write_bytes(Path(path).read_bytes())
                mapping[name] = {'original_path': path, 'sha256': expected}
            save(snapshot/'index.json', mapping)
        from decision_tree.learning.expanded_dataset import build_expanded_dataset
        dataset = build_expanded_dataset(output/'base', plan['arms'], plan)
        if not (output/'paired.json').exists():
            save(output/'paired.json', dataset)
        elif read(output/'paired.json') != dataset:
            raise ValueError('Rebuilt dataset differs from previous result')
        result = {'completed_games': plan['planned_games'], 'scene_count': plan['scene_count'],
                  'accepted_rows': len(dataset['rows']), 'rejected_pairs': len(dataset['rejected']),
                  'missing_anchor_slots': sum(s['missing_anchors'] for s in plan['scenes']),
                  'missing_candidate_slots': plan['scene_count']*4-len(plan['arms']),
                  'trained_model': False}
        if not (output/'workflow_result.json').exists():
            save(output/'workflow_result.json', result)
        return result
    finally:
        lock.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--manifest', required=True)
    prep.add_argument('--policy', required=True)
    prep.add_argument('--output-dir', required=True)
    run = sub.add_parser('collect')
    run.add_argument('--study-dir', required=True)
    run.add_argument('--output-dir', required=True)
    run.add_argument('--sc2-path')
    args = parser.parse_args()
    result = (prepare_expanded(args.manifest, args.policy, args.output_dir) if args.command == 'prepare'
              else collect(args.study_dir, args.output_dir, args.sc2_path))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
