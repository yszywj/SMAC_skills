"""Compare every eligible declared candidate at shared, frozen BASE anchors.

Sparse observations stay sparse: illegal candidates are never filled with zero.
Each alternative is an independent rerun with one complete changed invocation.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from decision_tree.experiment import digest, policy_contract
from decision_tree.schema import load_policy
from decision_tree.scripts.study import ROOT, DEFAULT_FLAGS, prepare, read, save
from decision_tree.scripts.expanded_study import check_sources, run_arm


# Enable spatial executors without enabling a second automatic dispatcher.
# Combat positioning is intentionally opt-in: it changes the default allocator.
CANDIDATE_FLAGS = [*DEFAULT_FLAGS, '--enable-regroup', '--enable-cohesive-search']
ANCHOR_RULE = {
    'contact': 'first eligible approach/engage call, otherwise first eligible call',
    'tactical': 'first later eligible call with >=0.10 effective health ratio loss, fewer live allies, or call_number >= contact+3; otherwise next eligible call',
    'third': 'first unused search/lost_contact call with an available spatial candidate; otherwise first eligible call >= tactical+3, otherwise next eligible call',
    'ordering': 'chosen anchors sorted by call_number; no reward or outcome is consulted',
    'candidates': 'all available candidates in policy declaration order; no random truncation',
}


def _count(value, name, low=1, high=None):
    if type(value) is not int or value < low or (high is not None and value > high):
        raise ValueError(f'{name} must be an integer >= {low}' + (f' and <= {high}' if high else ''))
    return value


def prepare_candidates(manifest_path, policy_path, output, *, flags=None, anchors=3,
                       candidate_ids=None, max_candidate_games=None):
    _count(anchors, 'anchors', high=3)
    if max_candidate_games is not None:
        _count(max_candidate_games, 'max_candidate_games', low=0)
    policy = load_policy(policy_path)
    if not policy.get('exploration', {}).get('enabled') or policy['exploration'].get('feature_schema') != 3:
        raise ValueError('Candidate study requires enabled feature_schema 3 exploration')
    if flags is not None and (not isinstance(flags, list) or not all(isinstance(f, str) for f in flags)):
        raise ValueError('flags must be a list of command arguments')
    flags = list(CANDIDATE_FLAGS if flags is None else flags)
    if '--auto-spatial-dispatch' in flags:
        raise ValueError('Candidate study preserves the initial dispatcher; auto-spatial-dispatch must stay off')
    result = prepare(manifest_path, policy_path, output, candidates=candidate_ids, flags=flags)
    protocol = read(Path(output)/'protocol.json')
    paths = list((ROOT/'decision_tree/learning').glob('*.py')) + list((ROOT/'decision_tree/scripts').glob('*.py'))
    protocol['sources'].update({str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    protocol.update(schema_version=3, kind='complete_candidate_call_study', anchor_rule=ANCHOR_RULE,
                    anchors_per_scene=anchors, candidates_per_anchor='all_available_declared',
                    max_candidate_games=max_candidate_games,
                    baseline='legacy initial dispatcher with declared executor capabilities; automatic spatial dispatch off',
                    training={'max_depth': 2, 'min_leaf_groups': 4, 'min_candidate_groups': 4,
                              'validation_fraction': .25, 'split_seed': 0},
                    return_definition='candidate whole-episode SMAC dense return minus BASE; exactly one changed call then frozen baseline dispatcher',
                    free_exploration='discovery only; not accepted as isolated-call labels')
    save(Path(output)/'candidate_protocol.json', protocol)
    return {**result, 'maximum_games': result['episodes_per_arm']*(1+anchors*len(protocol['candidate_ids'])),
            'exact_candidate_games': 'determined from BASE before candidate games start',
            'max_candidate_games': max_candidate_games}


def select_anchors(gates, count=3):
    _count(count, 'anchors', high=3)
    eligible = sorted((g for g in gates if g.get('opportunity') and g.get('candidate_calls')),
                      key=lambda g: g['call_number'])
    if not eligible:
        return []
    first = next((g for g in eligible if g['phase'] in ('approach', 'engage')), eligible[0])
    chosen = [(first, 'first_contact_or_available')]
    later = [g for g in eligible if g['call_number'] > first['call_number']]
    if count >= 2 and later:
        f = first['features']
        second = next((g for g in later if
                       g['features']['ally_effective_health_ratio'] <= f['ally_effective_health_ratio']-.10 or
                       g['features']['ally_count'] < f['ally_count'] or
                       g['call_number'] >= first['call_number']+3), None)
        chosen.append((second or later[0], 'loss_or_later' if second else 'next_boundary_fallback'))
    if count >= 3:
        used = {g['call_number'] for g, _ in chosen}
        spatial = next((g for g in eligible if g['call_number'] not in used and
                        g['phase'] in ('search', 'lost_contact') and
                        any(c['call']['skill'] in ('regroup', 'cohesive_search') for c in g['candidate_calls'])), None)
        if spatial:
            chosen.append((spatial, 'spatial_phase'))
        else:
            last_number = max(used)
            remaining = [g for g in eligible if g['call_number'] > last_number]
            if remaining:
                later_gate = next((g for g in remaining if g['call_number'] >= last_number+3), None)
                chosen.append((later_gate or remaining[0], 'third_later_boundary' if later_gate else 'third_boundary_fallback'))
    return sorted(chosen, key=lambda item: item[0]['call_number'])


def make_plan(base_path, output, protocol):
    from decision_tree.learning.dataset import read_run, episode
    base = read_run(base_path)
    policy = base['run.json']['skill_tree']['policy']
    if policy_contract(policy) != protocol['policy_contract'] or policy['selection']['mode'] != 'base':
        raise ValueError('Baseline policy contract mismatch')
    manifest = base['manifest.json']
    order = {cid: i for i, cid in enumerate(protocol['candidate_ids'])}
    arms, scenes, anchor_records = [], [], []
    inputs = Path(output)/'inputs'
    inputs.mkdir()
    for number, scene in enumerate(manifest['episodes'], 1):
        gates = select_anchors(episode(base, number)['gates'], protocol['anchors_per_scene'])
        scenes.append({'base_episode': number, 'group_id': scene['sha256'], 'sc2_seed': scene['sc2_seed'],
                       'anchors': len(gates), 'missing_anchors': protocol['anchors_per_scene']-len(gates)})
        for index, (gate, reason) in enumerate(gates, 1):
            available = gate['candidate_calls']
            if any(c['id'] not in order for c in available) or len({c['id'] for c in available}) != len(available):
                raise ValueError('Unknown or duplicate candidate at BASE anchor')
            available = sorted(available, key=lambda c: order[c['id']])
            anchor_records.append({'base_episode': number, 'anchor_id': f'a{index}',
                                   'call_number': gate['call_number'], 'step': gate['step'],
                                   'phase': gate['phase'], 'reason': reason,
                                   'available_candidate_ids': [c['id'] for c in available],
                                   'tested_candidate_ids': [c['id'] for c in available],
                                   'unavailable_candidate_ids': [cid for cid in order if cid not in {c['id'] for c in available}]})
            for rank, candidate in enumerate(available):
                name = f'ep{number:03d}_a{index}_c{rank+1}_{candidate["id"]}'
                single = deepcopy(manifest)
                single['episodes'] = [deepcopy(scene)]
                single['episodes'][0]['index'] = 0
                single['sha256'] = digest({k:v for k,v in single.items() if k != 'sha256'})
                arm = deepcopy(policy)
                arm['selection'] = {'mode': 'candidate', 'candidate_id': candidate['id'],
                                    'anchor_call_number': gate['call_number'],
                                    'max_interventions_per_episode': 1, 'min_gain': 0}
                arm = load_policy(arm)
                manifest_path, policy_path = inputs/(name+'_manifest.json'), inputs/(name+'_policy.json')
                save(manifest_path, single)
                save(policy_path, arm)
                arms.append({'name': name, 'path': str((Path(output)/name).resolve()),
                             'base_episode': number, 'arm_episode': 1, 'anchor_id': f'a{index}',
                             'anchor_call_number': gate['call_number'], 'anchor_phase': gate['phase'],
                             'anchor_reason': reason, 'anchor_step': gate['step'],
                             'candidate_id': candidate['id'], 'candidate_rank': rank,
                             'candidate_call': candidate['call'], 'sampling_probability': 1.0,
                             'manifest_sha256': single['sha256'], 'policy_sha256': digest(arm),
                             'manifest': str(manifest_path.resolve()), 'policy': str(policy_path.resolve())})
    cap = protocol.get('max_candidate_games')
    return {'schema_version': 3, 'kind': 'complete_candidate_anchor_plan', 'anchor_rule': ANCHOR_RULE,
            'baseline_path': str(Path(base_path).resolve()), 'base_hashes': base['hashes'],
            'scene_count': len(scenes), 'scenes': scenes, 'anchors': anchor_records, 'arms': arms,
            'planned_games': len(scenes)+len(arms), 'candidate_games': len(arms),
            'budget_exceeded': cap is not None and len(arms)>cap,
            'protocol_sha256': digest(protocol), 'label_unit': 'single call, independent rerun',
            'coverage': 'every available declared candidate at every selected anchor'}


def _snapshot(protocol, output):
    directory = Path(output)/'source_snapshot'
    if directory.exists():
        mapping = read(directory/'index.json')
        if {v['original_path']: v['sha256'] for v in mapping.values()} != protocol['sources']:
            raise ValueError('Snapshot source index changed')
        for name, info in mapping.items():
            if hashlib.sha256((directory/name).read_bytes()).hexdigest() != info['sha256']:
                raise ValueError('Snapshot bytes changed')
        return
    directory.mkdir()
    index = {}
    for i, (path, expected) in enumerate(sorted(protocol['sources'].items())):
        name = f'{i:04d}_{Path(path).name}'
        (directory/name).write_bytes(Path(path).read_bytes())
        index[name] = {'original_path': path, 'sha256': expected}
    save(directory/'index.json', index)


def collect_candidates(study, output, sc2_path=None, *, plan_only=False):
    study, output = Path(study).resolve(), Path(output).resolve()
    protocol = read(study/'candidate_protocol.json')
    check_sources(protocol)
    manifest = read(study/'manifest.json')
    from sc2_skill_adapter.bridge.scenarios import validate_manifest
    validate_manifest(manifest)
    if manifest['sha256'] != protocol['manifest_sha256'] or protocol['split'] != 'development':
        raise ValueError('Changed manifest or non-development collection')
    baseline = load_policy(study/'base.json')
    if baseline['selection']['mode'] != 'base' or policy_contract(baseline) != protocol['policy_contract']:
        raise ValueError('Changed baseline policy')
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
        result = {'completed_games': plan['scene_count'], 'scene_count': plan['scene_count'],
                  'planned_games': plan['planned_games'], 'candidate_games': len(plan['arms']),
                  'missing_anchor_slots': sum(s['missing_anchors'] for s in plan['scenes']),
                  'trained_model': False}
        print(f'BASE complete; {len(plan["arms"])} complete-candidate reruns planned', flush=True)
        if plan['budget_exceeded'] or plan_only:
            check_sources(protocol)
            _snapshot(protocol, output)
            return {**result, 'status': 'budget_exceeded' if plan['budget_exceeded'] else 'plan_ready',
                    'started_candidate_games': 0, 'plan': str(output/'anchor_plan.json')}
        for i, arm in enumerate(plan['arms'], 1):
            if read(arm['manifest'])['sha256'] != arm['manifest_sha256'] or digest(load_policy(arm['policy'])) != arm['policy_sha256']:
                raise ValueError('Prepared arm changed: '+arm['name'])
            run_arm(protocol, arm['manifest'], arm['policy'], arm['path'], sc2_path)
            print(f'Candidate {i}/{len(plan["arms"])} complete: {arm["name"]}', flush=True)
        check_sources(protocol)
        _snapshot(protocol, output)
        from decision_tree.learning.choice_dataset import build_choice_dataset
        dataset = build_choice_dataset(output/'base', plan['arms'], plan)
        data_path = output/'vector_paired.json'
        if not data_path.exists():
            save(data_path, dataset)
        elif read(data_path) != dataset:
            raise ValueError('Rebuilt dataset differs from previous result')
        result.update(status='complete', completed_games=plan['planned_games'],
                      accepted_rows=len(dataset['rows']), rejected_pairs=len(dataset['rejected']),
                      dataset_path=str(data_path))
        if not (output/'workflow_result.json').exists():
            save(output/'workflow_result.json', result)
        return result
    finally:
        lock.unlink()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--manifest', required=True)
    prep.add_argument('--policy', default=str(ROOT/'decision_tree/configs/candidate_v3.json'))
    prep.add_argument('--output-dir', required=True)
    prep.add_argument('--anchors', type=int, default=3)
    prep.add_argument('--candidate', action='append')
    prep.add_argument('--max-candidate-games', type=int)
    prep.add_argument('--flags-json')
    run = sub.add_parser('collect')
    run.add_argument('--study-dir', required=True)
    run.add_argument('--output-dir', required=True)
    run.add_argument('--sc2-path')
    run.add_argument('--plan-only', action='store_true', help='Run BASE and save the exact plan without candidate games')
    args = parser.parse_args(argv)
    result = (prepare_candidates(args.manifest, args.policy, args.output_dir,
                                 anchors=args.anchors, candidate_ids=args.candidate,
                                 max_candidate_games=args.max_candidate_games,
                                 flags=read(args.flags_json) if args.flags_json else None)
              if args.command == 'prepare' else
              collect_candidates(args.study_dir, args.output_dir, args.sc2_path, plan_only=args.plan_only))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
