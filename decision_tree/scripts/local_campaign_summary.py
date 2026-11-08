"""Account for completed, frozen campaign rounds without opening other rounds' results."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

from decision_tree.local.models import model_hash, predict, select_candidate
from decision_tree.local.runtime import digest, load_policy
from decision_tree.local.study import read, save_once


def require(condition, message):
    if not condition:
        raise ValueError(message)


def bound_json(path, sha):
    raw = Path(path).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == sha, 'File hash changed: '+str(path))
    return json.loads(raw)


def frozen_round(folder):
    """Verify model + structure + protocol before opening the evaluation result."""
    val = folder/'validation'
    require((val/'result.json').is_file(), 'Round lacks a completed validation: '+str(folder))
    protocol = read(val/'protocol.json')
    binding = protocol['bindings']['candidate']
    policy = load_policy(val/'candidate_policy.json')
    selected = policy.get('veto_model') or policy['model']
    require(digest(policy) == binding['policy_sha256'] and model_hash(selected) == binding['model_hash'],
            'Frozen validation policy binding mismatch')
    folders = [p.parent for p in folder.glob('*/model.json') if read(p) == selected]
    require(len(folders) == 1, 'Unable to identify a single frozen candidate export')
    export = folders[0]
    structure = read(export/'structure.json')
    require(digest(structure) == binding['structure_sha256'] and load_policy(export/'policy.json') == policy,
            'Frozen structure or export binding mismatch')
    report = read(export/'report.json')
    require(report['construction'] == 'LLM_supplied_predicates_and_topology'
            and report['model_hash'] == binding['model_hash'] and report['structure'] == structure
            and digest(report) == binding['report_sha256'], 'Frozen authored report mismatch')
    result = read(val/'result.json')
    require(result['kind'] == 'internal_validation' and result['protocol_sha256'] == digest(protocol),
            'Validation result is not bound to the frozen protocol')
    return protocol, policy, report, result


def base_coverage(base, expected_hash, policy, n):
    # A single-proposal selector agrees with BASE until its first proposal.
    # Count these prospective decisions separately from submitted validation actions.
    opportunities, first, unapplied = set(), {}, set()
    hasher = hashlib.sha256()
    with (base/'events.jsonl').open('rb') as stream:
        for line in stream:
            hasher.update(line)
            if b'"local_decision"' not in line:
                continue
            event = json.loads(line)
            if event['event'] != 'local_decision':
                continue
            require(event['selected_id'] == 'BASE', 'BASE trajectory contains an intervention')
            episode = event['episode']
            require(1 <= episode <= n, 'Out-of-manifest BASE decision')
            candidates = event['candidates']
            if candidates:
                opportunities.add(episode)
            if episode in first or not candidates:
                continue
            cid = select_candidate(policy['model'], candidates, policy['selection']['min_gain'])['candidate_id']
            if cid == 'BASE':
                continue
            chosen = next(c for c in candidates if c['id'] == cid)
            first[episode] = (event['step'], cid)
            if policy.get('veto_model'):
                prediction = predict(policy['veto_model'], chosen['family'], chosen['features'])
                if prediction['supported'] and prediction['value'] > 1e-9:
                    unapplied.add(episode)
    require(hasher.hexdigest() == expected_hash, 'BASE event stream hash mismatch')
    return opportunities, first, unapplied


def episode_records(run, ordinal, hashes, cache, *, scene, policy=None, pair=None):
    path = Path(run).resolve()
    key = (path, hashes['episodes.json'], hashes['summary.json'], hashes['run.json'], hashes['manifest.json'])
    if key not in cache:
        summary = bound_json(path/'summary.json', hashes['summary.json'])
        records = bound_json(path/'episodes.json', hashes['episodes.json'])
        metadata = bound_json(path/'run.json', hashes['run.json'])
        manifest = bound_json(path/'manifest.json', hashes['manifest.json'])
        require(summary['status'] == 'complete' and summary['completed_episodes'] == len(records) == len(manifest['episodes'])
                and manifest['sha256'] == metadata['manifest_sha256'] == summary['manifest_sha256'],
                'Incomplete source run: '+str(path))
        cache[key] = records, metadata['local_policy']['policy'], manifest['episodes']
    records, applied, episodes = cache[key]
    require(1 <= ordinal <= len(records), 'Episode ordinal out of range')
    require(all(episodes[ordinal-1][k] == scene[k] for k in ('sha256','sc2_seed')),
            'Scenario or SC2 seed differs in source run')
    if policy is not None:
        require(applied == policy, 'Source run used a different policy')
    if pair is not None:
        selection = applied['selection']
        require(selection['mode'] == 'forced' and selection['candidate_id'] == pair['candidate_id']
                and selection['anchor_step'] == pair['anchor_step'], 'Source intervention differs from pair')
    return records[ordinal-1], (str(path), ordinal)


def metrics(rows, key):
    records = [r[key] for r in rows]
    return {'wins': sum(r['outcome']=='win' for r in records),
            'win_rate': sum(r['outcome']=='win' for r in records)/len(records),
            'mean_return': math.fsum(r['return'] for r in records)/len(records),
            'proposals': sum(r['proposals'] for r in records), 'interventions': sum(r['interventions'] for r in records),
            'vetoes': sum(r['vetoes'] for r in records)}


def round_summary(folder, seen):
    protocol, policy, report, result = frozen_round(folder)
    manifest = read(folder/'manifest.json')
    split = read(folder/'split.json')
    require(manifest == read(folder/'study/manifest.json') and protocol['manifest_sha256'] == digest(manifest)
            and split == protocol['split'], 'Manifest/split differs from validation protocol')
    scenes = manifest['episodes']
    groups = {s['sha256'] for s in scenes}
    require(len(groups) == len(scenes) == 40, 'Expected 40 distinct scene groups')
    for g in groups:
        seen['groups'][g].append(folder.name)
    sources = [s for s in report['dataset_sources'] if Path(s['path']).resolve().is_relative_to(folder.resolve())]
    require(len(sources) == 1, 'Frozen report must bind one paired dataset from this round')
    source = sources[0]
    data = bound_json(source['path'], source['sha256'])
    require(data['kind'] == 'local_patch_dataset' and data['audited'] is True, 'Expected audited pairs')
    plan = read(folder/'study/plan.json')
    base = folder/'study/base'
    hashes = plan['base_hashes']
    require(hashes == protocol['base_run_hashes'] == data['provenance']['BASE']['hashes'], 'Different BASE runs')
    base_policy = load_policy(folder/'study/base_policy.json')
    incumbent = load_policy(folder/'validation/incumbent_policy.json')
    require(digest(incumbent) == protocol['bindings']['incumbent']['policy_sha256'], 'Different validator incumbent')
    cache = {}
    base_ids, pair_ids, val_ids, canonical = set(), set(), set(), set()
    for ordinal, scene in enumerate(scenes, 1):
        record, key = episode_records(base, ordinal, hashes, cache, scene=scene, policy=base_policy)
        require(record['scenario_sha256'] == scene['sha256'], 'BASE scene binding mismatch')
        base_ids.add(key)
    eligible, proposals, vetoed = base_coverage(base, hashes['events.jsonl'], policy, len(scenes))
    require(eligible == {s['base_episode'] for s in plan['scenes'] if s['anchors']}, 'Anchor coverage differs from BASE')
    provenance = {(a['entry']['base_episode'],a['entry']['anchor_step'],a['entry']['candidate_id']):a
                  for a in data['provenance']['arms']}
    require(len(provenance) == len(data['provenance']['arms']) == len(data['rows']), 'Pair provenance count mismatch')
    ordinals = {scene['sha256']:i for i,scene in enumerate(scenes,1)}
    families = Counter()
    for r in data['rows']:
        ordinal = ordinals[r['group_id']]
        require(r['sc2_seed'] == scenes[ordinal-1]['sc2_seed'] and r['audit']['commands_accepted'] is True,
                'Pair lacks scene/command audit')
        arm = provenance[(ordinal,r['anchor_step'],r['candidate_id'])]
        record, key = episode_records(arm['path'], arm['entry'].get('arm_episode',1), arm['hashes'], cache,
                                      scene=scenes[ordinal-1], pair=r)
        original, _ = episode_records(base, ordinal, hashes, cache, scene=scenes[ordinal-1], policy=base_policy)
        require(record['scenario_sha256'] == r['group_id'] and record['reward'] == r['candidate_return']
                and original['reward'] == r['base_return'] and r['gain'] == r['candidate_return']-r['base_return'],
                'Pair measured-return binding mismatch')
        pair_ids.add(key)
        canonical.add((r['group_id'],r['sc2_seed'],r['anchor_step'],r['effect_key']))
        families[r['candidate_family']] += 1
    require(len(pair_ids) == len(canonical) == len(data['rows']), 'Duplicate actual run or canonical paired effect')
    rows = result['rows']
    require(len(rows) == 10 and {r['group_id'] for r in rows} == set(split['validation_group_ids']),
            'Validation rows differ from the predeclared holdout')
    for r in rows:
        for label, used_policy in (('base',base_policy),('incumbent',incumbent),('candidate',policy)):
            v = r[label]
            record, key = episode_records(v['run'],v['ordinal'],v['hashes'],cache,
                                          scene=scenes[ordinals[r['group_id']]-1], policy=used_policy)
            require(record['scenario_sha256'] == r['group_id'] and record['reward'] == v['return']
                    and record['outcome'] == v['outcome'], 'Validation source result differs')
            val_ids.add(key)
        require(r['gain'] == r['candidate']['return'] - r['incumbent']['return'], 'Validation gain mismatch')
    extra_val = val_ids-base_ids-pair_ids
    seen['base'].update(base_ids)
    seen['pair'].update(pair_ids)
    seen['canonical'].update(canonical)
    seen['validation'].update(extra_val)
    m = {key:metrics(rows,key) for key in ('base','incumbent','candidate')}
    m['gain_vs_incumbent'] = math.fsum(r['gain'] for r in rows)/len(rows)
    m['gain_vs_base'] = math.fsum(r['candidate']['return']-r['base']['return'] for r in rows)/len(rows)
    require(math.isclose(m['gain_vs_incumbent'],result['mean_gain'],abs_tol=1e-9),'Validation mean differs')
    return {'round': folder.name, 'independent_scenes': len(groups), 'base_episodes': len(base_ids),
            'scenes_with_legal_patch_candidates': len(eligible), 'selected_anchors': len(plan['anchors']),
            'planned_arms': len(plan['arms']), 'audited_pairs': len(canonical), 'rejected_pairs': len(data['rejected']),
            'complete_anchor_rows': sum(r['anchor_complete'] for r in data['rows']), 'pairs_by_family': dict(families),
            'reference_first_proposal_anchors': (sum(a.get('reference_first_proposal') is True for a in plan['anchors'])
                                                if plan.get('reference_policy') else None),
            'new_selector_on_all_base_prefixes': {'proposals':len(proposals),'vetoes':len(vetoed),
                                                  'would_apply':len(proposals)-len(vetoed)},
            'extra_measured_validation_episodes': len(extra_val), 'unique_valid_games':len(base_ids|pair_ids|val_ids),
            'validation_decision':result['decision'], 'validation10':m, 'model_hash':report['model_hash'],
            'paired_file':source['path'], 'paired_sha256':source['sha256'],
            'validation_protocol_sha256':result['protocol_sha256']}


def summarize(campaign, rounds):
    require(bool(rounds) and len(rounds) == len(set(rounds)) and all(type(i) is int and i>=0 for i in rounds),
            'List distinct non-negative rounds explicitly')
    seen = {'groups':defaultdict(list), 'base':set(), 'pair':set(), 'canonical':set(), 'validation':set()}
    report = [round_summary(Path(campaign)/f'round_{i}',seen) for i in rounds]
    return {'schema_version':1,'scope':'explicit_completed_frozen_rounds','rounds':report,
            'totals':{'unique_scene_hashes':len(seen['groups']),
                      'repeated_scene_hashes':{g:ids for g,ids in seen['groups'].items() if len(ids)>1},
                      'unique_base_episodes':len(seen['base']), 'unique_canonical_pairs':len(seen['canonical']),
                      'distinct_paired_run_episodes':len(seen['pair']),
                      'extra_validation_episodes':len(seen['validation']-seen['base']-seen['pair']),
                      'unique_valid_games':len(seen['base']|seen['pair']|seen['validation'])}}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', required=True, type=Path)
    parser.add_argument('--rounds', nargs='+', type=int, required=True)
    parser.add_argument('--output', type=Path, help='new accounting snapshot')
    args = parser.parse_args()
    result = summarize(args.campaign,args.rounds)
    if args.output:
        save_once(args.output,result)
    print(json.dumps(result['totals'] if args.output else result,ensure_ascii=False,indent=2))
