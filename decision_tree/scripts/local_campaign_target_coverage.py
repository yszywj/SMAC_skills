"""Inspect attack coverage at a frozen selector's first proposal on training scenes."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

from decision_tree.local.authored import _source_groups, read, require, validate_split
from decision_tree.local.models import model_hash, select_candidate
from decision_tree.local.runtime import load_policy
from decision_tree.local.training import load_dataset
from skills.current.skills import damage_model, focus_fire


def coverage(observation, event, candidate):
    require(candidate['family'] == 'retarget' and candidate['duration_steps'] == 1
            and len(candidate['actors']) == 1, 'expected a single-actor retarget')
    actor, = candidate['actors']
    old = event['baseline_actions'][int(actor[1:])]
    new = candidate['actions'][actor]
    state = observation['state']
    require(type(old) is int and type(new) is int and old >= 6 and new >= 6 and old != new,
            'retarget requires two distinct attack actions')
    require(state['allies'][actor]['alive'] and set(candidate['actions']) == {actor}, 'invalid actor ownership')
    targets = []
    for action in (old, new):
        target = 'e'+str(action-6)
        require(str(action) in observation['questions'][actor]['criteria']
                and target in state['visible_enemies'] and state['visible_enemies'][target].get('alive', True)
                and state['relations'][actor][target].get('smac_attack_available'), 'illegal/unseen attack target')
        targets.append(target)
    others = [(f'u{i}', action) for i, action in enumerate(event['baseline_actions'])
              if f'u{i}' != actor and state['allies'][f'u{i}']['alive']]
    old_actors = [a for a, v in others if v == old]
    new_actors = [a for a, v in others if v == new]
    return {'actor': actor, 'old_target': targets[0], 'new_target': targets[1],
            'old_target_other_attackers': len(old_actors), 'new_target_other_attackers': len(new_actors),
            'old_target_other_actor_ids': old_actors, 'new_target_other_actor_ids': new_actors}


FIREPOWER_GROUPS = ('zero_budget', 'partial_below_target_hp', 'at_least_target_hp', 'unknown')


def remaining_firepower(observation, info):
    """One potential direct hit per other assigned, ready, confirmed-in-range actor."""
    state, target = observation['state'], info['old_target']
    target_action = str(6 + int(target[1:]))
    enemy = state['visible_enemies'][target]
    health, shield = enemy.get('health'), enemy.get('shield')
    valid_hp = all(type(x) in (float, int) and math.isfinite(x) and x >= 0 for x in (health, shield))
    denominator = health + shield if valid_hp else None
    credits, unknown = [], []
    subtotal = 0.0
    for actor in info['old_target_other_actor_ids']:
        require(actor != info['actor'] and state['allies'][actor]['alive'], 'invalid remaining attacker')
        relation = state['relations'][actor][target]
        require(target_action in observation['questions'][actor]['criteria'] and relation['smac_attack_available'],
                'remaining attacker has no legal old-target attack')
        ready, in_range = state['allies'][actor].get('weapon_ready'), relation.get('in_weapon_range')
        estimate = None
        if type(ready) is not bool or type(in_range) is not bool:
            status = 'unknown_readiness_or_range'
            unknown.append(actor)
        elif not ready or not in_range:
            status = 'excluded_cooldown_or_out_of_range'
        else:
            estimate = focus_fire.expected_damage(state, actor, target, parameters={'accurate_damage': True})
            if estimate is None:
                status = 'unknown_weapon_or_target_profile'
                unknown.append(actor)
            else:
                require(type(estimate) in (float,int) and math.isfinite(estimate) and estimate>=0,
                        'invalid direct-hit estimate')
                subtotal += estimate
                status = 'counted_direct_hit_estimate'
        credits.append({'actor': actor, 'weapon_ready': ready, 'in_weapon_range': in_range,
                        'status': status, 'direct_hit_estimate': estimate})
    numerator = None if unknown else subtotal
    ratio = numerator / denominator if numerator is not None and denominator is not None and denominator > 0 else None
    if ratio is None:
        category = 'unknown'
    elif ratio == 0:
        category = 'zero_budget'
    elif ratio < 1:
        category = 'partial_below_target_hp'
    else:
        category = 'at_least_target_hp'
    return {'category': category, 'candidate_actor_excluded': info['actor'], 'old_target': target,
            'numerator_direct_damage': numerator, 'known_direct_damage_subtotal': subtotal,
            'target_health': health, 'target_shield': shield, 'denominator_effective_hp': denominator,
            'budget_to_hp_ratio': ratio, 'unknown_actor_ids': unknown, 'actor_evidence': credits}


def check_damage_sources(base):
    raw = (Path(base['path'])/'run.json').read_bytes()
    require(hashlib.sha256(raw).hexdigest() == base['hashes']['run.json'], 'BASE metadata changed')
    recorded = {k.replace('\\','/'): v for k,v in json.loads(raw)['source_hashes'].items()}
    hashes = {}
    for module in (focus_fire, damage_model):
        path = Path(module.__file__).resolve()
        relative = 'skills/current/skills/' + path.name
        hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        require(recorded.get(relative) == hashes[relative], 'current damage source differs from frozen BASE: '+relative)
    return hashes


def first_proposals(base, episode_groups, model, *, with_firepower=False):
    """Only decision and observation events are decoded, and only training groups retained."""
    latest, chosen = {}, {}
    digest = hashlib.sha256()
    with (Path(base['path'])/'events.jsonl').open('rb') as stream:
        for line in stream:
            digest.update(line)
            if b'"skill_observation"' not in line and b'"local_decision"' not in line:
                continue
            event = json.loads(line)
            if event['event'] not in ('skill_observation', 'local_decision'):
                continue
            episode = event['episode']
            if episode not in episode_groups or episode in chosen:
                continue
            if event['event'] == 'skill_observation':
                latest[episode] = event
                continue
            cid = select_candidate(model, event['candidates'], 0)['candidate_id']
            if cid == 'BASE':
                continue
            candidate = next(c for c in event['candidates'] if c['id'] == cid)
            observation = latest[episode]
            require(event['step'] == observation['step'] and event['game_loop'] == observation['payload']['state']['game_loop'],
                    'reference decision and observation clocks differ')
            info = {'step': event['step'], 'candidate': candidate, 'game_loop': event['game_loop']}
            if candidate['family'] == 'retarget':
                info.update(coverage(observation['payload'], event, candidate))
                if with_firepower:
                    info['remaining_firepower'] = remaining_firepower(observation['payload'], info)
            chosen[episode] = info
    require(digest.hexdigest() == base['hashes']['events.jsonl'], 'BASE events changed since paired audit')
    return chosen


def summarize(records):
    by_group = defaultdict(list)
    for r in records:
        by_group[r['group_id']].append(r['veto_gain'])
    means = {g: sum(v)/len(v) for g, v in by_group.items()}
    scene_values = list(means.values())
    total = sum(scene_values)
    loo = [(total-v)/(len(means)-1) for v in scene_values] if len(means)>1 else []
    signs = lambda xs: {'positive': sum(v>1e-9 for v in xs), 'negative': sum(v< -1e-9 for v in xs),
                        'zero': sum(abs(v)<=1e-9 for v in xs)}
    return {'rows': len(records), 'groups': len(means), 'scene_equal_veto_gain': total/len(means) if means else None,
            'row_signs': signs([r['veto_gain'] for r in records]), 'scene_signs': signs(scene_values),
            'leave_one_scene_out_range': [min(loo),max(loo)] if loo else None, 'scene_means': means}


def inspect(dataset_paths, split, model, *, firepower_cohort=None):
    frozen_hash = model_hash(model)
    require(model['kind'] == 'local_gain_model', 'reference must be a gain model')
    require(firepower_cohort in (None,'prospective','historical'), 'invalid firepower cohort designation')
    require(firepower_cohort is None or len(dataset_paths) == 1,
            'use a separate firepower report per cohort')
    batches = [load_dataset(path) for path in dataset_paths]
    require(bool(batches), 'provide at least one audited paired dataset')
    groups, records, exclusions = set(), [], Counter()
    for batch in batches:
        source_groups = _source_groups(batch)
        require(not groups & source_groups, 'duplicate scene group across batches')
        groups.update(source_groups)
    train, _ = validate_split(split, {'rows': [], 'authored_group_ids': sorted(groups)})
    require(not groups & (set(model['training_group_ids'])|set(model.get('validation_group_ids', []))),
            'correction cohort overlaps gain model development scenes')
    for batch in batches:
        reference = load_policy(batch['protocol']['plan']['reference_policy'])
        require(reference['selection']['mode']=='model' and reference['selection']['min_gain']==0
                and model_hash(reference['model']) == frozen_hash and model['context_hash']==batch['context_hash']
                and model.get('game_version')==batch['game_version'], 'reference hash, threshold or context differs')
        base = batch['provenance']['BASE']
        manifest_path = Path(base['path'])/'manifest.json'
        raw_manifest = manifest_path.read_bytes()
        require(hashlib.sha256(raw_manifest).hexdigest() == base['hashes']['manifest.json'], 'BASE manifest changed')
        episodes = {i: scene for i, scene in enumerate(json.loads(raw_manifest)['episodes'], 1) if scene['sha256'] in train}
        if firepower_cohort is not None:
            damage_hashes = check_damage_sources(base)
        choices = first_proposals(base, episodes, model, with_firepower=firepower_cohort is not None)
        rows = {(r['group_id'], r['anchor_id'], r['candidate_id']):r for r in batch['rows'] if r['group_id'] in train}
        for i, scene in episodes.items():
            if i not in choices:
                exclusions['no_proposal'] += 1
                continue
            info = choices[i]
            candidate = info['candidate']
            if candidate['family'] != 'retarget':
                exclusions['first_proposal_other_family'] += 1
                continue
            anchor = f"{scene['sc2_seed']}:{info['step']}"
            key = (scene['sha256'], anchor, candidate['id'])
            row = rows.get(key)
            if row is None or not row['anchor_complete']:
                exclusions['missing_or_incomplete_first_anchor'] += 1
                continue
            require(row['reference_first_proposal'] is True and row['candidate_family'] == 'retarget'
                    and row['features'] == candidate['features'] and row['effect_key'] == candidate['effect_key'],
                    'audited reference marker, features or effect binding differs')
            records.append({**{k:v for k,v in info.items() if k!='candidate'},
                            'group_id': key[0], 'anchor_id': anchor, 'candidate_id': key[2],
                            'old_choice_gain': row['gain'], 'veto_gain': -row['gain']})
    sources = [{'paired': str(Path(path).resolve()), 'sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                'base_events_sha256': batch['provenance']['BASE']['hashes']['events.jsonl']}
               for path, batch in zip(dataset_paths, batches)]
    report = {'scope': 'training_only_first_frozen_gain_proposal', 'old_model_hash': frozen_hash, 'sources': sources,
            'label': 'BASE_return_minus_frozen_old_retarget_return', 'training_cohort_groups': len(train),
            'exclusions': dict(exclusions), 'old_target_no_other_attacker': summarize(
                [r for r in records if r['old_target_other_attackers']==0]), 'old_target_still_covered': summarize(
                [r for r in records if r['old_target_other_attackers']>0]), 'records': records}
    if firepower_cohort is not None:
        report['remaining_immediate_firepower'] = {
            'cohort_designation': firepower_cohort, 'damage_source_hashes': damage_hashes,
            'definition': 'One accurate_damage direct-hit estimate for each other original-target attacker '
                          'with known ready weapon and confirmed range, divided by current visible health plus shield.',
            'unknown_rule': 'Any unknown readiness/range, unknown eligible direct-hit estimate, or missing/nonpositive '
                            'denominator makes the ratio unknown. Known subtotal is still reported.',
            'groups': {name: summarize([r for r in records if r['remaining_firepower']['category'] == name])
                       for name in FIREPOWER_GROUPS}}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True, action='append')
    parser.add_argument('--split', type=Path, required=True)
    parser.add_argument('--old-model', type=Path, required=True)
    parser.add_argument('--firepower-cohort', choices=['prospective', 'historical'],
                        help='also inspect the predefined zero, partial, sufficient and unknown direct-hit budgets')
    parser.add_argument('--output', type=Path, help='new report file; otherwise print the full report')
    args = parser.parse_args()
    result = inspect(args.dataset, read(args.split), read(args.old_model), firepower_cohort=args.firepower_cohort)
    if args.output:
        with args.output.open('x', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='records'} if args.output else result,
                     ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
