"""Read-only training evidence for LLM-authored local trees.

summary uses predetermined semantic thresholds, never searches/ranks splits or
generates a tree. cases selects distinct training scenes and opens only those
episodes' local windows. validation is a separate command for an already-built
model report; it is never included in construction outputs.
"""
import argparse
from collections import defaultdict
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path


FAMILIES = ('retarget', 'retreat_veto', 'retreat_step', 'regroup_step')
# The original cuts were set before collection. The legal-attack count and
# recover-role cuts were added after the small early training-only snapshot,
# before opening the complete R0 training set. No holdout informs these cuts.
CUTS = {
    'retarget': [('actor_weapon_ready', [0.5]), ('target_health_ratio', [0.25, 0.5]),
                 ('target_effective_health', [40, 80]), ('target_distance', [4, 6]),
                 ('assigned_damage_known', [0.5]), ('assigned_damage_before', [20, 40]),
                 ('actor_threat_count', [1, 2])],
    'retreat_veto': [('actor_health_ratio', [0.25, 0.5, 0.75]),
                     ('actor_recent_loss_ratio', [0.05, 0.15]),
                     ('actor_weapon_ready', [0.5]), ('actor_threat_count', [1, 2]),
                     ('target_health_ratio', [0.25, 0.5]), ('baseline_recover', [0.5])],
    'retreat_step': [('actor_health_ratio', [0.25, 0.5, 0.75]),
                     ('actor_recent_loss_ratio', [0.05, 0.15]),
                     ('actor_weapon_ready', [0.5]), ('actor_threat_count', [1, 2]),
                     ('nearest_enemy_before', [3, 6]), ('nearest_enemy_after', [3, 6]),
                     ('actor_legal_attack_count', [1, 2])],
    'regroup_step': [('nearest_ally_before', [8, 12]), ('nearest_ally_after', [6, 10]),
                     ('nearest_enemy_before', [4, 8]), ('ally_alive_count', [5, 8]),
                     ('team_health_ratio', [0.5, 0.75])],
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def row_key(row):
    return row['group_id'], str(row['anchor_id']), row['candidate_id']


def stats(rows):
    by_scene = defaultdict(list)
    for row in rows:
        value = row['gain']
        require(type(value) in (int, float) and math.isfinite(value), 'non-finite training label')
        by_scene[row['group_id']].append(value)
    means = [sum(xs)/len(xs) for xs in by_scene.values()]
    return {'rows': len(rows), 'groups': len(means),
            'mean': sum(means)/len(means) if means else None,
            'positive': sum(r['gain'] > 1e-9 for r in rows),
            'negative': sum(r['gain'] < -1e-9 for r in rows),
            'zero': sum(abs(r['gain']) <= 1e-9 for r in rows)}


def load_training(prepared):
    prepared = Path(prepared).resolve()
    split = read(prepared/'split.json')
    training = set(split['training_group_ids'])
    validation = set(split['validation_group_ids'])
    require(training and validation and not training & validation, 'invalid fixed split')
    rows = read(prepared/'training_rows.json')
    require(isinstance(rows, list) and bool(rows), 'nonempty prepared training rows required')
    seen = set()
    for row in rows:
        require(row['group_id'] in training and row['group_id'] not in validation,
                'refusing to expose a non-training scene')
        require(row_key(row) not in seen, 'duplicate prepared training row')
        require(row['candidate_family'] in FAMILIES, 'unknown local family')
        seen.add(row_key(row))
    summary = read(prepared/'training_summary.json')
    for family in {r['candidate_family'] for r in rows}:
        measured = stats([r for r in rows if r['candidate_family'] == family])
        recorded = summary[family]
        require(recorded['rows'] == measured['rows'] and recorded['groups'] == measured['groups'],
                'prepared summary and rows disagree')
        require(math.isclose(recorded['scene_equal_mean_gain'], measured['mean'], abs_tol=1e-8),
                'prepared summary gain mismatch')
    return {'directory': prepared, 'rows': rows, 'training_ids': training,
            'validation_ids': validation, 'preparation': read(prepared/'preparation.json')}


def summarize(prepared, family=None):
    evidence = load_training(prepared)
    families = [family] if family else list(FAMILIES)
    result = {'scope': 'training_only', 'label': evidence['preparation']['kind'],
              'training_cohort_groups': len(evidence['training_ids']),
              'training_groups_with_rows': len({r['group_id'] for r in evidence['rows']}),
              'threshold_rule': 'fixed_semantic_cuts_v2_early_training_hypotheses', 'families': {}}
    for name in families:
        rows = [r for r in evidence['rows'] if r['candidate_family'] == name]
        conditions = []
        for feature, thresholds in CUTS[name]:
            for threshold in thresholds:
                left = [r for r in rows if r['features'][feature] <= threshold]
                right = [r for r in rows if r['features'][feature] > threshold]
                conditions.append({'feature': feature, 'threshold': threshold,
                                   'le': stats(left), 'gt': stats(right)})
        result['families'][name] = {'overall': stats(rows), 'conditions': conditions}
    return result


def select_cases(evidence, *, family=None, sign='negative', limit=3, candidate_id=None, group_id=None):
    require(sign in ('positive', 'negative', 'zero'), 'unsupported label sign')
    require(type(limit) is int and 1 <= limit <= 40, 'limit must be in [1,40]')
    if group_id is not None:
        require(group_id in evidence['training_ids'], 'requested scene is not in training')
    def wanted(row):
        value = row['gain']
        signed = value > 1e-9 if sign == 'positive' else value < -1e-9 if sign == 'negative' else abs(value) <= 1e-9
        return (signed and (family is None or row['candidate_family'] == family)
                and (candidate_id is None or row['candidate_id'] == candidate_id)
                and (group_id is None or row['group_id'] == group_id))
    candidates = sorted((r for r in evidence['rows'] if wanted(r)),
                        key=lambda r: (-abs(r['gain']), *row_key(r)))
    selected, groups = [], set()
    for row in candidates:
        if row['group_id'] in groups:
            continue
        selected.append(row)
        groups.add(row['group_id'])
        if len(selected) == limit:
            break
    return selected


def locate_case(evidence, row):
    """Resolve exact training pair through hashed preparation provenance."""
    require(row['group_id'] in evidence['training_ids'], 'case is outside training')
    found = []
    for source in evidence['preparation']['dataset_sources']:
        path = Path(source['path'])
        require(file_hash(path) == source['sha256'], 'paired source changed since preparation')
        data = read(path)
        originals = [r for r in data['rows'] if row_key(r) == row_key(row)]
        if not originals:
            continue
        require(len(originals) == 1, 'ambiguous original paired row')
        original = originals[0]
        require(original['features'] == row['features'], 'prepared case features differ from paired source')
        original_gain = row.get('original_gain', row['gain'])
        require(math.isclose(original['gain'], original_gain, abs_tol=1e-8), 'prepared label differs from original pair')
        base = data['provenance']['BASE']
        manifest_path = Path(base['path'])/'manifest.json'
        require(file_hash(manifest_path) == base['hashes']['manifest.json'], 'BASE manifest changed')
        manifest = read(manifest_path)
        episodes = [i for i, scene in enumerate(manifest['episodes'], 1)
                    if scene['sha256'] == row['group_id'] and scene['sc2_seed'] == row['sc2_seed']]
        require(len(episodes) == 1, 'ambiguous BASE scene and seed')
        for arm in data['provenance']['arms']:
            entry = arm['entry']
            if (entry['base_episode'] == episodes[0] and entry['anchor_step'] == row['anchor_step']
                    and entry['candidate_id'] == row['candidate_id']):
                found.append({'dataset': str(path), 'BASE': base, 'CANDIDATE': arm,
                              'base_episode': episodes[0], 'arm_episode': entry.get('arm_episode', 1),
                              'original_row': original})
    require(len(found) == 1, 'case must resolve to exactly one original pair')
    return found[0]


def _window(provenance, episode_number, anchor_step, radius):
    path = Path(provenance['path'])/'events.jsonl'
    require(file_hash(path) == provenance['hashes']['events.jsonl'], 'original event stream changed')
    records = defaultdict(dict)
    with path.open(encoding='utf-8-sig') as stream:
        for line in stream:
            event = json.loads(line)
            if event.get('episode') != episode_number:
                continue
            kind = event['event']
            if kind not in ('skill_observation', 'local_decision', 'skill_step', 'step'):
                continue
            step = event['step'] - (1 if kind == 'step' else 0)
            if max(0, anchor_step - radius) <= step <= anchor_step + radius:
                records[step][kind] = event
    require(anchor_step in records, 'anchor not present in original stream')
    return records


def _hp(unit):
    if not unit or unit.get('health') is None:
        return None
    return unit['health'] + (unit.get('shield') or 0)


def trace_case(evidence, row, radius=3):
    require(type(radius) is int and 0 <= radius <= 12, 'window radius must be in [0,12]')
    bound = locate_case(evidence, row)
    windows = {name: _window(bound[name], bound['base_episode'] if name == 'BASE' else bound['arm_episode'],
                             row['anchor_step'], radius) for name in ('BASE', 'CANDIDATE')}
    anchor = windows['BASE'][row['anchor_step']]['local_decision']
    candidates = [c for c in anchor['candidates'] if c['id'] == row['candidate_id']]
    require(len(candidates) == 1 and len(candidates[0]['actors']) == 1, 'invalid local candidate binding')
    candidate = candidates[0]
    actor = candidate['actors'][0]
    slot = int(actor[1:])
    candidate_action = candidate['actions'][actor]
    base_action = anchor['baseline_actions'][slot]
    tracked_action = candidate_action if candidate_action >= 6 else base_action
    tracked_target = f'e{tracked_action - 6}' if tracked_action >= 6 else None
    traces = []
    for name, records in windows.items():
        previous_hp, previous_step = None, None
        for step in sorted(records):
            events = records[step]
            state = events['skill_observation']['payload']['state']
            unit = state['allies'][actor]
            decision, submitted = events['local_decision'], events['step']
            action = decision['final_actions'][slot]
            target = f'e{action - 6}' if action >= 6 else None
            visible = state.get('visible_enemies', {})
            target_hp = _hp(visible.get(tracked_target))
            loss = (previous_hp - target_hp if previous_step == step - 1
                    and previous_hp is not None and target_hp is not None else None)
            role = decision['baseline_roles'].get(actor)
            chosen = next((c for c in decision['candidates'] if c['id'] == decision['selected_id']), None)
            if chosen is not None:
                role = chosen['roles'].get(actor, role)
            traces.append({'arm': name, 'step': step, 'game_loop': state['game_loop'], 'actor': actor,
                           'health': unit.get('health'), 'shield': unit.get('shield'), 'alive': unit['alive'],
                           'weapon_ready': unit.get('weapon_ready'), 'role': role,
                           'baseline_action': decision['baseline_actions'][slot], 'final_action': action,
                           'actual_target': target, 'actual_target_hp': _hp(visible.get(target)),
                           'tracked_target': tracked_target, 'tracked_target_hp': target_hp,
                           'tracked_target_observed_loss': loss,
                           'actor_recent_health_loss': unit.get('health_lost_since_previous_decision'),
                           'actor_recent_shield_loss': unit.get('shield_lost_since_previous_decision'),
                           'baseline_estimated_damage_to_tracked': decision['baseline_predicted_damage'].get(tracked_target),
                           'step_reward': submitted['reward'],
                           'submission': submitted['feedback'].get('actors', {}).get(actor, {}).get('submission')})
            previous_hp, previous_step = target_hp, step
    return {'group_id': row['group_id'], 'anchor_id': row['anchor_id'], 'anchor_step': row['anchor_step'],
            'candidate_id': row['candidate_id'], 'family': row['candidate_family'], 'actor': actor,
            'training_label': row['gain'], 'original_candidate_gain': bound['original_row']['gain'],
            'base_return': bound['original_row']['base_return'], 'candidate_return': bound['original_row']['candidate_return'],
            'features': deepcopy(row['features']), 'dataset': bound['dataset'],
            'base_path': bound['BASE']['path'], 'candidate_path': bound['CANDIDATE']['path'], 'trace': traces,
            'damage_note': 'Observed target health/shield change is not actor-attributed damage; missing visibility stays unknown. '
                           'Baseline predicted damage is an estimate and is not a reward label.'}


def cases(prepared, **options):
    radius = options.pop('radius', 3)
    evidence = load_training(prepared)
    rows = select_cases(evidence, **options)
    return {'scope': 'training_only', 'label': evidence['preparation']['kind'],
            'case_selection': 'largest_absolute_training_label_per_distinct_scene_then_stable_ID',
            'cases': [trace_case(evidence, row, radius) for row in rows]}


def validation_report(path):
    """Explicit post-construction validation output; no training diagnostics."""
    report = read(path)
    require(report.get('construction') == 'LLM_supplied_predicates_and_topology'
            and report.get('model_hash'), 'an already-built authored model report is required')
    require(isinstance(report.get('validation'), dict), 'missing frozen validation result')
    return {'scope': 'post_construction_validation_only', 'model_hash': report['model_hash'],
            'validation': deepcopy(report['validation']),
            'note': 'Using this feedback to revise a model makes these scenes development evidence for that revision.'}


def _n(value):
    if value is None:
        return '-'
    return f'{value:.4g}' if isinstance(value, float) else str(value)


def _stat(value):
    return f"n={value['rows']} g={value['groups']} +/0/-={value['positive']}/{value['zero']}/{value['negative']} mean={_n(value['mean'])}"


def render(result):
    lines = [f"scope={result['scope']}"]
    if 'families' in result:
        lines.append(f"label={result['label']} cohort={result['training_cohort_groups']} groups_with_rows={result['training_groups_with_rows']}")
        lines.append('Fixed semantic thresholds; means are equal-scene descriptions, not automatically selected splits.')
        for family, part in result['families'].items():
            lines.append(f"\n{family}: {_stat(part['overall'])}")
            for condition in part['conditions']:
                lines.append(f"  {condition['feature']} <= {_n(condition['threshold'])}: {_stat(condition['le'])} | >: {_stat(condition['gt'])}")
    elif 'cases' in result:
        lines.append(f"label={result['label']} cases={len(result['cases'])}; one case per distinct training scene")
        for case in result['cases']:
            lines.append(f"\n{case['candidate_id']} scene={case['group_id']} anchor={case['anchor_step']} label={_n(case['training_label'])} original_gain={_n(case['original_candidate_gain'])}")
            lines.append(f"returns BASE={_n(case['base_return'])} CANDIDATE={_n(case['candidate_return'])}; tracked actor={case['actor']}")
            lines.append('arm step loop hp shield role action target trackedHP observedLoss predictedDamage reward receipt')
            for r in case['trace']:
                values = [r[k] for k in ('arm', 'step', 'game_loop', 'health', 'shield', 'role', 'final_action',
                                         'actual_target', 'tracked_target_hp', 'tracked_target_observed_loss',
                                         'baseline_estimated_damage_to_tracked', 'step_reward', 'submission')]
                lines.append(' '.join(_n(v) for v in values))
            lines.append(case['damage_note'])
            lines.append('BASE: ' + case['base_path'])
            lines.append('CANDIDATE: ' + case['candidate_path'])
    else:
        lines.append(json.dumps(result, ensure_ascii=False, indent=2))
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    summary = sub.add_parser('summary', help='training-only family statistics at fixed semantic cuts')
    summary.add_argument('--prepared', type=Path, required=True)
    summary.add_argument('--family', choices=FAMILIES)
    case = sub.add_parser('cases', help='training-only representative signed-gain trajectory windows')
    case.add_argument('--prepared', type=Path, required=True)
    case.add_argument('--family', choices=FAMILIES)
    case.add_argument('--sign', choices=('positive', 'negative', 'zero'), default='negative')
    case.add_argument('--limit', type=int, default=3)
    case.add_argument('--candidate-id')
    case.add_argument('--group-id', help='must be a declared training scene')
    case.add_argument('--radius', type=int, default=3, help='decision steps before and after anchor, 0..12')
    validation = sub.add_parser('validation', help='explicit post-construction validation from a frozen authored report')
    validation.add_argument('--report', type=Path, required=True)
    for command in (summary, case, validation):
        command.add_argument('--format', choices=('text', 'json'), default='text')
    args = parser.parse_args(argv)
    if args.command == 'summary':
        result = summarize(args.prepared, args.family)
    elif args.command == 'cases':
        result = cases(args.prepared, family=args.family, sign=args.sign, limit=args.limit,
                       candidate_id=args.candidate_id, group_id=args.group_id, radius=args.radius)
    else:
        result = validation_report(args.report)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) if args.format == 'json' else render(result))


if __name__ == '__main__':
    main()
