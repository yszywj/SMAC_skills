"""Read-only reverse screen of shot_guard_only. One case = one actor/macro move.

Usage: python -B diagnose_hydra_shot_moves.py BATCH [--all-cases]
"""
import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path


LABEL = 'shot_guard_only'
LABELS = {'baseline', 'escape_only', LABEL, 'both'}
H = 8
EVENT = re.compile(rb'^\s*\{\s*"event"\s*:\s*"(episode_started|skill_observation|step|micro_window_closed)"')


def read_json(path):
    with path.open(encoding='utf-8') as f:
        return json.load(f)


def cohort(batch):
    comparison = read_json(batch / 'comparison.json')
    games = defaultdict(dict)
    hashes = set()
    for game in comparison['episodes']:
        shard, ep, sha = (game[k] for k in ('shard', 'episode', 'scenario_sha256'))
        if (set(game['versions']) != LABELS or ep in games[shard] or sha in hashes
                or not all(v['outcome'] in ('win', 'loss', 'draw', 'timeout') for v in game['versions'].values())):
            raise ValueError(f'invalid four-way episode {shard}/{ep}')
        games[shard][ep] = game
        hashes.add(sha)
    if len(hashes) != comparison['paired_scenarios'] or set(comparison['labels']) != LABELS:
        raise ValueError('inconsistent comparison cohort')
    return games


def windows(path, games):
    """Retain just two consecutive macro observations. Never bridge gaps or games."""
    prev = None
    started = set()
    with path.open('rb') as f:
        for line in f:
            if not EVENT.match(line):
                continue
            e = json.loads(line)
            ep, typ = e['episode'], e['event']
            if typ == 'episode_started' and prev:
                yield prev, None
                prev = None
            if ep not in games:
                continue
            if typ == 'episode_started':
                if ep in started or any(e[k] != games[ep][k] for k in ('initial_state_sha256', 'scenario_sha256')):
                    raise ValueError(f'{path}: identity mismatch for episode {ep}')
                started.add(ep)
            elif typ == 'skill_observation':
                state = e['payload']['state']
                if state['step_mul'] != H or state['episode'] != ep or ep not in started:
                    raise ValueError(f'{path}: invalid observation')
                if prev:
                    adjacent = ep == prev['ep'] and prev['state']['game_loop'] + H == state['game_loop']
                    yield prev, state if adjacent else None
                prev = dict(ep=ep, state=state)
            else:
                key, field = ('step', 'game_loop') if typ == 'step' else ('micro', 'issued_loop')
                if not prev or ep != prev['ep'] or e[field] != prev['state']['game_loop'] or key in prev:
                    raise ValueError(f'{path}: unmatched or duplicate {typ} {ep}/{e[field]}')
                prev[key] = e
    if prev:
        yield prev, None
    if started != games.keys():
        raise ValueError(f'{path}: missing paired episode start')


def source(row):
    skill, phase = row.get('original_skill'), row.get('original_phase')
    if row.get('provenance_rescue') is True or row.get('result') == 'exempt:urgent_rescue':
        return 'rescue'
    if skill == 'disengage_reenter' and phase == 'ESCAPE':
        return 'disengage_escape'
    if skill == 'combat_reposition' and (row.get('original_rationale') or '').startswith('retreat'):
        return 'combat_reposition_retreat'
    if skill == 'ranged_fire_formation' and phase == 'exclusive_control':
        role = row.get('formation_move_role')
        return 'formation_' + (role or 'unclassified')
    return 'unclassified'


def targets(state, actor, row):
    """Choose the evidence tier at L, before looking at the follow-up."""
    relations = state['relations'].get(actor, {})
    in_range = {e: str(u['tag']) for e, u in state['visible_enemies'].items()
                if u.get('alive') is True and u.get('tag') and relations.get(e, {}).get('in_weapon_range') is True}
    legal = {e: tag for e, tag in in_range.items() if relations[e].get('smac_attack_available') is True}
    orders = state['allies'][actor].get('orders') or []
    order_tag = None
    if orders and orders[0].get('ability_id') in (23, 3674) and orders[0].get('target_tag'):
        order_tag = str(orders[0]['target_tag'])
    target = row.get('current_target')
    if target in legal and legal[target] == str(row.get('target_tag')):
        return 'preserved_legal', {target: legal[target]}, order_tag
    native = {e: tag for e, tag in legal.items() if tag == order_tag}
    if native:
        return 'native_order_legal', native, order_tag
    if legal:
        return 'any_legal_only', legal, order_tag
    native = {e: tag for e, tag in in_range.items() if tag == order_tag and relations[e].get('smac_attack_available') is False}
    return 'native_order_only_not_reissuable', native, order_tag


def cooldown(unit):
    cd = unit.get('weapon_cooldown_loops')
    return cd if type(cd) in (float, int) and math.isfinite(cd) and cd >= 0 else None


def screen(frame, after, actor, action, row):
    state = frame['state']
    unit, loop = state['allies'][actor], state['game_loop']
    feedback = frame['step']['feedback']
    rec = feedback['actors'].get(actor, {})
    if (feedback.get('from_game_loop') != loop or rec.get('requested_action_id') != str(action)
            or rec.get('submission') != 'accepted' or (rec.get('response') or {}).get('code') != 1
            or (rec.get('command') or {}).get('ability_id') != 16):
        return 'unconfirmed_move', None
    # An audit of a different original action must not lend this move a motive or target.
    if not (row.get('original_kind') == 'move' and row.get('original_action') == row.get('final_action') == action
            and row.get('actor_tag') == unit.get('tag')):
        row = {}
    tier, before, order = targets(state, actor, row)
    if not before:
        return 'no_observed_target', None
    cd = cooldown(unit)
    if cd is None:
        return 'missing_cooldown', None
    if cd > H:
        return 'not_due', None
    if after is None or after['game_loop'] != loop + H or feedback.get('to_game_loop') != loop + H:
        return 'no_next_macro_observation', None
    nxt = after['allies'].get(actor, {})
    if not unit.get('tag') or nxt.get('tag') != unit['tag'] or nxt.get('type') != 'Hydralisk':
        return 'actor_missing_or_replaced', None
    if nxt.get('alive') is False:
        return 'actor_dead', None
    if nxt.get('alive') is not True:
        return 'actor_missing_or_replaced', None
    next_cd = cooldown(nxt)
    if next_cd is None:
        return 'missing_cooldown', None
    if next_cd > max(0, cd-H) + .02:  # match the existing report's rounded-CD restart proxy
        return 'restart_observed', None
    endpoints = {}
    visible = {str(u['tag']): e for e, u in after['visible_enemies'].items() if u.get('alive') is True}
    for target, tag in before.items():
        slot = visible.get(tag)
        rel = after['relations'].get(actor, {}).get(slot, {})
        endpoints[target] = dict(tag=tag, next_slot=slot, in_range=rel.get('in_weapon_range'),
                                 mask_legal=rel.get('smac_attack_available'))
    known = [v for v in endpoints.values() if type(v['in_range']) is bool]
    if not known:
        return 'target_missing_dead_or_range_unknown', None
    lost = any(v['in_range'] is False for v in known)
    ready = nxt.get('weapon_ready') is True
    if not (lost or ready):
        return 'neither_range_loss_nor_ready', None
    # An interior attack order does not prove a shot, but is not a move-only interval either.
    micro = frame.get('micro', {})
    if (micro.get('closed_loop') != loop + H or micro.get('interior_checks') != H-1
            or not isinstance(micro.get('records'), list) or len(micro['records']) != micro.get('authorizations')):
        return 'micro_window_missing_or_incomplete', None
    auths = [a for a in micro['records'] if a['actor'] == actor]
    if len(auths) > 1 or any(a['actor_tag'] != unit['tag'] or a['macro_action_id'] != action for a in auths):
        return 'micro_identity_mismatch', None
    auth = auths[0] if auths else {}
    if auth and auth.get('status') not in ('never_legal', 'target_lost', 'legal_control_off', 'submitted', 'already_ordered'):
        return 'micro_state_missing_or_conflicting', None
    risks = []
    if row.get('provenance_rescue') or row.get('result') == 'exempt:urgent_rescue':
        risks.append('rescue')
    if row.get('current_threats'):
        risks.append('under_threat')
    if row.get('move_sheds_threat'):
        risks.append('estimated_threat_relief')
    if row.get('improves_support') or row.get('result') == 'exempt:improves_support':
        risks.append('support_benefit')
    if row.get('health_fraction', 1) <= .35:
        risks.append('critical_hp')
    if any(row.get('observed_loss_history', ())):
        risks.append('recent_damage')
    if rec.get('observation_errors'):
        risks.append('interval_action_error')
    if not row:
        risks.append('no_matching_provenance')
    case = dict(actor=actor, actor_tag=unit['tag'], loop=loop, action=action, source=source(row),
                skill=row.get('original_skill'), phase=row.get('original_phase'), rationale=row.get('original_rationale'),
                formation_role=row.get('formation_move_role'), retreat_geometry_estimate=row.get('retreat_geometry'),
                b_result=row.get('result'), b_candidate=row.get('candidate_action'), veto=row.get('veto'),
                target_evidence=tier, native_order_tag=order, targets=endpoints, cooldown=[cd, next_cd],
                initially_ready=unit.get('weapon_ready'), next_ready=ready, lost_native_range=lost,
                risk_flags=risks, micro_status=auth.get('status'),
                micro_target=auth.get('target_tag'), micro_submitted_loop=auth.get('submitted_loop'),
                micro_response=auth.get('sc2_response'))
    mixed = auth.get('submitted_loop') is not None or auth.get('status') in ('submitted', 'already_ordered')
    return ('mixed_attack_window' if mixed else 'move_only_candidate'), case


def diagnose(batch, all_cases=False):
    games = cohort(batch)
    counts, sources, target_counts, signals, risks, b_results = (Counter() for _ in range(6))
    by_target = defaultdict(Counter)
    readiness = Counter()
    cases, examples = [], defaultdict(list)
    for shard, episodes in games.items():
        directory = batch / LABEL / shard
        rows, steps = {}, Counter()
        expected = {r['episode']: r for r in read_json(directory / 'episodes.json')}
        for ep, game in episodes.items():
            if any(expected[ep][k] != game[k] for k in ('scenario_sha256', 'initial_state_sha256')):
                raise ValueError(f'{shard}/{ep}: mismatched result')
            data = read_json(directory / 'hydra_switches' / f'episode-{ep:04}.json')
            if data['episode'] != ep:
                raise ValueError(f'{shard}/{ep}: mismatched switch artifact')
            for row in data['decisions']:
                if row['switch'] == 'B' and not row.get('followup'):
                    key = ep, row['game_loop'], row['actor']
                    if key in rows:
                        raise ValueError(f'duplicate switch row: {key}')
                    rows[key] = row
        for frame, after in windows(directory / 'events.jsonl', episodes):
            ep, state = frame['ep'], frame['state']
            if 'step' not in frame:
                raise ValueError(f'{shard}/{ep}: missing actual step')
            steps[ep] += 1
            for i, action in enumerate(frame['step']['actions']):
                actor = f'u{i}'
                unit = state['allies'].get(actor, {})
                if action not in (2, 3, 4, 5) or unit.get('type') != 'Hydralisk' or unit.get('alive') is not True:
                    continue
                row = rows.get((ep, state['game_loop'], actor), {})
                status, case = screen(frame, after, actor, action, row)
                counts[status] += 1
                if case is None:
                    continue
                case.update(id=f"{shard}/{ep:04}/L{case['loop']}/{actor}", status=status,
                            scenario_sha256=episodes[ep]['scenario_sha256'])
                if status == 'move_only_candidate':
                    sources[case['source']] += 1
                    target_counts[case['target_evidence']] += 1
                    by_target[case['target_evidence']][case['source']] += 1
                    readiness[f"{case['initially_ready']}->{case['next_ready']}"] += 1
                    risks.update(case['risk_flags'] or ['no_listed_flag_not_proof_safe'])
                    signals['lost_range' if case['lost_native_range'] else 'ready_without_observed_range_loss'] += 1
                    b_results[case['b_result'] or 'unknown'] += 1
                group = status + ':' + case['source'] + ':' + case['target_evidence']
                if len(examples[group]) < 2:
                    examples[group].append(case)
                if all_cases:
                    cases.append(case)
        if any(steps[ep] != expected[ep]['steps'] for ep in episodes):
            raise ValueError(f'{shard}: incomplete macro step coverage')
    result = dict(label=LABEL, paired_episodes=sum(map(len, games.values())), requested_hydra_moves=sum(counts.values()),
                  dispositions=counts, candidate_sources=sources, candidate_target_evidence=target_counts,
                  candidate_b_results=b_results, candidate_signals=signals, candidate_ready_transitions=readiness,
                  candidate_sources_by_target_evidence=by_target, candidate_risk_flags_overlapping=risks,
                  examples=examples, interpretation='Actor/accepted move windows, not unique shots. Same-tag visible target '
                  'at both ends; native range is estimated, separate from SMAC legality. Native-order-only targets cannot '
                  'be newly commanded. Absence of a macro CD restart is not proof of no shot, target, hit, damage or '
                  'benefit from holding position. Range loss need not be caused by this move. No counterfactual or rule change.')
    if all_cases:
        result['cases'] = cases
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('batch', type=Path, help='batch directory containing comparison.json')
    parser.add_argument('--all-cases', action='store_true', help='include every candidate and mixed-window ID with evidence')
    args = parser.parse_args()
    print(json.dumps(diagnose(args.batch, args.all_cases), indent=2, allow_nan=False))
