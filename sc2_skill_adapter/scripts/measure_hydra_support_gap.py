"""Measure observable counter-fire support in a completed Zerg matrix.

The denominator counts allied Hydralisk decision windows covered by an
ordinary enemy weapon. The numerator counts those with no other allied
Hydralisk able to attack a covering enemy in the same window according to
the legal menu, SMAC mask, native range and cooldown. This measures an
available response, regardless of which command the policy selected.

For an unanswered window we also follow distinct (supporter, threat) pairs
when an accepted MOVE aims closer from outside legal firing position. The
prediction uses a speed-limited endpoint and a stationary enemy. The next
observation independently checks actual menu, mask, native range and
cooldown. Both units and at least one pressured teammate must remain
identifiable. These associations do not establish the move's intent.
"""
import argparse
from collections import Counter
import json
from math import dist
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from skills.tactical_v1.context import Context, build_snapshot


def window_facts(payload):
    state = payload['state']
    hydras = {key: unit for key, unit in state['allies'].items()
              if unit['alive'] and unit['type'] == 'Hydralisk'}
    threats_by_actor = {}
    shooters = {}
    for actor, unit in hydras.items():
        legal = payload['questions'].get(actor, {}).get('criteria', {})
        cooldown = unit.get('weapon_cooldown_loops')
        threats = set()
        for eid, relation in state['relations'].get(actor, {}).items():
            enemy = state['visible_enemies'].get(eid)
            if not enemy or not enemy['alive'] or enemy['type'] == 'Baneling':
                continue
            if relation.get('in_enemy_weapon_range') is True:
                threats.add(eid)
            if (cooldown is not None and cooldown <= state['step_mul']
                    and relation.get('in_weapon_range') is True
                    and relation.get('smac_attack_available') is True
                    and str(6 + int(eid[1:])) in legal):
                shooters.setdefault(eid, set()).add(actor)
        if threats:
            threats_by_actor[actor] = threats
    unanswered = {actor: threats for actor, threats in threats_by_actor.items()
                  if not any(shooters.get(eid, set()) - {actor} for eid in threats)}
    return unanswered, len(threats_by_actor), shooters


def count_approaches(previous, feedback, following, next_shooters, totals):
    payload, unanswered = previous
    before, after = payload['state'], following['state']
    if not feedback or feedback.get('from_game_loop') != before['game_loop'] \
            or feedback.get('to_game_loop') != after['game_loop']:
        return
    context = None
    for actor, row in feedback['actors'].items():
        unit = before['allies'][actor]
        survivor = after['allies'].get(actor, {})
        if (unit['type'] != 'Hydralisk' or not survivor.get('alive') or unit['tag'] != survivor['tag']
                or row.get('submission') != 'accepted' or row.get('observation_errors')
                or row['command']['kind'] != 'move'):
            continue
        threats = {eid for aid, targets in unanswered.items() if aid != actor
                   and after['allies'].get(aid, {}).get('alive')
                   and after['allies'][aid]['tag'] == before['allies'][aid]['tag'] for eid in targets}
        for eid in threats:
            enemy = before['visible_enemies'][eid]
            observed = after['visible_enemies'].get(eid, {})
            if not observed.get('alive') or observed['tag'] != enemy['tag']:
                continue
            if context is None:
                context = Context(build_snapshot(payload, before['game_loop']))
            reach = context.effective_attack_reach(actor, eid)
            original = dist(unit['position'], enemy['position'])
            if reach is None or reach < 0 or original <= reach or context.speed(actor) is None:
                continue
            action = int(row['requested_action_id'])
            point = context.predict_position(actor, action).position
            if dist(point, enemy['position']) >= original:
                continue
            opportunity = context.projected_attack_opportunity(actor, eid, point)
            predicted = opportunity.available and opportunity.ready
            actual = actor in next_shooters.get(eid, ())
            totals['approaches'] += 1
            totals['predicted'] += predicted
            totals['observed'] += actual
            totals['both'] += predicted and actual


def analyze_group(events_path, expected=None, *, after_casualty=False, moves=None):
    """Stream one group's decisions, optionally selecting paired episode IDs."""
    numerator, denominator = 0, 0
    steps, per_episode, outcomes = Counter(), Counter(), {}
    previous = feedback = None
    with events_path.open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            episode = row.get('episode')
            if expected is not None and episode not in expected:
                continue
            if row['event'] == 'skill_observation':
                if row['step'] != steps[episode]:
                    raise ValueError(f'{events_path}: missing or duplicate step in episode {episode}')
                payload = row['payload']
                unanswered, d, shooters = window_facts(payload)
                if moves is not None and previous:
                    count_approaches(previous, feedback, payload, shooters, moves)
                allies = payload['state']['allies']
                if not after_casualty or 3 <= sum(u['alive'] for u in allies.values()) < len(allies):
                    numerator += len(unanswered)
                    denominator += d
                    per_episode[episode] += len(unanswered)
                    previous = (payload, unanswered) if unanswered else None
                else:
                    previous = None
                feedback = None
                steps[episode] += 1
            elif row['event'] == 'step':
                feedback = row.get('feedback')
            elif row['event'] == 'episode_completed':
                if row['steps'] != steps[episode] or episode in outcomes:
                    raise ValueError(f'{events_path}: incomplete or duplicate episode {episode}')
                if expected is not None and any(row[key] != expected[episode][key]
                        for key in ('scenario_sha256', 'initial_state_sha256', 'outcome')):
                    raise ValueError(f'{events_path}: baseline identity mismatch for episode {episode}')
                outcomes[episode] = row['outcome']
                previous = feedback = None
    if set(steps) != set(outcomes) or (expected is not None and set(outcomes) != set(expected)):
        raise ValueError(f'{events_path}: missing episode evidence')
    return numerator, denominator, per_episode, outcomes


def rate(count, total):
    return f'{count / total:.1%}' if total else 'n/a'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('batch', type=Path)
    parser.add_argument('side', nargs='?', default='current', choices=('current', 'baseline'))
    parser.add_argument('--paired-losses', action='store_true', help=(
        'Only matching scenario and initial state pairs lost by both versions, '
        'from first allied casualty while three or more allies remain'))
    args = parser.parse_args()
    matrix = json.loads((args.batch / 'matrix_summary.json').read_bytes())
    if matrix['status'] != 'complete':
        raise ValueError('Expected a completed matrix')
    directory = args.batch / 'current'
    paired = None
    if args.side == 'baseline' or args.paired_losses:
        paired = json.loads((args.batch / 'baseline_episodes.json').read_bytes())
    if args.side == 'baseline':
        directory = Path(matrix['baseline']) / 'current'
    total_n = total_d = total_games = 0
    all_moves = Counter()
    for group in matrix['groups']:
        if not group.startswith('z_'):
            continue
        expected = None if paired is None else {row['episode']: row for row in paired[group]}
        if args.paired_losses:
            current = json.loads((args.batch / 'current' / group / 'episodes.json').read_bytes())
            expected = {row['episode']: row for row in current
                        if (base := expected.get(row['episode'])) and base['outcome'] == row['outcome'] == 'loss'
                        and all(base[k] == row[k] for k in ('scenario_sha256', 'initial_state_sha256'))}
        if expected == {}:
            print(f'{group}: no matching episodes')
            continue
        n, d, per_episode, outcomes = analyze_group(directory / group / 'events.jsonl', expected,
            after_casualty=args.paired_losses, moves=all_moves)
        if paired is None and len(outcomes) != matrix['games_per_group']:
            raise ValueError(f'{group}: incomplete group')
        wins = sum(result == 'win' for result in outcomes.values())
        losses = [e for e, result in outcomes.items() if result == 'loss']
        print(f'{group}: unanswered {n}/{d} = {rate(n, d)}, wins {wins}/{len(outcomes)}, '
              f'unanswered-in-losses {sum(per_episode[e] for e in losses)} across {len(losses)} losses')
        total_n += n
        total_d += d
        total_games += len(outcomes)
    if not total_games and paired is None:
        raise ValueError('No completed Zerg episodes to analyze')
    print(f'TOTAL {args.side}: {total_n}/{total_d} = {rate(total_n, total_d)} ({total_games} games)')
    print(f"Accepted approach pairs with a follow-up: {all_moves['approaches']}; "
          f"predicted open {all_moves['predicted']}, observed open {all_moves['observed']}, "
          f"both {all_moves['both']}.")


if __name__ == '__main__':
    main()
