"""Shared pre-decision context for all candidates of a vector-output tree.

Semantic target slots are computed before choosing an invocation. Every
candidate at an anchor sees the SAME vector. Target IDs, rewards and outcomes
are never feature values. Volley/escape quantities are public heuristics, not
simulated damage or guarantees of safe navigation.
"""
import math

from .features import FEATURE_NAMES, extract_features
from .targeting import choose_target, target_metrics

TARGET_SLOTS = (('weakest', 'lowest_effective_health'), ('threat', 'highest_threat'),
                ('easy', 'easiest_focus'))
TARGET_METRICS = ('available', 'effective_health', 'health_ratio', 'attackable_fraction',
                  'ready_fraction', 'mean_distance', 'distance_known_fraction', 'threat_volley',
                  'threat_known', 'focus_damage', 'damage_known_fraction', 'overkill_ratio')
CONTEXT_NAMES = ('ally_threatened_fraction', 'retreat_increasing_distance_fraction',
                 'mean_nearest_enemy_distance', 'nearest_distance_known_fraction')
CHOICE_FEATURE_NAMES = tuple(FEATURE_NAMES) + tuple(
    f'target_{slot}_{metric}' for slot, _ in TARGET_SLOTS for metric in TARGET_METRICS) + CONTEXT_NAMES


def _number(value):
    return (float(value) if not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value) and value >= 0 else None)


def extract_choice_features(state, memory=None):
    result = extract_features(state, memory)
    for slot, rule in TARGET_SLOTS:
        metrics = target_metrics(state, choose_target(rule, state))
        for metric in TARGET_METRICS:
            result[f'target_{slot}_{metric}'] = float(metrics[metric])
    living = {a: u for a, u in state.get('allies', {}).items() if u.get('alive')}
    visible = {e for e, u in state.get('visible_enemies', {}).items()
               if u.get('alive', True) and (u.get('health', 0) or 0)+(u.get('shield', 0) or 0) > 0}
    nearest, threatened, retreat_ratios = [], 0, []
    for actor in living:
        relations = {e: r for e, r in state.get('relations', {}).get(actor, {}).items() if e in visible}
        distances = [value for r in relations.values() if (value := _number(r.get('distance'))) is not None]
        if distances:
            nearest.append(min(distances))
        exposed = any(r.get('in_enemy_weapon_range') is True or
                      r.get('enemy_weapon_range_status') == 'in_range' for r in relations.values())
        threatened += int(exposed)
        if not exposed or not distances:
            continue
        moves = []
        for move in state.get('movement_candidates', {}).get(actor, {}).values():
            moved = [v for e, d in move.get('distances_to_visible_enemies', {}).items()
                     if e in visible and (v := _number(d)) is not None]
            if moved:
                moves.append(float(min(moved) > min(distances) + 1e-6))
        if moves:
            retreat_ratios.append(sum(moves)/len(moves))
    result.update(ally_threatened_fraction=threatened/max(1, len(living)),
                  retreat_increasing_distance_fraction=sum(retreat_ratios)/max(1, len(retreat_ratios)),
                  mean_nearest_enemy_distance=sum(nearest)/max(1, len(nearest)),
                  nearest_distance_known_fraction=len(nearest)/max(1, len(living)))
    if set(result) != set(CHOICE_FEATURE_NAMES) or any(not math.isfinite(v) for v in result.values()):
        raise ValueError('invalid public choice feature vector')
    return result
