"""Visible-state target summaries and semantic rules, without observed outcomes.

Volley values use compatible public base damage times hits. They deliberately
ignore armor, upgrades, splash and cooldown: these are targeting heuristics,
not predicted actual damage or DPS (the public profile has no attack period).
"""
import math

TARGET_METRIC_NAMES = ('available', 'effective_health', 'health_ratio',
    'attackable_fraction', 'ready_fraction', 'mean_distance', 'distance_known_fraction',
    'threat_volley', 'threat_known', 'focus_damage', 'damage_known_fraction', 'overkill_ratio')


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value)


def _health(unit):
    hp, shield = _number(unit.get('health')), _number(unit.get('shield', 0))
    return None if hp is None or shield is None else max(0., hp) + max(0., shield)


def _volley(profile, target):
    if not profile.get('known') or not isinstance(target.get('is_flying'), bool):
        return None
    kinds = {'air', 'any'} if target['is_flying'] else {'ground', 'any'}
    if target.get('unit_type_id') == 4:  # Public Colossus weapon-domain exception.
        kinds.add('air')
    values = []
    for weapon in profile.get('weapons', []):
        if weapon.get('target_type') not in kinds:
            continue
        damage, hits = _number(weapon.get('base_damage')), _number(weapon.get('hits_per_attack'))
        if damage is not None and hits is not None and damage >= 0 and hits >= 1:
            values.append(damage * hits)
    return max(values) if values else None


def target_metrics(state, target):
    result = dict.fromkeys(TARGET_METRIC_NAMES, 0.)
    enemy = state.get('visible_enemies', {}).get(target)
    hp = _health(enemy) if enemy is not None else None
    if enemy is None or not enemy.get('alive', True) or hp is None or hp <= 0:
        return result
    living = {a:u for a,u in state.get('allies', {}).items() if u.get('alive', False)}
    profiles = state.get('unit_type_profiles', {})
    maximum = sum(max(0., _number(enemy.get(k)) or 0.) for k in ('max_health', 'max_shield'))
    legal, ready, distances, damage = [], [], [], []
    for actor, ally in living.items():
        relation = state.get('relations', {}).get(actor, {}).get(target, {})
        distance = _number(relation.get('distance'))
        if distance is not None and distance >= 0:
            distances.append(distance)
        if relation.get('smac_attack_available') is True:
            legal.append(actor)
            if ally.get('weapon_ready') is True and relation.get('in_weapon_range') is not False:
                ready.append(actor)
            volley = _volley(profiles.get(str(ally.get('unit_type_id')), {}), enemy)
            if volley is not None:
                damage.append(volley)
    profile = profiles.get(str(enemy.get('unit_type_id')), {})
    threats = [_volley(profile, ally) for ally in living.values()]
    threats = [value for value in threats if value is not None]
    focus = math.fsum(damage)
    denominator = max(1, len(living))
    result.update(available=1., effective_health=hp,
        health_ratio=min(1., hp/maximum) if maximum > 0 else 0.,
        attackable_fraction=len(legal)/denominator, ready_fraction=len(ready)/denominator,
        mean_distance=math.fsum(distances)/len(distances) if distances else 0.,
        distance_known_fraction=len(distances)/denominator,
        threat_volley=max(threats) if threats else 0., threat_known=float(bool(threats)),
        focus_damage=focus, damage_known_fraction=len(damage)/max(1, len(legal)),
        overkill_ratio=max(0., focus-hp)/hp)
    return result


def choose_target(rule, state):
    choices = {}
    for target in state.get('visible_enemies', {}):
        metrics = target_metrics(state, target)
        if metrics['available'] and metrics['attackable_fraction'] > 0:
            choices[target] = metrics
    if not choices:
        return None
    if rule == 'lowest_effective_health':
        return min(choices, key=lambda t: (choices[t]['effective_health'], t))
    if rule == 'highest_threat':
        known = [t for t in choices if choices[t]['threat_known']]
        return min(known, key=lambda t: (-choices[t]['threat_volley'],
                   choices[t]['effective_health'], t)) if known else None
    if rule == 'easiest_focus':
        return min(choices, key=lambda t: (-choices[t]['attackable_fraction'],
                   -choices[t]['ready_fraction'],
                   0 if choices[t]['distance_known_fraction'] == 1. else 1,
                   choices[t]['mean_distance'], choices[t]['effective_health'], t))
    raise ValueError('unknown semantic target rule: '+str(rule))
