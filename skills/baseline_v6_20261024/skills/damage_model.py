"""Opt-in catalog damage estimates; no simulation, hidden state or splash credit.

The unshielded formula is independently calibrated against the recorded native
Stalker 54 -> 37 HP transition. SC2's public protocol exposes attributes, bonus
damage and hits, but not the full effect/shield spillover pipeline. Crossing a
shield boundary therefore produces an explicit conservative interval. These are
potential single-attack estimates, not guaranteed damage or projectile tracking.
"""
import math

VERSION = "native-catalog-damage-v1"


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def _unknown(reason):
    return {"version": VERSION, "estimate": None, "lower_bound": None,
            "upper_bound": None, "status": "unknown", "reasons": [reason],
            "scope": "potential_single_target_attack; no_splash_or_projectile_credit"}


def expected_attack_damage_details(state, actor, enemy_id):
    """Return JSON-safe damage evidence; unknown inputs never become known zero."""
    attacker = state.get("allies", {}).get(actor, {})
    target = state.get("visible_enemies", {}).get(enemy_id, {})
    profiles = state.get("unit_type_profiles", {})
    own = profiles.get(str(attacker.get("unit_type_id")), {})
    other = profiles.get(str(target.get("unit_type_id")), {})
    if not own.get("known") or not other.get("known"):
        return _unknown("missing_catalog_profile")
    if not isinstance(target.get("is_flying"), bool):
        return _unknown("unknown_target_weapon_domain")
    kinds = {"air", "any"} if target["is_flying"] else {"ground", "any"}
    if target.get("unit_type_id") == 4:
        kinds.add("air")
    weapons = [w for w in own.get("weapons", []) if w.get("target_type") in kinds]
    if len(weapons) != 1:
        return _unknown("no_unique_compatible_catalog_weapon")
    weapon = weapons[0]
    base, hits = _number(weapon.get("base_damage")), _number(weapon.get("hits_per_attack"))
    armor, shield = _number(other.get("armor")), _number(target.get("shield"))
    if base is None or base < 0 or hits is None or hits < 1 or hits != int(hits):
        return _unknown("unknown_or_invalid_base_damage_or_hits")
    if armor is None or armor < 0 or shield is None or shield < 0:
        return _unknown("unknown_or_unsupported_target_armor_or_shield")
    for unit, field in ((attacker, "attack_upgrade_level"), (target, "armor_upgrade_level")):
        level = _number(unit.get(field))
        if level is None:
            return _unknown("unknown_" + field)
        if level != 0:
            return _unknown("unsupported_nonzero_" + field)
    bonuses = weapon.get("damage_bonus")
    attributes = other.get("attributes")
    if not isinstance(bonuses, list):
        return _unknown("unknown_weapon_damage_bonus")
    if not isinstance(attributes, list) or any(
            not isinstance(attribute, int) or isinstance(attribute, bool) for attribute in attributes):
        return _unknown("unknown_target_attributes")
    matched = []
    for entry in bonuses:
        if not isinstance(entry, dict):
            return _unknown("invalid_weapon_damage_bonus")
        bonus = _number(entry.get("bonus"))
        attribute = entry.get("attribute")
        if bonus is None or bonus < 0 or not isinstance(attribute, int) or isinstance(attribute, bool):
            return _unknown("invalid_weapon_damage_bonus")
        if attribute in attributes:
            matched.append(bonus)
    if len(matched) > 1:
        return _unknown("multiple_matching_bonus_combination_unsupported")
    bonus = matched[0] if matched else 0.0
    raw = base + bonus
    if not math.isfinite(raw):
        return _unknown("nonfinite_combined_damage")
    hull_hit = max(raw - armor, 0.5)
    lower = upper = hits * hull_hit
    status, reasons = "catalog_exact_unshielded", []
    shield_armor = None
    if shield > 0:
        level = _number(target.get("shield_upgrade_level"))
        shield_armor = _number(other.get("shield_armor_base"))
        if level is None:
            return _unknown("unknown_shield_upgrade_level")
        if level != 0:
            return _unknown("unsupported_nonzero_shield_upgrade_level")
        if shield_armor is None or shield_armor < 0:
            return _unknown("unknown_shield_armor")
        shield_hit = max(raw - shield_armor, 0.5)
        if shield >= hits * shield_hit:
            lower = upper = hits * shield_hit
            status = "catalog_shield_only_estimate"
            reasons = ["shield_armor_source=" + str(other.get("shield_armor_source", "provided"))]
        else:
            # Allow either/both armor layers to apply at spillover; do not assert
            # an unverified exact transfer formula. This deliberately credits
            # the conservative lower envelope for the focus budget.
            lower = hits * max(raw - armor - shield_armor, 0.5)
            # A crossing hit can encounter both minimum-damage floors.
            upper = hits * max(raw - min(armor, shield_armor), 1.0)
            status = "bounded_shield_transition"
            reasons = ["shield_spillover_formula_not_assumed_exact",
                       "shield_armor_source=" + str(other.get("shield_armor_source", "provided"))]
    if not math.isfinite(lower) or not math.isfinite(upper):
        return _unknown("nonfinite_attack_damage")
    return {"version": VERSION, "estimate": round(lower, 6),
            "lower_bound": round(lower, 6), "upper_bound": round(upper, 6),
            "status": status, "reasons": reasons, "raw_damage_per_hit": raw,
            "matching_attribute_bonus": bonus, "hits_per_attack": int(hits),
            "health_armor": armor, "shield_armor": shield_armor,
            "scope": "potential_single_target_attack; zero_upgrades; no_splash_or_projectile_credit",
            "assumptions": "catalog_damage_only; no_unmodeled_buffs_or_damage_modifiers"}
