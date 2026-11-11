"""Shared offline fixtures for tactical_v1 tests.

Purely semantic state dicts in the adapter's payload schema
(skills/current/tests/test_group_skills.py format extended with the fields
tactical_v1 reads). The world model is stated explicitly: a move realizes
its candidate destination unless blocked; nothing else changes.
"""
import copy
import unittest

from skills.tactical_v1.skill_manager import SkillManager

MOVE_DELTAS = {2: (0., 1.), 3: (0., -1.), 4: (1., 0.), 5: (-1., 0.)}
STOP = 1
DEAD = 0

UNIT_IDS = {"marine": 48, "marauder": 51, "medivac": 54, "zealot": 73,
            "stalker": 74, "colossus": 4, "zergling": 105, "baneling": 9,
            "hydralisk": 107}

# Minimal catalog-shaped weapon profiles (bridge combat_profile schema) so the
# shared calibrated damage model produces real estimates in offline fixtures.
GROUND, AIR = "ground", "air"
WEAPONS = {
    "marine": (6., 1, GROUND, 5.), "marauder": (10., 1, GROUND, 6.),
    "stalker": (13., 1, GROUND, 6.), "hydralisk": (12., 1, GROUND, 5.),
    "zealot": (8., 2, GROUND, 2.), "zergling": (5., 1, GROUND, 2.),
    "colossus": (20., 2, GROUND, 7.), "baneling": (16., 1, GROUND, 2.),
}


def type_profiles():
    profiles = {}
    for name, (damage, hits, target, reach) in WEAPONS.items():
        profiles[str(UNIT_IDS[name])] = {
            "known": True, "movement_speed_native": 2.25, "armor": 0.,
            "attributes": [1], "weapons": [
                {"target_type": target, "range": reach,
                 "base_damage": damage, "hits_per_attack": hits,
                 "damage_bonus": []}],
            "attack_upgrade_level": 0., "armor_upgrade_level": 0.}
    return profiles


def ally(position, name="marine", **extra):
    unit = {"tag": "ally", "alive": True, "type": name,
            "unit_type_id": UNIT_IDS[name], "position": list(position),
            "radius": .5, "is_flying": name == "medivac",
            "health": 45., "shield": 0., "max_health": 45., "max_shield": 0.,
            "weapon_cooldown_loops": 0., "weapon_ready": True,
            "effective_health": 45., "energy": None,
            "attack_upgrade_level": 0., "armor_upgrade_level": 0.}
    unit.update(extra)
    return unit


def enemy(position, name="zergling", **extra):
    unit = {"position": list(position), "radius": .5, "alive": True,
            "unit_type_id": UNIT_IDS[name], "is_flying": False, "type": name,
            "health": 35., "shield": 0., "max_health": 35., "max_shield": 0.,
            "attack_upgrade_level": 0., "armor_upgrade_level": 0.}
    unit.update(extra)
    return unit


def scene(allies, enemies=None, loop=80, step_mul=8, episode=1,
          distance=1.0, attack_range=5.0, map_name="10gen_terran",
          enemy_can_reach=True, search=(16., 16.)):
    """allies: list of (position, name) or dicts; enemies: dict e{i}->unit."""
    enemies = enemies or {}
    if allies and not isinstance(allies[0], (tuple, dict)):
        raise ValueError("allies must be (position, name) tuples or dicts")
    units = {}
    for i, item in enumerate(allies):
        unit = dict(item[1]) if isinstance(item, dict) and "position" in item else None
        if unit is None:
            position, name = item
            units[f"u{i}"] = ally(position, name)
        else:
            units[f"u{i}"] = unit
    state = {"allies": units,
             "visible_enemies": enemies, "game_loop": loop, "step_mul": step_mul,
             "episode": episode, "objective": "win", "map": {"name": map_name},
             "steps_remaining": 100, "search_location": list(search),
             "relations": {}, "movement_candidates": {}, "heal_candidates": {},
             "unit_type_profiles": type_profiles(),
             "action_contract": {
                 "candidate_center_distance_by_type": {
                     "marine": 5, "marauder": 6, "medivac": 4, "zealot": 2,
                     "stalker": 6, "colossus": 7, "zergling": 2, "baneling": 2,
                     "hydralisk": 5}},
             "previous_action_ids": {}}
    for actor, unit in units.items():
        state["relations"][actor] = {}
        for enemy_id, target in enemies.items():
            d = _dist(unit["position"], target["position"])
            in_range = d <= attack_range
            state["relations"][actor][enemy_id] = {
                "distance": round(d, 2), "in_weapon_range": in_range,
                "weapon_range_status": "in_range" if in_range else "out_of_range",
                "native_weapon_range": attack_range,
                "in_enemy_weapon_range": in_range and enemy_can_reach,
                "enemy_weapon_range_status":
                    "in_range" if (in_range and enemy_can_reach)
                    else ("no_compatible_weapon" if not enemy_can_reach
                          else "out_of_range"),
                "smac_attack_available": in_range,
                "smac_attack_distance_limit": attack_range + .5}
        state["movement_candidates"][actor] = {
            action: {"destination": [unit["position"][0] + dx * distance,
                                     unit["position"][1] + dy * distance],
                     "distances_to_visible_enemies": {
                         e: round(_dist([unit["position"][0] + dx * distance,
                                         unit["position"][1] + dy * distance],
                                        t["position"]), 2)
                         for e, t in enemies.items()}}
            for action, (dx, dy) in MOVE_DELTAS.items()}
    return state


def with_questions(state):
    """Build the per-actor legal-action menu the runner would supply."""
    questions = {}
    for actor, unit in state["allies"].items():
        if not unit["alive"]:
            continue
        criteria = {str(a): "ok" for a in (STOP, *MOVE_DELTAS)}
        for enemy_id in state["visible_enemies"]:
            if state["relations"][actor][enemy_id]["smac_attack_available"]:
                criteria[str(6 + int(enemy_id[1:]))] = "attack"
        for action, info in state.get("heal_candidates", {}).get(actor, {}).items():
            criteria[str(action)] = "heal"
        questions[actor] = {"type": "choice", "instructions": "", "criteria": criteria}
    return {"state": state, "questions": questions}


def payload(state):
    return {"state": state, "questions": {}}


def heal_candidates(state, healer, max_distance=4.0):
    candidates = {}
    for actor, unit in state["allies"].items():
        if actor == healer or unit.get("unit_type_id") == 54:
            continue
        if not (0 < unit["health"] < unit["max_health"]):
            continue
        d = _dist(state["allies"][healer]["position"], unit["position"])
        if d <= max_distance:
            candidates[str(6 + int(actor[1:]))] = {
                "target": actor,
                "missing_health": round(unit["max_health"] - unit["health"], 2),
                "distance": round(d, 2)}
    state["heal_candidates"][healer] = candidates
    return candidates


def advance(state, actions, distance=1.0, blocked=()):
    for actor, action in (actions or {}).items():
        if int(action) in MOVE_DELTAS and int(action) not in blocked:
            dx, dy = MOVE_DELTAS[int(action)]
            p = state["allies"][actor]["position"]
            state["allies"][actor]["position"] = [p[0] + dx * distance, p[1] + dy * distance]
    state["game_loop"] += state["step_mul"]
    for actor, unit in state["allies"].items():
        p = unit["position"]
        state["movement_candidates"][actor] = {
            action: {"destination": [p[0] + dx * distance, p[1] + dy * distance],
                     "distances_to_visible_enemies": {
                         e: round(_dist([p[0] + dx * distance, p[1] + dy * distance],
                                        t["position"]), 2)
                         for e, t in state["visible_enemies"].items()}}
            for action, (dx, dy) in MOVE_DELTAS.items()}


def _dist(a, b):
    import math
    return math.hypot(a[0] - b[0], a[1] - b[1])


class TacticalTestCase(unittest.TestCase):
    def manager(self, **config_overrides):
        manager = SkillManager()
        if config_overrides:
            from skills.tactical_v1.skill_manager import _deep_merge
            _deep_merge(manager.config, config_overrides)
            manager.config["_fingerprint"] = "test"
            manager.skills = __import__(
                "skills.tactical_v1.registry", fromlist=["build_skills"]).build_skills(
                manager.config)
            manager.coordinator = __import__(
                "skills.tactical_v1.coordinator", fromlist=["Coordinator"]).Coordinator(
                manager.config, manager.evaluator)
        return manager

    def step(self, manager, state, n_agents=None):
        n_agents = n_agents or len(state["allies"])
        actions, telemetry = manager.step(with_questions(copy.deepcopy(state)), n_agents)
        return actions, telemetry

    def skills_of(self, telemetry):
        return {actor: info["skill"] for actor, info in telemetry["units"].items()}
