"""Small semantic snapshots; no SC2 process or training stack is needed."""
import math


def unit(kind, position=(0, 0), health=None, energy=50):
    kind_id, maximum, flying = {
        "Marine": (48, 45, False),
        "Marauder": (51, 125, False),
        "Medivac": (54, 150, True),
    }[kind]
    return {
        "type": kind, "unit_type_id": kind_id, "alive": True,
        "position": list(position), "health": maximum if health is None else health,
        "max_health": maximum, "shield": 0, "max_shield": 0, "energy": energy,
        "is_flying": flying, "radius": .5, "orders": [],
        "weapon_ready": True, "weapon_cooldown_loops": 0,
        "health_lost_since_previous_decision": 0,
        "shield_lost_since_previous_decision": 0,
    }


def scene(units, enemies=None, *, episode=1, game_loop=8):
    allies = {f"u{i}": row for i, row in enumerate(units)}
    for i, row in enumerate(units):
        row["tag"] = str(1000 + i)
    enemies = enemies or {}
    state = {
        "allies": allies, "visible_enemies": enemies, "episode": episode,
        "game_loop": game_loop, "step_mul": 8, "steps_remaining": 100,
        "action_contract": {}, "relations": {}, "movement_candidates": {},
        "heal_candidates": {}, "legal_actions": {},
        "unit_type_profiles": {
            "48": {"known": True, "armor": 0, "movement_speed_native": 4,
                   "weapons": [{"target_type": "any", "range": 5,
                                "base_damage": 6, "hits_per_attack": 1}]},
            "51": {"known": True, "armor": 1, "movement_speed_native": 4,
                   "weapons": [{"target_type": "ground", "range": 6,
                                "base_damage": 10, "hits_per_attack": 1}]},
            "54": {"known": True, "armor": 1, "movement_speed_native": 4,
                   "weapons": []},
        },
    }
    for actor, row in allies.items():
        state["relations"][actor] = {}
        for enemy, other in enemies.items():
            distance = math.dist(row["position"], other["position"])
            compatible = other["type"] == "Marine" or (
                other["type"] == "Marauder" and not row["is_flying"])
            state["relations"][actor][enemy] = {
                "distance": distance, "in_enemy_weapon_range": compatible and distance <= 6,
                "enemy_weapon_range_status": (
                    "in_range" if distance <= 6 else "out_of_range") if compatible
                    else "no_compatible_weapon",
                "smac_attack_available": False, "in_weapon_range": False,
                "weapon_range_status": "no_compatible_weapon", "native_weapon_range": 0,
                "smac_attack_distance_limit": 6,
            }
        moves = {}
        for action, (dx, dy) in {2: (0, 1), 3: (0, -1), 4: (1, 0), 5: (-1, 0)}.items():
            point = [row["position"][0] + dx, row["position"][1] + dy]
            moves[str(action)] = {
                "destination": point, "distance_to_search_location": 0,
                "distances_to_visible_enemies": {
                    enemy: math.dist(point, other["position"])
                    for enemy, other in enemies.items()},
            }
        state["movement_candidates"][actor] = moves
        legal = [1, 2, 3, 4, 5]
        if row["type"] == "Medivac":
            candidates = {}
            for target, patient in allies.items():
                distance = math.dist(row["position"], patient["position"])
                if (patient["type"] != "Medivac" and patient["alive"]
                        and 0 < patient["health"] < patient["max_health"]
                        and row["energy"] >= 1 and distance <= 4):
                    action = 6 + int(target[1:])
                    candidates[str(action)] = {
                        "target": target, "distance": distance,
                        "missing_health": patient["max_health"] - patient["health"],
                    }
                    legal.append(action)
            state["heal_candidates"][actor] = candidates
        if not row["alive"] or row["health"] <= 0:
            legal = [0]
        state["legal_actions"][actor] = legal
    return state


def observe_heal(state, actor, target, *, keep=True):
    """Add an actually observed heal order, optionally with adapter KEEP."""
    state["allies"][actor]["orders"] = [{"ability_id": 386, "target_unit": target}]
    if keep:
        action = 6 + len(state["allies"])
        state["action_contract"]["keep_heal_order"] = {"enabled": True, "action_id": action}
        state.setdefault("keep_heal_order_candidates", {})[actor] = {
            "action_id": action, "target": target}
        state["legal_actions"][actor].append(action)
        return action


def payload(state):
    return {
        "state": state,
        "questions": {
            actor: {"type": "choice", "criteria": {str(a): str(a) for a in legal}}
            for actor, legal in state["legal_actions"].items()
            if state["allies"][actor]["alive"] and state["allies"][actor]["health"] > 0
        },
    }
