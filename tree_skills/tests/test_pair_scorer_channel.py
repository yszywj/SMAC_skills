"""Channel tests: default-off identity, A-plug equivalence, shadow identity,
fixed-participation legality, failure fallback, deterministic tie handling.

These use synthetic build_request snapshots (same shape as
skills/current/tests fixtures) — no SC2, no teacher, no training deps.
"""

import copy
import random
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from skills.current.skills import heal_assign
from tree_skills.medivac.pair_scorer import OriginalPairScorer


def unit(uid, health, max_health, type_="marine", pos=(0.0, 0.0), **extra):
    base = {"health": health, "max_health": max_health, "type": type_,
            "alive": True, "is_flying": type_ == "medivac",
            "unit_type_id": 54 if type_ == "medivac" else 0,
            "position": list(pos)}
    base.update(extra)
    return base


def make_state(seed=0):
    rng = random.Random(seed)
    allies = {
        "u0": unit("u0", 30, 45, pos=(rng.random() * 5, rng.random() * 5),
                   health_lost_since_previous_decision=8),
        "u1": unit("u1", 40, 45, pos=(rng.random() * 5, rng.random() * 5),
                   health_lost_since_previous_decision=4),
        "u2": unit("u2", 25, 45, pos=(rng.random() * 5, rng.random() * 5),
                   health_lost_since_previous_decision=15),
        "u3": unit("u3", 45, 45, pos=(rng.random() * 5, rng.random() * 5)),
        "u4": unit("u4", 150, 150, "medivac", pos=(2.0, 3.0), energy=180),
        "u5": unit("u5", 140, 150, "medivac", pos=(4.0, 1.0), energy=120),
    }
    state = {"allies": allies,
             "visible_enemies": {"e0": {"position": [12.0, 12.0], "health": 45,
                                        "shield": 0, "max_health": 45,
                                        "is_flying": False}},
             "relations": {a: {"e0": {"distance": 10 + rng.random() * 5,
                                      "in_enemy_weapon_range": False,
                                      "smac_attack_available": True,
                                      "in_weapon_range": False}}
                           for a in allies},
             "heal_candidates": {}, "movement_candidates": {}}
    # heal_candidates[healer][action_id] = {target, distance, missing_health}
    for healer, hi in (("u4", 0), ("u5", 1)):
        entries = {}
        for j, target in enumerate(("u0", "u1", "u2")):
            entries[str(6 + j)] = {"target": target,
                                   "distance": 2.0 + rng.random() * 4,
                                   "missing_health": allies[target]["max_health"] - allies[target]["health"]}
        state["heal_candidates"][healer] = entries
        state["movement_candidates"][healer] = {
            str(k): {"destination": [rng.random(), rng.random()],
                     "distances_to_visible_enemies": {"e0": 11.0}} for k in range(4)}
    return state


class TestChannel(unittest.TestCase):
    def test_default_off_untouched(self):
        state = make_state(1)
        base = heal_assign.assign_healing(copy.deepcopy(state), 6)
        again = heal_assign.assign_healing(copy.deepcopy(state), 6,
                                           pair_scorer=None, pair_scoring="off")
        self.assertEqual(base["assignments"], again["assignments"])
        self.assertNotIn("pair_scoring", again)

    def test_a_plug_equivalence_random(self):
        """Original scorer through the new channel must reproduce the
        original matching exactly, including deterministic ties."""
        for seed in range(60):
            state = make_state(seed)
            base = heal_assign.assign_healing(copy.deepcopy(state), 6)
            plugged = heal_assign.assign_healing(
                copy.deepcopy(state), 6, pair_scorer=OriginalPairScorer(),
                pair_scoring="on")
            self.assertEqual(base["assignments"], plugged["assignments"],
                             f"seed={seed}: {base['assignments']} vs "
                             f"{plugged['assignments']}")
            self.assertEqual(base["healer_actions"], plugged["healer_actions"])
            diag = plugged["pair_scoring"]
            self.assertEqual(diag["changed"], {})
            self.assertEqual(diag["mode"], "active")

    def test_shadow_does_not_change_actions(self):
        state = make_state(2)
        base = heal_assign.assign_healing(copy.deepcopy(state), 6)

        def weird(healer, target, context):
            return -100.0 if target == "u0" else 50.0  # all-negative allowed

        shadow = heal_assign.assign_healing(copy.deepcopy(state), 6,
                                            pair_scorer=weird,
                                            pair_scoring="shadow")
        self.assertEqual(base["assignments"], shadow["assignments"])
        self.assertEqual(base["healer_actions"], shadow["healer_actions"])
        self.assertEqual(shadow["pair_scoring"]["mode"], "shadow")

    def test_active_rescoring_changes_only_participation_set(self):
        # Deterministic direct test of the rescore contract on crafted edges.
        edges = {"u4": [(3.0, "u0"), (2.0, "u1"), (2.5, "u2")],
                 "u5": [(2.2, "u0"), (1.0, "u1"), (2.4, "u2")]}
        original = {"u4": "u0", "u5": "u2"}  # optimal under original scores
        state = make_state(3)

        def flip(healer, target, context):
            return 10.0 if target == "u1" else 1.0

        new, diag = heal_assign._apply_pair_scorer(
            state, original, edges, {}, {}, None, flip, active=True)
        self.assertEqual(set(new), set(original), "participation set changed")
        self.assertEqual(sorted(new.values()), sorted(set(new.values())),
                         "one-to-one violated")
        self.assertEqual(new, {"u4": "u1", "u5": "u0"},
                         "top-preferred u1 plus next best, not both on u1")
        self.assertEqual(diag["model_influence_rate"], 1.0)

    def test_all_negative_scores_still_assign(self):
        state = make_state(4)
        base = heal_assign.assign_healing(copy.deepcopy(state), 6)
        self.assertTrue(base["assignments"])

        def all_negative(healer, target, context):
            return -5.0

        rescored = heal_assign.assign_healing(copy.deepcopy(state), 6,
                                              pair_scorer=all_negative,
                                              pair_scoring="on")
        # fixed participation: every original participant still heals someone
        self.assertEqual(set(rescored["assignments"]), set(base["assignments"]))
        self.assertTrue(rescored["assignments"])

    def test_scorer_failure_keeps_original(self):
        state = make_state(5)

        def broken(healer, target, context):
            raise RuntimeError("boom")

        result = heal_assign.assign_healing(copy.deepcopy(state), 6,
                                            pair_scorer=broken,
                                            pair_scoring="on")
        base = heal_assign.assign_healing(copy.deepcopy(state), 6)
        self.assertEqual(base["assignments"], result["assignments"])
        self.assertTrue(result["pair_scoring"]["failures"])
        self.assertEqual(result["pair_scoring"]["changed"], {})

    def test_non_finite_score_rejected(self):
        state = make_state(6)

        def nan_scorer(healer, target, context):
            return float("nan") if target == "u0" else 1.0

        result = heal_assign.assign_healing(copy.deepcopy(state), 6,
                                            pair_scorer=nan_scorer,
                                            pair_scoring="on")
        base = heal_assign.assign_healing(copy.deepcopy(state), 6)
        self.assertEqual(base["assignments"], result["assignments"])
        self.assertTrue(result["pair_scoring"]["failures"])

    def test_no_reopening_of_illegal_targets(self):
        state = make_state(7)
        # u3 is full-health: never a legal candidate, scorer must not add it
        def sneaky(healer, target, context):
            return 99.0 if target == "u3" else 1.0

        result = heal_assign.assign_healing(copy.deepcopy(state), 6,
                                            pair_scorer=sneaky,
                                            pair_scoring="on")
        self.assertNotIn("u3", result["assignments"].values())

    def test_terran_combat_threading_off(self):
        from skills.current.skills.terran_combat import allocate_terran_combat
        # Reuse the project's own terran fixture shape (test_terran_combat).
        import math

        def tunit(kind, pos, health=None, energy=50):
            identity, maximum, flying = {'Marine': (48, 45, False),
                                         'Marauder': (51, 125, False),
                                         'Medivac': (54, 150, True)}[kind]
            return dict(tag=str(identity), type=kind, unit_type_id=identity,
                        alive=True, position=list(pos), radius=.5,
                        health=maximum if health is None else health,
                        max_health=maximum, shield=0, max_shield=0,
                        is_flying=flying, weapon_ready=True,
                        weapon_cooldown_loops=0, energy=energy, orders=[],
                        health_lost_since_previous_decision=0,
                        shield_lost_since_previous_decision=0)

        def tscene(units, enemies):
            profiles = {
                '48': dict(known=True, armor=0, movement_speed_native=4,
                           weapons=[dict(target_type='any', range=5, base_damage=6, hits_per_attack=1)]),
                '51': dict(known=True, armor=1, movement_speed_native=4,
                           weapons=[dict(target_type='ground', range=6, base_damage=10, hits_per_attack=1)]),
                '54': dict(known=True, armor=1, movement_speed_native=4, weapons=[])}
            allies = {f'u{i}': u for i, u in enumerate(units)}
            state = dict(allies=allies, visible_enemies=enemies, game_loop=8,
                         step_mul=8, steps_remaining=100, unit_type_profiles=profiles,
                         relations={}, movement_candidates={}, heal_candidates={})
            for actor, u in allies.items():
                state['relations'][actor] = {}
                for enemy, e in enemies.items():
                    distance = math.dist(u['position'], e['position'])
                    own = u['type'] != 'Medivac' and not (u['type'] == 'Marauder' and e['is_flying'])
                    foe = e['type'] != 'Medivac' and not (e['type'] == 'Marauder' and u['is_flying'])
                    state['relations'][actor][enemy] = dict(
                        distance=distance, smac_attack_available=own and distance <= 6,
                        native_weapon_range=5, smac_attack_distance_limit=6,
                        in_weapon_range=own and distance <= 6,
                        weapon_range_status=('in_range' if distance <= 6 else 'out_of_range') if own else 'no_compatible_weapon',
                        enemy_weapon_range_status=('in_range' if distance <= 6 else 'out_of_range') if foe else 'no_compatible_weapon',
                        in_enemy_weapon_range=foe and distance <= 6)
                state['movement_candidates'][actor] = {}
                for action, (dx, dy) in {'2': (0, 1), '3': (0, -1), '4': (1, 0), '5': (-1, 0)}.items():
                    dest = [u['position'][0] + dx, u['position'][1] + dy]
                    state['movement_candidates'][actor][action] = dict(
                        destination=dest, distance_to_search_location=0,
                        distances_to_visible_enemies={e: math.dist(dest, t['position']) for e, t in enemies.items()})
                if u['type'] == 'Medivac':
                    state['heal_candidates'][actor] = {
                        str(6 + int(t[1:])): dict(target=t, distance=math.dist(u['position'], v['position']),
                                                  missing_health=v['max_health'] - v['health'])
                        for t, v in allies.items()
                        if v['type'] != 'Medivac' and v['health'] < v['max_health']
                        and math.dist(u['position'], v['position']) <= 4}
            return state

        enemy = dict(tag='48', type='Marine', unit_type_id=48, alive=True,
                     position=[14.0, 0.0], radius=.5, health=45, max_health=45,
                     shield=0, max_shield=0, is_flying=False, weapon_ready=True,
                     weapon_cooldown_loops=0, energy=50, orders=[])
        state = tscene([tunit('Marine', (0, 0), 20), tunit('Marine', (1, 0), 35),
                        tunit('Medivac', (0.5, 1.0)), tunit('Medivac', (1.5, 1.0))],
                       {'e0': enemy})
        base = allocate_terran_combat(copy.deepcopy(state), 4)
        self.assertNotIn("pair_scoring", base["race_telemetry"]["healing"])
        plugged = allocate_terran_combat(copy.deepcopy(state), 4,
                                         pair_scorer=OriginalPairScorer(),
                                         pair_scoring="on")
        self.assertEqual(base["race_telemetry"]["healing"]["assignments"],
                         plugged["race_telemetry"]["healing"]["assignments"])
        self.assertIn("pair_scoring", plugged["race_telemetry"]["healing"])
        # full final action map must be identical under A-plug
        self.assertEqual(base["actions"], plugged["actions"])


if __name__ == "__main__":
    unittest.main()
