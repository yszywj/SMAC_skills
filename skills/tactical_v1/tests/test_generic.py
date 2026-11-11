"""C. Generic combat & spatial organization: six normal unit types across
G1/G2/G3/G4/G5/G6/G7 behaviours."""
import copy
import unittest

from .fixtures import (TacticalTestCase, MOVE_DELTAS, STOP, advance, ally,
                       enemy, scene, with_questions)


class NormalCombatTests(TacticalTestCase):
    def test_six_normal_types_engage_fight_retreat_and_return(self):
        manager = self.manager()
        names = ["marine", "marauder", "stalker", "hydralisk", "zealot", "zergling"]
        # Phase 1: entire column out of range -> coordinated approach.
        state = scene([((0., i), name) for i, name in enumerate(names)],
                      enemies={"e0": enemy((8., 3.))})
        actions, telemetry = self.step(manager, state)
        self.assertEqual(set(self.skills_of(telemetry).values()),
                         {"coordinated_engage"})
        # Phase 2: moved into range -> focus fire for everyone.
        state = scene([((6., i), name) for i, name in enumerate(names)],
                      enemies={"e0": enemy((8., 3.))})
        actions, telemetry = self.step(manager, state)
        self.assertEqual(set(self.skills_of(telemetry).values()), {"focus_fire"})
        self.assertTrue(all(a == 6 for a in actions))
        # Phase 3: critical front unit under fire -> disengage competes and wins.
        state = scene([((6., i), name) for i, name in enumerate(names)],
                      enemies={"e0": enemy((8., 3.)), "e1": enemy((7., 0.))})
        state["allies"]["u0"]["health"] = 5.
        state["allies"]["u0"]["health_lost_since_previous_decision"] = 10.
        state["allies"]["u0"]["weapon_cooldown_loops"] = 16.
        actions, telemetry = self.step(manager, state)
        self.assertIn(telemetry["units"]["u0"]["skill"],
                      ("disengage_reenter", "combat_reposition"))
        self.assertIn(actions[0], MOVE_DELTAS)
        # Phase 4: threat gone and health restored -> back to combat.
        state = scene([((6., i), name) for i, name in enumerate(names)],
                      enemies={"e0": enemy((8., 3.))})
        actions, telemetry = self.step(manager, state)
        self.assertEqual(telemetry["units"]["u0"]["skill"], "focus_fire")

    def test_front_fight_and_rear_regroup_coexist(self):
        state = scene([((8., 0.), "marine"), ((8., 1.), "marine"),
                       ((0., 20.), "marine")],
                      enemies={"e0": enemy((11., 0.))})
        actions, telemetry = self.step(self.manager(), state)
        skills = self.skills_of(telemetry)
        self.assertEqual(skills["u0"], "focus_fire")
        self.assertEqual(skills["u1"], "focus_fire")
        self.assertEqual(skills["u2"], "regroup")

    def test_melee_beyond_contact_not_counted_as_firing(self):
        # A melee unit out of range must not produce a focus_fire plan.
        state = scene([((0., 0.), "zealot")], enemies={"e0": enemy((9., 9.))})
        _, telemetry = self.step(self.manager(), state)
        self.assertEqual(telemetry["units"]["u0"]["skill"], "coordinated_engage")

    def test_cooldown_zero_does_not_force_movement(self):
        state = scene([((0., 0.), "marine")], enemies={"e0": enemy((3., 0.))})
        state["allies"]["u0"]["weapon_cooldown_loops"] = 0.
        actions, telemetry = self.step(self.manager(), state)
        self.assertEqual(actions[0], 6)   # attack, not a reflex move

    def test_long_cooldown_under_fire_allows_reposition(self):
        # Critical marine, long cooldown, TWO threats: the exposure risk of
        # attacking outweighs the foregone damage -> repositioning wins.
        state = scene([((0., 0.), "marine")],
                      enemies={"e0": enemy((3., 0.)), "e1": enemy((0., 3.))})
        state["allies"]["u0"]["weapon_cooldown_loops"] = 16.
        state["allies"]["u0"]["health"] = 12.
        state["allies"]["u0"]["health_lost_since_previous_decision"] = 8.
        actions, telemetry = self.step(self.manager(), state)
        self.assertIn(telemetry["units"]["u0"]["skill"],
                      ("combat_reposition", "disengage_reenter"))

    def test_single_threat_does_not_force_retreat(self):
        # One attacker, healthy-enough unit: risk competes but firing wins.
        state = scene([((0., 0.), "marine")], enemies={"e0": enemy((3., 0.))})
        state["allies"]["u0"]["weapon_cooldown_loops"] = 16.
        state["allies"]["u0"]["health"] = 40.
        state["allies"]["u0"]["health_lost_since_previous_decision"] = 4.
        actions, telemetry = self.step(self.manager(), state)
        self.assertEqual(actions[0], 6)

    def test_melee_does_not_blindly_kite(self):
        state = scene([((0., 0.), "zealot")], enemies={"e0": enemy((1., 0.))},
                      attack_range=2.0)
        state["allies"]["u0"]["weapon_cooldown_loops"] = 16.
        state["allies"]["u0"]["health_lost_since_previous_decision"] = 4.
        actions, telemetry = self.step(self.manager(), state)
        # In-contact melee with no improving shuffle keeps attacking.
        self.assertEqual(actions[0], 6)

    def test_no_visible_enemy_activates_search_not_attack(self):
        state = scene([((0., 0.), "marine"), ((1., 0.), "marine")])
        actions, telemetry = self.step(self.manager(), state)
        self.assertEqual(set(self.skills_of(telemetry).values()),
                         {"cohesive_search"})
        self.assertTrue(all(a in MOVE_DELTAS or a == STOP for a in actions))

    def test_stale_enemy_position_is_clue_not_target(self):
        from skills.tactical_v1.skills.cohesive_search import CohesiveSearchSkill
        from skills.tactical_v1.context import build_snapshot, Context
        from skills.tactical_v1.memory import TacticalMemory
        manager = self.manager()
        skill = CohesiveSearchSkill(manager.config)
        # The enemy was seen long ago; the clue drives the waypoint choice...
        clue_state = scene([((0., 0.), "marine")], loop=80,
                           enemies={"e0": enemy((12., 12.))})
        clue_snap = build_snapshot(with_questions(clue_state), 1)
        memory = TacticalMemory()
        memory.begin_step(clue_snap)
        memory.observe(clue_snap)
        # ...but is expired after the TTL: the waypoint falls back to the
        # map search location and is never presented as an attack goal.
        hidden = scene([((0., 0.), "marine")], loop=400)
        hidden_snap = build_snapshot(with_questions(hidden), 2)
        memory.begin_step(hidden_snap)
        memory.observe(hidden_snap)
        plans = skill.propose(Context(hidden_snap), memory, manager.telemetry)
        self.assertTrue(plans)
        self.assertTrue(all(p.goal != "e0" for p in plans))
        fresh_state = scene([((0., 0.), "marine")], loop=88,
                            enemies={"e0": enemy((12., 12.))})
        fresh_snap = build_snapshot(with_questions(fresh_state), 3)
        memory2 = TacticalMemory()
        memory2.begin_step(fresh_snap)
        memory2.observe(fresh_snap)
        fresh_snap.enemies = {}   # hidden again immediately: clue still fresh
        plans_fresh = skill.propose(Context(fresh_snap), memory2, manager.telemetry)
        self.assertTrue(all(p.goal != "e0" for p in plans_fresh))


class FormationTests(TacticalTestCase):
    def test_splash_threat_triggers_spread_candidate(self):
        # Clumped marines under an out-of-range colossus: spreading beats
        # approaching because the splash exposure proxy dominates.
        state = scene([((5., 5.), "marine"), ((5.3, 5.), "marine")],
                      enemies={"e0": enemy((9., 5.), "colossus")},
                      attack_range=3.0)
        _, telemetry = self.step(self.manager(), state)
        skills = self.skills_of(telemetry)
        self.assertIn("local_formation", skills.values())

    def test_formation_does_not_override_attack_unconditionally(self):
        # In range with a kill available: attack stays selectable; formation
        # only competes.
        state = scene([((5., 5.), "marine"), ((5.3, 5.), "marine")],
                      enemies={"e0": enemy((6., 5.), "colossus")})
        _, telemetry = self.step(self.manager(), state)
        self.assertIn("focus_fire", self.skills_of(telemetry).values())


if __name__ == "__main__":
    unittest.main()
