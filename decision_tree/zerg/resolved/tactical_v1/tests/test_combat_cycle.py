"""Recorded combat-cycle regressions, without running a game or changing masks."""
import copy
import math
import unittest

from skills.tactical_v1.context import Context, build_snapshot
from skills.tactical_v1.memory import TacticalMemory
from skills.tactical_v1.skill_manager import SkillManager
from skills.tactical_v1.skills.coordinated_engage import CoordinatedEngageSkill
from skills.tactical_v1.skills.combat_reposition import CombatRepositionSkill
from skills.tactical_v1.skills.disengage_reenter import DisengageReenterSkill
from skills.tactical_v1.skills.focus_fire import FocusFireSkill
from skills.tactical_v1.skills.splash_fire import SplashFireSkill
from skills.tactical_v1.telemetry import Telemetry
from .fixtures import enemy, scene, with_questions


class CombatCycleTests(unittest.TestCase):
    def setUp(self):
        self.memory = TacticalMemory()
        self.g2 = CoordinatedEngageSkill({})
        self.g4 = DisengageReenterSkill({})

    def context(self, state, observe=True):
        context = Context(build_snapshot(with_questions(state), state["game_loop"]))
        if observe:
            self.memory.begin_step(context.snapshot)
            self.memory.observe(context.snapshot)
            self.memory.update_task_facts(context.snapshot, context)
        return context

    def plans(self, skill, context):
        before = copy.deepcopy(self.memory.__dict__)
        plans = skill.propose(context, self.memory, Telemetry())
        self.assertEqual(before, self.memory.__dict__, "proposal mutated memory")
        return plans

    def commit(self, context, plan):
        self.memory.commit(context.snapshot, plan.current_actions, [plan], context=context)

    @staticmethod
    def physical_relations(state, weapon_range, command_range):
        for actor, unit in state["allies"].items():
            for target, other in state["visible_enemies"].items():
                gap = math.dist(unit["position"], other["position"])
                native = gap <= weapon_range + unit["radius"] + other["radius"]
                relation = state["relations"][actor][target]
                relation.update(distance=gap, native_weapon_range=weapon_range,
                                in_weapon_range=native,
                                weapon_range_status="in_range" if native else "out_of_range",
                                smac_attack_available=gap <= command_range,
                                smac_attack_distance_limit=command_range)

    def boundary(self, loop, x=25.133, cooldown=0., health=38., lost=0.):
        # seed173 / 6Z4S / ep4 u9: step59 x26.609, step60 x25.133.
        state = scene([((x, 20.358), "stalker")], loop=loop, distance=2.,
                      enemies={"e0": enemy((19.077, 19.836), "stalker", radius=.625),
                               "e1": enemy((18.443, 18.830), "stalker", radius=.625)},
                      attack_range=6.)
        state["allies"]["u0"].update(radius=.625, health=health, max_health=80.,
                                    shield=0., max_shield=80., weapon_cooldown_loops=cooldown,
                                    health_lost_since_previous_decision=lost)
        state["unit_type_profiles"]["74"]["movement_speed_native"] = 2.953125
        self.physical_relations(state, 6., 6.)
        return state

    def approach_boundary(self, lost=0.):
        context = self.context(self.boundary(474, x=26.609))
        plan = next(p for p in self.plans(self.g2, context) if p.current_actions["u0"] == 5)
        self.commit(context, plan)
        return self.context(self.boundary(482, health=38. - lost, lost=lost))

    def test_seed187_new_contact_wins_over_old_blocked_target(self):
        # Actual ep18 step13 u2 and the two relevant enemies, renamed u0.
        state = scene([((20.409, 12.955), "zealot")], loop=106, distance=2.,
                      enemies={"e0": enemy((19.050, 14.443), "zealot"),
                               "e2": enemy((21.382, 13.385), "stalker", radius=.625)})
        state["allies"]["u0"].update(health=100., max_health=100., shield=50., max_shield=50.)
        self.physical_relations(state, .10009765625, 2.)
        context = self.context(state)
        task = self.memory.create_instance("coordinated_engage", ["u0"], "e0",
                                           "approaching", 98, 96, goal_kind="enemy")
        self.memory.move_history["u0"] = [(82, (19.820, 13.108)), (98, (20.145, 12.111))]
        task.extra["approach_history"] = [(98, {"u0": (20.145, 12.111)})]
        self.assertEqual(context.contact_status("u0", "e2"), "in_range")
        self.assertEqual(context.snapshot.allies["u0"].weapon_cooldown_loops, 0.)
        self.assertFalse(self.g2._blocked(context, self.memory, "u0", "e0"))
        self.assertEqual(self.plans(self.g2, context), [])
        self.memory.update_task_facts(context.snapshot, context)
        self.assertNotIn(task.instance_id, self.memory.instances)
        self.assertFalse(self.memory.retry_records)

    def test_blockage_requires_full_committed_same_task_window(self):
        state = scene([((0., 0.), "stalker")], loop=80,
                      enemies={"e0": enemy((12., 0.), "stalker")})
        for loop in (80, 88, 96):
            state["game_loop"] = loop
            context = self.context(state)
            self.assertFalse(self.g2._blocked(context, self.memory, "u0", "e0"))
            self.commit(context, self.plans(self.g2, context)[0])
        state["game_loop"] = 104
        # Inspect before the normal observation-phase stall termination.
        context = self.context(state, observe=False)
        self.assertTrue(self.g2._blocked(context, self.memory, "u0", "e0"))
        self.assertFalse(self.g2._blocked(context, self.memory, "u0", "different"))

    def test_seed173_crosses_native_mask_edge_instead_of_reversing(self):
        context = self.approach_boundary()
        self.assertEqual(context.attackable_enemies("u0"), [])
        self.assertEqual(len(context.threats_to("u0")), 2)
        self.assertTrue(self.memory.cycle_holds(context, "u0", "approach"))
        self.assertEqual(self.plans(self.g4, context), [])
        candidates = self.plans(self.g2, context)
        self.assertTrue(candidates)
        self.assertTrue(all(p.current_actions["u0"] in context.legal_actions("u0") for p in candidates))
        self.assertTrue(any(context.projected_attack_opportunity(
            "u0", p.goal, context.predict_position("u0", p.current_actions["u0"]).position,
            horizon=16).available for p in candidates))
        # Forecasting never grants an illegal current attack command.
        self.assertIsNone(context.attack_action("u0", "e0"))

    def test_g2_does_not_label_direct_retreat_as_approaching(self):
        context = self.context(self.boundary(474, x=26.609))
        actions = {p.current_actions["u0"] for p in self.plans(self.g2, context)}
        self.assertIn(5, actions)
        self.assertNotIn(4, actions)

    def test_full_coordinator_keeps_boundary_crossing_after_real_approach_commit(self):
        manager = SkillManager()
        self.memory = manager.memory
        before = self.context(self.boundary(474, x=26.609))
        approach = next(p for p in self.plans(manager.skills["coordinated_engage"], before)
                        if p.current_actions["u0"] == 5)
        self.commit(before, approach)
        actions, summary = manager.step(with_questions(self.boundary(482)), 1)
        self.assertEqual(actions, [5])
        self.assertEqual(summary["units"]["u0"]["skill"], "coordinated_engage")
        manager.commit_final_actions(actions)

    def test_zero_cycle_commitment_does_not_block_either_direction(self):
        self.memory = TacticalMemory(encounter_config={"lifecycle": {"combat_cycle_commit_h": 0.}})
        # Low remaining reserve plus real pressure warrants an escape when
        # no approach commitment exists; two enemy IDs alone no longer do.
        context = self.approach_boundary(lost=.5)
        self.assertFalse(self.memory.cycle_holds(context, "u0", "approach"))
        escapes = self.plans(self.g4, context)
        self.assertTrue(escapes)
        self.commit(context, escapes[0])
        context = self.context(self.boundary(490, x=26.609, cooldown=16.))
        self.assertFalse(self.memory.cycle_holds(context, "u0", "escape"))
        self.assertTrue(self.plans(self.g2, context))

    def test_boundary_allowance_is_not_renewed_by_repeated_approach(self):
        context = self.approach_boundary()
        until = self.memory.combat_cycles["u0"]["until_loop"]
        self.commit(context, self.plans(self.g2, context)[0])
        context = self.context(self.boundary(490))
        self.commit(context, self.plans(self.g2, context)[0])
        self.assertEqual(self.memory.combat_cycles["u0"]["until_loop"], until)
        context = self.context(self.boundary(until))
        self.assertFalse(self.memory.cycle_holds(context, "u0", "approach"))

    def test_ordinary_chip_keeps_approach_but_severe_damage_interrupts(self):
        self.approach_boundary()
        context = self.context(self.boundary(490, health=37.5, lost=.5))
        self.assertFalse(self.memory.combat_emergency(context, "u0"))
        self.assertEqual(self.plans(self.g4, context), [])
        context = self.context(self.boundary(491, health=10., lost=27.5))
        self.assertTrue(self.memory.combat_emergency(context, "u0"))
        self.assertTrue(self.plans(self.g4, context))

    def test_escape_intent_survives_task_replacement_then_expires(self):
        context = self.context(self.boundary(482, cooldown=24., lost=.5))
        escape = self.plans(self.g4, context)[0]
        self.commit(context, escape)
        until = self.memory.combat_cycles["u0"]["until_loop"]
        context = self.context(self.boundary(490, x=26.609, cooldown=16.))
        # An unrelated stop replaces the task; it must not erase the encounter intent.
        self.memory.commit(context.snapshot, {"u0": 1}, [], context=context)
        self.assertFalse(self.memory.instances)
        self.assertTrue(self.memory.combat_cycles)
        self.assertEqual(self.plans(self.g2, context), [])
        context = self.context(self.boundary(until, x=26.609, cooldown=8.))
        self.assertTrue(self.plans(self.g2, context))

    def test_actor_identity_change_clears_cross_skill_intent(self):
        self.approach_boundary()
        state = self.boundary(490)
        state["allies"]["u0"]["tag"] = "replacement-unit"
        self.context(state)
        self.assertFalse(self.memory.combat_cycles)

    def test_observed_target_loss_clears_cross_skill_intent(self):
        self.approach_boundary()
        state = self.boundary(490)
        target = self.memory.combat_cycles["u0"]["target"]
        del state["visible_enemies"][target]
        del state["relations"]["u0"][target]
        self.context(state)
        self.assertFalse(self.memory.combat_cycles)

    def test_g3_preserves_real_next_shot_and_exposes_splash_objective(self):
        provider = SplashFireSkill({})
        g1 = FocusFireSkill({}, {"splash_fire": provider})
        g3 = CombatRepositionSkill({}, {"splash_fire": provider})
        state = scene([((0., 0.), "colossus")], loop=80,
                      enemies={"e0": enemy((2., 0.), "stalker"),
                               "e1": enemy((4., 0.), "stalker")}, attack_range=7.)
        context = self.context(state)
        fire = next(p for p in self.plans(g1, context) if p.goal == "e1")
        self.commit(context, fire)
        for loop in (88, 96):
            state["game_loop"] = loop
            state["allies"]["u0"]["weapon_cooldown_loops"] = 20.
            context = self.context(state)
            plans = self.plans(g3, context)
            self.assertTrue(plans)
            self.assertEqual({p.goal for p in plans}, {"e1"})
            objective = plans[0].goal_data["objective"]
            self.assertEqual(objective["next_attack_target"], "e1")
            self.assertTrue(objective["evaluate_splash"])
            self.assertEqual(objective["horizon_loops"], 16)
            self.commit(context, plans[0])
        self.assertIsNone(self.memory.instance_for("focus_fire", "u0"))
        self.assertEqual(self.memory.attack_history["u0"]["target"], "e1")


if __name__ == "__main__":
    unittest.main()
