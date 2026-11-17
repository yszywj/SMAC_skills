"""Melee frontage and detour regressions, including the recorded Z182 shuffle."""
import copy
import json
import math
from pathlib import Path
import unittest

from skills.tactical_v1.config import ConfigError, load, validate
from skills.tactical_v1.context import Context, build_snapshot
from skills.tactical_v1.ledger import Ledger
from skills.tactical_v1.memory import TacticalMemory
from skills.tactical_v1.skills.coordinated_engage import CoordinatedEngageSkill
from skills.tactical_v1.telemetry import Telemetry
from .fixtures import enemy, scene, with_questions


def native_scene(positions, target=(4., 0.), loop=80):
    state = scene([(p, 'zergling') for p in positions],
                  {'e0': enemy(target, 'hydralisk', radius=.625, health=90., max_health=90.)},
                  loop=loop, attack_range=.10009765625, distance=2.)
    state['unit_type_profiles']['105']['movement_speed_native'] = 2.953125
    state['unit_type_profiles']['105']['weapons'][0]['range'] = .10009765625
    for actor, unit in state['allies'].items():
        unit.update(radius=.375, health=35., max_health=35., tag=actor)
        relation = state['relations'][actor]['e0']
        relation['smac_attack_distance_limit'] = 2.
    return state


def context(state):
    return Context(build_snapshot(with_questions(state), state['game_loop']))


class MeleeContactRoutesTests(unittest.TestCase):
    def setUp(self):
        self.skill = CoordinatedEngageSkill({})

    def test_nearest_slot_tracks_actor_side_not_roster_index(self):
        c = context(native_scene([(-4., 0.), (4., 4.)], target=(4., 0.)))
        west, _ = self.skill._slot(c, 'u0', 'e0', 0)
        north, _ = self.skill._slot(c, 'u1', 'e0', 0)
        self.assertLess(west[0], 4.)
        self.assertGreater(north[1], 0.)
        self.assertAlmostEqual(west[1], 0.)
        self.assertAlmostEqual(north[0], 4.)

    def test_small_native_body_frontage_and_margin_are_not_legacy_spacing(self):
        c = context(native_scene([(-4., 0.)], target=(0., 0.)))
        points = {self.skill._slot(c, 'u0', 'e0', i)[1]: self.skill._slot(c, 'u0', 'e0', i)[0]
                  for i in range(16)}
        self.assertEqual(len(points), 8)
        for point in points.values():
            self.assertTrue(c.projected_attack_opportunity('u0', 'e0', point).available)
            self.assertGreaterEqual(math.dist(point, (0., 0.)), 1.)
        for i, a in enumerate(points.values()):
            for b in list(points.values())[i+1:]:
                self.assertGreaterEqual(math.dist(a, b), .75 + .05 - 1e-8)
        legacy = CoordinatedEngageSkill({'common_combat': {'joint_assessment_enabled': False}})
        self.assertEqual(len({legacy._slot(c, 'u0', 'e0', i)[1] for i in range(16)}), 4)
        config = load()
        self.assertEqual(config['coordinated_engage']['contact_slot_margin'], .05)
        config['coordinated_engage']['contact_slot_margin'] = -1.
        with self.assertRaises(ConfigError):
            validate(config)

    def test_alternative_slots_are_branch_claims_with_unique_ids_and_budget(self):
        c = context(native_scene([(-4., -.5), (-4., .5)], target=(4., 0.)))
        memory = TacticalMemory()
        before = copy.deepcopy(memory.__dict__)
        plans = self.skill.propose(c, memory, Telemetry())
        self.assertEqual(before, memory.__dict__)
        self.assertEqual(len(plans), len({p.proposal_id for p in plans}))
        first = [p for p in plans if p.controlled_actors == ('u0',)]
        second = [p for p in plans if p.controlled_actors == ('u1',)]
        self.assertGreaterEqual(len({p.claims.space[0].slot_id for p in first}), 2)
        self.assertLessEqual(len(first), 6)
        self.assertLessEqual(len(second), 6)
        a, b = next((a, b) for a in first for b in second
                    if a.claims.space[0].slot_id == b.claims.space[0].slot_id)
        ledger = Ledger(); ledger.reserve(a)
        self.assertTrue(any(reason.startswith('slot:') for reason in ledger.conflicts(b)))
        alternate = next(b for b in second if b.claims.space[0].slot_id != a.claims.space[0].slot_id)
        self.assertEqual(ledger.conflicts(alternate), [])

    def test_clear_corridor_negligible_radial_progress_is_not_approach(self):
        c = context(native_scene([(15.502, 8.815)], target=(21.542, 8.044)))
        goal, _ = self.skill._slot(c, 'u0', 'e0', 0)
        north_south = c.predict_position('u0', 3).position
        east = c.predict_position('u0', 4).position
        self.assertFalse(self.skill._useful_melee_route(c, TacticalMemory(), 'u0', 'e0', goal, north_south, False))
        self.assertTrue(self.skill._useful_melee_route(c, TacticalMemory(), 'u0', 'e0', goal, east, False))

    def test_visible_body_detour_is_preserved_without_claiming_safety(self):
        state = native_scene([(-4., 0.), (-2., 0.)], target=(0., 0.))
        c = context(state)
        goal = (-1.05, 0.)
        side = c.predict_position('u0', 2).position
        self.assertGreater(math.dist(side, (0., 0.)), 4.)
        self.assertTrue(self.skill._useful_melee_route(c, TacticalMemory(), 'u0', 'e0', goal, side, False))
        state['allies']['u1']['alive'] = False
        clear = context(state)
        self.assertFalse(self.skill._useful_melee_route(clear, TacticalMemory(), 'u0', 'e0', goal, side, False))

    def test_recent_same_target_return_does_not_become_a_fresh_detour(self):
        state = native_scene([(-4., -1.)], target=(0., 0.), loop=96)
        state['visible_enemies']['e0']['tag'] = 'stable'
        c = context(state); memory = TacticalMemory()
        task = memory.create_instance('coordinated_engage', ['u0'], 'e0', 'approaching', 80, 96, goal_kind='enemy')
        task.extra['approach_history'] = [(88, {'u0': (-4., 0.)})]
        task.extra['approach_target_history'] = [(88, 'e0', 'stable', (0., 0.))]
        before = copy.deepcopy(memory.__dict__)
        self.assertFalse(self.skill._useful_melee_route(c, memory, 'u0', 'e0', (-1.05, 0.), (-4., 0.), False))
        self.assertEqual(before, memory.__dict__)
        # Actual relative closure due to enemy motion is not stale self motion.
        task.extra['approach_target_history'] = [(88, 'e0', 'stable', (2., 0.))]
        self.assertTrue(self.skill._useful_melee_route(c, memory, 'u0', 'e0', (-1.05, 0.), (-4., 0.), False))

    def test_slot_reselection_does_not_renew_task_deadline_or_observed_progress(self):
        c = context(native_scene([(-4., 0.)], target=(4., 0.), loop=88))
        memory = TacticalMemory()
        task = memory.create_instance('coordinated_engage', ['u0'], 'e0', 'approaching', 80, 96, goal_kind='enemy')
        before = copy.deepcopy(task)
        plans = self.skill.propose(c, memory, Telemetry())
        self.assertTrue(plans)
        self.assertTrue(all(p.instance_id == task.instance_id and p.proposed_memory_delta['kind'] == 'continue' for p in plans))
        memory.commit(c.snapshot, plans[-1].current_actions, [plans[-1]], context=c)
        after = memory.instance_for('coordinated_engage', 'u0')
        self.assertEqual((after.instance_id, after.start_loop, after.max_loops, after.last_progress_loop),
                         (before.instance_id, before.start_loop, before.max_loops, before.last_progress_loop))

    def test_recorded_seed182_safe_side_shuffle_is_not_reissued_as_g2(self):
        path = Path(__file__).resolve().parents[3] / 'runs/tactical_20x3x3_20261115_173227/current/z_6l4h/events.jsonl'
        if not path.exists():
            self.skipTest('optional recorded native regression observations unavailable')
        observed = 0
        with path.open(encoding='utf8') as stream:
            for line in stream:
                row = json.loads(line)
                if row['event'] != 'skill_observation' or row['episode'] != 13 or row['step'] not in (22, 23, 24):
                    continue
                c = Context(build_snapshot(row['payload'], row['step'] + 1))
                memory = TacticalMemory()
                memory.create_instance('coordinated_engage', ['u2'], 'e4', 'approaching', 162, 96, goal_kind='enemy')
                plans = [p for p in self.skill.propose(c, memory, Telemetry()) if p.controlled_actors == ('u2',)]
                self.assertTrue(plans)
                self.assertEqual({p.current_actions['u2'] for p in plans}, {4})
                self.assertLess(plans[0].goal_data['waypoint'][0], c.snapshot.enemies['e4'].position[0])
                observed += 1
        self.assertEqual(observed, 3)


if __name__ == '__main__':
    unittest.main()
