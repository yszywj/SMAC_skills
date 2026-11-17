"""Generic candidates preserve physics and defer joint exchange admission."""
import copy
import unittest

from skills.tactical_v1.context import Context, build_snapshot
from skills.tactical_v1.combat_exchange import movement_exchange
from skills.tactical_v1.memory import TacticalMemory
from skills.tactical_v1.skills.coordinated_engage import CoordinatedEngageSkill
from skills.tactical_v1.skills.combat_reposition import CombatRepositionSkill
from skills.tactical_v1.skills.local_formation import LocalFormationSkill
from skills.tactical_v1.skills.regroup import RegroupSkill
from skills.tactical_v1.combat_groups import viable_rejoin_data
from skills.tactical_v1.joint_combat import assess_joint
from skills.tactical_v1.telemetry import Telemetry
from .fixtures import scene, enemy, with_questions


def context(raw):
    return Context(build_snapshot(with_questions(raw), raw['game_loop']))


class JointGenericContractTests(unittest.TestCase):
    def test_g2_preserves_contested_route_for_final_joint_assessment(self):
        raw = scene([((0., 0.), 'marine')],
            {f'e{i}': enemy((6.3, .1 * i), 'hydralisk') for i in range(3)}, distance=2.)
        raw['allies']['u0'].update(health=20., max_health=20.)
        c, memory = context(raw), TacticalMemory()
        prediction = c.predict_position('u0', 4).position
        self.assertFalse(movement_exchange(c, 'u0', prediction)['admissible'])
        saved = copy.deepcopy(memory.__dict__)
        new = CoordinatedEngageSkill({}).propose(c, memory, Telemetry())
        old = CoordinatedEngageSkill({'common_combat': {'joint_assessment_enabled': False}}).propose(
            c, memory, Telemetry())
        route = next(p for p in new if p.current_actions.get('u0') == 4)
        self.assertGreaterEqual(route.goal_data['movement_contract']['reserve_ratio'], 1.)
        self.assertFalse(any(p.current_actions.get('u0') == 4 for p in old))
        self.assertEqual(memory.__dict__, saved)

    def test_unknown_speed_still_cannot_invent_arrival(self):
        raw = scene([((0., 0.), 'marine')], {'e0': enemy((8., 0.))})
        raw['unit_type_profiles']['48'].pop('movement_speed_native')
        plans = CoordinatedEngageSkill({}).propose(context(raw), TacticalMemory(), Telemetry())
        self.assertFalse(any(p.commitment.phase == 'approaching' for p in plans))

    def test_formation_never_takes_burst_actor_even_as_atomic_partner(self):
        raw = scene([((0., 0.), 'marine'), ((.2, 0.), 'baneling')],
                    {'e0': enemy((8., 0.))})
        c = context(raw)
        plans = LocalFormationSkill({}).propose(c, TacticalMemory(), Telemetry())
        self.assertTrue(plans)
        self.assertTrue(all('u1' not in p.controlled_actors for p in plans))
        old = LocalFormationSkill({'common_combat': {'joint_assessment_enabled': False}}).propose(
            c, TacticalMemory(), Telemetry())
        self.assertTrue(any('u1' in p.controlled_actors for p in old))

    def test_g3_uses_command_native_reach_for_both_melee_and_ranged(self):
        for kind in ('marine', 'marauder', 'hydralisk', 'stalker', 'zergling', 'zealot'):
            with self.subTest(kind=kind):
                raw = scene([((0., 0.), kind)], {'e0': enemy((1., 0.))})
                raw['allies']['u0']['weapon_cooldown_loops'] = 24.
                raw['relations']['u0']['e0'].update(native_weapon_range=10., smac_attack_distance_limit=2.)
                c = context(raw)
                plans = CombatRepositionSkill({}).propose(c, TacticalMemory(), Telemetry())
                self.assertTrue(plans)
                self.assertTrue(all(p.goal_data['objective']['r_max'] == 2. for p in plans))

    def test_g3_unknown_reach_has_no_fabricated_five_unit_band(self):
        raw = scene([((0., 0.), 'marine')], {'e0': enemy((1., 0.))})
        raw['allies']['u0']['weapon_cooldown_loops'] = 24.
        raw['relations']['u0']['e0'].update(native_weapon_range=None)
        self.assertFalse(CombatRepositionSkill({}).propose(context(raw), TacticalMemory(), Telemetry()))

    def test_g3_band_width_uses_observed_catalog_speed(self):
        raw = scene([((0., 0.), 'marine')], {'e0': enemy((1., 0.))})
        raw['allies']['u0']['weapon_cooldown_loops'] = 24.
        def lower(state):
            return CombatRepositionSkill({}).propose(context(state), TacticalMemory(), Telemetry())[0].goal_data['objective']['r_min']
        slow = lower(raw)
        raw['unit_type_profiles']['48']['movement_speed_native'] = 4.
        self.assertLess(lower(raw), slow)

    def test_regroup_leaves_visible_burst_duty_but_can_gather_targetless_burst(self):
        raw = scene([((0., 0.), 'baneling'), ((10., 0.), 'marine'), ((11., 0.), 'marine')],
                    {'e0': enemy((12., 0.))})
        plans = RegroupSkill({}).propose(context(raw), TacticalMemory(), Telemetry())
        self.assertTrue(all('u0' not in p.controlled_actors for p in plans))
        raw = scene([((0., 0.), 'baneling'), ((10., 0.), 'marine'), ((11., 0.), 'marine')])
        plans = RegroupSkill({}).propose(context(raw), TacticalMemory(), Telemetry())
        self.assertTrue(any('u0' in p.controlled_actors for p in plans))

    def test_regroup_main_front_uses_compatible_fire_readiness_not_population(self):
        raw = scene([((x, 0.), 'marine') for x in (0., 10., 11., -10., -11., -12.)],
                    {'e0': enemy((15., 0.)), 'e1': enemy((-13., 0.))})
        for actor in ('u3', 'u4', 'u5'):
            raw['allies'][actor]['weapon_cooldown_loops'] = 24.
        c = context(raw)
        new = RegroupSkill({})._merge_options(c, 'u0')
        old = RegroupSkill({'common_combat': {'joint_assessment_enabled': False}})._merge_options(c, 'u0')
        self.assertEqual(set(new[0][0]), {'u1', 'u2'})
        self.assertEqual(set(old[0][0]), {'u3', 'u4', 'u5'})

    def test_rejoin_declaration_preserves_deadline_and_does_not_claim_healthy_cover_hp(self):
        raw = scene([((0., 0.), 'marine'), ((3., 0.), 'marine')],
                    {f'e{i}': enemy((6., i*.1), 'hydralisk') for i in range(3)})
        raw['allies']['u0'].update(health=20., max_health=20.)
        raw['allies']['u1'].update(health=200., max_health=200.)
        c = context(raw)
        data = {'joiners': ['u0'], 'supporters': ['u1'], 'roles': {'u0': 'join', 'u1': 'cover'},
                'support_radius': 6., 'local_cover_enabled': True, 'arrival_exposure_enabled': True,
                'joint_assessment_enabled': True, 'initial_threat_counts': {}}
        facts = movement_exchange(c, 'u0', moving=False)
        self.assertFalse(facts['admissible'])
        frozen = copy.deepcopy(data)
        self.assertIn('u0', viable_rejoin_data(c, data, 8.)['joiners'])
        self.assertEqual(data, frozen)
        self.assertIsNone(viable_rejoin_data(c, dict(data, joint_assessment_enabled=False), 8.))
        joint = assess_joint(c, {'u0': 4, 'u1': 6})
        self.assertGreaterEqual(joint['actors']['u0']['exposure']['reserve_ratio'], 1.)
        self.assertLess(joint['actors']['u1']['exposure']['reserve_ratio'], 1.)


if __name__ == '__main__':
    unittest.main()
