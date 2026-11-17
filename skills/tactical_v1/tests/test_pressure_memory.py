"""A cooldown gap is not proof that observed lethal exposure disappeared."""
import copy
import unittest

from skills.tactical_v1.combat_exchange import (imminent_exchange_danger,
    observed_pressure, exposure_window)
from skills.tactical_v1.context import Context, build_snapshot
from skills.tactical_v1.contracts import Commitment, ResourceClaims
from skills.tactical_v1.memory import TacticalMemory
from skills.tactical_v1.skills.common import make_plan
from skills.tactical_v1.skills.disengage_reenter import DisengageReenterSkill
from skills.tactical_v1.telemetry import Telemetry
from .fixtures import scene, enemy, with_questions


class PressureMemoryTests(unittest.TestCase):
    def state(self, loop=80, loss=20., actor='marine', x=0., health=29.):
        s = scene([((x, 0.), actor)], {
            f'e{i}': enemy((3., i * .1), 'zealot', tag=f'enemy-{i}')
            for i in range(3)}, loop=loop, distance=2.)
        s['allies']['u0'].update(health=health, max_health=100.,
            effective_health=health, health_lost_since_previous_decision=loss)
        return s

    def observe(self, memory, state):
        c = Context(build_snapshot(with_questions(state), state['game_loop']))
        memory.begin_step(c.snapshot)
        memory.observe(c.snapshot)
        memory.update_task_facts(c.snapshot, c)
        return c

    def test_one_quiet_window_keeps_same_lethal_threat_but_evidence_decays(self):
        for actor in ('marine', 'marauder', 'zergling', 'hydralisk', 'zealot', 'stalker'):
            with self.subTest(actor=actor):
                memory = TacticalMemory()
                c = self.observe(memory, self.state(actor=actor))
                self.assertTrue(imminent_exchange_danger(c, 'u0'))
                c = self.observe(memory, self.state(88, 0., actor))
                before = copy.deepcopy(memory.__dict__)
                self.assertEqual(observed_pressure(c, 'u0'), 10.)
                self.assertTrue(imminent_exchange_danger(c, 'u0'))
                self.assertTrue(DisengageReenterSkill({})._severe(c, 'u0'))
                self.assertEqual(memory.__dict__, before)
                c = self.observe(memory, self.state(96, 0., actor))
                self.assertFalse(imminent_exchange_danger(c, 'u0'))
                self.assertFalse(memory.pressure_memory.records)

    def test_pressure_expiry_enables_attack_again_without_renewing_on_query(self):
        memory = TacticalMemory()
        # The known volley (48) is below current reserve (50). Only the
        # observed loss trend over the cooling window makes this lethal;
        # expiry must not pretend an actually lethal volley became safe.
        state = self.state(loss=40., health=50.)
        state['allies']['u0']['weapon_cooldown_loops'] = 20.
        self.observe(memory, state)
        state = self.state(88, 0., health=50.)
        state['allies']['u0']['weapon_cooldown_loops'] = 20.
        c = self.observe(memory, state)
        skill = DisengageReenterSkill({})
        self.assertFalse(skill._protect_shot(c, 'u0'))
        self.assertTrue(skill.propose(c, memory, Telemetry()))
        c = self.observe(memory, self.state(96, 0., health=50.))
        self.assertTrue(skill._protect_shot(c, 'u0'))
        self.assertFalse(skill.propose(c, memory, Telemetry()))

    def test_no_blanket_low_health_retreat_or_persistence_after_small_chip(self):
        memory = TacticalMemory()
        state = self.state(loss=0.)
        # One known 16-damage opponent, not three already lethal opponents.
        state['visible_enemies'] = {'e0': state['visible_enemies']['e0']}
        state['relations']['u0'] = {'e0': state['relations']['u0']['e0']}
        c = self.observe(memory, state)
        self.assertFalse(imminent_exchange_danger(c, 'u0'))
        self.assertFalse(DisengageReenterSkill({})._severe(c, 'u0'))
        self.observe(memory, self.state(88, loss=1., health=90.))
        c = self.observe(memory, self.state(96, loss=0., health=90.))
        self.assertFalse(imminent_exchange_danger(c, 'u0'))

    def test_actual_relief_and_replaced_enemy_release_old_damage(self):
        for mode in ('relief', 'enemy_tag', 'actor_tag', 'episode'):
            with self.subTest(mode=mode):
                memory = TacticalMemory()
                self.observe(memory, self.state())
                s = self.state(88, loss=0., x=-2. if mode == 'relief' else 0.)
                if mode == 'enemy_tag':
                    for unit in s['visible_enemies'].values():
                        unit['tag'] += '-new'
                elif mode == 'actor_tag':
                    s['allies']['u0']['tag'] = 'replacement'
                elif mode == 'episode':
                    s['episode'] = 2
                c = self.observe(memory, s)
                self.assertEqual(observed_pressure(c, 'u0'), 0.)
                self.assertFalse(imminent_exchange_danger(c, 'u0'))
                self.assertFalse(memory.pressure_memory.records)

    def test_pressure_ablation_preserves_old_imminent_and_excludes_special_units(self):
        memory = TacticalMemory(encounter_config={
            'common_combat': {'pressure_memory_enabled': False}})
        c = self.observe(memory, self.state())
        self.assertTrue(imminent_exchange_danger(c, 'u0'))
        c = self.observe(memory, self.state(88, loss=0.))
        self.assertFalse(imminent_exchange_danger(c, 'u0'))
        for kind in ('medivac', 'baneling'):
            memory = TacticalMemory()
            self.observe(memory, self.state(actor=kind))
            self.observe(memory, self.state(88, loss=0., actor=kind))
            self.assertFalse(memory.pressure_memory.records)

    def test_unknown_damage_does_not_erase_observed_pressure_or_claim_safe_reserve(self):
        memory = TacticalMemory()
        s = self.state()
        s['unit_type_profiles'].pop('73')
        self.observe(memory, s)
        s = self.state(88, loss=0.)
        s['unit_type_profiles'].pop('73')
        c = self.observe(memory, s)
        self.assertEqual(observed_pressure(c, 'u0'), 10.)
        facts = exposure_window(c, 'u0')
        self.assertTrue(facts['unknown'])
        self.assertLessEqual(facts['viability'], .5)
        self.assertTrue(imminent_exchange_danger(c, 'u0'))

    def test_remembered_collapse_interrupts_committed_cross_skill_intent(self):
        for enabled in (True, False):
            with self.subTest(enabled=enabled):
                memory = TacticalMemory(encounter_config={
                    'common_combat': {'pressure_memory_enabled': enabled}})
                s = self.state()
                for unit in s['visible_enemies'].values():
                    unit['position'][0] = 2.  # both ends remain under the same exposure
                c = self.observe(memory, s)
                plan = make_plan('coordinated_engage', ('u0',), 'e0', {'u0': 4}, [],
                    ResourceClaims(controls=('u0',)), Commitment('approaching'), {},
                    c.snapshot.snapshot_id, 'test', goal_kind='enemy')
                memory.commit(c.snapshot, {'u0': 4}, [plan], context=c)
                s = self.state(88, loss=0., x=1.125)
                for unit in s['visible_enemies'].values():
                    unit['position'][0] = 2.
                c = self.observe(memory, s)
                before = copy.deepcopy(memory.__dict__)
                self.assertEqual(memory.combat_emergency(c, 'u0'), enabled)
                if enabled:
                    self.assertEqual(memory.movement_conflict(c, 'u0', 5), 0.)
                else:
                    self.assertGreater(memory.movement_conflict(c, 'u0', 5), 0.)
                self.assertEqual(memory.__dict__, before)


if __name__ == '__main__':
    unittest.main()
