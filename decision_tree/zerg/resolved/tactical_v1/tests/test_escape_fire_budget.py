"""Short survival evidence and a real cooling move -> legal-return timeline."""
import copy
import json
import math
from pathlib import Path
import unittest

from skills.tactical_v1.combat_exchange import escape_urgency, next_fire_delay
from skills.tactical_v1.combat_progress import CombatProgress
from skills.tactical_v1.context import Context, build_snapshot
from skills.tactical_v1.memory import TacticalMemory
from skills.tactical_v1.skills.disengage_reenter import DisengageReenterSkill
from skills.tactical_v1.telemetry import Telemetry
from .fixtures import enemy, scene, with_questions


class QuietMemory:
    def combat_emergency(self, *_):
        return False

    def movement_blockage(self, *_):
        return 0.


def state(loop=80, position=(0., 0.), cooldown=10.11, health=125., loss=0.,
          targets=None, native=6., command=6., speed=2.25):
    raw = scene([(position, 'marauder')], targets or {
        'e0': enemy((5.8, 0.), 'marine'), 'e1': enemy((5.8, .5), 'marine')},
        loop=loop, distance=2., attack_range=native)
    raw['allies']['u0'].update(health=health, max_health=125., effective_health=health,
        weapon_cooldown_loops=cooldown, health_lost_since_previous_decision=loss)
    raw['unit_type_profiles']['51'].update(armor=1., attributes=[2], movement_speed_native=speed)
    raw['unit_type_profiles']['51']['weapons'][0]['attack_period_loops'] = 24.
    for e, target in raw['visible_enemies'].items():
        d = math.dist(position, target['position'])
        relation = raw['relations']['u0'][e]
        relation.update(in_weapon_range=d <= native + 1.,
            weapon_range_status='in_range' if d <= native + 1. else 'out_of_range',
            smac_attack_available=d <= command, smac_attack_distance_limit=command,
            in_enemy_weapon_range=d <= 6.,
            enemy_weapon_range_status='in_range' if d <= 6. else 'out_of_range')
    return raw


def context(raw):
    return Context(build_snapshot(with_questions(raw), raw['game_loop']))


class EscapeFireBudgetTests(unittest.TestCase):
    def test_full_reserve_two_frontal_threats_do_not_generate_cooling_escape(self):
        c = context(state())
        self.assertEqual(len(c.risk_at('u0', (0., 0.)).threats), 2)
        self.assertGreater(c.snapshot.allies['u0'].weapon_cooldown_loops, c.H)
        evidence = escape_urgency(c, 'u0')
        self.assertFalse(evidence['crossfire'])
        self.assertFalse(evidence['urgent'])
        skill = DisengageReenterSkill({})
        self.assertFalse(skill._entry(c, 'u0'))
        self.assertFalse(skill.propose(c, TacticalMemory(), Telemetry()))
        old = DisengageReenterSkill({'common_combat': {'joint_assessment_enabled': False}})
        self.assertTrue(old._entry(c, 'u0'))

    def test_real_lethal_volley_does_not_require_a_previous_damage_sample(self):
        c = context(state(health=8., cooldown=0.))
        self.assertEqual(c.snapshot.allies['u0'].health_lost, 0.)
        self.assertTrue(escape_urgency(c, 'u0')['immediate'])
        skill = DisengageReenterSkill({})
        self.assertTrue(skill._entry(c, 'u0'))
        self.assertFalse(skill._protect_shot(c, 'u0'))
        self.assertEqual(CombatProgress().next_fire_cost(c, 'u0', 5, QuietMemory()), 0.)

    def test_pressure_or_actual_crossfire_with_limited_reserve_keeps_escape(self):
        c = context(state(health=40., loss=3.))
        facts = escape_urgency(c, 'u0')
        self.assertFalse(facts['immediate'])
        self.assertTrue(facts['urgent'])
        self.assertTrue(DisengageReenterSkill({})._entry(c, 'u0'))
        c = context(state(health=40., targets={'e0': enemy((5., 0.), 'marine'),
                                             'e1': enemy((-5., 0.), 'marine')}))
        facts = escape_urgency(c, 'u0')
        self.assertTrue(facts['crossfire'])
        self.assertTrue(facts['urgent'])
        # A healthy durable unit does not flee merely because firing
        # directions differ; the short exchange must consume real reserve.
        c = context(state(targets={'e0': enemy((5., 0.), 'marine'),
                                   'e1': enemy((-5., 0.), 'marine')}))
        self.assertFalse(escape_urgency(c, 'u0')['urgent'])

    def test_unknown_volley_is_not_certain_lethal_without_observed_trend(self):
        raw = state(health=20.)
        raw['unit_type_profiles']['48'].pop('weapons')
        c = context(raw)
        self.assertTrue(escape_urgency(c, 'u0')['exposure']['unknown'])
        self.assertFalse(escape_urgency(c, 'u0')['immediate'])
        raw['allies']['u0']['health_lost_since_previous_decision'] = 12.
        self.assertTrue(escape_urgency(context(raw), 'u0')['immediate'])

    def test_generic_interrupt_and_nonlethal_relief_do_not_erase_return_cost(self):
        c, memory, progress = context(state()), QuietMemory(), CombatProgress()
        memory.combat_emergency = lambda *_: True  # new threats, no injury
        self.assertGreater(next_fire_delay(c, 'u0', 5), 0.)
        frozen = copy.deepcopy(progress.__dict__)
        self.assertFalse(progress._safety_override(c, 'u0', 5, memory))
        self.assertEqual(progress.next_fire_cost(c, 'u0', 5, memory),
                         next_fire_delay(c, 'u0', 5))
        self.assertEqual(progress.__dict__, frozen)
        old = CombatProgress({'common_combat': {'joint_assessment_enabled': False}})
        self.assertEqual(old.next_fire_cost(c, 'u0', 5, memory), 0.)

    def test_fresh_critical_collapse_can_have_an_unknown_damage_source(self):
        raw = state(health=5., loss=30., targets={'e0': enemy((30., 0.), 'marine')})
        c = context(raw)
        self.assertFalse(c.risk_at('u0', c.snapshot.allies['u0'].position).threats)
        self.assertTrue(escape_urgency(c, 'u0')['observed_collapse'])
        self.assertTrue(CombatProgress()._safety_override(c, 'u0', 5, QuietMemory()))
        # No unobserved source gets extrapolated into another escape window.
        raw['allies']['u0']['health_lost_since_previous_decision'] = 0.
        self.assertFalse(escape_urgency(context(raw), 'u0')['immediate'])

    def test_observed_blockage_remains_an_exception_without_deadline_refresh(self):
        progress, memory = CombatProgress(), QuietMemory()
        before = context(state())
        progress.observe(before)
        progress.commit(before, {'u0': 5})
        after = context(state(loop=88, position=(-1.125, 0.), cooldown=2.11))
        progress.observe(after)
        until = progress.output_duty(after, 'u0')['until_loop']
        memory.movement_blockage = lambda *_: 1.
        self.assertEqual(progress.next_fire_cost(after, 'u0', 5, memory), 0.)
        self.assertEqual(progress.output_duty(after, 'u0')['until_loop'], until)

    def test_departure_return_uses_actual_speed_and_both_range_boundaries(self):
        for native, command in ((6., 6.), (5., 20.)):
            with self.subTest(native=native, command=command):
                progress, memory = CombatProgress(), QuietMemory()
                c = context(state(native=native, command=command))
                stride = 2.25 * 8. / 16.
                end = c.predict_position('u0', 5).position
                self.assertAlmostEqual(math.dist(c.snapshot.allies['u0'].position, end), stride)
                # Effective center reach is 6 in both cases. The move takes
                # one H; the 0.925 return gap cannot fit in the remaining CD.
                expected = 1. + (5.8 + stride - 6.) / stride - 10.11 / 8.
                self.assertAlmostEqual(progress.next_fire_cost(c, 'u0', 5, memory), expected)
                progress.observe(c)
                progress.commit(c, {'u0': 5})
                c = context(state(loop=88, position=end, cooldown=2.11,
                                  native=native, command=command))
                progress.observe(c)
                duty = progress.output_duty(c, 'u0')
                self.assertEqual(duty['cycles'], 0)
                self.assertFalse(any(c.contact_status('u0', e) == 'in_range'
                                     for e in c.attackable_enemies('u0')))
                self.assertGreater(progress.next_fire_cost(c, 'u0', 5, memory), 0.)
                closing_cost = progress.next_fire_cost(c, 'u0', 4, memory)
                if command > native + 1.:
                    # With a wider command mask, legal ATTACK can implicitly
                    # close the 0.925 native gap in 6.58 loops. MOVE occupies
                    # all8, so this small delay is avoidable even while closing.
                    self.assertTrue(c.can_issue_attack('u0', 'e0'))
                    self.assertAlmostEqual(closing_cost, 1. - (5.8 + stride - 6.) / stride)
                    self.assertEqual(progress.next_fire_cost(c, 'u0', 6, memory), 0.)
                else:
                    self.assertEqual(closing_cost, 0.)
                # A further lateral detour leaves the now-ready weapon out
                # of contact; returning at the real stride restores contact.
                detour = context(state(loop=96, position=(-stride, stride), cooldown=0.,
                                       native=native, command=command))
                self.assertFalse(any(detour.contact_status('u0', e) == 'in_range'
                                     for e in detour.attackable_enemies('u0')))
                progress.commit(c, {'u0': 4})
                c = context(state(loop=96, position=(0., 0.), cooldown=0.,
                                  native=native, command=command))
                progress.observe(c)
                self.assertEqual(c.contact_status('u0', 'e0'), 'in_range')
                self.assertEqual(progress.output_duty(c, 'u0')['until_loop'], duty['until_loop'])
                self.assertEqual(progress.output_duty(c, 'u0')['cycles'], 0)
                self.assertEqual(progress.next_fire_cost(c, 'u0', 6, memory), 0.)
                progress.commit(c, {'u0': 6})
                c = context(state(loop=104, cooldown=16., native=native, command=command))
                progress.observe(c)
                self.assertEqual(progress.output_duty(c, 'u0')['cycles'], 1)

    def test_return_time_changes_with_observed_native_speed(self):
        slow = context(state(speed=1.125))
        normal = context(state())
        self.assertLess(next_fire_delay(slow, 'u0', 5), next_fire_delay(normal, 'u0', 5))
        self.assertEqual(next_fire_delay(context(state(cooldown=24.)), 'u0', 5), 0.)

    def test_recorded_seed172_u9_step10_no_healthy_escape_or_free_detour(self):
        path = Path(__file__).resolve().parents[3] / 'runs' / \
            'tactical_20x3x3_20261115_173227/current/t_6m4r/events.jsonl'
        if not path.exists():
            self.skipTest('recorded batch is not distributed with the source tree')
        with path.open(encoding='utf-8') as stream:
            record = next(json.loads(line) for i, line in enumerate(stream, 1) if i == 296)
        self.assertEqual((record['event'], record['episode'], record['step']),
                         ('skill_observation', 3, 10))
        c = Context(build_snapshot(record['payload'], 10))
        self.assertEqual(c.snapshot.allies['u9'].health, 125.)
        self.assertAlmostEqual(c.snapshot.allies['u9'].weapon_cooldown_loops, 10.11)
        self.assertFalse(DisengageReenterSkill({})._entry(c, 'u9'))
        self.assertGreater(CombatProgress().next_fire_cost(c, 'u9', 5, QuietMemory()), 0.)


if __name__ == '__main__':
    unittest.main()
