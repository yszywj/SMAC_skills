"""Sequence tests for observed progress and skill-independent retreat limits."""
import copy
import unittest

from skills.tactical_v1.context import Context, build_snapshot
from skills.tactical_v1.contracts import Commitment, ResourceClaims
from skills.tactical_v1.memory import TacticalMemory
from skills.tactical_v1.skills.common import make_plan
from .fixtures import enemy, scene, with_questions


class BattleIntentTests(unittest.TestCase):
    def setUp(self):
        self.memory = TacticalMemory()

    def observe(self, state):
        context = Context(build_snapshot(with_questions(state), state['game_loop']))
        self.memory.begin_step(context.snapshot)
        self.memory.observe(context.snapshot)
        self.memory.update_task_facts(context.snapshot, context)
        return context

    def state(self, loop=80, name='marine', hp=35., enemy_x=4.):
        return scene([((0., 0.), name), ((0., 1.), name)],
                     enemies={'e0': enemy((enemy_x, 0.), health=hp)}, loop=loop)

    def commit(self, c, skill='disengage_reenter', action=5):
        p = make_plan(skill, ('u0',), 'safe', {'u0': action}, [],
                      ResourceClaims(controls=('u0',)), Commitment('ESCAPE'), {},
                      c.snapshot.snapshot_id, 'test')
        self.memory.commit(c.snapshot, {'u0': action, 'u1': 1}, [p], context=c)

    def test_actual_stall_changes_phase_damage_resumes_engagement(self):
        self.observe(self.state())
        c = self.observe(self.state(112))
        self.assertEqual(self.memory.battle_state(c)['phase'], 'press')
        c = self.observe(self.state(120, hp=25.))
        self.assertEqual(self.memory.battle_state(c)['phase'], 'engage')
        self.assertEqual(self.memory.battle_state(c)['stalled_h'], 0.)

    def test_disappearance_is_not_damage_and_visibility_return_not_new_progress(self):
        self.observe(self.state())
        s = self.state(88)
        s['visible_enemies'] = {}
        s['relations'] = {a: {} for a in s['allies']}
        c = self.observe(s)
        self.assertEqual(self.memory.battle_state(c)['observed_net_damage'], 0.)
        c = self.observe(self.state(96, hp=1.))
        self.assertEqual(self.memory.battle_state(c)['observed_net_damage'], 0.)
        self.assertEqual(self.memory.battle_state(c)['stalled_h'], 2.)

    def test_proposal_queries_do_not_renew_target_or_history(self):
        c = self.observe(self.state())
        old = copy.deepcopy(self.memory.__dict__)
        for _ in range(3):
            self.memory.battle_state(c)
            self.memory.battle_target(c, 'u0')
            self.memory.retreat_exhausted(c, 'u0')
            self.memory.update_task_facts(c.snapshot, c)
        self.assertEqual(self.memory.__dict__, old)

    def test_retreat_failure_survives_skill_switch_and_stops_after_actual_attack(self):
        c = self.observe(self.state(enemy_x=2.))
        self.commit(c)
        c = self.observe(self.state(88, enemy_x=2.))  # a blocked escape did not help
        self.commit(c, skill='combat_reposition')
        c = self.observe(self.state(96, enemy_x=2.))
        self.assertTrue(self.memory.retreat_exhausted(c, 'u0'))
        self.commit(c, skill='focus_fire', action=6)
        c = self.observe(self.state(104))
        self.assertTrue(self.memory.retreat_exhausted(c, 'u0'))
        self.commit(c, skill='focus_fire', action=6)
        s = self.state(112)
        s['allies']['u0']['weapon_cooldown_loops'] = 16.
        c = self.observe(s)  # the command alone did not prove actual firing
        self.assertFalse(self.memory.retreat_exhausted(c, 'u0'))

    def test_real_risk_reduction_clears_failed_streak(self):
        c = self.observe(self.state(enemy_x=2.))
        self.commit(c)
        c = self.observe(self.state(88, enemy_x=2.))
        self.commit(c)
        c = self.observe(self.state(96, enemy_x=2.))
        self.assertTrue(self.memory.retreat_exhausted(c, 'u0'))
        self.commit(c)
        c = self.observe(self.state(104, enemy_x=4.))
        self.assertFalse(self.memory.retreat_exhausted(c, 'u0'))

    def test_unknown_speed_incompatible_domain_and_healer_do_not_get_chase_goal(self):
        for condition in ('speed', 'domain', 'healer', 'mask'):
            with self.subTest(condition=condition):
                self.memory = TacticalMemory()
                s = self.state(enemy_x=8.)
                s['allies']['u1']['health_lost_since_previous_decision'] = 1.
                if condition == 'speed':
                    s['unit_type_profiles']['48'].pop('movement_speed_native')
                elif condition == 'domain':
                    for a in s['relations']:
                        s['relations'][a]['e0']['weapon_range_status'] = 'no_compatible_weapon'
                elif condition == 'healer':
                    s['allies']['u0']['type'] = 'medivac'
                    s['allies']['u0']['unit_type_id'] = 54
                else:
                    s['action_contract'] = {}
                    for a in s['relations']:
                        s['relations'][a]['e0'].pop('smac_attack_distance_limit')
                c = self.observe(s)
                self.assertIsNone(self.memory.battle_target(c, 'u0'))

    def test_episode_and_identity_change_drop_retreat_evidence(self):
        c = self.observe(self.state())
        self.commit(c)
        c = self.observe(self.state(88))
        self.commit(c)
        c = self.observe(self.state(96))
        self.assertTrue(self.memory.retreat_exhausted(c, 'u0'))
        s = self.state(104)
        s['allies']['u0']['tag'] = 'replacement'
        c = self.observe(s)
        self.assertFalse(self.memory.retreat_exhausted(c, 'u0'))
        s = self.state(0)
        s['episode'] = 2
        c = self.observe(s)
        self.assertEqual(self.memory.battle_state(c)['stalled_h'], 0.)

    def test_same_progress_rules_for_melee_ranged_and_splash(self):
        for name in ('zealot', 'stalker', 'colossus', 'marine', 'zergling'):
            with self.subTest(name=name):
                self.memory = TacticalMemory()
                self.observe(self.state(name=name, enemy_x=1.))
                c = self.observe(self.state(112, name=name, enemy_x=1.))
                self.assertEqual(self.memory.battle_state(c)['phase'], 'press')
                self.assertEqual(self.memory.battle_target(c, 'u0'), 'e0')


if __name__ == '__main__':
    unittest.main()
