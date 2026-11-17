"""Local fire arrival, finite safe waits, and provider-independent time cost."""
import copy
import unittest

from skills.tactical_v1.context import Context, build_snapshot
from skills.tactical_v1.joint_combat import assess_joint
from skills.tactical_v1.local_entry import (assess_local_entry, wait_opportunity,
                                           wait_remaining_loops)
from skills.tactical_v1.memory import TacticalMemory
from skills.tactical_v1.skills.coordinated_engage import CoordinatedEngageSkill
from skills.tactical_v1.telemetry import Telemetry
from .fixtures import enemy, scene, with_questions


def state(loop=80):
    return scene([((0., 0.), 'marine'), ((1.5, 0.), 'marine')],
                 {'e0': enemy((7.5, 0.), 'hydralisk')}, loop=loop, distance=2.)


def context(raw):
    return Context(build_snapshot(with_questions(raw), raw['game_loop']))


def measured(c, actions, memory=None, plans=()):
    return assess_local_entry(c, actions, assess_joint(c, actions), plans, memory)


class LocalEntryTests(unittest.TestCase):
    def test_safe_wait_requires_actual_final_peer_entry_and_is_provider_independent(self):
        c, memory = context(state()), TacticalMemory()
        self.assertIsNotNone(wait_opportunity(c, memory, 'u0', 'e0'))
        moving = measured(c, {'u0': 1, 'u1': 4}, memory)
        stopped = measured(c, {'u0': 1, 'u1': 1}, memory)
        withdrawing = measured(c, {'u0': 1, 'u1': 5}, memory)
        self.assertIn('u0', moving['valid_wait_actors'])
        self.assertNotIn('u0', stopped['valid_wait_actors'])
        self.assertNotIn('u0', withdrawing['valid_wait_actors'])
        self.assertIn('u0', stopped['invalid_wait_actors'])
        self.assertIn('u0', withdrawing['invalid_wait_actors'])
        self.assertNotIn('u0', moving['invalid_wait_actors'])
        self.assertEqual(moving, measured(c, {'u0': 1, 'u1': 4}, memory,
                                        plans=(object(),)))

    def test_new_exposure_is_not_free_when_before_support_is_zero(self):
        c = context(state())
        alone = measured(c, {'u0': 4, 'u1': 1})
        together = measured(c, {'u0': 4, 'u1': 4})
        self.assertGreater(alone['cost'], 0.)
        self.assertLess(together['cost'], alone['cost'])
        self.assertLessEqual(abs(alone['cost']), 1.)

    def test_existing_exposed_front_can_gain_cover_even_before_support_zero(self):
        c = context(state())
        idle = measured(c, {'u0': 1, 'u1': 4})
        cover = measured(c, {'u0': 4, 'u1': 4})
        first = {a: value for front in idle['fronts'] for a, value in front['actor_costs'].items()}
        second = {a: value for front in cover['fronts'] for a, value in front['actor_costs'].items()}
        self.assertEqual(first['u1'], 0.)
        self.assertLess(second['u1'], 0.)

    def test_wait_deadline_survives_switch_to_baseline_and_new_plan(self):
        memory, skill = TacticalMemory(), CoordinatedEngageSkill({})
        original = None
        for loop in (80, 88, 96):
            c = context(state(loop))
            memory.begin_step(c.snapshot)
            memory.observe(c.snapshot)
            memory.update_task_facts(c.snapshot, c)
            plans = skill.propose(c, memory, Telemetry())
            waits = [p for p in plans if p.controlled_actors == ('u0',)
                     and p.commitment.phase == 'waiting']
            if loop == 80:
                self.assertTrue(waits)
                original = waits[0]
                memory.commit(c.snapshot, {'u0': 1, 'u1': 4}, [original], context=c)
            elif loop == 88:
                self.assertEqual(wait_remaining_loops(c, memory, 'u0'), 8)
                # The peer failed to move in this fixture: it cannot keep
                # promising the same arrival merely by issuing East again.
                self.assertFalse(waits)
                memory.commit(c.snapshot, {'u0': 1, 'u1': 4}, [], context=c)
            else:
                self.assertNotIn(original.instance_id, memory.instances)
                self.assertEqual(wait_remaining_loops(c, memory, 'u0'), 0)
                self.assertFalse(waits)
                facts = measured(c, {'u0': 1, 'u1': 4}, memory)
                self.assertFalse(facts['waits']['u0']['valid'])
                self.assertIn('u0', facts['invalid_wait_actors'])

    def test_read_only_queries_and_actor_identity_reset(self):
        c, memory = context(state()), TacticalMemory()
        memory.combat_progress.history['u0'] = [
            {'loop': 72, 'tag': c.snapshot.allies['u0'].tag, 'action': 1, 'fired': False},
            {'loop': 80, 'tag': c.snapshot.allies['u0'].tag, 'action': 1, 'fired': False}]
        before = copy.deepcopy(memory.__dict__)
        self.assertEqual(wait_remaining_loops(c, memory, 'u0'), 0)
        measured(c, {'u0': 1, 'u1': 4}, memory)
        self.assertEqual(memory.__dict__, before)
        raw = state()
        raw['allies']['u0']['tag'] = 'replacement'
        self.assertEqual(wait_remaining_loops(context(raw), memory, 'u0'), 16)

    def test_baseline_first_wait_reduces_new_g2_phase_budget(self):
        c, memory = context(state()), TacticalMemory()
        memory.combat_progress.history['u0'] = [
            {'loop': 80, 'tag': c.snapshot.allies['u0'].tag, 'action': 1, 'fired': False}]
        plans = CoordinatedEngageSkill({}).propose(c, memory, Telemetry())
        wait = next(p for p in plans if p.controlled_actors == ('u0',)
                    and p.commitment.phase == 'waiting')
        self.assertEqual(wait.proposed_memory_delta['extra']['phase_max_loops'], 8)
        old = CoordinatedEngageSkill({'common_combat': {'joint_assessment_enabled': False}})
        self.assertFalse(any(p.commitment.phase == 'waiting' for p in
                             old.propose(c, memory, Telemetry())))

    def test_ready_shooter_is_never_a_wait_candidate_and_escape_has_no_entry_cost(self):
        raw = state()
        raw['allies']['u0']['position'] = [3., 0.]
        # Rebuild legal menus/relations for a genuinely firing actor.
        raw = scene([((3., 0.), 'marine'), ((1.5, 0.), 'marine')],
                    {'e0': enemy((7.5, 0.), 'hydralisk')}, distance=2.)
        c = context(raw)
        self.assertIsNone(wait_opportunity(c, TacticalMemory(), 'u0', 'e0'))
        self.assertNotIn('u0', measured(c, {'u0': 6, 'u1': 4})['waits'])
        facts = assess_joint(c, {'u0': 5, 'u1': 5})
        # Confirm the fixture is a real risk-reducing branch.
        relieving = [a for a, row in facts['actors'].items() if row['relief']]
        self.assertTrue(relieving)
        result = assess_local_entry(c, {'u0': 5, 'u1': 5}, facts)
        self.assertTrue(all(a not in f['actor_costs'] for a in relieving for f in result['fronts']))

    def test_unreachable_peer_and_other_front_cannot_validate_wait(self):
        c = context(state())
        facts = copy.deepcopy(assess_joint(c, {'u0': 1, 'u1': 4}))
        facts['actors']['u1']['target_etas'] = {'e0': 3.}
        late = assess_local_entry(c, {'u0': 1, 'u1': 4}, facts)
        self.assertFalse(late['waits']['u0']['valid'])
        facts['actors']['u1']['target_etas'] = {}
        facts['actors']['u1']['contingency_target_etas'] = {'e0': 1.}
        backup = assess_local_entry(c, {'u0': 1, 'u1': 4}, facts)
        self.assertFalse(backup['waits']['u0']['valid'])
        raw = state()
        raw['visible_enemies']['e1'] = enemy((30., 0.), 'hydralisk')
        for relations in raw['relations'].values():
            relations['e1'] = dict(relations['e0'], smac_attack_available=False)
        remote = context(raw)
        facts = copy.deepcopy(assess_joint(remote, {'u0': 1, 'u1': 4}))
        facts['actors']['u1']['target_etas'] = {'e1': 1.}
        elsewhere = assess_local_entry(remote, {'u0': 1, 'u1': 4}, facts)
        self.assertFalse(elsewhere['waits']['u0']['valid'])

    def test_contingency_shooter_is_not_mislabeled_as_idle(self):
        c = context(state())
        facts = copy.deepcopy(assess_joint(c, {'u0': 1, 'u1': 4}))
        facts['actors']['u0']['contingency_target_etas'] = {'e0': 1.}
        result = assess_local_entry(c, {'u0': 1, 'u1': 4}, facts)
        self.assertNotIn('u0', result['waits'])

    def test_collapsing_receiver_cannot_supply_timely_cover(self):
        def sample(health):
            raw = scene([((0., 0.), 'marine'), ((3.2, 0.), 'marine')],
                        {'e0': enemy((8., 0.), 'hydralisk', health=90.)}, distance=2.)
            raw['allies']['u1']['health'] = health
            return context(raw)
        healthy = measured(sample(45.), {'u0': 4, 'u1': 6})
        dying = measured(sample(1.), {'u0': 4, 'u1': 6})
        actor_cost = lambda result: next(f['actor_costs']['u0'] for f in result['fronts']
                                         if 'u0' in f['actor_costs'])
        self.assertGreater(actor_cost(dying), actor_cost(healthy))
        raw = state()
        raw['allies']['u1']['health'] = 1.
        c = context(raw)
        self.assertIsNone(wait_opportunity(c, TacticalMemory(), 'u0', 'e0'))
        self.assertFalse(measured(c, {'u0': 1, 'u1': 4})['waits']['u0']['valid'])

    def test_many_receivers_are_bounded_per_front_not_pairwise(self):
        raw = scene([((0., i * .1), 'marine') for i in range(10)],
                    {'e0': enemy((7.5, 0.), 'hydralisk')}, distance=2.)
        c = context(raw)
        result = measured(c, {a: 4 for a in c.snapshot.allies})
        self.assertEqual(len(result['fronts']), 1)
        self.assertLessEqual(abs(result['cost']), 1.)
        stopped = measured(c, {a: 1 for a in c.snapshot.allies})
        self.assertEqual(stopped['cost'], 0.)
        self.assertEqual(len(stopped['invalid_wait_actors']), 10)

    def test_unknown_speed_is_not_a_wait_promise(self):
        raw = state()
        raw['unit_type_profiles']['48'].pop('movement_speed_native')
        c = context(raw)
        self.assertIsNone(wait_opportunity(c, TacticalMemory(), 'u0', 'e0'))
        self.assertNotIn('u0', measured(c, {'u0': 1, 'u1': 4})['waits'])


if __name__ == '__main__':
    unittest.main()
