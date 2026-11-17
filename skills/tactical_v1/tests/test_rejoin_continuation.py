"""Continuation protects observable useful progress, not task labels or age."""
import copy
import unittest

from skills.tactical_v1.combat_groups import rejoin_continuation_opportunity
from skills.tactical_v1.context import Context, build_snapshot
from skills.tactical_v1.evaluator import Evaluator
from skills.tactical_v1.memory import TacticalMemory
from skills.tactical_v1.skills.combat_rejoin import CombatRejoinSkill
from skills.tactical_v1.telemetry import Telemetry
from .fixtures import enemy, scene, with_questions


class RejoinContinuationTests(unittest.TestCase):
    def context(self, xs=(8., 7., 16.), loop=80):
        state = scene([((x, 0.), 'stalker') for x in xs],
                      enemies={'e0': enemy((18., 0.))}, loop=loop)
        return Context(build_snapshot(with_questions(state), loop))

    def observe(self, memory, context):
        memory.begin_step(context.snapshot)
        memory.observe(context.snapshot)
        memory.update_task_facts(context.snapshot, context)

    def start(self):
        context, memory, skill = self.context(), TacticalMemory(), CombatRejoinSkill({})
        self.observe(memory, context)
        plan = skill.propose(context, memory, Telemetry())[0]
        memory.commit(context.snapshot, plan.current_actions, [plan], context=context)
        return context, memory, skill, memory.instances[plan.instance_id]

    def progressing(self):
        _, memory, skill, instance = self.start()
        context = self.context(xs=(9.5, 7., 16.), loop=96)
        self.observe(memory, context)
        return context, memory, skill, instance

    def test_only_recently_observed_unfinished_joiner_is_eligible(self):
        initial, memory, skill, instance = self.start()
        self.assertEqual(rejoin_continuation_opportunity(initial, instance), {})
        current = self.context(xs=(9.5, 7., 16.), loop=96)
        self.observe(memory, current)
        before = copy.deepcopy(instance.extra)
        weights = rejoin_continuation_opportunity(current, instance)
        self.assertEqual(set(weights), {'u0'})
        self.assertGreater(weights['u0'], 0.)
        self.assertLessEqual(weights['u0'], 1.)
        skill.propose(current, memory, Telemetry())
        self.assertEqual(instance.extra, before)
        self.assertEqual(instance.start_loop, 80)
        self.assertEqual(instance.max_loops, 64)

    def test_stale_progress_and_insufficient_remaining_budget_have_no_claim(self):
        context, _, _, instance = self.progressing()
        instance.extra['joiner_progress']['u0']['last_progress_loop'] = 72
        self.assertEqual(rejoin_continuation_opportunity(context, instance), {})
        instance.extra['joiner_progress']['u0']['last_progress_loop'] = 96
        instance.max_loops = 24  # only one window remains; ETA is longer
        self.assertEqual(rejoin_continuation_opportunity(context, instance), {})
        instance.max_loops = 16
        self.assertEqual(rejoin_continuation_opportunity(context, instance), {})

    def test_collapsing_cover_releases_continuation_without_new_hit(self):
        _, memory, _, instance = self.progressing()
        state = scene([((9.5, 0.), 'stalker'), ((7., 0.), 'stalker'),
                       ((16., 0.), 'stalker')], enemies={'e0': enemy((18., 0.))}, loop=96)
        state['allies']['u2']['health'] = 8.
        context = Context(build_snapshot(with_questions(state), 96))
        self.assertEqual(rejoin_continuation_opportunity(context, instance), {})

    def test_cost_uses_final_progress_not_skill_label(self):
        context, memory, skill, instance = self.progressing()
        evaluator = Evaluator()
        plan = skill.propose(context, memory, Telemetry())[0]
        actions = plan.current_actions
        positions = {a: context.predict_position(a, action).position for a, action in actions.items()}
        self.assertEqual(evaluator._rejoin_abandonment(context, actions, positions, memory, set()), 0.)
        stalled = dict(actions, u0=1)
        stationary = {a: context.predict_position(a, action).position for a, action in stalled.items()}
        loss = evaluator._rejoin_abandonment(context, stalled, stationary, memory, set())
        self.assertGreater(loss, 0.)
        self.assertLessEqual(loss, 1.)
        # Existing minimum commitment is a separate cost, never doubled.
        instance.extra['min_loops'] = 24
        self.assertEqual(evaluator._rejoin_abandonment(context, stalled, stationary, memory, set()), 0.)

    def test_real_contact_and_expired_task_do_not_charge_abandonment(self):
        context, memory, _, instance = self.progressing()
        evaluator = Evaluator()
        instance.max_loops = 16
        actions = {'u0': 1, 'u1': 1, 'u2': 6}
        self.assertEqual(evaluator._rejoin_abandonment(context, actions, {}, memory, set()), 0.)
        instance.max_loops = 64
        contact = self.context(xs=(13., 7., 16.), loop=96)
        actions['u0'] = 6
        self.assertEqual(evaluator._rejoin_abandonment(contact, actions, {}, memory, set()), 0.)

    def test_receiver_movement_does_not_leave_joiner_chasing_old_anchor(self):
        state = scene([((0., 0.), 'stalker'), ((0., 8.), 'stalker')],
                      enemies={'e0': enemy((0., 10.))})
        context = Context(build_snapshot(with_questions(state), 80))
        data = {'joiners': ['u0'], 'supporters': ['u1'], 'anchor': [-8., 0.],
                'support_radius': 6., 'roles': {'u0': 'join', 'u1': 'cover'}}
        before = copy.deepcopy(data)
        self.assertEqual(CombatRejoinSkill({})._join_move(context, 'u0', data), 2)
        self.assertEqual(data, before)


if __name__ == '__main__':
    unittest.main()
