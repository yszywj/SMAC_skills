"""Route, coverage and output invariants for a local counter-fire pair."""
import unittest

from skills.tactical_v1.context import Context, build_snapshot, distance
from skills.tactical_v1.hydra_support import counter_fire_support_plans, pressured_hydras
from .fixtures import enemy, scene, with_questions
from .test_hydra_support import support_scene


def candidates(state, actor='u0', supporter='u1'):
    context = Context(build_snapshot(with_questions(state), 1))
    pairs = counter_fire_support_plans(context, None, 'test')
    return context, [p for p in pairs if p.goal_data['support_pair'] == [actor, supporter]]


class CounterFireRegressionTests(unittest.TestCase):
    def test_enemy_coverage_honors_observation_and_enemy_weapon(self):
        state = scene([((0., 0.), 'hydralisk')], {'e0': enemy((4., 0.), 'zergling')})
        rel = state['relations']['u0']['e0']
        # A Hydralisk can shoot four units; the opposing Zergling cannot.
        for flag, status in [(False, 'out_of_range'), (None, 'unknown')]:
            rel.update(in_enemy_weapon_range=flag, enemy_weapon_range_status=status)
            context = Context(build_snapshot(with_questions(state), 1))
            self.assertFalse(pressured_hydras(context))
        state['visible_enemies']['e0']['position'] = [1., 0.]
        self.assertTrue(pressured_hydras(Context(build_snapshot(with_questions(state), 1))))

    def test_supporter_keeps_a_ready_shot_on_another_enemy(self):
        allies = [((0., 0.), 'hydralisk'), ((-.9, 1.8), 'hydralisk')]
        threats = {'e0': enemy((4.7, 0.), 'hydralisk')}
        _, plans = candidates(scene(allies, threats))
        self.assertTrue(any(p.goal == 'e0' for p in plans))
        threats['e1'] = enemy((-2., 2.), 'zergling')
        state = scene(allies, threats)
        state['relations']['u0']['e1'].update(in_enemy_weapon_range=False,
                                             enemy_weapon_range_status='out_of_range')
        context, plans = candidates(state)
        self.assertGreater(context.fire_estimate('u1', 'e1').window_shots, 0)
        self.assertFalse(any(p.goal == 'e0' for p in plans))

    def test_bounded_two_step_route_makes_progress_without_claiming_damage(self):
        state = support_scene((6., 12.), (7.5, 10.), (4., 10.), smac_limit=2.2)
        context, plans = candidates(state)
        plan = next(p for p in plans if p.goal_data['prospective_fire'])
        point = context.predict_position('u1', plan.current_actions['u1']).position
        gap = distance(point, context.snapshot.enemies['e0'].position)
        self.assertGreater(gap, context.effective_attack_reach('u1', 'e0'))
        self.assertLess(gap, context.distance('u1', 'e0'))
        self.assertEqual(plan.claims.fire, (('e0', 12.),))

    def test_blocked_support_route_and_attack_before_withdrawal(self):
        # The only west step opening B's mask would enter A's body.
        _, plans = candidates(support_scene((6., 10.), (7.5, 10.), (4., 10.), smac_limit=3.))
        self.assertFalse(plans)
        for cooldown, expected in [(0., {'attack'}), (1., {'attack', 'move'})]:
            with self.subTest(cooldown=cooldown):
                context, plans = candidates(support_scene(
                    (6., 10.), (9., 12.), (10., 10.), a_cooldown=cooldown))
                self.assertEqual({context.decode('u0', p.current_actions['u0']).kind
                                  for p in plans}, expected)
                for plan in plans:
                    self.assertEqual(context.decode('u1', plan.current_actions['u1']).kind,
                                     'attack')

    def test_supporter_limit_is_applied_after_route_feasibility(self):
        state = scene([((0., 0.), 'hydralisk'), ((-.9, 1.8), 'hydralisk'),
                       ((-.8, -1.8), 'hydralisk'), ((-1., 2.8), 'hydralisk')],
                      {'e0': enemy((4.7, 0.), 'hydralisk')})
        for actor in ('u1', 'u2'):
            state['movement_candidates'][actor] = {}
        _, plans = candidates(state, supporter='u3')
        self.assertTrue(any(plan.goal_data['prospective_fire'] for plan in plans))

    def test_stationary_cooling_member_keeps_legal_cover_without_damage_credit(self):
        state = support_scene((6., 10.), (8., 10.), (9., 10.), a_cooldown=12.)
        state['movement_candidates']['u0'] = {}
        context, plans = candidates(state)
        self.assertTrue(plans)
        for plan in plans:
            self.assertFalse(plan.goal_data['withdrawing'])
            self.assertEqual(context.decode('u0', plan.current_actions['u0']).kind, 'attack')
            self.assertEqual(plan.claims.fire, (('e0', 12.),))

    def test_one_slot_for_feasible_unanswered_pressure(self):
        state = scene([((6., 10.), 'hydralisk'), ((7., 12.), 'hydralisk'),
                       ((23., 10.), 'hydralisk'), ((21.5, 12.), 'hydralisk')],
                      {'e0': enemy((10., 10.), 'hydralisk'),
                       'e1': enemy((28., 10.), 'hydralisk')})
        state['allies']['u0']['health'] = 26.
        state['allies']['u1']['health'] = 30.
        # u2's own ready shot on e1 cannot count as covering fire from a peer.
        for open_route, expected in [(True, {'u0', 'u2'}), (False, {'u0', 'u1'})]:
            if not open_route:
                state['movement_candidates']['u3'] = {}
            context = Context(build_snapshot(with_questions(state), 1))
            plans = counter_fire_support_plans(context, None, 'test')
            self.assertEqual({p.goal_data['support_pair'][0] for p in plans}, expected)
            if open_route:
                self.assertTrue(any(p.goal_data['support_pair'] == ['u2', 'u3']
                                    and p.goal_data['prospective_fire'] for p in plans))
