"""Regression contracts for risk symmetry and timed ordinary fire support."""
import unittest

from skills.tactical_v1.context import Context, build_snapshot
from skills.tactical_v1.evaluator import Evaluator
from skills.tactical_v1.joint_combat import assess_joint
from .fixtures import enemy, scene, with_questions


def context(raw):
    return Context(build_snapshot(with_questions(raw), raw['game_loop']))


class CommonExchangeRevisionTests(unittest.TestCase):
    def test_holding_or_attacking_cannot_escape_the_same_lethal_exposure_cost(self):
        raw = scene([((0., 0.), 'marine')], {'e0': enemy((3., 0.), 'hydralisk')})
        raw['allies']['u0']['health'] = 3.
        c, scorer = context(raw), Evaluator()
        costs = [scorer.score_joint(c, [], {'u0': a})[1]['risk'] for a in (1, 2, 6)]
        self.assertLessEqual(costs[0], -3.2)
        self.assertEqual(costs[0], costs[1])
        self.assertEqual(costs[0], costs[2])

    def test_explicit_legacy_ablation_keeps_previous_risk_model(self):
        raw = scene([((0., 0.), 'marine')], {'e0': enemy((3., 0.), 'hydralisk')})
        raw['allies']['u0']['health'] = 3.
        c = context(raw)
        old = Evaluator(common_combat={'joint_assessment_enabled': False})
        self.assertGreater(old.score_joint(c, [], {'u0': 6})[1]['risk'], -3.2)

    def cooling_pair(self, cooldown=16.):
        raw = scene([((0., 0.), 'marine'), ((0., 1.), 'marauder')],
                    {'e0': enemy((4.75, 1.), 'hydralisk')})
        for unit in raw['allies'].values():
            unit.update(weapon_cooldown_loops=cooldown,
                        orders=[{'kind': 'attack', 'target_unit': 'e0'}])
        return context(raw)

    def test_cooling_peer_has_same_retained_volley_before_and_after(self):
        row = assess_joint(self.cooling_pair(), {'u0': 6, 'u1': 6})['actors']['u0']
        self.assertGreater(row['support_before'], 0.)
        self.assertAlmostEqual(row['support_before'], row['support_after'])
        self.assertEqual(row['support_loss'], 0.)

    def test_cooling_peer_leaving_range_loses_upcoming_cover(self):
        c = self.cooling_pair()
        stay = assess_joint(c, {'u0': 6, 'u1': 6})['actors']['u0']
        leave = assess_joint(c, {'u0': 6, 'u1': 5})['actors']['u0']
        self.assertAlmostEqual(stay['support_before'], leave['support_before'])
        self.assertGreater(leave['support_loss'], 0.)
        self.assertLess(leave['support_after'], stay['support_after'])

    def test_shot_outside_shared_window_is_not_counted_on_either_side(self):
        row = assess_joint(self.cooling_pair(32.), {'u0': 6, 'u1': 6})['actors']['u0']
        self.assertEqual(row['support_before'], 0.)
        self.assertEqual(row['support_after'], 0.)

    def test_later_shot_gets_less_credit_without_changing_horizons(self):
        ready = assess_joint(self.cooling_pair(0.), {'u0': 6, 'u1': 6})['actors']['u0']
        later = assess_joint(self.cooling_pair(16.), {'u0': 6, 'u1': 6})['actors']['u0']
        self.assertGreater(ready['support_before'], later['support_before'])
        self.assertEqual(ready['support_window_h'], later['support_window_h'])
        self.assertEqual(later['support_before'], later['support_after'])

    def test_cached_branch_cannot_rewrite_an_earlier_support_result(self):
        c = self.cooling_pair()
        first = assess_joint(c, {'u0': 6, 'u1': 5})['actors']['u0']
        saved = dict(first)
        assess_joint(c, {'u0': 2, 'u1': 6})
        self.assertEqual(first, saved)
        self.assertEqual(first, assess_joint(c, {'u1': 5, 'u0': 6})['actors']['u0'])


if __name__ == '__main__':
    unittest.main()
