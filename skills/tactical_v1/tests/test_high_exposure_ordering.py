"""The high-risk tail must distinguish relief without changing normal trades."""
import unittest

from skills.tactical_v1.context import Context, build_snapshot
from skills.tactical_v1.evaluator import Evaluator, reserve_risk_penalty
from .fixtures import enemy, scene, with_questions


class HighExposureOrderingTests(unittest.TestCase):
    def test_normal_exchange_scale_is_preserved_and_extreme_cost_is_bounded(self):
        for ratio in (0., .25, .75, 1., 1.5, 2.):
            self.assertEqual(reserve_risk_penalty(ratio), ratio * ratio)
        values = [reserve_risk_penalty(r) for r in (2., 2.85, 95./30., 4., 8., 1000.)]
        self.assertEqual(values, sorted(set(values)))
        self.assertLess(values[-1], 8.)

    def test_more_exposure_is_not_a_tie_after_both_routes_exceed_twice_reserve(self):
        evaluator = Evaluator()
        costs = []
        for health in (5., 4., 3.):
            raw = scene([((0., 0.), 'marine')], {'e0': enemy((2., 0.), 'hydralisk')})
            raw['allies']['u0']['health'] = health
            c = Context(build_snapshot(with_questions(raw), 1))
            costs.append(evaluator.score_joint(c, [], {'u0': 6})[1]['risk'])
        self.assertGreater(costs[0], costs[1])
        self.assertGreater(costs[1], costs[2])

    def test_same_geometry_gets_same_tail_for_attack_stop_and_move(self):
        raw = scene([((0., 0.), 'marine')], {'e0': enemy((2., 0.), 'hydralisk')})
        raw['allies']['u0']['health'] = 3.
        c = Context(build_snapshot(with_questions(raw), 1))
        evaluator = Evaluator()
        costs = [evaluator.score_joint(c, [], {'u0': a})[1]['risk'] for a in (1, 2, 6)]
        self.assertAlmostEqual(costs[0], costs[1])
        self.assertAlmostEqual(costs[0], costs[2])


if __name__ == '__main__':
    unittest.main()
