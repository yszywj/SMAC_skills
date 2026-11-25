"""Split-search regression tests. All rows, including timing inputs, are synthetic."""
import math
import os
import random
import time
import unittest
from unittest.mock import patch

from decision_tree.zerg import training as t


def exhaustive(rows, names, weights, labels, ix, spec, depth=0):
    """Oracle: partition and recompute both weighted residual sums at every cut."""
    def stats(part):
        mass = math.fsum(weights[i] for i in part)
        if mass <= 0:
            return 0.0, 0.0
        avg = math.fsum(weights[i] * labels[i] for i in part) / mass
        return avg, math.fsum(weights[i] * (labels[i] - avg)**2 for i in part)

    def groups(part):
        return len({rows[i]['group_id'] for i in part})

    def supported(part):
        return len(part) >= spec['minimum_leaf_support'] and groups(part) >= spec['minimum_leaf_groups']

    avg, loss = stats(ix)
    best = None
    if depth < spec['max_depth'] and loss > 1e-12:
        for name in names:
            values = sorted({rows[i]['features'][name] for i in ix})
            for low, high in zip(values, values[1:]):
                cut = low / 2 + high / 2
                if cut >= high:
                    cut = low
                l = [i for i in ix if rows[i]['features'][name] <= cut]
                r = [i for i in ix if rows[i]['features'][name] > cut]
                if supported(l) and supported(r):
                    score = stats(l)[1] + stats(r)[1]
                    if score < loss - 1e-12:
                        best, loss = (name, cut, l, r), score
    if best is None:
        return dict(value=avg, samples=len(ix), groups=groups(ix), supported=supported(ix))
    name, cut, l, r = best
    return dict(feature=name, threshold=cut,
                left=exhaustive(rows, names, weights, labels, l, spec, depth+1),
                right=exhaustive(rows, names, weights, labels, r, spec, depth+1))


def row(g, x, y, **more):
    return dict(group_id=str(g), decision_key=str(g), features=dict(x=x, **more), y=y)


class ThresholdSweep(unittest.TestCase):
    def check(self, rows, names=('x',), weights=None, **capacity):
        spec = dict(t.DEFAULT_CAPACITY, minimum_leaf_support=1, minimum_leaf_groups=1, max_depth=3)
        spec.update(capacity)
        body, _ = t.fit_threshold_forest(rows, names, 'y', capacity=spec, weights=weights)
        weights = t.group_weights_for(rows) if weights is None else weights
        ix = list(range(len(rows)))
        labels = [r['y'] for r in rows]
        current = [body['base_value']] * len(rows)
        for j, tree in enumerate(body['trees']):
            sample, targets = ix, labels
            if spec['estimator_type'] == 'bagged_threshold_tree':
                groups = sorted({r['group_id'] for r in rows})
                counts = t._bootstrap_scene_counts(groups, len(groups), spec['random_seed'], j)
                sample = [i for i in ix for _ in range(counts[rows[i]['group_id']])]
            if spec['estimator_type'] == 'boosted_threshold_tree':
                targets = [a-b for a, b in zip(labels, current)]
            expected = exhaustive(rows, names, weights, targets, sample, spec)
            self.assertEqual(tree['root'], expected)
            if spec['estimator_type'] == 'boosted_threshold_tree':
                for i in ix:
                    node = expected
                    while 'feature' in node:
                        node = node['left' if rows[i]['features'][node['feature']] <= node['threshold'] else 'right']
                    current[i] += spec['learning_rate'] * node['value']
        return body['trees'][0]['root']

    def test_weighted_forests(self):
        rng = random.Random(829)
        rows = [row(g, rng.randrange(9), rng.uniform(-2, 3), z=rng.randrange(6))
                for g in range(14) for _ in range(1 + g % 6)]
        for i, r in enumerate(rows):
            r['decision_key'] += ':' + str(i % 2)
        rng.shuffle(rows)
        for weights in (None, [rng.choice([0, 0.03, 0.2, 1.5, 10.1]) for _ in rows]):
            for est, count in [('single', 1), ('bagged', 5), ('boosted', 5)]:
                with self.subTest(est=est, custom_weights=weights is not None):
                    self.check(rows, ('z', 'x'), weights, estimator_type=est+'_threshold_tree',
                               number_of_trees=count, learning_rate=0.3, minimum_leaf_support=4,
                               minimum_leaf_groups=3)

    def test_bootstrap_occurrences_and_distinct_groups(self):
        rows = [row(g, x, x) for g, x in enumerate([0, 0, 1, 1, 2, 2])]
        counts = dict(zip('012345', [3, 0, 1, 0, 1, 1]))
        with patch.object(t, '_bootstrap_scene_counts', return_value=counts):
            root = self.check(rows, estimator_type='bagged_threshold_tree', number_of_trees=2,
                              minimum_leaf_support=2, minimum_leaf_groups=2)
        self.assertEqual(root['threshold'], 1.5)  # 0.5 has three left rows, only one group
        self.assertEqual((root['left']['samples'], root['left']['groups']), (4, 2))

    def test_ties_midpoint_and_tolerance(self):
        for delta in [0., 1e-13, 1e-10]:
            rows = [row(i, i, y, z=i) for i, y in enumerate([1., 0., 1.+delta])]
            root = self.check(rows, ('z', 'x'))
            self.assertEqual((root['feature'], root['threshold']), ('z', 1.5 if delta > 1e-12 else 0.5))
        low = math.nextafter(1., 2.)
        high = math.nextafter(low, 2.)  # midpoint rounds up; threshold must be low
        rows = [row(i, v, i//2) for i, v in enumerate([high, high, low, low])]
        self.assertEqual(self.check(rows, minimum_leaf_support=2)['threshold'], low)
        self.assertNotIn('feature', self.check(rows, minimum_leaf_support=3))
        self.assertFalse(self.check(rows, minimum_leaf_groups=5)['supported'])
        self.assertNotIn('feature', self.check([row(0, 0, 0), row(1, 1, 1e-6)]))

    def test_loss_roundoff(self):
        rows = [row(i, i, 1e15 + d) for i, d in enumerate([0, 3, 4, 9, 11, 12, 5])]
        self.check(rows, weights=[1e10, 2e-8, 0.1, 0.3, 1, 1e-9, 0.8])
        self.check(rows, weights=[10, -1, 2, 3, 1, 0, 1])
        # Exact least squares prefers 3.5 here, but stats() rounds child
        # means to the spacing at 1e16. Keep the legacy decision at 0.5.
        rows = [row(i, i, 1e16 + d) for i, d in enumerate([8, 12, 10, 12, 6, 6])]
        root = self.check(rows, weights=[5, 1, 5, 1, 0.1, 1], max_depth=1)
        self.assertEqual(root['threshold'], 0.5)

    def test_high_cardinality_row_visits(self):
        # Deterministic complexity guard rather than a machine-speed assertion.
        visits = 0

        class Counted(dict):
            def __getitem__(self, key):
                nonlocal visits
                visits += 1
                return super().__getitem__(key)

        rows = [row(i % 480, i, i % 7) for i in range(480 * 4)]
        with patch.object(t, '_row_vector', side_effect=lambda r, *_: Counted(r['features'])):
            t.fit_threshold_forest(rows, ['x'], 'y', capacity={'max_depth': 1})
        self.assertLessEqual(visits, 4 * len(rows))

    @unittest.skipUnless(os.getenv('BENCH_THRESHOLD_SWEEP'), 'opt-in synthetic timing')
    def test_timing(self):
        rng = random.Random(431)
        names = ['f%d' % i for i in range(12)]
        rows = []
        for g in range(480):
            for step in range(24):
                for c in range(4):
                    features = {name: rng.random() for name in names}
                    rows.append(dict(group_id=g, decision_key=step, features=features,
                                     y=float(features[names[0]] + features[names[1]] > 1)))
        spec = dict(t.DEFAULT_CAPACITY)
        small = rows[:8*24*4]
        tic = time.perf_counter()
        oracle = exhaustive(small, names, t.group_weights_for(small), [r['y'] for r in small],
                            list(range(len(small))), spec)
        old = time.perf_counter() - tic
        for sample in [small, rows]:
            tic = time.perf_counter()
            body, _ = t.fit_threshold_forest(sample, names, 'y', capacity=spec)
            elapsed = time.perf_counter() - tic
            if sample is small:
                self.assertEqual(oracle, body['trees'][0]['root'])
            print(f'{len(sample)} rows: optimized {elapsed:.3f}s' +
                  (f', exhaustive {old:.3f}s' if sample is small else ''))
