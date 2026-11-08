import unittest
from decision_tree.local.study import select_anchors


class LocalAnchorCoverageTests(unittest.TestCase):
    def test_late_retreat_is_covered_without_using_rewards(self):
        events = [dict(event='local_decision', step=t, candidates=[{'family': f} for f in families], reward=reward)
                  for t, families, reward in [(0, ['regroup_step'], 10), (4, ['retarget'], 5),
                                             (8, ['retarget', 'retreat_veto', 'retreat_step'], -10)]]
        selected = select_anchors(events, 2, 4, 'family_coverage')
        self.assertEqual([e['step'] for e in selected], [0, 8])
        for event in events:
            event['reward'] *= -100
        self.assertEqual([e['step'] for e in select_anchors(events, 2, 4, 'family_coverage')], [0, 8])

    def test_frozen_policy_first_opportunity_is_mandatory(self):
        events = [dict(event='local_decision', step=t, candidates=[{'family': 'retarget'}]) for t in range(12)]
        selected = select_anchors(events, 3, 4, 'family_coverage', required_step=7)
        steps = [e['step'] for e in selected]
        self.assertIn(7, steps)
        self.assertTrue(all(abs(a-b) >= 4 for i, a in enumerate(steps) for b in steps[i+1:]))
        with self.assertRaisesRegex(ValueError, 'missing'):
            select_anchors(events, 3, 4, 'family_coverage', required_step=99)

    def test_empty_episode_is_not_filled_with_fake_candidates(self):
        self.assertEqual(select_anchors([], 3, 4, 'family_coverage'), [])


if __name__ == '__main__':
    unittest.main()
