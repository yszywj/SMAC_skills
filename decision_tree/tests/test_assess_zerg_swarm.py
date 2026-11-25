import unittest

from decision_tree.scripts.linux.assess_zerg_swarm import metrics


class SwarmReport(unittest.TestCase):
    def test_distinguishes_losses_from_draws(self):
        w=dict(outcome='win',reward=20,enemy_elimination_fraction=1.)
        d=dict(outcome='draw',reward=8,enemy_elimination_fraction=.5)
        s=dict(outcome='loss',reward=6,enemy_elimination_fraction=.5)
        pairs=[dict(v2=s,swarm=w),dict(v2=w,swarm=d),dict(v2=w,swarm=s)]
        r=metrics(pairs)
        self.assertEqual((r['gained'],r['lost'],r['win_to_loss']),(1,2,1))
        self.assertEqual((r['severe_to_win'],r['win_to_severe'],r['posthoc_oracle_wins']),(1,1,3))
        self.assertEqual(r['paired_win_exact_p'],1.)

    def test_unmatched_does_not_become_zero_gain(self):
        self.assertEqual(metrics([]),{'pairs':0})
