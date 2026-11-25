import unittest
from decision_tree.scripts.linux.evaluate_zerg_v2_gate import assess
from decision_tree.scripts.linux.zerg_severe_panel import partition


class PanelGateTest(unittest.TestCase):
    def test_draw_and_severe(self):
        a=dict(result='draw',enemy_left=7,reward=1); b=dict(result='win',enemy_left=0,reward=3)
        c=dict(result='loss',enemy_left=5,reward=0)
        out=partition([(a,b,{}),(b,c,{}),(c,c,{})])
        self.assertEqual((out['tree_wins'],out['nonwin_to_win'],out['win_to_nonwin']),(1,1,1))
        self.assertEqual((out['both_severe'],out['tree_severe']),(1,2))

    def test_tiny_pilot_even_if_all_win_is_not_confirmation(self):
        a=dict(outcome='loss',enemy_elimination_fraction=.4,reward=4);b=dict(outcome='win',enemy_elimination_fraction=1.,reward=15)
        out=assess([{'v1':a,'v2':b}]*4,fresh=False,resamples=16)
        self.assertEqual((out['gained'],out['lost'],out['severe_v1'],out['severe_v2']),(4,0,4,0))
        self.assertEqual(out['paired_exact_two_sided_win_p'],.125)
        self.assertEqual(out['decision'],'not established')
