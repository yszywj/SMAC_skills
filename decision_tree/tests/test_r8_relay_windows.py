import unittest

from decision_tree.scripts.linux.r8_relay_windows import match


def frames(n):
    return [dict(loop=2+8*i,obs=f's{i}',fb='no-micro',actions=[4],accepted=[True],
                 metrics=dict(alive=2,capable=1,shared=1,hp=30)) for i in range(n)]


class MatchedWindows(unittest.TestCase):
    def test_first_action_is_not_a_proven_outcome(self):
        a,b=frames(12),frames(12);b[1].update(actions=[3],relay=dict(owner='custom',actions=[3],r6_actions=[4]));b[5]['metrics']['capable']=2
        b[9]['metrics']['alive']=1
        status,w=match(a,b)
        self.assertEqual((status,w['4']['capable'],w['8']['alive']),('macro',1,-1))

    def test_end_is_missing_not_zero(self):
        a,b=frames(4),frames(4);b[1].update(actions=[3],relay=dict(owner='menu',actions=[3],r6_actions=[4]))
        self.assertEqual(match(a,b),('macro',{}))

    def test_prior_unmatched_micro_or_state_excludes_comparison(self):
        a,b=frames(10),frames(10);b[4]['actions']=[3];b[2]['fb']='another-order'
        self.assertEqual(match(a,b),('different_execution_feedback',{}))
        b[2]['fb']='no-micro';b[4]['obs']='different'
        self.assertEqual(match(a,b),('different_pre_observation',{}))
