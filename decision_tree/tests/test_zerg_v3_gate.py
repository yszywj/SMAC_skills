import unittest

from decision_tree.scripts.linux.evaluate_zerg_v3_gate import summarize


class TestV3Gate(unittest.TestCase):
    def test_development_excludes_unreproduced_endpoints(self):
        old=dict(initial='start', outcome='loss',reward=2.,enemy_elimination_fraction=.3)
        won=dict(old,outcome='win',reward=18.,enemy_elimination_fraction=1.)
        rows=[dict(pair_id='a',scenario_sha256='s1',v2=old,v3=won),
              dict(pair_id='b',scenario_sha256='s2',v2=old,v3=old)]
        report=dict(pairs=2,paired_outcomes=rows,replay_batch='regression')
        parent=dict(paired_outcomes=[dict(scenario_sha256='s1',v1=old,v2=old),
                                    dict(scenario_sha256='s2',v1=old,v2=dict(old,reward=3.))])
        result=summarize(report,parent,resamples=16)
        self.assertEqual(result['gained'],1)
        self.assertEqual(result['historical_v2_endpoints_not_reproduced'],['b'])
        self.assertEqual(result['historical_cohorts']['historic_both_severe']['v3_win'],1)
        self.assertEqual(result['decision'],'not established')

    def test_duplicate_and_unpaired_rejected(self):
        a=dict(initial='one',outcome='loss',reward=2,enemy_elimination_fraction=.4)
        row=dict(scenario_sha256='s',v2=a,v3=dict(a,initial='two'))
        for rows in ([row],[row,row]):
            with self.assertRaises(ValueError):summarize(dict(pairs=len(rows),paired_outcomes=rows,replay_batch=None))
