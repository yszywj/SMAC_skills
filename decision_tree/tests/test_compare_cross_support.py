import unittest
from decision_tree.scripts.linux.compare_cross_support import cohorts,summarize


class CrossSupportReportTest(unittest.TestCase):
    def test_cohorts_keep_actual_previous_gains_and_drops(self):
        a=dict(strata=['both_loss'],online_outcome='loss',outcome='win')
        self.assertEqual(cohorts(a),['all','both_loss','previous_gain'])
        self.assertIn('previous_lost',cohorts(dict(a,online_outcome='win',outcome='draw')))

    def test_changes_incremental_not_both_against_router(self):
        def make(outcomes):
            r={}
            for k,o in zip(('parent','guard','front'),outcomes):
                r.update({k:o,k+'_severe':False,k+'_reward':int(o=='win'),k+'_trace':{'cross':2}})
            return r
        group=summarize(list(map(make,[('loss','win','loss'),('win','win','win')])))
        self.assertEqual(group['parent_to_guard']['net_wins'],1)
        self.assertEqual(group['guard_to_front']['lost'],1)
        self.assertEqual(group['front']['cross_target_within_four'],2)
