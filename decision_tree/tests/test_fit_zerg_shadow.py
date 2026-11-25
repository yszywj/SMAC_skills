import unittest

from decision_tree.scripts.linux.fit_zerg_shadow import change_count, evaluate, split
from decision_tree.zerg.dataset import episode_group
from decision_tree.zerg.tests.test_v11 import _records
from decision_tree.zerg.training import ZergTrainingBackend


class ShadowFitTest(unittest.TestCase):
    def test_action_ids_are_compared_numerically(self):
        c = {'current_actions': {'u0': '2', 'u1': '6'}, 'proposed_actions': {'u0': 2, 'u1': 7, 'u2': 3}}
        self.assertEqual(change_count(c), 2)

    def test_stratified_split_and_no_tie_credit_for_list_order(self):
        records = _records(20)
        episodes = {episode_group(r): {'seed': r['seed'], 'roster': r['scene_config_id']}
                    for r in records}
        held = split(episodes, 3, 42)
        self.assertEqual(held, split(dict(reversed(list(episodes.items()))), 3, 42))
        for roster in ('cfg0', 'cfg1'):
            self.assertEqual(sum(episodes[g]['roster'] == roster for g in held), 3)
        for r in records:
            r['teacher_choice'] = r['candidates'][0]['candidate_id']
            for c in r['candidates']:
                c['features'] = {'candidate_action_change_count': 1.0}
        model, report = ZergTrainingBackend().train('imitation', records, scorer='joint_scorer',
            execution_context_hash='1' * 64, source_fingerprint='0' * 64, validation_episodes=held)
        self.assertEqual(report['training']['n_scenarios'], 14)
        self.assertEqual(report['validation']['n_scenarios'], 6)
        metrics = evaluate(model, [r for r in records if episode_group(r) in held])['overall']
        self.assertEqual(metrics['tie_rate'], 1.)
        self.assertAlmostEqual(metrics['tie_neutral_top1'], 1/3)
