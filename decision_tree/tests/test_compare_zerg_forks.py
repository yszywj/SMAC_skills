import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from decision_tree.scripts.linux import compare_zerg_forks as cmp


class PairComparison(unittest.TestCase):
    def test_pair_deltas_and_missing_horizons(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            p = dict(game_loop=10, seed=12, roster='test', changed={'u0': {'A': 3, 'B': 4}})
            (run/'plan.json').write_text(json.dumps({'pairs': [p]}))
            (run/'design.json').write_text(json.dumps({'pair_indices': [1]}))
            for i, arm in enumerate('AB'):
                path = run / 'pair-001' / arm / 'runner'
                path.mkdir(parents=True)
                ep = dict(initial_state_sha256='same', scenario_sha256='same',
                          reward=1+i, outcome=['loss','win'][i])
                (path/'episodes.json').write_text(json.dumps([ep]))
                ally = dict(tag='1', alive=True, health=10-i, position=[2*i, 1])
                obs = dict(event='skill_observation', payload={'state': {'game_loop': 18, 'allies': {'u0': ally}}})
                (path/'events.jsonl').write_text(json.dumps(obs) + '\n')
            with patch.object(cmp, 'validate_arm', return_value={'engine_receipt': 'accepted'}):
                report = cmp.compare(run)
                self.assertEqual((report['accepted_pairs'], report['more_wins'], report['mean_return_delta']), (1, 1, 1))
                after = report['pairs'][0]['after_macro_steps']
                self.assertEqual(after[1]['max_ally_position_distance'], 2)
                self.assertIsNone(after[3])
            with patch.object(cmp, 'validate_arm', return_value={'engine_receipt': 'unknown'}):
                report = cmp.compare(run)
                self.assertEqual(report['accepted_pairs'], 0)
                self.assertIsNone(report['mean_return_delta'])
                self.assertEqual(len(report['excluded']), 1)
