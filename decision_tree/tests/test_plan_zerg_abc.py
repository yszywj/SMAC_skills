import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from decision_tree.scripts.linux import plan_zerg_abc as planner
from decision_tree.scripts.linux import compare_zerg_forks


class ScreenTest(unittest.TestCase):
    def test_shadow_peek_before_commit(self):
        from decision_tree.scripts.linux.zerg_fork_preflight import replay_prefix
        from types import SimpleNamespace
        calls = []
        fork = SimpleNamespace(anchor=0, step=lambda *args: ([1],{}),
                               peek_c=lambda *args: calls.append('peek') or {'actions': {'u0':2}},
                               committed=lambda *args: calls.append('checked'))
        ctrl = SimpleNamespace(active=True,manager=None,commit_final_actions=lambda _: calls.append('commit'))
        obs = {'event':'skill_observation','episode':1,'step':0,'payload':{'state':{'game_loop':2}}}
        proposal = replay_prefix(fork,ctrl,[obs],peek_hash='test')
        self.assertEqual(proposal['actions']['u0'],2)
        self.assertEqual(calls,['peek','commit','checked'])

    def test_no_overwriting_frozen_plans(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp)/'plan.json'; planner.write(path, {'pairs':[1]})
            with self.assertRaises(FileExistsError): planner.write(path, {})
            self.assertEqual(json.loads(path.read_text())['pairs'], [1])

    def test_old_comparison_refuses_three_arms(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp); planner.write(root/'plan.json', {'pairs':[{'C':{}}]}); planner.write(root/'design.json',{})
            with self.assertRaisesRegex(ValueError,'A/B/C'):
                compare_zerg_forks.compare(root)
