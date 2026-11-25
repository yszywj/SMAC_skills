"""Offline archive regression: phase changes, native opening, v2-shaped scores."""
import json
from pathlib import Path
import unittest

from decision_tree.zerg import policy_distill as cart
from decision_tree.zerg.policy_repair_v2 import v2_choose
from decision_tree.zerg.policy_route_alt import choose


class TestRoute(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.panel = Path(__file__).resolve().parents[3] / 'runs/zerg_v1_v2_regression_20261124_185519'
        if not cls.panel.is_dir():
            raise unittest.SkipTest('requires frozen November panel')
        cls.model = cart.load_model(cls.panel/'model.json')

    def fixtures(self, pair):
        arm = self.panel/pair/'v2'
        decisions = json.loads((arm/'runner/distill/episode-0001.json').read_text())['decisions']
        with (arm/'runner/events.jsonl').open() as events, (arm/'shadow/shadow.jsonl').open() as shadows:
            obs = (json.loads(line)['payload'] for line in events if '"skill_observation"' in line[:150])
            for pre, sh, d in zip(obs, shadows, decisions, strict=True):
                cards = {c['candidate_id']:c for c in json.loads(sh)['candidates']}
                menu = [cards[k]['proposed_actions'] for k in d['scores'] if cards[k].get('origin') != 'distill_opening_support']
                yield pre, menu, d

    def test_both_phases_and_native_candidates(self):
        for i,(pre,menu,d) in enumerate(self.fixtures('pair-029')):
            if i not in (1, 2):
                continue
            result = choose(self.model, pre, menu)
            self.assertEqual(result['reason'], 'route_' + ('rendezvous' if i == 1 else 'shared_front'))
            self.assertEqual(result['audit']['v2_actions'], list(cart.vector(pre['state'], d['final_actions'])))
            self.assertIn(tuple(result['actions']), [cart.vector(pre['state'], a) for a in menu])
            self.assertTrue(all(a == b or max(a,b)<6 for a,b in zip(result['actions'],result['audit']['v2_actions'])))
            permuted = choose(self.model, pre, reversed(menu + menu))
            self.assertEqual(permuted['scores'], result['scores'][::-1] * 2)
            for k in ('score','actions','unique_count','top_ties','reason'):
                self.assertEqual(permuted[k], result[k])
            singleton = choose(self.model, pre, [d['final_actions']], previous_actions=[1]*10)
            self.assertEqual(singleton['actions'], result['audit']['v2_actions'])
            self.assertFalse(singleton['audit']['v3_triggered'])

    def test_runtime_opening_priority(self):
        for pre,menu,d in self.fixtures('pair-018'):
            if d['selection_reason'] == 'generated_opening_support':
                base = v2_choose(self.model, pre, menu); alt = choose(self.model, pre, menu)
                self.assertEqual(alt['reason'], 'no_existing_formation_fix')
                for k in ('score','actions','scores','top_ties','unique_count','reason'):
                    self.assertEqual(alt[k], base[k])
                return
        self.fail('missing historical opening')
