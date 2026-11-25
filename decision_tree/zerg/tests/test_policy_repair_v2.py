"""No engine. The integration case is read from a real matched severe loss."""
import copy
import json
from pathlib import Path
import unittest

from decision_tree.zerg import policy_distill as v1
from decision_tree.zerg.policy_repair_v2 import formation, v2_choose


class TestFormation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        panel = Path(__file__).resolve().parents[3] / 'runs/zerg_policy_eval_20261124_162949'
        arm = panel / 'pair-159/tree'
        if not arm.is_dir():
            raise unittest.SkipTest('requires the November development archive')
        cls.model = v1.load_model(panel/'model.json')
        with (arm/'runner/events.jsonl').open() as events:
            cls.pre = next(json.loads(l)['payload'] for l in events if '"skill_observation"' in l[:150])
        with (arm/'shadow/shadow.jsonl').open() as shadows:
            row = json.loads(next(shadows))
        decision = json.loads((arm/'runner/distill/episode-0001.json').read_text())['decisions'][0]
        cls.menu = [c['proposed_actions'] for c in row['candidates'] if c['candidate_id'] in decision['scores']]
        cls.actual = list(v1.vector(cls.pre['state'],decision['final_actions']))

    def test_real_repair_preserves_attacks_and_uses_existing_vector(self):
        out = v2_choose(self.model,self.pre,self.menu)
        self.assertEqual(out['reason'],'protect_hydra_support_gap')
        self.assertEqual(out['audit']['v1_actions'],self.actual)
        i = [list(v1.vector(self.pre['state'],a)) for a in self.menu].index(out['actions'])
        audit = out['audit']
        self.assertLessEqual(audit['candidate_escort_costs'][i], audit['v1_escort_cost']-.5)
        self.assertLessEqual(audit['candidate_gaps'][i],audit['v1_gap'])
        self.assertTrue(all(x==y or max(x,y)<6 for x,y in zip(out['actions'],self.actual)))
        self.assertEqual(out['score'],out['scores'][i]); self.assertEqual(out['score'],max(out['scores']))

    def test_permutation_duplicates_and_singleton(self):
        first = v2_choose(self.model,self.pre,self.menu)
        permuted = v2_choose(self.model,self.pre,list(reversed(self.menu))+self.menu)
        for k in ('score','actions','unique_count','top_ties','reason'):
            self.assertEqual(first[k],permuted[k])
        self.assertEqual(permuted['scores'],first['scores'][::-1]+first['scores'])
        singleton = v2_choose(self.model,self.pre,[self.actual])
        self.assertEqual(singleton['actions'],self.actual)
        self.assertEqual(singleton['reason'],'no_existing_formation_fix')

    def test_guard_noop_and_previous_actions(self):
        pre = copy.deepcopy(self.pre['state'])
        for invalidation in ('damaged','late'):
            with self.subTest(invalidation=invalidation):
                pre['allies']['u0']['health']=pre['allies']['u0']['max_health']-(invalidation=='damaged')
                pre['game_loop']=83 if invalidation=='late' else 2
                prev=[1]*10
                expected=v1.choose(self.model,pre,self.menu,prev)
                out=v2_choose(self.model,pre,self.menu,prev)
                self.assertEqual(out['reason'],'outside_formation_guard')
                self.assertEqual(expected,{k:out[k] for k in expected})


class TestGate(unittest.TestCase):
    def test_no_heavy_roster(self):
        unit=dict(type='Hydralisk',alive=True,health=10,max_health=10,position=[1,1])
        self.assertFalse(formation(dict(game_loop=2,allies={'u%d'%i:unit for i in range(5)})))
