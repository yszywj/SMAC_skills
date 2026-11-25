import json
from pathlib import Path
import tempfile
import unittest

from decision_tree.zerg import policy_distill as p


def state():
    return dict(game_loop=2, allies={'u0':dict(alive=True,type='Zergling',position=[2,1],health=30,max_health=35,weapon_ready=True)},
                visible_enemies={'e0':dict(type='Baneling',health=10,max_health=30)},
                previous_action_ids={'u0':'2'}, relations={'u0':{'e0':dict(distance=2.,smac_attack_available=True,in_weapon_range=False)}},
                movement_candidates={'u0':{'3':{'distances_to_visible_enemies':{'e0':3}}}})


class PolicyTest(unittest.TestCase):
    def test_cart_episode_and_beat_weights(self):
        import numpy as np
        from decision_tree.scripts.linux.fit_zerg_policy import fit
        # One episode has a singleton; the other has three beats with four
        # candidates each. A pooled-row fit would predict 4/13, not 5/8.
        def game(n, beats):
            rows = np.zeros((n*beats,len(p.FEATURES)))
            for i in range(beats):
                rows[i*n,0] = 1
            return rows, [([(i,) for i in range(n)],0,'salt',n)]*beats
        games = {'short':game(1,1),'long':game(4,3)}
        nodes, _ = fit(games,sorted(games),0,min_leaf=1)
        self.assertAlmostEqual(nodes[0][0],5/8)
        nodes, info = fit(games,sorted(games),1,min_leaf=1)
        self.assertEqual(info['leaves'],2)
        self.assertEqual(p.score(nodes,[0]),0); self.assertAlmostEqual(p.score(nodes,[1]),1)

    def test_semantics(self):
        x = dict(zip(p.FEATURES,p.project(state(),[6])))
        self.assertEqual(len(x),111)
        self.assertEqual(x['zergling.attack_baneling'],1)
        self.assertEqual(x['all.attack_distance_sum'],2)
        self.assertEqual(x['all.attack_wounded'],1)
        self.assertEqual(x['hydralisk.attack'],0)
        x = dict(zip(p.FEATURES,p.project({'state':state()},[3])))
        self.assertEqual(x['all.move_closure_sum'],-1)
        self.assertEqual(x['all.reverse_move'],1)
        self.assertEqual(x['all.move_with_attack_available'],1)

    def test_pre_only_and_duplicates(self):
        s = state(); model = {'nodes':[[.5]]}; a = p.choose(model,s,[[1],[6]])
        s.update(teacher_choice='not a feature', model_scores=[1000], return_value=-100)
        b = p.choose(model,s,[[6],[1],[1]])
        self.assertEqual(a['actions'],b['actions'])
        self.assertEqual(b['scores'],[.5,.5,.5]); self.assertEqual(b['top_ties'],2)
        selected = set()
        for tick in range(100):
            s['game_loop'] = tick
            selected.add(tuple(p.choose(model,s,[[1],[6]])['actions']))
        self.assertEqual(selected,{(1,),(6,)})

    def test_reject_incomplete_and_malformed(self):
        for v in ([],[True],[16],['2']):
            with self.assertRaises(ValueError):
                p.project(state(),v)
        m = dict(version=p.VERSION,features=p.FEATURES,schema_hash=p.SCHEMA_HASH,
                 source_hash='1'*64,ctx_hash='2'*64,split_hash='3'*64,nodes=[[0,0,0,1],[1]])
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'bad.json'; path.write_text(json.dumps(m))
            with self.assertRaises(ValueError):
                p.load_model(path)

    def test_real_episode_json_round_trip(self):
        root = Path(__file__).resolve().parents[3]; out = root/'runs/zerg_policy_distill_v1'
        if not (out/'model.json').is_file():
            self.skipTest('real fitted artifact not installed')
        m = p.load_model(out/'model.json')
        meta = next(v for v in json.loads((out/'split.json').read_text()).values() if v['partition']=='validation')
        w = root/meta['batch']/meta['worker']
        with (w/'runner/events.jsonl').open() as src, (w/'shadow/shadow.jsonl').open() as rows:
            count = 0
            for event in map(json.loads,src):
                if event['event'] != 'skill_observation':
                    continue
                s = event['payload']; r = json.loads(next(rows)); count += 1
                self.assertEqual(s['state']['game_loop'],r['game_loop'])
                candidates = [c['proposed_actions'] for c in r['candidates'] if c['candidate_family']=='joint_plan' and c['legal']]
                result = p.choose(m,s,candidates)
                other = p.choose(m,s,list(reversed(candidates)))
                self.assertEqual(result['actions'],other['actions']); self.assertEqual(result['scores'],other['scores'][::-1])
                self.assertEqual(result['score'],max(result['scores']))
            self.assertIsNone(next(rows,None)); self.assertGreater(count,1)


if __name__ == '__main__':
    unittest.main()
