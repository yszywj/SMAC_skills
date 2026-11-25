"""Paired archived decisions and full scorer, no SC2 or rebuilt hypothetical menus."""
import copy
import json
from pathlib import Path
import unittest

from decision_tree.zerg.policy_distill import load_model, vector
from decision_tree.zerg.policy_repair_v2 import v2_choose
from decision_tree.zerg.policy_repair_v3 import v3_choose


PANEL = Path(__file__).resolve().parents[3] / 'runs/zerg_v1_v2_regression_20261124_185519'


def snapshot(pair, step):
    root = PANEL / pair / 'v2'
    with (root/'runner/events.jsonl').open() as f:
        pre = next(ev['payload'] for line in f if '"skill_observation"' in line[:150]
                   if (ev := json.loads(line))['step']==step)
    with (root/'shadow/shadow.jsonl').open() as f:
        cards = {c['candidate_id']:c for i,line in enumerate(f) if i==step for c in json.loads(line)['candidates']}
    d = json.loads((root/'runner/distill/episode-0001.json').read_text())['decisions'][step]
    menu = [cards[i]['proposed_actions'] for i in d['scores'] if cards[i].get('origin')!='distill_opening_support']
    return pre,menu,d


@unittest.skipUnless(PANEL.is_dir(), 'requires the recorded regression panel')
class TestEngagement(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = load_model(PANEL/'model.json')

    def test_early_known_volley(self):
        pre,menu,d = snapshot('pair-017',3)
        out=v3_choose(self.model,pre,menu); audit=out['audit']; eng=audit['engagement']
        self.assertEqual(audit['v2_actions'], list(vector(pre['state'],d['final_actions'])))
        self.assertTrue(audit['v3_triggered']); self.assertEqual(out['reason'],'improve_ready_volley')
        self.assertEqual((eng['before']['excess'],eng['after']['excess']), (13,1))
        self.assertEqual(eng['before']['actors'],eng['after']['actors'])
        self.assertEqual(eng['before']['lethal'],eng['after']['lethal'])
        idx=[list(vector(pre['state'],a)) for a in menu].index(out['actions'])
        self.assertEqual(out['scores'][idx],max(out['scores']))
        perm=v3_choose(self.model,pre,menu[::-1]+menu)
        for k in ('score','actions','unique_count','top_ties','reason'):
            self.assertEqual(out[k],perm[k])
        self.assertEqual(perm['scores'],out['scores'][::-1]+out['scores'])

    def test_abstain_on_unknown_weapon_or_empty_competition(self):
        pre,menu,_ = snapshot('pair-017',3)
        unknown=copy.deepcopy(pre); unknown['state']['unit_type_profiles']={}
        for obs,options in ((unknown,menu),(pre,menu[:1])):
            old=v2_choose(self.model,obs,options); out=v3_choose(self.model,obs,options)
            self.assertFalse(out['audit']['v3_triggered'])
            self.assertEqual({k:out[k] for k in old if k!='audit'}, {k:v for k,v in old.items() if k!='audit'})

    def test_opening_priority(self):
        for pair,step,reason in [('pair-011',1,'protect_hydra_support_gap'),('pair-018',0,'no_existing_formation_fix')]:
            pre,menu,d=snapshot(pair,step); old=v2_choose(self.model,pre,menu); out=v3_choose(self.model,pre,menu)
            self.assertEqual(out['reason'],reason);self.assertEqual(out['actions'],old['actions'])
            self.assertFalse(out['audit']['v3_triggered']); self.assertEqual(out['scores'],old['scores'])
            expected=d['selector_audit'].get('before_opening_actions',list(vector(pre['state'],d['final_actions'])))
            self.assertEqual(out['actions'],expected)


if __name__=='__main__': unittest.main()
