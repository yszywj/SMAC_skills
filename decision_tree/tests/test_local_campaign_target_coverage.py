"""Synthetic coverage checks. No real rollout or held-out outcome is inspected."""
from pathlib import Path
import json
import tempfile
import unittest

from decision_tree.local.dataset import build_dataset
from decision_tree.local.models import fit_gain
from decision_tree.local.runtime import load_policy
from decision_tree.local.study import save_once
from decision_tree.scripts.local_campaign_target_coverage import coverage, inspect, remaining_firepower, summarize
from decision_tree.tests.test_local_dataset import receipt, replace_events
from decision_tree.tests.test_local_pipeline import paired_batch
from skills.local_v1 import FEATURE_NAMES


class TargetCoverageTests(unittest.TestCase):
    def firepower_input(self):
        allies = {a:{'alive':True, 'weapon_ready':True, 'unit_type_id':73, 'attack_upgrade_level':0}
                  for a in ('u0','u1','u2')}
        enemy = {'health':15, 'shield':5, 'is_flying':False, 'unit_type_id':73,
                 'armor_upgrade_level':0, 'shield_upgrade_level':0}
        profile = {'known':True, 'armor':0, 'shield_armor_base':0, 'attributes':[1],
                   'weapons':[{'target_type':'ground','base_damage':10,'hits_per_attack':1,'damage_bonus':[]}]}
        obs = {'state':{'allies':allies,'visible_enemies':{'e0':enemy},'unit_type_profiles':{'73':profile},
                        'relations':{a:{'e0':{'in_weapon_range':True,'smac_attack_available':True}} for a in allies}},
               'questions':{a:{'criteria':{'6':'attack'}} for a in allies}}
        return obs, {'actor':'u0','old_target':'e0','old_target_other_actor_ids':[]}

    def test_predefined_budget_bins_and_actual_accurate_damage(self):
        obs, info = self.firepower_input()
        for units, ratio, name in [([],0,'zero_budget'), (['u1'],0.5,'partial_below_target_hp'),
                                   (['u1','u2'],1,'at_least_target_hp')]:
            info['old_target_other_actor_ids'] = units
            result = remaining_firepower(obs,info)
            self.assertEqual((result['category'],result['budget_to_hp_ratio']), (name,ratio))
            self.assertEqual(result['denominator_effective_hp'],20)
            self.assertEqual([x['direct_hit_estimate'] for x in result['actor_evidence']], [10]*len(units))
        obs['state']['allies']['u1']['weapon_ready'] = False
        obs['state']['relations']['u2']['e0']['in_weapon_range'] = False
        result = remaining_firepower(obs,info)
        self.assertEqual(result['category'],'zero_budget')
        self.assertTrue(all(x['direct_hit_estimate'] is None for x in result['actor_evidence']))

    def test_unknown_budget_records_partial_known_amount_without_imputing_zero(self):
        obs, info = self.firepower_input()
        info['old_target_other_actor_ids'] = ['u1','u2']
        for mutation, expected in [('range','unknown_readiness_or_range'),('weapon','unknown_weapon_or_target_profile')]:
            obs['state']['relations']['u2']['e0']['in_weapon_range'] = None if mutation=='range' else True
            obs['state']['allies']['u2']['unit_type_id'] = 73 if mutation=='range' else 99
            result = remaining_firepower(obs,info)
            self.assertEqual(result['category'],'unknown')
            self.assertEqual(result['known_direct_damage_subtotal'],10)
            self.assertEqual(result['unknown_actor_ids'],['u2'])
            self.assertEqual(result['actor_evidence'][1]['status'],expected)
            self.assertIsNone(result['numerator_direct_damage'])
            self.assertIsNone(result['budget_to_hp_ratio'])
        info['old_target_other_actor_ids'] = []
        obs['state']['visible_enemies']['e0']['health'] = None
        result = remaining_firepower(obs,info)
        self.assertEqual(result['category'],'unknown')
        self.assertIsNone(result['denominator_effective_hp'])

    def test_actor_is_excluded_counts_use_real_orders_and_attack_legality(self):
        observation = {'state': {'allies': {f'u{i}': {'alive': i!=3} for i in range(4)},
                                'visible_enemies': {'e0': {}, 'e1': {}},
                                'relations': {'u0': {e:{'smac_attack_available': True} for e in ('e0','e1')}}},
                       'questions': {'u0': {'criteria': {'6':'attack','7':'attack'}}}}
        decision = {'baseline_actions':[6,6,7,7]}
        candidate = {'family':'retarget','actors':['u0'],'actions':{'u0':7},'duration_steps':1}
        pairs = coverage(observation, decision, candidate)
        self.assertEqual(pairs['old_target_other_actor_ids'], ['u1'])
        self.assertEqual(pairs['new_target_other_actor_ids'], ['u2'])
        del observation['questions']['u0']['criteria']['6']
        with self.assertRaisesRegex(ValueError, 'illegal/unseen'):
            coverage(observation, decision, candidate)

    def test_scene_weighted_contrast_and_deletion_range(self):
        result = summarize([{'group_id':g,'veto_gain':v} for g,v in [('A',3),('A',3),('B',-1),('C',0)]])
        self.assertAlmostEqual(result['scene_equal_veto_gain'], 2/3)
        self.assertEqual(result['row_signs'], {'positive':2,'negative':1,'zero':1})
        self.assertEqual(result['leave_one_scene_out_range'], [-0.5,1.5])
        self.assertIsNone(summarize([])['scene_equal_veto_gain'])

    def test_audited_first_proposals_and_training_split(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _, data, base = paired_batch(root/'batch', 'coverage-contract', -2)
            entries = [r['entry'] for r in data['provenance']['arms']]
            # No legal attack on the first observation, then exactly one tested
            # alternative at observation 1. Records come from the project fixture.
            for directory in [base, *[Path(item['path']) for item in entries]]:
                events = [json.loads(line) for line in (directory/'events.jsonl').read_text().splitlines()]
                for event in events:
                    if event['event']=='skill_observation' and event['step']==0:
                        payload = event['payload']
                        for relation in payload['state']['relations']['u0'].values():
                            relation['smac_attack_available'] = False
                        menu = payload['questions']['u0']['criteria']
                        payload['questions']['u0']['criteria'] = {k:v for k,v in menu.items() if int(k)<6}
                    elif event['event']=='local_decision' and event['step']==0:
                        event.update(baseline_actions=[1], final_actions=[1], candidates=[],
                                     baseline_roles={'u0':'reposition'}, baseline_predicted_damage={})
                    elif event['event']=='step' and event['step']==1:
                        event.update(actions=[1], feedback=receipt(payload, [1]))
                replace_events(directory, events)
            rows = [{'group_id':f'old{i}', 'candidate_family':'retarget', 'gain':2,
                     'features':data['rows'][0]['features']} for i in range(4)]
            model = fit_gain(rows, FEATURE_NAMES, data['context_hash'])
            model['game_version'] = data['game_version']
            reference = load_policy({**data['policy'], 'model':model,
                                     'selection':{'mode':'model','min_gain':0,'max_interventions':1}})
            ids = [r['group_id'] for r in data['rows']]
            plan = {'reference_policy':reference, 'scenes':[{'group_id':g} for g in ids]}
            dataset = root/'audited.json'
            save_once(dataset, build_dataset(base, entries, plan))
            split = {'schema_version':1, 'kind':'local_scene_split', 'training_group_ids':ids[:6],
                     'validation_group_ids':ids[6:]}
            result = inspect([dataset], split, model)
            self.assertEqual({r['group_id'] for r in result['records']}, set(ids[:6]))
            self.assertEqual(result['old_target_no_other_attacker']['scene_equal_veto_gain'], 2)
            self.assertEqual(result['old_target_still_covered']['rows'], 0)
            changed = {**model, 'max_depth':1}
            with self.assertRaisesRegex(ValueError, 'reference hash'):
                inspect([dataset], split, changed)


if __name__ == '__main__':
    unittest.main()
