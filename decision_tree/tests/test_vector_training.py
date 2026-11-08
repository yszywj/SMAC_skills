"""Candidate evidence must not leak across vector-leaf outputs."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from decision_tree.descriptors import candidate_features
from decision_tree.features import FEATURE_NAMES
from decision_tree.learning.vector_train import (evaluate_tested_candidates, fit_vector_tree,
    load_dataset, predict_estimates, select_vector, train_vector, validate_rows, candidate_balanced_weights)
from decision_tree.tests.test_expanded_learning import data_fixture, training_fixture
from decision_tree.tests.test_dataset import IDS


def rows_fixture(n=8, gain_a=2., gain_b=-1.):
    rows = []
    template = data_fixture()['rows'][0]
    for i in range(n):
        for cid, gain in zip(IDS, (gain_a, gain_b)):
            r = deepcopy(template)
            r.update(group_id=f'group_{i}', sc2_seed=i, candidate_id=cid, gain=gain,
                     candidate_return=r['base_return']+gain)
            r['gate']['candidate_id'] = cid
            if cid == IDS[1]:
                r['candidate_call'] = {**r['base_call'], 'posture':'defensive'}
            r['features'] = candidate_features(r['state_features'],r['candidate_call'],r['base_call'])
            rows.append(r)
    return rows


def fit(rows, ids=IDS, **kwargs):
    return fit_vector_tree(rows,list(ids),context_hash='fixture',**kwargs)


class VectorTrainingTests(unittest.TestCase):
    def test_one_candidate_positive_never_raises_another_candidate_score(self):
        rows = rows_fixture(gain_a=3.,gain_b=0.)
        model = fit(rows,max_depth=0)
        estimates = predict_estimates(model,rows[0]['state_features'])
        self.assertEqual(estimates[IDS[0]]['gain'],3.)
        self.assertEqual(estimates[IDS[1]]['gain'],0.)
        self.assertEqual(select_vector(model,rows[0]['state_features'],list(IDS))[0],IDS[0])

    def test_unobserved_candidate_is_null_and_never_eligible(self):
        rows = rows_fixture()
        model = fit(rows,ids=(*IDS,'unobserved'),max_depth=0)
        unseen = model['root']['estimates']['unobserved']
        self.assertEqual(unseen,{'gain':None,'n_samples':0,'n_groups':0,'group_ids':[],'supported':False})
        self.assertEqual(select_vector(model,rows[0]['state_features'],['unobserved'])[0],'BASE')

    def test_many_rows_from_one_scenario_are_not_four_scenarios(self):
        rows = rows_fixture()
        # Keep A in a single original scenario, despite many distinct anchors.
        rows = [r for r in rows if r['candidate_id']==IDS[1] or r['group_id']=='group_0']
        a = next(r for r in rows if r['candidate_id']==IDS[0])
        for anchor in range(2,12):
            r=deepcopy(a)
            r['anchor_call_number']=anchor
            r['gate']['anchor_call_number']=anchor
            rows.append(r)
        model=fit(rows,max_depth=0)
        estimate=model['root']['estimates'][IDS[0]]
        self.assertEqual(estimate['n_samples'],11)
        self.assertEqual(estimate['n_groups'],1)
        self.assertFalse(estimate['supported'])
        self.assertEqual(select_vector(model,a['state_features'],list(IDS))[0],'BASE')

    def test_candidate_estimate_averages_scenarios_not_repeated_anchors(self):
        rows=rows_fixture(4,gain_a=0.)
        for r in rows:
            if r['candidate_id']==IDS[0] and r['group_id']=='group_0':
                r['gain']=8.
                r['candidate_return']=r['base_return']+8.
        a=next(r for r in rows if r['candidate_id']==IDS[0])
        for anchor in range(2,10):
            r=deepcopy(a)
            r['anchor_call_number']=anchor
            r['gate']['anchor_call_number']=anchor
            rows.append(r)
        model=fit(rows,max_depth=0)
        self.assertEqual(model['root']['estimates'][IDS[0]]['gain'],2.)
        self.assertAlmostEqual(model['root']['weight_sum'],4.)

    def test_unequal_candidate_coverage_has_coherent_fixed_weights(self):
        rows=rows_fixture(4,gain_a=0.,gain_b=0.)
        rows=[r for r in rows if r['candidate_id']==IDS[0] or r['group_id']=='group_0']
        a=next(r for r in rows if r['candidate_id']==IDS[0])
        a['gain']=8.
        a['candidate_return']=a['base_return']+8.
        for anchor in range(2,6):
            r=deepcopy(a)
            r['anchor_call_number']=anchor
            r['gate']['anchor_call_number']=anchor
            rows.append(r)
        weights=candidate_balanced_weights(rows)
        for group in {r['group_id'] for r in rows}:
            self.assertAlmostEqual(sum(w for r,w in zip(rows,weights) if r['group_id']==group),1.)
        for cid in IDS:
            self.assertAlmostEqual(sum(w for r,w in zip(rows,weights)
                                       if r['group_id']=='group_0' and r['candidate_id']==cid),.5)
        model=fit(rows,max_depth=0)
        mean=model['root']['estimates'][IDS[0]]['gain']
        self.assertAlmostEqual(mean,4./3.5)
        # The fitted mean makes the derivative of the actual SSE exactly zero.
        derivative=sum(w*(r['gain']-mean) for r,w in zip(rows,weights) if r['candidate_id']==IDS[0])
        self.assertAlmostEqual(derivative,0.)

    def test_leaf_mean_retains_global_weights_when_group_spans_leaves(self):
        rows=rows_fixture(8,gain_a=0.,gain_b=0.)
        for r in rows:
            r['state_features']['game_loop']=0 if r['sc2_seed']<4 else 10
            r['gain']=(1. if r['sc2_seed']<4 else 6.) if r['candidate_id']==IDS[0] else 0.
            r['candidate_return']=r['base_return']+r['gain']
            r['features']=candidate_features(r['state_features'],r['candidate_call'],r['base_call'])
        a=deepcopy(rows[0])
        a['anchor_call_number']=2
        a['gate']['anchor_call_number']=2
        a['state_features']['game_loop']=10
        a['gain']=10.
        a['candidate_return']=a['base_return']+10.
        a['features']=candidate_features(a['state_features'],a['candidate_call'],a['base_call'])
        rows.append(a)
        model=fit(rows,max_depth=1)
        self.assertEqual(model['root']['feature'],'game_loop')
        weights=candidate_balanced_weights(rows)
        for values in (rows[0]['state_features'],a['state_features']):
            mean=predict_estimates(model,values)[IDS[0]]['gain']
            side=values['game_loop']<=model['root']['threshold']
            derivative=sum(w*(r['gain']-mean) for r,w in zip(rows,weights)
                           if r['candidate_id']==IDS[0] and
                           (r['state_features']['game_loop']<=model['root']['threshold'])==side)
            self.assertAlmostEqual(derivative,0.)

    def test_equal_or_nearly_equal_positive_candidates_fall_back_to_base(self):
        for difference in (0.,5e-10):
            rows=rows_fixture(gain_a=1.,gain_b=1.+difference)
            model=fit(rows,max_depth=0)
            selected,_,reason=select_vector(model,rows[0]['state_features'],list(IDS))
            self.assertEqual((selected,reason),('BASE','ambiguous_positive_gain'))

    def test_all_zero_candidate_returns_preserve_base(self):
        rows=rows_fixture(gain_a=0.,gain_b=0.)
        model=fit(rows,max_depth=0)
        self.assertEqual(select_vector(model,rows[0]['state_features'],list(IDS))[0],'BASE')
        report=evaluate_tested_candidates(model,rows)
        self.assertEqual(report['baseline_fallback_rate'],1.)
        self.assertEqual(report['mean_regret_to_tested_best'],0.)
        self.assertIsNone(report['rank_agreement_supported_pairs'])

    def test_near_zero_positive_return_does_not_override_base(self):
        rows=rows_fixture(gain_a=5e-10,gain_b=-1.)
        model=fit(rows,max_depth=0)
        self.assertEqual(select_vector(model,rows[0]['state_features'],list(IDS))[0],'BASE')

    def test_runtime_and_offline_selection_agree_on_ties_and_support(self):
        from decision_tree.policy import choose
        for a,b in ((2.,0.),(1.,1.+5e-10),(5e-10,-1.),(-1.,-2.)):
            rows=rows_fixture(gain_a=a,gain_b=b)
            model=fit(rows,max_depth=0)
            policy=data_fixture()['policy_contract']
            policy['selection']={'mode':'model','model':model,'max_interventions_per_episode':1,'min_gain':0.}
            available=[{'id':r['candidate_id'],'call':r['candidate_call']} for r in rows[:2]]
            runtime=choose(policy,rows[0]['state_features'],available,rows[0]['base_call'])
            offline=select_vector(model,rows[0]['state_features'],list(IDS))
            self.assertEqual(runtime['candidate_id'],offline[0])

    def test_candidate_specific_state_split_learns_opposite_effects(self):
        rows=rows_fixture(12)
        for r in rows:
            i=r['sc2_seed']
            r['state_features']['game_loop']=i
            r['gain']=(2. if i<6 else -2.)*(1. if r['candidate_id']==IDS[0] else -1.)
            r['candidate_return']=r['base_return']+r['gain']
            r['features']=candidate_features(r['state_features'],r['candidate_call'],r['base_call'])
        model=fit(rows,max_depth=2)
        self.assertEqual(model['root']['feature'],'game_loop')
        self.assertEqual(select_vector(model,rows[0]['state_features'],list(IDS))[0],IDS[0])
        self.assertEqual(select_vector(model,rows[-1]['state_features'],list(IDS))[0],IDS[1])

    def test_one_anchor_cannot_route_candidates_using_different_state(self):
        rows=rows_fixture()
        rows[1]['state_features']['game_loop']+=1
        rows[1]['features']=candidate_features(rows[1]['state_features'],rows[1]['candidate_call'],rows[1]['base_call'])
        with self.assertRaisesRegex(ValueError,'share state'):
            fit(rows)

    def test_future_outcome_or_candidate_id_feature_is_rejected(self):
        rows=rows_fixture()
        rows[0]['vector_features']={**rows[0]['state_features'],'final_reward':20.}
        with self.assertRaisesRegex(ValueError,'allowlist'):
            fit(rows)

    def test_selection_metrics_expose_harm_and_wrong_ranking(self):
        training=rows_fixture(gain_a=3.,gain_b=-1.)
        model=fit(training,max_depth=0)
        validation=rows_fixture(gain_a=-1.,gain_b=2.)
        report=evaluate_tested_candidates(model,validation)
        self.assertEqual(report['selector_mean_gain'],-1.)
        self.assertEqual(report['mean_regret_to_tested_best'],3.)
        self.assertEqual(report['harmful_selection_rate_per_anchor'],1.)
        self.assertEqual(report['rank_agreement_supported_pairs'],0.)
        self.assertEqual(report['supported_prediction_coverage'],1.)

    def test_available_mask_ignores_unsupported_and_absent_candidates(self):
        rows=rows_fixture(gain_a=3.,gain_b=1.)
        model=fit(rows,max_depth=0)
        self.assertEqual(select_vector(model,rows[0]['state_features'],[IDS[1]])[0],IDS[1])
        with self.assertRaisesRegex(ValueError,'unknown'):
            select_vector(model,rows[0]['state_features'],['not_declared'])

    def test_export_uses_disjoint_scene_groups_and_declared_context(self):
        data=training_fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            path=root/'pairs.json'
            path.write_text(json.dumps(data),encoding='utf-8')
            policy,model,report=train_vector(path,root/'model')
            split=report['split']
            self.assertFalse(set(split['train_groups'])&set(split['validation_groups']))
            self.assertEqual(model['schema_version'],3)
            self.assertEqual(model['context'],data['context'])
            self.assertEqual(model['candidate_ids'],data['candidate_ids'])
            self.assertEqual(policy['selection']['model'],model)
            self.assertEqual(report['status'],'trained_not_game_validated')
            self.assertTrue((root/'model/report.json').exists())

    def test_sealed_test_dataset_and_missing_vector_schema_rejected(self):
        data=training_fixture()
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'pairs.json'
            data['split']='test'
            path.write_text(json.dumps(data),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'sealed'):
                load_dataset(path)
            data['split']='development'
            data['schema_version']=3
            path.write_text(json.dumps(data),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'explicit public'):
                load_dataset(path)

    def test_choice_context_schema_exports_and_requires_policy_binding(self):
        from decision_tree.choice_features import CHOICE_FEATURE_NAMES
        from decision_tree.experiment import digest
        data=training_fixture()
        data['schema_version']=3
        data['vector_feature_names']=list(CHOICE_FEATURE_NAMES)
        data['policy_contract']['exploration']['feature_schema']=3
        data['context']['policy_contract']=deepcopy(data['policy_contract'])
        data['context_hash']=digest(data['context'])
        for row in data['rows']:
            row['state_features']={name:row['state_features'].get(name,0.) for name in CHOICE_FEATURE_NAMES}
            row['vector_features']=deepcopy(row['state_features'])
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'pairs.json'
            path.write_text(json.dumps(data),encoding='utf-8')
            _,model,_=train_vector(path)
            self.assertEqual(model['feature_names'],list(CHOICE_FEATURE_NAMES))
            data['policy_contract']['exploration'].pop('feature_schema')
            data['context']['policy_contract']=deepcopy(data['policy_contract'])
            data['context_hash']=digest(data['context'])
            path.write_text(json.dumps(data),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'declared in policy context'):
                load_dataset(path)


if __name__=='__main__':
    unittest.main()
