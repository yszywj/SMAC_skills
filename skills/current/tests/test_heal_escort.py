"""Behavioral boundaries of local healing/retreat proposals."""
import copy
import unittest
from unittest.mock import patch

from skills.current.skills import heal_escort
from skills.current.tests.test_terran_combat import scene, unit


def fixture():
    state = scene([unit('Marine', (3.8, 0), 10), unit('Medivac', (0, 0))])
    state['allies']['u0']['tag'] = 'patient'
    state['allies']['u1']['tag'] = 'healer'
    state['action_contract'] = {'candidate_center_distance_by_type': {'medivac': 4}}
    state['movement_candidates']['u0'] = {a:r for a,r in state['movement_candidates']['u0'].items() if a in ('2','4')}
    healing = {'assignments': {'u1':'u0'}, 'self_preserving':{}}
    return state, healing, [4,6]


def choose(state, healing, actions, **kwargs):
    return heal_escort.select_heal_escort(state, 'u0', actions[0], 'retreat', healing, actions,
                                         pressure_guard=lambda _:True, **kwargs)


class HealEscortTests(unittest.TestCase):
    def test_keeps_range_without_requiring_closer_than_current_position(self):
        state, healing, actions = fixture()
        row = choose(state, healing, actions)
        self.assertTrue(row['applied'])
        self.assertEqual(row['selected_action'], 2)
        self.assertEqual(row['pair_status'], 'pending')
        self.assertLess(row['baseline_predicted_clearance'],0)
        self.assertGreater(row['selected_predicted_clearance'],0)
        self.assertGreater(4-row['selected_predicted_clearance'],3.8)

    def test_baseline_in_range_is_not_optimized_for_extra_closeness(self):
        s,h,a = fixture();a[0]=2
        row=choose(s,h,a)
        self.assertFalse(row['applied'])
        self.assertEqual(row['reason'],'baseline_retains_range_reference')

    def test_attack_recovery_and_stop_are_never_overridden(self):
        s,h,a=fixture()
        for role,action in [('attack',6),('recover',2),('retreat',1),('reposition',4)]:
            with self.subTest(role=role,action=action):
                row=heal_escort.select_heal_escort(s,'u0',action,role,h,a,pressure_guard=lambda _:True)
                self.assertFalse(row['applied']);self.assertEqual(row['selected_action'],action)

    def test_unknown_conflicting_or_invalid_range_does_not_invent_threshold(self):
        for contract,brief in [({},{}),({'medivac':None},{}),({'medivac':4},{'medivac':5}),({'medivac':float('nan')},{})]:
            with self.subTest(contract=contract,brief=brief):
                s,h,a=fixture();s['action_contract']['candidate_center_distance_by_type']=contract
                s['scenario_brief']={'candidate_ranges':brief}
                self.assertEqual(choose(s,h,a)['reason'],'unknown_public_heal_range')
        s,h,a=fixture();s['action_contract']={};s['scenario_brief']={'candidate_ranges':{'medivac':4}}
        self.assertTrue(choose(s,h,a)['applied'])

    def test_unknown_patient_speed_or_window_keeps_baseline(self):
        for value in (None,0,float('nan'),True):
            s,h,a=fixture();s['unit_type_profiles']['48']['movement_speed_native']=value
            self.assertEqual(choose(s,h,a)['reason'],'unknown_patient_travel')
        s,h,a=fixture();s.pop('step_mul')
        self.assertEqual(choose(s,h,a)['reason'],'unknown_patient_travel')

    def test_one_window_speed_not_nominal_full_command_distance(self):
        s,h,a=fixture();s['unit_type_profiles']['48']['movement_speed_native']=.2
        self.assertEqual(choose(s,h,a)['reason'],'baseline_retains_range_reference')

    def test_requires_unique_pair_and_actual_final_heal_command(self):
        for replacement in (1,2,7):
            s,h,a=fixture();a[1]=replacement
            self.assertEqual(choose(s,h,a)['reason'],'final_healer_action_not_legal_heal')
        s,h,a=fixture();h['assignments']={}
        self.assertEqual(choose(s,h,a)['reason'],'no_unique_planned_pair')
        s,h,a=fixture();s['heal_candidates']['u1']={}
        self.assertEqual(choose(s,h,a)['reason'],'final_healer_action_not_legal_heal')
        s,h,a=fixture();h['assignments']['u1']='u9'
        self.assertEqual(choose(s,h,a)['reason'],'no_unique_planned_pair')
        s,h,a=fixture();s['legal_actions']={'u1':[1,2]}
        self.assertEqual(choose(s,h,a)['reason'],'final_healer_action_not_legal_heal')

    def test_shared_guard_checks_original_combat_baseline_too(self):
        s,h,a=fixture()
        with patch.object(heal_escort,'retreat_override_allowed',side_effect=lambda state,actor,baseline,*args:baseline==4) as guard:
            row=choose(s,h,a,combat_baseline=3)
        self.assertFalse(row['applied'])
        self.assertEqual([call.args[2] for call in guard.call_args_list],[4,3])

    def test_unavailable_or_self_preserving_healer_cannot_anchor_retreat(self):
        for field,value in [('energy',0),('energy',None),('alive',False)]:
            s,h,a=fixture();s['allies']['u1'][field]=value
            self.assertEqual(choose(s,h,a)['reason'],'healer_unavailable')
        s,h,a=fixture();h['self_preserving']={'u1':True}
        self.assertEqual(choose(s,h,a)['reason'],'healer_unavailable')

    def test_observed_and_new_plan_are_separate(self):
        s,h,a=fixture();s['allies']['u1']['orders']=[dict(ability_id=386,target_unit='u0')]
        self.assertEqual(choose(s,h,a)['pair_status'],'observed')
        s['allies']['u1']['orders']=[dict(ability_id=386,target_unit='u9')]
        self.assertEqual(choose(s,h,a)['pair_status'],'pending')
        self.assertEqual(s['allies']['u1']['orders'][0]['target_unit'],'u9')

    def test_recent_healer_away_motion_marks_reference_uncertain(self):
        s,h,a=fixture()
        hist={'u1':dict(tag='healer',loop=0,position=[1,0],action=5)}
        self.assertEqual(choose(s,h,a,histories=hist)['reason'],'healer_recently_moving_away')
        hist['u1']['loop']=-8
        self.assertTrue(choose(s,h,a,histories=hist)['applied'])

    def test_explicit_legal_mask_is_respected(self):
        s,h,a=fixture();s['legal_actions']={'u0':{4}}
        self.assertFalse(choose(s,h,a)['applied'])
        s['legal_actions']=None
        self.assertTrue(choose(s,h,a)['applied'])

    def test_visible_exposure_and_native_pressure_are_not_traded_for_healing(self):
        s,h,a=fixture()
        enemy=unit('Marine',(3.8,1.6));s['visible_enemies']={'e0':enemy}
        s['unit_type_profiles']['48']['weapons'][0]['range']=.1
        self.assertFalse(choose(s,h,a)['applied'])
        s,h,a=fixture()
        row=heal_escort.select_heal_escort(s,'u0',4,'retreat',h,a,pressure_guard=lambda _:False)
        self.assertFalse(row['applied'])
        self.assertFalse(heal_escort.select_heal_escort(s,'u0',4,'retreat',h,a)['applied'])

    def test_reversal_is_not_allowed_for_treatment(self):
        s,h,a=fixture();history={'u0':dict(tag='patient',loop=0,action=3,position=[3.8,1])}
        self.assertFalse(choose(s,h,a,histories=history)['applied'])

    def test_later_reentry_shot_is_not_allowed_for_treatment(self):
        s,h,a=fixture();s['visible_enemies']={'e0':unit('Marine',(10,0))}
        s['relations']['u0']['e0']=dict(native_weapon_range=5,smac_attack_distance_limit=6,
            weapon_range_status='out_of_range',distance=6.2)
        self.assertFalse(choose(s,h,a)['applied'])

    def test_disabled_pure_and_deterministic(self):
        s,h,a=fixture();before=copy.deepcopy((s,h,a));first=choose(s,h,a)
        self.assertEqual(first,choose(s,h,a));self.assertEqual((s,h,a),before)
        with patch.dict(heal_escort.PARAMS,enabled=False):
            row=choose(s,h,a)
        self.assertEqual(row['selected_action'],4);self.assertEqual(row['reason'],'disabled')

    def test_continuity_confirms_next_observation_without_claiming_heal_amount(self):
        s,h,a=fixture();_,prior=heal_escort.observe_heal_continuity(s,h,a)
        self.assertEqual(prior['u1']['pair_status'],'pending')
        self.assertEqual(prior['u1']['observed_legal_windows'],0)
        s['game_loop']=16;s['allies']['u1']['orders']=[dict(ability_id=386,target_unit='u0')]
        observed,memory=heal_escort.observe_heal_continuity(s,h,a,prior)
        self.assertEqual(observed['u1']['reason'],'observed_legal_order')
        self.assertEqual(memory['u1']['observed_legal_windows'],1)
        self.assertNotIn('healed_amount',observed['u1'])
        self.assertEqual(observed['u1']['actual_center_gap'],3.8)

    def test_continuity_distinguishes_range_death_full_energy_and_order(self):
        for reason,edit in [
            ('outside_public_candidate_range',lambda s:s['allies']['u0'].update(position=[4.3,0])),
            ('patient_dead',lambda s:s['allies']['u0'].update(alive=False,health=0)),
            ('healer_dead',lambda s:s['allies']['u1'].update(alive=False)),
            ('patient_full_health',lambda s:s['allies']['u0'].update(health=45)),
            ('insufficient_or_unknown_energy',lambda s:s['allies']['u1'].update(energy=0)),
            ('different_or_missing_observed_order',lambda s:None),
            ('no_legal_heal_candidate',lambda s:s['heal_candidates'].update(u1={}))]:
            with self.subTest(reason=reason):
                s,h,a=fixture();_,prior=heal_escort.observe_heal_continuity(s,h,a)
                s['game_loop']=16;edit(s)
                observed,_=heal_escort.observe_heal_continuity(s,h,a,prior)
                self.assertEqual(observed['u1']['reason'],reason)
                self.assertEqual(observed['u1']['observed_legal_windows'],0)

    def test_stale_or_reused_tags_do_not_confirm_old_pair(self):
        s,h,a=fixture();_,prior=heal_escort.observe_heal_continuity(s,h,a)
        s['game_loop']=24
        self.assertEqual(heal_escort.observe_heal_continuity(s,h,a,prior)[0],{})
        s['game_loop']=16;s['allies']['u0']['tag']='different'
        self.assertEqual(heal_escort.observe_heal_continuity(s,h,a,prior)[0],{})


class V6ApproachTests(unittest.TestCase):
    def setUp(self):
        from skills.current.skills import heal_escort, terran_motion
        legacy=patch.dict(heal_escort.V7_PARAMS,enabled=False);legacy.start();self.addCleanup(legacy.stop)
        motion=patch.object(terran_motion,"V7_ENABLED",False);motion.start();self.addCleanup(motion.stop)
        switch=patch.dict(heal_escort.V6_PARAMS,{k:True for k in heal_escort.V6_PARAMS})
        switch.start();self.addCleanup(switch.stop)

    def _case(self,healer_x=0,loop=8):
        s=scene([unit('Marine',(6.5,0),20),unit('Medivac',(healer_x,0))])
        s['game_loop']=loop;s['action_contract']={'candidate_center_distance_by_type':{'medivac':4}}
        return s,{'healers':['u1'],'assignments':{},'observed_orders':{},'self_preserving':{}}

    def _coordinate(self,s,h,memory=None,roles=None,actions=None,healer_safe=True):
        return heal_escort.coordinate_v6(s,actions or [1,2],roles or {'u0':'recover','u1':'support'},
            h,{},memory or {},lambda *args:{'allowed':True,'exceptions':[]},lambda *args:healer_safe,{})

    def test_pre_pair_outside_heal_range_only_moves_medical_and_protects_recover(self):
        s,h=self._case();original=copy.deepcopy((s,h));r=self._coordinate(s,h)
        self.assertEqual(r['actions'],[1,4]);self.assertEqual(h['assignments'],{})
        self.assertEqual(r['approach_intents']['u1']['mode'],'approach_not_healing')
        self.assertEqual(r['approach_decisions']['u1']['patient_role'],'recover')
        self.assertEqual(r['approach_decisions']['u1']['patient_baseline_action'],1)
        self.assertEqual((s,h),original)

    def test_observed_attack_patient_is_monitored_without_losing_attack(self):
        s,h=self._case();s['visible_enemies']={'e0':unit('Marine',(10,0))}
        s['relations']['u0']['e0']={'smac_attack_available':True,'distance':3.5}
        s['allies']['u0']['orders']=[dict(ability_id=23,target_unit='e0')]
        memory={'v6_motion':{'u0':dict(tag='48',loop=0,position=[6.5,0],action=6,kind='attack',target='e0',
                                      observed_attack_target='e0',predicted_center=[6.5,0])}}
        r=self._coordinate(s,h,memory,{'u0':'attack','u1':'support'},[6,2])
        self.assertEqual(r['actions'],[6,4])
        self.assertEqual(r['pair_monitor']['u1']['patient_prediction']['kind'],'attack')
        self.assertEqual(r['pair_monitor']['u1']['patient_prediction']['target'],'e0')

    def test_approach_ttl_expires_then_allows_a_new_bounded_intent(self):
        memory={};results=[]
        for x,loop in [(0,8),(1,16),(2,24),(2,32)]:
            s,h=self._case(x,loop);r=self._coordinate(s,h,memory);results.append(r)
            memory={'v6_approach_intents':r['approach_intents']}
        self.assertEqual(results[0]['approach_intents']['u1']['windows'],1)
        self.assertEqual(results[1]['approach_intents']['u1']['windows'],2)
        self.assertEqual(results[2]['approach_decisions']['u1']['reason'],'approach_intent_expired')
        self.assertEqual(results[3]['approach_intents']['u1']['windows'],1)

    def test_no_observed_approach_progress_ends_the_intent(self):
        s,h=self._case();first=self._coordinate(s,h);s['game_loop']=16
        r=self._coordinate(s,h,{'v6_approach_intents':first['approach_intents']})
        self.assertFalse(r['approach_decisions']['u1']['selected'])
        self.assertEqual(r['approach_decisions']['u1']['reason'],'approach_intent_no_observed_progress')
        self.assertEqual(r['approach_outcomes']['u1']['gap_progress'],0)

    def test_occupied_target_and_observed_healer_are_not_prepaired(self):
        s,h=self._case();h['observed_orders']={'u1':'u0'}
        s['allies']['u1']['orders']=[dict(ability_id=386,target_unit='u0')]
        self.assertEqual(self._coordinate(s,h)['approach_intents'],{})
        s,h=self._case();h['observed_orders']={'other_medical':'u0'}
        self.assertEqual(self._coordinate(s,h)['approach_intents'],{})

    def test_two_idle_medical_units_do_not_share_one_intent_target(self):
        s,h=self._case();extra=scene([unit('Marine',(6.5,0),20),unit('Medivac',(0,0)),unit('Medivac',(0,1))])
        extra['action_contract']=s['action_contract'];h['healers'].append('u2')
        r=self._coordinate(extra,h,roles={'u0':'recover','u1':'support','u2':'support'},actions=[1,2,2])
        self.assertEqual(len(r['approach_intents']),1)

    def test_unknown_range_disabled_component_and_unsafe_medical_motion_fall_back(self):
        s,h=self._case()
        self.assertEqual(self._coordinate(s,h,healer_safe=False)['actions'],[1,2])
        with patch.dict(heal_escort.V6_PARAMS,approach=False):
            self.assertEqual(self._coordinate(s,h)['actions'],[1,2])
        s['action_contract']={}
        self.assertEqual(self._coordinate(s,h)['actions'],[1,2])


if __name__=='__main__':unittest.main()
