"""Behavioral tests of Terran composition using real combat allocators."""
import copy
import math
import unittest
from unittest.mock import patch

from skills.current.skills import terran_combat
from skills.current.skills.terran_combat import allocate_terran_combat, _support_retreat


def unit(kind, pos, health=None, energy=50):
    identity, maximum, flying = {'Marine':(48,45,False),'Marauder':(51,125,False),
                                 'Medivac':(54,150,True)}[kind]
    return dict(tag=str(identity),type=kind,unit_type_id=identity,alive=True,position=list(pos),radius=.5,
                health=maximum if health is None else health,max_health=maximum,shield=0,max_shield=0,
                is_flying=flying,weapon_ready=True,weapon_cooldown_loops=0,energy=energy,orders=[],
                health_lost_since_previous_decision=0,shield_lost_since_previous_decision=0)


def scene(units, enemies=None):
    enemies=enemies or {}
    profiles={
        '48':dict(known=True,armor=0,movement_speed_native=4,weapons=[dict(target_type='any',range=5,base_damage=6,hits_per_attack=1)]),
        '51':dict(known=True,armor=1,movement_speed_native=4,weapons=[dict(target_type='ground',range=6,base_damage=10,hits_per_attack=1)]),
        '54':dict(known=True,armor=1,movement_speed_native=4,weapons=[])}
    allies={f'u{i}':u for i,u in enumerate(units)}
    state=dict(allies=allies,visible_enemies=enemies,game_loop=8,step_mul=8,steps_remaining=100,
               unit_type_profiles=profiles,relations={},movement_candidates={},heal_candidates={})
    for actor,u in allies.items():
        state['relations'][actor]={}
        for enemy,e in enemies.items():
            distance=math.dist(u['position'],e['position'])
            own_compatible=u['type']!='Medivac' and not (u['type']=='Marauder' and e['is_flying'])
            enemy_compatible=e['type']!='Medivac' and not (e['type']=='Marauder' and u['is_flying'])
            state['relations'][actor][enemy]=dict(distance=distance,smac_attack_available=own_compatible and distance<=6,
                native_weapon_range=5,smac_attack_distance_limit=6,in_weapon_range=own_compatible and distance<=6,
                weapon_range_status=('in_range' if distance<=6 else 'out_of_range') if own_compatible else 'no_compatible_weapon',
                enemy_weapon_range_status=('in_range' if distance<=6 else 'out_of_range') if enemy_compatible else 'no_compatible_weapon',
                in_enemy_weapon_range=enemy_compatible and distance<=6)
        state['movement_candidates'][actor]={}
        for action,(dx,dy) in {'2':(0,1),'3':(0,-1),'4':(1,0),'5':(-1,0)}.items():
            dest=[u['position'][0]+dx,u['position'][1]+dy]
            state['movement_candidates'][actor][action]=dict(destination=dest,distance_to_search_location=0,
                distances_to_visible_enemies={e:math.dist(dest,t['position']) for e,t in enemies.items()})
        if u['type']=='Medivac':
            state['heal_candidates'][actor]={str(6+int(t[1:])):dict(target=t,distance=math.dist(u['position'],v['position']),missing_health=v['max_health']-v['health'])
                for t,v in allies.items() if v['type']!='Medivac' and v['health']<v['max_health'] and math.dist(u['position'],v['position'])<=4}
    return state


class TerranCombatTests(unittest.TestCase):
    def setUp(self):
        # Preserve every existing v4/v5 contract as an explicit compatibility
        # suite; v6 behavior is covered separately below.
        switch=patch.dict(terran_combat.V6_PARAMS,enabled=False)
        switch.start();self.addCleanup(switch.stop)

    def _escort_case(self, baseline=4):
        s=scene([unit('Marine',(3.8,0),10),unit('Medivac',(0,0))])
        s['action_contract']={'candidate_center_distance_by_type':{'medivac':4}}
        s['movement_candidates']['u0']={a:r for a,r in s['movement_candidates']['u0'].items() if a in ('2','4')}
        base=terran_combat.allocate_adaptive_focus_fire(s,2)
        base['actions'][0]=baseline;base['roles']['u0']='retreat'
        base['retreat_units']=['u0'];base['retreat_actions']['u0']=baseline
        base['retreat_reasons']['u0']='base_retreat'
        base['movement_diagnostics']['u0']={'pre_overlay':True}
        return s,base

    def test_escort_integrates_after_frozen_healing_and_counts_one_final_move(self):
        s,base=self._escort_case();original=copy.deepcopy(s)
        self.assertEqual(_support_retreat(s,'u0',4,{'u1':'u0'}),4)
        with patch.object(terran_combat,'allocate_adaptive_focus_fire',return_value=base):
            result=allocate_terran_combat(s,2)
        self.assertEqual(result['actions'],[2,6])
        self.assertEqual(result['race_memory']['support_retreat_uses'],{'u0':1})
        self.assertEqual(result['race_memory']['actors']['u0']['action'],2)
        self.assertEqual(result['retreat_reasons']['u0'],'bounded_heal_escort')
        self.assertNotIn('u0',result['movement_diagnostics'])
        self.assertEqual(result['race_telemetry']['healing']['assignments'],{'u1':'u0'})
        self.assertEqual(result['predicted_damage'],{})
        self.assertEqual(s,original)

    def test_escort_existing_budget_is_not_expanded_and_disable_keeps_legacy_action(self):
        for uses,enabled,expected,total in [(0,False,4,0),(1,True,2,2),(2,True,4,2)]:
            s,base=self._escort_case();memory={'support_retreat_uses':{'u0':uses}}
            original=copy.deepcopy(memory)
            with self.subTest(uses=uses,enabled=enabled), patch.dict(terran_combat.ESCORT_PARAMS,enabled=enabled), \
                    patch.object(terran_combat,'allocate_adaptive_focus_fire',return_value=base):
                result=allocate_terran_combat(s,2,race_memory=memory)
            self.assertEqual(result['actions'],[expected,6])
            self.assertEqual(result['race_memory']['support_retreat_uses'],{'u0':total})
            self.assertEqual(memory,original)

    def test_escort_returning_to_combat_baseline_has_no_redirect_budget_charge(self):
        # Isolate arbitration even if a preceding overlay proposes another
        # move: proposal.applied is relative to v4, budget to the combat plan.
        s,base=self._escort_case(baseline=2)
        with patch.object(terran_combat,'allocate_adaptive_focus_fire',return_value=base), \
                patch.object(terran_combat,'_support_retreat',return_value=4):
            result=allocate_terran_combat(s,2)
        decision=result['race_telemetry']['heal_escort']['decisions']['u0']
        self.assertTrue(decision['applied'])
        self.assertEqual((decision['combat_baseline_action'],decision['baseline_action'],decision['selected_action']),(2,4,2))
        self.assertEqual(result['actions'],[2,6])
        self.assertNotIn('u0',result['race_memory']['support_retreat_uses'])
        self.assertNotIn('u0',result['race_telemetry']['changes'])

    def test_joint_healing_assigns_different_wounded_units(self):
        s=scene([unit('Marine',(0,0),20),unit('Marauder',(1,0),80),unit('Medivac',(0,1)),unit('Medivac',(1,1))])
        result=allocate_terran_combat(s,4)
        assignments=result['race_memory']['heal_assignments']
        self.assertEqual(set(assignments),{'u2','u3'})
        self.assertEqual(set(assignments.values()),{'u0','u1'})
        for healer,target in assignments.items():
            self.assertEqual(result['actions'][int(healer[1:])],6+int(target[1:]))
        self.assertEqual(result['attack_assignments'],{})

    def test_empty_energy_follows_friends_instead_of_enemy(self):
        s=scene([unit('Marine',(-8,0)),unit('Medivac',(0,0),energy=0)],{'e0':unit('Marauder',(4,0))})
        result=allocate_terran_combat(s,2)
        self.assertEqual(result['actions'][1],5)
        self.assertEqual(result['race_memory']['heal_assignments'],{})

    def test_ground_only_enemy_does_not_interrupt_critical_medivac_healing(self):
        s=scene([unit('Marine',(0,0),20),unit('Medivac',(0,1),20)],{'e0':unit('Marauder',(1,1))})
        result=allocate_terran_combat(s,2)
        self.assertEqual(result['actions'][1],6)
        self.assertEqual(result['race_telemetry']['air_threats']['u1'],[])

    def test_air_capable_enemy_triggers_critical_medivac_escape(self):
        s=scene([unit('Marine',(-8,0),20),unit('Medivac',(0,0),20)],{'e0':unit('Marine',(5.5,0))})
        result=allocate_terran_combat(s,2)
        self.assertEqual(result['actions'][1],5)
        self.assertEqual(result['race_telemetry']['air_threats']['u1'],['e0'])

    def test_incompatible_attack_mask_is_filtered_and_budget_clean(self):
        s=scene([unit('Marauder',(0,0))],{'e0':unit('Medivac',(2,0))})
        s['relations']['u0']['e0'].update(smac_attack_available=True,weapon_range_status='in_range')
        result=allocate_terran_combat(s,1)
        self.assertLess(result['actions'][0],6)
        self.assertEqual(result['attack_assignments'],{})
        self.assertEqual(result['predicted_damage'],{})
        self.assertEqual(result['race_telemetry']['filtered_incompatible_attacks'],[['u0','e0']])

    def test_idle_stop_is_explicit_cancel_not_fake_hold(self):
        s=scene([unit('Medivac',(0,0),energy=0)])
        s['movement_candidates']['u0']={}
        s['legal_actions']={'u0':{1}}
        result=allocate_terran_combat(s,1)
        self.assertEqual(result['actions'],[1])
        self.assertEqual(result['race_memory']['heal_assignments'],{})
        self.assertIn('never order retention',result['race_telemetry']['order_semantics'])

    def test_stricter_mask_respected_for_support_and_attack(self):
        s=scene([unit('Marine',(0,0)),unit('Medivac',(-5,0),energy=0)],{'e0':unit('Marine',(2,0))})
        s['legal_actions']={'u0':{1},'u1':{2}}
        result=allocate_terran_combat(s,2)
        self.assertEqual(result['actions'],[1,2])
        self.assertEqual(result['attack_assignments'],{})

    def test_inputs_and_cross_window_memory_not_mutated(self):
        s=scene([unit('Marine',(0,0),25),unit('Marauder',(1,0),80),unit('Medivac',(0,1))])
        memory={'heal_assignments':{'u2':'u1'},'support_retreat_uses':{'u0':1}}
        options={'unit_states':{'u2':{'role':'reposition','engagement':{'target':'e9'}}}}
        before=copy.deepcopy((s,memory,options))
        result=allocate_terran_combat(s,3,race_memory=memory,**options)
        self.assertEqual((s,memory,options),before)
        # An unconfirmed previous plan must not outweigh current urgency.
        self.assertEqual(result['race_memory']['heal_assignments'],{'u2':'u0'})
        self.assertIn('u2',result['race_telemetry']['reset_actor_histories'])
        self.assertNotIn('u2',result['engagement_state_updates'])

    def test_support_retreat_never_introduces_new_exposure(self):
        s=scene([unit('Marine',(0,0),10),unit('Medivac',(0,3))],{'e0':unit('Marine',(2,0))})
        s['unit_type_profiles']['48']['weapons'][0]['range']=1
        self.assertEqual(_support_retreat(s,'u0',5,{'u1':'u0'}),2)
        # A new visible threat covers the northern route, so keep west.
        s['visible_enemies']['e1']=unit('Marine',(0,2))
        s['relations']['u0']['e1']=dict(enemy_weapon_range_status='in_range',in_enemy_weapon_range=True,distance=2)
        self.assertEqual(_support_retreat(s,'u0',5,{'u1':'u0'}),5)

    def test_healers_keep_observed_orders_despite_opposite_planned_swap(self):
        s=scene([unit('Marine',(0,0),20),unit('Marauder',(1,0),80),
                 unit('Medivac',(0,1)),unit('Medivac',(1,1))])
        s['allies']['u2']['orders']=[dict(ability_id=386,target_unit='u1')]
        s['allies']['u3']['orders']=[dict(ability_id=386,target_unit='u0')]
        r=allocate_terran_combat(s,4,race_memory={'heal_assignments':{'u2':'u0','u3':'u1'}})
        self.assertEqual(r['actions'][2:],[7,6])
        self.assertEqual(r['race_telemetry']['healing']['locked_orders'],{'u2':'u1','u3':'u0'})

    def test_handoff_waits_for_observed_release_of_exhausted_healer(self):
        s=scene([unit('Marine',(0,0),10),unit('Medivac',(0,1),energy=.98),unit('Medivac',(1,1))])
        s['allies']['u1']['orders']=[dict(ability_id=386,target_unit='u0')]
        r=allocate_terran_combat(s,3)
        self.assertEqual(r['race_memory']['heal_assignments'],{})
        self.assertTrue(all(a < 6 for a in r['actions'][1:]))
        s['allies']['u1']['orders']=[]
        s['game_loop']+=8
        r=allocate_terran_combat(s,3,race_memory=r['race_memory'])
        self.assertEqual(r['race_memory']['heal_assignments'],{'u2':'u0'})
        self.assertEqual(r['actions'][2],6)

    def test_support_retreat_requires_actual_legal_heal_assignment(self):
        s=scene([unit('Marine',(0,0),10),unit('Medivac',(0,17))],{'e0':unit('Marine',(2,0))})
        self.assertEqual(_support_retreat(s,'u0',5,{'u1':'u0'}),5)
        s=scene([unit('Marine',(0,0),10),unit('Medivac',(0,3))],{'e0':unit('Marine',(2,0))})
        self.assertEqual(_support_retreat(s,'u0',5,{}),5)

    def test_support_retreat_does_not_reverse_last_move(self):
        s=scene([unit('Marine',(0,0),10),unit('Medivac',(0,3))],{'e0':unit('Marine',(2,0))})
        s['unit_type_profiles']['48']['weapons'][0]['range']=1
        history=dict(tag=s['allies']['u0']['tag'],loop=0,action=3,position=[0,1])
        self.assertEqual(_support_retreat(s,'u0',5,{'u1':'u0'},history=history),5)

    def test_idle_healer_approaches_remote_wounded_ally(self):
        s=scene([unit('Marine',(8,0),10),unit('Marine',(-1,0)),unit('Medivac',(0,0))])
        r=allocate_terran_combat(s,3)
        self.assertEqual(r['actions'][2],4)
        self.assertEqual(r['race_telemetry']['changes']['u2']['reason'],'approach_wounded_ally')

    def test_rescue_switch_from_stable_patient_to_critical_unoccupied_patient(self):
        s=scene([unit('Marine',(0,0),42.3),unit('Marine',(1,0),7),unit('Medivac',(0,1))])
        s['allies']['u2']['orders']=[dict(ability_id=386,target_unit='u0')]
        before=copy.deepcopy(s)
        r=allocate_terran_combat(s,3)
        self.assertEqual(r['actions'][2],7)
        healing=r['race_telemetry']['healing']
        self.assertEqual(healing['rescue_requests'],{'u2':{'from':'u0','to':'u1'}})
        self.assertEqual(healing['observed_orders'],{'u2':'u0'})
        self.assertEqual(r['race_memory']['heal_order_state']['u2']['pending_target'],'u1')
        self.assertEqual(s,before)

    def test_rescue_switch_never_steals_observed_target_from_another_healer(self):
        s=scene([unit('Marine',(0,0),42.3),unit('Marine',(1,0),7),
                 unit('Medivac',(0,1)),unit('Medivac',(1,1),energy=.98)])
        s['allies']['u2']['orders']=[dict(ability_id=386,target_unit='u0')]
        s['allies']['u3']['orders']=[dict(ability_id=386,target_unit='u1')]
        r=allocate_terran_combat(s,4)
        self.assertEqual(r['actions'][2],6)
        self.assertEqual(r['race_telemetry']['healing']['rescue_requests'],{})
        s['allies']['u3']['orders']=[]
        s['game_loop']+=8
        r=allocate_terran_combat(s,4,race_memory=r['race_memory'])
        self.assertEqual(r['actions'][2],7)

    def test_rescue_requires_stable_current_patient_or_recent_loss_is_protected(self):
        s=scene([unit('Marine',(0,0),30),unit('Marine',(1,0),7),unit('Medivac',(0,1))])
        s['allies']['u0']['health_lost_since_previous_decision']=9
        s['allies']['u2']['orders']=[dict(ability_id=386,target_unit='u0')]
        r=allocate_terran_combat(s,3)
        self.assertEqual(r['actions'][2],6)
        self.assertEqual(r['race_telemetry']['healing']['rescue_options'],{})

    def test_rapidly_dropping_patient_can_trigger_controlled_rescue(self):
        s=scene([unit('Marine',(0,0),42),unit('Marine',(1,0),20),unit('Medivac',(0,1))])
        s['allies']['u1']['health_lost_since_previous_decision']=12
        s['allies']['u2']['orders']=[dict(ability_id=386,target_unit='u0')]
        r=allocate_terran_combat(s,3)
        self.assertEqual(r['actions'][2],7)

    def test_rescue_confirmation_and_cooldown_prevent_target_ping_pong(self):
        s=scene([unit('Marine',(0,0),42.3),unit('Marine',(1,0),7),unit('Medivac',(0,1))])
        s['allies']['u2']['orders']=[dict(ability_id=386,target_unit='u0')]
        r=allocate_terran_combat(s,3)
        memory=r['race_memory']
        # Actual next observation confirms the switch. Reverse the health
        # urgency, but do not let the next plan immediately switch back.
        s=scene([unit('Marine',(0,0),7),unit('Marine',(1,0),42.3),unit('Medivac',(0,1))])
        s['allies']['u2']['orders']=[dict(ability_id=386,target_unit='u1')]
        for loop in (16,24,32):
            s['game_loop']=loop
            r=allocate_terran_combat(s,3,race_memory=memory)
            if loop==16:
                self.assertEqual(r['race_telemetry']['healing']['rescue_confirmations']['u2']['status'],'observed')
            self.assertEqual(r['actions'][2],6 if loop==32 else 7)
            memory=r['race_memory']

    def test_unconfirmed_rescue_plan_does_not_become_observed_order(self):
        s=scene([unit('Marine',(0,0),42.3),unit('Marine',(1,0),7),unit('Medivac',(0,1))])
        s['allies']['u2']['orders']=[dict(ability_id=386,target_unit='u0')]
        r=allocate_terran_combat(s,3)
        s['game_loop']+=8
        r=allocate_terran_combat(s,3,race_memory=r['race_memory'])
        healing=r['race_telemetry']['healing']
        self.assertEqual(healing['observed_orders'],{'u2':'u0'})
        self.assertEqual(healing['rescue_confirmations']['u2']['status'],'different_order_observed')
        self.assertEqual(r['actions'][2],6)
        self.assertEqual(healing['rescue_requests'],{})

    def test_stale_rescue_memory_cannot_block_a_new_healer(self):
        s=scene([unit('Marine',(0,0),42.3),unit('Marine',(1,0),7),unit('Medivac',(0,1))])
        s['allies']['u2']['orders']=[dict(ability_id=386,target_unit='u0')]
        memory={'heal_order_state':{'u2':dict(tag='old-unit',loop=0,last_rescue_request_loop=0,pending_target='u9')}}
        r=allocate_terran_combat(s,3,race_memory=memory)
        self.assertEqual(r['actions'][2],7)
        self.assertEqual(r['race_telemetry']['healing']['rescue_confirmations'],{})

    def test_idle_healers_keep_v2_wounded_approach_without_coverage_split(self):
        s=scene([unit('Marine',(-8,0),10),unit('Marine',(8,0),10),
                 unit('Medivac',(0,0)),unit('Medivac',(0,0))])
        r=allocate_terran_combat(s,4)
        # Frozen v2 chooses the same wounded ally in this symmetric scene.
        # The v4 ablation must not add active coverage partitioning.
        self.assertEqual(r['actions'][2:],[5,5])
        self.assertEqual(r['race_telemetry']['coverage'],{})
        self.assertEqual(r['race_telemetry']['healing']['rescue_requests'],{})
        self.assertTrue(all(r['race_telemetry']['changes'][h]['reason']=='approach_wounded_ally'
                            for h in ('u2','u3')))
        self.assertEqual(r['race_memory']['heal_assignments'],{})

    def test_v2_approach_keeps_valid_healing_and_does_not_approach_air_threat(self):
        s=scene([unit('Marine',(0,0),15),unit('Marine',(8,0),10),
                 unit('Medivac',(0,1)),unit('Medivac',(0,0))],{'e0':unit('Marine',(6,0))})
        s['allies']['u2']['orders']=[dict(ability_id=386,target_unit='u0')]
        r=allocate_terran_combat(s,4)
        self.assertEqual(r['actions'][2],6)
        self.assertEqual(r['race_telemetry']['coverage'],{})
        self.assertEqual(r['actions'][3],2)
        self.assertEqual(r['race_telemetry']['healing']['rescue_requests'],{})

    def test_idle_healer_follows_v2_main_friendly_group(self):
        s=scene([unit('Marine',(-8,0)),unit('Marine',(-8,1)),unit('Marine',(8,0)),
                 unit('Medivac',(0,0))])
        r=allocate_terran_combat(s,4)
        self.assertEqual(r['actions'][3],5)
        self.assertEqual(r['race_telemetry']['changes']['u3']['reason'],
                         'air_threat_avoidance_or_friendly_support')
        self.assertEqual(r['race_telemetry']['healing']['rescue_requests'],{})

    def test_old_coverage_memory_cannot_change_v2_support_actions(self):
        s=scene([unit('Marine',(-8,0),10),unit('Marine',(8,0),10),
                 unit('Medivac',(0,0)),unit('Medivac',(0,0))])
        expected=allocate_terran_combat(s,4)
        for goals in ({'u2':'u1','u3':'u0'}, {'u2':'u1','u3':'u1'}, {'u2':'missing'}):
            memory={'coverage_targets':goals}
            before=copy.deepcopy(memory)
            with self.subTest(goals=goals):
                r=allocate_terran_combat(s,4,race_memory=memory)
                self.assertEqual(r,expected)
                self.assertEqual(memory,before)
                self.assertNotIn('coverage_targets',r['race_memory'])
                self.assertEqual(r['race_telemetry']['version'],'terran-specialization-v5')
                self.assertTrue(r['race_telemetry']['coverage_policy'].startswith('disabled'))

    def test_redirected_retreat_limit_persists_across_windows(self):
        s=scene([unit('Marine',(0,0),10),unit('Medivac',(0,3))],{'e0':unit('Marine',(2,0))})
        s['unit_type_profiles']['48']['weapons'][0]['range']=1
        s['allies']['u0'].update(weapon_ready=False,weapon_cooldown_loops=20)
        memory={}
        actions=[]
        # Observed blocked travel: positions unchanged, fresh snapshot time.
        for tick in range(3):
            s['game_loop']=8*(tick+1)
            r=allocate_terran_combat(s,2,race_memory=memory)
            memory=r['race_memory'];actions.append(r['actions'][0])
            self.assertNotIn('u0',r['attack_assignments'])
            self.assertEqual(r['predicted_damage'],{})
        self.assertEqual(actions,[2,2,5])
        self.assertEqual(memory['support_retreat_uses'],{'u0':2})


class TerranV6Tests(unittest.TestCase):
    def setUp(self):
        from skills.current.skills import heal_escort, terran_motion
        legacy=patch.dict(heal_escort.V7_PARAMS,enabled=False);legacy.start();self.addCleanup(legacy.stop)
        motion=patch.object(terran_motion,"V7_ENABLED",False);motion.start();self.addCleanup(motion.stop)
        switch=patch.dict(terran_combat.V6_PARAMS,{k:True for k in terran_combat.V6_PARAMS})
        switch.start();self.addCleanup(switch.stop)

    def _anticipation_case(self):
        s=scene([unit('Marine',(2.9,0),10),unit('Medivac',(0,0))])
        s['action_contract']={'candidate_center_distance_by_type':{'medivac':4}}
        s['movement_candidates']['u0']={a:r for a,r in s['movement_candidates']['u0'].items() if a in ('2','4')}
        s['allies']['u1']['orders']=[dict(ability_id=386,target_unit='u0')]
        base=terran_combat.allocate_adaptive_focus_fire(s,2)
        base['actions'][0]=4;base['roles']['u0']='retreat';base['retreat_units']=['u0']
        base['retreat_actions']['u0']=4;base['retreat_reasons']['u0']='base_retreat'
        memory={'v6_motion':{'u1':dict(tag='54',loop=0,position=[0,0],action=6,kind='heal',target='u0',
                                     observed_heal_target='u0',predicted_center=[0,0])}}
        return s,base,memory

    def _keep_scene(self):
        s=scene([unit('Marine',(4.2,0),40),unit('Medivac',(0,0))])
        s['action_contract']={'candidate_center_distance_by_type':{'medivac':4},
                              'keep_heal_order':{'enabled':True,'action_id':16}}
        s['allies']['u1']['orders']=[dict(ability_id=386,target_unit='u0')]
        s['keep_heal_order_candidates']={'u1':{'action_id':16,'target':'u0'}}
        return s

    def test_uncertainty_can_trigger_before_nominal_patient_exits_range(self):
        s,base,memory=self._anticipation_case()
        with patch.object(terran_combat,'allocate_adaptive_focus_fire',return_value=base):
            r=allocate_terran_combat(s,2,race_memory=memory)
        d=r['race_telemetry']['terran_coordination']['decisions']['u0']
        self.assertLess(d['baseline_geometry']['center_gap'],4)
        self.assertGreater(d['baseline_geometry']['upper_gap'],4)
        self.assertEqual(r['actions'],[2,6]);self.assertTrue(d['applied'])
        self.assertEqual(r['race_memory']['support_retreat_uses'],{'u0':1})

    def test_legacy_and_new_coordination_share_existing_two_move_budget(self):
        for used,expected in [(1,2),(2,4)]:
            s,base,memory=self._anticipation_case();memory['support_retreat_uses']={'u0':used}
            with patch.object(terran_combat,'allocate_adaptive_focus_fire',return_value=base):
                r=allocate_terran_combat(s,2,race_memory=memory)
            self.assertEqual(r['actions'][0],expected)
            self.assertEqual(r['race_memory']['support_retreat_uses']['u0'],2)

    def test_blocked_reversal_is_accounted_once_when_selected(self):
        s,base,memory=self._anticipation_case()
        memory['actors']={'u0':dict(tag='48',loop=0,position=[2.9,0],action=3)}
        with patch.object(terran_combat,'allocate_adaptive_focus_fire',return_value=base):
            r=allocate_terran_combat(s,2,race_memory=memory)
        self.assertEqual(r['actions'][0],2)
        self.assertTrue(r['race_memory']['v6_reversal_used']['u0'])
        self.assertEqual(r['race_memory']['support_retreat_uses']['u0'],1)

    def test_recover_stop_is_monitored_but_not_replaced(self):
        s,base,memory=self._anticipation_case();base['actions'][0]=1;base['roles']['u0']='recover'
        base['retreat_units']=[];base['retreat_actions']={}
        with patch.object(terran_combat,'allocate_adaptive_focus_fire',return_value=base):
            r=allocate_terran_combat(s,2,race_memory=memory)
        self.assertEqual(r['actions'][0],1)
        self.assertEqual(r['race_telemetry']['terran_coordination']['pair_monitor']['u1']['patient_role'],'recover')
        self.assertFalse(r['race_telemetry']['terran_coordination']['decisions']['u0']['applied'])

    def test_explicit_keep_preserves_order_without_fabricated_heal_assignment(self):
        s=self._keep_scene();before=copy.deepcopy(s);r=allocate_terran_combat(s,2)
        self.assertEqual(r['actions'][1],16);self.assertEqual(r['race_telemetry']['healing']['assignments'],{})
        self.assertTrue(r['race_telemetry']['terran_coordination']['keep_order']['u1']['selected'])
        self.assertEqual(s,before)
        s['game_loop']+=8
        again=allocate_terran_combat(s,2,race_memory=r['race_memory'])
        self.assertNotEqual(again['actions'][1],16)
        self.assertEqual(again['race_telemetry']['terran_coordination']['keep_order']['u1']['reason'],
                         'preserved_order_window_exhausted')
        self.assertTrue(again['race_telemetry']['terran_coordination']['keep_outcomes']['u1']['order_same'])

    def test_keep_requires_contract_energy_and_observed_target(self):
        for change in ('disabled','no_order','energy','wrong_candidate','unknown_motion'):
            s=self._keep_scene()
            if change=='disabled':s['action_contract']['keep_heal_order']['enabled']=False
            elif change=='no_order':s['allies']['u1']['orders']=[]
            elif change=='energy':s['allies']['u1']['energy']=.99
            elif change=='wrong_candidate':s['keep_heal_order_candidates']['u1']['target']='u9'
            else:s['unit_type_profiles']['54']['movement_speed_native']=None
            with self.subTest(change=change):self.assertNotEqual(allocate_terran_combat(s,2)['actions'][1],16)

    def test_keep_cannot_replace_legal_heal_or_necessary_rescue(self):
        s=self._keep_scene();s['allies']['u0']['position']=[3.9,0]
        s['heal_candidates']['u1']={'6':dict(target='u0',distance=3.9,missing_health=5)}
        self.assertEqual(allocate_terran_combat(s,2)['actions'][1],6)
        s=scene([unit('Marine',(4.2,0),40),unit('Marine',(3,0),1),unit('Medivac',(0,0))])
        s['action_contract']={'candidate_center_distance_by_type':{'medivac':4},'keep_heal_order':{'enabled':True,'action_id':16}}
        s['keep_heal_order_candidates']={'u2':{'action_id':16,'target':'u0'}}
        s['allies']['u2']['orders']=[dict(ability_id=386,target_unit='u0')]
        r=allocate_terran_combat(s,3)
        self.assertEqual(r['actions'][2],7)
        self.assertEqual(r['race_telemetry']['terran_coordination']['keep_order']['u2']['reason'],'necessary_legal_rescue')

    def test_keep_uncertainty_cannot_add_weapon_exposure(self):
        s=self._keep_scene();e=unit('Marine',(-7.5,0));s['visible_enemies']={'e0':e}
        for actor in s['allies']:
            s['relations'][actor]['e0']=dict(distance=math.dist(s['allies'][actor]['position'],e['position']),
                smac_attack_available=False,weapon_range_status='out_of_range',enemy_weapon_range_status='out_of_range',in_enemy_weapon_range=False)
        r=allocate_terran_combat(s,2);row=r['race_telemetry']['terran_coordination']['keep_order']['u1']
        self.assertFalse(row['selected']);self.assertEqual(row['reason'],'unsafe_preserved_order_motion')
        self.assertGreater(row['safety']['possible_compatible_threats'],row['safety']['current_compatible_threats'])

    def test_keep_followup_logs_lost_orders_death_full_and_energy(self):
        from skills.current.skills.heal_escort import select_keep_orders
        first=self._keep_scene();previous=allocate_terran_combat(first,2)['race_memory']['v6_keep_order']
        for reason in ('order_changed_or_missing','patient_dead','healer_dead','patient_full_health','insufficient_or_unknown_energy'):
            s=copy.deepcopy(first);s['game_loop']=16
            if reason=='order_changed_or_missing':s['allies']['u1']['orders']=[]
            elif reason=='patient_dead':s['allies']['u0']['alive']=False
            elif reason=='healer_dead':s['allies']['u1']['alive']=False
            elif reason=='patient_full_health':s['allies']['u0']['health']=45
            else:s['allies']['u1']['energy']=0
            _,_,outcomes=select_keep_orders(s,{'observed_orders':{}},previous,lambda *args:True,terran_combat.PARAMS)
            self.assertEqual(outcomes['u1']['reason'],reason)

    def test_all_v6_off_never_uses_supplied_keep_candidate(self):
        s=self._keep_scene()
        with patch.dict(terran_combat.V6_PARAMS,enabled=False):
            r=allocate_terran_combat(s,2)
        self.assertNotEqual(r['actions'][1],16)
        self.assertEqual(r['race_telemetry']['version'],'terran-specialization-v5')
        self.assertNotIn('v6_motion',r['race_memory'])


if __name__=='__main__':unittest.main()
