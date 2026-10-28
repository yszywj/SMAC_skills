"""Behavioral tests of Terran composition using real combat allocators."""
import copy
import math
import unittest

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


if __name__=='__main__':unittest.main()
