"""Behavioral boundaries for the first visible-only Zerg specialization."""
import copy
import math
import unittest

from skills.current.skills.adaptive_focus_fire import allocate_adaptive_focus_fire
from skills.current.skills.zerg_combat import allocate_zerg_combat
from skills.current.skills.race_movement_guard import retreat_override_allowed


TYPES = {'Baneling': (9, None, 30), 'Zergling': (105, .1, 35), 'Hydralisk': (107, 5., 80)}


def scene(own=('Hydralisk',), enemies=('Hydralisk',), own_positions=None, enemy_positions=None):
    own_positions = own_positions or [(0., i * .1) for i in range(len(own))]
    enemy_positions = enemy_positions or [(1.5, i * .1) for i in range(len(enemies))]
    def unit(name, position, tag):
        unit_id, reach, hp = TYPES[name]
        return {'tag': tag, 'type': name, 'unit_type_id': unit_id, 'position': list(position),
                'alive': True, 'health': float(hp), 'max_health': float(hp), 'shield': 0., 'max_shield': 0.,
                'is_flying': False, 'radius': .375, 'weapon_ready': True, 'weapon_cooldown_loops': 0.,
                'health_lost_since_previous_decision': 0., 'shield_lost_since_previous_decision': 0.,
                'attack_upgrade_level': 0, 'armor_upgrade_level': 0, 'shield_upgrade_level': 0,
                'orders': []}
    allies = {f'u{i}': unit(name, own_positions[i], f'ally-{i}') for i, name in enumerate(own)}
    visible = {f'e{i}': unit(name, enemy_positions[i], f'enemy-{i}') for i, name in enumerate(enemies)}
    profiles = {}
    for name, (unit_id, reach, _) in TYPES.items():
        profiles[str(unit_id)] = {'known': True, 'movement_speed_native': 2.5, 'armor': 0.,
                                 'attributes': [1, 3], 'weapons': [] if reach is None else [
                                     {'range': reach, 'target_type': 'ground', 'base_damage': 5. if name == 'Zergling' else 12.,
                                      'hits_per_attack': 1, 'damage_bonus': []}]}
    relations, moves = {}, {}
    for actor, ally in allies.items():
        reach = TYPES[ally['type']][1]
        mask_reach = 2. if reach is None or reach < 1. else reach
        relations[actor] = {}
        for enemy, target in visible.items():
            d = math.dist(ally['position'], target['position'])
            enemy_reach = TYPES[target['type']][1]
            relations[actor][enemy] = {'distance': d, 'native_weapon_range': reach,
                'in_weapon_range': None if reach is None else d <= reach + .75,
                'weapon_range_status': 'unknown' if reach is None else 'known',
                'in_enemy_weapon_range': None if enemy_reach is None else d <= enemy_reach + .75,
                'smac_attack_available': d <= mask_reach, 'smac_attack_distance_limit': mask_reach}
        x, y = ally['position']
        moves[actor] = {str(a): {'destination': list(pos), 'distance_to_search_location': math.dist(pos, (16., 16.)),
                                'distances_to_visible_enemies': {e: math.dist(pos, u['position']) for e, u in visible.items()}}
                        for a, pos in ((2, (x, y+2)), (3, (x, y-2)), (4, (x+2, y)), (5, (x-2, y)))}
    return {'allies': allies, 'visible_enemies': visible, 'unit_type_profiles': profiles,
            'relations': relations, 'movement_candidates': moves, 'heal_candidates': {},
            'game_loop': 80, 'step_mul': 8, 'search_location': [16., 16.], 'steps_remaining': 100}


class ZergCombatTests(unittest.TestCase):
    def test_melee_keeps_team_finishing_target_instead_of_nearest_full_health_unit(self):
        state=scene(('Zergling',),('Zergling','Hydralisk'),enemy_positions=[(.69,0),(.8,0)])
        state['visible_enemies']['e1']['health']=1
        result=allocate_zerg_combat(state,1)
        self.assertEqual(result['actions'],[7])
        self.assertEqual(result['race_telemetry']['actors']['u0']['reason'],'inherit_team_focus_target')
        self.assertEqual(result['race_telemetry']['reset_actor_histories'],[])

    def test_melee_preserves_base_retreat_and_diagnostic_reason(self):
        state=scene(('Zergling',),enemy_positions=[(2,0)])
        state['allies']['u0'].update(health=9,weapon_ready=False,weapon_cooldown_loops=20)
        base=allocate_adaptive_focus_fire(state,1)
        self.assertEqual(base['roles']['u0'],'retreat')
        result=allocate_zerg_combat(state,1)
        self.assertEqual(result['actions'],base['actions'])
        self.assertEqual(result['retreat_reasons'],base['retreat_reasons'])

    def test_v2_ready_shot_distance_boundary_is_preserved(self):
        # This ablation deliberately restores v2's fixed threshold, including
        # its limitations; observed enemy velocity must not change that policy.
        for distance, expected_attack in ((1.5, False), (1.501, True)):
            with self.subTest(distance=distance):
                state=scene(enemies=('Baneling',),enemy_positions=[(distance,0)])
                result=allocate_zerg_combat(state,1)
                self.assertEqual(result['actions'][0] >= 6,expected_attack)
        state=scene(('Zergling',),('Zergling','Baneling'),enemy_positions=[(.69,0),(1.667,0)])
        memory={'visible_banelings':dict(loop=72,units={'e1':dict(tag='enemy-1',position=[2.6,0])})}
        result=allocate_zerg_combat(state,1,race_memory=memory)
        self.assertEqual(result['actions'],[6])
        self.assertEqual(result['race_telemetry']['actors']['u0']['avoidance_skipped'],'preserve_fire_window')

    def test_v2_reverse_move_restriction_only_relaxes_at_contact(self):
        memory={'actors':{'u0':dict(tag='ally-0',loop=72,action=4,position=[-1.25,0])}}
        for distance, expected_action in ((1.4,2),(.8,5)):
            with self.subTest(distance=distance):
                state=scene(enemies=('Baneling',),enemy_positions=[(distance,0)])
                state['allies']['u0'].update(weapon_ready=False,weapon_cooldown_loops=20)
                result=allocate_zerg_combat(state,1,race_memory=memory)
                self.assertEqual(result['actions'],[expected_action])
                self.assertEqual(result['race_telemetry']['actors']['u0']['contact_emergency'],distance < 1.)

    def test_v3_enemy_motion_memory_is_unused_and_not_retained(self):
        for distance in (.8,1.4,1.667,2.929):
            for streak in (0,2):
                for cooldown in (0,20):
                    with self.subTest(distance=distance,streak=streak,cooldown=cooldown):
                        state=scene(enemies=('Baneling',),enemy_positions=[(distance,0)])
                        state['allies']['u0'].update(weapon_ready=cooldown == 0,weapon_cooldown_loops=cooldown)
                        memory={'actors':{'u0':dict(tag='ally-0',loop=72,action=5,escape_streak=streak)}}
                        baseline=allocate_zerg_combat(state,1,race_memory=memory)
                        memory['visible_banelings']=dict(loop=72,units={'e0':dict(tag='enemy-0',position=[100,0])})
                        saved=copy.deepcopy((state,memory))
                        result=allocate_zerg_combat(state,1,race_memory=memory)
                        self.assertEqual(result,baseline)
                        self.assertEqual((state,memory),saved)
                        self.assertNotIn('visible_banelings',result['race_memory'])
                        self.assertNotIn('baneling_contact_risk',result['race_telemetry']['actors'].get('u0',{}))

    def test_safe_window_returns_to_team_attack_even_with_long_escape_history(self):
        state=scene(('Zergling',),('Zergling','Baneling'),enemy_positions=[(.69,0),(3.07,0)])
        memory={'actors':{'u0':dict(tag='ally-0',loop=72,action=5,escape_streak=8)}}
        result=allocate_zerg_combat(state,1,race_memory=memory)
        self.assertEqual(result['actions'],[6])
        self.assertEqual(result['race_memory']['actors']['u0']['escape_streak'],0)

    def test_distant_baneling_does_not_cancel_in_range_melee_shot(self):
        state=scene(('Zergling',),('Zergling','Baneling'),enemy_positions=[(.69,0),(3.07,0)])
        r=allocate_zerg_combat(state,1)
        self.assertEqual(r['actions'],[6])
        self.assertEqual(r['race_telemetry']['actors']['u0']['avoidance_skipped'],'preserve_fire_window')

    def test_escape_budget_returns_to_attack_but_contact_emergency_can_continue(self):
        state=scene(enemies=('Baneling',),enemy_positions=[(1.4,0)])
        memory={'actors':{'u0':dict(tag='ally-0',loop=72,action=5,escape_streak=2)}}
        r=allocate_zerg_combat(state,1,race_memory=memory)
        self.assertEqual(r['actions'],[6])
        self.assertEqual(r['race_memory']['actors']['u0']['escape_streak'],0)
        state=scene(enemies=('Baneling',),enemy_positions=[(.8,0)])
        r=allocate_zerg_combat(state,1,race_memory=memory)
        self.assertIn(r['actions'][0],(2,3,4,5))
        self.assertTrue(r['race_telemetry']['actors']['u0']['contact_emergency'])

    def test_lethal_interception_requires_native_range_and_ready_window(self):
        state=scene(('Zergling',),('Baneling',),enemy_positions=[(.7,0)])
        state['visible_enemies']['e0']['health']=4
        r=allocate_zerg_combat(state,1,parameters={'accurate_damage':True})
        self.assertEqual(r['actions'],[6])
        self.assertEqual(r['race_telemetry']['actors']['u0']['reason'],'feasible_lethal_baneling_interception')
        state['relations']['u0']['e0']['in_weapon_range']=False
        r=allocate_zerg_combat(state,1,parameters={'accurate_damage':True})
        self.assertIn(r['actions'][0],(2,3,4,5))
        state['relations']['u0']['e0']['in_weapon_range']=True
        state['allies']['u0'].update(weapon_ready=False,weapon_cooldown_loops=20)
        r=allocate_zerg_combat(state,1,parameters={'accurate_damage':True})
        self.assertIn(r['actions'][0],(2,3,4,5))

    def test_hydralisk_screen_does_not_replace_equally_risky_base_retreat(self):
        state=scene(('Hydralisk','Zergling'),own_positions=[(0,0),(0,2)],enemy_positions=[(2,0)])
        state['allies']['u0'].update(health=10,weapon_ready=False,weapon_cooldown_loops=20)
        base=allocate_adaptive_focus_fire(state,2)
        r=allocate_zerg_combat(state,2)
        self.assertEqual(r['actions'][0],base['actions'][0])
        self.assertNotEqual(r['race_telemetry']['actors'].get('u0',{}).get('reason'),'hydralisk_retreat_toward_screen')

    def test_retreat_overlay_cannot_delay_reentry_or_reverse_move(self):
        state=scene(enemy_positions=[(4.7,0)])
        # North retains an attack next window; west adds another movement window.
        self.assertFalse(retreat_override_allowed(state,'u0',2,5))
        history=dict(tag='ally-0',loop=72,action=3,position=[0,1.25])
        self.assertFalse(retreat_override_allowed(state,'u0',5,2,history=history))

    def assert_legal(self, state, result):
        for actor, unit in state['allies'].items():
            action = result['actions'][int(actor[1:])]
            if not unit['alive']:
                self.assertEqual(action, 0)
                continue
            legal = {1, *map(int, state['movement_candidates'][actor])}
            legal.update(6 + int(e[1:]) for e, r in state['relations'][actor].items()
                         if r['smac_attack_available'] and e in state['visible_enemies'])
            self.assertIn(action, legal)
        for actor, target in result['attack_assignments'].items():
            self.assertEqual(result['actions'][int(actor[1:])], 6+int(target[1:]))

    def test_baneling_unknown_effect_is_not_shooting_damage_or_retreat(self):
        state = scene(('Baneling',))
        state['allies']['u0']['health'] = 1.
        history = {'actors': {'u0': {'tag': 'ally-0', 'target': 'e0', 'loop': 72}}}
        before = copy.deepcopy((state, history))
        result = allocate_zerg_combat(state, 1, race_memory=history, all_in=True, stall=100, contacted=True,
                                      unit_states={'u0': {'role': 'retreat', 'retreat_cycles': 99}})
        self.assertEqual(result['actions'], [6])
        self.assertEqual(result['roles']['u0'], 'baneling_strike')
        self.assertFalse(result['all_in'])
        self.assertFalse(result['terminated'])
        self.assertEqual(result['retreat_units'], [])
        self.assertEqual(result['predicted_damage'], {})
        self.assertEqual(result['damage_model']['unknown_damage_actors'], ['u0'])
        self.assertIsNone(result['race_telemetry']['actors']['u0']['explosion_radius'])
        self.assertEqual((state, history), before)
        self.assert_legal(state, result)

    def test_multiple_banelings_allocate_separate_visible_groups(self):
        state = scene(('Baneling', 'Baneling'), ('Hydralisk',)*4,
                      enemy_positions=[(3., 0.), (3., .3), (-3., 0.), (-3., .3)])
        result = allocate_zerg_combat(state, 2)
        targets = result['race_telemetry']['baneling_targets']
        self.assertNotEqual(int(targets['u0'][1:])//2, int(targets['u1'][1:])//2)
        self.assertEqual(result['predicted_damage'], {})
        self.assert_legal(state, result)

    def test_spread_overrides_legal_attacks_and_reserves_different_destinations(self):
        state = scene(('Hydralisk', 'Hydralisk'), ('Baneling',),
                      own_positions=[(0., 0.), (0., 0.)], enemy_positions=[(1., 0.)])
        baseline = allocate_adaptive_focus_fire(state, 2)
        self.assertTrue(all(a >= 6 for a in baseline['actions']))
        result = allocate_zerg_combat(state, 2, parameters={'engagement_commitment': True, 'accurate_damage': True})
        self.assertTrue(all(a in (2, 3, 4, 5) for a in result['actions']))
        self.assertNotEqual(*result['actions'])
        self.assertEqual(result['attack_assignments'], {})
        self.assertEqual(result['predicted_damage'], {})
        self.assertEqual(result['engagement_state_updates'], {})
        self.assertEqual(result['race_telemetry']['reset_actor_histories'], ['u0', 'u1'])
        for row in result['race_telemetry']['actors'].values():
            self.assertEqual(row['reason'], 'anti_baneling_geometric_spread')
            self.assertGreater(row['predicted_distance_after'], row['distance_before'])
        self.assert_legal(state, result)

    def test_no_banelings_preserves_hydralisk_baseline_and_damage(self):
        state = scene(enemy_positions=[(4., 0.)])
        before = copy.deepcopy(state)
        baseline = allocate_adaptive_focus_fire(state, 1)
        result = allocate_zerg_combat(state, 1)
        self.assertEqual(result['actions'], baseline['actions'])
        self.assertEqual(result['predicted_damage'], {e:d for e,d in baseline['predicted_damage'].items() if d})
        self.assertGreater(result['predicted_damage']['e0'], 0)
        self.assertEqual(result['race_telemetry']['reset_actor_histories'], [])
        self.assertEqual(state, before)

    def test_dead_banelings_do_not_reduce_ordinary_all_in_force_ratio(self):
        state = scene(('Hydralisk', 'Baneling', 'Baneling', 'Baneling'))
        for actor in ('u1', 'u2', 'u3'):
            state['allies'][actor].update(alive=False, health=0.)
        result = allocate_zerg_combat(state, 4, contacted=True, stall=100)
        self.assertFalse(result['all_in'])
        self.assertEqual(result['actions'][1:], [0, 0, 0])
        self.assertEqual(set(result['roles']), {'u0'})
        self.assert_legal(state, result)

    def test_all_dead_returns_only_noop(self):
        state = scene(('Baneling',))
        state['allies']['u0'].update(alive=False, health=0.)
        result = allocate_zerg_combat(state, 1)
        self.assertEqual(result['actions'], [0])
        self.assertTrue(result['terminated'])
        self.assertEqual(result['race_memory'], {'actors': {}})

    def test_fog_drops_hidden_baneling_target_and_uses_only_legal_search(self):
        state = scene(('Baneling',), ())
        memory = {'actors': {'u0': {'tag': 'ally-0', 'loop': 72, 'target': 'e99'}}}
        result = allocate_zerg_combat(state, 1, race_memory=memory)
        self.assertEqual(result['race_telemetry']['baneling_targets'], {})
        self.assertIsNone(result['race_memory']['actors']['u0']['target'])
        self.assertEqual(result['attack_assignments'], {})
        self.assert_legal(state, result)

    def test_movement_mask_and_stop_fallback(self):
        state = scene(('Baneling',), enemy_positions=[(10., 0.)])
        state['movement_candidates']['u0'] = {'2': state['movement_candidates']['u0']['2']}
        result = allocate_zerg_combat(state, 1)
        self.assertEqual(result['actions'], [2])
        self.assert_legal(state, result)
        state['movement_candidates']['u0'] = {}
        self.assertEqual(allocate_zerg_combat(state, 1)['actions'], [1])

    def test_melee_targets_keep_shared_damage_allocation_without_inventing_hits(self):
        state = scene(('Zergling', 'Zergling'), ('Hydralisk', 'Hydralisk'),
                      enemy_positions=[(.8, 0.), (.8, .1)])
        result = allocate_zerg_combat(state, 2)
        baseline = allocate_adaptive_focus_fire(state,2)
        self.assertEqual(result['attack_assignments'], baseline['attack_assignments'])
        self.assertEqual(result['predicted_damage'], {e:d for e,d in baseline['predicted_damage'].items() if d})
        self.assert_legal(state, result)

    def test_observed_melee_blockage_moves_and_invalidates_old_attack_prediction(self):
        state = scene(('Zergling',), enemy_positions=[(1.8, 0.)])
        memory = {'actors': {'u0': {'tag': 'ally-0', 'loop': 72, 'target': 'e0', 'action': 6,
                                    'position': [0., 0.], 'blocked_steps': 1}}}
        result = allocate_zerg_combat(state, 1, race_memory=memory,
                                     parameters={'engagement_commitment': True, 'approach_progress_guard': True})
        self.assertIn(result['actions'][0], (2, 3, 4, 5))
        self.assertEqual(result['attack_assignments'], {})
        self.assertEqual(result['predicted_damage'], {})
        self.assertEqual(result['engagement_state_updates'], {})
        self.assertEqual(result['approach_state_updates'], {})
        self.assertIn('u0', result['race_telemetry']['reset_actor_histories'])
        self.assert_legal(state, result)


if __name__ == '__main__':
    unittest.main()
