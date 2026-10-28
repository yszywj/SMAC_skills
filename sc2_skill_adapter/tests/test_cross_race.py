import copy
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
from s2clientprotocol import data_pb2
from bridge.config import load_config, validate_config, public_brief, RACE_UNITS, RACE_RANGES, map_file
from bridge.environment import RecordedEnv, StarCraft2Env, audit_unit_balance, is_medivac
from bridge.policy import build_request
from bridge.scenarios import generate_manifest, validate_manifest
from bridge import runner, skill_runtime
from test_bridge import FakeEnv, unit


class CrossRaceTests(unittest.TestCase):
    def test_three_native_manifests_deterministic_and_correct_map(self):
        for race in RACE_UNITS:
            args=load_config(10,race);args['unit_balance']='native'
            a=generate_manifest(args,2,170)
            self.assertEqual(a,generate_manifest(args,2,170))
            validate_manifest(a)
            self.assertTrue(map_file(args).is_file())
            self.assertEqual(public_brief(args)['candidate_ranges'],RACE_RANGES[race])
            for ep in a['episodes']:
                team=ep['episode_config']['team_gen']
                self.assertEqual(team['ally_team'],team['enemy_team'])
                self.assertEqual(len(team['ally_team']),10)

    def test_race_mismatch_rejected(self):
        args=load_config(10,'terran')
        args['capability_config']['team_gen']['unit_types']=['stalker']
        with self.assertRaises(ValueError):validate_config(args)

    def test_native_ids_mapped_for_both_owners(self):
        for race,ids in RACE_UNITS.items():
            env=RecordedEnv.__new__(RecordedEnv)
            env.unit_balance='native';env.map_name='10gen_'+race
            env.ally_unit_map={};env.enemy_unit_map={}
            with patch.object(StarCraft2Env,'_init_ally_unit_types'):
                env._init_ally_unit_types(0)
            self.assertEqual(env.ally_unit_map,ids)
            self.assertEqual(env.enemy_unit_map,ids)
            for name,uid in ids.items():self.assertEqual(getattr(env,name+'_id'),uid)

    def test_terran_healing_is_not_enemy_attack(self):
        env=FakeEnv();env.map_name='10gen_terran'
        env.scenario_brief=public_brief(load_config(5,'terran'))
        for units in (env.agents,env.enemies):
            for u in units.values():u.unit_type=48
        env.agents[0].unit_type=54
        env.agents[1].health=40
        env.unit_shoot_range=lambda i:4 if i==0 else 5
        env.get_avail_agent_actions=lambda i:([0,1,1,1,1,1,0,1,0,0,0] if i==0 else [0]+[1]*10)
        data=NS(units={54:'Medivac',48:'Marine'},unit_stats={})
        payload=build_request(env,data,'unused',1)
        self.assertEqual(payload['state']['heal_candidates']['u0']['7']['target'],'u1')
        self.assertIn('Heal ally',payload['questions']['u0']['criteria']['7'])
        self.assertFalse(any(r['smac_attack_available'] for r in payload['state']['relations']['u0'].values()))
        self.assertNotIn('e1',payload['state']['visible_enemies'])
        manager=skill_runtime.load_manager('baseline_v7_20261024')()
        manager.select('adaptive_focus_fire',posture='balanced')
        actions,_=manager.step(payload,5)
        self.assertEqual(actions[0],7)
        env.agents[1].health=env.agents[1].health_max
        full=build_request(env,data,'unused',1)
        self.assertEqual(full['state']['heal_candidates']['u0'],{})
        self.assertNotIn('7',full['questions']['u0']['criteria'])

    def test_zerg_snapshot_actions_and_no_hidden_enemies(self):
        env=FakeEnv();env.map_name='10gen_zerg'
        env.scenario_brief=public_brief(load_config(5,'zerg'))
        for units in (env.agents,env.enemies):
            for i,u in units.items():u.unit_type=[105,9,107][i%3]
        env.unit_shoot_range=lambda i:5 if env.agents[i].unit_type==107 else 2
        data=NS(units={105:'Zergling',9:'Baneling',107:'Hydralisk'},unit_stats={})
        payload=build_request(env,data,'unused',1)
        self.assertEqual(set(payload['state']['visible_enemies']),{'e0'})
        self.assertFalse(payload['state']['heal_candidates'])
        manager=skill_runtime.load_manager('baseline_v7_20261024')()
        manager.select('adaptive_focus_fire',posture='balanced')
        actions,_=manager.step(payload,5)
        for i,a in enumerate(actions):self.assertIn(str(a),payload['questions'][f'u{i}']['criteria'])

    def test_ground_only_weapon_cannot_target_flying_medivac(self):
        env=FakeEnv();env.map_name='10gen_terran'
        env.agents[0].unit_type=51
        env.agents[1].unit_type=48
        env.enemies[0].unit_type=54;env.enemies[0].is_flying=True
        env.unit_shoot_range=lambda i:6
        marauder=data_pb2.UnitTypeData(unit_id=51,name='Marauder')
        marauder.weapons.add(type=1,range=6,damage=10,attacks=1)
        marine=data_pb2.UnitTypeData(unit_id=48,name='Marine')
        marine.weapons.add(type=3,range=5,damage=6,attacks=1)
        data=NS(units={51:'Marauder',48:'Marine',54:'Medivac'},unit_stats={51:marauder,48:marine})
        payload=build_request(env,data,'unused',1)
        self.assertNotIn('6',payload['questions']['u0']['criteria'])
        self.assertFalse(payload['state']['relations']['u0']['e0']['smac_attack_available'])
        self.assertIn('6',payload['questions']['u1']['criteria'])
        self.assertTrue(env.get_avail_agent_actions(0)[6])  # upstream mask is unchanged

    def test_new_race_balance_rejects_attribute_mismatch(self):
        for race,uid in [('terran',48),('zerg',105)]:
            env=FakeEnv();env.map_name='10gen_'+race
            for units in (env.agents,env.enemies):
                for u in units.values():u.unit_type=uid
            data=NS(unit_stats={uid:data_pb2.UnitTypeData(unit_id=uid,name=race)})
            self.assertTrue(audit_unit_balance(env,data)['passed'])
            env.enemies[0].health_max+=1
            with self.assertRaises(RuntimeError):audit_unit_balance(env,data)

    def test_v7_freeze_is_exact_historical_skill_source(self):
        import hashlib,json
        from bridge.config import ROOT
        historical=json.loads((ROOT/'runs/native-10v10-v7-skill-n40-seed170/run.json').read_text())
        for path in skill_runtime.source_files('baseline_v7_20261024'):
            key='skills/current/'+path.relative_to(skill_runtime.SOURCES['baseline_v7_20261024']).as_posix()
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),historical['skill_source_hashes'][key])

if __name__=='__main__':unittest.main()
