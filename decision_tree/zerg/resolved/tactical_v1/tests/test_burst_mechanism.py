"""Native-effect boundaries, final-action accounting and S3 lifecycle tests."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import unittest

from skills.tactical_v1.burst import approach_value, damage
from skills.tactical_v1.context import Context, build_snapshot
from skills.tactical_v1.effects import physical_effects
from skills.tactical_v1.memory import TacticalMemory
from skills.tactical_v1.skills.burst_engage import BurstEngageSkill
from skills.tactical_v1.telemetry import Telemetry
from .fixtures import enemy, scene, with_questions


def native_scene(allies=None, enemies=None, **kwargs):
    state = scene(allies or [((0., 0.), "baneling")],
                  enemies if enemies is not None else {"e0": enemy((1.5, 0.), "hydralisk", health=90., max_health=90., radius=.625)},
                  attack_range=2., **kwargs)
    state['unit_type_profiles']['9'] = {'known':True, 'movement_speed_native':2.5,
                                       'armor':0., 'attributes':[3], 'weapons':[]}
    for unit in state['allies'].values():
        if unit['type']=='baneling':unit.update(health=30.,max_health=30.,radius=.375)
    for actor, relations in state['relations'].items():
        for enemy_id, relation in relations.items():
            if state['allies'][actor]['type']=='baneling':
                relation.update(in_weapon_range=None,native_weapon_range=None,weapon_range_status='unknown')
            if state['visible_enemies'][enemy_id]['type']=='baneling':
                relation.update(in_enemy_weapon_range=None,enemy_weapon_range_status='unknown')
    return state


def context(state):
    return Context(build_snapshot(with_questions(state),1))


class BurstMechanismTests(unittest.TestCase):
    def test_native_profile_is_measured_bound_and_not_command_radius(self):
        c=context(native_scene())
        p=c.burst_profile()
        self.assertTrue(p.damage_known)
        self.assertLess(p.blast_radius, p.blast_radius_upper)
        self.assertGreater(p.blast_radius,2.)
        self.assertLess(p.trigger_reach,.25)
        self.assertEqual(damage(c,'e0'),35.)

    def test_catalog_and_upgrade_mismatch_never_fabricate_damage(self):
        state=native_scene();state['unit_type_profiles']['9']['armor']=1.
        self.assertFalse(context(state).burst_profile().geometry_known)
        self.assertIsNone(damage(context(state),'e0'))
        state=native_scene();state['allies']['u0']['attack_upgrade_level']=1
        self.assertTrue(context(state).burst_profile().geometry_known)
        self.assertIsNone(damage(context(state),'e0'))

    def test_light_armor_air_and_unknown_shield_remain_separate(self):
        state=native_scene();state['unit_type_profiles']['107']['attributes']=[2,3]
        state['unit_type_profiles']['107']['armor']=1.
        self.assertEqual(damage(context(state),'e0'),15.)
        state['visible_enemies']['e0']['is_flying']=True
        self.assertEqual(damage(context(state),'e0'),0.)
        state['visible_enemies']['e0'].update(is_flying=False,shield=10.)
        self.assertIsNone(damage(context(state),'e0'))

    def test_far_enemy_baneling_not_global_imminent_danger(self):
        state=native_scene([((0.,0.),'hydralisk')],{'e0':enemy((8.,0.),'baneling')})
        c=context(state)
        self.assertEqual(c.risk_at('u0',(0.,0.)).effect_weapon_near,())
        self.assertEqual(c.forecast_risk('u0',(0.,0.)).effect_weapon_near,())
        self.assertIn('e0',c.forecast_risk('u0',(4.5,0.)).effect_weapon_near)

    def test_unknown_damage_does_not_destroy_known_geometry(self):
        state=native_scene([((0.,0.),'hydralisk')],{'e0':enemy((8.,0.),'baneling')})
        state['unit_type_profiles']['107'].pop('armor')
        c=context(state)
        self.assertIsNone(c.enemy_attack_damage('e0','u0'))
        self.assertEqual(c.forecast_risk('u0',(0.,0.)).threats,())
        self.assertIn('e0',c.forecast_risk('u0',(6.,0.)).effect_weapon_near)

    def test_ground_blast_cannot_threaten_flying_healer(self):
        c=context(native_scene([((0.,0.),'medivac')],{'e0':enemy((.1,0.),'baneling')}))
        self.assertEqual(c.forecast_risk('u0',(0.,0.)).threats,())
        self.assertEqual(c.risk_at('u0',(0.,0.)).effect_weapon_near,())

    def test_effect_uses_source_trigger_center_not_enemy_center(self):
        state=native_scene(enemies={'e0':enemy((1.5,0.),'hydralisk',health=90.,radius=.625),
                                    'e1':enemy((3.4,0.),'hydralisk',health=90.,radius=.625)})
        c=context(state);report=physical_effects(c,{'u0':6})
        opportunity = report.burst_outcomes['u0'].opportunity
        self.assertGreater(opportunity, 0.)
        self.assertLess(opportunity, 1., 'a moving approach under possible fire is not certain')
        self.assertEqual(report.burst_damage_potential,{'e0':35. * opportunity})
        self.assertEqual(report.damage_known,{})
        self.assertEqual(report.coverage,{})
        self.assertEqual(report.planned_loss,{'u0':opportunity})
        self.assertLess(report.burst_outcomes['u0'].position[0],.5)
        self.assertEqual(approach_value(c,{'u0':6},report),0.)

    def test_death_trigger_can_contribute_without_attack_command(self):
        state=native_scene();state['allies']['u0']['health']=1.
        report=physical_effects(context(state),{'u0':4})
        self.assertEqual(report.burst_outcomes['u0'].trigger,'death_at_origin_opportunity')
        opportunity = report.burst_outcomes['u0'].opportunity
        self.assertGreater(opportunity, 0.)
        self.assertLess(opportunity, 1., 'possible enemy targeting is not guaranteed death')
        self.assertEqual(report.burst_damage_potential,{'e0':35. * opportunity})
        self.assertEqual(report.planned_loss,{'u0':opportunity})

    def test_disabled_mechanism_cannot_create_consumption_or_effect(self):
        report=physical_effects(context(native_scene()),{'u0':6},mechanisms={'burst':False,'splash':True})
        self.assertEqual(report.burst_damage_potential,{})
        self.assertEqual(report.planned_loss,{})

    def test_enemy_killed_near_allies_retains_on_death_exposure(self):
        state=native_scene([((0.,0.),'hydralisk')],{'e0':enemy((1.,0.),'baneling',health=1.,max_health=30.)})
        report=physical_effects(context(state),{'u0':6})
        self.assertGreater(report.damage_known['e0'],1.)
        self.assertEqual(report.enemy_death_burst['u0'],35.)

    def test_approach_value_is_final_action_based_bounded_and_survival_sensitive(self):
        state=native_scene(enemies={'e0':enemy((5.,0.),'hydralisk',health=90.,radius=.625)})
        c=context(state)
        toward=approach_value(c,{'u0':4},physical_effects(c,{'u0':4}))
        away=approach_value(c,{'u0':5},physical_effects(c,{'u0':5}))
        self.assertGreater(toward,0.);self.assertLessEqual(toward,1.)
        self.assertEqual(away,0.)
        state['allies']['u0']['health']=10.
        injured=context(state)
        self.assertLess(approach_value(injured,{'u0':4},physical_effects(injured,{'u0':4})),toward)

    def test_s3_goal_phase_transition_and_bounded_commitment(self):
        c=context(native_scene());memory=TacticalMemory()
        memory.create_instance('burst_engage',['u0'],'e0','approaching',72,48,goal_kind='enemy')
        plans=BurstEngageSkill({}).propose(c,memory,Telemetry())
        attack=next(p for p in plans if p.current_actions.get('u0')==6)
        self.assertEqual(attack.goal_kind,'enemy')
        self.assertEqual(attack.goal_data['objective']['kind'],'burst_region')
        self.assertEqual(attack.proposed_memory_delta['kind'],'transition')
        self.assertEqual(attack.proposed_memory_delta['phase'],'bursting')
        self.assertEqual(attack.commitment.min_loops,c.H)
        self.assertEqual(attack.commitment.max_loops,6*c.H)

    def test_s3_proposals_have_distinct_residual_regions(self):
        state=native_scene([((0.,0.),'baneling'),((0.,1.),'baneling')],
             {'e0':enemy((5.,0.),'zergling'), 'e1':enemy((0.,6.),'zergling')})
        plans=BurstEngageSkill({}).propose(context(state),TacticalMemory(),Telemetry())
        first={}
        for plan in plans:first.setdefault(plan.controlled_actors[0],plan.goal)
        self.assertNotEqual(first['u0'],first['u1'])

    def test_s1_effects_match_frozen_before_for_pure_colossus(self):
        root=Path(__file__).resolve().parents[3]
        path=root/'runs/tactical_all_races_10v10_9x20_mechanisms_20261115/before_source_snapshot/skills/tactical_v1/effects.py'
        if not path.exists():self.skipTest('optional frozen before snapshot unavailable')
        spec=importlib.util.spec_from_file_location('skills.tactical_v1._before_effects_test',path)
        old=importlib.util.module_from_spec(spec);sys.modules[spec.name]=old;spec.loader.exec_module(old)
        for cooldown in (0.,20.):
            state=scene([((0.,0.),'colossus')],{'e0':enemy((5.,0.),health=150.),
                        'e1':enemy((5.5,.5),health=150.),'e2':enemy((5.5,-.5),is_flying=True)},attack_range=7.)
            state['allies']['u0']['weapon_cooldown_loops']=cooldown;c=context(state)
            expected=old.physical_effects(c,{'u0':6},splash_reliability=.5)
            actual=physical_effects(c,{'u0':6},splash_reliability=.5)
            for field in ('damage_known','damage_bounds','coverage','coverage_raw','coverage_sources','planned_loss'):
                self.assertEqual(getattr(actual,field),getattr(expected,field),field)

    def test_z171_recorded_far_baneling_no_longer_threatens_every_backliner(self):
        root=Path(__file__).resolve().parents[3]
        log=root/'runs/tactical_all_races_10v10_9x20_firecycle_20261114/current/z_4l5h1b/events.jsonl'
        if not log.exists():self.skipTest('optional Z171 log unavailable')
        checked=0
        with log.open(encoding='utf8') as stream:
            for line in stream:
                row=json.loads(line)
                if row.get('episode')!=2 or row.get('event')!='skill_observation':continue
                c=Context(build_snapshot(row['payload'],row.get('step',0)+1))
                for actor,unit in c.snapshot.allies.items():
                    if not unit.alive:continue
                    for enemy_id,target in c.snapshot.enemies.items():
                        if target.type_name!='baneling':continue
                        from skills.tactical_v1.context import distance
                        if distance(unit.position,target.position)>7.:
                            self.assertNotIn(enemy_id,c.forecast_risk(actor,unit.position).effect_weapon_near)
                            checked+=1
        self.assertGreater(checked,0)


if __name__=='__main__':unittest.main()
