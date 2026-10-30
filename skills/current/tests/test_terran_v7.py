import copy
import unittest
from skills.current.skills import terran_combat, terran_motion
from skills.current.tests import test_terran_combat as fixtures
from skills.current.tests.test_terran_combat import scene, unit


class TerranV7Tests(unittest.TestCase):
    def test_productive_order_can_continue_without_new_candidate(self):
        s=fixtures.TerranV6Tests()._keep_scene();memory={}
        for _ in range(7):
            before=copy.deepcopy(s)
            r=terran_combat.allocate_terran_combat(s,2,race_memory=memory)
            self.assertEqual(r['actions'][1],16)
            self.assertEqual(r['race_telemetry']['healing']['assignments'],{})
            self.assertEqual(s,before)
            memory=r['race_memory'];s['game_loop']+=8;s['allies']['u1']['energy']-=1.2

    def test_stalled_order_exits_and_does_not_reset_budget(self):
        s=fixtures.TerranV6Tests()._keep_scene();memory={};actions=[]
        for _ in range(5):
            r=terran_combat.allocate_terran_combat(s,2,race_memory=memory)
            actions.append(r['actions'][1]);memory=r['race_memory'];s['game_loop']+=8
        self.assertEqual(actions[:2],[16,16]);self.assertNotIn(16,actions[2:])

    def test_progressing_follow_has_bounded_unconfirmed_duration(self):
        s=fixtures.TerranV6Tests()._keep_scene();s['allies']['u0']['position']=[8,0];memory={};a=[]
        for _ in range(6):
            r=terran_combat.allocate_terran_combat(s,2,race_memory=memory);a.append(r['actions'][1])
            memory=r['race_memory'];s['game_loop']+=8;s['allies']['u0']['position'][0]-=.15
        self.assertEqual(a[:4],[16]*4);self.assertNotIn(16,a[4:])

    def test_startup_and_acceleration_do_not_inherit_other_phase_error(self):
        s=scene([unit('Medivac',(0,0))]);s['unit_type_profiles']['54']['movement_speed_native']=2.5
        past={'tag':'54','loop':0,'position':[0,0],'kind':'heal','predicted_center':[1.25,0]}
        r=terran_motion.estimate(s,'u0',4,past)
        self.assertEqual(r['phase'],'startup');self.assertAlmostEqual(r['center'][0],.31625)
        self.assertLess(r['radius'],.3)
        s['allies']['u0']['position']=[.316,0];past['phase']='startup'
        r=terran_motion.estimate(s,'u0',4,past)
        self.assertEqual(r['phase'],'accelerating');self.assertLess(r['radius'],.4)

    def test_medical_motion_change_does_not_affect_ground_forecast(self):
        s=scene([unit('Marine',(0,0))]);r=terran_motion.estimate(s,'u0',4)
        self.assertEqual(r['center'],[1.,0.])


if __name__=='__main__':unittest.main()
