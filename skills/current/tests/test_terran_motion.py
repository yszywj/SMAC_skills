import copy
import unittest

from skills.current.skills import terran_motion as motion
from skills.current.tests.test_terran_combat import scene,unit


class TerranMotionTests(unittest.TestCase):
    def test_move_uses_one_window_and_turn_and_crowding_allowances(self):
        s=scene([unit('Marine',(0,0))]);plain=motion.estimate(s,'u0',4)
        self.assertEqual(plain['center'],[1.,0.])
        history={'tag':'48','loop':0,'position':[0,-.5],'action':2}
        turning=motion.estimate(s,'u0',4,history)
        self.assertGreater(turning['radius'],plain['radius'])
        s['allies']['u1']=unit('Marine',(.7,0))
        self.assertGreater(motion.estimate(s,'u0',4)['radius'],plain['radius'])

    def test_confirmed_stationary_attack_narrows_but_new_target_is_uncertain(self):
        s=scene([unit('Marine',(0,0))],{'e0':unit('Marine',(4,0))})
        s['allies']['u0']['orders']=[dict(ability_id=23,target_unit='e0')]
        past={'tag':'48','loop':0,'position':[0,0],'action':6,'kind':'attack','target':'e0',
              'observed_attack_target':'e0','predicted_center':[0,0]}
        known=motion.estimate(s,'u0',6,past,'attack','e0')
        pending=motion.estimate(s,'u0',7,past,'attack','e1')
        self.assertEqual(known['center'],[0.,0.]);self.assertLess(known['radius'],pending['radius'])
        self.assertIn('same_order_observed_velocity',known['reasons'])

    def test_heal_to_keep_uses_only_observed_same_target_history(self):
        s=scene([unit('Marine',(4,0),20),unit('Medivac',(0,0))])
        s['allies']['u1']['orders']=[dict(ability_id=386,target_unit='u0')]
        past={'tag':'54','loop':0,'position':[-.5,0],'action':6,'kind':'heal','target':'u0',
              'observed_heal_target':'u0','predicted_center':[0,0]}
        current=motion.estimate(s,'u1',16,past,'keep','u0')
        self.assertEqual(current['center'],[.5,0.]);self.assertLess(current['radius'],1)
        past['observed_heal_target']=None
        pending=motion.estimate(s,'u1',16,past,'keep','u0')
        self.assertIn('automatic_motion_not_yet_observed',pending['reasons'])
        self.assertGreater(pending['radius'],current['radius'])

    def test_unknown_speed_and_stale_history_do_not_claim_a_forecast(self):
        s=scene([unit('Marine',(0,0))]);past={'tag':'old','loop':0,'position':[9,9],'predicted_center':[8,8]}
        self.assertNotIn('previous_prediction_error',motion.estimate(s,'u0',4,past))
        s['unit_type_profiles']['48']['movement_speed_native']=None
        self.assertFalse(motion.estimate(s,'u0',4)['known'])

    def test_large_prior_error_is_retained_and_marked_unreliable(self):
        s=scene([unit('Marine',(0,0))]);past={'tag':'48','loop':0,'position':[0,0],'predicted_center':[2,0]}
        row=motion.estimate(s,'u0',4,past)
        self.assertGreaterEqual(row['radius'],2);self.assertFalse(row['reliable'])

    def test_swept_uncertainty_detects_fire_crossing_and_missing_prediction(self):
        threats={'e0':([0,0],.5)}
        self.assertEqual(motion.possible_exposure(threats,[-2,0],{'known':True,'center':[2,0],'radius':.1}),1)
        self.assertIsNone(motion.possible_exposure(threats,[-2,0],{'known':False}))

    def test_prediction_and_memory_are_pure_and_deterministic(self):
        s=scene([unit('Marine',(0,0))]);before=copy.deepcopy(s)
        row=motion.estimate(s,'u0',4);memory=motion.remember(s,{'u0':row})
        self.assertEqual(row,motion.estimate(s,'u0',4));self.assertEqual(s,before)
        self.assertEqual(memory['u0']['action'],4)


if __name__=='__main__':unittest.main()
