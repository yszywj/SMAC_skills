import unittest

from decision_tree.phase2.collection import select
from skills.local_v1 import FEATURE_NAMES


def step(number, distance):
    features = dict.fromkeys(FEATURE_NAMES, 0.0)
    features.update(ally_alive_count=10.0, nearest_enemy_before=distance)
    return {'event':'local_decision', 'step':number, 'candidates':[{
        'id':'regroup_step:u0:2','family':'regroup_step','features':features}]}


def receipt(action):
    return {'feedback':{'actors':{'u0':{'submission':'accepted','requested_action_id':str(action)}}}}


class AnchorSelection(unittest.TestCase):
    def test_adjacent_wait_then_consumed_observes_only_previous_receipts(self):
        steps = [step(0,15.0), step(1,8.0)]
        probes = {'old':{'mode':'regroup'}, 'patient':{'mode':'regroup','rules':{'far':'wait'}}}
        selected = select(steps, {1:receipt(6), 2:receipt(2)}, probes, supplement=False)
        self.assertEqual(list(selected), [0,1])
        self.assertEqual(selected[1][0]['action'],'execute')
        self.assertEqual(selected[1][0]['bucket'],'near_engaged')
        self.assertEqual(len(selected[0]),2)

    def test_no_supplement_does_not_force_other_family(self):
        event = step(0,5.0)
        event['candidates'][0].update(family='retarget', id='retarget:u0:6')
        probes = {'old':{'mode':'regroup'}}
        self.assertEqual(select([event],{1:receipt(6)},probes,supplement=False), {})
        self.assertEqual(select([event],{1:receipt(6)},probes,supplement=True)[0][0]['source'],
                         'outcome_blind_family_coverage_supplement')
