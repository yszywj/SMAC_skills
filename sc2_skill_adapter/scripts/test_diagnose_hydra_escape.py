"""Offline regression cases for prefix matching and sampled landing evidence."""
import unittest
from copy import deepcopy

if __package__:
    from .diagnose_hydra_escape import diagnose_move, prefix, previous_beat, status
else:
    from diagnose_hydra_escape import diagnose_move, prefix, previous_beat, status


def scene():
    actor = dict(tag='11', type='Hydralisk', unit_type_id=1, position=[0, 0], health=10, alive=True, radius=.5)
    enemy = dict(tag='12', unit_type_id=1, position=[-1.5, 0], radius=.5)
    return dict(allies={'u0': actor}, visible_enemies={'e0': enemy},
                unit_type_profiles={'1': dict(movement_speed_native=2, weapons=[dict(target_type='ground', range=1)])},
                movement_candidates={'u0': {'4': {'destination': [2, 0]}}},
                relations={'u0': {'e0': {'in_enemy_weapon_range': True}}})


class TestEscape(unittest.TestCase):
    def test_same_snapshot_not_same_prefix(self):
        step = ([], {'actors': {}, 'micro_submissions': []})
        a = ({0:scene(), 8:scene()}, {0:step, 8:deepcopy(step)})
        b = deepcopy(a)
        b[1][0][1]['micro_submissions'].append(dict(submitted_loop=6, actor_tag='20', target_tag='30', sc2_response=[1]))
        self.assertEqual(prefix(a, b, 8), (6, 'micro_command', []))
        b[1][0][1]['micro_submissions'][0]['sc2_response'] = [5]
        self.assertEqual(prefix(a, b, 8)[1], 'no_observed_divergence')
        b[0][8]['allies']['u0']['health'] = 9
        self.assertEqual(prefix(a, b, 8)[:2], (8, 'observed_state'))

    def test_previous_menu_not_policy_candidate(self):
        prev = dict(actor_tag='11', result='unchanged:weapon_ready_prefer_fire', candidate_action=None)
        got = previous_beat(scene(), prev, 'u0')
        self.assertEqual((got['recorded_candidate'], got['static_shed_moves']), (None, ['4']))
        self.assertEqual(status({}, 7, '11', 8), 'censored')

    def test_arrival_and_visibility_are_not_survival_or_death(self):
        state = scene()
        row = dict(actor='u0', actor_tag='11', game_loop=0, candidate_destination=[1,0], threat_tags={'e0':'12'},
                   current_threats=['e0'], worthwhile_move={'pursuer_covered_by':None, 'shed':['e0']}, ally_support={'e0':[]})
        after = deepcopy(state)
        after['visible_enemies']['e0']['tag'] = 'different'
        got = diagnose_move(row, {}, {0:state, 8:after}, {}, 8)
        self.assertEqual(got['arrival'], 'not_at_sampled_endpoint')
        self.assertEqual(got['samples'][0]['original_threat_unseen'], ['e0'])
        self.assertEqual(got['samples'][0]['original_threat_dead'], [])
        self.assertEqual(got['samples'][1]['survival'], 'censored')
        self.assertEqual(diagnose_move(row, {}, {0:state}, {'11':3}, 8)['arrival'], 'dead_before_next_sample')
        after['allies']['u0']['position'] = [1.1,0]
        self.assertEqual(diagnose_move(row, {}, {0:state, 8:after}, {}, 8)['first_arrival'], 8)


if __name__ == '__main__':
    unittest.main()
