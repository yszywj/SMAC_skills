"""Offline visible-observation, endpoint-scoring and split boundary checks."""
import copy
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import contact_front_shadow as shadow


def obs(loop, x, visible=True):
    # Two distant fronts; dog 0 is unsupported while dog 1 has hydra 2.
    def unit(kind, pos):
        return dict(type=kind, unit_type_id=1, position=pos, alive=True)

    allies = {'u0': unit('Zergling', [x, 0]), 'u1': unit('Zergling', [x, 20]),
              'u2': unit('Hydralisk', [5, 22])}
    enemies = {'e3': unit('Zergling', [6, 0]), 'e8': unit('Hydralisk', [6, 20]),
               'e9': unit('Hydralisk', [6, 23])} if visible else {}
    s = dict(game_loop=loop, step_mul=8, allies=allies, visible_enemies=enemies, relations={},
             unit_type_profiles={'1': {'movement_speed_native': 2}},
             action_contract={'candidate_center_distance_by_type': {'zergling': 2, 'hydralisk': 5}})
    questions = {}
    for k, u in allies.items():
        rel = s['relations'][k] = {}
        menu = questions[k] = {'criteria': {'1': 'stop'}}
        for e, v in enemies.items():
            legal = math.dist(u['position'], v['position']) <= (5 if k == 'u2' else 2)
            rel[e] = dict(smac_attack_available=legal, in_weapon_range=legal, in_enemy_weapon_range=False)
            if legal:
                menu['criteria'][str(6 + int(e[1:]))] = 'attack'
    return dict(state=s, questions=questions)


class ShadowTest(unittest.TestCase):
    def test_prefix_only_and_no_actions(self):
        ep = shadow.Episode()
        plain = obs(2, 1)
        ep.observe(plain, 'test')
        saved = copy.deepcopy(ep.updates)
        other = copy.deepcopy(plain)
        other['state'].update(layout='center', future={'e9': [99, 99]}, previous_action_ids={'u0': 9},
                              scenario_brief={'enemy_spawn': [0, 0]}, search_location=[77, 77])
        before = copy.deepcopy(other)
        alt = shadow.Episode()
        alt.observe(other, 'test')
        self.assertEqual(other, before)
        self.assertEqual(ep.updates, alt.updates)
        for loop, x in ((10, 2), (18, 3), (26, 4)):
            ep.observe(obs(loop, x), 'test')
        ep.observe({}, 'post contact: not inspected')
        self.assertEqual(ep.updates[:1], saved)
        self.assertEqual([u['loop'] for u in ep.updates], [2, 10, 18])
        summary = shadow.summarize([ep.result()])
        self.assertEqual(summary['earliest']['lead_loops_hist'], {24: 1})
        self.assertEqual(summary['last']['lead_loops_hist'], {8: 1})
        self.assertEqual(summary['last']['eligible'], 1)

    def test_tied_dogs_labels_and_same_front_joint(self):
        ep = shadow.Episode()
        ep.observe(obs(2, 4), 'test')
        self.assertEqual(ep.result()['status'], 'already_in_contact')
        self.assertEqual(ep.updates, [])
        a, b = (ep.label['dogs'][k] for k in ('u0', 'u1'))
        self.assertEqual(ep.label['canonical'], 'u0')
        self.assertEqual((a['support'], b['support']), ([], ['u2']))
        self.assertEqual(b['targets'], {'e8': [6, 20]})
        self.assertEqual(b['cluster'], {'e8': [6, 20], 'e9': [6, 23]})
        self.assertEqual(b['point'], [6, 21.5])
        pred = dict(dog='u1', dog_position=[4, 20], point=b['point'], cluster=list(b['cluster']), support=['u2'])
        scores = shadow.score(pred, ep.label)
        self.assertTrue(scores['joint_with_cluster'])
        self.assertFalse(scores['canonical_dog'])
        self.assertFalse(shadow.score(dict(pred, support=[]), ep.label)['joint'])
        self.assertFalse(shadow.score(dict(pred, point=a['point']), ep.label)['joint'])
        pred.update(dog='u0', support=[], point=a['point'])
        self.assertTrue(shadow.score(pred, ep.label)['both_empty'])
        self.assertTrue(shadow.score(pred, ep.label)['joint'])
        self.assertFalse(shadow.score(dict(pred, support=['u2']), ep.label)['joint'])
        a['support'] = None
        self.assertIsNone(shadow.score(pred, ep.label)['joint'])

    def test_vision_loss_abstention_not_stale_or_dropped(self):
        ep = shadow.Episode()
        ep.observe(obs(2, 2, False), 'test')
        self.assertEqual(ep.result()['status'], 'no_vision')
        ep.observe(obs(10, 2), 'test')
        self.assertEqual(ep.result()['status'], 'no_contact')
        ep.observe(obs(18, 2, False), 'test')
        ep.observe(obs(26, 4), 'test')
        summary = shadow.summarize([ep.result()])
        self.assertEqual(summary['earliest']['nearest']['abstained'], 0)
        self.assertEqual(summary['last']['kinematic']['abstained'], 1)
        self.assertEqual(summary['last']['eligible'], 1)
        self.assertEqual(summary['last']['lead_loops_hist'], {8: 1})

    def test_history_changes_kinematics_not_nearest(self):
        current = shadow.Frame.observe(obs(10, 2)['state'])
        prev = shadow.Frame.observe(obs(2, 0)['state'])
        with_history = shadow.forecasts(current, prev)
        cold = shadow.forecasts(current)
        self.assertEqual(with_history['nearest'], cold['nearest'])
        self.assertNotEqual(with_history['kinematic'], cold['kinematic'])
        self.assertEqual(with_history['kinematic']['projected_loops'], 8)

    def test_split_and_sequence_guards(self):
        a = {'shard': [dict(seed=1, scenario='A')]}
        for seed, scenario in ((1, 'B'), (2, 'A')):
            with self.assertRaisesRegex(ValueError, 'overlap'):
                shadow.check_splits([a, {'other': [dict(seed=seed, scenario=scenario)]}])
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            shadow.check_splits([a, {'one': a['shard'] * 2}])
        ep = shadow.Episode()
        ep.observe(obs(2, 2), 'test')
        with self.assertRaisesRegex(ValueError, 'unordered'):
            ep.observe(obs(18, 4), 'test')
        bad = obs(10, 4)
        bad['questions']['u0']['criteria'] = {}
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            ep.observe(bad, 'test')


if __name__ == '__main__':
    unittest.main()
