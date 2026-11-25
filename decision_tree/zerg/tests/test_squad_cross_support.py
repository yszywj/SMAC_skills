import unittest
from copy import deepcopy
from math import dist

from decision_tree.zerg.squad_cross_support import CrossSupportRendezvous
from decision_tree.zerg.squad_rendezvous import SquadRendezvous
from decision_tree.zerg.tests.test_squad_rendezvous import frame


def enemy(data, eid, point, native, limit):
    p, masks, _ = data
    s = p['state']
    s['visible_enemies'][eid] = dict(alive=True, position=point, radius=.5)
    for i, (uid, u) in enumerate(s['allies'].items()):
        u['radius'] = .5
        s.setdefault('relations', {}).setdefault(uid, {})[eid] = dict(
            native_weapon_range=native, smac_attack_distance_limit=limit)
        masks[i][6 + int(eid[1:])] = int(dist(u['position'], point) <= limit)


def split():
    data = frame()
    enemy(data, 'e0', (15.5, 0), 4, 5)
    _, masks, base = data
    base[2:4] = [4, 5]
    masks[3][4] = 0  # Rear cannot move east to its own centroid.
    return data


def stagger(limit=3):
    data = frame((0, 1, 2, 12, 13, 14))
    enemy(data, 'e0', (12, 6), 2, limit)
    return data


class CrossSupportTest(unittest.TestCase):
    def test_guard_is_causal_veto(self):
        data = split()
        original = deepcopy(data)
        parent = SquadRendezvous().propose(*data)
        out = CrossSupportRendezvous().propose(*data)
        self.assertEqual(out['reason'], 'veto_split_contact')
        self.assertEqual(out['actions'], [*parent['actions'][:2], 4, *parent['actions'][3:]])
        self.assertEqual(out['veto'][0]['targets'], ['e0'])
        self.assertEqual(data, original)

    def test_guard_negative_controls(self):
        for case in ('no_radii', 'short_native', 'short_mask_range', 'unknown', 'incompatible',
                     'no_baseline_lane', 'no_separate_contact', 'new_lane'):
            with self.subTest(case=case):
                data = split()
                s = data[0]['state']
                r = s['relations']['u2']['e0']
                if case == 'no_radii':
                    del s['visible_enemies']['e0']['radius']
                elif case == 'short_native':
                    r['native_weapon_range'] = 3
                elif case == 'short_mask_range':
                    r['smac_attack_distance_limit'] = 4
                elif case == 'unknown':
                    del r['native_weapon_range']
                elif case == 'incompatible':
                    r['weapon_range_status'] = 'no_compatible_weapon'
                elif case == 'no_baseline_lane':
                    data[2][2] = 2
                elif case == 'no_separate_contact':
                    data[1][3][4] = 1
                else:
                    enemy(data, 'e1', (7, 4), 3.1, 4.1)
                self.assertEqual(CrossSupportRendezvous().propose(*data), SquadRendezvous().propose(*data))

    def test_front_changes_both_roles_guard_releases(self):
        data = stagger()
        self.assertEqual(CrossSupportRendezvous().propose(*data)['reason'], 'formed_or_blocked')
        out = CrossSupportRendezvous('front').propose(*data)
        self.assertEqual(out['reason'], 'front_sync')
        self.assertEqual(out['actions'][:3], [4, 4, 4])
        self.assertEqual(out['actions'][3], 5)  # Legal side-step toward rear, not deep pursuit.
        self.assertGreaterEqual(out['front']['eta'][1] - out['front']['eta'][0], 1)
        data[1][3][4:6] = [0, 0]
        self.assertEqual(CrossSupportRendezvous('front').propose(*data)['actions'][3], 1)

    def test_local_shared_mask_keeps_orders_but_rear_can_approach(self):
        data = stagger(6.2)
        data[2][3:5] = [6, 6]
        policy = CrossSupportRendezvous('front')
        out = policy.propose(*data)
        self.assertFalse(out['handoff'])
        self.assertEqual(out['actions'], [4, 4, 4, 6, 6, 2])
        data[1][2][6] = 1
        self.assertEqual(policy.propose(*data)['reason'], 'cross_squad_shared_target')
        self.assertTrue(policy.propose(*stagger())['handoff'])

    def test_no_guess_and_no_shared_front(self):
        for data in (frame(), stagger()):
            if data[0]['state']['visible_enemies']:
                # No legal range-closing first step for the rear.
                for row in data[1][:3]:
                    row[2] = row[4] = 0
                data[2][:3] = [5]*3
            for mode in ('guard', 'front'):
                self.assertEqual(CrossSupportRendezvous(mode).propose(*data), SquadRendezvous().propose(*data))

    def test_safety_after_parent_latch(self):
        for reason in ('timeout', 'close_enemy', 'casualty_or_damage'):
            with self.subTest(reason=reason):
                policy, data = CrossSupportRendezvous('front'), stagger(6.2)
                for _ in range(8 if reason == 'timeout' else 1):
                    self.assertFalse(policy.propose(*data)['handoff'])
                s = data[0]['state']
                if reason == 'close_enemy':
                    s['visible_enemies']['e0']['position'] = (12, 2.5)
                elif reason == 'casualty_or_damage':
                    s['allies']['u5']['shield_lost_since_previous_decision'] = 1
                self.assertEqual(policy.propose(*data), dict(actions=data[2], phase='v2', handoff=True, reason=reason))
                self.assertTrue(policy.propose(*frame())['handoff'])

    def test_compact_even_with_local_shared_mask(self):
        data = frame(range(6))
        enemy(data, 'e0', (1, 5), 2, 5.2)
        self.assertEqual(CrossSupportRendezvous('front').propose(*data)['reason'], 'compact_start')

    def test_validation(self):
        with self.assertRaises(ValueError):
            CrossSupportRendezvous('adaptive')
        data = frame()
        data[2][0] = 6
        with self.assertRaises(ValueError):
            CrossSupportRendezvous('front').propose(*data)
