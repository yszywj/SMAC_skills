import unittest

from decision_tree.zerg.squad_rendezvous import SquadRendezvous


def frame(xs=(0, 5, 9, 22, 26, 31)):
    s = dict(allies={}, visible_enemies={}, movement_candidates={})
    for i, x in enumerate(xs):
        uid = f'u{i}'
        s['allies'][uid] = dict(position=(x, 0), alive=True)
        s['movement_candidates'][uid] = {str(a): dict(destination=(x + dx, dy))
                                        for a, (dx, dy) in enumerate(((0, 2), (0, -2), (2, 0), (-2, 0)), 2)}
    return {'state': s}, [[0, 1, 1, 1, 1, 1] + [0]*10 for _ in xs], [2]*len(xs)


class SquadTest(unittest.TestCase):
    def test_local_groups_not_global_center(self):
        for mode in ('adaptive', 'fixed'):
            p, masks, base = frame()
            masks[0][6] = 1
            base[0] = 6
            masks[-1][5] = 0
            out = SquadRendezvous(mode).propose(p, masks, base)
            self.assertFalse(out['handoff'])
            self.assertEqual(out['actions'], [6, 2, 5, 4, 2, 2])
            self.assertEqual(base, [6, 2, 2, 2, 2, 2])

    def test_safety_releases_whole_vector_permanently(self):
        for mode in ('fixed', 'adaptive'):
            for reason in ('shared_target', 'close_enemy', 'casualty_or_damage'):
                with self.subTest(mode=mode, reason=reason):
                    policy = SquadRendezvous(mode)
                    p, masks, base = frame()
                    self.assertFalse(policy.propose(p, masks, base)['handoff'])
                    if reason == 'shared_target':
                        masks[1][9] = masks[3][9] = 1
                    elif reason == 'close_enemy':
                        p['state']['visible_enemies']['e2'] = dict(alive=True, position=(.3, 0))
                    else:
                        p['state']['allies']['u2']['health_lost_since_previous_decision'] = 1
                    out = policy.propose(p, masks, base)
                    self.assertEqual((out['reason'], out['handoff'], out['actions']), (reason, True, base))
                    p, masks, _ = frame()
                    self.assertEqual(policy.propose(p, masks, [3]*6)['actions'], [3]*6)

    def test_timeouts_and_compact_gate(self):
        for mode, limit in [('fixed', 4), ('adaptive', 8)]:
            self.assertEqual(SquadRendezvous(mode).propose(*frame(range(6)))['reason'], 'compact_start')
            policy = SquadRendezvous(mode)
            # Static observations simulate collision/stall, no reliance on progress.
            for _ in range(limit):
                self.assertFalse(policy.propose(*frame())['handoff'])
            self.assertEqual(policy.propose(*frame())['reason'], 'timeout')

    def test_formed_squads_timing_ablation(self):
        data = frame((0, 1, 2, 20, 21, 22))
        adaptive = SquadRendezvous().propose(*data)
        fixed = SquadRendezvous('fixed').propose(*data)
        self.assertTrue(adaptive['handoff'])
        self.assertFalse(fixed['handoff'])
        self.assertEqual(fixed['actions'], data[-1])

    def test_reject_bad_input(self):
        with self.assertRaises(ValueError):
            SquadRendezvous('unknown')
        p, masks, base = frame()
        base[1] = -1
        with self.assertRaises(ValueError):
            SquadRendezvous().propose(p, masks, base)
