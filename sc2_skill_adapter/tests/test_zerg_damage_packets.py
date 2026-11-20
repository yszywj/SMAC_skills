import unittest

from scripts.zerg_damage_packets import infer_packets


class ZergDamagePacketTests(unittest.TestCase):
    def test_single_and_combined_surviving_shots(self):
        counts = {'Zergling': 1, 'Hydralisk': 1}
        self.assertEqual(infer_packets(5, counts, 'Baneling'), {'Zergling': 5})
        self.assertEqual(infer_packets(11.82, counts, 'Hydralisk'), {'Hydralisk': 11.82})
        assigned = infer_packets(16.8, counts, 'Zergling')
        self.assertAlmostEqual(assigned['Zergling'], 5 * 16.8 / 17)
        self.assertAlmostEqual(sum(assigned.values()), 16.8)

    def test_only_light_victims_receive_bonus(self):
        counts = {'Baneling': 1}
        self.assertEqual(infer_packets(35, counts, 'Hydralisk'), {'Baneling': 35})
        self.assertEqual(infer_packets(16, counts, 'Baneling'), {'Baneling': 16})
        self.assertEqual(infer_packets(35, counts, 'Baneling'), {})

    def test_common_contribution_across_ambiguous_mixtures(self):
        counts = {'Zergling': 7, 'Hydralisk': 1, 'Baneling': 1}
        self.assertEqual(infer_packets(35, counts, 'Hydralisk'), {})
        self.assertEqual(infer_packets(47, counts, 'Hydralisk'), {'Hydralisk': 12})

    def test_lethal_attacker_only_receives_effective_final_hit(self):
        counts = {'Hydralisk': 1, 'Zergling': 1}
        assigned = infer_packets(15, counts, 'Hydralisk', killer_type='Zergling')
        self.assertEqual(assigned['Zergling'], 3)
        self.assertAlmostEqual(assigned['Hydralisk'], 12 * 15 / 15.41)
        self.assertLessEqual(sum(assigned.values()), 15)
        # An earlier full 12-damage hit may have left a sliver through regen.
        # This permits a real last hit with no strictly positive lower bound.
        self.assertNotIn('Zergling', infer_packets(12, counts, 'Hydralisk', 'Zergling'))

    def test_infeasible_counts_or_loss_leave_damage_unassigned(self):
        self.assertEqual(infer_packets(17, {'Hydralisk': 1}, 'Hydralisk'), {})
        self.assertEqual(infer_packets(11.4, {'Hydralisk': 1}, 'Hydralisk'), {})
        self.assertEqual(infer_packets(5, {'Hydralisk': 1}, 'Hydralisk', 'Zergling'), {})
        self.assertEqual(infer_packets(None, {'Hydralisk': 1}, 'Hydralisk'), {})
