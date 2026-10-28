"""Native protocol facts survive the SMACv2 policy adapter."""
import unittest
from s2clientprotocol import data_pb2, raw_pb2
from bridge.policy import combat_profile, unit_record


class AccurateDamageProfileTests(unittest.TestCase):
    def test_raw_catalog_bonus_attributes_and_explicit_shield_prior(self):
        weapon = data_pb2.Weapon(type=data_pb2.Weapon.Any, range=6, damage=13, attacks=1,
                                damage_bonus=[data_pb2.DamageBonus(attribute=data_pb2.Armored, bonus=5)])
        stat = data_pb2.UnitTypeData(armor=1, attributes=[data_pb2.Armored], weapons=[weapon])
        profile = combat_profile(stat, 74)
        self.assertEqual(profile["attributes"], [2])
        self.assertEqual(profile["weapons"][0]["damage_bonus"], [{"attribute": 2, "bonus": 5}])
        self.assertEqual(profile["shield_armor_base"], 0)
        self.assertEqual(profile["shield_armor_source"], "native_protoss_zero_upgrade_prior")
        self.assertIsNone(combat_profile(stat, 999)["shield_armor_base"])

    def test_raw_unit_upgrades_are_preserved_on_both_sides(self):
        unit = raw_pb2.Unit(unit_type=74, health=80, health_max=80, shield=80, shield_max=80,
                            attack_upgrade_level=0, armor_upgrade_level=0, shield_upgrade_level=0)
        for own in (False, True):
            row = unit_record(unit, "Stalker", own=own)
            self.assertEqual([row[field] for field in ("attack_upgrade_level", "armor_upgrade_level", "shield_upgrade_level")], [0, 0, 0])

    def test_absent_proto_scalars_are_unknown_not_default_zero(self):
        unit = raw_pb2.Unit(unit_type=74, health=80, health_max=80)
        row = unit_record(unit, "Stalker")
        for field in ("attack_upgrade_level", "armor_upgrade_level", "shield_upgrade_level"):
            self.assertIsNone(row[field])
        profile = combat_profile(data_pb2.UnitTypeData(weapons=[data_pb2.Weapon(type=3, range=6)]), 74)
        self.assertIsNone(profile["armor"])
        self.assertIsNone(profile["weapons"][0]["base_damage"])
        self.assertIsNone(profile["weapons"][0]["hits_per_attack"])


if __name__ == "__main__":
    unittest.main()
