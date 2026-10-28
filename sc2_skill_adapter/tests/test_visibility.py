"""Exercise the actual SMACv2 observation method without launching SC2."""
import math
from types import SimpleNamespace as NS
import unittest

import numpy as np
from bridge.environment import StarCraft2Env, visible_enemies


def unit(x, y, hp):
    return NS(pos=NS(x=x, y=y), health=hp, health_max=100, shield=0)


class ObservationFixture:
    obs_starcraft = True
    obs_pathing_grid = obs_terrain_height = conic_fov = mask_enemies = False
    obs_all_health = obs_own_health = True
    obs_own_pos = obs_timestep_number = obs_last_action = debug = False
    stochastic_health = stochastic_attack = False
    shield_bits_enemy = shield_bits_ally = unit_type_bits = 0
    n_agents, n_enemies, n_actions_move, n_actions_no_attack = 1, 2, 4, 6
    prob_obs_enemy = 1.0
    distance = staticmethod(lambda x, y, ex, ey: math.hypot(x-ex, y-ey))
    get_obs_move_feats_size = lambda self: 4
    get_obs_enemy_feats_size = lambda self: (2, 5)
    get_obs_ally_feats_size = lambda self: (0, 5)
    get_obs_own_feats_size = lambda self: 1
    unit_sight_range = lambda self, actor: 9
    get_unit_by_id = lambda self, actor: self.agents[actor]
    get_avail_agent_actions = get_true_avail_agent_actions = lambda self, actor: [0, 1, 1, 1, 1, 1, 1, 0]
    get_obs_agent = StarCraft2Env.get_obs_agent

    def __init__(self, second_distance=9.7, second_hp=0):
        self.agents = {0: unit(0, 0, 100)}
        self.enemies = {0: unit(4, 0, 50), 1: unit(0, second_distance, second_hp)}
        self.enemy_tags = [0, None]
        self.obs_enemies = np.array([[1.], [0.]])

    def rows(self, fully_observable=False):
        return self.get_obs_agent(0, fully_observable)[4:14].reshape(2, 5)


class VisibilityRegressionTests(unittest.TestCase):
    def test_dead_enemy_never_erases_live_enemy_or_registers_as_observer(self):
        for distance in (9.7, 8.7):
            env = ObservationFixture(distance)
            for _ in range(2):
                rows = env.rows()
                self.assertTrue(np.any(rows[0]))
                self.assertFalse(np.any(rows[1]))
                self.assertIsNone(env.enemy_tags[1])
                self.assertEqual(set(visible_enemies(env)), {0})

    def test_unseen_live_enemy_never_erases_seen_enemy(self):
        env = ObservationFixture(second_distance=12, second_hp=100)
        self.assertTrue(np.any(env.rows()[0]))
        self.assertFalse(np.any(env.rows()[1]))

    def test_epo_mask_keeps_only_its_own_row_hidden(self):
        env = ObservationFixture(second_distance=5, second_hp=100)
        env.enemy_tags[1] = 1  # Previously assigned to another observer.
        env.prob_obs_enemy = 0
        self.assertTrue(np.any(env.rows()[0]))
        self.assertFalse(np.any(env.rows()[1]))

    def test_fully_observable_still_excludes_dead_units(self):
        env = ObservationFixture(second_distance=12, second_hp=100)
        self.assertTrue(np.all(np.any(env.rows(True), axis=1)))
        env.enemies[1].health = 0
        self.assertFalse(np.any(env.rows(True)[1]))


if __name__ == "__main__":
    unittest.main()
