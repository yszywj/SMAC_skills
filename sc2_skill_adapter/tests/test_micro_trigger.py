"""Offline micro windows: real SMAC masks/command builder, no controller or game."""
import math
from types import SimpleNamespace as NS
import unittest

from s2clientprotocol import data_pb2, raw_pb2, sc2api_pb2 as sc_pb

from bridge.environment import StarCraft2Env
from bridge.micro_trigger import (ACTOR_DEAD, ALREADY_ORDERED, LEGAL_CONTROL_OFF,
                                  MicroAttackWindow, NEVER_LEGAL, SUBMITTED, TARGET_LOST)


ACTOR, TARGET = 2**63 + 17, 2**63 + 19
INTENT = dict(kind='approach_then_attack', actor='u0', actor_tag=str(ACTOR),
              target='e1', target_tag=str(TARGET), issued_loop=20)


class WindowEnv:
    n_agents, n_enemies, n_actions, n_actions_no_attack = 1, 2, 8, 6
    hydralisk_id, map_type = 107, 'zerg_gen'
    conic_fov = action_mask = stochastic_attack = debug = False
    distance = staticmethod(lambda x, y, ex, ey: math.hypot(x-ex, y-ey))
    can_move = lambda self, unit, direction: True
    unit_shoot_range = lambda self, i: 5
    get_unit_by_id = lambda self, i: self.agents[i]
    get_obs_move_feats_size = lambda self: 0
    get_obs_enemy_feats_size = lambda self: (2, 1)
    get_obs_agent = lambda self, i: [1, int(self.team_sees_target)]
    get_true_avail_agent_actions = StarCraft2Env.get_true_avail_agent_actions
    get_avail_agent_actions = StarCraft2Env.get_avail_agent_actions
    get_can_shoot = StarCraft2Env.get_can_shoot
    get_agent_action = StarCraft2Env.get_agent_action

    def __init__(self):
        self._obs = NS(observation=NS(game_loop=20))
        def unit(tag, x):
            u = raw_pb2.Unit(tag=tag, unit_type=107, health=90, display_type=raw_pb2.Visible)
            u.pos.x = x
            return u
        self.agents = {0: unit(ACTOR, 0)}
        self.enemies = {0: unit(TARGET - 1, 1), 1: unit(TARGET, 5.01)}
        self.team_sees_target = True
        self.sent = []

    def submit_micro_commands(self, request):
        self.sent.append(request)
        return sc_pb.ResponseAction(result=[1])

    def window(self, execute=True, intents=None, **kwargs):
        return MicroAttackWindow(self, [INTENT] if intents is None else intents, 1, 20, 8, execute, **kwargs)

    def check(self, window, delta, offset=None):
        self._obs.observation.game_loop = 20 + delta
        window.on_interior_loop(delta if offset is None else offset)


class MicroWindowTests(unittest.TestCase):
    def test_off_on_first_true_legal_and_one_exact_command(self):
        for execute in (False, True):
            env = WindowEnv()
            w = env.window(execute)
            self.assertEqual(env.get_avail_agent_actions(0)[7], 1)       # unmasked
            self.assertEqual(env.get_true_avail_agent_actions(0)[7], 0)  # out of range
            # A closer, always legal enemy is not a replacement target.
            for delta in range(1, 7):
                env.check(w, delta)
            env.enemies[1].pos.x = 5
            env.agents[0].weapon_cooldown = 10
            for delta in (7, 7, 8):
                env.check(w, delta)
            r, = w.close(28)
            self.assertEqual((r['checked_loops'], r['first_legal_loop']), (7, 27))
            self.assertEqual(r['status'], SUBMITTED if execute else LEGAL_CONTROL_OFF)
            self.assertEqual(r['submitted_loop'], 27 if execute else None)
            self.assertEqual(len(env.sent), int(execute))
            if execute:
                req, = env.sent
                self.assertIsInstance(req, sc_pb.RequestAction)
                cmd, = req.actions
                order = cmd.action_raw.unit_command
                self.assertEqual((list(order.unit_tags), order.target_unit_tag, order.ability_id,
                                  order.queue_command), ([ACTOR], TARGET, 23, False))

    def test_clock_and_terminal_close_never_send(self):
        for early in (True, False):
            env = WindowEnv()
            env.enemies[1].pos.x = 4
            w = env.window()
            for delta, offset in ((-1, 1), (0, 0), (0, 1), (1, 2)):
                env.check(w, delta, offset)
            if early:
                w.close(21)  # battle ended before next interior callback
            else:
                env.check(w, 8, 7)  # stale offset cannot extend the window
            env.check(w, 2)
            self.assertEqual(w.close(28)[0]['status'], NEVER_LEGAL)
            self.assertIsNone(w.records[0]['first_legal_loop'])
            self.assertEqual(env.sent, [])

    def test_loss_of_identity_or_visibility_revokes_even_if_restored(self):
        for loss in ('actor_tag', 'actor_dead', 'type', 'target_tag', 'target_dead', 'snapshot', 'team'):
            with self.subTest(loss=loss):
                env = WindowEnv()
                w = env.window()
                a, t = env.agents[0], env.enemies[1]
                t.pos.x = 4
                if loss == 'actor_tag': a.tag += 9
                if loss == 'actor_dead': a.health = 0
                if loss == 'type': a.unit_type = 105
                if loss == 'target_tag': t.tag += 9
                if loss == 'target_dead': t.health = 0
                if loss == 'snapshot': t.display_type = raw_pb2.Snapshot
                if loss == 'team': env.team_sees_target = False
                env.check(w, 1)
                a.tag, t.tag, a.health, t.health, a.unit_type = ACTOR, TARGET, 90, 90, 107
                t.display_type, env.team_sees_target = raw_pb2.Visible, True
                env.check(w, 2)
                self.assertEqual(w.records[0]['status'], ACTOR_DEAD if loss in
                                 ('actor_tag', 'actor_dead', 'type') else TARGET_LOST)
                self.assertEqual(env.sent, [])

    def test_only_effective_unit_target_attack_suppresses_send(self):
        for ability, tag, quiet in ((23, TARGET, True), (3674, TARGET, True), (23, TARGET-1, False),
                                    (16, TARGET, False), (23, None, False)):
            with self.subTest(ability=ability, tag=tag):
                env = WindowEnv()
                env.enemies[1].pos.x = 4
                orders = env.agents[0].orders
                order = orders.add(ability_id=ability)
                if tag is None:
                    order.target_world_space_pos.x = 4
                else:
                    order.target_unit_tag = tag
                orders.add(ability_id=23, target_unit_tag=TARGET)  # queued, not effective
                w = env.window()
                env.check(w, 1)
                self.assertEqual(w.records[0]['status'], ALREADY_ORDERED if quiet else SUBMITTED)
                self.assertEqual(len(env.sent), int(not quiet))

    def test_stale_ambiguous_or_unbound_grants_do_not_send(self):
        env = WindowEnv()
        for changes in ({'issued_loop': 12}, {'actor': 'e0'}, {'target': 'e0'}, {'actor_tag': None},
                        {'target_tag': 'e1'}, {'kind': 'search'}):
            self.assertEqual(env.window(intents=[{**INTENT, **changes}]).records, [])
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            env.window(intents=[INTENT, INTENT])
        self.assertEqual(env.sent, [])


class AllAttackersTests(unittest.TestCase):
    types = {'marine': 48, 'marauder': 51, 'medivac': 54, 'stalker': 74, 'zealot': 73,
             'colossus': 4, 'baneling': 9, 'zergling': 105, 'hydralisk': 107}

    def setup_pair(self, name, weapons=(), flying=False):
        env = WindowEnv()
        for role, type_id in self.types.items():
            setattr(env, role + '_id', type_id)
        actor, target = env.agents[0], env.enemies[1]
        actor.unit_type = self.types[name]
        actor.radius = target.radius = .375
        target.is_flying = flying
        target.pos.x = .75
        stat = data_pb2.UnitTypeData(unit_id=actor.unit_type)
        for kind, reach in weapons:
            stat.weapons.add(type=kind, range=reach)
        data = NS(unit_stats={actor.unit_type: stat})
        return env, data

    def test_eight_attackers_only_and_legacy_isolation(self):
        for name in self.types:
            with self.subTest(name=name):
                weapons = () if name in ('baneling', 'hydralisk') else [(3, 4)]
                env, data = self.setup_pair(name, weapons)
                if name != 'hydralisk':
                    self.assertEqual(env.window(static_data=data).records, [])
                    self.assertEqual(env.window(execute=False, static_data=data).records, [])
                w = env.window(all_attackers=True, static_data=data)
                for delta in range(1, 9):
                    env.check(w, delta)
                if name == 'medivac':  # forbidden even with a spurious catalog weapon
                    self.assertEqual((w.records, env.sent), ([], []))
                    continue
                row, = w.records
                self.assertEqual((row['actor_type'], row['actor_type_id']), (name, self.types[name]))
                self.assertEqual((row['first_legal_loop'], row['submitted_loop'], row['submitted_action_id'],
                                  row['status'], row['sc2_response']), (21, 21, 7, SUBMITTED, [1]))
                request, = env.sent
                cmd, = request.actions
                order = cmd.action_raw.unit_command
                self.assertEqual((list(order.unit_tags), order.target_unit_tag, order.ability_id,
                                  order.queue_command), ([ACTOR], TARGET, 23, False))

    def test_two_open_menus_do_not_interrupt_a_melee_approach_outside_contact(self):
        for name, weapons in [('zergling', [(1, .125)]), ('baneling', [])]:
            with self.subTest(name=name):
                env, data = self.setup_pair(name, weapons)
                env.agents[0].orders.add(ability_id=16).target_world_space_pos.x = 2.
                env.enemies[1].pos.x = 2.
                w = env.window(all_attackers=True, static_data=data)
                self.assertEqual(env.get_true_avail_agent_actions(0)[7], 1)
                self.assertEqual(env.get_avail_agent_actions(0)[7], 1)
                for delta in range(1, 5):
                    env.check(w, delta)
                self.assertIsNone(w.records[0]['first_legal_loop'])
                self.assertEqual(env.sent, [])
                env.enemies[1].pos.x = .75 + (.125 if weapons else 0.)
                env.check(w, 5)
                env.check(w, 6)
                self.assertEqual((w.records[0]['submitted_loop'], len(env.sent)), (25, 1))

    def test_unknown_or_incompatible_profiles_and_unauthorized_intents_do_not_register(self):
        for name, weapons, flying in [('zergling', [(1, 2)], True), ('baneling', [], True),
                                      ('zealot', [], False), ('stalker', [(2, 6)], False)]:
            with self.subTest(name=name):
                env, data = self.setup_pair(name, weapons, flying)
                self.assertEqual(env.window(all_attackers=True, static_data=data).records, [])
        env, data = self.setup_pair('marine', [(3, 5)], flying=True)
        self.assertEqual(env.window(all_attackers=True, static_data=NS(unit_stats={})).records, [])
        self.assertEqual(env.window(all_attackers=True).records, [])
        for kind in ('move', 'retreat', 'rescue'):
            self.assertEqual(env.window(intents=[dict(INTENT, kind=kind)],
                                        all_attackers=True, static_data=data).records, [])
        env.agents[0].unit_type = 12345
        data.unit_stats[12345] = data.unit_stats[48]
        self.assertEqual(env.window(all_attackers=True, static_data=data).records, [])

    def test_recheck_live_geometry_target_plane_and_both_menus(self):
        env, data = self.setup_pair('marauder', [(1, 1.)])
        w = env.window(all_attackers=True, static_data=data)
        target = env.enemies[1]
        target.is_flying = True
        env.check(w, 1)  # target plane no longer compatible
        target.is_flying = False
        target.pos.x = 1.5
        target.radius = .1
        env.check(w, 2)  # 1+.375+.1 < 1.5
        target.radius = .25
        for delta, mask in [(3, 'get_true_avail_agent_actions'), (4, 'get_avail_agent_actions')]:
            setattr(env, mask, lambda _: [0]*8)
            env.check(w, delta)
            delattr(env, mask)
        self.assertIsNone(w.records[0]['first_legal_loop'])
        self.assertEqual(env.sent, [])
        # Same tag is now in a different enemy slot. Record the action actually sent.
        env.enemies[0], env.enemies[1] = env.enemies[1], env.enemies[0]
        env.check(w, 5)
        self.assertEqual((w.records[0]['first_legal_loop'], w.records[0]['submitted_action_id']), (25, 6))
        self.assertEqual(env.sent[0].actions[0].action_raw.unit_command.target_unit_tag, TARGET)

    def test_anti_air_weapon_can_attack_visible_air_and_already_ordered_is_not_a_send(self):
        env, data = self.setup_pair('stalker', [(1, 6), (2, 3)], flying=True)
        env.enemies[1].pos.x = 4  # in ground range but outside air reach of 3+.75
        env.agents[0].orders.add(ability_id=3674, target_unit_tag=TARGET)
        w = env.window(all_attackers=True, static_data=data)
        env.check(w, 1)
        env.enemies[1].pos.x = 3.75
        env.check(w, 2)
        self.assertEqual((w.records[0]['first_legal_loop'], w.records[0]['status'],
                          w.records[0]['submitted_loop'], env.sent), (22, ALREADY_ORDERED, None, []))
