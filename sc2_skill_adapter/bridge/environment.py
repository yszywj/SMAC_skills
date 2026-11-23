"""Thin SMACv2 integration; leave upstream dynamics, masks and rewards intact."""
import copy
from collections import Counter
import hashlib
import os
from unittest.mock import patch

import numpy as np

from .config import map_file, RACE_UNITS, race_for_map, source_path, validate_config
from .scenarios import reset_config
from .keep_order import raw_keep_heal_target

source_path()
from smacv2.env.starcraft2.starcraft2 import StarCraft2Env
from smacv2.env.starcraft2.wrapper import StarCraftCapabilityEnvWrapper
from smacv2.env.starcraft2.maps import smac_maps
from pysc2.lib.units import Protoss


def is_medivac(unit, data):
    name = getattr(data, "units", {}).get(unit.unit_type, "")
    return unit.unit_type == 54 or name.lower().startswith("medivac")


def visible_enemies(env):
    """Use the actual enemy feature block, including all observation gating.

    Raw SC2 observations and env.enemies may include units outside the SMACv2
    per-unit sight radius. Attack masks are not a visibility mask either.
    """
    offset = env.get_obs_move_feats_size()
    count, width = env.get_obs_enemy_feats_size()
    visible = set()
    for agent_id, unit in env.agents.items():
        if unit.health <= 0:
            continue
        obs = np.asarray(env.get_obs_agent(agent_id))
        enemies = obs[offset:offset + count * width].reshape(count, width)
        visible.update(int(i) for i in np.flatnonzero(np.any(enemies != 0, axis=1)))
    return {i: env.enemies[i] for i in sorted(visible) if env.enemies[i].health > 0}


class RecordingController:
    def __init__(self, controller, env):
        self.controller, self.env = controller, env

    def __getattr__(self, name):
        return getattr(self.controller, name)

    def actions(self, request):
        result = self.controller.actions(request)
        if self.env.collecting:
            self.env.last_action_results.append(result)
        return result


class RecordedEnv(StarCraft2Env):
    @property
    def game(self):
        return self._join_request

    @game.setter
    def game(self, request):
        # SMACv2 assigns self.game immediately before controller.join_game.
        if request is not None:
            request.options.score = True
        self._join_request = request

    def __init__(self, **kwargs):
        self.unit_balance = kwargs.pop("unit_balance", "original")
        self.terran_keep_heal_order = kwargs.pop("terran_keep_heal_order", False)
        if self.terran_keep_heal_order and (self.unit_balance != 'native'
                or kwargs.get('map_name') != '10gen_terran' or kwargs.get('heuristic_ai', False)
                or kwargs.get('conic_fov', False)):
            raise ValueError('KeepHealOrder requires native standard Terran control')
        self.collecting = False
        self.last_submissions = []
        self.last_action_results = []
        super().__init__(**kwargs)
        if self.terran_keep_heal_order:
            self.keep_heal_order_action_id = self.n_actions
            self.n_actions += 1
            # Upstream reset/step and dimension methods use self.n_actions.
            # Refresh the two arrays/names created during the base constructor.
            self.last_action = np.zeros((self.n_agents, self.n_actions))
            self.state_feature_names = self.build_state_feature_names()
            self.obs_feature_names = self.build_obs_feature_names()

    def keep_heal_order_target(self, agent_id):
        return raw_keep_heal_target(self.agents, agent_id) if self.terran_keep_heal_order else None

    def get_avail_agent_actions(self, agent_id):
        available = super().get_avail_agent_actions(agent_id)
        if getattr(self, 'terran_keep_heal_order', False):
            available[self.keep_heal_order_action_id] = int(self.keep_heal_order_target(agent_id) is not None)
        return available

    def _init_ally_unit_types(self, min_unit_type):
        super()._init_ally_unit_types(min_unit_type)
        if self.unit_balance == "native":
            # One SC2 catalog definition per type for BOTH owners. This also
            # intentionally restores normal automatic acquisition on allies.
            race = race_for_map(getattr(self, "map_name", "10gen_protoss"))
            for name, native_id in RACE_UNITS[race].items():
                setattr(self, name + "_id", native_id)
                self.ally_unit_map[name] = native_id
                self.enemy_unit_map[name] = native_id

    def _launch(self):
        path = map_file({'map_name': self.map_name}).resolve(strict=True)
        # Each logical 10gen class normally points at the same 32x32_flat filename.
        with patch.multiple(getattr(smac_maps, self.map_name), directory=str(path.parent), filename=path.stem):
            super()._launch()
        self._controller = RecordingController(self._controller, self)

    def init_units(self, ally_team, enemy_team, episode_config):
        if os.environ.get('SMAC_DAMAGE_MAP_DIR') and any(
                u.owner in (1, 2) for u in self._obs.observation.raw_data.units):
            raise RuntimeError('Damage map has preplaced combatants; expected instrumented 32x32_flat')
        return super().init_units(ally_team, enemy_team, episode_config)

    def get_agent_action(self, a_id, action):
        if (getattr(self, 'terran_keep_heal_order', False)
                and action == self.keep_heal_order_action_id):
            if not self.get_avail_agent_actions(a_id)[action]:
                raise ValueError(f'u{a_id}: no eligible observed healing order to retain')
            if self.collecting:
                self.last_submissions.append({
                    'actor': f'u{a_id}', 'action_id': action, 'command': None,
                    'submission': 'retained_order'})
            return None
        command = super().get_agent_action(a_id, action)
        if self.collecting:
            self.last_submissions.append({
                "actor": f"u{a_id}", "action_id": action, "command": command,
                "submission": "submitted" if command is not None else "dead_noop"})
        return command

    def step(self, actions):
        self.last_submissions, self.last_action_results = [], []
        before_restarts = self.force_restarts
        self.collecting = True
        try:
            result = super().step(actions)
        finally:
            self.collecting = False
        if self.force_restarts != before_restarts or not self.last_action_results:
            raise RuntimeError("SC2 transport/restart failure; episode is invalid, not a loss.")
        return result

    def submit_micro_commands(self, request):
        """Submit authorized micro commands WITHOUT stepping.

        Sent outside the collection window, so they never enter
        last_submissions/last_action_results and can never be mistaken for
        the macro batch by action_feedback.
        """
        return self._controller.actions(request)

    def step_micro_window(self, actions, hook, macro_loops):
        """Split only the engine clock. Native step owns actions and settlement."""
        if self.stochastic_health or self.heuristic_ai:
            raise RuntimeError("micro execution requires standard health and policy actions")
        if macro_loops != self._step_mul or macro_loops < 1:
            raise RuntimeError(
                f"micro window of {macro_loops} loops requires step_mul "
                f"{macro_loops}, configured {self._step_mul}")

        controller = self._controller
        step, observe = controller.step, controller.observe
        start = self._obs.observation.game_loop
        interval_errors = []

        def frame(offset):
            obs = observe()
            if obs.observation.game_loop != start + offset:
                raise RuntimeError(f"Expected SC2 loop {start + offset}, got {obs.observation.game_loop}")
            return obs

        def split_step(count):
            if count != macro_loops:
                raise RuntimeError(f"Unexpected native step size: {count}")
            # update_units mutates missing units in place and overwrites previous_*.
            # Use private rosters throughout the interior; never call it on macro state.
            allies, enemies = copy.deepcopy(self.agents), copy.deepcopy(self.enemies)
            rng = np.random.get_state()  # get_obs_agent draws even at prob_obs_enemy=1
            try:
                with patch.multiple(self, collecting=False, agents=allies, enemies=enemies,
                                    enemy_tags=self.enemy_tags.copy(), obs_enemies=self.obs_enemies.copy(),
                                    _obs=self._obs):
                    for offset in range(1, count):
                        step(1)
                        self._obs = obs = frame(offset)
                        if obs.player_result:
                            raise RuntimeError("SC2 game ended before the SMAC macro boundary")
                        interval_errors.extend(obs.action_errors)
                        by_tag = {u.tag: u for u in obs.observation.raw_data.units}
                        for roster in (allies, enemies):
                            for slot, old in roster.items():
                                if old.tag in by_tag:
                                    roster[slot] = by_tag[old.tag]
                                else:
                                    # Do not mutate a protobuf owned by an earlier raw frame.
                                    dead = copy.deepcopy(old)
                                    dead.health = 0
                                    roster[slot] = dead
                        hook.on_interior_loop(offset)
            finally:
                np.random.set_state(rng)
            step(1)  # closing observation is consumed by native step, not by the hook

        def closing_observation():
            obs = frame(macro_loops)
            if interval_errors:
                # feedback.py expects errors for the whole interval, not only the last loop.
                obs = copy.deepcopy(obs)
                errors = interval_errors + list(obs.action_errors)
                del obs.action_errors[:]
                obs.action_errors.extend(errors)
            return obs

        # Patch this controller instance only. If native step restarts SC2, never
        # put the old controller back onto the newly launched environment.
        with patch.multiple(controller, step=split_step, observe=closing_observation):
            reward, terminated, info = self.step(actions)
        info["micro_advanced_loops"] = macro_loops
        return reward, terminated, info


def audit_unit_balance(env, static_data):
    """Full-roster audit only; caller must never put this into model input."""
    race = race_for_map(getattr(env, "map_name", "10gen_protoss"))
    expected_ids = set(RACE_UNITS[race].values())
    rosters = {side: Counter(u.unit_type for u in units.values())
               for side, units in (("ally", env.agents), ("enemy", env.enemies))}
    if rosters["ally"] != rosters["enemy"] or not set(rosters["ally"]) <= expected_ids:
        raise RuntimeError("Native balance audit failed: type/count mismatch")
    fields = ("health_max", "shield_max", "energy_max", "radius",
              "attack_upgrade_level", "armor_upgrade_level", "shield_upgrade_level")
    profiles = {}
    for unit_type, count in sorted(rosters["ally"].items()):
        groups = {
            side: [u for u in units.values() if u.unit_type == unit_type]
            for side, units in (("ally", env.agents), ("enemy", env.enemies))
        }
        stats = {side: {tuple(getattr(u, name) for name in fields) for u in units}
                 for side, units in groups.items()}
        if stats["ally"] != stats["enemy"] or len(stats["ally"]) != 1:
            raise RuntimeError(f"Native balance audit failed: attributes/upgrades differ for {unit_type}")
        if any(tuple(u.buff_ids) for units in groups.values() for u in units):
            raise RuntimeError("Unexpected initial buffs; cannot certify equal combat attributes")
        data = static_data.unit_stats[unit_type]
        profiles[data.name] = {
            "unit_type_id": unit_type, "count_per_side": count,
            "shared_attributes": dict(zip(fields, next(iter(stats["ally"])))),
            "shared_catalog_sha256": hashlib.sha256(data.SerializeToString()).hexdigest(),
            "movement_speed": data.movement_speed, "armor": data.armor,
            "weapons": [
                {"type": w.type, "range": w.range, "damage": w.damage,
                 "attacks": w.attacks, "speed": w.speed,
                 "damage_bonus": [{"attribute": b.attribute, "bonus": b.bonus} for b in w.damage_bonus]}
                for w in data.weapons],
            "initial_health_shields": {
                side: [[u.health, u.shield] for u in units] for side, units in groups.items()},
        }
    return {"passed": True, "mode": "native", "profiles": profiles,
            "scope": "Same unit catalog IDs and maximum attributes/upgrades; no initial buffs. "
                     "Does not equate AI policy, observation or command cadence."}


class ReplayableEnv(StarCraftCapabilityEnvWrapper):
    def __init__(self, args, *, terran_keep_heal_order=False):
        validate_config(args)
        kwargs = copy.deepcopy(args)
        self.distribution_config = kwargs["capability_config"]
        self.env_key_to_distribution_map = {}
        self._parse_distribution_config()
        self.env = RecordedEnv(**kwargs, **({'terran_keep_heal_order': True} if terran_keep_heal_order else {}))

    def reset_record(self, record):
        # Sampled once by the manifest builder. Do not resample on failed resets.
        before_restarts = self.env.force_restarts
        obs, _state = self.env.reset(reset_config(record))
        if self.env.force_restarts != before_restarts:
            raise RuntimeError("Reset restarted SC2; refuse to silently replace a paired episode")
        if len(self.env.agents) != self.env.n_agents or len(self.env.enemies) != self.env.n_enemies:
            raise RuntimeError("Reset produced an incomplete roster")
        return obs
