"""Thin SMACv2 integration; leave upstream dynamics, masks and rewards intact."""
import copy
from collections import Counter
import hashlib

import numpy as np

from .config import MAP_FILE, RACE_UNITS, race_for_map, source_path, validate_config
from .scenarios import reset_config
from .keep_order import raw_keep_heal_target

source_path()
from smacv2.env.starcraft2.starcraft2 import StarCraft2Env
from smacv2.env.starcraft2.wrapper import StarCraftCapabilityEnvWrapper
from smacv2.env.starcraft2.maps.smac_maps import SMACMap
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
        # Load the sibling map directly, without copying into the SC2 install.
        SMACMap.directory = str(MAP_FILE.parent.resolve())
        super()._launch()
        self._controller = RecordingController(self._controller, self)

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
