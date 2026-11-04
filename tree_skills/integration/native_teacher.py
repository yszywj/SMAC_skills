"""Frozen actor diagnostic: native masked argmax, without healing rule gates.

The medivac arm replaces only medivac actions after the rule team proposes
actions. The all arm never invokes the rule controller. Both advance exactly
the same observation/history service used by the conditional scoring arm.
"""
import hashlib
import math
from pathlib import Path

from .pair_runtime import env_dims
from ..teacher.action_mapping import SmacActionMap
from ..teacher.inference import TeacherServiceClient


def validate_control(args, method, config):
    mode = getattr(args, "native_teacher_control", "off")
    if mode == "off":
        return
    if (method != "adaptive-skill" or getattr(args, "race_profile", None) != "terran"
            or getattr(args, "skill_version", None) != "current"
            or config.get("map_name") != "10gen_terran"
            or config.get("unit_balance") != "native"
            or config.get("obs_last_action") or config.get("conic_fov")):
        raise ValueError("Native teacher diagnostics require current Terran/native standard observations")
    if getattr(args, "pair_mode", "off") != "off":
        raise ValueError("Native teacher control and pair scoring are mutually exclusive")
    if not getattr(args, "native_teacher_checkpoint", None):
        raise ValueError("Native teacher control requires an explicit checkpoint")


def masked_native_argmax(logits, avail, native_n_actions):
    """Exclude the keep extension and cross-environment padding; ties use ID."""
    if len(logits) < native_n_actions or len(avail) < native_n_actions:
        raise ValueError("Truncated teacher head or native availability mask")
    if not all(math.isfinite(float(x)) for x in logits):
        raise ValueError("Non-finite teacher logits")
    legal = [a for a in range(native_n_actions) if avail[a]]
    if not legal:
        raise ValueError("No legal native action; refusing a policy fallback")
    return max(legal, key=lambda a: (float(logits[a]), -a))


class NativeTeacherRuntime:
    def __init__(self, control, checkpoint, last_token="on", client=None):
        if control not in ("medivac", "all"):
            raise ValueError(control)
        self.control = control
        self.checkpoint = str(checkpoint)
        self.client = client if client is not None else TeacherServiceClient(
            checkpoint, last_token=last_token)
        self._dims = None
        self._episode = None
        self._key = None
        self.steps = 0

    def choose(self, episode, game_loop, env, rule_actions=None):
        key = (episode, game_loop)
        if self._key == key:
            raise ValueError("Duplicate teacher decision boundary")
        dims = env_dims(env)
        if self._dims != dims:
            self.client.init(dims)
            self._dims = dims
        if self._episode != episode:
            self.client.reset_episode(episode)
            self._episode = episode
        mapping = SmacActionMap(env.n_agents, env.n_enemies,
                                teacher_action_size=self.client.model_args["action_size"])
        agents = [{"agent": a, "obs": [float(x) for x in env.get_obs_agent(a)],
                   "avail": [int(x) for x in env.get_avail_agent_actions(a)]}
                  for a in range(env.n_agents)]
        logits = self.client.logits(episode, game_loop, agents)
        if set(logits) != set(range(env.n_agents)):
            raise ValueError("Teacher response does not cover the full team")
        if self.control == "medivac":
            if rule_actions is None or len(rule_actions) != env.n_agents:
                raise ValueError("Medivac control requires complete rule actions")
            actions = list(rule_actions)
        else:
            if rule_actions is not None:
                raise ValueError("All-teacher control must bypass the rule controller")
            actions = [None] * env.n_agents
        decisions = []
        for row in agents:
            a = row["agent"]
            controlled = self.control == "all" or env.agents[a].unit_type == env.medivac_id
            selected = masked_native_argmax(logits[a], row["avail"], mapping.native_n_actions)
            if controlled:
                actions[a] = selected
            decisions.append({"agent": a, "controlled": controlled,
                              "unit_type": int(env.agents[a].unit_type),
                              "alive": bool(env.agents[a].health > 0),
                              "native_avail": row["avail"][:mapping.native_n_actions],
                              "logits": logits[a], "teacher_action": selected,
                              "rule_action": rule_actions[a] if rule_actions is not None else None,
                              "final_action": actions[a]})
        self._key = key
        self.steps += 1
        return actions, {"control": self.control, "selection": "native_masked_argmax",
                         "decisions": decisions}

    def describe(self):
        base = Path(__file__).resolve().parents[1]
        files = ("integration/native_teacher.py", "integration/pair_runtime.py",
                 "teacher/inference.py", "teacher/service.py", "teacher/obs_builder.py")
        return {"control": self.control, "checkpoint": self.checkpoint,
                "checkpoint_sha256": self.client.model_sha256,
                "env_indicator": self.client.env_indicator,
                "effective_history_len": self.client.history_len,
                "last_token": self.client.last_token,
                "selection": "native_masked_argmax", "failure_policy": "strict",
                "rule_controller_executed": self.control != "all",
                "keep_extension_allowed_for_teacher": False,
                "teacher_steps": self.steps,
                "source_sha256": {name: hashlib.sha256((base / name).read_bytes()).hexdigest()
                                  for name in files}}

    def close(self):
        self.client.close()
