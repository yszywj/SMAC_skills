"""P0-2: raw per-agent SMACv2 observations -> MARL-GPT model input.

Runs ONLY in the teacher environment (imports the official archived
implementation under tree_skills/external/marl-gpt-main). Pipeline mirrors
the official InferenceSmacWrapper + InferenceHistoryWrapper flow:

  raw obs vector (env.get_obs_agent(a))
    -> decompose_obs slicing (move / enemy / ally / own)      [decomposer.py]
    -> per-(episode, agent) history deque, zero front-pad,
       latest frame first                                    [general_envs_wrapeers.py,
                                                                env_observation.stack_dict_tensor]
    -> smacTokenizer.get_obs_pos_vectors (official class, terran attr map)
    -> append_agent_race (terran code 0, attr 16)            [wrapper.py]
    -> reshape_obs pad to checkpoint block_size (value 0)    [env_observation.py]
    -> optional last_token (attr n_attr-1, env indicator 1)  [inference.py]

Differences from the official path are deliberate and recorded:
- history_len: the checkpoint's model_args carry history_len=10, but that is
  the TIME-EMBEDDING TABLE capacity, not the trained sequence length. With
  terran 10v10 token layout (162 tokens/frame) the checkpoint block_size
  (700) only fits 4 frames + race token (4*162+1=649 <= 700); 10 frames
  would need 1621 positions. The official SMAC configs also use 4. The
  builder therefore defaults to history_len=4 and REJECTS configurations
  that would exceed block_size instead of truncating; the chosen value is
  recorded in INFERENCE_VALIDATION.md and must match across A/B/C/D runs;
- history is maintained per (episode_id, agent_id) server-side; a repeated
  request for the same (episode, agent, game_loop) reuses the existing
  history instead of appending again;
- feature sizes come from OUR live environment instead of instantiating a
  second StarCraftCapabilityEnvWrapper.
"""

import os
import sys

import numpy as np
import torch

MARL_GPT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                        "external", "marl-gpt-main"))
if MARL_GPT not in sys.path:
    sys.path.insert(0, MARL_GPT)

from envs.smac_env.obs_tokenizer import smacTokenizer  # noqa: E402
from utils.env_observation import reshape_obs  # noqa: E402

RACE_CODE_TERRAN = 0
RACE_POS_CODE = 16


class ObsBuildError(ValueError):
    pass


class SmacObsBuilder:
    def __init__(self, *, n_agents, n_enemies, move_feats, own_feats,
                 obs_nf_en, obs_nf_al, history_len, block_size, vocab_size,
                 action_size, obs_with_race=True, last_token=False,
                 env_indicator=1, n_attr=129):
        self.n_agents, self.n_enemies = n_agents, n_enemies
        self.move_feats, self.own_feats = move_feats, own_feats
        self.obs_nf_en, self.obs_nf_al = obs_nf_en, obs_nf_al
        self.history_len = history_len
        self.block_size, self.vocab_size = block_size, vocab_size
        self.action_size = action_size
        self.obs_with_race = obs_with_race
        self.last_token = last_token
        self.env_indicator = env_indicator
        self.n_attr = n_attr
        # smacTokenizer("terran") reads a CWD-relative path; load the attr
        # map from the archive explicitly so the service is CWD-independent
        import json as _json
        with open(os.path.join(MARL_GPT, "envs", "smac_env",
                               "attr_for_all_map.json"), encoding="utf-8") as fh:
            attr_map = _json.load(fh)
        self.tokenizer = smacTokenizer("terran", pos_dict=attr_map["terran"])
        # (episode_id, agent_id) -> {"loop": int, "frames": [obs vectors]}
        self._history = {}
        per_frame = (move_feats + own_feats) + (n_agents - 1) * obs_nf_al \
            + n_enemies * obs_nf_en + (1 if obs_with_race else 0)
        self.per_frame_tokens = per_frame
        needed = per_frame * history_len + (1 if last_token else 0)
        if needed > block_size:
            raise ObsBuildError(
                f"tokenized input needs {needed} positions but the checkpoint "
                f"block_size is {block_size}; refusing to truncate")

    # -- decomposition (official SC2Decomposer.decompose_obs slicing) ------
    def decompose(self, obs_vector):
        obs = np.asarray(obs_vector, dtype=np.float32)
        expected = (self.move_feats + self.n_enemies * self.obs_nf_en
                    + (self.n_agents - 1) * self.obs_nf_al + self.own_feats)
        if obs.shape[-1] != expected:
            raise ObsBuildError(
                f"obs vector has {obs.shape[-1]} features, layout expects "
                f"{expected} (move {self.move_feats} + enemy "
                f"{self.n_enemies}x{self.obs_nf_en} + ally "
                f"{self.n_agents - 1}x{self.obs_nf_al} + own {self.own_feats})")
        base = self.move_feats
        enemy = obs[base:base + self.n_enemies * self.obs_nf_en].reshape(
            self.n_enemies, self.obs_nf_en)
        base += self.n_enemies * self.obs_nf_en
        ally = obs[base:base + (self.n_agents - 1) * self.obs_nf_al].reshape(
            self.n_agents - 1, self.obs_nf_al)
        base += (self.n_agents - 1) * self.obs_nf_al
        own = np.concatenate([obs[:self.move_feats], obs[base:base + self.own_feats]])
        return own, ally, enemy

    # -- history management -----------------------------------------------
    def reset_episode(self, episode_id):
        for key in [k for k in self._history if k[0] == episode_id]:
            del self._history[key]

    def observe(self, episode_id, agent_id, game_loop, obs_vector):
        """Advance one agent's history by one real environment step.

        Idempotent per (episode, agent, game_loop): re-sending the same
        loop reuses the recorded frame and does NOT append again.
        """
        key = (episode_id, agent_id)
        record = self._history.get(key)
        if record is None:
            record = {"loop": None, "frames": []}
            self._history[key] = record
        if record["loop"] == game_loop:
            return  # duplicate request for the same step
        record["loop"] = game_loop
        record["frames"].append(np.asarray(obs_vector, dtype=np.float32))
        record["frames"] = record["frames"][-self.history_len:]

    def build_input(self, episode_id, agent_id):
        """Tokenized, padded model input dict for one agent (batch 1)."""
        record = self._history[(episode_id, agent_id)]
        frames = record["frames"]
        # official stack_dict_tensor: zero FRONT-pad to history_len, then
        # flip so the newest frame is first
        padded = [np.zeros_like(frames[0])] * (self.history_len - len(frames)) + frames
        stacked = np.stack(padded[::-1])  # newest first, shape (H, obs_dim)

        per_agent = []
        for frame in stacked:  # newest -> oldest
            own, ally, enemy = self.decompose(frame)
            per_agent.append({"own": own, "ally": ally, "enemy": enemy})

        def collect(kind):
            return torch.tensor(np.stack([p[kind] for p in per_agent]),
                                dtype=torch.float32).unsqueeze(0)

        # official tokenizer with history dim: obs_own (1, H, f),
        # obs_ally (1, 1, H, n-1, f), obs_enemy (1, 1, H, n, f)
        observation = {
            "obs_own": collect("own"),
            "obs_ally": collect("ally").unsqueeze(0),
            "obs_enemy": collect("enemy").unsqueeze(0),
        }
        obs, group_pos, agent_pos, time_pos, attr_pos = \
            self.tokenizer.get_obs_pos_vectors(observation)
        result = {"obs": obs, "group_pos": group_pos, "agent_pos": agent_pos,
                  "time_pos": time_pos, "attr_pos": attr_pos}
        if self.obs_with_race:
            # official append_agent_race (batch of 1)
            result["obs"] = torch.cat(
                [result["obs"], torch.full((1, 1), float(RACE_CODE_TERRAN))], dim=-1)
            for key in ("group_pos", "agent_pos", "time_pos", "attr_pos"):
                result[key] = torch.cat(
                    [result[key], torch.zeros(1, dtype=torch.int)], dim=-1)
            result["attr_pos"][..., -1] = RACE_POS_CODE
        result = {k: (v if v.dim() >= 2 else v.unsqueeze(0))
                  for k, v in result.items()}
        result = reshape_obs(result, self.block_size, self.action_size)
        result["obs"] = result["obs"].float()
        if self.last_token:
            result["attr_pos"][..., -1] = self.n_attr - 1
            result["obs"][..., -1] = float(self.env_indicator)
        self._validate(result)
        return result

    def _validate(self, result):
        length = result["obs"].shape[-1]
        if length > self.block_size:
            raise ObsBuildError(f"input length {length} > block_size {self.block_size}")
        for key in ("group_pos", "agent_pos", "time_pos"):
            if int(result[key].max()) >= {"group_pos": 2, "agent_pos": 20,
                                          "time_pos": self.history_len}[key]:
                raise ObsBuildError(f"{key} index out of embedding range")
        if int(result["attr_pos"].max()) >= self.n_attr:
            raise ObsBuildError("attr_pos index out of embedding range")
