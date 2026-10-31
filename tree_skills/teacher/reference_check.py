"""G2 real-snapshot parity: our obs builder vs an official-code reference.

Teacher-env script. Input: a real collection JSONL (collect_real.py) that
carries, per decision step, every agent's RAW env.get_obs_agent() vector.
For sampled (episode, agent) it replays the recorded observation sequence
through BOTH pipelines:

  ours     : tree_skills.teacher.obs_builder.SmacObsBuilder
  reference: the official functions from the archived implementation —
             utils.env_observation.stack_dict_tensor for history stacking
             (zero front-pad, newest-first), the official smacTokenizer,
             a verbatim port of InferenceSmacWrapper.append_agent_race, and
             the official reshape_obs padding.

Documented limitation: the official SC2Decomposer cannot be instantiated
without spawning its own SC2 env, so the move/enemy/ally/own SLICING of the
raw vector is replicated from decomposer.decompose_obs (six documented
lines) instead of executed; everything downstream of the slicing — the
error-prone part (history, positions, race token, padding) — runs OFFICIAL
code. Elementwise float equality is required.

Usage:
  python tree_skills/teacher/reference_check.py --collect <file.jsonl> \
      [--last-token on|off] [--samples 10]
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
MARL_GPT = os.path.join(REPO, "tree_skills", "external", "marl-gpt-main")
for p in (MARL_GPT, REPO):
    if p not in sys.path:
        sys.path.insert(0, p)

from utils.env_observation import reshape_obs, stack_dict_tensor  # noqa: E402
from envs.smac_env.obs_tokenizer import smacTokenizer  # noqa: E402

from tree_skills.teacher.obs_builder import (  # noqa: E402
    ObsBuildError, SmacObsBuilder)

DIMS = dict(n_agents=10, n_enemies=10, move_feats=4, own_feats=6,
            obs_nf_en=8, obs_nf_al=8)
HISTORY_LEN = 4
BLOCK = 700
VOCAB = 400
ACT = 20
N_ATTR = 129


def decompose_reference(obs_vector):
    """Verbatim port of SC2Decomposer.decompose_obs slicing for the frozen
    10v10 terran layout (see limitation note in the module docstring)."""
    obs = np.asarray(obs_vector, dtype=np.float32)
    move_feats = obs[:DIMS["move_feats"]]
    base = DIMS["move_feats"]
    enemy = [obs[base + i * DIMS["obs_nf_en"]:base + (i + 1) * DIMS["obs_nf_en"]]
             for i in range(DIMS["n_enemies"])]
    base += DIMS["n_enemies"] * DIMS["obs_nf_en"]
    ally = [obs[base + i * DIMS["obs_nf_al"]:base + (i + 1) * DIMS["obs_nf_al"]]
            for i in range(DIMS["n_agents"] - 1)]
    base += DIMS["obs_nf_al"] * (DIMS["n_agents"] - 1)
    own = np.concatenate([move_feats, obs[base:base + DIMS["own_feats"]]])
    return own, ally, enemy


class OfficialReference:
    """Pipeline assembled from official functions only (post-slicing)."""

    def __init__(self, last_token):
        with open(os.path.join(MARL_GPT, "envs", "smac_env",
                               "attr_for_all_map.json"), encoding="utf-8") as fh:
            attr = json.load(fh)
        self.tokenizer = smacTokenizer("terran", pos_dict=attr["terran"])
        self.last_token = last_token

    def build(self, frames_oldest_first):
        # official history stacking: per-step dicts (each key batch-leading,
        # batch=1 for our single-agent replay), then stack_dict_tensor
        steps = []
        for frame in frames_oldest_first:
            own, ally, enemy = decompose_reference(frame)
            steps.append({"obs_own": torch.tensor(np.asarray([own])),
                          "obs_ally": torch.tensor(np.asarray([ally])),
                          "obs_enemy": torch.tensor(np.asarray([enemy]))})
        stacked = stack_dict_tensor(steps, HISTORY_LEN, 1, "cpu")
        observation = {k: v for k, v in stacked.items()}  # (1, H, ...)
        obs, group, agent, time_, attr = self.tokenizer.get_obs_pos_vectors(observation)
        result = {"obs": obs, "group_pos": group, "agent_pos": agent,
                  "time_pos": time_, "attr_pos": attr}
        # verbatim InferenceSmacWrapper.append_agent_race (batch 1, terran 0)
        result["obs"] = torch.cat([result["obs"], torch.zeros(1, 1)], dim=-1)
        for key in ("group_pos", "agent_pos", "time_pos", "attr_pos"):
            result[key] = torch.cat([result[key], torch.zeros(1, dtype=torch.int)],
                                    dim=-1)
        result["attr_pos"][..., -1] = 16
        result = {k: (v if v.dim() >= 2 else v.unsqueeze(0))
                  for k, v in result.items()}
        result = reshape_obs(result, BLOCK, ACT)
        if self.last_token:
            result["attr_pos"][..., -1] = N_ATTR - 1
            result["obs"][..., -1] = 1.0
        return result


def replay(collect_rows, agent_id, episode):
    frames = []
    for row in collect_rows:
        if row["episode"] != episode:
            continue
        teacher = row.get("teacher_raw") or {}
        for entry in teacher.get("agents") or []:
            if entry["agent"] == agent_id:
                frames.append(entry["obs"])
    return frames  # ordered by game_loop as recorded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collect", required=True)
    parser.add_argument("--last-token", choices=("on", "off"), default="on")
    parser.add_argument("--samples", type=int, default=10)
    args = parser.parse_args()

    rows = [json.loads(line) for line in open(args.collect, encoding="utf-8") if line.strip()]
    episodes = sorted({r["episode"] for r in rows})
    ours = SmacObsBuilder(history_len=HISTORY_LEN, block_size=BLOCK,
                          vocab_size=VOCAB, action_size=ACT, n_attr=N_ATTR,
                          obs_with_race=True, last_token=(args.last_token == "on"),
                          **DIMS)
    reference = OfficialReference(args.last_token == "on")

    checked = mismatches = 0
    report = []
    for episode in episodes:
        for agent in range(DIMS["n_agents"]):
            frames = replay(rows, agent, episode)
            if len(frames) < 3:
                continue  # need some real history
            for cut in (min(3, len(frames)), len(frames)):
                seq = frames[:cut]
                # ours: feed sequentially
                ours.reset_episode(episode)
                for loop, obs_vec in enumerate(seq):
                    ours.observe(episode, agent, loop, obs_vec)
                mine = ours.build_input(episode, agent)
                # reference: official zero front-pad; keep the NEWEST
                # HISTORY_LEN frames exactly like the builder's deque
                zero = np.zeros_like(np.asarray(seq[0], dtype=np.float32))
                window = ([zero] * max(0, HISTORY_LEN - len(seq))
                          + list(seq))[-HISTORY_LEN:]
                ref = reference.build(window)
                checked += 1
                row_ok = True
                for key in ("obs", "group_pos", "agent_pos", "time_pos", "attr_pos"):
                    a = np.asarray(mine[key])
                    b = np.asarray(ref[key])
                    if a.shape != b.shape or not np.array_equal(a, b):
                        row_ok = False
                        report.append({"episode": episode, "agent": agent,
                                       "cut": cut, "key": key,
                                       "shape_a": list(a.shape),
                                       "shape_b": list(b.shape)})
                mismatches += 0 if row_ok else 1
                if checked >= args.samples * 2:
                    break
            if checked >= args.samples * 2:
                break
        if checked >= args.samples * 2:
            break
    result = {"collect": args.collect, "last_token": args.last_token,
              "comparisons": checked, "mismatches": mismatches,
              "limitation": "slicing replicated from decomposer.decompose_obs; "
                            "history/positions/race/padding run official code",
              "details": report[:10]}
    print(json.dumps(result, indent=1))
    out = os.path.join(HERE, "REFERENCE_CHECK_run.json")
    payload = {args.last_token: result}
    if os.path.exists(out):
        payload = {**json.load(open(out, encoding="utf-8")), **payload}
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1)
    if checked == 0:
        raise SystemExit("no comparable agent sequences found in collection")
    if mismatches:
        raise SystemExit("REFERENCE PARITY FAILED")
    print("REFERENCE PARITY OK")


if __name__ == "__main__":
    main()
