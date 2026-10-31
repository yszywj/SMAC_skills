"""G2 inference validation: run the resident teacher service end-to-end on
synthetic-but-layout-faithful raw observations and through the REAL
injection channel (assign_healing, mode "on").

Teacher-env script (torch). This validates plumbing, mapping, determinism,
capacity and failure modes. It does NOT validate game-level teacher
quality (that is G4, real A/B evaluation) — logits here come from
synthetic observation vectors, and the run is marked synthetic_only.

Usage (teacher venv):
  python tree_skills/teacher/validate_inference.py [--last-token on|off]
"""

import argparse
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)

CHECKPOINT = os.path.join(HERE, "checkpoints", "marl-gpt-main.pt")

# terran 10v10, obs_own_pos=True, unit_type_bits=3, conic_fov=False
DIMS = dict(n_agents=10, n_enemies=10, move_feats=4, own_feats=6,
            obs_nf_en=8, obs_nf_al=8)
OBS_DIM = (DIMS["move_feats"] + DIMS["n_enemies"] * DIMS["obs_nf_en"]
           + (DIMS["n_agents"] - 1) * DIMS["obs_nf_al"] + DIMS["own_feats"])  # 162


def synthetic_obs(rng, alive=True):
    if not alive:
        return [0.0] * OBS_DIM
    obs = [rng.uniform(-1, 1) for _ in range(OBS_DIM)]
    return obs


def run(last_token):
    results = {"synthetic_only": True, "last_token": last_token,
               "obs_dim": OBS_DIM, "dims": DIMS}

    from tree_skills.teacher.inference import (TeacherPairScorer,
                                               TeacherServiceClient)
    client = TeacherServiceClient(CHECKPOINT, last_token=last_token)
    results["model_sha256"] = client.model_sha256
    results["model_args_subset"] = {k: client.model_args[k] for k in
                                    ("block_size", "history_len", "action_size",
                                     "n_attr", "vocab_size", "n_agents")}
    info = client.init(DIMS)
    results["per_frame_tokens"] = info["per_frame_tokens"]

    rng = random.Random(7)
    # 3 steps of history for the whole team, agent 3 dead on step 2
    all_logits = []
    for loop in range(3):
        agents = [{"agent": a, "obs": synthetic_obs(rng, alive=not (loop == 1 and a == 3)),
                   "avail": [1 if i in (1, 2, 3, 4, 5) or i >= 6 else 0 for i in range(16)]}
                  for a in range(10)]
        logits = client.logits("ep-validate", loop, agents)
        all_logits.append(logits)
    results["steps"] = 3
    results["logits_length"] = len(all_logits[0][0])
    results["logits_all_finite"] = all(
        v == v and abs(v) != float("inf")
        for step in all_logits for row in step.values() for v in row)
    # determinism: fresh service, same obs sequence -> identical logits
    client2 = TeacherServiceClient(CHECKPOINT, last_token=last_token)
    client2.init(DIMS)
    rng = random.Random(7)
    again = []
    for loop in range(3):
        agents = [{"agent": a, "obs": synthetic_obs(rng, alive=not (loop == 1 and a == 3)),
                   "avail": [1] * 16} for a in range(10)]
        again.append(client2.logits("ep-validate", loop, agents))
    results["deterministic_across_service_restarts"] = all(
        all_logits[s][a] == again[s][a] for s in range(3) for a in range(10))

    # duplicate (episode, agent, loop) must not silently double-advance:
    # the server treats a repeated loop as idempotent — verify via scorer guard
    scorer = TeacherPairScorer(client2, n_agents=10, n_enemies=10)
    rng = random.Random(9)
    agents = [{"agent": a, "obs": synthetic_obs(rng), "avail": [1] * 16}
              for a in range(10)]
    scorer.begin_step("ep-score", 0, agents)
    try:
        scorer.begin_step("ep-score", 0, agents)
        results["duplicate_step_guard"] = "MISSING"
    except Exception as exc:
        results["duplicate_step_guard"] = type(exc).__name__
    # candidate scoring via heal slots, through the real injection layer
    from tree_skills.teacher.inference import TeacherServiceError
    ok = scorer("u4", "u0", {})  # context unused by teacher scorer
    results["scorer_heal_slot_u4_u0"] = ok
    try:
        scorer("u4", "u16", {})  # out-of-range ally
        results["out_of_range_ally_guard"] = "MISSING"
    except Exception as exc:
        results["out_of_range_ally_guard"] = type(exc).__name__
    client.close()
    client2.close()
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--last-token", choices=["on", "off"], default="off")
    args = parser.parse_args()
    results = run(args.last_token)
    out = os.path.join(HERE, "INFERENCE_VALIDATION_run.json")
    payload = {args.last_token: results}
    if os.path.exists(out):
        with open(out, encoding="utf-8") as fh:
            payload = {**json.load(fh), args.last_token: results}
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1)
    print(json.dumps(results, indent=1))
    failures = [k for k, v in results.items()
                if isinstance(v, str) and v == "MISSING"]
    if failures or not results["logits_all_finite"] or \
            not results["deterministic_across_service_restarts"]:
        raise SystemExit("INFERENCE VALIDATION FAILED")
    print("INFERENCE VALIDATION OK (synthetic_only; G2 plumbing evidence, "
          "not game-level teacher quality)")


if __name__ == "__main__":
    main()
