"""P0-3: resident MARL-GPT teacher inference service.

Runs in the TEACHER environment (torch + archived official code). The
checkpoint is loaded ONCE at startup; requests are JSONL on stdin, one JSON
response line per request on stdout. Deterministic: eval mode, no_grad,
fixed seeds, CPU (or cuda if available).

Request / response protocol (version-stamped, auditable):

  {"op": "init", "dims": {n_agents, n_enemies, move_feats, own_feats,
                          obs_nf_en, obs_nf_al}}
      -> {"ok": true, "model_sha256": ..., "model_args": {...},
          "history_len": H, "last_token": bool}
  {"op": "logits", "id": <int>, "episode": <str>, "game_loop": <int>,
   "agents": [{"agent": <int>, "obs": [...], "avail": [...]}]}
      -> {"id": ..., "results": [{"agent": a, "logits": [...20 floats...],
          "n_native_actions": 16}], "elapsed_ms": ...}
      (batched: all agents of one step in ONE forward pass; per-agent
      histories advance exactly once — repeated (agent, game_loop) pairs
      reuse the recorded frame)
  {"op": "reset", "episode": <str>}  -> {"ok": true}
  {"op": "ping"} -> {"pong": model_sha256}

Errors are structured: {"id": ..., "error": {"kind": ..., "detail": ...}}
and never crash the resident process.
"""

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
MARL_GPT = os.path.join(REPO, "tree_skills", "external", "marl-gpt-main")
for p in (MARL_GPT, REPO):
    if p not in sys.path:
        sys.path.insert(0, p)


def load_model(checkpoint_path, device):
    import torch
    from gpt.model_ac import CriticGPTConfig, CriticWithLoss

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model_args = dict(checkpoint["model_args"])
    state = checkpoint["model"]
    state = {k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v
             for k, v in state.items()}
    config = CriticGPTConfig(**model_args)
    model = CriticWithLoss(config)
    result = model.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f"strict load mismatch: {result}")
    model.to(device).eval()
    return model, model_args


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--last-token", choices=["on", "off"], default="off",
                        help="append the env-indicator last token or not")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    import torch
    torch.manual_seed(0)
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    from tree_skills.teacher import preflight
    from tree_skills.teacher.obs_builder import SmacObsBuilder

    model, model_args = load_model(args.checkpoint, device)
    import hashlib
    with open(args.checkpoint, "rb") as fh:
        model_sha = hashlib.sha256(fh.read()).hexdigest()
    builder = None  # created on init (needs live env dims)
    # history_len 4: model table allows 10, but block_size 700 fits only
    # 4 frames of the terran 10v10 token layout (see obs_builder docstring)
    base = dict(n_agents=10, n_enemies=10, history_len=4,
                block_size=model_args["block_size"], vocab_size=model_args["vocab_size"],
                action_size=model_args["action_size"], n_attr=model_args["n_attr"],
                obs_with_race=True,
                last_token=(args.last_token == "on"))

    sys.stdout.write(json.dumps({"ok": True, "model_sha256": model_sha,
                                 "model_args": model_args,
                                 "history_len": model_args["history_len"],
                                 "last_token": args.last_token}) + "\n")
    sys.stdout.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            if request.get("op") == "ping":
                reply = {"pong": model_sha}
            elif request.get("op") == "init":
                builder = SmacObsBuilder(**{**base, **request["dims"]})
                reply = {"ok": True, "per_frame_tokens": builder.per_frame_tokens}
            elif request.get("op") == "reset":
                if builder is not None:  # resets before init are no-ops
                    builder.reset_episode(request["episode"])
                reply = {"ok": True}
            elif request.get("op") == "logits":
                started = time.perf_counter()
                episode, loop = request["episode"], request["game_loop"]
                rows = []
                for entry in request["agents"]:
                    builder.observe(episode, entry["agent"], loop, entry["obs"])
                for entry in request["agents"]:
                    rows.append((entry["agent"], builder.build_input(episode, entry["agent"])))
                batched = {k: torch.cat([x[k] for _, x in rows]).to(device)
                           for k in rows[0][1]}
                with torch.no_grad():
                    act_logits, _ = type(model).__mro__[1].forward(model, batched)
                if not torch.isfinite(act_logits).all():
                    raise ValueError("non-finite teacher logits")
                n_native = 6 + max(builder.n_agents, builder.n_enemies)
                results = [{"agent": a, "logits": [float(v) for v in act_logits[i]],
                            "n_native_actions": n_native}
                           for i, (a, _) in enumerate(rows)]
                reply = {"id": request.get("id"), "results": results,
                         "elapsed_ms": round((time.perf_counter() - started) * 1000, 3)}
            else:
                reply = {"id": request.get("id"),
                         "error": {"kind": "unknown_op",
                                   "detail": request.get("op")}}
        except Exception as exc:
            reply = {"error": {"kind": type(exc).__name__, "detail": str(exc)}}
        sys.stdout.write(json.dumps(reply) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
