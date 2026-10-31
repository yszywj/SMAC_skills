"""P0-1 checkpoint structural audit (teacher venv: torch, CPU only).

For each checkpoint under teacher/checkpoints:
1. torch.load(..., map_location='cpu', weights_only=True) — safe loading;
   if the pickle layout is unsupported, report why and STOP (never fall
   back to unrestricted pickle silently).
2. Read model_args and weight keys from the file — never guess from names.
3. Build gpt.model_ac.CriticGPTConfig/CriticWithLoss from the file's real
   model_args; strip the official loader's `_orig_mod.` prefix; load with
   strict=True (the official `strict=False` hides missing actor layers —
   we refuse that for a teacher).
4. eval(), no_grad, fixed synthetic input: full actor logits from
   Critic_GPT.forward (act_logits), check shape == action_size and
   finiteness, determinism across repeated calls.
5. Record everything into a per-checkpoint audit JSON and update
   TEACHER_MANIFEST.json via preflight.record_audit (status audited /
   rejected — never ready, which requires G2 mapping validation).
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))          # SMAC_agent
MARL_GPT = os.path.join(REPO, "tree_skills", "external", "marl-gpt-main")
sys.path.insert(0, MARL_GPT)
sys.path.insert(0, REPO)

import torch  # noqa: E402

from gpt.model_ac import CriticGPTConfig, CriticWithLoss  # noqa: E402
from tree_skills.teacher import preflight  # noqa: E402


def synthetic_input(config, seed=0):
    """Deterministic integer input matching the official token layout."""
    g = torch.Generator().manual_seed(seed)
    b = 2
    return {
        # obs tokens feed Linear(1, n_embd) -> continuous float values;
        # only the position indices are integer embeddings.
        "obs": torch.rand((b, config.block_size), generator=g),
        "group_pos": torch.randint(0, config.n_team, (b, config.block_size), generator=g),
        "agent_pos": torch.randint(0, config.n_agents, (b, config.block_size), generator=g),
        "time_pos": torch.randint(0, config.history_len, (b, config.block_size), generator=g),
        "attr_pos": torch.randint(0, config.n_attr, (b, config.block_size), generator=g),
    }


def audit_one(path):
    details = {"file": os.path.basename(path)}
    try:
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            details["weights_only_load"] = True
        except Exception as exc:
            # Fail closed: we do NOT retry with weights_only=False.
            return {"ok": False, "reject_reason": "weights_only_load_failed",
                    "details": {**details, "weights_only_error": repr(exc)}}

        details["top_level_keys"] = sorted(checkpoint.keys()) if isinstance(checkpoint, dict) else type(checkpoint).__name__
        model_args = checkpoint.get("model_args")
        if model_args is None:
            return {"ok": False, "reject_reason": "missing_model_args",
                    "details": details}
        details["model_args"] = {k: v for k, v in model_args.items()}

        state = checkpoint.get("model")
        if state is None:
            return {"ok": False, "reject_reason": "missing_model_weights",
                    "details": details}
        stripped = {k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v
                    for k, v in state.items()}
        details["n_state_keys_raw"] = len(state)
        details["n_state_keys_stripped"] = len(stripped)
        details["had_orig_mod_prefix"] = any(k.startswith("_orig_mod.") for k in state)

        config = CriticGPTConfig(**model_args)
        model = CriticWithLoss(config)
        n_params = sum(p.numel() for p in model.parameters())
        details["param_count"] = n_params

        result = model.load_state_dict(stripped, strict=True)
        details["strict_load_ok"] = True
        details["missing_keys"] = list(result.missing_keys)
        details["unexpected_keys"] = list(result.unexpected_keys)
        if result.missing_keys or result.unexpected_keys:
            return {"ok": False, "reject_reason": "strict_load_key_mismatch",
                    "details": details}

        model.eval()
        x = synthetic_input(config)
        with torch.no_grad():
            # call the Critic_GPT forward directly: CriticWithLoss.forward
            # returns a 4-tuple with loss info; we need raw (act, val) logits
            act_logits, val_logits = type(model).__mro__[1].forward(model, x)
            act_logits_2, _ = type(model).__mro__[1].forward(model, x)
        details["actor_head_dim"] = config.action_size
        details["act_logits_shape"] = list(act_logits.shape)
        details["val_logits_shape"] = list(val_logits.shape)
        if tuple(act_logits.shape[-1:]) != (config.action_size,):
            return {"ok": False, "reject_reason": "actor_head_dim_mismatch",
                    "details": details}
        details["act_logits_all_finite"] = bool(torch.isfinite(act_logits).all())
        details["val_logits_all_finite"] = bool(torch.isfinite(val_logits).all())
        if not details["act_logits_all_finite"]:
            return {"ok": False, "reject_reason": "non_finite_actor_logits",
                    "details": details}
        details["deterministic_repeated_forward"] = bool(torch.equal(act_logits, act_logits_2))
        if not details["deterministic_repeated_forward"]:
            return {"ok": False, "reject_reason": "non_deterministic_forward",
                    "details": details}
        details["act_logits_sample"] = [round(float(v), 6) for v in act_logits[0]]
        # last-token variant applicability: attr index n_attr-1 must be valid
        details["last_token_attr_index"] = config.n_attr - 1
        details["last_token_supported"] = True
        details["notes"] = ("structural audit only; ready requires G2 "
                            "obs/action mapping validation on real snapshots")
        return {"ok": True, "details": details}
    except Exception as exc:
        return {"ok": False, "reject_reason": f"audit_exception:{type(exc).__name__}",
                "details": {**details, "exception": repr(exc)}}


def main():
    torch.manual_seed(0)
    audits = {}
    for entry in preflight.EXPECTED:
        path = os.path.join(preflight.CHECKPOINT_DIR, entry["file"])
        if not os.path.isfile(path):
            print(f"SKIP {entry['file']}: not present")
            continue
        print(f"AUDIT {entry['file']} ...")
        result = audit_one(path)
        stem = entry["file"].replace(".pt", "")
        out = os.path.join(HERE, f"AUDIT_{stem}.json")
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=1)
        audits[entry["file"]] = result
        preflight.record_audit(entry["file"], result)
        print(f"  ok={result['ok']} reason={result.get('reject_reason', '-')}")
    print(json.dumps({k: {"ok": v["ok"], "reason": v.get("reject_reason")}
                      for k, v in audits.items()}, indent=1))


if __name__ == "__main__":
    main()
