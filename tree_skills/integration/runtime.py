"""Integration runtime: build injection kwargs for the original skill stack.

Modes (task §12): off / shadow / teacher-score / tree-score. The bridge or
runner passes these as extra kwargs to SkillManager/allocate_terran_combat;
with mode "off" nothing is created, imported or changed — the original
behaviour is bit-for-bit identical.

Usage inside a runner:
    kwargs = integration_kwargs(mode, model_dir=...)
    actions, telemetry = skill_manager.step(payload, n_agents, **kwargs)
(threading depends on the runner's call site; see docs/EXPERIMENT_PROTOCOL)
"""

import json
import os


def integration_kwargs(mode, model_dir=None, config_path=None, scorer_mode=None):
    """Kwargs for allocate_terran_combat / assign_healing.

    Returns {} for mode "off" — no scorer object, no heavy imports, no
    random draws, no network. Schemas are validated eagerly at startup.

    ``shadow`` records the scorer's proposed matching without applying it;
    it needs ``scorer_mode`` ("teacher-score" or "tree-score") to know
    WHICH model to shadow, or ``model_dir`` (implies tree-score).
    """
    if config_path:
        with open(config_path, "r", encoding="utf-8") as fh:
            cfg = json.load(fh)
        mode = cfg.get("mode", mode)
        model_dir = model_dir or cfg.get("model_dir")
        scorer_mode = scorer_mode or cfg.get("scorer_mode")
    if mode == "off":
        return {}
    from ..medivac.pair_scorer import make_pair_scorer
    if mode == "shadow":
        underlying = scorer_mode or ("tree-score" if model_dir else None)
        if underlying is None:
            raise ValueError("shadow mode needs scorer_mode or model_dir")
        return {"pair_scorer": make_pair_scorer(underlying, model_dir=model_dir),
                "pair_scoring": "shadow"}
    if mode == "a-plug":
        return {"pair_scorer": make_pair_scorer("a-plug"), "pair_scoring": "on"}
    return {"pair_scorer": make_pair_scorer(mode, model_dir=model_dir),
            "pair_scoring": "on"}
