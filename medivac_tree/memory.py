"""Episode/tag/clock-scoped observed history, not imagined order execution."""

from copy import deepcopy


def fresh_memory(memory, state, tree_hash):
    loop = state.get("game_loop")
    step = state.get("step_mul", 8)
    if isinstance(loop, bool) or not isinstance(loop, int) or loop < 0:
        raise ValueError("game_loop must be a nonnegative integer")
    if isinstance(step, bool) or not isinstance(step, int) or step < 1:
        raise ValueError("step_mul must be a positive integer")
    prior = memory or {}
    reason = None
    if prior:
        if prior.get("schema_version") != 1 or prior.get("tree_hash") != tree_hash:
            reason = "schema_or_tree_changed"
        elif prior.get("episode") != state.get("episode"):
            reason = "episode_changed"
        elif prior.get("roster") != {a: u.get("tag") for a, u in state["allies"].items()}:
            reason = "unit_identity_changed"
        elif loop < prior.get("loop", -1):
            reason = "clock_reset"
        elif loop not in (prior.get("loop"), prior.get("loop", -1) + step):
            reason = "noncontiguous_clock"
    return ({} if reason else deepcopy(prior)), reason


def actor_memory(prior, actor, unit, loop, step):
    record = prior.get("actors", {}).get(actor, {})
    if record.get("tag") != unit.get("tag") or record.get("loop") != loop - step:
        return {}
    return deepcopy(record)
