"""Deployment-faithful forced-participation matching (shared reference).

Same algorithm as skills/current/skills/heal_assign.py::_forced_matching —
the injection layer keeps its own copy so the skills package stays
importable without tree_skills; this module is the reference used by
training/evaluation metrics and by consistency tests that assert the two
implementations agree bit-for-bit on random inputs.

Every participating healer MUST take exactly one target (no idle column,
negative edges allowed). Deterministic: healers sorted by unit number,
options by (score desc, target id), strict-improvement updates.
Inputs larger than MAX_HEALERS raise — never a silent fallback.
"""

MAX_HEALERS = 8


def _key(healer):
    return int(healer[1:])


def forced_matching(healers, edges):
    """Return {healer: target} or None when infeasible (no legal edge)."""
    healers = sorted(healers, key=_key)
    if len(healers) > MAX_HEALERS:
        raise ValueError(
            f"forced matching is defined for up to {MAX_HEALERS} "
            f"participating healers, got {len(healers)}; refusing to "
            "silently fall back to a different solver")
    for healer in healers:
        if not edges.get(healer):
            return None
    best = {"score": None, "assign": None}

    def recurse(index, used, total, assign):
        if index == len(healers):
            if best["score"] is None or total > best["score"]:
                best["score"], best["assign"] = total, dict(assign)
            return
        for score, target in edges[healers[index]]:
            if target in used:
                continue
            assign[healers[index]] = target
            recurse(index + 1, used | {target}, total + score, assign)
            del assign[healers[index]]

    recurse(0, frozenset(), 0.0, {})
    return best["assign"]
