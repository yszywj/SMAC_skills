"""Per-snapshot object-value view over the legacy pure evaluator.

This is a read-only ADAPTER, not a second implementation: all estimation
logic lives in ``skills/current/unit_value.py`` and is imported verbatim.
The result is computed once per decision snapshot and cached on the Context;
queries below never create, renew or commit tasks.

Three explicitly separated uses (no overlap with the evaluator's damage /
kill / risk ledgers — these values only ORDER candidate generation and
duty construction; they are never added to score_joint):
  * preservation value  — ally importance for healing/rescue/damage soaking;
  * target value        — enemy importance for ordinary target ordering;
  * immediate threat    — NOT computed here; the baneling intercept module
    (intercept.py) supplies short-horizon threat facts from actual effect
    geometry, because object value is not a loss prediction.

Unknown units or missing fields fall back to the legacy evaluator's own
neutral handling (base 1.0, factor 1.0, null components) — nothing here
invents mechanics. The legacy manual weights are initial priors only.
"""
import sys
from pathlib import Path

_WORKSPACE = Path(__file__).resolve().parents[2]
if str(_WORKSPACE) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE))

from skills.current.unit_value import (  # noqa: E402  (shared pure module)
    DEFAULT_PARAMETERS as _LEGACY_DEFAULTS,
    evaluate_unit_values as _legacy_evaluate,
    validate_parameters as _legacy_validate)

# Config keys accepted by the tactical_v1 "unit_value" section. The legacy
# parameter surface is reused unchanged so both entries stay one schema.
CONFIG_KEYS = tuple(sorted(k for k in _LEGACY_DEFAULTS if k != "mode"))
DEFAULTS = {"enabled": True, "mode": "contextual", "spatial_context": True,
            "w_matchup": .2, "w_support": .2, "w_scarcity": .2, "scale": 2.,
            "focus_value_weight": .2, "focus_switch_margin": .05,
            "retreat_value_margin": .05, "retreat_threshold_upper_bound": .6,
            "heal_value_weight": .1}


def parameters_from_config(config):
    """Map the tactical_v1 config section onto the legacy parameter dict.

    The legacy "off" default is never used here: when this section is
    disabled the whole view short-circuits to neutral values instead.
    """
    section = (config or {}).get("unit_value", {})
    params = {key: section.get(key, DEFAULTS[key]) for key in CONFIG_KEYS}
    params["mode"] = section.get("mode", DEFAULTS["mode"])
    params["spatial_context"] = bool(section.get("spatial_context",
                                                 DEFAULTS["spatial_context"]))
    return params


def evaluate(context):
    """One evaluation per snapshot; disabled/failed entry is fully neutral."""
    cached = getattr(context, "_unit_value_rows", None)
    if cached is not None:
        return cached
    rows = {"allies": {}, "visible_enemies": {}, "version": None, "enabled": False}
    config = getattr(context, "unit_value_parameters", None)
    if config is None or not config.get("enabled", True) or config.get('mode') == 'off':
        context._unit_value_rows = rows
        return rows
    try:
        params = _legacy_validate(parameters_from_config({'unit_value': config}))
        rows = _legacy_evaluate(context.snapshot.raw_state, params)
        rows["enabled"] = True
    except Exception:            # unknown shapes stay neutral, never fatal
        rows = {"allies": {}, "visible_enemies": {},
                "version": None, "enabled": False, "error": "evaluation_failed"}
    context._unit_value_rows = rows
    return rows


def preservation_value(context, actor) -> float:
    """Ally protection importance in [0,1]; unknown/missing is neutral 0."""
    if not context.snapshot.allies.get(actor) \
            or not context.snapshot.allies[actor].alive:
        return 0.
    row = evaluate(context)["allies"].get(actor, {})
    value = row.get("preservation_value", row.get("normalized_importance"))
    try:
        return min(1., max(0., float(value))) if value is not None else 0.
    except (TypeError, ValueError):
        return 0.


def target_value(context, enemy_id) -> float:
    """Enemy combat importance in [0,1] for ordinary target ORDERING only."""
    if enemy_id not in context.snapshot.enemies:
        return 0.
    row = evaluate(context)["visible_enemies"].get(enemy_id, {})
    value = row.get("target_priority_value", row.get("normalized_importance"))
    try:
        return min(1., max(0., float(value))) if value is not None else 0.
    except (TypeError, ValueError):
        return 0.


def summary(context, limit=4):
    """Compact diagnostic rows; predicted ordering, never observed damage."""
    rows = evaluate(context)
    if not rows.get("enabled"):
        return {"enabled": False}
    allies = sorted(((preservation_value(context, a), a) for a in rows["allies"]),
                    reverse=True)[:limit]
    enemies = sorted(((target_value(context, e), e) for e in rows["visible_enemies"]),
                     reverse=True)[:limit]
    return {"enabled": True, "version": rows.get("version"),
            "mode": rows.get("mode"),
            "top_preservation": [{"actor": a, "value": round(v, 3)}
                                 for v, a in allies if v > 0],
            "top_targets": [{"enemy": e, "value": round(v, 3)}
                            for v, e in enemies if v > 0]}
