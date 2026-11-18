"""Strict configuration validation (R5).

Unknown keys, wrong types, out-of-range or non-finite values are rejected.
Boolean flags must be real bools (no `bool("false")` acceptance); files must
be JSON objects.
"""
import json
import math
from pathlib import Path
from typing import Any, Dict, Optional

DEFAULTS_PATH = Path(__file__).resolve().parent / "defaults.json"


class ConfigError(ValueError):
    pass


def _bool_spec():  return (bool, None, None)
def _int_spec(lo, hi):  return (int, lo, hi)
def _float_spec(lo, hi):  return (float, lo, hi)


def _str_spec(options):
    return (str, tuple(options), None)


# schema: section -> {key: (type, lo/min, hi/max or options)}
_SCHEMA = {
    "enabled": {name: _bool_spec() for name in (
        "focus_fire", "coordinated_engage", "combat_reposition",
        "disengage_reenter", "regroup", "local_formation", "cohesive_search",
        "splash_fire", "heal_support", "burst_engage", "precombat_rally",
        "combat_rejoin")},
    "weights": {
        "useful_damage": _float_spec(0.0, 10.0),
        "kill": _float_spec(0.0, 10.0),
        "heal": _float_spec(0.0, 10.0),
        "coverage": _float_spec(0.0, 10.0),
        "tactic_progress": _float_spec(0.0, 10.0),
        "link": _float_spec(0.0, 10.0),
        "meeting": _float_spec(0.0, 10.0),
        "role": _float_spec(0.0, 10.0),
        "risk": _float_spec(0.0, 10.0),
        "congestion": _float_spec(0.0, 10.0),
        "switch": _float_spec(0.0, 10.0),
        "reentry_cost": _float_spec(0.0, 10.0),
        "planned_loss": _float_spec(0.0, 10.0),
        "hold_value": _float_spec(0.0, 1.0),
        "support_break": _float_spec(0.0, 10.0),
        "attack_continuity": _float_spec(0.0, 10.0),
        "rally_wait_cost": _float_spec(0.0, 1.0),
        "future_attack": _float_spec(0.0, 10.0),
        "target_threat": _float_spec(0.0, 10.0),
        "combat_idle": _float_spec(0.0, 10.0),
        "movement_continuity": _float_spec(0.0, 10.0),
        "blocked_move": _float_spec(0.0, 10.0),
        "unsupported_frontline": _float_spec(0.0, 10.0),
        "sustained_contact": _float_spec(0.0, 10.0),
    },
    "forecasting": {
        "forecast_reach_discount": _float_spec(0.0, 1.0),
        "uncertain_position_floor": _float_spec(0.0, 1.0),
    },
    "common_combat": {
        "joint_assessment_enabled": _bool_spec(),
        "mechanism_recovery_enabled": _bool_spec(),
        "next_fire_budget_enabled": _bool_spec(),
        "pressure_memory_enabled": _bool_spec(),
        "shared_front_enabled": _bool_spec(),
        "local_cover_enabled": _bool_spec(),
        "arrival_exposure_enabled": _bool_spec(),
        "reentry_window_enabled": _bool_spec(),
        # Complete-plan joint kill may satisfy a rescue duty (task 3).
        "joint_rescue_enabled": _bool_spec(),
        # Melee/zergling rescue eligibility under ordinary fire (task 4).
        "melee_survivable_rescue_enabled": _bool_spec(),
        # Bounded "delay one shot for damage reduction" budget (task 5).
        "bounded_fire_delay_enabled": _bool_spec(),
        # Engineering initial value (NOT a validated optimum); windows of
        # postponed fire regained per committed real attack.
        "fire_delay_max_h": _float_spec(0.5, 2.0),
    },
    "search": {
        "beam_width": _int_spec(1, 64),
        "max_plan_evaluations": _int_spec(1, 100000),
        "rounds": _int_spec(1, 8),
        "max_candidates_per_unit": _int_spec(1, 16),
        "max_atomic_candidates": _int_spec(0, 64),
    },
    "lifecycle": {  # multiples of the control window H (engineering starts)
        "no_progress_window_h": _float_spec(0.5, 10.0),
        "retry_cooldown_h": _float_spec(0.0, 10.0),
        "engage_target_max_h": _float_spec(1.0, 40.0),
        "engage_wait_max_h": _float_spec(0.0, 8.0),
        "escape_max_h": _float_spec(1.0, 10.0),
        "stabilize_max_h": _float_spec(0.0, 10.0),
        "reenter_max_h": _float_spec(1.0, 20.0),
        "disengage_total_max_h": _float_spec(2.0, 40.0),
        "regroup_max_h": _float_spec(1.0, 30.0),
        "formation_max_h": _float_spec(0.5, 10.0),
        "search_waypoint_max_h": _float_spec(1.0, 40.0),
        "observation_ttl_loops": _int_spec(16, 100000),
        "search_revisit_cooldown_loops": _int_spec(16, 100000),
        "combat_cycle_commit_h": _float_spec(0.0, 10.0),
        "combat_cycle_memory_h": _float_spec(1.0, 40.0),
    },
    "contact_progress": {
        "stall_h": _float_spec(1.0, 12.0),
        "deadline_h": _float_spec(2.0, 24.0),
    },
    "battle_intent": {
        "commit_h": _float_spec(1.0, 12.0),
        "stall_h": _float_spec(2.0, 12.0),
        "quiet_h": _float_spec(2.0, 20.0),
        "failed_retreats": _int_spec(1, 6),
    },
    "focus_fire": {
        "target_planning_enabled": _bool_spec(),
        "kill_window_h": _float_spec(0.5, 10.0),
        "joint_kill_max_members": _int_spec(0, 10),
        "urgency_value": _float_spec(0.0, 1.0),
    },
    "coordinated_engage": {"slot_spacing": _float_spec(0.2, 5.0),
                           "contact_slot_margin": _float_spec(0.0, 1.0)},
    "precombat_rally": {
        "min_members": _int_spec(2, 10),
        "ready_radius": _float_spec(1.0, 15.0),
        "max_member_distance": _float_spec(2.0, 40.0),
        "trigger_enemy_distance": _float_spec(2.0, 40.0),
        "ready_fraction": _float_spec(0.5, 1.0),
        "max_h": _float_spec(1.0, 30.0),
        "min_commit_h": _float_spec(0.0, 5.0),
        "no_progress_h": _float_spec(0.5, 10.0),
        "rearm_quiet_h": _float_spec(1.0, 20.0),
    },
    "combat_reposition": {},
    "combat_rejoin": {
        "support_radius": _float_spec(1.0, 20.0),
        "min_members": _int_spec(2, 10),
        "max_members": _int_spec(2, 10),
        "max_h": _float_spec(1.0, 30.0),
        "min_commit_h": _float_spec(0.0, 10.0),
        "no_progress_h": _float_spec(0.5, 10.0),
        "min_support_ratio": _float_spec(0.1, 5.0),
    },
    "disengage_reenter": {
        "critical_health_fraction": _float_spec(0.01, 1.0),
        "reenter_health_fraction": _float_spec(0.01, 1.0),
        "escape_risk_threshold": _int_spec(1, 10),
    },
    "regroup": {"support_radius": _float_spec(1.0, 30.0)},
    "local_formation": {
        "spread_radius": _float_spec(0.5, 10.0),
        "min_pair_distance": _float_spec(0.2, 5.0),
    },
    "cohesive_search": {
        "arrival_tolerance": _float_spec(0.5, 10.0),
        "cohesion_radius": _float_spec(2.0, 40.0),
    },
    "splash_fire": {
        "coverage_reliability": _float_spec(0.0, 1.0),
        "coverage_radius": _float_spec(0.5, 10.0),
        "coverage_weight": _float_spec(0.05, 1.0),
    },
    "heal_support": {
        "follow_distance": _float_spec(1.0, 15.0),
        "switch_margin": _float_spec(0.0, 1.0),
    },
    "burst_engage": {
        "proxy_radius": _float_spec(0.5, 10.0),
        "value_per_enemy": _float_spec(0.05, 5.0),
        "min_exchange_value": _float_spec(0.0, 10.0),
        "self_cost": _float_spec(0.0, 5.0),
        "lonely_enemy_exception": _bool_spec(),
        # Short-window enemy motion prediction for burst-point timing.
        "motion_prediction_enabled": _bool_spec(),
        # Only state-backed strong exchanges become forced cashout duties.
        "selective_commitment_enabled": _bool_spec(),
    },
    # Legacy pure-evaluator weights reused through unit_value_view; manual
    # engineering priors, bounded by the legacy parameter surface.
    "unit_value": {
        "enabled": _bool_spec(),
        "mode": _str_spec(("off", "basic", "contextual")),
        "spatial_context": _bool_spec(),
        "w_matchup": _float_spec(0.0, 1.0),
        "w_support": _float_spec(0.0, 1.0),
        "w_scarcity": _float_spec(0.0, 1.0),
        "scale": _float_spec(0.1, 10.0),
        "focus_value_weight": _float_spec(0.0, 0.5),
        "focus_switch_margin": _float_spec(0.0, 1.0),
        "retreat_value_margin": _float_spec(0.0, 0.2),
        "retreat_threshold_upper_bound": _float_spec(0.0, 1.0),
        "heal_value_weight": _float_spec(0.0, 0.5),
    },
    "intercept": {
        "enabled": _bool_spec(),
        # Require a conservative (upper-bound) kill time before calling an
        # interception "safe"; unknown timing degrades to loss-control.
        "timing_guard_enabled": _bool_spec(),
        # Loss-control interception duties when members are already exposed.
        "loss_control_enabled": _bool_spec(),
        # Relative margin required before deferring a currently positive
        # one-use exchange for a strictly better reachable point.
        "burst_defer_margin": _float_spec(0.0, 5.0),
    },
    "telemetry": {"top_k_candidates": _int_spec(1, 20)},
}
_BOOL_KEYS = {"_fingerprint"}
_BOOL_KEYS = set()   # fingerprint injected at runtime, never user-settable


def _check(value, spec, path):
    kind, lo, hi = spec
    if kind is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{path} must be a boolean, got {value!r}")
        return
    if kind is str:
        if not isinstance(value, str) or value not in lo:
            raise ConfigError(f"{path} must be one of {sorted(lo)}")
        return
    if isinstance(value, bool) or not isinstance(value, kind):
        raise ConfigError(f"{path} must be {kind.__name__}, got {value!r}")
    if not math.isfinite(value):
        raise ConfigError(f"{path} must be finite, got {value!r}")
    if lo is not None and value < lo:
        raise ConfigError(f"{path} below minimum {lo}: {value}")
    if hi is not None and value > hi:
        raise ConfigError(f"{path} above maximum {hi}: {value}")


def validate(config: Dict[str, Any]) -> Dict[str, Any]:
    for section, entries in config.items():
        if section == "_fingerprint":
            continue
        if section not in _SCHEMA:
            raise ConfigError(f"unknown config section: {section}")
        if not isinstance(entries, dict):
            raise ConfigError(f"config section {section} must be an object")
        schema = _SCHEMA[section]
        for key, value in entries.items():
            if key not in schema:
                raise ConfigError(f"unknown config key: {section}.{key}")
            _check(value, schema[key], f"{section}.{key}")
    # required sections
    for required in ("enabled", "weights", "search", "lifecycle"):
        if required not in config:
            raise ConfigError(f"missing required config section: {required}")
    return config


def load(path: Optional[str] = None) -> Dict[str, Any]:
    with open(DEFAULTS_PATH, encoding="utf-8") as handle:
        config = json.load(handle)
    if path is not None:
        override_file = Path(path)
        if not override_file.is_file():
            raise ConfigError(f"--tactical-config: no such file: {override_file}")
        override = json.loads(override_file.read_text(encoding="utf-8"))
        if not isinstance(override, dict):
            raise ConfigError("tactical config must be a JSON object")
        _deep_merge(config, override)
    return validate(config)


def _deep_merge(base: Dict, override: Dict) -> None:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
