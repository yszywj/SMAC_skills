"""Skill lifecycle: activation, multi-step execution, termination, telemetry."""

from time import perf_counter
from copy import deepcopy

from .unit_value import evaluate_unit_values, validate_parameters
from .skill_policy import (ADAPTIVE_FOCUS_FIRE, COHESIVE_SEARCH, COVER_RETREAT,
                           DIRECT_CONTROL, FOCUS_FIRE_BUDGET, HEAL_ASSIGN, REGROUP)
from .skills.focus_fire import allocate_focus_fire
from .skills.cover_retreat import allocate_cover_retreat, PARAMS as COVER_PARAMS
from .skills.heal_assign import assign_healing, PARAMS as HEAL_PARAMS
from .skills.adaptive_focus_fire import (PARAMS as ADAPTIVE_PARAMS,
                                         allocate_adaptive_focus_fire)
from .skills.regroup import allocate_regroup, PARAMS as REGROUP_PARAMS
from .skills.cohesive_search import allocate_cohesive_search, PARAMS as SEARCH_PARAMS
from .skills.group_geometry import cohesion_report

# Spatial group skills (v8), all OFF by default: disabled flags must leave the
# legacy action path, candidate order and telemetry semantics untouched.
SPATIAL_DEFAULTS = {
    **{k: v for k, v in REGROUP_PARAMS.items()},
    **{k: v for k, v in SEARCH_PARAMS.items()},
    "enable_regroup": False,
    "enable_cohesive_search": False,
    "enable_combat_positioning": False,
    "auto_spatial_dispatch": False,
    "dispatch_enter_share": 0.45,     # split worse than this -> consider regroup
    "dispatch_exit_share": 0.8,       # coherence above this exits regroup phase
    "dispatch_min_windows": 2,        # minimum phase duration (windows) vs flapping
    "dispatch_cooldown_windows": 4,   # failed-phase retry cooldown
    "positioning_spacing": "normal",  # compact | normal | spread
    "positioning_layout": "free",     # free | front_back
}

SPATIAL_BOOLS = ("enable_regroup", "enable_cohesive_search", "enable_combat_positioning",
                 "auto_spatial_dispatch")
SPATIAL_RATIOS = ("main_group_min_share", "regroup_entry_share", "regroup_target_share",
                  "dispatch_enter_share", "dispatch_exit_share", "search_severe_split_share")


def validate_spatial_parameters(params):
    """Strict validation: booleans must be real booleans (a string "false" is
    an error, not truthy), ratios bounded to (0, 1], counts integral and
    non-negative, distances/TTLs finite and non-negative, enums exact."""
    merged = {**SPATIAL_DEFAULTS, **(params or {})}
    for key in SPATIAL_BOOLS:
        value = merged.get(key, False)
        if not isinstance(value, bool):
            raise ValueError(f"spatial parameter {key} must be a bool, got {value!r}")
        merged[key] = value
    for key in ("positioning_spacing",):
        if merged[key] not in ("compact", "normal", "spread"):
            raise ValueError(f"spatial parameter {key} must be compact|normal|spread")
    for key in ("positioning_layout",):
        if merged[key] not in ("free", "front_back"):
            raise ValueError(f"spatial parameter {key} must be free|front_back")
    for key, value in merged.items():
        if key in SPATIAL_BOOLS or key in ("positioning_spacing", "positioning_layout"):
            continue
        if key.endswith("_windows") or key.endswith("_limit") or key.endswith("_capacity") \
                or key in ("search_memory_ttl_loops",):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"spatial parameter {key} must be an int")
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            if key in SPATIAL_RATIOS and not 0.0 < value <= 1.0:
                raise ValueError(f"spatial parameter {key} must be in (0, 1]")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            number = float(value)
            if number != number or number in (float("inf"), float("-inf")):
                raise ValueError(f"spatial parameter {key} must be finite")
            if number < 0:
                raise ValueError(f"spatial parameter {key} must be >= 0")
    if merged["dispatch_enter_share"] >= merged["dispatch_exit_share"]:
        raise ValueError("dispatch_enter_share must be below dispatch_exit_share")
    return merged


class SkillManager:
    """Tracks the currently selected control mode and one active skill run.

    One manager instance spans the whole experiment; ``reset`` must be called
    at every episode start so no state leaks across episodes.
    """

    def __init__(self, max_steps=4, risk_threshold=None, safe_streak_required=2,
                 heal_switch_margin=None, adaptive_params=None, combat_params=None,
                 unit_value_params=None, spatial_params=None, race_profile="off"):
        if race_profile not in ("off", "terran", "zerg"):
            raise ValueError("race_profile must be off, terran or zerg")
        self.race_profile = race_profile
        self.max_steps = max_steps
        self.safe_streak_required = max(1, safe_streak_required)
        # Combat positioning lives inside the combat mode: its switch rides on
        # combat_params but is validated with the other spatial flags.
        self.spatial_params = validate_spatial_parameters(
            {**(spatial_params or {}),
             **({"combat_positioning": bool(spatial_params.get("enable_combat_positioning")),
                 "positioning_spacing": spatial_params.get("positioning_spacing", "normal"),
                 "positioning_layout": spatial_params.get("positioning_layout", "free")}
                if spatial_params and spatial_params.get("enable_combat_positioning") else {})})
        self.cover_params = dict(COVER_PARAMS)
        if risk_threshold is not None:
            self.cover_params["risk_threshold"] = risk_threshold
        self.heal_params = dict(HEAL_PARAMS)
        if heal_switch_margin is not None:
            self.heal_params["switch_margin"] = heal_switch_margin
        self.adaptive_params = dict(ADAPTIVE_PARAMS)
        if adaptive_params:
            self.adaptive_params.update(adaptive_params)
        # Shared Sharpy-inspired combat mechanisms (window fire, move scoring,
        # target scoring): one source of truth for every skill.
        self.combat_params = dict(combat_params or {})
        if self.spatial_params.get("enable_combat_positioning"):
            self.combat_params.update(
                combat_positioning=True,
                positioning_spacing=self.spatial_params.get("positioning_spacing", "normal"),
                positioning_layout=self.spatial_params.get("positioning_layout", "free"))
        self.unit_value_params = validate_parameters(unit_value_params)
        # Fingerprint of every output-relevant spatial setting: cache entries
        # are valid only for the exact configuration that produced them.
        self._spatial_config_key = tuple(sorted(
            (key, value) for key, value in self.spatial_params.items()))
        self.stats = {"activations": 0, "steps": 0, "terminations": {},
                      "reassignments": 0, "estimated_overkill": 0.0,
                      "predicted_damage": 0.0, "selections": {},
                      "fallback_steps": 0, "role_reassignments": 0,
                      "safe_streak_exits": 0, "retreat_unit_steps": 0,
                      "heal_assignments_created": 0, "heal_target_switches": 0,
                      "heal_self_preserves": 0, "heal_no_target_steps": 0,
                      "adaptive_engage_steps": 0, "adaptive_retreat_steps": 0,
                      "adaptive_all_in_steps": 0, "adaptive_retreat_unit_steps": 0,
                      "adaptive_reengagements": 0, "adaptive_all_in_triggers": 0}
        self.stats.update(immediate_damage=0.0, window_only_damage=0.0,
                          movement_evaluations=0, kept_attack_positions=0,
                          adaptive_recovery_unit_steps=0,
                          coordinated_approach_unit_steps=0,
                          safe_reentry_evaluations=0,
                          critical_escape_evaluations=0, critical_escape_applied=0,
                          planned_reentry_evaluations=0, planned_reentry_applied=0,
                          engagement_evaluations=0, engagement_shot_proxies=0,
                          engagement_cycles_detected=0, engagement_retreat_blocks=0,
                          engagement_commitments_started=0, engagement_commitments_fulfilled=0,
                          engagement_commitments_timed_out=0, engagement_danger_overrides=0,
                          engagement_repeated_moves_blocked=0,
                          target_switches=0, unit_value_evaluations=0,
                          unit_value_evaluation_ms=0.0, unit_value_influenced_decisions=0)
        self.stats.update(regroup_steps=0, cohesive_search_steps=0, spatial_interrupts=0,
                          spatial_timeouts=0, spatial_dispatch_switches=0)
        self.reset()

    def reset(self):
        """Clear per-episode execution state; cumulative stats survive for the run."""
        self._race_memory = {}
        self._race_episode = None
        self.current_skill = None
        self.skill_start_step = None
        self.elapsed_steps = 0
        self.mode_remaining = 0
        self.assigned_units = []
        self.target_assignments = {}
        self.priority_target = None
        self.termination_reason = None
        self.last_observation_step = None
        self.retreat_units = []
        self.cover_units = []
        self.retreat_direction = None
        self.safe_streak = 0
        self._pending_damage = {}
        # AdaptiveFocusFire episode-scoped history: survives 4-step skill
        # windows so re-activation cannot reset retreat caps or All-in.
        self.posture = "balanced"
        self.unit_states = {}
        self.all_in = False
        self.all_in_reason = None
        self.stall_steps = 0
        self.contacted = False
        self._last_enemy_health = None
        self._last_move = {}  # per-unit previous movement direction (anti ping-pong)
        self._last_seen_enemies = {}
        self._unit_values = None  # Per-call sidecar, never a cross-snapshot cache.
        self._engagement_cached_step = None
        # Spatial skill state (v8.1). Memory/budgets/cooldowns survive skill
        # windows within an episode but never a reset. The observation clock
        # (_spatial_observed_loop) makes per-snapshot updates idempotent; time
        # running backwards clears the whole spatial store.
        self._regroup_memory = {}
        self._search_memory = {}
        self._spatial_cache = {}   # (loop, skill, actors, config) -> (partial, telemetry)
        self._dispatch_state = {"phase": "idle", "windows_in_phase": 0,
                                "cooldown_until": {REGROUP: None, COHESIVE_SEARCH: None},
                                "last_reason": None}
        self._spatial_observed_loop = None
        self._dispatch_last_loop = None
        self._episode_key = None
        self._pending_spatial_moves = None  # (loop, {actor: proposal}) awaiting commit

    # -- selection -------------------------------------------------------

    def select(self, skill, priority_target=None, posture=None):
        """JEV (or --skill-mode always) chose a mode; hold it for one window."""
        if skill not in (DIRECT_CONTROL, FOCUS_FIRE_BUDGET, COVER_RETREAT,
                         HEAL_ASSIGN, ADAPTIVE_FOCUS_FIRE, REGROUP, COHESIVE_SEARCH):
            raise ValueError(f"Unknown skill: {skill}")
        if skill == REGROUP and not self.spatial_params["enable_regroup"]:
            raise ValueError("regroup requires enable_regroup")
        if skill == COHESIVE_SEARCH and not self.spatial_params["enable_cohesive_search"]:
            raise ValueError("cohesive_search requires enable_cohesive_search")
        if self.race_profile != "off" and (
                skill != self.current_skill or (posture and posture != self.posture)):
            self._engagement_cached_step = None
        self.current_skill = skill
        self.priority_target = priority_target
        if posture:
            self.posture = posture
        self.mode_remaining = self.max_steps
        self.stats["selections"][skill] = self.stats["selections"].get(skill, 0) + 1
        if skill != DIRECT_CONTROL:
            self._start_skill()

    def _start_skill(self):
        self.skill_start_step = None
        self.elapsed_steps = 0
        self.assigned_units = []
        if self.current_skill != ADAPTIVE_FOCUS_FIRE:
            self.target_assignments = {}
        self.termination_reason = None
        self.retreat_units = []
        self.cover_units = []
        self.retreat_direction = None
        self.safe_streak = 0
        self._pending_damage = {}
        self.stats["activations"] += 1

    @property
    def active(self):
        return (self.current_skill in (FOCUS_FIRE_BUDGET, COVER_RETREAT,
                                       HEAL_ASSIGN, ADAPTIVE_FOCUS_FIRE,
                                       REGROUP, COHESIVE_SEARCH)
                and self.termination_reason is None)

    # -- execution -------------------------------------------------------

    def _record_diagnostics(self, result, telemetry):
        """Preserve allocator diagnostics in the actual skill_step event."""
        for key in ("immediate_damage", "window_only_damage", "movement_diagnostics",
                    "approach_diagnostics", "safe_reentry_diagnostics",
                    "critical_escape_diagnostics", "planned_reentry_diagnostics",
                    "damage_model", "engagement_diagnostics"):
            telemetry[key] = result.get(key, {})
        for key, parameter in (("window_fire_enabled", "window_fire"),
                               ("target_scoring_enabled", "target_scoring"),
                               ("range_preserving_move_enabled", "range_preserving_move"),
                               ("coordinated_approach_enabled", "coordinated_approach"),
                               ("safe_reentry_enabled", "safe_reentry"),
                               ("accurate_damage_enabled", "accurate_damage"),
                               ("critical_escape_enabled", "critical_escape"),
                               ("planned_reentry_enabled", "planned_reentry"),
                               ("engagement_commitment_enabled", "engagement_commitment"),
                               ("approach_progress_guard_enabled", "approach_progress_guard")):
            telemetry[key] = result.get(key, bool(self.combat_params.get(parameter)))
        for key in ("immediate_damage", "window_only_damage"):
            self.stats[key] += sum(telemetry[key].values())
        self.stats["movement_evaluations"] += len(telemetry["movement_diagnostics"])
        self.stats["coordinated_approach_unit_steps"] += len(telemetry["approach_diagnostics"])
        self.stats["safe_reentry_evaluations"] += len(telemetry["safe_reentry_diagnostics"])
        for mechanism in ("critical_escape", "planned_reentry"):
            rows = telemetry[mechanism + "_diagnostics"]
            self.stats[mechanism + "_evaluations"] += len(rows)
            self.stats[mechanism + "_applied"] += sum(bool(row.get("applied")) for row in rows.values())
        engagement_rows = telemetry['engagement_diagnostics'].values()
        self.stats['engagement_evaluations'] += len(telemetry['engagement_diagnostics'])
        for counter, field in (('shot_proxies', 'shot_proxy'), ('cycles_detected', 'cycle_detected'),
                               ('retreat_blocks', 'retreat_blocked'), ('commitments_started', 'commitment_started'),
                               ('commitments_fulfilled', 'commitment_fulfilled'), ('commitments_timed_out', 'commitment_timed_out'),
                               ('danger_overrides', 'danger_override'), ('repeated_moves_blocked', 'repeated_move_blocked')):
            self.stats['engagement_' + counter] += sum(bool(row.get(field)) for row in engagement_rows)
        self.stats["kept_attack_positions"] += sum(
            row.get("reason") == "keep_attack_position"
            for row in telemetry["movement_diagnostics"].values())
        if self._unit_values is not None:
            telemetry["unit_value_diagnostics"] = result.get("unit_value_diagnostics", {})

    def step(self, payload, n_agents):
        """Run one local decision step of the active skill on a fresh snapshot.

        Returns (actions, telemetry). ``actions`` is always a full
        n_agents-length list of legal snapshot actions; spatial skills instead
        return a partial {index: action} proposal for the caller to merge
        (the heal_assign precedent). The unified observation entry runs first:
        whatever skill — or direct control — executes, the spatial observation
        clock and shared enemy ledger advance exactly once per snapshot.
        """
        self._unit_values = None
        self._observe(payload["state"])
        elapsed_ms = 0.0
        if self.unit_value_params["mode"] != "off":
            started = perf_counter()
            self._unit_values = evaluate_unit_values(payload["state"], self.unit_value_params)
            elapsed_ms = (perf_counter() - started) * 1000.0
            self.stats["unit_value_evaluations"] += 1
            self.stats["unit_value_evaluation_ms"] += elapsed_ms
        previous_assignments = dict(self.target_assignments)
        if self.current_skill == COVER_RETREAT:
            actions, telemetry = self._step_cover_retreat(payload, n_agents)
        elif self.current_skill == HEAL_ASSIGN:
            actions, telemetry = self._step_heal_assign(payload, n_agents)
        elif self.current_skill == ADAPTIVE_FOCUS_FIRE:
            actions, telemetry = self._step_adaptive(payload, n_agents)
        elif self.current_skill == REGROUP and self.spatial_params["enable_regroup"]:
            return self._step_regroup(payload, n_agents)
        elif self.current_skill == COHESIVE_SEARCH and self.spatial_params["enable_cohesive_search"]:
            return self._step_cohesive_search(payload, n_agents)
        else:
            actions, telemetry = self._step_focus_fire(payload, n_agents)
        if self.current_skill != HEAL_ASSIGN:
            switches = sum(actor in previous_assignments and previous_assignments[actor] != target
                           for actor, target in self.target_assignments.items())
            self.stats["target_switches"] += switches
            telemetry["target_switches"] = switches
        telemetry["unit_value_mode"] = self.unit_value_params["mode"]
        telemetry["unit_value_evaluation_ms"] = elapsed_ms
        telemetry["unit_value_evaluation_calls"] = int(self._unit_values is not None)
        if self._unit_values is not None:
            # Local event only: never inject this into payload sent to a planner/model.
            telemetry["unit_value_estimate"] = self._unit_values
            telemetry["unit_value_estimate_kind"] = "heuristic tactical importance, not win contribution"
            influence = sum(row.get("influenced_decisions", int(bool(row.get("changed"))))
                            for row in telemetry.get("unit_value_diagnostics", {}).values())
            telemetry["unit_value_influenced_decisions"] = influence
            self.stats["unit_value_influenced_decisions"] += influence
        return actions, telemetry

    def _step_focus_fire(self, payload, n_agents):
        state = payload["state"]
        result = allocate_focus_fire(state, n_agents, priority_target=self.priority_target,
                                     previous_assignments=self.target_assignments or None,
                                     parameters=self.combat_params, unit_values=self._unit_values)
        reassigned = bool(self.target_assignments) and result["assignments"] != self.target_assignments
        if reassigned:
            self.stats["reassignments"] += 1
        self.last_observation_step = state["game_loop"]
        self.skill_start_step = self.skill_start_step if self.skill_start_step is not None \
            else state["game_loop"]
        self.elapsed_steps += 1
        self.assigned_units = result["participating_units"]
        self.target_assignments = result["assignments"]
        self.stats["steps"] += 1
        self.stats["estimated_overkill"] += sum(result["estimated_overkill"].values())
        self.stats["predicted_damage"] += sum(result["predicted_damage"].values())
        # Cross-step pending-damage ledgers are disabled: observations cannot
        # reliably separate landed from in-flight attacks, and subtracting a
        # stale ledger double-counted damage already visible in health.
        telemetry = {"skill": "focus_fire_budget", "skill_step": self.elapsed_steps,
                     "skill_start_step": self.skill_start_step,
                     "participating_units": result["participating_units"],
                     "visible_targets": sorted(state["visible_enemies"]),
                     "target_priority": result["target_priority"],
                     "target_assignments": result["assignments"],
                     "predicted_damage": result["predicted_damage"],
                     "pending_damage_applied": result["pending_damage_applied"],
                     "estimated_overkill": result["estimated_overkill"],
                     "budget_deficit": result["budget_deficit"],
                     "reassigned": reassigned,
                     "unknown_damage_attackers": result["unknown_damage_attackers"],
                     "unassigned_units": result["unassigned_units"],
                     "fallback_actions": result["fallback_actions"]}
        self._record_diagnostics(result, telemetry)
        if result["terminated"]:
            self._terminate(result["termination_reason"])
        elif self.elapsed_steps >= self.max_steps:
            self._terminate("max_steps")
        if self.termination_reason is not None:
            telemetry["termination_reason"] = self.termination_reason
        return result["actions"], telemetry

    def _step_cover_retreat(self, payload, n_agents):
        state = payload["state"]
        # Per-unit hysteresis: the previous step's retreating set decides who
        # is judged at the tightened continue threshold; first-step and newly
        # endangered units always use the full entry threshold.
        result = allocate_cover_retreat(state, n_agents,
                                        parameters={**self.cover_params, **self.combat_params},
                                        previous_assignments=self.target_assignments or None,
                                        retreating_units=self.retreat_units,
                                        move_history=self._last_move, unit_values=self._unit_values)
        self.last_observation_step = state["game_loop"]
        self.skill_start_step = self.skill_start_step if self.skill_start_step is not None \
            else state["game_loop"]
        self.elapsed_steps += 1
        self.stats["steps"] += 1
        self.stats["retreat_unit_steps"] += len(result["retreat_units"])
        if (self.retreat_units or self.cover_units) and \
                (result["retreat_units"] != self.retreat_units
                 or result["cover_units"] != self.cover_units):
            self.stats["role_reassignments"] += 1
        self.retreat_units = result["retreat_units"]
        self.cover_units = result["cover_units"]
        self.target_assignments = result["cover_assignments"]
        for actor in self.retreat_units:
            action = result["actions"][int(actor[1:])]
            if action in (2, 3, 4, 5):
                self._last_move[actor] = action
        if result["retreat_units"]:
            first = result["retreat_units"][0]
            action = result["actions"][int(first[1:])]
            self.retreat_direction = action if action in (2, 3, 4, 5) else None
        # Exit hysteresis: only leave after N consecutive threat-free steps,
        # so low-health-but-safe units do not oscillate retreat/attack.
        if result["safe"]:
            self.safe_streak += 1
        else:
            self.safe_streak = 0
        telemetry = {"skill": "cover_retreat", "skill_step": self.elapsed_steps,
                     "skill_start_step": self.skill_start_step,
                     "risks": result["risks"],
                     "retreat_units": result["retreat_units"],
                     "cover_units": result["cover_units"],
                     "other_units": result["other_units"],
                     "retreat_actions": result["retreat_actions"],
                     "retreat_direction": self.retreat_direction,
                     "threat_priority": result["threat_priority"],
                     "cover_assignments": result["cover_assignments"],
                     "cover_predicted_damage": result["cover_budget"],
                     "cover_estimated_overkill": result["cover_overkill"],
                     "safe_streak": self.safe_streak,
                     "safe_streak_required": self.safe_streak_required}
        self._record_diagnostics(result, telemetry)
        if result["terminated"] and result["termination_reason"] != "no_retreat_needed":
            self._terminate(result["termination_reason"])
        elif self.safe_streak >= self.safe_streak_required:
            self.stats["safe_streak_exits"] += 1
            self._terminate("safe_streak")
        elif self.elapsed_steps >= self.max_steps:
            self._terminate("max_steps")
        if self.termination_reason is not None:
            telemetry["termination_reason"] = self.termination_reason
        return result["actions"], telemetry

    def _step_heal_assign(self, payload, n_agents):
        """One HealAssign step. Returns (partial_actions, telemetry).

        ``partial_actions`` maps actor index -> action for healer slots only;
        the caller merges a declared base policy for every other unit, so no
        unit ever receives two action sources in one step.
        """
        state = payload["state"]
        result = assign_healing(state, n_agents,
                                previous_assignments=self.target_assignments or None,
                                parameters=self.heal_params, unit_values=self._unit_values)
        self.last_observation_step = state["game_loop"]
        self.skill_start_step = self.skill_start_step if self.skill_start_step is not None \
            else state["game_loop"]
        self.elapsed_steps += 1
        self.stats["steps"] += 1
        if not result["available"]:
            self.stats["heal_no_target_steps"] += 1
            telemetry = {"skill": "heal_assign", "skill_step": self.elapsed_steps,
                         "skill_start_step": self.skill_start_step,
                         "available": False}
            self._terminate(result["termination_reason"])
            telemetry["termination_reason"] = self.termination_reason
            return {}, telemetry
        switches = sum(1 for healer, previous in (self.target_assignments or {}).items()
                       if healer in result["assignments"] and result["assignments"][healer] != previous)
        self.stats["heal_assignments_created"] += len(result["assignments"])
        self.stats["heal_target_switches"] += switches
        self.stats["heal_self_preserves"] += len(result["self_preserving"])
        if not result["assignments"] and not result["self_preserving"] \
                and not result.get("approach_targets"):
            self.stats["heal_no_target_steps"] += 1
        partial = {int(actor[1:]): action for actor, action in result["healer_actions"].items()
                   if action is not None}
        self.assigned_units = sorted(result["healer_actions"], key=lambda a: int(a[1:]))
        self.target_assignments = result["assignments"]
        telemetry = {"skill": "heal_assign", "skill_step": self.elapsed_steps,
                     "skill_start_step": self.skill_start_step,
                     "available": True, "healers": result["healers"],
                     "wounded_targets": result["wounded_targets"],
                     "assignments": result["assignments"],
                     "partial_actions": partial,
                     "urgency": result["urgency"],
                     "missing_health": result["missing_health"],
                     "healer_energy": {h: state["allies"][h].get("energy") for h in result["healers"]},
                     "retained_orders": result["retained_orders"],
                     "released": result["released"],
                     "self_preserving": result["self_preserving"],
                     "unassigned_healers": result["unassigned_healers"],
                     "target_switches": switches}
        if self._unit_values is not None:
            telemetry["unit_value_diagnostics"] = result.get("unit_value_diagnostics", {})
        if result["terminated"]:
            self._terminate(result["termination_reason"])
        elif self.elapsed_steps >= self.max_steps:
            self._terminate("max_steps")
        if self.termination_reason is not None:
            telemetry["termination_reason"] = self.termination_reason
        return partial, telemetry

    def _step_adaptive(self, payload, n_agents):
        """One AdaptiveFocusFire step with persistent per-unit role history."""
        state = payload["state"]
        engagement_enabled = bool(self.combat_params.get('engagement_commitment',
                                                        self.adaptive_params.get('engagement_commitment')))
        cache_enabled = engagement_enabled or self.race_profile != "off"
        if self.race_profile != "off":
            episode = state.get("episode")
            if (self._race_episode is not None and episode is not None
                    and episode != self._race_episode):
                self._race_memory = {}
                self._engagement_cached_step = None
                self.unit_states, self._last_move, self._last_seen_enemies = {}, {}, {}
                self._last_enemy_health = None
                self.all_in, self.all_in_reason, self.stall_steps, self.contacted = False, None, 0, False
                self.target_assignments = {}
            self._race_episode = episode
        cached = self._engagement_cached_step
        if cache_enabled and cached is not None:
            if state['game_loop'] == cached[0]:
                return deepcopy(cached[1]), deepcopy(cached[2])
            if state['game_loop'] < cached[0]:
                # A new observation timeline cannot inherit an old promise or shot.
                self.unit_states, self._last_move, self._last_seen_enemies = {}, {}, {}
                self._last_enemy_health = None
                self.all_in, self.all_in_reason, self.stall_steps, self.contacted = False, None, 0, False
                self._race_memory = {}
                if self.race_profile != "off":
                    self.target_assignments = {}
        # Observable combat progress, fog-safe: compare per-enemy health only
        # over enemies visible in BOTH steps, so an enemy entering or leaving
        # the fog is never misread as damage dealt or undone.
        current_health = {enemy_id: e["health"] + e["shield"]
                          for enemy_id, e in state["visible_enemies"].items()}
        overlap = set(current_health) & set(self._last_enemy_health or {})
        damage_observed = any(current_health[e] < self._last_enemy_health[e] - 1e-6
                              for e in overlap)
        if damage_observed:
            self.stall_steps = 0
            self.contacted = True  # Observable damage proves contact.
            # Damage describes the interval that just finished, before this
            # step's role selection. Delayed hits during retreat/recovery count.
            # This is team-observed progress, not per-unit damage attribution.
            if not engagement_enabled:
                for history in self.unit_states.values():
                    history["progress_since_retreat"] = True
                    history["retreat_cycles"] = 0
        elif self.contacted and state["visible_enemies"] and overlap:
            self.stall_steps += 1
        if not self.contacted and state["visible_enemies"] and any(
                relation.get("smac_attack_available")
                for row in state.get("relations", {}).values()
                for relation in row.values()):
            self.contacted = True  # Visible enemy plus a legal attack edge = engagement.
        self._last_enemy_health = current_health
        loop = state["game_loop"]
        # Single shared enemy ledger: when any spatial feature is enabled, the
        # unified _observe() entry already refreshed it for this snapshot with
        # one TTL (the spatial one), so this legacy update — with its own,
        # shorter TTL — must not run and fight it. With all spatial features
        # off, the original v7 semantics stay byte-identical.
        if not (self._spatial_features_enabled()
                and self._spatial_observed_loop == loop):
            for enemy, unit in state["visible_enemies"].items():
                self._last_seen_enemies[enemy] = {"position": list(unit["position"]), "game_loop": loop}
            ttl = self.adaptive_params["search_memory_steps"] * state.get("step_mul", 8)
            self._last_seen_enemies = {
                e: record for e, record in self._last_seen_enemies.items()
                if 0 <= loop - record["game_loop"] <= ttl}
        search_hint = None
        if not state["visible_enemies"] and self._last_seen_enemies:
            enemy, record = max(self._last_seen_enemies.items(),
                                key=lambda item: (item[1]["game_loop"], -int(item[0][1:])))
            search_hint = {**record, "enemy": enemy, "age_loops": loop - record["game_loop"]}
            state = {**state, "search_hint": search_hint}

        allocator = allocate_adaptive_focus_fire
        race_kwargs = {}
        if self.race_profile == "terran":
            from .skills.terran_combat import allocate_terran_combat
            allocator = allocate_terran_combat
        elif self.race_profile == "zerg":
            from .skills.zerg_combat import allocate_zerg_combat
            allocator = allocate_zerg_combat
        if self.race_profile != "off":
            race_kwargs["race_memory"] = self._race_memory
        result = allocator(
            state, n_agents, unit_states=self.unit_states, all_in=self.all_in,
            stall=self.stall_steps, contacted=self.contacted, posture=self.posture,
            parameters={**self.adaptive_params, **self.combat_params},
            previous_attack_assignments=self.target_assignments or None,
            move_history=self._last_move, unit_values=self._unit_values, **race_kwargs)
        if self.race_profile != "off":
            self._race_memory = deepcopy(result.get("race_memory", {}))
            for actor in result.get("race_telemetry", {}).get("reset_actor_histories", []):
                # Discard navigation/fire promises for overridden proposals,
                # but retain bounded retreat lifecycle counters.
                if actor in self.unit_states:
                    self.unit_states[actor] = {k: v for k, v in self.unit_states[actor].items()
                        if k in ("role", "retreat_steps", "retreat_cycles", "recover_steps",
                                 "progress_since_retreat")}
                self._last_move.pop(actor, None)
        for actor in result["roles"]:
            action = result["actions"][int(actor[1:])]
            if action in (2, 3, 4, 5):
                self._last_move[actor] = action
        all_in_started_now = result["all_in"] and not self.all_in
        previous_roles = {a: v.get("role") for a, v in self.unit_states.items()}
        retreat_ended = sorted((a for a, role in result["roles"].items()
                                if role != "retreat" and previous_roles.get(a) == "retreat"),
                               key=lambda a: int(a[1:]))
        self.last_observation_step = state["game_loop"]
        self.skill_start_step = self.skill_start_step if self.skill_start_step is not None \
            else state["game_loop"]
        self.elapsed_steps += 1
        self.stats["steps"] += 1
        if result["all_in"] and not self.all_in:
            self.stats["adaptive_all_in_triggers"] += 1
        self.all_in, self.all_in_reason = result["all_in"], result["all_in_reason"]
        self.stats["adaptive_all_in_steps" if self.all_in else
                   ("adaptive_retreat_steps" if result["retreat_units"]
                    else "adaptive_engage_steps")] += 1
        self.stats["adaptive_retreat_unit_steps"] += len(result["retreat_units"])
        self.stats["adaptive_recovery_unit_steps"] += len(result["reentry_decisions"])
        self.stats["adaptive_reengagements"] += len(result["reengaged"])
        if isinstance(result.get("predicted_damage"), dict):
            self.stats["predicted_damage"] += sum(result["predicted_damage"].values())
        if isinstance(result.get("estimated_overkill"), dict):
            self.stats["estimated_overkill"] += sum(result["estimated_overkill"].values())

        # Per-unit role history. A retreat episode only counts toward the
        # unproductive-cycle tally when the unit's intervening attack stint
        # produced no observable enemy damage: normal dodge -> re-engage ->
        # damage cycles are free, repeated fruitless retreats are not.
        for actor, role in result["roles"].items():
            unit_state = self.unit_states.setdefault(
                actor, {"role": "attack", "retreat_steps": 0, "retreat_cycles": 0})
            if role == "retreat":
                if unit_state.get("role") != "retreat":
                    if not unit_state.pop("progress_since_retreat", False):
                        unit_state["retreat_cycles"] += 1
                    unit_state["retreat_steps"] = 1
                else:
                    unit_state["retreat_steps"] += 1
            elif role == "recover":
                unit_state["recover_steps"] = unit_state.get("recover_steps", 0) + 1
            else:
                if unit_state.get("role") == "retreat":
                    if not self.combat_params.get("range_preserving_move"):
                        unit_state["retreat_steps"] = 0
                if role in ("attack", "all_in_attack"):
                    unit_state["retreat_steps"] = 0
                    unit_state["recover_steps"] = 0
            unit_state["role"] = role
        for actor, update in result.get("approach_state_updates", {}).items():
            # Navigation history shares the episode lifetime of role history.
            # It must survive a new four-step activation but never a reset.
            if actor in self.unit_states:
                self.unit_states[actor].update(update)
        for actor, update in result.get('engagement_state_updates', {}).items():
            if actor in self.unit_states:
                self.unit_states[actor].update(update)
        self.unit_states = {a: v for a, v in self.unit_states.items()
                            if a in state["allies"] and state["allies"][a]["alive"]}
        self.target_assignments = result["attack_assignments"]

        telemetry = {"skill": "adaptive_focus_fire", "skill_step": self.elapsed_steps,
                     "skill_start_step": self.skill_start_step,
                     "posture": self.posture, "phase": result["phase"],
                     "all_in": self.all_in, "all_in_reason": self.all_in_reason,
                     "attack_units": result["attack_units"],
                     "retreat_units": result["retreat_units"],
                     "support_units": result["support_units"],
                     "roles": result["roles"],
                     "retreat_reasons": result["retreat_reasons"],
                     "retreat_actions": result["retreat_actions"],
                     "reengaged": result["reengaged"],
                     "attack_assignments": result["attack_assignments"],
                     "predicted_damage": result["predicted_damage"],
                     "estimated_overkill": result["estimated_overkill"],
                     "stall_steps": self.stall_steps,
                     "all_in_started_now": result["all_in_reason"] if all_in_started_now else None,
                     "retreat_ended": retreat_ended,
                     "retreat_state": {a: dict(self.unit_states.get(a, {}))
                                       for a in result["retreat_units"]},
                     "reentry_decisions": result["reentry_decisions"],
                     "search_hint": search_hint,
                     "observed_team_damage_progress": damage_observed,
                     "progress_basis": "visible enemy health decrease in the preceding interval"}
        self._record_diagnostics(result, telemetry)
        if self.race_profile != "off":
            telemetry["race_profile"] = self.race_profile
            telemetry["race_telemetry"] = result.get("race_telemetry", {})
        if result["terminated"]:
            self._terminate(result["termination_reason"])
        elif self.elapsed_steps >= self.max_steps:
            self._terminate("max_steps")
        if self.termination_reason is not None:
            telemetry["termination_reason"] = self.termination_reason
        if cache_enabled:
            self._engagement_cached_step = (state['game_loop'], deepcopy(result['actions']), deepcopy(telemetry))
        return result["actions"], telemetry

    # -- spatial group skills (v8.1) ----------------------------------------
    # Pipeline: (1) _observe() updates observation facts once per unique
    # snapshot, whatever skill (or direct control) is running; (2) allocators
    # propose partial actions; (3) the runner merges and mask-validates; (4)
    # commit_final_actions() records what actually executed. Execution history
    # is never written from an overwritten or rejected proposal.

    def _spatial_features_enabled(self):
        return bool(self.spatial_params["enable_regroup"]
                    or self.spatial_params["enable_cohesive_search"]
                    or self.spatial_params["enable_combat_positioning"]
                    or self.spatial_params["auto_spatial_dispatch"])

    def _clear_spatial_memory(self):
        """Episode change / time regression: budgets, waypoints, failure
        directions, position assignments, caches and cooldowns all go."""
        self._regroup_memory = {}
        self._search_memory = {}
        self._spatial_cache = {}
        self._dispatch_state = {"phase": "idle", "windows_in_phase": 0,
                                "cooldown_until": {REGROUP: None, COHESIVE_SEARCH: None},
                                "last_reason": "cleared"}
        self._pending_spatial_moves = None

    def _update_enemy_ledger(self, state, ttl_loops):
        """Single shared last-seen-enemy ledger (also read by _step_adaptive).

        Old positions are search hints only, never attack targets; leaving
        vision is not death — only the TTL or a fresh observation removes or
        replaces a record.
        """
        loop = state["game_loop"]
        for enemy, unit in state["visible_enemies"].items():
            self._last_seen_enemies[enemy] = {"position": list(unit["position"]),
                                              "game_loop": loop}
        self._last_seen_enemies = {
            e: record for e, record in self._last_seen_enemies.items()
            if 0 <= loop - record["game_loop"] <= ttl_loops}

    def _observe(self, state):
        """Unified observation entry: runs on every unique decision snapshot
        while any spatial feature is enabled (no-op otherwise, so the legacy
        path is byte-identical when the flags are off). Never issues actions,
        so direct control is untouched."""
        if not self._spatial_features_enabled():
            return
        episode = state.get("episode")
        if self._episode_key is not None and episode is not None and episode != self._episode_key:
            self._clear_spatial_memory()  # new episode: never inherit budgets
        if episode is not None:
            self._episode_key = episode
        loop = state["game_loop"]
        if self._spatial_observed_loop is not None:
            if loop == self._spatial_observed_loop:
                return  # same snapshot: updates stay idempotent
            if loop < self._spatial_observed_loop:
                # Observation clock went backwards: a new timeline cannot
                # inherit the old episode's spatial state.
                self._clear_spatial_memory()
        self._update_enemy_ledger(state, self.spatial_params["search_memory_ttl_loops"])
        self._spatial_observed_loop = loop

    def _spatial_partial(self, result):
        """Actor-keyed partial proposal -> index-keyed partial for the caller.

        Only units with an emitted action are taken over; everyone else keeps
        the caller's declared base policy. The final action array stays
        n_agents long with the environment's index mapping.
        """
        return {int(actor[1:]): action for actor, action in result["actions"].items()}

    def _cooldown_active(self, mode, loop):
        until = self._dispatch_state["cooldown_until"].get(mode)
        return until is not None and loop < until

    def _set_cooldown(self, mode, loop, windows):
        step_mul = getattr(self, "_last_step_mul", None) or 8
        self._dispatch_state["cooldown_until"][mode] = loop + windows * step_mul

    def _flush_pending_moves(self):
        """If the runner never called commit_final_actions for the previous
        snapshot, the proposal is the best available record of what ran; it is
        applied once, never twice (documented fallback for unwired paths)."""
        if self._pending_spatial_moves is not None:
            _, proposals = self._pending_spatial_moves
            for actor, action in proposals.items():
                if action in (2, 3, 4, 5):
                    self._last_move[actor] = action
            self._pending_spatial_moves = None

    def _cached_spatial_step(self, skill, loop, actors_key, compute):
        """Snapshot+config-bound cache. The same snapshot, skill, controlled
        set and configuration always yields the same outcome, returned as a
        copy; a changed intent (different skill/actors) or configuration
        computes fresh. Observation facts were already updated once."""
        key = (loop, skill, actors_key, self._spatial_config_key)
        if key in self._spatial_cache:
            partial, telemetry = self._spatial_cache[key]
            return deepcopy(partial), deepcopy(telemetry)
        self._flush_pending_moves()
        partial, telemetry = compute()
        self._spatial_cache[key] = (deepcopy(partial), deepcopy(telemetry))
        if len(self._spatial_cache) > 8:  # bounded: recent snapshots only
            self._spatial_cache = dict(sorted(self._spatial_cache.items(),
                                              key=lambda item: item[0][0])[-8:])
        return partial, telemetry

    def _spatial_preamble(self, payload, n_agents):
        state = payload["state"]
        self._last_step_mul = state.get("step_mul", 8)
        self._observe(state)
        self._flush_pending_moves()
        controlled = [a for a, u in sorted(state["allies"].items())
                      if u.get("alive") and int(a[1:]) < n_agents]
        self.last_observation_step = state["game_loop"]
        self.skill_start_step = self.skill_start_step if self.skill_start_step is not None \
            else state["game_loop"]
        # Per-snapshot bookkeeping happens in compute(): a cached (repeated)
        # snapshot must not re-charge steps, budgets or statistics.
        return state, controlled

    def _charge_spatial_step(self):
        self.elapsed_steps += 1
        self.stats["steps"] += 1

    def _finish_spatial_step(self, result, telemetry, loop):
        """Shared termination bookkeeping; interrupted/blocked/timeout are
        never reported as success."""
        if result["status"] == "interrupted":
            self.stats["spatial_interrupts"] += 1
            self._terminate("interrupted_" + result["reason"])
        elif result["status"] == "timeout":
            self.stats["spatial_timeouts"] += 1
            self._terminate(f"{telemetry['skill']}_timeout")
        elif result["status"] == "blocked":
            self._terminate(f"{telemetry['skill']}_{result['reason']}")
        elif result["status"] == "success":
            self._terminate(f"{telemetry['skill']}_complete")
        elif self.elapsed_steps >= self.max_steps:
            self._terminate("max_steps")
        if self.termination_reason is not None:
            telemetry["termination_reason"] = self.termination_reason
        return self._spatial_partial(result), telemetry

    def _step_regroup(self, payload, n_agents):
        state, controlled = self._spatial_preamble(payload, n_agents)
        loop = state["game_loop"]

        def compute():
            self._charge_spatial_step()
            result = allocate_regroup(state, controlled, self._regroup_memory,
                                      self.spatial_params, self._last_move)
            self.stats["regroup_steps"] += 1
            self._regroup_memory = {**self._regroup_memory, **result["state_updates"]}
            if result["state_updates"].get("phase") == "cooldown":
                self._set_cooldown(REGROUP, loop,
                                   self.spatial_params["regroup_retry_cooldown_windows"])
            telemetry = {"skill": "regroup", "skill_step": self.elapsed_steps,
                         "skill_start_step": self.skill_start_step,
                         "status": result["status"], "reason": result["reason"],
                         "requested_units": result.get("requested_units", controlled),
                         "controlled_units": result["controlled_units"],
                         "partial_actions": self._spatial_partial(result),
                         "observed_cohesion": result["diagnostics"].get("observed_cohesion"),
                         "controlled_cohesion": result["diagnostics"].get("controlled_cohesion"),
                         "anchor": result["diagnostics"].get("anchor"),
                         "blocked_stragglers": result["diagnostics"].get("blocked_stragglers", {}),
                         "regroup_budget_windows": self._regroup_memory.get("elapsed_windows", 0)}
            self._pending_spatial_moves = (loop, dict(result["actions"]))
            return self._finish_spatial_step(result, telemetry, loop)

        return self._cached_spatial_step(REGROUP, loop, tuple(controlled), compute)

    def _step_cohesive_search(self, payload, n_agents):
        state, controlled = self._spatial_preamble(payload, n_agents)
        loop = state["game_loop"]

        def compute():
            self._charge_spatial_step()
            derived = state
            hint_source = None
            if not state["visible_enemies"] and "search_hint" not in state \
                    and "search_locations" not in state and self._last_seen_enemies:
                # The shared ledger becomes the search lead: position,
                # observation loop, age and source travel WITH the call (a
                # runner-provided hint keeps priority inside the allocator).
                enemy, record = max(self._last_seen_enemies.items(),
                                    key=lambda item: (item[1]["game_loop"],
                                                      -int(item[0][1:])))
                hint_source = "last_seen_enemy"
                derived = {**state, "search_hint": {
                    "position": record["position"], "game_loop": record["game_loop"],
                    "age_loops": loop - record["game_loop"], "source": hint_source,
                    "generation": record["game_loop"], "enemy": enemy}}
            result = allocate_cohesive_search(derived, controlled, self._search_memory,
                                              self.spatial_params, self._last_move)
            self.stats["cohesive_search_steps"] += 1
            self._search_memory = {**self._search_memory, **result["state_updates"]}
            if result["state_updates"].get("phase") == "cooldown":
                self._set_cooldown(COHESIVE_SEARCH, loop,
                                   self.spatial_params["search_waypoint_ttl_windows"])
            if result["state_updates"].get("phase") == "needs_regroup":
                # A search verdict of "severe split" must actually influence
                # the next dispatch, not just terminate this window.
                self._dispatch_state["phase"] = "needs_regroup"
                self._dispatch_state["last_reason"] = "search_needs_regroup"
            telemetry = {"skill": "cohesive_search", "skill_step": self.elapsed_steps,
                         "skill_start_step": self.skill_start_step,
                         "status": result["status"], "reason": result["reason"],
                         "requested_units": result.get("requested_units", controlled),
                         "controlled_units": result["controlled_units"],
                         "partial_actions": self._spatial_partial(result),
                         "observed_cohesion": result["diagnostics"].get("observed_cohesion"),
                         "controlled_cohesion": result["diagnostics"].get("controlled_cohesion"),
                         "waypoint": result["diagnostics"].get("waypoint"),
                         "waypoint_source": result["diagnostics"].get("waypoint_source"),
                         "degenerate_mode": result["diagnostics"].get("degenerate", False),
                         "visited_waypoints": self._search_memory.get("visited_waypoints", []),
                         "search_budget_windows": self._search_memory.get("elapsed_windows", 0)}
            if hint_source:
                telemetry["injected_search_hint"] = hint_source
            self._pending_spatial_moves = (loop, dict(result["actions"]))
            return self._finish_spatial_step(result, telemetry, loop)

        return self._cached_spatial_step(COHESIVE_SEARCH, loop, tuple(controlled), compute)

    def commit_final_actions(self, final_actions):
        """Record what actually executed after merge + mask validation.

        Only the pending snapshot's controlled actors are recorded, from the
        FINAL array: a proposal that was overwritten or rejected never enters
        the execution history. Idempotent per snapshot.
        """
        if self._pending_spatial_moves is None:
            return False
        _, proposals = self._pending_spatial_moves
        final_moves = self._regroup_memory.get("last_final_moves")
        for actor in proposals:
            index = int(actor[1:])
            if 0 <= index < len(final_actions):
                action = final_actions[index]
                if action in (2, 3, 4, 5):
                    self._last_move[actor] = action
                if final_moves is not None and actor in final_moves:
                    self._regroup_memory["last_final_moves"][actor] = action
        self._pending_spatial_moves = None
        return True

    def spatial_dispatch(self, payload):
        """Explicit rule scheduler for local no-model running (auto dispatch).

        Returns REGROUP / COHESIVE_SEARCH / None (None = existing combat
        controller keeps control). Enter/exit hysteresis, minimum phase length
        and absolute-loop failure cooldowns prevent regroup<->search flapping;
        contact and observed damage interrupt immediately regardless of the
        minimum duration. Judgement uses share AND compactness: a long chain
        is one component but is not "concentrated".
        """
        if not self.spatial_params["auto_spatial_dispatch"]:
            return None
        state = payload["state"]
        self._last_step_mul = state.get("step_mul", 8)
        self._observe(state)
        loop = state["game_loop"]
        dispatch = self._dispatch_state
        contact = bool(state["visible_enemies"]) or any(
            relation.get("smac_attack_available")
            for row in state.get("relations", {}).values()
            for relation in row.values())
        damaged = any((u.get("health_lost_since_previous_decision", 0)
                       + u.get("shield_lost_since_previous_decision", 0)) > 0
                      for u in state["allies"].values() if u.get("alive"))
        if contact or damaged:
            if dispatch["phase"] in (REGROUP, COHESIVE_SEARCH, "needs_regroup"):
                self.stats["spatial_dispatch_switches"] += 1
            dispatch.update(phase="combat", windows_in_phase=0,
                            last_reason="contact_or_damage")
            self._dispatch_last_loop = loop
            return None
        report = cohesion_report(state, self.spatial_params)
        share = report["main_share"]
        radius = report["main_compactness"]["quantile_radius"]
        compact = radius is not None and radius <= self.spatial_params["target_radius"]
        minimum = self.spatial_params["dispatch_min_windows"]

        def advance_window(default_phase):
            if self._dispatch_last_loop != loop:
                dispatch["windows_in_phase"] += 1
                self._dispatch_last_loop = loop
            return default_phase

        if dispatch["phase"] == REGROUP:
            if (share >= self.spatial_params["dispatch_exit_share"]
                    and radius is not None
                    and radius <= self.spatial_params["regroup_target_radius"]
                    and not report["isolated"]):
                dispatch.update(phase="idle", windows_in_phase=0, last_reason="regrouped")
                self._dispatch_last_loop = loop
            elif dispatch["windows_in_phase"] < minimum:
                return advance_window(REGROUP)
        elif dispatch["phase"] == COHESIVE_SEARCH:
            if dispatch["windows_in_phase"] < minimum:
                return advance_window(COHESIVE_SEARCH)
        split = (len(report["groups"]) > 1
                 and (share < self.spatial_params["dispatch_exit_share"]
                      or report["isolated"]
                      or len(report["outliers"]) >= len(report["main_group"])))
        loose_chain = radius is not None and radius > self.spatial_params["target_radius"]
        if split or (loose_chain and not compact):
            if self.spatial_params["enable_regroup"] and not self._cooldown_active(REGROUP, loop):
                if dispatch["phase"] != REGROUP:
                    self.stats["spatial_dispatch_switches"] += 1
                dispatch.update(phase=REGROUP, windows_in_phase=1,
                                last_reason="severe_split" if split else "not_compact")
                self._dispatch_last_loop = loop
                return REGROUP
            dispatch.update(last_reason="regroup_unavailable_or_cooling")
            self._dispatch_last_loop = loop
            return None  # explicit degradation: never loop on an unusable mode
        if (self.spatial_params["enable_cohesive_search"]
                and not self._cooldown_active(COHESIVE_SEARCH, loop)
                and not state["visible_enemies"]):
            if dispatch["phase"] != COHESIVE_SEARCH:
                self.stats["spatial_dispatch_switches"] += 1
            dispatch.update(phase=COHESIVE_SEARCH, windows_in_phase=1,
                            last_reason="no_contact_advance")
            self._dispatch_last_loop = loop
            return COHESIVE_SEARCH
        self._dispatch_last_loop = loop
        return None

    def _terminate(self, reason):
        self.termination_reason = reason
        # A genuinely terminated skill must hand control back at the next
        # decision boundary instead of keeping a stale mode window.
        if self.current_skill != DIRECT_CONTROL:
            self.mode_remaining = 0
        self.stats["terminations"][reason] = self.stats["terminations"].get(reason, 0) + 1

    def spend_mode_step(self):
        """Consume one decision step of the current mode window."""
        if self.mode_remaining > 0:
            self.mode_remaining -= 1

    def record_fallback(self):
        self.stats["fallback_steps"] += 1
