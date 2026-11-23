"""Versioned feature schema and deterministic extractors for Zerg skills.

Four fixed layers:

- unit:      per-actor features (each unit type uses its own subset)
- team:      whole-force features shared by every candidate of a decision
- skill:     skill-lifecycle features shared by every candidate
- candidate: per-candidate features (descriptions of the proposal itself,
             never of its realised outcome)

Hard constraints enforced here:

- no future information is read (snapshot + committed memory only);
- episode ids, seeds and unit tags are association keys, never features;
- missing values are NaN, never folded into 0.0;
- extraction is read-only: Snapshot/Context/Memory are never mutated;
- the same state always yields the same feature vector (sorted iteration,
  no wall-clock, no RNG).

Renaming or removing a feature requires bumping SCHEMA_VERSION; adding a
feature requires appending it in a new minor-compatible revision and bumping
the version too — the digest below is part of every model fingerprint.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

SCHEMA_NAME = "zerg_tactical_features"
SCHEMA_VERSION = 1

MISSING = float("nan")

UNIT_TYPE_ALIASES: Dict[str, Tuple[str, ...]] = {
    # canonical -> observed SMAC type_name spellings
    "zergling": ("zergling",),
    "hydralisk": ("hydralisk",),
    "baneling": ("baneling",),
    "ultralisk": ("ultralisk",),
}

# Ordered vocabularies for discrete fields. Index encoding: unknown/absent
# maps to -1, never silently to index 0.
SKILL_VOCABULARY: Tuple[str, ...] = (
    "focus_fire", "coordinated_engage", "combat_reposition",
    "disengage_reenter", "regroup", "local_formation", "cohesive_search",
    "precombat_rally", "combat_rejoin", "ranged_fire_formation",
    "splash_fire", "heal_support", "burst_engage", "baseline")

PHASE_VOCABULARY: Tuple[str, ...] = (
    "approach", "engage", "fire", "disengage", "reposition", "regroup",
    "search", "rally", "rejoin", "hold", "support", "unknown")

GOAL_KIND_VOCABULARY: Tuple[str, ...] = (
    "enemy", "ally", "position", "group_link", "meeting", "internal")

DUTY_VOCABULARY: Tuple[str, ...] = (
    "attack", "move", "rescue", "intercept", "heal", "stop", "unknown")


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    layer: str                       # unit | team | skill | candidate
    unit_types: Tuple[str, ...] = ()  # unit layer only; empty = all types
    allow_missing: bool = True
    description: str = ""


def _u(name: str, unit_types: Sequence[str], allow_missing: bool = True,
       description: str = "") -> FeatureSpec:
    return FeatureSpec(name=name, layer="unit",
                       unit_types=tuple(unit_types),
                       allow_missing=allow_missing, description=description)


UNIT_FEATURES: Tuple[FeatureSpec, ...] = (
    _u("health_ratio", ("zergling", "hydralisk", "baneling"),
       False, "health / max_health"),
    _u("recent_health_loss", ("zergling", "hydralisk", "baneling"),
       False, "health lost since the previous decision step"),
    _u("shield_ratio", ("zergling", "hydralisk", "baneling"),
       False, "shield / max_shield; 0 when the type has no shield"),
    _u("weapon_ready", ("zergling", "hydralisk", "baneling"),
       False, "1.0 when the weapon can fire this step"),
    _u("weapon_cooldown", ("zergling", "hydralisk", "baneling"),
       False, "remaining weapon cooldown in loops"),
    _u("nearest_enemy_distance", ("zergling", "hydralisk", "baneling"),
       True, "distance to the nearest visible enemy (world units)"),
    _u("nearest_ally_distance", ("zergling", "hydralisk", "baneling"),
       True, "distance to the nearest other alive ally"),
    _u("attackable_enemy_count", ("zergling", "hydralisk", "baneling"),
       False, "visible enemies this actor may attack right now"),
    _u("enemy_threat_count", ("zergling", "hydralisk", "baneling"),
       True, "enemies whose weapon currently threatens this actor"),
    _u("predicted_incoming_damage", ("zergling", "hydralisk", "baneling"),
       True, "context risk estimate of incoming damage this window"),
    _u("retreat_quality", ("zergling", "hydralisk", "baneling"),
       True, "quality of the available withdrawal route (skill-sourced)"),
    _u("support_distance", ("zergling", "hydralisk", "baneling"),
       True, "distance to nearest ranged/ready support ally"),
    _u("target_remaining_health", ("zergling", "hydralisk"),
       True, "effective health of the actor's current target, if any"),
    _u("target_threat", ("zergling", "hydralisk"),
       True, "threat the actor's current target poses to this actor"),
    _u("movement_opportunity", ("zergling", "hydralisk", "baneling"),
       False, "number of legal movement actions"),
    _u("fire_opportunity", ("zergling", "hydralisk", "baneling"),
       False, "number of legal attack actions"),
)

TEAM_FEATURES: Tuple[FeatureSpec, ...] = (
    FeatureSpec("ally_alive_count", "team", (), False),
    FeatureSpec("enemy_visible_count", "team", (), False),
    FeatureSpec("ally_effective_health", "team", (), False),
    FeatureSpec("enemy_effective_health", "team", (), False),
    FeatureSpec("local_ally_count", "team", (), False,
                "alive allies within the local radius of the force centroid"),
    FeatureSpec("local_enemy_count", "team", (), False),
    FeatureSpec("local_force_ratio", "team", (), False,
                "local effective-health ratio, allies / enemies"),
    FeatureSpec("team_dispersion", "team", (), False,
                "mean pairwise ally distance"),
    FeatureSpec("frontline_support", "team", (), False,
                "ready melee attackers adjacent to the nearest enemy"),
    FeatureSpec("ranged_support", "team", (), False,
                "ready ranged attackers with a target in range"),
    FeatureSpec("combat_pressure", "team", (), False,
                "total threat edges onto alive allies"),
    FeatureSpec("recent_kills", "team", (), True,
                "kills recorded by committed memory this episode"),
    FeatureSpec("recent_losses", "team", (), True,
                "ally deaths recorded by committed memory this episode"),
)

SKILL_FEATURES: Tuple[FeatureSpec, ...] = (
    FeatureSpec("current_skill", "skill", (), False,
                "index into SKILL_VOCABULARY, -1 unknown/none"),
    FeatureSpec("previous_skill", "skill", (), False,
                "index into SKILL_VOCABULARY, -1 unknown/none"),
    FeatureSpec("skill_duration", "skill", (), True,
                "loops the current skill instance has run"),
    FeatureSpec("skill_switch_cost", "skill", (), True,
                "remaining minimum-loops commitment of the current instance"),
    FeatureSpec("current_commitment_phase", "skill", (), False,
                "index into PHASE_VOCABULARY, -1 unknown"),
    FeatureSpec("task_progress", "skill", (), True,
                "0..1 progress signal of the active task, memory-sourced"),
    FeatureSpec("task_staleness", "skill", (), True,
                "loops since the last recorded progress"),
    FeatureSpec("previous_skill_success", "skill", (), True,
                "1.0/0.0 outcome flag of the previous instance"),
    FeatureSpec("repeated_failure_count", "skill", (), True,
                "consecutive failed retries of the current task"),
)

CANDIDATE_FEATURES: Tuple[FeatureSpec, ...] = (
    FeatureSpec("candidate_skill", "candidate", (), False,
                "index into SKILL_VOCABULARY"),
    FeatureSpec("candidate_phase", "candidate", (), False,
                "index into PHASE_VOCABULARY"),
    FeatureSpec("candidate_actor_count", "candidate", (), False),
    FeatureSpec("candidate_action_change_count", "candidate", (), False,
                "controlled actors whose proposed action differs from the "
                "last executed action"),
    FeatureSpec("candidate_target_type", "candidate", (), False,
                "index into GOAL_KIND_VOCABULARY"),
    FeatureSpec("candidate_predicted_damage", "candidate", (), False,
                "declared expected damage effect total (declared, not "
                "realised)"),
    FeatureSpec("candidate_predicted_loss", "candidate", (), False,
                "declared expected loss effect total"),
    FeatureSpec("candidate_risk", "candidate", (), True,
                "fraction of controlled actors inside enemy weapon range"),
    FeatureSpec("candidate_support_gain", "candidate", (), True,
                "reserved; support-network gain estimate"),
    FeatureSpec("candidate_progress", "candidate", (), True,
                "declared task progress signal of the plan"),
    FeatureSpec("candidate_resource_conflict", "candidate", (), False,
                "1.0 when the plan's claims collide with a peer plan at the "
                "same decision point"),
    FeatureSpec("candidate_execution_duty", "candidate", (), True,
                "index into DUTY_VOCABULARY of the dominant duty"),
)

ALL_SPECS: Tuple[FeatureSpec, ...] = (UNIT_FEATURES + TEAM_FEATURES
                                      + SKILL_FEATURES + CANDIDATE_FEATURES)


def encode_index(value: Any, vocabulary: Sequence[str]) -> int:
    """Explicit discrete encoding: index into a fixed vocabulary, -1 when
    the value is absent or unknown. Never falls back to index 0."""
    if value is None:
        return -1
    text = str(value)
    try:
        return vocabulary.index(text)
    except ValueError:
        return -1


def _num(value: Any) -> float:
    if value is None:
        return MISSING
    try:
        result = float(value)
    except (TypeError, ValueError):
        return MISSING
    if math.isnan(result) or math.isinf(result):
        return MISSING
    return result


def _bool(value: Any) -> float:
    if value is None:
        return MISSING
    return 1.0 if value else 0.0


def _sorted_actors(allies: Mapping[str, Any]) -> List[str]:
    """Sort 'u12' style ids by numeric index — stable and independent of the
    mapping's insertion order (determinism requirement)."""
    def key(actor: str) -> Tuple[int, str]:
        suffix = actor[1:] if actor.startswith("u") else actor
        try:
            return (int(suffix), actor)
        except ValueError:
            return (10**9, actor)
    return sorted((a for a in allies), key=key)


class ZergFeatureSchema:
    """Immutable registry of the fixed feature space."""

    def __init__(self, name: str = SCHEMA_NAME, version: int = SCHEMA_VERSION):
        self.name = name
        self.version = version
        self._specs: Dict[str, FeatureSpec] = {s.name: s for s in ALL_SPECS}

    # -- lookups ------------------------------------------------------------
    def spec(self, name: str) -> FeatureSpec:
        return self._specs[name]

    def names(self, layer: Optional[str] = None,
              unit_type: Optional[str] = None) -> Tuple[str, ...]:
        picked = []
        for item in ALL_SPECS:
            if layer is not None and item.layer != layer:
                continue
            if unit_type is not None and item.layer == "unit" \
                    and unit_type not in item.unit_types:
                continue
            picked.append(item.name)
        return tuple(picked)

    def unit_types(self) -> Tuple[str, ...]:
        covered: Tuple[str, ...] = ()
        for item in UNIT_FEATURES:
            for unit_type in item.unit_types:
                if unit_type not in covered:
                    covered += (unit_type,)
        return covered

    # -- validation ---------------------------------------------------------
    def validate(self, layer: str, features: Mapping[str, float],
                 unit_type: Optional[str] = None) -> List[str]:
        problems = []
        allowed = set(self.names(layer=layer, unit_type=unit_type))
        unknown = sorted(set(features) - allowed)
        if unknown:
            problems.append(f"unknown {layer} features: {unknown}")
        for name in features:
            item = self._specs.get(name)
            if item is None:
                continue
            value = features[name]
            if isinstance(value, float) and math.isnan(value) \
                    and not item.allow_missing:
                problems.append(f"{name} is missing but required")
        return problems

    # -- identity -----------------------------------------------------------
    def digest(self) -> str:
        payload = {
            "name": self.name, "version": self.version,
            "specs": [
                {"name": s.name, "layer": s.layer,
                 "unit_types": list(s.unit_types),
                 "allow_missing": s.allow_missing}
                for s in ALL_SPECS],
            "vocabularies": {
                "skill": list(SKILL_VOCABULARY),
                "phase": list(PHASE_VOCABULARY),
                "goal_kind": list(GOAL_KIND_VOCABULARY),
                "duty": list(DUTY_VOCABULARY)},
        }
        blob = repr(sorted(payload.items(), key=str)).encode()
        return hashlib.sha256(blob).hexdigest()[:16]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name, "version": self.version, "digest": self.digest(),
            "unit_types": list(self.unit_types()),
            "layers": {
                layer: list(self.names(layer=layer))
                for layer in ("unit", "team", "skill", "candidate")},
            "unit_feature_subsets": {
                unit_type: list(self.names(layer="unit", unit_type=unit_type))
                for unit_type in self.unit_types()},
            "vocabularies": {
                "skill": list(SKILL_VOCABULARY),
                "phase": list(PHASE_VOCABULARY),
                "goal_kind": list(GOAL_KIND_VOCABULARY),
                "duty": list(DUTY_VOCABULARY)},
        }


DEFAULT_SCHEMA = ZergFeatureSchema()


class StateFeatureExtractor:
    """Read-only extraction of unit/team/skill features from one snapshot.

    ``context`` and ``memory`` are the tactical_v1 objects; everything is
    accessed defensively so the extractor also works on plain test doubles
    that expose the Snapshot contract. Missing capabilities yield MISSING,
    never a fabricated 0.
    """

    def __init__(self, schema: ZergFeatureSchema = DEFAULT_SCHEMA,
                 local_radius: float = 6.0):
        self.schema = schema
        self.local_radius = float(local_radius)

    # -- unit layer ---------------------------------------------------------
    def unit_type_of(self, actor: str, snapshot) -> Optional[str]:
        unit = snapshot.allies.get(actor)
        if unit is None:
            return None
        type_name = str(getattr(unit, "type_name", "") or "").lower()
        for canonical, aliases in UNIT_TYPE_ALIASES.items():
            if type_name in aliases:
                return canonical
        return None

    def extract_unit_features(self, actor: str, snapshot, context=None,
                              plan=None) -> Dict[str, float]:
        unit = snapshot.allies.get(actor)
        if unit is None:
            return {}
        relations = snapshot.relations.get(actor, {}) or {}
        enemies = snapshot.enemies
        distances = [row.distance for row in relations.values()
                     if getattr(row, "distance", None) is not None]
        attackable = sum(1 for row in relations.values()
                         if getattr(row, "smac_attack_available", False))
        ally_positions = [(peer, other.position)
                          for peer, other in snapshot.allies.items()
                          if peer != actor and getattr(other, "alive", False)
                          and other.position is not None]
        if unit.position is not None and ally_positions:
            nearest_ally = min(
                math.dist(unit.position, pos) for _, pos in ally_positions)
        else:
            nearest_ally = MISSING
        threats, incoming = self._risk_fields(actor, unit, context)
        support = self._support_distance(actor, unit, snapshot)
        target_health, target_threat = self._target_fields(
            actor, unit, relations, plan, enemies)
        movement = len(snapshot.movement_candidates.get(actor, {}) or {})
        movement = float(movement) if movement else float(
            self._legal_move_count(actor, context))
        features = {
            "health_ratio": _num(getattr(unit, "health_fraction", None)),
            "recent_health_loss": _num(getattr(unit, "health_lost", None)),
            "shield_ratio": self._shield_ratio(unit),
            "weapon_ready": _bool(getattr(unit, "weapon_ready", None)),
            "weapon_cooldown": _num(getattr(unit, "weapon_cooldown_loops", None)),
            "nearest_enemy_distance": _num(min(distances) if distances else None),
            "nearest_ally_distance": _num(nearest_ally),
            "attackable_enemy_count": float(attackable),
            "enemy_threat_count": _num(threats),
            "predicted_incoming_damage": _num(incoming),
            "retreat_quality": self._retreat_quality(actor, context),
            "support_distance": _num(support),
            "target_remaining_health": _num(target_health),
            "target_threat": _num(target_threat),
            "movement_opportunity": _num(movement),
            "fire_opportunity": float(attackable),
        }
        unit_type = self.unit_type_of(actor, snapshot)
        if unit_type:
            allowed = self.schema.names(layer="unit", unit_type=unit_type)
            return {name: features[name] for name in allowed}
        return {name: value for name, value in features.items()
                if value == value}  # drop MISSING for unmapped types

    @staticmethod
    def _shield_ratio(unit) -> float:
        max_shield = getattr(unit, "max_shield", 0.0) or 0.0
        if max_shield <= 0:
            return 0.0
        return float((getattr(unit, "shield", 0.0) or 0.0) / max_shield)

    @staticmethod
    def _risk_fields(actor: str, unit, context) -> Tuple[Any, Any]:
        if context is None:
            return MISSING, MISSING
        risk = None
        try:
            risk = context.risk_at(actor, unit.position)
        except Exception:
            risk = None
        if risk is None:
            return MISSING, MISSING
        threats = getattr(risk, "threats", None)
        if threats is None:
            return MISSING, MISSING
        count = float(len(threats))
        damage = getattr(risk, "expected_damage", None)
        if damage is None:
            damage = getattr(risk, "damage_estimate", None)
        if damage is None:
            forecast = None
            try:
                forecast = context.forecast_risk(actor)
            except Exception:
                forecast = None
            if forecast is not None:
                damage = getattr(forecast, "damage", None)
                if damage is None:
                    damage = getattr(forecast, "expected_damage", None)
        return count, damage

    def _support_distance(self, actor: str, unit, snapshot) -> Any:
        best = None
        for peer, other in snapshot.allies.items():
            if peer == actor or not getattr(other, "alive", False):
                continue
            type_name = str(getattr(other, "type_name", "") or "").lower()
            ranged = type_name in ("hydralisk", "roach", "corruptor")
            if not ranged or not getattr(other, "weapon_ready", False):
                continue
            if unit.position is None or other.position is None:
                continue
            distance = math.dist(unit.position, other.position)
            best = distance if best is None else min(best, distance)
        return best

    @staticmethod
    def _legal_move_count(actor: str, context) -> int:
        if context is None:
            return 0
        try:
            legal = context.legal_actions(actor)
        except Exception:
            return 0
        count = 0
        for action in legal or ():
            try:
                if context.decode(actor, action).kind == "move":
                    count += 1
            except Exception:
                continue
        return count

    @staticmethod
    def _retreat_quality(actor: str, context) -> Any:
        provider = getattr(context, "retreat_quality", None)
        if callable(provider):
            try:
                return provider(actor)
            except Exception:
                return MISSING
        return MISSING

    def _target_fields(self, actor, unit, relations, plan, enemies):
        target_id = None
        if plan is not None:
            goal = getattr(plan, "goal", None)
            if goal in relations:
                target_id = goal
            else:
                for effect in getattr(plan, "expected_effects", ()) or ():
                    if getattr(effect, "kind", "") == "damage" \
                            and getattr(effect, "target", "") in relations:
                        target_id = effect.target
                        break
        if target_id is None:
            return MISSING, MISSING
        enemy = enemies.get(target_id)
        remaining = getattr(enemy, "effective_health", None) \
            if enemy is not None else None
        relation = relations.get(target_id)
        in_their_range = getattr(relation, "in_enemy_weapon_range", None) \
            if relation is not None else None
        return remaining, _bool(in_their_range)

    # -- team layer ---------------------------------------------------------
    def extract_team_features(self, snapshot, context=None) -> Dict[str, float]:
        alive = [(actor, unit) for actor, unit in snapshot.allies.items()
                 if getattr(unit, "alive", False)]
        enemies = list(snapshot.enemies.values())
        ally_eh = sum((getattr(u, "health", 0.0) or 0.0)
                      + (getattr(u, "shield", 0.0) or 0.0)
                      for _, u in alive)
        enemy_eh = sum((getattr(e, "health", 0.0) or 0.0)
                       + (getattr(e, "shield", 0.0) or 0.0) for e in enemies)
        positions = [u.position for _, u in alive if u.position is not None]
        if len(positions) >= 2:
            total = 0.0
            pairs = 0
            for i in range(len(positions)):
                for j in range(i + 1, len(positions)):
                    total += math.dist(positions[i], positions[j])
                    pairs += 1
            dispersion = total / pairs
        else:
            dispersion = 0.0
        centroid = (
            (sum(p[0] for p in positions) / len(positions),
             sum(p[1] for p in positions) / len(positions))
            if positions else None)
        local_allies = sum(
            1 for p in positions
            if centroid is not None and math.dist(p, centroid) <= self.local_radius)
        local_enemies = sum(
            1 for e in enemies
            if centroid is not None and e.position is not None
            and math.dist(e.position, centroid) <= self.local_radius)
        ratio = (ally_eh / enemy_eh) if enemy_eh > 0 else (
            MISSING if ally_eh <= 0 else 1.0)
        melee_ready = sum(
            1 for _, u in alive
            if getattr(u, "weapon_ready", False)
            and str(getattr(u, "type_name", "")).lower()
            in ("zergling", "ultralisk", "baneling"))
        ranged_ready = sum(
            1 for _, u in alive
            if getattr(u, "weapon_ready", False)
            and str(getattr(u, "type_name", "")).lower() == "hydralisk")
        pressure = self._total_pressure(context, alive)
        kills, losses = self._casualty_fields(context)
        return {
            "ally_alive_count": float(len(alive)),
            "enemy_visible_count": float(len(enemies)),
            "ally_effective_health": float(ally_eh),
            "enemy_effective_health": float(enemy_eh),
            "local_ally_count": float(local_allies),
            "local_enemy_count": float(local_enemies),
            "local_force_ratio": _num(ratio),
            "team_dispersion": float(dispersion),
            "frontline_support": float(melee_ready),
            "ranged_support": float(ranged_ready),
            "combat_pressure": _num(pressure),
            "recent_kills": _num(kills),
            "recent_losses": _num(losses),
        }

    @staticmethod
    def _total_pressure(context, alive) -> Any:
        if context is None:
            return MISSING
        total = 0
        for actor, unit in alive:
            risk = None
            try:
                risk = context.risk_at(actor, unit.position)
            except Exception:
                risk = None
            if risk is not None and getattr(risk, "threats", None):
                total += len(risk.threats)
        return float(total)

    @staticmethod
    def _casualty_fields(context) -> Tuple[Any, Any]:
        memory = getattr(context, "deployment_memory", None)
        if memory is None:
            return MISSING, MISSING
        kills = getattr(memory, "kills", None)
        if kills is None:
            kills = getattr(memory, "confirmed_kills", None)
        losses = getattr(memory, "losses", None)
        if losses is None:
            losses = getattr(memory, "fallen_actors", None)
        kill_count = len(kills) if isinstance(kills, (list, dict, tuple, set)) \
            else None
        loss_count = len(losses) if isinstance(losses, (list, dict, tuple, set)) \
            else None
        return kill_count, loss_count

    # -- skill layer --------------------------------------------------------
    def extract_skill_features(self, snapshot, context=None) -> Dict[str, float]:
        memory = getattr(context, "deployment_memory", None)
        current = getattr(memory, "current_skill", None) \
            if memory is not None else None
        if current is None and memory is not None:
            instances = getattr(memory, "instances", None)
            if isinstance(instances, (list, dict)):
                pool = list(instances.values()) if isinstance(instances, dict) \
                    else list(instances)
                if pool:
                    latest = pool[-1]
                    current = getattr(latest, "skill", None)
        previous = getattr(memory, "previous_skill", None) \
            if memory is not None else None
        duration, switch_cost, phase, progress, staleness, success, failures = \
            self._lifecycle_fields(memory, snapshot)
        return {
            "current_skill": float(encode_index(current, SKILL_VOCABULARY)),
            "previous_skill": float(encode_index(previous, SKILL_VOCABULARY)),
            "skill_duration": _num(duration),
            "skill_switch_cost": _num(switch_cost),
            "current_commitment_phase": float(
                encode_index(phase, PHASE_VOCABULARY)),
            "task_progress": _num(progress),
            "task_staleness": _num(staleness),
            "previous_skill_success": _num(success),
            "repeated_failure_count": _num(failures),
        }

    @staticmethod
    def _lifecycle_fields(memory, snapshot):
        if memory is None:
            return (MISSING,) * 7
        instance = None
        active = getattr(memory, "active_instance", None)
        if active is not None:
            instance = active
        else:
            instances = getattr(memory, "instances", None)
            if isinstance(instances, dict) and instances:
                instance = list(instances.values())[-1]
            elif isinstance(instances, list) and instances:
                instance = instances[-1]
        if instance is None:
            return (MISSING,) * 7
        loop = getattr(snapshot, "game_loop", 0) or 0
        duration = loop - getattr(instance, "start_loop", loop)
        budget = getattr(instance, "budget_loops", None)
        phase = getattr(instance, "phase", None)
        extra = getattr(instance, "extra", None) or {}
        progress = extra.get("progress")
        staleness = loop - getattr(instance, "last_progress_loop", loop)
        success = extra.get("last_success")
        failures = extra.get("repeated_failures")
        return duration, budget, phase, progress, staleness, success, failures

    # -- aggregate ----------------------------------------------------------
    def extract_state(self, snapshot, context=None,
                      plans: Optional[Sequence[Any]] = None
                      ) -> Dict[str, Any]:
        """Global (team+skill) features plus per-actor unit features.

        Returns {"team": {...}, "skill": {...}, "units": {actor: {...}},
        "unit_types": {actor: canonical_type}}.
        """
        plan_by_actor: Dict[str, Any] = {}
        for plan in plans or ():
            for actor in getattr(plan, "controlled_actors", ()) or ():
                plan_by_actor.setdefault(actor, plan)
        units: Dict[str, Dict[str, float]] = {}
        unit_types: Dict[str, str] = {}
        for actor in _sorted_actors(snapshot.allies):
            features = self.extract_unit_features(
                actor, snapshot, context=context, plan=plan_by_actor.get(actor))
            if features:
                units[actor] = features
                resolved = self.unit_type_of(actor, snapshot)
                if resolved:
                    unit_types[actor] = resolved
        return {
            "team": self.extract_team_features(snapshot, context=context),
            "skill": self.extract_skill_features(snapshot, context=context),
            "units": units,
            "unit_types": unit_types,
        }


def extract_candidate_features(plan, snapshot, context=None,
                               peer_plans: Optional[Sequence[Any]] = None,
                               duty: Optional[str] = None) -> Dict[str, float]:
    """Candidate-layer features describing the proposal itself.

    Everything is read from the declared plan (effects are declared
    predictions, never realised outcomes). ``peer_plans`` supplies the other
    candidates of the SAME decision point so resource conflicts are computed
    without any execution information.
    """
    actors = tuple(getattr(plan, "controlled_actors", ()) or ())
    previous = getattr(snapshot, "previous_action_ids", {}) or {}
    current_actions = getattr(plan, "current_actions", {}) or {}
    changes = sum(1 for actor, action in current_actions.items()
                  if previous.get(actor) != action)
    effects = getattr(plan, "expected_effects", ()) or ()
    damage = sum(float(getattr(e, "amount", 0.0) or 0.0)
                 for e in effects if getattr(e, "kind", "") == "damage"
                 and getattr(e, "amount", None) is not None)
    loss = sum(float(getattr(e, "amount", 0.0) or 0.0)
               for e in effects if getattr(e, "kind", "") == "loss"
               and getattr(e, "amount", None) is not None)
    conflicts = 0.0
    if peer_plans:
        own = set(actors) | set(getattr(getattr(plan, "claims", None),
                                        "controls", ()) or ())
        for peer in peer_plans:
            if peer is plan:
                continue
            peer_actors = set(getattr(peer, "controlled_actors", ()) or ())
            if own & peer_actors:
                conflicts = 1.0
                break
    risk = _risk_fraction(plan, snapshot)
    plan_features = getattr(plan, "features", {}) or {}
    progress = plan_features.get("progress", plan_features.get("task_progress"))
    commitment = getattr(plan, "commitment", None)
    phase = getattr(commitment, "phase", None) if commitment is not None else None
    return {
        "candidate_skill": float(
            encode_index(getattr(plan, "primary_skill", None),
                         SKILL_VOCABULARY)),
        "candidate_phase": float(encode_index(phase, PHASE_VOCABULARY)),
        "candidate_actor_count": float(len(actors)),
        "candidate_action_change_count": float(changes),
        "candidate_target_type": float(
            encode_index(getattr(plan, "goal_kind", None),
                         GOAL_KIND_VOCABULARY)),
        "candidate_predicted_damage": damage,
        "candidate_predicted_loss": loss,
        "candidate_risk": _num(risk),
        "candidate_support_gain": MISSING,   # reserved; no estimator in v1
        "candidate_progress": _num(progress),
        "candidate_resource_conflict": conflicts,
        "candidate_execution_duty": float(
            encode_index(duty, DUTY_VOCABULARY)),
    }


def _risk_fraction(plan, snapshot) -> Any:
    actors = getattr(plan, "controlled_actors", ()) or ()
    relations = getattr(snapshot, "relations", {}) or {}
    allies = getattr(snapshot, "allies", {}) or {}
    exposed, counted = 0, 0
    for actor in actors:
        unit = allies.get(actor)
        if unit is None or not getattr(unit, "alive", False):
            continue
        counted += 1
        rows = relations.get(actor, {}) or {}
        if any(getattr(row, "in_enemy_weapon_range", False)
               for row in rows.values()):
            exposed += 1
    if not counted:
        return MISSING
    return exposed / counted
