"""Snapshot construction and the shared combat queries (R1–R3).

Three SEPARATE queries — never one `can_attack` doing everything:
  can_issue_attack / contact_status / fire_estimate.
Movement prediction and threat evaluation share one risk standard for the
current and every candidate position.
"""
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .actions import (ContractError, DEAD_NOOP, MOVE_ACTIONS, STOP,
                      SemanticAction, decode)
from .capabilities import Capability, canonical_type, capability_for
from .contracts import (CALIBRATED, EnemyView, Relation, Snapshot, UnitView,
                        HEURISTIC)


def _position(value) -> Tuple[float, float]:
    return (float(value[0]), float(value[1]))


def _num(value, default=0.0) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def _env_index(key: str, prefix: str) -> int:
    """The REAL environment index is carried by the actor/enemy key
    (u{i}/e{i} are SMAC agent indices, not visibility order)."""
    if not (len(key) > 1 and key[0] == prefix and key[1:].isdigit()):
        raise ContractError(f"malformed {prefix}-side key: {key}")
    return int(key[1:])


def build_snapshot(payload: Dict, snapshot_id: int) -> Snapshot:
    state = payload["state"]
    questions = payload.get("questions", {})
    allies: Dict[str, UnitView] = {}
    n_enemies = int(state.get("n_enemies")
                    or max((_env_index(e, "e") for e in state.get(
                        "visible_enemies", {})), default=-1) + 1)
    for actor, unit in state["allies"].items():
        index = _env_index(actor, "u")
        type_key = canonical_type(unit.get("type"), unit.get("unit_type_id"))
        allies[actor] = UnitView(
            actor=actor, index=index, tag=unit.get("tag", actor),
            type_name=type_key, unit_type_id=int(unit.get("unit_type_id") or -1),
            position=_position(unit["position"]), radius=_num(unit.get("radius"), .5),
            is_flying=bool(unit.get("is_flying")), alive=bool(unit.get("alive", True)),
            health=_num(unit.get("health")), max_health=_num(unit.get("max_health")),
            shield=_num(unit.get("shield")), max_shield=_num(unit.get("max_shield")),
            weapon_cooldown_loops=_num(unit.get("weapon_cooldown_loops")),
            weapon_ready=bool(unit.get("weapon_ready", True)),
            energy=(None if unit.get("energy") is None else _num(unit.get("energy"))),
            health_lost=_num(unit.get("health_lost_since_previous_decision")),
            shield_lost=_num(unit.get("shield_lost_since_previous_decision")))
    enemies: Dict[str, EnemyView] = {}
    for enemy_id, unit in state["visible_enemies"].items():
        enemies[enemy_id] = EnemyView(
            enemy_id=enemy_id, index=_env_index(enemy_id, "e"),
            type_name=canonical_type(unit.get("type"), unit.get("unit_type_id")),
            unit_type_id=int(unit.get("unit_type_id") or -1),
            position=_position(unit["position"]), radius=_num(unit.get("radius"), .5),
            is_flying=bool(unit.get("is_flying")),
            health=_num(unit.get("health")), max_health=_num(unit.get("max_health")),
            shield=_num(unit.get("shield")), max_shield=_num(unit.get("max_shield")))
    relations: Dict[str, Dict[str, Relation]] = {}
    for actor, per_enemy in state.get("relations", {}).items():
        relations[actor] = {}
        for enemy_id, raw in per_enemy.items():
            relations[actor][enemy_id] = Relation(
                distance=_num(raw.get("distance"), float("inf")),
                in_weapon_range=raw.get("in_weapon_range"),
                weapon_range_status=raw.get("weapon_range_status", "unknown"),
                native_weapon_range=(None if raw.get("native_weapon_range") is None
                                     else _num(raw["native_weapon_range"])),
                in_enemy_weapon_range=raw.get("in_enemy_weapon_range"),
                enemy_weapon_range_status=raw.get("enemy_weapon_range_status",
                                                  "unknown"),
                smac_attack_available=bool(raw.get("smac_attack_available")))
    movement = {actor: {int(a): info for a, info in cands.items()}
                for actor, cands in state.get("movement_candidates", {}).items()}
    heals = {actor: {int(a): info for a, info in cands.items()}
             for actor, cands in state.get("heal_candidates", {}).items()}
    keep = state.get("keep_heal_order_candidates", {})
    keep_contract = state.get("action_contract", {}).get("keep_heal_order", {})
    keep_action_id = int(keep_contract.get("action_id", -1)) \
        if keep_contract.get("enabled") else None
    legal: Dict[str, frozenset] = {}
    for actor, unit in allies.items():
        if not unit.alive:
            legal[actor] = frozenset({DEAD_NOOP})
            continue
        mask = state.get("legal_actions", {}).get(actor)
        criteria = questions.get(actor, {}).get("criteria", {})
        if mask is not None:
            legal[actor] = frozenset(int(a) for a in mask)
        elif criteria:
            legal[actor] = frozenset(int(a) for a in criteria)
        else:
            # A living unit without a legal action menu is a CONTRACT error;
            # fabricating four move directions here would hide it (R1).
            raise ContractError(
                f"living actor {actor} has no legal action menu "
                "(neither questions.criteria nor state.legal_actions)")
    return Snapshot(
        snapshot_id=snapshot_id,
        game_loop=int(state.get("game_loop", 0)),
        step_mul=int(state.get("step_mul", 8)),
        steps_remaining=int(state.get("steps_remaining", 0)),
        map_name=state.get("map", "unknown") if isinstance(state.get("map"), str)
        else state.get("map", {}).get("name", "unknown"),
        episode=int(state.get("episode", 0)),
        search_location=_position(state.get("search_location", (0.0, 0.0))),
        allies=allies, enemies=enemies, relations=relations,
        movement_candidates=movement, heal_candidates=heals, legal=legal,
        keep_heal_order=keep,
        previous_action_ids=dict(state.get("previous_action_ids", {})),
        raw_state=state,
        n_allies=max(u.index for u in allies.values()) + 1 if allies else 0,
        n_enemies=n_enemies,
        keep_heal_action_id=keep_action_id)


def distance(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


IN_RANGE = "in_range"
OUT_OF_RANGE = "out_of_range"
CONTACT_UNKNOWN = "unknown"


@dataclass(frozen=True)
class FireEstimate:
    """Window-qualified attack feasibility (R2). H = control window."""
    window_shots: int                 # conservative: 0 or 1 (multi only with verified data)
    damage_lo: Optional[float]        # None = unknown, never a fake zero credit
    damage_est: Optional[float]
    damage_hi: Optional[float]
    evidence: str
    effective_after_loops: Optional[int]   # cooldown when a shot can land
    delay_known: bool


@dataclass(frozen=True)
class RiskReport:
    threats: Tuple[str, ...]
    heuristic: bool                   # True when ranges were proxied
    effect_weapon_near: Tuple[str, ...]   # baneling-like proximity danger


@dataclass(frozen=True)
class PredictedPosition:
    position: Tuple[float, float]
    certain: bool                     # False when speed/path data is missing
    source: str


class Context:
    """Read-only access layer. The single risk/geometry standard."""

    def __init__(self, snapshot: Snapshot):
        self.snapshot = snapshot
        self.capabilities: Dict[str, Capability] = {
            actor: capability_for(unit.type_name, unit.is_flying)
            for actor, unit in snapshot.allies.items()}
        self.enemy_capabilities: Dict[str, Capability] = {
            enemy_id: capability_for(enemy.type_name, enemy.is_flying)
            for enemy_id, enemy in snapshot.enemies.items()}
        self.H = max(snapshot.step_mul, 1)   # unified evaluation window (R2)

    # ---------------------------------------------------------------- legality
    def legal_actions(self, actor: str) -> frozenset:
        return self.snapshot.legal.get(actor, frozenset({DEAD_NOOP}))

    def decode(self, actor: str, action: int) -> SemanticAction:
        unit = self.snapshot.allies.get(actor)
        keep_target = None
        if self.snapshot.keep_heal_action_id is not None \
                and action == self.snapshot.keep_heal_action_id:
            keep_target = self.snapshot.keep_heal_order.get(actor, {}).get("target")
        return decode(action,
                      is_healer=self.capabilities[actor].healer if actor in self.capabilities else False,
                      n_enemies=self.snapshot.n_enemies,
                      n_allies=self.snapshot.n_allies,
                      keep_heal_action_id=self.snapshot.keep_heal_action_id,
                      keep_heal_target=keep_target)

    def attack_action(self, actor: str, enemy_id: str) -> Optional[int]:
        """Slot id for a VISIBLE enemy, iff the mask allows submitting it."""
        enemy = self.snapshot.enemies.get(enemy_id)
        if enemy is None:
            return None
        action = 6 + enemy.index
        return action if action in self.legal_actions(actor) else None

    def heal_action(self, actor: str, patient: str) -> Optional[int]:
        for action, info in self.snapshot.heal_candidates.get(actor, {}).items():
            if info.get("target") == patient and action in self.legal_actions(actor):
                return action
        return None

    def keep_heal_action(self, actor: str) -> Optional[int]:
        candidate = self.snapshot.keep_heal_order.get(actor)
        if not candidate:
            return None
        action = int(candidate["action_id"])
        return action if action in self.legal_actions(actor) else None

    def heal_occupancy(self) -> Dict[str, str]:
        """OBSERVED heal ownership: patient actor -> healer actor, from real
        order records in the snapshot (S2.2). A mask never proves that two
        medivacs may treat the same patient simultaneously."""
        owners: Dict[str, str] = {}
        for actor, unit in self.snapshot.raw_state.get("allies", {}).items():
            if not self.capabilities.get(actor) or \
                    not self.capabilities[actor].healer:
                continue
            orders = unit.get("orders") or []
            for order in orders:
                if order.get("kind") == "heal" and order.get("target_unit"):
                    target = order["target_unit"]
                    if target in self.snapshot.allies:
                        owners.setdefault(target, actor)
        return owners

    # ------------------------------------------------- the three attack queries
    def can_issue_attack(self, actor: str, enemy_id: str) -> bool:
        """Query 1: may an attack command be submitted RIGHT NOW?"""
        relation = self.snapshot.relations.get(actor, {}).get(enemy_id)
        if relation is None or not relation.smac_attack_available:
            return False
        return self.attack_action(actor, enemy_id) is not None

    def contact_status(self, actor: str, enemy_id: str) -> str:
        """Query 2: inside/outside actual weapon reach, or unknown."""
        relation = self.snapshot.relations.get(actor, {}).get(enemy_id)
        if relation is None:
            return CONTACT_UNKNOWN
        if relation.weapon_range_status == "no_compatible_weapon":
            return OUT_OF_RANGE       # domain mismatch can never be in range
        if relation.in_weapon_range is True:
            return IN_RANGE
        if relation.in_weapon_range is False:
            return OUT_OF_RANGE
        if relation.native_weapon_range is not None \
                and math.isfinite(relation.distance):
            return (IN_RANGE if relation.distance <= relation.native_weapon_range
                    else OUT_OF_RANGE)
        return CONTACT_UNKNOWN

    def fire_estimate(self, actor: str, enemy_id: str,
                      horizon: Optional[int] = None) -> FireEstimate:
        """Query 3: can a shot LAND inside the window, when, with what effect.

        A legal command with cooldown longer than H keeps no window damage;
        out-of-physical-range attacks score no immediate damage either. Only
        verified catalog data yields a damage estimate; unknown stays
        unknown (None), never a fabricated number.
        """
        horizon = horizon or self.H
        unit = self.snapshot.allies[actor]
        contact = self.contact_status(actor, enemy_id)
        if contact != IN_RANGE:
            return FireEstimate(0, None, None, None, HEURISTIC, None, False)
        cooldown = unit.weapon_cooldown_loops
        window_shots = 1 if cooldown <= horizon else 0
        amount_lo = amount_est = amount_hi = None
        evidence = HEURISTIC
        if window_shots >= 1:
            result = self._damage(actor, enemy_id)
            if result is not None:
                amount_lo, amount_est, amount_hi = result
                evidence = CALIBRATED
        return FireEstimate(window_shots, amount_lo, amount_est, amount_hi,
                            evidence, int(cooldown) if cooldown >= 0 else None,
                            True)

    def _damage(self, actor: str, enemy_id: str):
        try:
            from skills.current.skills.damage_model import (
                expected_attack_damage_details)
        except ImportError as exc:   # explicit, never a silent blanket (R5)
            raise ContractError(f"shared damage model unavailable: {exc}")
        details = expected_attack_damage_details(
            self.snapshot.raw_state, actor, enemy_id)
        if details.get("estimate") is not None and details.get("status") not in (
                None, "unknown"):
            est = float(details["estimate"])
            lo, hi = details.get("lower_bound"), details.get("upper_bound")
            return (_num(lo, est) if lo is not None else est, est,
                    _num(hi, est) if hi is not None else est)
        return None

    # -------------------------------------------------------- movement & risk
    def speed(self, actor: str) -> Optional[float]:
        profile = self.snapshot.raw_state.get("unit_type_profiles", {}).get(
            str(self.snapshot.allies[actor].unit_type_id), {})
        value = profile.get("movement_speed_native")
        return float(value) if isinstance(value, (int, float)) \
            and math.isfinite(value) and value > 0 else None

    def predict_position(self, actor: str, action: int,
                         horizon: Optional[int] = None) -> PredictedPosition:
        """Command destination vs. arrival point within H (R3).

        SC2 runs 22.4 game loops per real second and catalog speeds are
        units/second — an engineering assumption, flagged uncertain whenever
        speed data is missing: the full step destination is then NOT claimed
        as certainly reached.
        """
        horizon = horizon or self.H
        unit = self.snapshot.allies[actor]
        semantic = self.decode(actor, action)
        if semantic.kind != "move":
            # Attack/heal commands may trigger native auto-approach: certain
            # stationary ONLY when already in effective contact (B6).
            if semantic.kind == "attack" and semantic.target in self.snapshot.enemies:
                contact = self.contact_status(actor, semantic.target)
                return PredictedPosition(
                    unit.position, contact == IN_RANGE,
                    f"attack_command_contact_{contact}")
            return PredictedPosition(unit.position, True,
                                     f"hold:{semantic.kind}")
        info = self.snapshot.movement_candidates.get(actor, {}).get(action)
        destination = info.get("destination") if info else None
        if destination is None:
            return PredictedPosition(unit.position, False, "no_candidate_data")
        destination = _position(destination)
        speed = self.speed(actor)
        if speed is None:
            return PredictedPosition(destination, False, "unknown_speed")
        reach = speed * horizon / 22.4
        gap = distance(unit.position, destination)
        if reach >= gap:
            return PredictedPosition(destination, True, "speed_sufficient")
        fraction = reach / gap if gap > 0 else 1.0
        point = (unit.position[0] + (destination[0] - unit.position[0]) * fraction,
                 unit.position[1] + (destination[1] - unit.position[1]) * fraction)
        return PredictedPosition(point, False, "partial_reach")

    def _enemy_weapon_range(self, enemy_id: str,
                            target_flying: bool) -> Optional[float]:
        """Real catalog weapon range for the matching target domain; the
        action-contract candidate table is NOT enemy weapon truth (R3)."""
        enemy = self.snapshot.enemies[enemy_id]
        profile = self.snapshot.raw_state.get("unit_type_profiles", {}).get(
            str(enemy.unit_type_id), {})
        weapons = profile.get("weapons")
        if not isinstance(weapons, list) or not profile.get("known"):
            return None
        domain = "air" if target_flying else "ground"
        ranges = [w.get("range") for w in weapons
                  if isinstance(w, dict) and w.get("target_type") in (domain, "any")]
        values = [r for r in ranges if isinstance(r, (int, float))
                  and math.isfinite(r) and r > 0]
        return max(values) if values else None

    def risk_at(self, actor: str, position,
                horizon: Optional[int] = None) -> RiskReport:
        """Visible threats whose weapon can plausibly reach ``position``.

        Domain-checked (ground weapons cannot threaten a flying Medivac),
        all threats considered (not just the nearest), same standard for
        current and candidate positions. Missing weapon data falls back to a
        clearly-flagged geometric proxy — unknown is not safe, and not
        certain damage either.
        """
        unit = self.snapshot.allies[actor]
        threats: List[str] = []
        effect_near: List[str] = []
        heuristic = False
        for enemy_id, enemy in sorted(self.snapshot.enemies.items()):
            relation = self.snapshot.relations.get(actor, {}).get(enemy_id)
            if relation is not None and \
                    relation.enemy_weapon_range_status == "no_compatible_weapon":
                continue   # cannot ever hit this actor's domain
            weapon_range = self._enemy_weapon_range(enemy_id, unit.is_flying)
            d = distance(position, enemy.position) - enemy.radius - unit.radius
            capability = self.enemy_capabilities.get(enemy_id)
            if capability is not None and capability.suicide_attack:
                if d <= 2.0:
                    effect_near.append(enemy_id)   # baneling proximity proxy
                continue
            if weapon_range is None:
                heuristic = True
                if d <= 5.0:
                    threats.append(enemy_id)
                continue
            if d <= weapon_range:
                threats.append(enemy_id)
        return RiskReport(tuple(threats), heuristic, tuple(effect_near))

    def threats_to(self, actor: str) -> List[str]:
        return list(self.risk_at(actor,
                                 self.snapshot.allies[actor].position).threats)

    # ---------------------------------------------------------------- helpers
    def attackable_enemies(self, actor: str) -> List[str]:
        """Issuable attacks, nearest FIRST (ascending distance)."""
        return sorted((e for e in self.snapshot.enemies
                       if self.can_issue_attack(actor, e)),
                      key=lambda e: (self.snapshot.relations[actor][e].distance, e))

    def nearest_enemy(self, actor: str) -> Optional[str]:
        relations = self.snapshot.relations.get(actor, {})
        if not relations:
            return None
        return min(relations, key=lambda e: (relations[e].distance, e))

    def distance(self, actor: str, enemy_id: str) -> float:
        relation = self.snapshot.relations.get(actor, {}).get(enemy_id)
        if relation is not None:
            return relation.distance
        unit = self.snapshot.allies.get(actor)
        enemy = self.snapshot.enemies.get(enemy_id)
        if unit is None or enemy is None:
            return float("inf")
        return distance(unit.position, enemy.position)

    def move_destinations(self, actor: str) -> Dict[int, Tuple[float, float]]:
        result = {}
        for action, info in self.snapshot.movement_candidates.get(actor, {}).items():
            destination = info.get("destination")
            if destination is not None and action in self.legal_actions(actor):
                result[action] = _position(destination)
        return result

    def engagement_groups(self, link_radius: Optional[float] = None) -> List[List[str]]:
        """Connected ally groups; flying supports do NOT chain distant
        ground groups together (G2.1)."""
        allies = [a for a, u in self.snapshot.allies.items() if u.alive]
        radius = link_radius if link_radius is not None else 8.0
        adjacency = {a: set() for a in allies}
        for i, a in enumerate(allies):
            for b in allies[i + 1:]:
                if self.capabilities[a].flying != self.capabilities[b].flying:
                    continue
                if distance(self.snapshot.allies[a].position,
                            self.snapshot.allies[b].position) <= radius:
                    adjacency[a].add(b)
                    adjacency[b].add(a)
        seen, groups = set(), []
        for actor in sorted(allies):
            if actor in seen:
                continue
            stack, component = [actor], []
            seen.add(actor)
            while stack:
                current = stack.pop()
                component.append(current)
                for other in adjacency[current]:
                    if other not in seen:
                        seen.add(other)
                        stack.append(other)
            groups.append(sorted(component))
        return groups
