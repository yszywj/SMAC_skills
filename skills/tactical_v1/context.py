"""Snapshot construction and read-only context helpers.

The context never reads hidden enemy state: it is a strict view over the
adapter payload built by bridge/policy.build_request (team-shared vision).
"""
import math
from typing import Dict, Iterable, List, Optional, Tuple

from .capabilities import Capability, canonical_type, capability_for
from .contracts import (CALIBRATED, EnemyView, Relation, Snapshot, UnitView,
                        HEURISTIC)

MOVE_ACTIONS = (2, 3, 4, 5)          # N/S/E/W per ACTION_CONTRACT
STOP = 1
DEAD_NOOP = 0


def _position(value) -> Tuple[float, float]:
    return (float(value[0]), float(value[1]))


def _num(value, default=0.0) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def build_snapshot(payload: Dict, snapshot_id: int) -> Snapshot:
    """Convert an adapter payload into an immutable Snapshot."""
    state = payload["state"]
    questions = payload.get("questions", {})
    allies: Dict[str, UnitView] = {}
    for index, (actor, unit) in enumerate(sorted(state["allies"].items())):
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
    for index, (enemy_id, unit) in enumerate(sorted(state["visible_enemies"].items())):
        enemies[enemy_id] = EnemyView(
            enemy_id=enemy_id, index=index,
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
                enemy_weapon_range_status=raw.get("enemy_weapon_range_status", "unknown"),
                smac_attack_available=bool(raw.get("smac_attack_available")))
    movement = {actor: {int(a): info for a, info in cands.items()}
                for actor, cands in state.get("movement_candidates", {}).items()}
    heals = {actor: {int(a): info for a, info in cands.items()}
             for actor, cands in state.get("heal_candidates", {}).items()}
    legal: Dict[str, frozenset] = {}
    for actor, unit in allies.items():
        if not unit.alive:
            legal[actor] = frozenset({DEAD_NOOP})
        else:
            criteria = questions.get(actor, {}).get("criteria", {})
            mask = state.get("legal_actions", {}).get(actor)
            if mask is not None:
                legal[actor] = frozenset(int(a) for a in mask)
            elif criteria:
                legal[actor] = frozenset(int(a) for a in criteria)
            else:  # offline fixtures without a question menu
                legal[actor] = frozenset((STOP, *MOVE_ACTIONS))
    keep = state.get("keep_heal_order_candidates", {})
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
        raw_state=state)


def distance(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


class Context:
    """Convenience read-only access layer over a Snapshot."""

    def __init__(self, snapshot: Snapshot):
        self.snapshot = snapshot
        self.capabilities: Dict[str, Capability] = {
            actor: capability_for(unit.type_name, unit.is_flying)
            for actor, unit in snapshot.allies.items()}
        self.enemy_capabilities: Dict[str, Capability] = {
            enemy_id: capability_for(enemy.type_name, enemy.is_flying)
            for enemy_id, enemy in snapshot.enemies.items()}

    # -- legality -----------------------------------------------------------
    def legal_actions(self, actor: str) -> frozenset:
        return self.snapshot.legal.get(actor, frozenset({DEAD_NOOP}))

    def can_attack(self, actor: str, enemy_id: str) -> bool:
        """A legal attack action AND a compatible weapon (both required)."""
        relation = self.snapshot.relations.get(actor, {}).get(enemy_id)
        if relation is None or not relation.smac_attack_available:
            return False
        return (6 + self.snapshot.enemies[enemy_id].index) in self.legal_actions(actor)

    def attack_action(self, actor: str, enemy_id: str) -> Optional[int]:
        return (6 + self.snapshot.enemies[enemy_id].index
                if self.can_attack(actor, enemy_id) else None)

    def move_destinations(self, actor: str) -> Dict[int, Tuple[float, float]]:
        result = {}
        for action, info in self.snapshot.movement_candidates.get(actor, {}).items():
            destination = info.get("destination")
            if destination is not None and action in self.legal_actions(actor):
                result[action] = _position(destination)
        return result

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

    # -- combat helpers -----------------------------------------------------
    def attackable_enemies(self, actor: str) -> List[str]:
        return sorted((e for e in self.snapshot.enemies if self.can_attack(actor, e)),
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

    def expected_damage(self, actor: str, enemy_id: str) -> Tuple[float, str, float]:
        """(amount, evidence, confidence). Uses the shared damage model when
        the raw state carries weapon data; otherwise a conservative lower
        bound of 0 with HEURISTIC evidence (never a fabricated number)."""
        try:
            from skills.current.skills.damage_model import expected_attack_damage_details
            details = expected_attack_damage_details(self.snapshot.raw_state, actor, enemy_id)
            amount = _num(details.get("estimate"), default=None) if isinstance(
                details.get("estimate"), (int, float)) else None
            if amount is not None and amount >= 0:
                # Catalog-calibrated single-attack estimate (no splash credit).
                return amount, CALIBRATED, 0.8
        except Exception:
            pass
        return 0.0, HEURISTIC, 0.0

    def threats_to(self, actor: str) -> List[str]:
        """Visible enemies whose weapon can currently reach this actor."""
        unit = self.snapshot.allies[actor]
        result = []
        for enemy_id in self.snapshot.enemies:
            relation = self.snapshot.relations.get(actor, {}).get(enemy_id)
            if relation is None:
                continue
            status = relation.enemy_weapon_range_status
            if status in ("no_compatible_weapon",):
                continue  # e.g. ground weapon vs flying Medivac
            if relation.in_enemy_weapon_range:
                result.append(enemy_id)
        return result

    def incoming_pressure(self, actor: str) -> int:
        return len(self.threats_to(actor))

    def engagement_groups(self) -> List[List[str]]:
        """Local fight clusters: allies connected by support distance."""
        allies = list(self.snapshot.allies)
        support = 8.0
        adjacency = {a: set() for a in allies}
        for i, a in enumerate(allies):
            for b in allies[i + 1:]:
                if distance(self.snapshot.allies[a].position,
                            self.snapshot.allies[b].position) <= support:
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
