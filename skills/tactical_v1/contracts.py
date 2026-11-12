"""Shared data contracts for the tactical_v1 controller.

Skills only read snapshots and committed memory, and return CandidatePlans.
Nothing in this module mutates environment state or submits actions.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# Evidence grades for expected effects. Anything below VERIFIED is an
# estimate and must never enter a "certain kill/heal" budget.
VERIFIED = "verified"        # from the environment's own legal/action contract
CALIBRATED = "calibrated"    # checked against recorded games of this adapter
HEURISTIC = "heuristic"      # geometric or density proxy, unknown magnitude


@dataclass(frozen=True)
class UnitView:
    """A controlled ally as seen through the allowed team-vision snapshot."""
    actor: str
    index: int                      # REAL SMAC agent index (from the u{i} key)
    tag: Any
    type_name: str
    unit_type_id: int
    position: Tuple[float, float]
    radius: float
    is_flying: bool
    alive: bool
    health: float
    max_health: float
    shield: float
    max_shield: float
    weapon_cooldown_loops: float
    weapon_ready: bool
    energy: Optional[float]
    health_lost: float              # since the previous decision step
    shield_lost: float

    @property
    def effective_health(self) -> float:
        return self.health + self.shield

    @property
    def health_fraction(self) -> float:
        return self.health / self.max_health if self.max_health else 1.0


@dataclass(frozen=True)
class EnemyView:
    """A visible enemy. Hidden enemies are unknown, never fabricated."""
    enemy_id: str
    index: int                      # REAL enemy index (from the e{i} key)
    type_name: str
    unit_type_id: int
    position: Tuple[float, float]
    radius: float
    is_flying: bool
    health: float
    max_health: float
    shield: float
    max_shield: float

    @property
    def effective_health(self) -> float:
        return self.health + self.shield


@dataclass(frozen=True)
class Relation:
    distance: float
    in_weapon_range: Optional[bool]
    weapon_range_status: str
    native_weapon_range: Optional[float]
    in_enemy_weapon_range: Optional[bool]
    enemy_weapon_range_status: str
    smac_attack_available: bool


@dataclass(frozen=True)
class Snapshot:
    """Immutable view of one decision step."""
    snapshot_id: int                       # monotonic per-episode counter
    game_loop: int
    step_mul: int
    steps_remaining: int
    map_name: str
    episode: int
    search_location: Tuple[float, float]
    allies: Dict[str, UnitView]
    enemies: Dict[str, EnemyView]          # visible only
    relations: Dict[str, Dict[str, Relation]]
    movement_candidates: Dict[str, Dict[int, Dict[str, Any]]]   # actor -> action -> info
    heal_candidates: Dict[str, Dict[int, Dict[str, Any]]]
    legal: Dict[str, frozenset]            # actor -> legal action ids (raw mask view)
    keep_heal_order: Dict[str, Dict[str, Any]]   # actor -> candidate (extension only)
    previous_action_ids: Dict[str, str]
    raw_state: Dict[str, Any]              # read-only reference for shared helpers
    n_allies: int                          # REAL roster size (not alive count)
    n_enemies: int                         # REAL enemy count (not visible count)
    keep_heal_action_id: Optional[int]     # extension slot id when enabled


@dataclass(frozen=True)
class EffectClaim:
    """One predicted effect with its time window and evidence grade.
    amount=None means unknown magnitude (opportunity only), never a fake 0."""
    kind: str                              # damage | heal | coverage | position | loss
    target: str                            # enemy id / ally id / actor
    amount: Optional[float]
    window_loops: int
    evidence: str                          # VERIFIED / CALIBRATED / HEURISTIC
    source: str                            # human-readable provenance
    lower: Optional[float] = None          # explicit interval when available
    upper: Optional[float] = None


@dataclass(frozen=True)
class SpaceClaim:
    """Structured spatial reservation (R5). Soft congestion by default; only
    an explicit shared slot id is a hard mutual exclusion."""
    actor: str
    position: Tuple[float, float]          # predicted position within the window
    window_loops: int
    collision_class: str                   # ground | air
    radius: float
    slot_id: Optional[str] = None          # hard-exclusion contact slot key


@dataclass(frozen=True)
class ResourceClaims:
    """Claims on the four separated resource classes."""
    controls: Tuple[str, ...] = ()         # action control rights
    fire: Tuple[Tuple[str, float], ...] = ()   # (enemy_id, expected window damage)
    healing: Tuple[Tuple[str, float], ...] = ()  # (ally_id, expected window heal proxy)
    space: Tuple[SpaceClaim, ...] = ()


@dataclass(frozen=True)
class Dependency:
    """Executable dependency condition (R5); resolved by the validator."""
    kind: str          # member | target_valid | fire_window | heal_owner | slot
    refs: Tuple[str, ...]   # actor / enemy / ally ids the condition binds


@dataclass
class Commitment:
    """How long a plan wants to persist and how it ends. Budgets are in
    loops, converted from H multiples at task creation (2.3)."""
    phase: str
    min_loops: int = 0                      # soft switch cost, never blocks illegal
    max_loops: Optional[int] = None
    interrupt_on: Tuple[str, ...] = ("target_lost", "illegal", "danger_change")


@dataclass
class CandidatePlan:
    proposal_id: str
    snapshot_id: int
    config_fingerprint: str
    primary_skill: str                     # G1..G7 / S1..S3 / baseline
    specialization_provider: Optional[str] # S1/S2/S3 when the mechanism model is used
    instance_id: Optional[str]
    controlled_actors: Tuple[str, ...]
    beneficiaries: Tuple[str, ...]
    goal: str                              # enemy id / position / area / ally
    current_actions: Dict[str, int]
    expected_effects: List[EffectClaim]
    claims: ResourceClaims
    dependencies: Tuple[Dependency, ...]
    commitment: Commitment
    features: Dict[str, float]             # named, capped evaluator inputs (no raw_bonus)
    proposed_memory_delta: Dict[str, Any]  # applied only if selected
    atomic: bool = False                   # multi-unit plans execute whole
    rationale: str = ""
    # 1.2: real, evaluator-readable goal. goal_kind/goal_data describe WHERE
    # and WHY; task_key is stable across steps for the same duty + real unit
    # identities + normalized target (independent of proposal_id/order).
    goal_kind: str = "internal"            # enemy/ally/position/group_link/meeting/internal
    goal_data: Dict[str, Any] = field(default_factory=dict)
    task_key: str = ""


@dataclass
class SkillInstance:
    """A running skill instance; one skill type may have many instances.
    Duration is measured by game_loop deltas, never by call counts (R4)."""
    instance_id: str
    skill: str
    actors: Tuple[str, ...]
    ally_tags: Tuple[Any, ...]
    goal: str
    goal_kind: str                  # enemy | ally | position | area | internal
    phase: str
    start_loop: int
    phase_start_loop: int
    last_progress_loop: int
    budget_loops: int                      # consumed duration, loop-based
    executed_steps: int                    # actual committed decision steps
    max_loops: Optional[int]
    retry_after_loop: Optional[int] = None
    termination_reason: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StepResult:
    actions: List[int]
    telemetry: Dict[str, Any]
