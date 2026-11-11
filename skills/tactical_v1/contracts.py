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
    index: int                      # SMAC agent index; deterministic tie-break only
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
    index: int
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


@dataclass
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


@dataclass
class EffectClaim:
    """One predicted effect with its time window and evidence grade."""
    kind: str                              # damage | heal | position | exposure
    target: str                            # enemy id / ally id / actor
    amount: float
    window_loops: int
    evidence: str                          # VERIFIED / CALIBRATED / HEURISTIC
    source: str                            # human-readable provenance
    confidence: float = 1.0                # <= 1; < 1 for heuristic estimates


@dataclass
class ResourceClaims:
    """Claims on the four separated resource classes."""
    controls: Tuple[str, ...] = ()         # action control rights
    fire: Tuple[Tuple[str, float], ...] = ()   # (enemy_id, expected damage)
    healing: Tuple[Tuple[str, float], ...] = ()  # (ally_id, expected heal)
    space: Tuple[str, ...] = ()            # coarse destination keys per actor


@dataclass
class Commitment:
    """How long a plan wants to persist and how it ends."""
    phase: str
    min_steps: int                         # hysteresis against flapping
    max_steps: Optional[int]               # timeout -> instance fails
    interrupt_on: Tuple[str, ...] = ("target_lost", "illegal", "danger_change")


@dataclass
class CandidatePlan:
    proposal_id: str
    snapshot_id: int
    config_fingerprint: str
    primary_skill: str                     # G1..G7 / S1..S3 / fallback
    specialization_provider: Optional[str] # S1/S2/S3 when the mechanism model is used
    instance_id: Optional[str]
    controlled_actors: Tuple[str, ...]
    beneficiaries: Tuple[str, ...]
    goal: str                              # enemy id / position / area / ally
    current_actions: Dict[str, int]
    expected_effects: List[EffectClaim]
    claims: ResourceClaims
    dependencies: Tuple[str, ...]          # conditions that must co-hold
    commitment: Commitment
    features: Dict[str, float]             # evaluator inputs
    proposed_memory_delta: Dict[str, Any]  # applied only if selected
    rationale: str = ""


@dataclass
class SkillInstance:
    """A running skill instance; one skill type may have many instances."""
    instance_id: str
    skill: str
    actors: Tuple[str, ...]
    goal: str
    phase: str
    start_loop: int
    last_progress_loop: int
    committed_until_step: int              # hysteresis: no free switch before this
    budget_steps: int                      # steps consumed
    max_steps: Optional[int]
    extra: Dict[str, Any] = field(default_factory=dict)

    def advance(self, loop: int, snapshot_id: int) -> None:
        """Only the coordinator may advance an instance after commit."""
        self.budget_steps += 1

    def mark_progress(self, loop: int) -> None:
        self.last_progress_loop = loop


@dataclass
class StepResult:
    actions: List[int]
    telemetry: Dict[str, Any]
