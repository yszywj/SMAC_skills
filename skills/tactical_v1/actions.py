"""Single source of truth for SMACv2 action encoding/decoding (R1).

Every module must use these helpers; skills never compute `6 + j` on their
own. The raw action id is ambiguous (attack e{j} vs heal u{j} share the
band), so decoding always needs the actor's role and the enemy/ally index
boundaries of the CURRENT environment.
"""
from dataclasses import dataclass
from typing import Optional

STOP = 1
DEAD_NOOP = 0
MOVE_ACTIONS = (2, 3, 4, 5)   # N/S/E/W
ATTACK_HEAL_BASE = 6
KEEP_HEAL_ORDER = "keep_heal_order"   # extension action; id from the contract


class ContractError(ValueError):
    """The payload violates the action/state contract. Never papered over."""


@dataclass(frozen=True)
class SemanticAction:
    action: int
    kind: str            # dead | stop | move | attack | heal | keep_heal | invalid
    target: Optional[str]   # "e{j}" for attack, "u{j}" for heal/keep_heal
    target_index: Optional[int]

    @property
    def valid(self) -> bool:
        return self.kind != "invalid"


def attack_action_id(enemy_index: int) -> int:
    return ATTACK_HEAL_BASE + enemy_index


def heal_action_id(ally_index: int) -> int:
    return ATTACK_HEAL_BASE + ally_index


def decode(action: int, *, is_healer: bool, n_enemies: int, n_allies: int,
           keep_heal_action_id: Optional[int] = None,
           keep_heal_target: Optional[str] = None) -> SemanticAction:
    """Decode one raw action for one actor.

    `n_enemies`/`n_allies` are the environment's real roster sizes (NOT the
    visible-count); `e5` stays attack slot 11 even if it is the only visible
    enemy. Values >= 6 are NEVER assumed to be attacks without the role and
    boundaries: for a healer the band means heal u{j}; the keep-heal-order
    extension id is decoded separately.
    """
    if action == DEAD_NOOP:
        return SemanticAction(action, "dead", None, None)
    if action == STOP:
        return SemanticAction(action, "stop", None, None)
    if action in MOVE_ACTIONS:
        return SemanticAction(action, "move", None, None)
    if keep_heal_action_id is not None and action == keep_heal_action_id:
        return SemanticAction(action, KEEP_HEAL_ORDER, keep_heal_target,
                              (int(keep_heal_target[1:])
                               if keep_heal_target else None))
    index = action - ATTACK_HEAL_BASE
    if index < 0:
        return SemanticAction(action, "invalid", None, None)
    if is_healer:
        if 0 <= index < n_allies:
            return SemanticAction(action, "heal", f"u{index}", index)
        return SemanticAction(action, "invalid", None, None)
    if 0 <= index < n_enemies:
        return SemanticAction(action, "attack", f"e{index}", index)
    return SemanticAction(action, "invalid", None, None)


def encode_semantic(kind: str, target_index: Optional[int] = None) -> int:
    if kind == "stop":
        return STOP
    if kind == "dead":
        return DEAD_NOOP
    if kind in ("attack", "heal"):
        if target_index is None:
            raise ContractError(f"{kind} needs a target index")
        return ATTACK_HEAL_BASE + target_index
    raise ContractError(f"cannot encode kind {kind}")
