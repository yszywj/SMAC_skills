"""A-class (skill-internal) proposal collection without re-calling propose.

Phase 4 contract:

- the ONLY source of A-class candidates is the plans each skill already
  returned to the Coordinator inside the normal ``step`` flow. The
  collector wraps each skill instance's ``propose`` method once; the
  wrapper records what the Coordinator requested and returns the plans
  untouched — ``propose`` is never invoked a second time, Memory is never
  touched, the Coordinator's candidate set and the decision lifecycle
  (including the step cache) are unchanged, and Shadow mode never alters
  actions.
- per-decision output: for every recorded skill, the candidate records
  (target/phase/goal/controlled actors/ResourceClaims/Dependencies/rule
  feature channels/source proposal id) plus the decision key that links
  them to the B-class joint candidates of the SAME decision point.
- A-class candidates keep ``candidate_family="skill_internal"`` and their
  own candidate groups (one group per decision point per skill); they are
  never counted as complete joint plans and joint plans are never split
  into per-unit "action samples".
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from .candidates import FAMILY_SKILL_INTERNAL, candidate_from_plan
from .features import DEFAULT_SCHEMA, ZergFeatureSchema

PRIORITY_SKILLS = ("focus_fire", "coordinated_engage", "combat_reposition",
                   "disengage_reenter", "ranged_fire_formation",
                   "burst_engage")


class SkillProposalCollector:
    """Passive recorder over the skill instances owned by a SkillManager."""

    def __init__(self, manager: Any = None,
                 enabled: Sequence[str] = PRIORITY_SKILLS,
                 schema: ZergFeatureSchema = DEFAULT_SCHEMA):
        self.enabled = tuple(enabled)
        self.schema = schema
        self.propose_calls: Dict[str, int] = {name: 0 for name in self.enabled}
        # (snapshot_id, game_loop) -> {skill: [ZergCandidate]}
        self._decisions: Dict[Tuple[int, int], Dict[str, List[Any]]] = {}
        self._manager: Optional[Any] = None

    # -- installation -------------------------------------------------------
    def install(self, manager: Any) -> "SkillProposalCollector":
        """Wrap ``skill.propose`` on the manager's OWN skill instances,
        exactly once per skill. The wrapper is transparent: same plans out,
        no extra calls, no memory writes."""
        self._manager = manager
        for name in self.enabled:
            skill = manager.skills.get(name)
            if skill is None or not getattr(skill, "enabled", True):
                continue
            if getattr(skill, "_zerg_propose_wrapped", False):
                continue
            original = skill.propose
            collector = self

            def wrapped(context, memory, telemetry, ledger=None,
                        _original=original, _name=name):
                plans = _original(context, memory, telemetry, ledger)
                collector._record(_name, context, plans)
                return plans

            skill.propose = wrapped
            skill._zerg_propose_wrapped = True
        return self

    # -- recording ----------------------------------------------------------
    def _record(self, skill_name: str, context, plans: Sequence[Any]) -> None:
        if skill_name not in self.propose_calls:
            return
        self.propose_calls[skill_name] += 1
        snapshot = getattr(context, "snapshot", None)
        if snapshot is None:
            return
        key = (int(getattr(snapshot, "snapshot_id", 0) or 0),
               int(getattr(snapshot, "game_loop", 0) or 0))
        bucket = self._decisions.setdefault(key, {})
        stored = bucket.setdefault(skill_name, [])
        for plan in plans or ():
            stored.append(candidate_from_plan(
                plan, snapshot, context=context,
                family=FAMILY_SKILL_INTERNAL, peer_plans=list(plans or ()),
                schema=self.schema))

    # -- retrieval ----------------------------------------------------------
    def decision_key(self, snapshot_id: int, game_loop: int) -> Tuple[int, int]:
        return (int(snapshot_id), int(game_loop))

    def candidates_for(self, snapshot_id: int, game_loop: int
                       ) -> Dict[str, List[Any]]:
        """ZergCandidate objects per skill for one decision point. Read-only;
        calling this never triggers propose."""
        return dict(self._decisions.get(
            self.decision_key(snapshot_id, game_loop), {}))

    def records_for(self, snapshot_id: int, game_loop: int
                    ) -> Dict[str, List[Dict[str, Any]]]:
        """Serialisable candidate records per skill, for the decision
        record's ``skill_proposals`` section."""
        return {skill: [candidate.to_record() for candidate in candidates]
                for skill, candidates in self.candidates_for(
                    snapshot_id, game_loop).items()}

    def forget_before(self, snapshot_id: int, game_loop: int) -> None:
        """Drop decision buffers strictly older than the given key (memory
        hygiene for long episodes; committed decisions are already
        persisted by the runtime's record writer)."""
        stale = [key for key in self._decisions if key < (snapshot_id,
                                                          game_loop)]
        for key in stale:
            del self._decisions[key]

    def summary(self) -> Dict[str, Any]:
        return {"propose_calls": dict(self.propose_calls),
                "buffered_decisions": len(self._decisions)}
