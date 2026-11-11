"""S2 HealSupport: joint healing, sustained execution, safe escort.

Patient/healer graph from LEGAL heal candidates plus OBSERVED orders; the
verified single-patient occupancy rule is respected (an observed order's
patient stays owned until a later observation confirms release). Every
medivac keeps ALL legal patients as candidates (no first-match break) —
capacity is the coordinator's job. Heal rate is NOT calibrated: the window
value is an explicit opportunity proxy, and a moving patient halves it.
"""
from typing import Dict, List, Optional

from ..context import Context, STOP, distance
from ..contracts import Commitment, Dependency, EffectClaim, HEURISTIC, ResourceClaims
from .common import best_move_toward, make_plan, move_progress, space_claim


class HealSupportSkill:
    name = "heal_support"

    def __init__(self, config: Dict):
        self.config = config
        params = config.get("heal_support", {})
        self.follow_distance = params.get("follow_distance", 5.0)
        self.switch_margin = params.get("switch_margin", 0.3)

    def _urgency(self, context: Context, patient: str) -> float:
        """Missing HEALTH fraction x visible pressure. Shields never count."""
        unit = context.snapshot.allies[patient]
        missing = max(unit.max_health - unit.health, 0.0)
        pressure = len(context.threats_to(patient))
        return missing / max(unit.max_health, 1.0) + 0.1 * min(pressure, 3)

    def propose(self, context: Context, memory, telemetry, ledger=None) -> List:
        fingerprint = self.config.get("_fingerprint", "")
        snapshot_id = context.snapshot.snapshot_id
        plans: List = []
        medivacs = [a for a, cap in sorted(context.capabilities.items())
                    if cap.healer and context.snapshot.allies[a].alive]
        if not medivacs:
            return plans
        telemetry.count(self.name, "applicable", len(medivacs))
        owners = context.heal_occupancy()          # observed occupancy
        for actor in medivacs:
            instance = memory.instance_for(self.name, actor)
            # 1. observed keep-heal order (explicit extension only)
            keep = context.keep_heal_action(actor)
            if keep is not None:
                target = context.snapshot.keep_heal_order.get(actor, {}).get("target")
                plans.append(make_plan(
                    self.name, {actor: keep}, target or "observed_order",
                    {actor: keep},
                    [EffectClaim("heal", target or "", None, max(context.H, 1),
                                 HEURISTIC, "observed_heal_order_retained")],
                    ResourceClaims(controls=(actor,)),
                    Commitment("sustained_heal", 0,
                               max_loops=30 * max(context.H, 1)),
                    {}, snapshot_id, fingerprint, provider="heal_support",
                    beneficiaries=(target,) if target else (),
                    instance_id=instance.instance_id if instance else None,
                    rationale=f"keep heal order on {target}"))
            # 2. ALL legal patients as candidates (occupancy-aware urgency)
            candidates: List[str] = []
            for action, info in sorted(context.snapshot.heal_candidates
                                       .get(actor, {}).items()):
                patient = info.get("target")
                if patient and patient in context.snapshot.allies:
                    candidates.append(patient)
            candidates.sort(key=lambda p: (-self._urgency(context, p), p))
            for patient in candidates:
                action = context.heal_action(actor, patient)
                if action is None:
                    continue
                owner = owners.get(patient)
                taken = owner is not None and owner != actor
                switch = 0.2 if instance and instance.goal not in (
                        patient, "observed_order") else 0.0
                plans.append(make_plan(
                    self.name, {actor: action}, patient, {actor: action},
                    [EffectClaim("heal", patient, None, max(context.H, 1),
                                 HEURISTIC,
                                 "heal_opportunity_proxy_taken" if taken
                                 else "heal_opportunity_proxy")],
                    ResourceClaims(controls=(actor,),
                                   healing=((patient, 1.0),)),
                    Commitment("sustained_heal", 0, max_loops=30 * max(context.H, 1)),
                    {"switch_cost": switch}, snapshot_id, fingerprint,
                    provider="heal_support", beneficiaries=(patient,),
                    dependencies=(Dependency("heal_owner", (patient, actor)),),
                    instance_id=instance.instance_id if instance else None,
                    memory_delta=({"kind": "create", "skill": self.name,
                                   "actors": [actor], "goal": patient,
                                   "phase": "sustained_heal",
                                   "max_loops": 30 * max(context.H, 1)}
                                  if instance is None else None),
                    rationale=f"heal {patient}" +
                              (" (occupied by observed order)" if taken else "")))
            # 3. approach the neediest unowned patient
            owned = set(owners) - {p for p, a in owners.items() if a == actor}
            approachable = [p for p in sorted(
                context.snapshot.allies,
                key=lambda p: (-self._urgency(context, p), p))
                if 0 < context.snapshot.allies[p].health
                < context.snapshot.allies[p].max_health
                and not context.capabilities[p].healer
                and p not in owned]
            if approachable and not candidates:
                patient = approachable[0]
                move = best_move_toward(
                    context, actor, context.snapshot.allies[patient].position)
                if move is not None:
                    plans.append(make_plan(
                        self.name, {actor: move}, patient, {actor: move},
                        [EffectClaim("position", actor,
                                     move_progress(context, actor, move,
                                                   context.snapshot.allies[patient].position),
                                     max(context.H, 1), HEURISTIC,
                                     "approach_patient")],
                        ResourceClaims(controls=(actor,),
                                       space=(space_claim(context, actor, move),)),
                        Commitment("approach_patient", 0,
                                   max_loops=12 * max(context.H, 1)),
                        {"progress": abs(move_progress(
                            context, actor, move,
                            context.snapshot.allies[patient].position))},
                        snapshot_id, fingerprint, provider="heal_support",
                        beneficiaries=(patient,),
                        rationale=f"approach patient {patient}"))
            # 4. own danger: shared safe-move RETREAT candidate competes with
            #    healing (S2.10) — never skipped because of the healer class.
            risk = context.risk_at(actor, context.snapshot.allies[actor].position)
            if risk.threats:   # only real domain-compatible threats count
                from .common import best_move_away
                move = best_move_away(context, actor, list(risk.threats))
                if move is not None:
                    plans.append(make_plan(
                        self.name, {actor: move}, "escape", {actor: move},
                        [EffectClaim("position", actor,
                                     move_progress(context, actor, move,
                                                   (0., 0.)), max(context.H, 1),
                                     HEURISTIC, "medivac_self_danger_move")],
                        ResourceClaims(controls=(actor,),
                                       space=(space_claim(context, actor, move),)),
                        Commitment("escape", 0, max_loops=3 * max(context.H, 1)),
                        {}, snapshot_id, fingerprint, provider="heal_support",
                        rationale="leave domain-compatible danger"))
            # 5. no patients: escort the largest effective fighting group
            if not candidates and not approachable:
                goal = self._escort_point(context, actor)
                if goal is not None:
                    if distance(context.snapshot.allies[actor].position, goal) \
                            > self.follow_distance:
                        move = best_move_toward(context, actor, goal)
                        if move is not None:
                            plans.append(make_plan(
                                self.name, {actor: move}, "escort", {actor: move},
                                [EffectClaim("position", actor,
                                             move_progress(context, actor, move, goal),
                                             max(context.H, 1), HEURISTIC,
                                             "escort_follow")],
                                ResourceClaims(
                                    controls=(actor,),
                                    space=(space_claim(context, actor, move),)),
                                Commitment("escort", 0, max_loops=6 * max(context.H, 1)),
                                {"progress": abs(move_progress(
                                    context, actor, move, goal))},
                                snapshot_id, fingerprint,
                                provider="heal_support",
                                rationale="follow main fighting group"))
                    else:
                        plans.append(make_plan(
                            self.name, {actor: STOP}, "escort", {actor: STOP},
                            [EffectClaim("position", actor, None, max(context.H, 1),
                                         HEURISTIC, "escort_hold")],
                            ResourceClaims(controls=(actor,)),
                            Commitment("escort", 0, max_loops=6 * max(context.H, 1)),
                            {"hold_value": 0.5}, snapshot_id, fingerprint,
                            provider="heal_support",
                            rationale="hold escort position"))
        telemetry.count(self.name, "proposed", len(plans))
        return plans

    def _escort_point(self, context: Context, actor: str):
        """Largest effective fighting group's centroid — never the nearest
        enemy, never the unconditional whole-team center."""
        groups = [g for g in context.engagement_groups()
                  if any(context.attackable_enemies(a) or
                         context.snapshot.enemies for a in g)]
        if not groups:
            return None
        group = max(groups, key=lambda g: (len(g), g))
        positions = [context.snapshot.allies[a].position for a in group]
        n = len(positions)
        return (sum(p[0] for p in positions) / n,
                sum(p[1] for p in positions) / n)
