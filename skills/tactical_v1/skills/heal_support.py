"""S2 HealSupport: Medivac sustained healing.

Heal need is based on healable HEALTH loss only (shields are not healable).
The observed heal occupation (a committed heal order visible in the current
environment) and this step's new assignment are tracked separately. A
healed ally keeps its own action; only plans that explicitly need the
patient to move claim its control. Normal attack/kiting logic never applies
to a Medivac.
"""
from typing import Dict, List, Optional

from ..context import Context, STOP, distance
from ..contracts import (CALIBRATED, CandidatePlan, Commitment, EffectClaim,
                         HEURISTIC, ResourceClaims)
from .common import best_move_toward, make_plan


class HealSupportSkill:
    name = "heal_support"

    def __init__(self, config: Dict):
        self.config = config
        params = config.get("heal_support", {})
        self.max_heal_per_step = params.get("max_heal_per_step", 8.0)
        self.switch_margin = params.get("switch_margin", 6.0)
        self.follow_distance = params.get("follow_distance", 5.0)

    def _patients(self, context: Context) -> List[str]:
        """Allies with healable health loss, worst first."""
        def need(actor):
            unit = context.snapshot.allies[actor]
            return (-(unit.max_health - unit.health), actor)
        return sorted((a for a, u in sorted(context.snapshot.allies.items())
                       if u.alive and u.health < u.max_health and not
                       context.capabilities[a].healer), key=need)

    def propose(self, context: Context, memory, telemetry, ledger=None
                ) -> List[CandidatePlan]:
        fingerprint = self.config.get("_fingerprint", "")
        snapshot_id = context.snapshot.snapshot_id
        plans: List[CandidatePlan] = []
        medivacs = [a for a, cap in sorted(context.capabilities.items())
                    if cap.healer and context.snapshot.allies[a].alive]
        if not medivacs:
            return plans
        telemetry.count(self.name, "applicable", len(medivacs))
        for actor in medivacs:
            # 1. Observed keep-heal order: the env exposes a dedicated legal
            #    action only when the extension is enabled. Prefer it when the
            #    current patient still needs healing.
            keep = context.keep_heal_action(actor)
            if keep is not None:
                candidate = context.snapshot.keep_heal_order.get(actor, {})
                target = candidate.get("target")
                plans.append(make_plan(
                    self.name, {actor: keep}, target or "observed_order",
                    {actor: keep},
                    [EffectClaim("heal", target or "", self.max_heal_per_step,
                                 max(context.snapshot.step_mul, 1), CALIBRATED,
                                 "observed_heal_order_retained", 0.9)],
                    ResourceClaims(controls=(actor,),
                                   healing=((target, self.max_heal_per_step),)
                                   if target else ()),
                    Commitment("sustained_heal", min_steps=1, max_steps=8),
                    {"switch_cost": 0.0},
                    snapshot_id, fingerprint, provider="heal_support",
                    beneficiaries=(target,) if target else (),
                    rationale=f"keep heal order on {target}"))
            # 2. New heal assignment on the neediest patient in range.
            chosen: Optional[str] = None
            for patient in self._patients(context):
                action = context.heal_action(actor, patient)
                if action is not None:
                    already = (ledger.healing.get(patient, 0.0)
                               if ledger else 0.0)
                    patient_unit = context.snapshot.allies[patient]
                    missing = max(patient_unit.max_health - patient_unit.health
                                  - already, 0.0)
                    plans.append(make_plan(
                        self.name, {actor: action}, patient, {actor: action},
                        [EffectClaim("heal", patient,
                                     min(missing, self.max_heal_per_step),
                                     max(context.snapshot.step_mul, 1),
                                     CALIBRATED, "smac_heal_action", 0.9)],
                        ResourceClaims(controls=(actor,),
                                       healing=((patient,
                                                 min(missing, self.max_heal_per_step)),)),
                        Commitment("sustained_heal", min_steps=2, max_steps=30),
                        {"switch_cost": 0.0}, snapshot_id, fingerprint,
                        provider="heal_support", beneficiaries=(patient,),
                        rationale=f"heal {patient}"))
                    chosen = patient
                    break
            # 3. No heal action in range: approach the neediest patient.
            if keep is None and chosen is None:
                patients = self._patients(context)
                if patients:
                    patient = patients[0]
                    move = best_move_toward(
                        context, actor, context.snapshot.allies[patient].position)
                    if move is not None:
                        progress = _progress(context, actor, move,
                                             context.snapshot.allies[patient].position)
                        plans.append(make_plan(
                            self.name, {actor: move}, patient, {actor: move},
                            [EffectClaim("position", actor, progress,
                                         max(context.snapshot.step_mul, 1),
                                         HEURISTIC, "approach_patient", 0.7)],
                            ResourceClaims(controls=(actor,), space=(f"{actor}@{move}",)),
                            Commitment("approach_patient", min_steps=0, max_steps=12),
                            {"move_opportunity_cost": 0.0}, snapshot_id,
                            fingerprint, provider="heal_support",
                            beneficiaries=(patient,),
                            rationale=f"approach patient {patient}"))
                else:
                    # 4. No patients: follow the main combat group, never the enemy.
                    group = _combat_group_center(context, actor)
                    if group is not None:
                        if distance(context.snapshot.allies[actor].position, group) \
                                > self.follow_distance:
                            move = best_move_toward(context, actor, group)
                            if move is not None:
                                plans.append(make_plan(
                                    self.name, {actor: move}, "follow_group",
                                    {actor: move},
                                    [EffectClaim("position", actor,
                                                 _progress(context, actor, move, group),
                                                 max(context.snapshot.step_mul, 1),
                                                 HEURISTIC, "escort_follow", 0.6)],
                                    ResourceClaims(controls=(actor,)),
                                    Commitment("escort", min_steps=0, max_steps=6),
                                    {"support_follow": 0.3}, snapshot_id,
                                    fingerprint, provider="heal_support",
                                    rationale="follow combat group"))
                        else:
                            plans.append(make_plan(
                                self.name, {actor: STOP}, "escort_hold",
                                {actor: STOP},
                                [EffectClaim("position", actor, 0.0,
                                             max(context.snapshot.step_mul, 1),
                                             HEURISTIC, "escort_hold", 0.6)],
                                ResourceClaims(controls=(actor,)),
                                Commitment("escort", min_steps=0, max_steps=6),
                                {"support_follow": 0.2}, snapshot_id,
                                fingerprint, provider="heal_support",
                                rationale="hold escort position"))
        telemetry.count(self.name, "proposed", len(plans))
        return plans


def _progress(context, actor, action, target) -> float:
    from .common import move_progress
    return move_progress(context, actor, action, target)


def _combat_group_center(context: Context, actor: str):
    allies = [u.position for a, u in sorted(context.snapshot.allies.items())
              if u.alive and not context.capabilities[a].healer]
    if not allies:
        return None
    n = len(allies)
    return (sum(p[0] for p in allies) / n, sum(p[1] for p in allies) / n)
