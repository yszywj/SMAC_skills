"""S2 HealSupport: joint healing, sustained execution, safe escort (D8).

Observed orders keep ownership until a later observation releases them; NEW
assignments this step obey the verified capacity (one active healer per
patient). All legal patients stay candidates; medivac+patient joint moves
(atomic, ≤2) pay the patient's lost output explicitly. Follow targets exist
even with no visible enemy. Signed progress; no propose-side mutation.
"""
from typing import Dict, List, Optional

from ..actions import KEEP_HEAL_ORDER
from ..context import Context, STOP, distance
from ..contracts import Commitment, Dependency, EffectClaim, HEURISTIC, ResourceClaims
from .common import best_move_away, best_move_toward, make_plan, move_progress, space_claim


class HealSupportSkill:
    name = "heal_support"

    def __init__(self, config: Dict):
        self.config = config
        params = config.get("heal_support", {})
        self.follow_distance = params.get("follow_distance", 5.0)
        self.switch_margin = params.get("switch_margin", 0.3)

    def _urgency(self, context: Context, patient: str) -> float:
        unit = context.snapshot.allies[patient]
        missing = max(unit.max_health - unit.health, 0.0)
        pressure = len(context.threats_to(patient))
        return missing / max(unit.max_health, 1.0) + 0.1 * min(pressure, 3)

    def _escort_point(self, context: Context, actor: str):
        """Best follow target: the wounded-serving squad or the largest
        GROUND fighting group — works with and without visible enemies."""
        ground = [a for a, u in sorted(context.snapshot.allies.items())
                  if u.alive and not u.is_flying
                  and not context.capabilities[a].healer]
        if not ground:
            return None
        wounded_near = sorted(
            ground, key=lambda a: (-self._urgency(context, a), a))
        focus = wounded_near[0] if self._urgency(context, wounded_near[0]) > 0 \
            else max(ground, key=lambda a: (1, a))
        squad = next((g for g in context.engagement_groups()
                      if focus in g), [focus])
        positions = [context.snapshot.allies[a].position for a in squad]
        n = len(positions)
        return (sum(p[0] for p in positions) / n,
                sum(p[1] for p in positions) / n)

    def propose(self, context: Context, memory, telemetry, ledger=None) -> List:
        fingerprint = self.config.get("_fingerprint", "")
        snapshot_id = context.snapshot.snapshot_id
        plans: List = []
        medivacs = [a for a, cap in sorted(context.capabilities.items())
                    if cap.healer and context.snapshot.allies[a].alive]
        if not medivacs:
            return plans
        telemetry.count(self.name, "applicable", len(medivacs))
        owners = context.heal_occupancy()
        for actor in medivacs:
            instance = memory.instance_for(self.name, actor)
            # 1. observed keep-heal order: explicit extension action only
            keep = context.keep_heal_action(actor)
            if keep is not None:
                target = context.snapshot.keep_heal_order.get(actor, {}).get(
                    "target")
                plans.append(make_plan(
                    self.name, {actor: keep}, target or "observed_order",
                    {actor: keep},
                    [EffectClaim("heal", target or "", None, max(context.H, 1),
                                 HEURISTIC, "observed_heal_order_retained")],
                    ResourceClaims(controls=(actor,)),
                    Commitment("sustained_heal", 0, max_loops=30 * context.H),
                    {}, snapshot_id, fingerprint, provider="heal_support",
                    beneficiaries=(target,) if target else (),
                    dependencies=(Dependency("heal_owner",
                                             (target, actor)),) if target else (),
                    instance_id=instance.instance_id if instance else None,
                    rationale=f"keep heal order on {target}"))
            # 2. ALL legal patients (occupancy-aware, urgency-ordered)
            patients = []
            for action, info in sorted(context.snapshot.heal_candidates
                                       .get(actor, {}).items()):
                patient = info.get("target")
                if patient and patient in context.snapshot.allies:
                    patients.append(patient)
            patients.sort(key=lambda p: (-self._urgency(context, p), p))
            for patient in patients:
                action = context.heal_action(actor, patient)
                if action is None:
                    continue
                taken = owners.get(patient) not in (None, actor)
                switch = 0.2 if instance is not None and \
                    instance.goal not in (patient, "observed_order") else 0.0
                plans.append(make_plan(
                    self.name, {actor: action}, patient, {actor: action},
                    [EffectClaim("heal", patient, None, max(context.H, 1),
                                 HEURISTIC,
                                 "heal_opportunity_taken" if taken
                                 else "heal_opportunity")],
                    ResourceClaims(controls=(actor,),
                                   healing=((patient, 1.0),)),
                    Commitment("sustained_heal", 0, max_loops=30 * context.H),
                    {"switch_cost": switch}, snapshot_id, fingerprint,
                    provider="heal_support", beneficiaries=(patient,),
                    dependencies=(Dependency("heal_owner", (patient, actor)),),
                    instance_id=instance.instance_id if instance else None,
                    memory_delta=({"kind": "create", "skill": self.name,
                                   "actors": [actor], "goal": patient,
                                   "goal_kind": "ally",
                                   "phase": "sustained_heal",
                                   "max_loops": 30 * context.H}
                                  if instance is None else
                                  {"kind": "retarget",
                                   "instance_id": instance.instance_id,
                                   "goal": patient, "goal_kind": "ally"}
                                  if instance.goal != patient else None),
                    rationale=f"heal {patient}" +
                              (" (owned by observed order)" if taken else "")))
            # 3. joint medivac+patient move (atomic, few): both step closer
            if patients:
                joint_budget = 2
                for patient in patients[:joint_budget]:
                    goal = context.snapshot.allies[patient].position
                    meet = ((goal[0] + context.snapshot.allies[actor].position[0]) / 2,
                            (goal[1] + context.snapshot.allies[actor].position[1]) / 2)
                    m_move = best_move_toward(context, actor, meet)
                    p_move = best_move_toward(context, patient, meet)
                    if m_move is None or p_move is None:
                        continue
                    # patient pays lost output for leaving its position
                    patient_attack = context.attackable_enemies(patient)
                    plans.append(make_plan(
                        self.name, {actor: m_move, patient: p_move},
                        patient, {actor: m_move, patient: p_move},
                        [EffectClaim("position", actor,
                                     move_progress(context, actor, m_move, meet),
                                     max(context.H, 1), HEURISTIC,
                                     "medivac_patient_meet"),
                         EffectClaim("position", patient,
                                     move_progress(context, patient, p_move, meet),
                                     max(context.H, 1), HEURISTIC,
                                     "patient_approach")],
                        ResourceClaims(
                            controls=(actor, patient),
                            space=(space_claim(context, actor, m_move),
                                   space_claim(context, patient, p_move))),
                        Commitment("meet_patient", 0, max_loops=6 * context.H),
                        {"switch_cost": 0.3 if patient_attack else 0.0},
                        snapshot_id, fingerprint, provider="heal_support",
                        atomic=True,
                        beneficiaries=(patient,),
                        dependencies=(Dependency("member", (actor, patient)),),
                        rationale=f"joint approach: {actor} and {patient}"))
            # 4. own danger: shared safe-move escape competes (S2.10)
            risk = context.risk_at(actor, context.snapshot.allies[actor].position)
            if risk.threats:
                move = best_move_away(context, actor, list(risk.threats))
                if move is not None:
                    plans.append(make_plan(
                        self.name, {actor: move}, "escape", {actor: move},
                        [EffectClaim("position", actor,
                                     move_progress(context, actor, move, (0., 0.)),
                                     max(context.H, 1), HEURISTIC,
                                     "medivac_self_danger_move")],
                        ResourceClaims(controls=(actor,),
                                       space=(space_claim(context, actor, move),)),
                        Commitment("escape", 0, max_loops=3 * context.H),
                        {}, snapshot_id, fingerprint, provider="heal_support",
                        rationale="leave domain-compatible danger"))
            # 5. escort: follow the effective friendly squad (also without
            #    visible enemies)
            if not patients:
                goal = self._escort_point(context, actor)
                if goal is not None:
                    if distance(context.snapshot.allies[actor].position, goal) \
                            > self.follow_distance:
                        move = best_move_toward(context, actor, goal)
                        if move is not None:
                            plans.append(make_plan(
                                self.name, {actor: move}, "escort",
                                {actor: move},
                                [EffectClaim("position", actor,
                                             move_progress(context, actor, move,
                                                           goal),
                                             max(context.H, 1), HEURISTIC,
                                             "escort_follow")],
                                ResourceClaims(
                                    controls=(actor,),
                                    space=(space_claim(context, actor, move),)),
                                Commitment("escort", 0, max_loops=6 * context.H),
                                {}, snapshot_id, fingerprint,
                                provider="heal_support",
                                rationale="follow effective friendly squad"))
                    else:
                        plans.append(make_plan(
                            self.name, {actor: STOP}, "escort", {actor: STOP},
                            [EffectClaim("position", actor, None,
                                         max(context.H, 1), HEURISTIC,
                                         "escort_hold")],
                            ResourceClaims(controls=(actor,)),
                            Commitment("escort", 0, max_loops=6 * context.H),
                            {"hold_value": 0.4}, snapshot_id, fingerprint,
                            provider="heal_support",
                            rationale="hold escort position"))
        telemetry.count(self.name, "proposed", len(plans))
        return plans
