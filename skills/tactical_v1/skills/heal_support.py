"""S2 HealSupport: joint healing, sustained execution, safe escort (D8).

Observed orders keep ownership until a later observation releases them; NEW
assignments this step obey the verified capacity (one active healer per
patient). All legal patients stay candidates; medivac+patient joint moves
(atomic, ≤2) pay the patient's lost output explicitly. Follow targets exist
even with no visible enemy. Signed progress; no propose-side mutation.
"""
import copy
from typing import Dict, List

from ..context import Context, STOP, distance
from ..contracts import Commitment, Dependency, EffectClaim, HEURISTIC, ResourceClaims
from ..healing import (approach_options, approach_facts, eligible_patient,
                       energy_ready, observed_patient, rescue_deadline, rescue_priority)
from .common import (best_move_away, best_move_toward, make_plan,
                     move_progress, space_claim, task_key_for)


class HealSupportSkill:
    name = "heal_support"

    def __init__(self, config: Dict):
        self.config = config
        params = config.get("heal_support", {})
        self.follow_distance = params.get("follow_distance", 5.0)
        self.switch_margin = params.get("switch_margin", 0.3)

    def _urgency(self, context: Context, patient: str) -> float:
        return rescue_priority(context, patient)

    def _escort_target(self, context: Context, actor: str):
        """Best follow target: the wounded-serving squad or the largest
        GROUND fighting group — works with and without visible enemies."""
        ground = [a for a, u in sorted(context.snapshot.allies.items())
                  if u.alive and not u.is_flying
                  and not context.capabilities[a].healer]
        if not ground:
            return None
        wounded_near = sorted(
            ground, key=lambda a: (-self._urgency(context, a), a))
        squads = [tuple(a for a in g if a in ground)
                  for g in context.engagement_groups()]
        squads = [g for g in squads if g]
        if self._urgency(context, wounded_near[0]) > 0:
            focus = wounded_near[0]
            squad = next((g for g in squads if focus in g), (focus,))
        else:
            origin = context.snapshot.allies[actor].position
            squad = min(squads or [(a,) for a in ground], key=lambda g: (
                -len(g), min(distance(origin, context.snapshot.allies[a].position) for a in g), g))
        positions = [context.snapshot.allies[a].position for a in squad]
        n = len(positions)
        return ((sum(p[0] for p in positions) / n,
                 sum(p[1] for p in positions) / n), tuple(sorted(squad)))

    def _with_tasks(self, context: Context, memory, plans: List) -> List:
        """Attach a pure, commit-only lifecycle change to every S2 duty.

        A meeting owns both units; healing owns only its healer. Following
        and holding the same squad share one budget even as its centre moves.
        A changed duty creates a replacement, never relabels an old heal task.
        """
        result = []
        loop = context.snapshot.game_loop
        for plan in plans:
            actors = plan.controlled_actors
            phase = plan.commitment.phase
            data = copy.deepcopy(plan.goal_data)
            members = tuple(data.get("escort_members", ()))
            if phase == "meet_patient":
                active = next((i for i in memory.instances.values()
                               if i.skill == self.name and i.phase == phase
                               and i.goal == plan.goal and data.get("healer") in i.actors
                               and i.termination_reason is None), None)
                # Moving just the healer and moving the pair are alternate
                # duties, not a way to restart the same rescue deadline.
                if active is not None and active.actors != actors:
                    continue
            duty_key = "|".join((phase, plan.goal_kind, plan.goal,
                                 ",".join(members)))
            # Escort retries follow the squad's identity, not a transient
            # waypoint cell that could change while the failed duty cools down.
            retry_key = memory.task_retry_key(
                self.name, (data["healer"],) if phase == "meet_patient" else actors,
                "internal" if phase == "escort" else plan.goal_kind,
                "escort:" + ",".join(members) if phase == "escort"
                else "approach:" + plan.goal if phase == "meet_patient"
                else plan.goal, data)
            instance = next((i for i in memory.instances.values()
                             if i.skill == self.name and i.actors == actors
                             and i.termination_reason is None
                             and i.extra.get("duty_key") == duty_key), None)
            if instance is not None and instance.max_loops is not None \
                    and loop - instance.start_loop >= instance.max_loops:
                continue            # observation phase owns timeout/finish
            if memory.retry_blocked(*retry_key, loop):
                continue
            data.update(duty_key=duty_key, retry_key=retry_key)
            plan.task_key = task_key_for(self.name, actors, plan.goal_kind,
                                         plan.goal, duty_key)
            plan.instance_id = instance.instance_id if instance else None
            if instance is None:
                plan.proposed_memory_delta = {
                    "kind": "create", "skill": self.name,
                    "actors": list(actors), "goal": plan.goal,
                    "goal_kind": plan.goal_kind, "phase": phase,
                    "max_loops": plan.commitment.max_loops, "extra": data}
            else:
                plan.proposed_memory_delta = {
                    "kind": "continue", "instance_id": instance.instance_id,
                    "goal": plan.goal, "goal_kind": plan.goal_kind,
                    "extra": data}
            result.append(plan)
        return result

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
        assignments = approach_options(context, memory, medivacs)
        for actor in medivacs:
            instance = memory.instance_for(self.name, actor)
            # 1. observed keep-heal order: explicit extension action only
            keep = context.keep_heal_action(actor)
            if keep is not None and energy_ready(context, actor):
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
                    goal_kind="ally" if target else "internal",
                    goal_data={"target_id": target} if target else {},
                    rationale=f"keep heal order on {target}"))
            # 2. ALL legal patients (occupancy-aware, urgency-ordered)
            patients = []
            for action, info in sorted(context.snapshot.heal_candidates
                                       .get(actor, {}).items()):
                patient = info.get("target")
                if patient and eligible_patient(context, patient) and energy_ready(context, actor) \
                        and owners.get(patient) in (None, actor):
                    patients.append(patient)
            patients.sort(key=lambda p: (-self._urgency(context, p), p))
            for patient in patients:
                action = context.heal_action(actor, patient)
                if action is None:
                    continue
                taken = owners.get(patient) not in (None, actor)
                previous = observed_patient(context, actor) or (instance.goal if instance else None)
                switch = self.switch_margin if previous not in (None, patient, "observed_order") else 0.0
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
                    goal_kind="ally",
                    goal_data={"target_id": patient},
                    rationale=f"heal {patient}" +
                              (" (owned by observed order)" if taken else "")))
            # 3. The healer approaches first. A second atomic option may move
            # the patient only when it saves meaningful arrival time, costs no
            # ready legal shot and can arrive before predicted lethal exposure.
            for patient in assignments.get(actor, ()):
                if patient in patients:
                    continue
                deadline = rescue_deadline(context, memory, actor, patient)
                remaining = max(0., (deadline - context.snapshot.game_loop) / max(context.H, 1))
                previous = observed_patient(context, actor) or (instance.goal if instance else None)
                switch = self.switch_margin if previous not in (None, patient, "observed_order", "escort", "escape") else 0.
                goal = context.snapshot.allies[patient].position
                solo = best_move_toward(context, actor, goal)
                if solo is not None and approach_facts(context, actor, patient, {actor: solo}, remaining):
                    plans.append(make_plan(
                        self.name, {actor: solo}, patient, {actor: solo},
                        [EffectClaim("position", actor, None, max(context.H, 1), HEURISTIC,
                                     "healer_only_approach_uncalibrated")],
                        ResourceClaims(controls=(actor,), space=(space_claim(context, actor, solo),)),
                        Commitment("meet_patient", 0, max_loops=6 * context.H), {"switch_cost": switch},
                        snapshot_id, fingerprint, provider="heal_support", beneficiaries=(patient,),
                        dependencies=(Dependency("heal_owner", (patient, actor)),),
                        goal_kind="meeting", goal_data={"healer": actor, "patient": patient,
                            "position": list(goal), "approach_mode": "healer_only",
                            "rescue_deadline_loop": deadline,
                            "objective": {"kind": "meeting"}},
                        rationale="healer approaches; patient retains combat control"))
                for patient in (patient,):
                    goal = context.snapshot.allies[patient].position
                    meet = ((goal[0] + context.snapshot.allies[actor].position[0]) / 2,
                            (goal[1] + context.snapshot.allies[actor].position[1]) / 2)
                    m_move = best_move_toward(context, actor, meet)
                    p_move = best_move_toward(context, patient, meet)
                    if m_move is None or p_move is None:
                        continue
                    if not approach_facts(context, actor, patient, {actor: m_move, patient: p_move}, remaining):
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
                        {"switch_cost": switch + (0.3 if patient_attack else 0.0)},
                        snapshot_id, fingerprint, provider="heal_support",
                        atomic=True,
                        beneficiaries=(patient,),
                        dependencies=(Dependency("member", (actor, patient)),
                                      Dependency("heal_owner", (patient, actor))),
                        goal_kind="meeting",
                        goal_data={"healer": actor, "patient": patient,
                                   "position": list(meet),
                                   "approach_mode": "patient_assisted",
                                   "rescue_deadline_loop": deadline,
                                   "objective": {"kind": "meeting"}},
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
                escort = self._escort_target(context, actor)
                if escort is not None:
                    goal, members = escort
                    goal_data = {"waypoint": list(goal),
                                 "escort_members": list(members),
                                 "follow_distance": self.follow_distance,
                                 "objective": {"kind": "escort_band",
                                               "position": list(goal),
                                               "r_min": 1.0,
                                               "r_max": self.follow_distance}}
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
                                goal_kind="position",
                                goal_data=goal_data,
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
                            goal_kind="position", goal_data=goal_data,
                            rationale="hold escort position"))
        plans = self._with_tasks(context, memory, plans)
        telemetry.count(self.name, "proposed", len(plans))
        return plans
