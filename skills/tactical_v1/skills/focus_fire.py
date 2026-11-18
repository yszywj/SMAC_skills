"""G1 FocusFire: ordinary fire allocation and joint interception (D1).

Candidate classes (original target / kill-completable / high real threat /
nearby effective) are INTERLEAVED so no class exhausts the quota by id
order. Colossus candidates come from the enabled S1 provider's target
ranking. Joint interception members are picked by window opportunity and
effect timing, with one alternative combination so important members may
keep healing or escaping instead.
"""
from typing import Dict, List, Optional

from ..context import Context, IN_RANGE
from ..battle_pressure import enemy_pressure
from ..targeting import next_shot_target, planning_enabled, shared_target_plan
from ..unit_value_view import target_value as enemy_target_value
from ..contracts import (Commitment, Dependency, EffectClaim, HEURISTIC,
                         ResourceClaims)
from .common import h_loops, make_plan, space_claim


class FocusFireSkill:
    name = "focus_fire"

    def __init__(self, config: Dict, providers: Optional[Dict] = None):
        self.config = config
        params = config.get("focus_fire", {})
        self.kill_window_h = params.get("kill_window_h", 3.0)
        self.joint_kill_max_members = params.get("joint_kill_max_members", 5)
        self.max_candidates = config.get("search", {}).get(
            "max_candidates_per_unit", 6)
        self.providers = providers or {}

    # -- impact model -------------------------------------------------------
    def _impact(self, context: Context, actor: str, enemy_id: str
                ) -> List[EffectClaim]:
        provider = self.providers.get("splash_fire")
        capability = context.capabilities[actor]
        if capability.specialization == "splash_fire" and provider is not None:
            return provider.impact_claims(context, actor, enemy_id)
        estimate = context.fire_estimate(actor, enemy_id)
        window = max(context.H, 1)
        if estimate.window_shots >= 1 and estimate.damage_est is not None:
            return [EffectClaim("damage", enemy_id, estimate.damage_est, window,
                                estimate.evidence,
                                "shared_damage_model_window_shot",
                                lower=estimate.damage_lo, upper=estimate.damage_hi)]
        return [EffectClaim("damage", enemy_id, None, window, HEURISTIC,
                            "attack_command_without_window_damage")]

    def _window_damage(self, context: Context, actor: str, enemy_id: str) -> float:
        for effect in self._impact(context, actor, enemy_id):
            if effect.kind == "damage" and effect.target == enemy_id \
                    and effect.amount is not None:
                return effect.amount
        return 0.0

    # -- candidate ordering -------------------------------------------------
    def _ordered_targets(self, context: Context, actor: str, issuable,
                         original, assigned=None) -> List[str]:
        provider = self.providers.get("splash_fire")
        capability = context.capabilities[actor]
        if capability.specialization == "splash_fire" and provider is not None:
            ranked = provider.target_candidates(context, actor)
            # interleave the provider ranking with killable/threat/nearby
            classes = [ranked]
        else:
            classes = []
        killable = [e for e in issuable
                    if self._window_damage(context, actor, e)
                    >= context.snapshot.enemies[e].effective_health]
        # A boolean "threatens anyone" leaves all ordinary attackers sorted
        # by ID, so an eligible high-pressure rear attacker can miss the
        # candidate quota entirely. Rank compatible objective pressure;
        # known immediate kills and the established target remain classes.
        # Object-value ordering (legacy pure evaluator, adapted): a bounded
        # candidate-generation preference ONLY — damage, kills and threat
        # relief stay in their own evaluator ledgers, so this never adds a
        # second score for the same volley.
        valuable = sorted((e for e in issuable if enemy_target_value(context, e) > 0),
                          key=lambda e: (-enemy_target_value(context, e), e))
        high_threat = sorted((e for e in issuable if enemy_pressure(context, e) > 0),
                             key=lambda e: (-enemy_pressure(context, e), e))
        nearby = sorted(issuable, key=lambda e: (
            context.snapshot.relations[actor][e].distance, e))
        classes = [killable, [original] if original else [],
                   [assigned] if assigned in issuable else [], high_threat,
                   valuable, nearby, *classes]
        # round-robin interleave: one from each class per round
        ordered, seen = [], set()
        for round_index in range(max(len(c) for c in classes if c)):
            for group in classes:
                if not group:
                    continue
                if round_index < len(group) and group[round_index] not in seen:
                    seen.add(group[round_index])
                    ordered.append(group[round_index])
        return ordered

    def propose(self, context: Context, memory, telemetry,
                ledger=None) -> List:
        fingerprint = self.config.get("_fingerprint", "")
        snapshot_id = context.snapshot.snapshot_id
        plans: List = []
        target_plan = shared_target_plan(context, memory, self.config)
        attackers: Dict[str, List[str]] = {}
        for actor, unit in sorted(context.snapshot.allies.items()):
            if not unit.alive:
                continue
            capability = context.capabilities[actor]
            if capability.healer or capability.suicide_attack:
                continue
            issuable = [e for e in sorted(context.snapshot.enemies)
                        if context.can_issue_attack(actor, e)]
            if issuable:
                attackers[actor] = issuable
                telemetry.count(self.name, "applicable")
        for actor, issuable in sorted(attackers.items()):
            instance = memory.instance_for(self.name, actor)
            # A cooldown reposition replaces the G1 task, but it must not
            # erase the actual shot target that G3 positioned around.
            original = (next_shot_target(context, memory, actor, issuable)
                        if planning_enabled(self.config) else
                        memory.recent_attack_target(context, actor, issuable))
            assignment = target_plan.assignments.get(actor)
            for enemy_id in self._ordered_targets(
                    context, actor, issuable, original,
                    assignment.target if assignment else None)[:self.max_candidates]:
                action = context.attack_action(actor, enemy_id)
                if action is None:
                    continue
                amount = self._window_damage(context, actor, enemy_id)
                effects = self._impact(context, actor, enemy_id)
                capability = context.capabilities[actor]
                provider = ("splash_fire"
                            if capability.specialization == "splash_fire"
                            and self.providers.get("splash_fire") else None)
                delta = None
                if instance is None:
                    delta = {"kind": "create", "skill": self.name,
                             "actors": [actor], "goal": enemy_id,
                             "goal_kind": "enemy", "phase": "attacking"}
                elif instance.goal != enemy_id:
                    delta = {"kind": "retarget", "instance_id":
                             instance.instance_id, "goal": enemy_id,
                             "goal_kind": "enemy"}   # keep budget, update goal
                switch = 0.2 if original is not None and original != enemy_id \
                    else 0.0
                plans.append(make_plan(
                    self.name, {actor: action}, enemy_id, {actor: action},
                    effects,
                    ResourceClaims(controls=(actor,),
                                   fire=((enemy_id, amount),) if amount else ()),
                    Commitment("attacking", min_loops=0),
                    {"switch_cost": switch},
                    snapshot_id, fingerprint, provider=provider,
                    instance_id=instance.instance_id if instance else None,
                    memory_delta=delta,
                    goal_kind="enemy",
                    goal_data={"target_id": enemy_id,
                               "allocated_target": assignment.target if assignment else None,
                               "target_horizon_loops": target_plan.horizon_loops},
                    rationale=f"attack {enemy_id}"))
        plans.extend(self._joint_interceptions(
            context, memory, telemetry, attackers, fingerprint, snapshot_id,
            target_plan))
        telemetry.count(self.name, "proposed", len(plans))
        return plans

    def _joint_interceptions(self, context, memory, telemetry, attackers,
                             fingerprint, snapshot_id, target_plan=None) -> List:
        plans: List = []
        max_atomic = self.config.get("search", {}).get(
            "max_atomic_candidates", 12)
        produced = 0
        allocated_targets = {assignment.target for assignment in
                             (target_plan.assignments.values() if target_plan else ())}
        # -- enemy bombers: complete spread + minimal-sufficient fire plans,
        # built from the shared intercept facts (same short-horizon truth as
        # the execution duties). The death-burst risk floor stays in the
        # evaluator; these plans claim only the window-qualified shots.
        from ..intercept import intercept_plan
        for row in intercept_plan(context, memory, self.config):
            if produced >= max_atomic:
                break
            members = {actor: action for actor, action, _ in row["shooters"]}
            members.update(dict(row["spread"]))
            if len(members) < 1:
                continue
            # Candidate contract for the specialized whole-group validation
            # (validate_plans -> validate_intercept_plan): the plan's OWN
            # member/action lists and claim type are frozen here; final
            # eligibility is RECOMPUTED from final actions, never trusted
            # from these declared facts.
            shooter_actions = {actor: action for actor, action, _ in row["shooters"]}
            spread_actions = dict(row["spread"])
            intercept_type = ('safe_intercept' if row.get('safe')
                              else 'loss_control_intercept'
                              if row.get('type') == 'loss_control_intercept'
                              else 'ordinary_joint_fire' if row['shooters']
                              else 'spread_only')
            requires_lethal = intercept_type in ('safe_intercept',
                                                 'loss_control_intercept')
            effects, fire = [], []
            for actor, action, damage_lo in row["shooters"]:
                estimate = context.fire_estimate(actor, row["enemy"])
                effects.append(EffectClaim(
                    "damage", row["enemy"], estimate.damage_est, context.H,
                    estimate.evidence, "shared_damage_model_window_shot",
                    lower=estimate.damage_lo, upper=estimate.damage_hi))
                fire.append((row["enemy"], damage_lo))
            for actor, _ in row["spread"]:
                effects.append(EffectClaim("position", actor, .5,
                                           max(context.H, 1), HEURISTIC,
                                           "burst_spread_loss_reduction"))
            produced += 1
            plans.append(make_plan(
                self.name, dict(members), row["enemy"], dict(members),
                effects,
                ResourceClaims(controls=tuple(sorted(members)), fire=tuple(fire)),
                Commitment("intercepting", 0,
                           max_loops=h_loops(context, self.kill_window_h)),
                {}, snapshot_id, fingerprint, atomic=True,
                goal_kind="enemy",
                goal_data={"target_id": row["enemy"],
                           "validation_kind": "bomber_intercept",
                           "intercept_type": intercept_type,
                           "shooter_actions": shooter_actions,
                           "spread_actions": spread_actions,
                           "protected_members": list(row.get('protected_members', ())),
                           "already_exposed_members": list(row.get('already_exposed', ())),
                           "conservative_kill_loops": row.get('conservative_kill_loops'),
                           "requires_lethal_damage": requires_lethal,
                           "entry_h": round(row["entry_h"], 3),
                           "spread_members": [a for a, _ in row["spread"]]},
                dependencies=(Dependency("member", tuple(sorted(members))),
                              Dependency("target_valid", (row["enemy"],)))
                # fire_window (kill claim) only for plans that actually claim
                # a kill; ordinary joint fire must not be forced lethal.
                + ((Dependency("fire_window", (row["enemy"],)),)
                   if requires_lethal and row['shooters'] else ()),
                rationale=("joint explosion defense (" + intercept_type
                           + ") against " + row["enemy"]
                           + f" with {len(row['shooters'])} shooters"
                           + (f", {len(row['spread'])} spreading" if row["spread"] else ""))))
        # -- ordinary allocated high-pressure targets keep the generic
        # minimal-sufficient joint kill path. Imminent threats to members
        # in danger are added here: they must not be missed just because
        # the shared target plan allocated elsewhere — without a real
        # candidate entry the assess()-side joint rescue substitution would
        # have nothing to substitute with.
        threat_targets = set(allocated_targets)
        from ..combat_exchange import danger_evidence
        for member in sorted(context.snapshot.allies):
            unit = context.snapshot.allies.get(member)
            if unit is None or not unit.alive:
                continue
            cap = context.capabilities[member]
            if cap.healer or cap.suicide_attack:
                continue
            if not danger_evidence(context, member, horizon_h=2.)['immediate']:
                continue
            threat_targets.update(e for e in context.forecast_risk(
                member, unit.position).reachable_threats
                if e in context.snapshot.enemies)
        for enemy_id, enemy in sorted(context.snapshot.enemies.items()):
            if not context.enemy_capabilities[enemy_id].suicide_attack and not (
                    enemy_id in threat_targets
                    and enemy_pressure(context, enemy_id) > 0):
                continue
            # member pool ranked by opportunity: window damage, effect delay,
            # then the member's own foregone output (opportunity cost)
            pool = []
            for actor in sorted(attackers):
                if not context.can_issue_attack(actor, enemy_id):
                    continue
                estimate = context.fire_estimate(actor, enemy_id)
                if estimate.window_shots < 1 or not estimate.damage_est:
                    continue
                foregone = max(
                    (context.fire_estimate(actor, other).damage_est or 0.0)
                    for other in context.attackable_enemies(actor)
                    if other != enemy_id) if len(
                    context.attackable_enemies(actor)) > 1 else 0.0
                pool.append((estimate.effective_after_loops or 0,
                             -estimate.damage_est, foregone, actor,
                             estimate.damage_est))
            pool.sort()
            # primary minimal-sufficient set + one alternative combination
            # (the alternative leaves the best-timing member free to keep
            # healing / escaping / other output)
            for skip_first_member in (False, True):
                if produced >= max_atomic:
                    break
                cumulative, members, effects, fire = 0.0, {}, [], []
                member_pool = pool[1:] if (skip_first_member and pool) else pool
                for delay, neg_damage, foregone, actor, damage in member_pool:
                    if len(members) >= self.joint_kill_max_members:
                        break
                    action = context.attack_action(actor, enemy_id)
                    members[actor] = action
                    effects.extend(self._impact(context, actor, enemy_id))
                    fire.append((enemy_id, damage))
                    cumulative += damage
                    if cumulative >= enemy.effective_health and len(members) >= 2:
                        produced += 1
                        plans.append(make_plan(
                            self.name, dict(members), enemy_id, dict(members),
                            effects,
                            ResourceClaims(controls=tuple(sorted(members)),
                                           fire=tuple(fire)),
                            Commitment("intercepting", 0,
                                       max_loops=h_loops(context,
                                                         self.kill_window_h)),
                            {},
                            snapshot_id, fingerprint, atomic=True,
                            goal_kind="enemy",
                            goal_data={"target_id": enemy_id},
                            dependencies=(Dependency("member",
                                                     tuple(sorted(members))),
                                         Dependency("target_valid", (enemy_id,)),
                                         Dependency("fire_window", (enemy_id,))),
                            rationale=f"attempted joint interception of "
                                      f"{enemy_id} with {len(members)} attackers"))
                        break
        return plans
