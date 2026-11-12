"""Final validation and bounded repair (A4).

Structured results: every problem is {type, proposal_id, actor, target,
drop_whole} — repair consumes these directly, never string-splitting. All
checks run on the COMPLETE final action list (baseline entries included).
"""
from typing import Dict, List, Optional, Tuple

from .actions import DEAD_NOOP, KEEP_HEAL_ORDER
from .context import Context
from .contracts import CandidatePlan, Dependency
from .ledger import Ledger

KNOWN_DEPENDENCY_KINDS = ("member", "target_valid", "fire_window",
                          "heal_owner", "slot")


def _problem(kind: str, drop_whole: bool, proposal_id: str = "",
             actor: str = "", target: str = "") -> Dict[str, object]:
    return {"type": kind, "proposal_id": proposal_id, "actor": actor,
            "target": target, "drop_whole": drop_whole}


def validate_actions(actions: List[int], context: Context,
                     n_agents: int) -> List[Dict[str, object]]:
    problems: List[Dict[str, object]] = []
    if len(actions) != n_agents:
        problems.append(_problem("length", False,
                                 target=f"{len(actions)}!={n_agents}"))
    for i, action in enumerate(actions):
        actor = f"u{i}"
        if not isinstance(action, int) or isinstance(action, bool):
            problems.append(_problem("type", False, actor=actor,
                                     target=repr(action)))
            continue
        unit = context.snapshot.allies.get(actor)
        legal = context.legal_actions(actor)
        if unit is not None and not unit.alive:
            if action != DEAD_NOOP:
                problems.append(_problem("dead_unit_action", True, actor=actor,
                                         target=str(action)))
            continue
        if action not in legal:
            problems.append(_problem("illegal_action", True, actor=actor,
                                     target=str(action)))
    return problems


def validate_plans(plans: List[CandidatePlan], context: Context,
                   actions: Dict[str, int],
                   report=None) -> List[Dict[str, object]]:
    """Structural + dependency validation of the plan set against the FINAL
    actions. ``report`` is the physical effects of ``actions`` and carries
    the window-qualified fire sources and heal assignments."""
    problems: List[Dict[str, object]] = []
    seen_control: Dict[str, str] = {}
    plan_by_id = {p.proposal_id: p for p in plans}
    for plan in plans:
        if set(plan.current_actions) != set(plan.controlled_actors):
            problems.append(_problem("actions_actors_mismatch", True,
                                     plan.proposal_id))
        for actor, action in plan.current_actions.items():
            holder = seen_control.get(actor)
            if holder is not None and holder != plan.proposal_id:
                problems.append(_problem("double_control", True,
                                         plan.proposal_id, actor, holder))
            seen_control[actor] = plan.proposal_id
            if actions.get(actor) != action:
                problems.append(_problem("plan_action_changed", True,
                                         plan.proposal_id, actor,
                                         str(action)))
            semantic = context.decode(actor, action)
            if not semantic.valid:
                problems.append(_problem("undecodable_action", True,
                                         plan.proposal_id, actor, str(action)))
                continue
            if semantic.kind == "attack":
                if plan.goal != semantic.target and not any(
                        e.target == semantic.target
                        for e in plan.expected_effects):
                    problems.append(_problem("goal_mismatch", True,
                                             plan.proposal_id, actor,
                                             semantic.target))
            elif semantic.kind in ("heal", KEEP_HEAL_ORDER):
                if semantic.target and semantic.target not in plan.beneficiaries \
                        and plan.goal not in (semantic.target, "observed_order"):
                    problems.append(_problem("heal_target_mismatch", True,
                                             plan.proposal_id, actor,
                                             semantic.target))
        # atomic completeness: every declared member is in THIS plan with the
        # task's action (real check, not list-membership tautology)
        if plan.atomic:
            for dependency in plan.dependencies:
                if dependency.kind != "member":
                    continue
                for actor in dependency.refs:
                    if actor not in plan.controlled_actors:
                        problems.append(_problem("atomic_member_missing", True,
                                                 plan.proposal_id, actor))
                    elif context.decode(actor, plan.current_actions.get(
                            actor, DEAD_NOOP)).kind != "attack":
                        problems.append(_problem("atomic_member_not_attacking",
                                                 True, plan.proposal_id, actor))
        # structured dependencies (A4)
        for dependency in plan.dependencies:
            if dependency.kind not in KNOWN_DEPENDENCY_KINDS:
                problems.append(_problem("unknown_dependency_kind", True,
                                         plan.proposal_id,
                                         target=dependency.kind))
            elif dependency.kind == "member":
                for actor in dependency.refs:
                    if actor not in context.snapshot.allies \
                            or not context.snapshot.allies[actor].alive:
                        problems.append(_problem("member_dead", True,
                                                 plan.proposal_id, actor))
                    elif actions.get(actor) != plan.current_actions.get(actor):
                        problems.append(_problem("member_action_lost", True,
                                                 plan.proposal_id, actor))
            elif dependency.kind == "target_valid":
                for enemy_id in dependency.refs:
                    if enemy_id not in context.snapshot.enemies:
                        problems.append(_problem("target_lost", True,
                                                 plan.proposal_id,
                                                 target=enemy_id))
            elif dependency.kind == "fire_window":
                # every promised fire member is window-qualified NOW; the
                # total is re-derived from the physical report, not strings
                for enemy_id in dependency.refs:
                    qualified = (report.window_sources.get(enemy_id, [])
                                 if report else [])
                    promised = [a for a in plan.controlled_actors
                                if plan.current_actions.get(a) is not None
                                and context.decode(
                                    a, plan.current_actions[a]).target == enemy_id]
                    missing = [a for a in promised if a not in qualified]
                    if missing:
                        problems.append(_problem("fire_window_not_met", True,
                                                 plan.proposal_id,
                                                 target=",".join(missing)))
            elif dependency.kind == "heal_owner":
                patient, healer = dependency.refs[0], dependency.refs[1]
                owners = context.heal_occupancy()
                owner = owners.get(patient)
                if owner is not None and owner != healer:
                    problems.append(_problem("heal_occupied_by_observed_order",
                                             True, plan.proposal_id, healer,
                                             patient))
                if report is not None:
                    assigned = report.new_heal_assignments.get(patient)
                    if assigned is not None and assigned != healer:
                        problems.append(_problem("heal_capacity_exceeded", True,
                                                 plan.proposal_id, healer,
                                                 patient))
    # protocol heal capacity across the WHOLE final plan: a patient newly
    # treated by two healers is a hard violation even for baseline rows (D8)
    if report is not None:
        for patient, (first, second) in report.heal_capacity_conflicts.items():
            problems.append(_problem("heal_capacity_conflict", True,
                                     "", second, patient,
                                     ))
    return problems


def complete_and_validate(plans, context, actions, report=None):
    problems = validate_actions([actions[f"u{i}"] for i in range(len(actions))],
                                context, len(actions))
    problems.extend(validate_plans(plans, context, actions, report))
    return problems


def repair(plans: List[CandidatePlan], context: Context,
           actions: Dict[str, int], memory, evaluator,
           rounds: int = 2, skills=None) -> Tuple[List[CandidatePlan],
                                                  Dict[str, int],
                                                  List[str]]:
    """Whole-plan atomic release of anything invalid, refill from the
    context-aware baseline, rescore — bounded rounds (A4)."""
    from .baseline import baseline_actions
    log: List[str] = []
    for _ in range(rounds + 1):
        report = evaluator.effects(context, actions)
        problems = validate_actions(
            [actions[f"u{i}"] for i in range(len(actions))], context,
            len(actions))
        problems += validate_plans(plans, context, actions, report)
        if not problems:
            break
        drop_ids = {p["proposal_id"] for p in problems if p["drop_whole"]}
        actor_problems = {p["actor"] for p in problems if p["actor"]}
        plans = [p for p in plans if p.proposal_id not in drop_ids]
        refill = baseline_actions(context, memory, skills,
                                  reserved=dict(actions),
                                  reserved_plans=plans)
        for actor in list(actions):
            plan = next((p for p in plans if actor in p.controlled_actors), None)
            if plan is None and actor in actor_problems:
                actions[actor] = refill.get(actor, actions[actor])
            elif plan is not None:
                actions[actor] = plan.current_actions[actor]
        log.append(f"repair_released:{sorted(drop_ids)}")
    return plans, actions, log
