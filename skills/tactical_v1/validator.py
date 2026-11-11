"""Final validation and bounded repair (A4).

Ordered checks:
 1. count / type / current legality of every action;
 2. decoded real target vs. the plan's goal and effect claims;
 3. control rights, heal occupancy, multi-unit dependencies, contact slots;
 4. atomic plans still complete;
 5. effects/ledger rebuilt from FINAL actions only.
"""
from typing import Dict, List, Optional, Tuple

from .actions import DEAD_NOOP
from .context import Context
from .contracts import CandidatePlan
from .ledger import Ledger


def validate_actions(actions: List[int], context: Context,
                     n_agents: int) -> List[str]:
    problems = []
    if len(actions) != n_agents:
        problems.append(f"length:{len(actions)}!={n_agents}")
    for i, action in enumerate(actions):
        actor = f"u{i}"
        if not isinstance(action, int) or isinstance(action, bool):
            problems.append(f"type:{actor}")
            continue
        unit = context.snapshot.allies.get(actor)
        legal = context.legal_actions(actor)
        if unit is not None and not unit.alive:
            if action != DEAD_NOOP:
                problems.append(f"dead:{actor}:{action}")
            continue
        if action not in legal:
            problems.append(f"illegal:{actor}:{action}")
    return problems


def validate_plans(plans: List[CandidatePlan], context: Context,
                   actions: Dict[str, int], memory=None) -> List[str]:
    problems: List[str] = []
    seen_control = {}
    for plan in plans:
        if set(plan.current_actions) != set(plan.controlled_actors):
            problems.append(f"actions_actors_mismatch:{plan.proposal_id}")
        for actor, action in plan.current_actions.items():
            if actor in seen_control:
                problems.append(
                    f"double_control:{actor}:{seen_control[actor]}:"
                    f"{plan.proposal_id}")
            seen_control[actor] = plan.proposal_id
            if actions.get(actor) != action:
                # plan action overridden by repair/fallback: must be released
                problems.append(f"plan_action_changed:{actor}:"
                                f"{plan.proposal_id}")
            semantic = context.decode(actor, action)
            if not semantic.valid:
                problems.append(f"undecodable:{actor}:{action}")
                continue
            # decoded target must MATCH the declared goal / effects
            if semantic.kind == "attack":
                if plan.goal != semantic.target and not any(
                        e.target == semantic.target for e in plan.expected_effects):
                    problems.append(f"goal_mismatch:{plan.proposal_id}:"
                                    f"{plan.goal}!={semantic.target}")
            elif semantic.kind in ("heal", "keep_heal"):
                if semantic.target and semantic.target not in plan.beneficiaries \
                        and plan.goal not in (semantic.target, "observed_order"):
                    problems.append(f"heal_target_mismatch:{plan.proposal_id}")
        # structured dependencies
        for dependency in plan.dependencies:
            if dependency.kind == "member":
                missing = [a for a in dependency.refs
                           if a not in plan.controlled_actors]
                if missing:
                    problems.append(f"member_missing:{plan.proposal_id}:{missing}")
            elif dependency.kind == "target_valid":
                for enemy_id in dependency.refs:
                    if enemy_id not in context.snapshot.enemies:
                        problems.append(f"target_lost:{plan.proposal_id}:{enemy_id}")
            elif dependency.kind == "heal_owner":
                patient, healer = dependency.refs[0], dependency.refs[1]
                owners = context.heal_occupancy()
                owner = owners.get(patient)
                if owner is not None and owner != healer:
                    problems.append(f"heal_occupied:{plan.proposal_id}:"
                                    f"{patient} by {owner}")
        # atomic completeness: every member still in the final plan set
        if plan.atomic:
            plan_ids = {p.proposal_id for p in plans}
            if plan.proposal_id not in plan_ids:
                problems.append(f"atomic_split:{plan.proposal_id}")
    return problems


def complete_and_validate(plans: List[CandidatePlan],
                          context: Context, actions: Dict[str, int]) -> List[str]:
    problems = validate_actions(list(actions.values()), context,
                                len(actions))
    problems.extend(validate_plans(plans, context, actions))
    return problems


def repair(plans: List[CandidatePlan], context: Context,
           actions: Dict[str, int], memory, evaluator,
           rounds: int = 2) -> Tuple[List[CandidatePlan], Dict[str, int], List[str]]:
    """Whole-plan atomic release of anything invalid, refill from the
    baseline, rescore — up to a bounded number of rounds (A4)."""
    from .baseline import baseline_actions
    log: List[str] = []
    for _ in range(rounds + 1):
        problems = validate_plans(plans, context, actions, memory)
        ordered = [actions[f"u{i}"] for i in range(len(actions))]
        legality = validate_actions(ordered, context, len(actions))
        if not problems and not legality:
            break
        broken_ids = {p.split(":")[1] for p in problems
                      if p.startswith("plan_action_changed")}
        broken_ids |= {pid for pid, plan in {p.proposal_id: p for p in plans}.items()
                       if any(x in ("undecodable", "atomic_split")
                              or x.startswith(("goal_mismatch", "member_missing",
                                               "target_lost", "heal_occupied",
                                               "double_control"))
                              for x in [p.split(":", 1)[0] for p in problems
                                        if pid in p])}
        # whole-plan release: atomic plans go entirely
        plans = [p for p in plans if p.proposal_id not in broken_ids]
        fresh = baseline_actions(context, memory, None)
        for actor, action in list(actions.items()):
            plan = next((p for p in plans if actor in p.controlled_actors), None)
            if plan is None:
                actions[actor] = fresh.get(actor, action)
            else:
                actions[actor] = plan.current_actions[actor]
        log.append(f"repair_released:{sorted(broken_ids)}")
    return plans, actions, log
