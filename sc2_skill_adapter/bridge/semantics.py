"""Public SMACv2 semantics and distribution-only Astra input."""
import copy

KEEP_HEAL_ORDER_DESCRIPTION = (
    'Keep the actually observed heal order: submit no native SC2 command for this unit. '
    'The existing target may be outside the new-heal candidate range; this does not prove healing or following.')

ACTION_CONTRACT = {
    "attack_candidate_rule": "Use supplied legal criteria from SMACv2 masks and team visibility.",
    "candidate_center_distance_by_type": {"zealot": 2, "stalker": 6, "colossus": 7},
    "range_estimate": "Native weapon range plus unit radii is an estimate, not proof of firing.",
    "attack_order": "Accepted is not fired. Cooldown and movement affect actual attacks.",
    "abilities_enabled": False,
    "actions": "0 dead only; 1 Stop cancels order; 2/3/4/5 N/S/E/W; 6+j attacks e{j}.",
}


def compact_brief(brief):
    return copy.deepcopy(brief)


def planning_input(payload, env):
    # Intentionally ignore the payload. No initial episode or hidden state enters
    # a plan reused across randomized episodes.
    return {
        "scenario": copy.deepcopy(env.scenario_brief),
        "action_contract": copy.deepcopy(ACTION_CONTRACT),
        "episode_limit": env.episode_limit,
        "output_map_field": env.map_name,
        "planning_scope": "One conditional plan for this distribution and team size.",
    }


PLANNER_INSTRUCTIONS = """You are Astra, the tactical planner for SMACv2 Protoss micro.
Return only the requested JSON plan. Do not use tools, files, web, or code.
All facts are in INPUT JSON. Generate a concise conditional plan reused across
independent randomized episodes at this team size. Do not bind roles to unit IDs,
tags, an initial lineup, or positions. Use unit-type roles. Actual lineups and
starting positions vary. Write English, about 250-450 words, <=6000 JSON UTF-8 bytes;
text fields <=1000 characters; lists <=6 entries, each <=500 characters;
unit_roles 1-12 entries, unit_type <=80 and guidance <=500 characters.
Set map to output_map_field. Public distribution settings are not live enemy state.
The game pauses while models think. JEV selects one legal Choice per living ally
on one shared snapshot; all batches finish before advancing step_mul loops.
Actions: 0 dead-unit no-op; 1 Stop cancels an order; 2/3/4/5 move N/S/E/W by
move_amount; 6+j attacks visible enemy e{j}. No hold-order, spells, blink,
production, or macro. No skill allocator or shared same-frame reservations.
Candidate center-distance thresholds with this configuration are Zealot 2,
Stalker 6, Colossus 7; native weapon range and cooldown are separate facts.
Movement can interrupt firing; command acceptance does not prove damage.
JEV receives all allies and the union of their SMACv2-visible enemies, along with
health/shields, cooldowns, profiles, legal moves/attacks, distances, prior actions,
and native command feedback. Unseen enemies are unknown, not dead. No get_state()
or sampled hidden spawn positions are supplied. Large requests preserve core
teammate fields across batches; pending choices are never shared.
Give useful conditional targeting, frontline/ranged roles, and cooldown movement
guidance. Never invent unseen enemy positions or deterministic damage outcomes.
The plan is advisory: JEV chooses every concrete action from unchanged legal criteria.
"""
