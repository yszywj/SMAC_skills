"""Apply at most one local change per step, with per-actor re-entry cooldown."""
from copy import deepcopy
import re

from decision_tree.local.runtime import LocalController
from decision_tree.phase2.policy import REGROUP_POLICY
from skills.local_v1 import apply_candidate
from skills.local_v3.library import catalog, recommended_candidate
from skills.local_v2.library import resolve_action
from .policy import gate, normalize_policy
from .corrections import choose


CARRIER = {
    "schema_version": 1, "kind": "local_patch_policy",
    "selection": {"mode": "base", "max_interventions": 1, "min_gain": 0.0},
    "candidates": {"max_candidates": 8, "max_per_family": 2},
}


def normalize_forced(forced):
    if forced is None:
        return None
    if (not isinstance(forced, dict) or set(forced) != {"step", "binding"}
            or type(forced["step"]) is not int or forced["step"] < 0
            or not isinstance(forced["binding"], str)
            or not re.fullmatch(r"(?:BASE|(?:regroup_step|retarget|retreat_step|retreat_veto):u\d+:\d+)",
                                forced["binding"])):
        raise ValueError("Forced override requires a nonnegative step and legal binding syntax")
    return deepcopy(forced)


class ConditionalController(LocalController):
    def __init__(self, manager, policy=None, *, posture="balanced", context_hash=None, forced=None):
        self.conditional_policy = normalize_policy(policy)
        self.forced = normalize_forced(forced)
        if context_hash != REGROUP_POLICY["model"]["context_hash"]:
            raise ValueError("Conditional skills require the frozen adaptive context")
        super().__init__(manager, CARRIER, posture=posture, context_hash=context_hash)

    def reset(self):
        super().reset()
        self.last_change = {}

    def _decide(self, payload, baseline):
        policy = self.conditional_policy
        menu = catalog(payload, baseline)
        candidates, eligible, considered = [], [], []
        budget = policy["max_interventions"]
        exhausted = budget is not None and self.interventions >= budget
        for row in menu["bindings"]:
            candidate = recommended_candidate(payload, baseline, row)
            reason = gate(row, candidate, policy)
            actor = row["actor"]
            if reason == "eligible":
                if exhausted:
                    reason = "budget_exhausted"
                elif self.step_number - self.last_change.get(actor, -policy["actor_cooldown_steps"]) < policy["actor_cooldown_steps"]:
                    reason = "wait_actor_cooldown"
            considered.append({"family": row["family"], "actor": actor,
                               "recommended_action": row["recommended_action"], "reason": reason})
            if candidate is not None:
                candidates.append(candidate)
            if reason == "eligible":
                eligible.append(candidate)
        # Last intervention breaks ties fairly before stable actor IDs.
        eligible.sort(key=lambda c: (policy["priority"].index(c["family"]),
                                     self.last_change.get(c["actors"][0], -1),
                                     int(c["actors"][0][1:]), c["id"]))
        selected = eligible[0] if eligible else None
        allowed_actors = {actor for actor, unit in payload["state"]["allies"].items()
                          if unit.get("alive") and self.step_number - self.last_change.get(
                              actor, -policy["actor_cooldown_steps"]) >= policy["actor_cooldown_steps"]}
        selected, corrections = choose(policy, payload, baseline, menu, selected, candidates,
                                       allowed_actors=allowed_actors, exhausted=exhausted)
        natural_selected = selected["id"] if selected else "BASE"
        is_forced = self.forced is not None and self.step_number == self.forced["step"]
        if is_forced:
            binding = self.forced["binding"]
            if binding == "BASE":
                selected = None
            else:
                if not any(binding in effect["aliases"] for effect in menu["effects"].values()):
                    raise ValueError("Forced binding is absent from the current complete catalogue")
                family, actor, action = binding.split(":")
                selected = resolve_action(payload, baseline, family, actor, int(action))
        result = apply_candidate(payload, baseline, selected) if selected else baseline
        if selected:
            self.last_change[selected["actors"][0]] = self.step_number
            self.interventions += 1
            self.proposals_used += 1
        # Counts describe changed commands submitted to the manager. Native
        # acceptance stays in the runner's subsequent step feedback.
        self.events.append({
            "event": "conditional_local_decision", "step": self.step_number,
            "game_loop": payload["state"]["game_loop"], "baseline_actions": list(baseline["actions"]),
            "baseline_roles": deepcopy(baseline.get("roles", {})),
            "baseline_predicted_damage": deepcopy(baseline.get("predicted_damage") or {}),
            "selected_id": selected["id"] if selected else "BASE",
            "proposed_id": self.forced["binding"] if is_forced else selected["id"] if selected else "BASE",
            "vetoed": False,
            "reason": "forced_override" if is_forced else "execute" if selected else "budget_exhausted" if exhausted else "wait",
            "candidates": candidates, "considered": considered, "catalog": menu,
            "final_actions": list(result["actions"]), "predictions": {},
            "interventions": self.interventions, "proposals_used": self.proposals_used,
            "counter_semantics": "changed_command_submissions",
            "conditional_policy": deepcopy(policy),
            "forced": is_forced, "natural_selected_id": natural_selected,
            "corrections": corrections,
        })
        return result
