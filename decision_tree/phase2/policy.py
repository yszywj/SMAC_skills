"""Pure deployable selector, shared by BASE-prefix replay and native rollouts."""
from copy import deepcopy
from math import isfinite
import re

from decision_tree.local.models import select_candidate
from decision_tree.local.runtime import load_policy
from skills.local_v1 import FEATURE_NAMES
from skills.precontact_regroup import POLICY_PATH


REGROUP_POLICY = load_policy(POLICY_PATH)
DEFAULT_RULES = {
    "unknown": "execute",
    "near_pre": "execute",
    "near_engaged": "execute",
    "far": "decline",
}


def validate_rule(rule, depth=0):
    if isinstance(rule, str):
        if rule not in ("execute", "decline", "wait"):
            raise ValueError("Unknown action in phase2 tree")
        return
    if (depth >= 2 or not isinstance(rule, dict)
            or set(rule) != {"feature", "threshold", "left", "right"}
            or rule["feature"] not in FEATURE_NAMES or type(rule["threshold"]) not in (int, float)
            or not isfinite(rule["threshold"])):
        raise ValueError("Expected a depth-at-most-two tree on public candidate features")
    validate_rule(rule["left"], depth + 1)
    validate_rule(rule["right"], depth + 1)


def normalize_policy(source):
    """Validate and fill all defaults before recording or executing a policy."""
    policy = deepcopy(source)
    fields = {"base": {"mode"}, "forced": {"mode", "anchor_step", "candidate_id"},
              "regroup": {"mode", "rules", "fallback"}}
    if not isinstance(policy, dict) or policy.get("mode") not in fields:
        raise ValueError("Expected phase2 base, forced, or regroup policy")
    mode = policy["mode"]
    if set(policy) - fields[mode]:
        raise ValueError("Unknown phase2 policy fields")
    if mode == "forced":
        if (type(policy.get("anchor_step")) is not int or policy["anchor_step"] < 0
                or not re.fullmatch(r"(?:regroup_step|retarget|retreat_step|retreat_veto):u\d+:\d+",
                                    str(policy.get("candidate_id")))):
            raise ValueError("A forced policy needs a valid anchor_step and candidate_id")
    if mode != "regroup":
        return policy
    rules = policy.get("rules", {})
    if not isinstance(rules, dict) or set(rules) - set(DEFAULT_RULES):
        raise ValueError("Unknown regroup bucket")
    policy["rules"] = {**DEFAULT_RULES, **rules}
    for rule in policy["rules"].values():
        validate_rule(rule)
    fallback = policy.setdefault("fallback", None)
    if fallback is not None:
        allowed = {"family", "rule", "max_actor_health_ratio", "require_team_fired"}
        if (not isinstance(fallback, dict) or set(fallback) - allowed
                or fallback.get("family") not in ("retarget", "retreat_step")):
            raise ValueError("Expected one retarget or retreat_step fallback")
        threshold = fallback.setdefault("max_actor_health_ratio", 1.0)
        if (type(threshold) not in (int, float) or not 0 <= threshold <= 1
                or type(fallback.setdefault("require_team_fired", True)) is not bool):
            raise ValueError("Invalid fallback eligibility")
        validate_rule(fallback.setdefault("rule", "execute"))
    return policy


def accepted_attack(feedback):
    """Whether the last completed SC2 step accepted an attack command (Protoss)."""
    if feedback is None:
        return False
    return any(receipt["submission"] == "accepted"
               and int(receipt["requested_action_id"]) >= 6
               for receipt in feedback["actors"].values())


def _action(rule, features):
    while isinstance(rule, dict):
        branch = "left" if features[rule["feature"]] <= rule["threshold"] else "right"
        rule = rule[branch]
    return rule


def regroup_bucket(candidate, team_fired_before):
    distance = candidate["features"]["nearest_enemy_before"]
    if distance < 0:
        return "unknown"
    if distance > 12:
        return "far"
    return "near_engaged" if team_fired_before else "near_pre"


def _fallback(candidates, policy, team_fired_before):
    if policy is None or (policy["require_team_fired"] and not team_fired_before):
        return None
    for candidate in candidates:
        if candidate["family"] != policy["family"]:
            continue
        features = candidate["features"]
        if (features["actor_health_ratio"] <= policy["max_actor_health_ratio"]
                and (candidate["family"] != "retarget" or features["actor_weapon_ready"] > 0.5)):
            return candidate
    return None


class Selector:
    """One instance per episode. Only execute and decline spend its opportunity."""

    def __init__(self, policy):
        self.policy = normalize_policy(policy)
        self.consumed = False

    def decide(self, candidates, step, team_fired_before):
        considered, bucket, action = None, None, "none"
        mode = self.policy["mode"]
        if self.consumed:
            action = "spent"
        elif mode == "forced" and step == self.policy["anchor_step"]:
            considered = next((c for c in candidates if c["id"] == self.policy["candidate_id"]), None)
            if considered is None:
                raise ValueError("Forced candidate absent at its declared anchor")
            action, bucket = "execute", "forced"
        elif mode == "regroup":
            regroup = [c for c in candidates if c["family"] == "regroup_step"]
            choice = select_candidate(REGROUP_POLICY["model"], regroup, 0)["candidate_id"]
            if choice != "BASE":
                considered = next(c for c in regroup if c["id"] == choice)
                bucket = regroup_bucket(considered, team_fired_before)
                action = _action(self.policy["rules"][bucket], considered["features"])
            else:
                fallback = self.policy["fallback"]
                considered = _fallback(candidates, fallback, team_fired_before)
                if considered is not None:
                    bucket = considered["family"]
                    action = _action(fallback["rule"], considered["features"])
        cid = considered["id"] if considered is not None else "BASE"
        if action in ("execute", "decline"):
            self.consumed = True
        return {"considered_id": cid, "selected_id": cid if action == "execute" else "BASE",
                "action": action, "bucket": bucket, "consumed": self.consumed,
                "team_fired_before": team_fired_before}
