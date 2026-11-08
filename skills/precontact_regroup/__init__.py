"""接敌前的单兵靠拢，使用 adaptive 的单单位、单决策步提交接口。"""

from pathlib import Path

from decision_tree.local.models import predict, select_candidate
from decision_tree.local.runtime import LocalController, load_policy
from skills.local_v1 import generate_candidates


POLICY_PATH = Path(__file__).with_name("policy.json")
_POLICY = load_policy(POLICY_PATH)


def propose(payload, baseline):
    """Return (candidate, vetoed) for this observation, without changing state.

    (None, False) leaves the opportunity open. (candidate, True) consumes
    the proposal but keeps BASE. For (candidate, False) the caller may pass
    candidate to skills.local_v1.apply_candidate before the manager commits.
    """
    menu = generate_candidates(payload, baseline, _POLICY["candidates"])
    regroup = [candidate for candidate in menu if candidate["family"] == "regroup_step"]
    choice = select_candidate(_POLICY["model"], regroup, _POLICY["selection"]["min_gain"])
    if choice["candidate_id"] == "BASE":
        return None, False
    candidate = next(item for item in regroup if item["id"] == choice["candidate_id"])
    correction = predict(_POLICY["veto_model"], candidate["family"], candidate["features"])
    return candidate, correction["supported"] and correction["value"] > 1e-9


def attach(manager, *, context_hash, posture="balanced"):
    """Install the skill before adaptive commits its actions and unit histories.

    Pass the current runner's execution context. Drive the returned controller
    with step(), commit the final actions through the manager, and reset the
    controller and manager at each episode boundary.
    """
    return LocalController(manager, _POLICY, posture=posture, context_hash=context_hash)
