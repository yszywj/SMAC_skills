"""Audit a one-step override followed by the same repeated conditional policy."""
from dataclasses import dataclass, field
import json
from pathlib import Path

from decision_tree.learning.dataset import episode as recorded_episode, index, require
from decision_tree.local.dataset import comparable_observation
from decision_tree.local.runtime import digest
from skills.local_v2.explore import checksum
from .run import check_completion


FILES = ("run.json", "summary.json", "manifest.json", "episodes.json", "events.jsonl")


def _stamps(path):
    return {name: (file.stat().st_size, file.stat().st_mtime_ns)
            for name in FILES for file in [path/name]}


@dataclass
class AuditedRun:
    path: Path
    headers: dict
    hashes: dict
    stamps: dict
    cached_episode: int | None = None
    cached_arm: dict | None = field(default=None, repr=False)

    def unchanged(self):
        require(_stamps(self.path) == self.stamps, "Audited run changed on disk")

    def episode(self, number):
        self.unchanged()
        if self.cached_episode != number:
            events = []
            with (self.path/"events.jsonl").open(encoding="utf-8") as stream:
                for line in stream:
                    event = json.loads(line)
                    if event.get("episode") == number:
                        events.append(event)
            self.cached_arm = recorded_episode({**self.headers, "events.jsonl": events}, number)
            self.cached_episode = number
        return self.cached_arm


def read_audited_run(path):
    path = Path(path).resolve()
    stamps = _stamps(path)
    headers = {name: json.loads((path/name).read_text(encoding="utf-8")) for name in FILES[:-1]}
    metadata = headers["run.json"]["conditional_skills"]
    check_completion(path, metadata, len(headers["manifest.json"]["episodes"]))
    hashes = {name: checksum(path/name) for name in FILES}
    require(stamps == _stamps(path), "Run changed during audit")
    return AuditedRun(path, headers, hashes, stamps)


def audit_counterfactual(incumbent_run, branch_run, episode, step, binding):
    """Return measured terminal deltas after checking the exact pre-override path.

    Both arguments accept a directory or read_audited_run()'s cache. The
    incumbent may contain many episodes; the branch contains its one matched
    scene. Later actions may differ as the same policy reacts to the new state.
    """
    incumbent = incumbent_run if isinstance(incumbent_run, AuditedRun) else read_audited_run(incumbent_run)
    branch = branch_run if isinstance(branch_run, AuditedRun) else read_audited_run(branch_run)
    incumbent.unchanged()
    branch.unchanged()
    left, right = incumbent.headers, branch.headers
    imeta, bmeta = (source["run.json"]["conditional_skills"] for source in (left, right))
    require(imeta.get("forced") is None, "Incumbent already has a forced override")
    require(bmeta.get("forced") == {"step": step, "binding": binding}, "Wrong declared override")
    require(type(episode) is int and 1 <= episode <= len(left["episodes.json"]), "Invalid incumbent episode")
    require(type(step) is int and step >= 0, "Invalid anchor step")
    require(len(right["episodes.json"]) == 1, "Branch must contain exactly one scene")
    for key in ("policy", "policy_sha256", "source_hashes"):
        require(imeta[key] == bmeta[key], "Changed policy or implementation across fork")
    for key in ("env_args", "skill_parameters", "source_hashes", "versions", "action_extensions"):
        require(left["run.json"].get(key) == right["run.json"].get(key), "Changed execution setting: " + key)
    scene = left["manifest.json"]["episodes"][episode-1]
    other_scene = right["manifest.json"]["episodes"][0]
    for key in ("sha256", "sc2_seed", "episode_config"):
        require(scene[key] == other_scene[key], "Wrong paired scene or random seed")
    old, new = incumbent.episode(episode), branch.episode(1)
    for key in ("initial_state_sha256", "scenario_sha256", "game_version"):
        require(old["start"][key] == new["start"][key], "Initial-state mismatch: " + key)
    decisions = [index(arm["events"], "conditional_local_decision", "step") for arm in (old, new)]
    require(step in decisions[0] and step in decisions[1], "Override anchor was not reached")
    prefix = digest({key: old["start"][key] for key in
                     ("initial_state_sha256", "scenario_sha256", "game_version")})
    baseline_fields = ("game_loop", "baseline_actions", "baseline_roles", "baseline_predicted_damage", "catalog")
    before_fields = ("selected_id", "final_actions", "interventions", "proposals_used", "natural_selected_id", "corrections")
    for t in range(step+1):
        old_obs = comparable_observation(old["observations"][t]["payload"])
        new_obs = comparable_observation(new["observations"][t]["payload"])
        require(old_obs == new_obs, "Observable prefix differs before override")
        a, b = decisions[0][t], decisions[1][t]
        baseline = {key: a[key] for key in baseline_fields}
        require(baseline == {key: b[key] for key in baseline_fields}, "Adaptive baseline differs at paired prefix")
        prefix = digest([prefix, old_obs, baseline])
        if t < step:
            require(all(a[key] == b[key] for key in before_fields), "Earlier conditional intervention differs")
            submitted = {key: old["steps"][t+1][key] for key in
                         ("actions", "reward", "feedback", "game_loop", "terminated")}
            require(submitted == {key: new["steps"][t+1][key] for key in submitted}, "Executed prefix differs")
            prefix = digest([prefix, submitted])
    anchor_old, anchor_new = decisions[0][step], decisions[1][step]
    require(anchor_new["forced"] and anchor_new["proposed_id"] == binding, "Missing exact forced binding")
    effect = "BASE"
    if binding != "BASE":
        effect = next((key for key, option in anchor_old["catalog"]["effects"].items() if binding in option["aliases"]), None)
        require(effect is not None, "Override is absent from incumbent action catalogue")
    selected = anchor_old["selected_id"]
    original_effect = "BASE" if selected == "BASE" else next(
        key for key, option in anchor_old["catalog"]["effects"].items() if selected in option["aliases"])
    reference = {"return": old["record"]["reward"], "outcome": old["record"]["outcome"]}
    candidate = {"return": new["record"]["reward"], "outcome": new["record"]["outcome"]}
    return {
        "scope": "single_override_then_same_conditional_policy", "group_id": scene["sha256"],
        "seed": scene["sc2_seed"], "episode": episode, "step": step, "binding": binding,
        "anchor_sha256": prefix, "effect": effect, "incumbent_effect": original_effect,
        "different_action": effect != original_effect, "candidate": candidate, "reference": reference,
        "branch": candidate, "incumbent": reference,
        "return_gain": candidate["return"] - reference["return"],
        "win_gain": int(candidate["outcome"] == "win") - int(reference["outcome"] == "win"),
        "incumbent_sources": incumbent.hashes, "branch_sources": branch.hashes,
    }
