"""Pure standard-library, single-tree Medivac direct-action runtime."""

from copy import deepcopy
import hashlib
import json
import operator
from pathlib import Path

from .bindings import build_bindings
from .candidates import FEATURE_NAMES, FEATURE_SCHEMA_VERSION, build_features, evaluate_candidate
from .facts import (actor_index, finite, heal_candidates, hp_ratio, is_medivac,
                    legal_actions, observed_target, threats)
from .memory import actor_memory, fresh_memory
from .idle_recovery import observed_context, stop_windows
from .selector import POLICY_KIND, fingerprint as comparable_fingerprint, select, validate_policy
from .validate import validate_tree

_OPS = {"<=": operator.le, ">=": operator.ge, "<": operator.lt,
        ">": operator.gt, "==": operator.eq, "!=": operator.ne}


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                      allow_nan=False, default=lambda v: sorted(v) if isinstance(v, (set, frozenset)) else str(v))


class MedivacTree:
    """A shared action tree. Each decision is bound and validated atomically.

    The initial tree inherits selected old priorities, not old numerical
    scoring, global matching, or changes to ground-unit actions.
    """

    def __init__(self, tree_path=None):
        self.tree_path = Path(tree_path) if tree_path is not None else Path(__file__).with_name("tree_v0.json")
        self.policy_document = json.loads(self.tree_path.read_text(encoding="utf-8"))
        self.return_policy = self.policy_document.get("policy_kind") == POLICY_KIND if isinstance(self.policy_document, dict) else False
        info = validate_policy(self.policy_document) if self.return_policy else validate_tree(self.policy_document)
        self.document = self.policy_document["base_tree"] if self.return_policy else self.policy_document
        self.parameters = info["parameters"]
        self.tree_hash = comparable_fingerprint({"policy": self.policy_document, "runtime_sources": info["runtime_source_hashes"]}) if self.return_policy else hashlib.sha256(_json(self.document).encode("utf-8")).hexdigest()
        self._metadata = {"version": self.document["version"], "schema_version": 1,
                          "feature_schema_version": 1, "tree_hash": self.tree_hash,
                          "tree_path": str(self.tree_path.resolve()), "node_count": info["node_count"],
                          "parameters": deepcopy(self.parameters), "method": "single_direct_action_tree",
                          "baseline_equivalence": False, "teacher": None,
                          "ground_coordination": False}
        if self.return_policy:
            self.candidate = info["candidate"]
            self.selection = info["selection"]
            self.comparison_contract_hash = info["comparison_contract_hash"]
            self._metadata.update(version=self.policy_document["version"], policy_kind=POLICY_KIND,
                                  method=POLICY_KIND, selection_mode=self.selection["mode"],
                                  comparison_contract_hash=self.comparison_contract_hash,
                                  runtime_source_hashes=info["runtime_source_hashes"],
                                  candidate=deepcopy(self.candidate), base_tree_hash=comparable_fingerprint(self.document),
                                  feature_schema_version=FEATURE_SCHEMA_VERSION, base_feature_schema_version=1,
                                  feature_schema={"version": FEATURE_SCHEMA_VERSION, "names": list(FEATURE_NAMES)})

    @property
    def metadata(self):
        return deepcopy(self._metadata)

    def _walk(self, facts):
        node = self.document["root"]
        path = []
        while "intent" not in node:
            value = facts[node["feature"]]
            result = bool(_OPS[node["op"]](value, node["threshold"]))
            path.append({"node": node["id"], "feature": node["feature"], "value": value,
                         "op": node["op"], "threshold": node["threshold"], "result": result})
            node = node["yes"] if result else node["no"]
        return node["id"], node["intent"], path

    def decide(self, state, n_agents, memory=None):
        if not isinstance(state, dict) or not isinstance(state.get("allies"), dict):
            raise ValueError("state.allies must be a mapping")
        if isinstance(n_agents, bool) or not isinstance(n_agents, int) or n_agents < 1:
            raise ValueError("n_agents must be a positive integer")
        medivacs = [a for a, u in state["allies"].items() if is_medivac(u)]
        if any(actor_index(a) >= n_agents for a in medivacs):
            raise ValueError("Medivac actor index exceeds n_agents")
        prior, reset_reason = fresh_memory(memory, state, self.tree_hash)
        fingerprint = hashlib.sha256(_json(state).encode("utf-8")).hexdigest()
        if prior.get("loop") == state["game_loop"] and prior.get("_cache"):
            if prior.get("state_fingerprint") != fingerprint:
                raise ValueError("same decision loop received a changed snapshot")
            return {**deepcopy(prior["_cache"]), "memory": deepcopy(prior)}
        loop, step = state["game_loop"], state.get("step_mul", 8)
        opportunity_consumed = False
        if self.return_policy:
            # A history reset caused by a roster change or skipped observation
            # must not introduce a second intervention in the same episode.
            same_episode = (isinstance(memory, dict) and memory.get("tree_hash") == self.tree_hash
                            and memory.get("episode") == state.get("episode")
                            and loop >= memory.get("loop", -1))
            opportunity_consumed = bool(same_episode and memory.get("return_opportunity_consumed"))
        observed = {}
        observed_conflicts = {}
        for actor in sorted(medivacs, key=actor_index):
            target = observed_target(state, actor)
            if target is not None:
                if target in observed:
                    observed_conflicts.setdefault(target, [observed[target]]).append(actor)
                else:
                    observed[target] = actor
        reservations = dict(observed)
        legal = {a: legal_actions(state, a) for a in medivacs}
        for actor in medivacs:
            if "tag" not in state["allies"][actor]:
                raise ValueError(f"{actor}: observed tag is required for memory isolation")
        def scheduling_key(actor):
            unit = state["allies"][actor]
            if not unit.get("alive"):
                return (-1, 0., actor_index(actor))
            candidates = heal_candidates(state, actor, legal[actor])
            count = sum(observed.get(t, actor) == actor for t in candidates)
            return (count, hp_ratio(unit), actor_index(actor))
        processing_order = sorted(medivacs, key=scheduling_key)
        actions, assignments, decisions, records = {}, {}, {}, {}
        for actor in processing_order:
            unit = state["allies"][actor]
            old = actor_memory(prior, actor, unit, loop, step)
            before_reservations = dict(reservations)
            if unit.get("alive"):
                threat_rows = threats(state, actor, self.parameters)
                bindings, details = build_bindings(state, actor, legal[actor], old, reservations, threat_rows, self.parameters)
                endangered = hp_ratio(unit) < self.parameters["danger_hp_ratio"] and any(r["near"] for r in threat_rows)
                facts = {"alive": True, "endangered": endangered,
                         "can_retreat": endangered and "RETREAT" in bindings,
                         "has_rescue": "HEAL_RESCUE" in bindings, "can_keep": "KEEP_HEAL" in bindings,
                         "has_current": "HEAL_CURRENT" in bindings, "has_new": "HEAL_NEW" in bindings,
                         "has_approach": "APPROACH" in bindings, "has_support": "SUPPORT" in bindings,
                         "has_support_move": bindings.get("SUPPORT", {}).get("action", 1) != 1,
                         "can_stop": 1 in legal[actor], "health_ratio": hp_ratio(unit),
                         "energy": float(unit["energy"]) if finite(unit.get("energy")) else 0.,
                         "energy_known": finite(unit.get("energy")), "threat_count": sum(r["near"] for r in threat_rows),
                         "candidate_count": details["candidate_count"],
                         "keep_stalled": details["keep"].get("stalled", 0),
                         "keep_unconfirmed": details["keep"].get("unconfirmed", 0),
                         "approach_windows": details["approach"].get("windows", 0),
                         "rescue_cooldown_ready": details["rescue_cooldown_ready"]}
            else:
                threat_rows, details = [], {}
                bindings = {"NOOP": {"action": 0, "target": None, "reason": "dead_unit_noop", "selection_key": []}}
                facts = {key: False for key in ("alive", "endangered", "can_retreat", "has_rescue", "can_keep", "has_current", "has_new", "has_approach", "has_support", "has_support_move", "can_stop", "energy_known", "rescue_cooldown_ready")}
                facts.update({key: 0 for key in ("health_ratio", "energy", "threat_count", "candidate_count", "keep_stalled", "keep_unconfirmed", "approach_windows")})
            leaf, intent, path = self._walk(facts)
            if intent not in bindings:
                raise ValueError(f"{actor}: leaf {leaf} intent {intent} has no legal binding")
            binding = bindings[intent]
            return_selection = None
            candidate_audit, selected_proposal, idle_windows = None, None, 0
            idle_generator = self.return_policy and self.candidate.get("generator") == "idle_recovery"
            if self.return_policy:
                if idle_generator:
                    idle_windows = stop_windows(state, actor, old, intent, binding, self.candidate["event"])
                proposal_context = {"state": state, "actor": actor, "legal": legal[actor], "old": old,
                                    "reservations": before_reservations, "threat_rows": threat_rows,
                                    "stop_windows": idle_windows}
                if not opportunity_consumed or idle_generator:
                    proposal_context["observed"] = observed_context(state, actor, before_reservations, threat_rows)
                    proposal, candidate_audit = evaluate_candidate(self.candidate, bindings, intent, binding, loop,
                                                                    context=proposal_context)
                else:
                    proposal = None
                if opportunity_consumed:
                    proposal = None
                    if candidate_audit is not None:
                        candidate_audit.update(eligible=False, proposal_reason=candidate_audit["reason"],
                                               reason="episode_opportunity_consumed")
                if proposal is not None:
                    # Candidate generation and inference are pure. Commit only
                    # the selected binding below, using the unchanged executor.
                    features = build_features(state, actor, facts, intent, binding, proposal["binding"],
                                              context=proposal_context)
                    selected, predicted_gain, selector_path = select(self.selection, features)
                    before_memory = {key: deepcopy(prior[key]) for key in
                                     ("schema_version", "episode", "loop", "actors", "roster") if key in prior}
                    before_memory["current_step_records"] = deepcopy(records)
                    fingerprints = {"state_fingerprint": comparable_fingerprint(state),
                                    "memory_fingerprint": comparable_fingerprint(before_memory),
                                    "reservations_fingerprint": comparable_fingerprint(before_reservations)}
                    context = {**fingerprints, "actor": actor, "game_loop": loop}
                    return_selection = {"opportunity": True, "features": features,
                                        "base": {"intent": intent, "action": binding["action"], "target": binding["target"]},
                                        "candidate": {"intent": proposal["intent"], "action": proposal["binding"]["action"],
                                                      "target": proposal["binding"]["target"]},
                                        "selected": selected, "predicted_gain": predicted_gain, "path": selector_path,
                                        "comparison_contract_hash": self.comparison_contract_hash,
                                        **fingerprints, "context_fingerprint": comparable_fingerprint(context)}
                    opportunity_consumed = True
                    if selected == "CANDIDATE":
                        intent, binding = proposal["intent"], proposal["binding"]
                        selected_proposal = proposal
            action, target = binding["action"], binding["target"]
            if action not in legal[actor] or (not unit.get("alive") and action != 0):
                raise ValueError(f"{actor}: tree emitted an illegal action {action}")
            record = {"tag": unit["tag"], "loop": loop, "order_target": details.get("current_target"),
                      "order_target_tag": details.get("current_target_tag"),
                      "last_rescue_loop": old.get("last_rescue_loop"), "last_action": action,
                      "last_intent": intent, "last_target": target, "keep": {}, "approach": {}}
            if intent == "HEAL_RESCUE":
                record["last_rescue_loop"] = loop
            if details.get("keep") and (intent == "KEEP_HEAL" or details.get("keep_rejections") or old.get("keep")):
                record["keep"] = deepcopy(details["keep"])
            if intent == "APPROACH" or details.get("approach_rejections"):
                record["approach"] = deepcopy(details["approach"])
            if selected_proposal is not None and "approach_memory" in selected_proposal:
                record["approach"] = deepcopy(selected_proposal["approach_memory"])
            if idle_generator:
                record["idle_stop_windows"] = idle_windows if intent == "STOP" and action == 1 else 0
            if target is not None and intent in ("HEAL_RESCUE", "HEAL_CURRENT", "HEAL_NEW", "KEEP_HEAL", "APPROACH"):
                owner = reservations.get(target)
                if owner is not None and owner != actor:
                    raise ValueError(f"{actor}: target {target} already reserved by {owner}")
                reservations[target] = actor
            if intent in ("HEAL_RESCUE", "HEAL_CURRENT", "HEAL_NEW", "KEEP_HEAL"):
                assignments[actor] = target
            actions[actor], records[actor] = action, record
            decisions[actor] = {"path": path, "leaf": leaf, "intent": intent, "target": target,
                                "action": action, "reason": binding["reason"], "binding": deepcopy(binding),
                                "facts": facts, "bindings": deepcopy(bindings), "binding_details": details,
                                "visible_threats": threat_rows, "memory_before": old,
                                "memory_after": deepcopy(record), "reservation_before": before_reservations,
                                "reservation_after": dict(reservations), "processing_key": list(scheduling_key(actor)),
                                "override": None}
            if candidate_audit is not None:
                decisions[actor]["candidate_audit"] = deepcopy(candidate_audit)
            if return_selection is not None:
                decisions[actor]["return_selection"] = return_selection
                if return_selection["selected"] == "CANDIDATE":
                    decisions[actor]["override"] = {"kind": POLICY_KIND, "base": deepcopy(return_selection["base"]),
                                                    "reason": "one_paired_intervention"}
        result = {"actions": actions, "assignments": assignments, "decisions": decisions,
                  "processing_order": processing_order, "observed_reservations": observed,
                  "observed_orders": {a: t for a in medivacs if (t := observed_target(state, a)) is not None},
                  "observed_conflicts": observed_conflicts, "reservations": reservations,
                  "memory_reset_reason": reset_reason, "metadata": self.metadata}
        cached_result = deepcopy(result)
        cached_result["memory_reset_reason"] = None  # reset diagnostics are not persistent policy state
        new_memory = {"schema_version": 1, "tree_hash": self.tree_hash, "episode": state.get("episode"),
                      "loop": loop, "actors": records, "state_fingerprint": fingerprint,
                      "roster": {a: u.get("tag") for a, u in state["allies"].items()},
                      "_cache": cached_result}
        if self.return_policy:
            new_memory["return_opportunity_consumed"] = opportunity_consumed
        return {**result, "memory": new_memory}
