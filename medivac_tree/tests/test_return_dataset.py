"""Audit recorded episode pairs using real selector decisions, without SC2.

The terminal returns and feedback are fixtures; all decision fields, proposals,
features and fingerprints are produced by the deployed MedivacTree runtime.
"""
from copy import deepcopy
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

from medivac_tree import MedivacTree
from medivac_tree.candidates import FEATURE_NAMES, FEATURE_SCHEMA_VERSION
from medivac_tree.selector import comparison_contract_hash
from medivac_tree.learning.dataset import build_dataset, digest, main
from medivac_tree.tests.fixtures import observe_heal, payload, scene, unit


def write(path, name, value):
    if name.endswith("jsonl"):
        text = "".join(json.dumps(e) + "\n" for e in value)
    else:
        text = json.dumps(value)
    (path / name).write_text(text, encoding="utf-8")


def read(path, name):
    text = (path / name).read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines()] if name.endswith("jsonl") else json.loads(text)


def make_run(path, mode, reward=2., keep=True, *, candidate=None, initial_state=None, model_gain=1.):
    path.mkdir(parents=True)
    document = {"schema_version": 1, "version": "audit-" + mode,
                "policy_kind": "paired_return_selector", "base_tree": MedivacTree().document,
                "candidate": candidate or {"intent": "HEAL_NEW", "base_intents": ["KEEP_HEAL", "HEAL_CURRENT"],
                              "min_game_loop": 16}, "selection": {"mode": mode}}
    if mode == "model":
        document["selection"]["model"] = {
            "schema_version": 1, "model_kind": "gain_regression_tree", "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "feature_names": list(FEATURE_NAMES),
            "comparison_contract_hash": comparison_contract_hash(document["base_tree"], document["candidate"]),
            "root": {"id": "gain", "gain": model_gain, "n_samples": 4}}
    write(path, "policy.json", document)
    tree = MedivacTree(path / "policy.json")
    if initial_state is None:
        state = scene([unit("Medivac"), unit("Marine", (5 if keep else 1, 0), health=20), unit("Marine", (0, 2), health=25)])
        observe_heal(state, "u0", "u1", keep=keep)
        state["allies"]["u0"]["orders"][0]["target_tag"] = state["allies"]["u1"]["tag"]
    else:
        state = deepcopy(initial_state)
    n_agents = len(state["allies"])
    config = {"units": [u["type"] for u in state["allies"].values()], "test": "return_dataset"}
    scenario_hash, initial_hash = digest(config), digest(state)
    manifest = {"version": 1, "seed": 123, "env_args": {"map_name": "fixture"},
                "episodes": [{"index": 0, "sc2_seed": 123, "episode_config": config, "sha256": scenario_hash}]}
    manifest["sha256"] = digest(manifest)
    common = {"manifest_sha256": manifest["sha256"], "skill_source_hashes": {"test.py": "a" * 64},
              "skill_parameters": {"test": True}, "action_extensions": {"keep_heal": keep}}
    run = {**common, "medivac_policy": "tree", "native_teacher_control": "off", "scenario_count": 1,
           "method": "adaptive-skill", "skill_version": "fixture", "medical_patient_coordination": False,
           "medivac_tree_metadata": tree.metadata, "env_args": manifest["env_args"],
           "versions": {"pysc2": "fixture"}, "observation_protocol": {"version": 1},
           "source_hashes": {"runtime.py": "b" * 64, str(path / "policy.json"): digest(document)}}
    start = {"event": "episode_started", "episode": 1, "scenario_sha256": scenario_hash,
             "initial_state_sha256": initial_hash, "game_loop": 8, "game_version": "fixture"}
    events, memory = [start], None
    for index, loop in enumerate((8, 16, 24)):
        state["game_loop"] = loop
        observation = payload(deepcopy(state))
        result = tree.decide(state, n_agents, memory)
        memory = result["memory"]
        actions = [result["actions"].get("u" + str(j), 1) for j in range(n_agents)]
        decisions = []
        for actor, decision in result["decisions"].items():
            decisions.append({**deepcopy(decision), "actor": actor, "executed": True,
                              "proposed_action": decision["action"], "final_action": decision["action"],
                              "override_reason": None})
        feedback = {"from_game_loop": loop, "to_game_loop": loop + 8, "response_mapping": "exact",
                    "actors": {}, "unmatched_observation_errors": [], "raw_response_results": []}
        for j, action in enumerate(actions):
            decision = result["decisions"].get("u" + str(j), {})
            retained = decision.get("intent") == "KEEP_HEAL"
            target = decision.get("target")
            command = {"kind": "heal" if target else "stop", "target_unit": target,
                       "ability_id": 386 if target else 4,
                       "target_tag": state["allies"][target]["tag"] if target else None}
            if action in (2, 3, 4, 5):
                command = {"kind": "move", "ability_id": 16, "target_unit": None, "target_tag": None,
                           "target_position": state["movement_candidates"]["u" + str(j)][str(action)]["destination"]}
            feedback["actors"]["u" + str(j)] = {"requested_action_id": str(action),
                "command": None if retained else command, "submission": "retained_order" if retained else "accepted",
                "response": None if retained else {"code": 1, "name": "Success"}, "observation_errors": []}
        events.extend([
            {"event": "skill_observation", "episode": 1, "step": index, "payload": observation},
            {"event": "medivac_tree_step", "episode": 1, "step": index, "game_loop": loop, "mode": "tree",
             "candidate_metadata": tree.metadata, "decisions": decisions},
            {"event": "step", "episode": 1, "step": index + 1, "game_loop": loop, "actions": actions,
             "reward": reward if index == 2 else 0., "terminated": index == 2, "feedback": feedback}])
    record = {"episode": 1, "scenario_sha256": scenario_hash, "initial_state_sha256": initial_hash,
              "outcome": "win", "steps": 3, "reward": reward, "info": {"battle_won": True}}
    events.append({"event": "episode_completed", **record})
    for name, value in (("run.json", run), ("manifest.json", manifest), ("events.jsonl", events),
                        ("episodes.json", [record]), ("summary.json", {**common, "status": "complete", "error": None,
                         "completed_episodes": 1, "expected_episodes": 1, "pair_mode": "off"})):
        write(path, name, value)


class ReturnDatasetTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.base, self.candidate = self.root / "base", self.root / "candidate"
        make_run(self.base, "base", 2.)
        make_run(self.candidate, "candidate", 3.)

    def mutate_events(self, change, path=None):
        path = path or self.candidate
        events = read(path, "events.jsonl")
        change(events)
        write(path, "events.jsonl", events)

    def rejection(self, reason):
        result = build_dataset(self.base, self.candidate)
        self.assertEqual(result["rows"], [])
        self.assertTrue(any(reason in r for reject in result["rejected"] for r in reject["reasons"]), result)

    @staticmethod
    def gate(events):
        return next(e["decisions"][0] for e in events if e["event"] == "medivac_tree_step" and e["step"] == 1)

    def test_real_runtime_fields_form_one_episode_label(self):
        result = build_dataset(self.base, self.candidate)
        self.assertEqual(result["rejected"], [])
        self.assertEqual(len(result["rows"]), 1)
        row = result["rows"][0]
        self.assertEqual(row["target_gain"], 1.)
        self.assertEqual(row["evidence_kind"], "matched_observed_prefix")
        self.assertEqual(row["prefix_steps"], 1)
        self.assertEqual(row["base"]["intent"], "KEEP_HEAL")
        self.assertEqual(row["feedback"]["base"]["submission"], "retained_order")
        self.assertEqual(set(row["features"]), set(FEATURE_NAMES))
        self.assertEqual(result["feature_schema"]["version"], FEATURE_SCHEMA_VERSION)
        self.assertEqual(row["group_id"], row["scenario_sha256"])

    def test_positive_zero_negative_and_timeout_returns_are_retained(self):
        for value in (1., 2., 3.):
            arm = self.root / ("reward-" + str(value))
            make_run(arm, "candidate", value)
            records = read(arm, "episodes.json")
            records[0]["outcome"] = "timeout"
            write(arm, "episodes.json", records)
            events = read(arm, "events.jsonl")
            events[-1]["outcome"] = "timeout"
            write(arm, "events.jsonl", events)
            result = build_dataset(self.base, arm)
            self.assertEqual(result["rejected"], [], result)
            self.assertEqual(result["rows"][0]["target_gain"], value - 2.)

    def test_current_heal_feedback_is_accepted(self):
        base, candidate = self.root / "no-keep-base", self.root / "no-keep-candidate"
        make_run(base, "base", keep=False)
        make_run(candidate, "candidate", keep=False)
        result = build_dataset(base, candidate)
        self.assertEqual(result["rejected"], [])
        self.assertEqual(result["rows"][0]["base"]["intent"], "HEAL_CURRENT")

    def test_feature_schema_drift_is_rejected(self):
        run = read(self.candidate, "run.json")
        run["medivac_tree_metadata"]["feature_schema"]["version"] = FEATURE_SCHEMA_VERSION - 1
        write(self.candidate, "run.json", run)
        self.rejection("feature_schema_mismatch")

    def test_idle_stop_to_approach_move_uses_real_runtime_and_exact_feedback(self):
        candidate = {"intent": "APPROACH", "base_intents": ["STOP"], "min_game_loop": 0,
            "generator": "idle_recovery", "event": {"min_stop_windows": 3, "min_energy": 5,
            "min_health_ratio": .6, "max_target_distance": 12, "min_progress": .1, "max_extra_exposure": 1}}
        state = scene([unit("Medivac"), unit("Marine", (0, 8), health=20)],
                      {"e0": unit("Marine", (0, 7))})
        base, other = self.root / "idle-base", self.root / "idle-candidate"
        make_run(base, "base", candidate=candidate, initial_state=state)
        make_run(other, "candidate", candidate=candidate, initial_state=state)
        result = build_dataset(base, other)
        self.assertEqual(result["rejected"], [], result)
        self.assertEqual(len(result["rows"]), 1)
        row = result["rows"][0]
        self.assertEqual(row["base"]["intent"], "STOP")
        self.assertEqual(row["candidate"]["intent"], "APPROACH")
        self.assertEqual(row["feedback"]["candidate"]["command"]["kind"], "move")
        self.assertEqual(row["features"]["stop_windows"], 3.)
        self.assertEqual(row["prefix_steps"], 2)
        events = read(other, "events.jsonl")
        next(e for e in events if e["event"] == "step" and e["step"] == 3)["feedback"]["actors"]["u0"]["command"]["kind"] = "stop"
        write(other, "events.jsonl", events)
        self.assertEqual(build_dataset(base, other)["rows"], [])

    def test_no_opportunity_and_duplicate_opportunity_are_rejected(self):
        original = read(self.candidate, "events.jsonl")
        self.mutate_events(lambda events: self.gate(events).pop("return_selection"))
        self.rejection("not_exactly_one_opportunity")
        write(self.candidate, "events.jsonl", original)
        def duplicate(events):
            events_tree = next(e for e in events if e["event"] == "medivac_tree_step" and e["step"] == 2)
            events_tree["decisions"][0]["return_selection"] = deepcopy(self.gate(events)["return_selection"])
        self.mutate_events(duplicate)
        self.rejection("not_exactly_one_opportunity")

    def test_prefix_observation_or_action_difference_is_rejected(self):
        original = read(self.candidate, "events.jsonl")
        for event_type, reason in (("skill_observation", "observed_prefix_mismatch"), ("step", "action_or_feedback_prefix_mismatch")):
            with self.subTest(event_type=event_type):
                write(self.candidate, "events.jsonl", original)
                def change(events):
                    event = next(e for e in events if e["event"] == event_type)
                    if event_type == "step":
                        event["actions"][1] = 2
                    else:
                        event["payload"]["state"]["allies"]["u1"]["health"] -= 1
                self.mutate_events(change)
                self.rejection(reason)

    def test_gate_features_context_and_prediction_are_audited(self):
        original = read(self.candidate, "events.jsonl")
        cases = (("features", {"invalid": 0}, "invalid_opportunity_features"),
                 ("context_fingerprint", "0" * 64, "context_fingerprint_invalid"),
                 ("reservations_fingerprint", "0" * 64, "reservation_fingerprint_invalid"),
                 ("predicted_gain", 0., "unexpected_forced_arm_prediction"),
                 ("selected", "BASE", "arm_selection_mismatch"))
        for key, value, reason in cases:
            with self.subTest(key=key):
                write(self.candidate, "events.jsonl", original)
                self.mutate_events(lambda events: self.gate(events)["return_selection"].update({key: value}))
                self.rejection(reason)

    def test_missing_failed_or_ambiguous_feedback_is_rejected(self):
        original = read(self.candidate, "events.jsonl")
        cases = (("response_mapping", "heuristic", "feedback_not_exact"),
                 ("response", None, "feedback_failed"),
                 ("observation_errors", [{"code": 9}], "feedback_observation_error"),
                 ("command", None, "feedback_command_missing"))
        for key, value, reason in cases:
            with self.subTest(key=key):
                write(self.candidate, "events.jsonl", original)
                def change(events):
                    feedback = next(e for e in events if e["event"] == "step" and e["step"] == 2)["feedback"]
                    (feedback if key == "response_mapping" else feedback["actors"]["u0"])[key] = value
                self.mutate_events(change)
                self.rejection(reason)

    def test_illegal_or_second_actor_action_change_is_rejected(self):
        original = read(self.candidate, "events.jsonl")
        for action, reason in ((999, "illegal_final_action"), (2, "multiple_actor_intervention")):
            write(self.candidate, "events.jsonl", original)
            self.mutate_events(lambda events: next(e for e in events if e["event"] == "step" and e["step"] == 2)["actions"].__setitem__(1, action))
            self.rejection(reason)

    def test_wrong_heal_command_or_retained_target_is_rejected(self):
        self.mutate_events(lambda events: next(e for e in events if e["event"] == "step" and e["step"] == 2)
                           ["feedback"]["actors"]["u0"]["command"].update(kind="move"))
        self.rejection("feedback_target_mismatch")
        # Both prefixes still agree, but a purported retained target is not
        # the target of the actually observed order.
        make_run(self.root / "fresh-candidate", "candidate")
        self.candidate = self.root / "fresh-candidate"
        def change_order(events):
            event = next(e for e in events if e["event"] == "skill_observation" and e["step"] == 1)
            event["payload"]["state"]["allies"]["u0"]["orders"][0]["target_unit"] = "u2"
        self.mutate_events(change_order, self.base)
        self.mutate_events(change_order, self.candidate)
        self.rejection("retained_target_not_observed")

    def test_base_already_targeted_requires_matching_observed_order_and_command(self):
        self.base, self.candidate = self.root / "already-base", self.root / "already-candidate"
        make_run(self.base, "base", keep=False)
        make_run(self.candidate, "candidate", keep=False)
        def reject_redundant(events):
            feedback = next(e for e in events if e["event"] == "step" and e["step"] == 2)["feedback"]["actors"]["u0"]
            feedback.update(submission="rejected", response={"code": 206, "name": "AlreadyTargeted"})
        self.mutate_events(reject_redundant, self.base)
        result = build_dataset(self.base, self.candidate)
        self.assertEqual(result["rejected"], [])
        row = result["rows"][0]
        self.assertEqual(row["feedback"]["base"]["submission"], "rejected")
        self.assertEqual(row["feedback"]["base"]["response"], {"code": 206, "name": "AlreadyTargeted"})
        self.assertEqual(row["feedback_audit"]["base"]["acceptance_semantics"], "observed_existing_order")
        self.assertFalse(row["feedback_audit"]["base"]["healing_effect_confirmed"])
        original_base = read(self.base, "events.jsonl")
        original_candidate = read(self.candidate, "events.jsonl")
        cases = (("no_order", "existing_heal_order_not_observed"),
                 ("wrong_order_target", "existing_heal_order_not_observed"),
                 ("wrong_command_target", "existing_heal_command_mismatch"),
                 ("wrong_command_tag", "existing_heal_command_mismatch"),
                 ("wrong_code", "feedback_failed"), ("candidate_rejected", "feedback_failed"))
        for change, reason in cases:
            with self.subTest(change=change):
                write(self.base, "events.jsonl", original_base)
                write(self.candidate, "events.jsonl", original_candidate)
                if change in ("no_order", "wrong_order_target"):
                    def alter_order(events):
                        actor = next(e for e in events if e["event"] == "skill_observation" and e["step"] == 1)["payload"]["state"]["allies"]["u0"]
                        if change == "no_order":
                            actor["orders"] = []
                        else:
                            actor["orders"][0]["target_unit"] = "u2"
                    self.mutate_events(alter_order, self.base)
                    self.mutate_events(alter_order, self.candidate)
                elif change == "candidate_rejected":
                    self.mutate_events(reject_redundant, self.candidate)
                else:
                    def alter_feedback(events):
                        feedback = next(e for e in events if e["event"] == "step" and e["step"] == 2)["feedback"]["actors"]["u0"]
                        if change == "wrong_code":
                            feedback["response"]["code"] = 9
                        else:
                            key = "target_unit" if change == "wrong_command_target" else "target_tag"
                            feedback["command"][key] = "wrong"
                    self.mutate_events(alter_feedback, self.base)
                self.rejection(reason)

    def test_changed_runtime_source_and_incomplete_run_are_rejected(self):
        run = read(self.candidate, "run.json")
        original = deepcopy(run)
        run["source_hashes"]["runtime.py"] = "c" * 64
        write(self.candidate, "run.json", run)
        self.rejection("source_hashes_mismatch")
        write(self.candidate, "run.json", original)
        summary = read(self.candidate, "summary.json")
        summary["status"] = "failed"
        write(self.candidate, "summary.json", summary)
        self.rejection("run_not_complete")

    def test_missing_file_is_preserved_as_rejection(self):
        (self.candidate / "episodes.json").unlink()
        self.rejection("episodes.json")

    def test_baseline_repeat_is_noise_only_never_a_label(self):
        repeat = self.root / "repeat"
        make_run(repeat, "base", 1.5)
        result = build_dataset(self.base, repeat, baseline_noise=True)
        self.assertEqual(result["rows"], [])
        self.assertEqual(result["rejected"], [])
        self.assertEqual(result["noise_audit"][0]["return_difference"], -.5)
        self.assertFalse(result["noise_audit"][0]["training_label"])
        self.assertEqual(result["noise_audit"][0]["first_divergent_step"], 2)

    def test_cli_writes_once_and_refuses_overwrite(self):
        output = self.root / "dataset.json"
        argv = ["--base-run", str(self.base), "--candidate-run", str(self.candidate), "--output", str(output)]
        with redirect_stdout(io.StringIO()):
            self.assertEqual(main(argv), 0)
        original = output.read_bytes()
        with self.assertRaises(SystemExit), redirect_stderr(io.StringIO()):
            main(argv)
        self.assertEqual(output.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
