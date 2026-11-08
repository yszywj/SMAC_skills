"""Small synthetic, offline tests; no native game or large trace is read."""
from copy import deepcopy
import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest

from decision_tree.scripts import conditional_completion_check as check


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


class Fixture:
    def __init__(self, directory):
        self.project = Path(directory).resolve()
        self.root = self.project / "campaign"
        self.root.mkdir()
        source, auditor = self.project / "frozen.py", self.project / "compat.py"
        source.write_text("# frozen\n", encoding="utf-8")
        auditor.write_text("# compatibility auditor\n", encoding="utf-8")
        self.sources = {"frozen.py": check.checksum(source)}
        self.compat = {"kind": "json_tuple_list_audit_compatibility", "script": str(auditor),
                       "sha256": check.checksum(auditor), "runtime_sources_modified": False,
                       "native_decision_rules_modified": False}
        save(self.root / "audit_compatibility.json", self.compat)
        self.attempts, self.jobs = [], []
        self.initial = {"kind": "fixture_initial", "corrections": []}
        save(self.root / "initial_policy.json", self.initial)
        self.protocol = {"kind": "conditional_v3_bounded_3x40", "rounds": [], "source_hashes": self.sources,
                         "max_native_attempts": 780,
                         "budget": {"planned_games_max": 720, "scenes_per_round": 40,
                                    "train_scenes": 30, "holdout_scenes": 10,
                                    "candidate_branches_per_train_scene": 4, "max_steps_per_scene": 4,
                                    "max_effects_per_step": 8, "max_effects_per_family_per_step": 2}}
        for index in range(1, 4):
            self.round(index)
        save(self.root / "protocol.json", self.protocol)
        save(self.root / "source_revision.json", {"original_source_hashes": self.sources,
                                                   "source_hashes": self.sources, "reason": "fixture"})
        with sqlite3.connect(self.root / "native_attempts.sqlite3") as conn:
            conn.execute("CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            conn.execute("CREATE TABLE jobs(output TEXT PRIMARY KEY, identity TEXT NOT NULL)")
            conn.execute("CREATE TABLE attempts(id INTEGER PRIMARY KEY, output TEXT, attempt INTEGER, path TEXT, slots INTEGER, state TEXT, pid INTEGER, started REAL, returncode INTEGER, error TEXT)")
            conn.execute("INSERT INTO meta VALUES('protocol', ?)", (check.digest(self.protocol),))
            conn.executemany("INSERT INTO jobs VALUES (?, ?)", self.jobs)
            conn.executemany("INSERT INTO attempts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", self.attempts)

    def run(self, path, manifest, policy=None, forced=None, recovery=False, win=False):
        groups = check.manifest_groups(manifest)
        metadata = {"policy": policy, "policy_sha256": check.digest(policy), "forced": forced,
                    "source_hashes": self.sources}
        native = {"manifest_sha256": manifest["sha256"]}
        summary = {"manifest_sha256": manifest["sha256"], "status": "complete", "error": None,
                   "completed_episodes": len(groups), "expected_episodes": len(groups)}
        if policy is not None:
            native["conditional_skills"] = metadata
            summary["conditional_skills"] = metadata
        records = [{"episode": index + 1, "scenario_sha256": group,
                    "initial_state_sha256": "initial_" + group,
                    "reward": 20.0 if win else 10.0, "outcome": "win" if win else "loss"}
                   for index, group in enumerate(groups)]
        for name, value in (("run.json", native), ("summary.json", summary),
                            ("manifest.json", manifest), ("episodes.json", records)):
            save(path / name, value)
        (path / "events.jsonl").write_text('{"fixture": true}\n', encoding="utf-8")
        number = len(self.attempts) + 1
        self.attempts.append((number, str(path), 1, str(path), len(groups),
                              "failed" if recovery else "exited", None, 0.0, 1 if recovery else 0, None))
        identity = {"command": ["fixture", "--manifest", str(path / "manifest.json")],
                    "inputs": {str(path / "manifest.json"): check.checksum(path / "manifest.json")},
                    "episode_slots": len(groups)}
        self.jobs.append((str(path), json.dumps(identity)))
        hashes = {name: check.checksum(path / name) for name in (*check.SMALL_EVIDENCE, "events.jsonl")}
        if recovery:
            save(path / "audit_recovery.json", {"kind": "json_representation_audit_recovery",
                 "native_attempt_id": number, "original_native_returncode": 1,
                 "native_ledger_modified": False, "source_hashes": self.sources,
                 "auditor_sha256": self.compat["sha256"], "completed_episodes": len(groups),
                 "evidence_sha256": hashes})
            console = self.root / "launch_state" / f"attempt_{number:05d}" / "console.log"
            console.parent.mkdir(parents=True)
            console.write_text(check.AUDIT_ERROR, encoding="utf-8")
        return {row["scenario_sha256"]: row for row in records}, hashes

    def round(self, index):
        folder = self.root / f"round_{index}"
        scenes = []
        for number in range(40):
            config = {"fixture_round": index, "fixture_scene": number}
            scenes.append({"index": number, "sc2_seed": index * 1000 + number,
                           "episode_config": config, "sha256": check.digest(config)})
        body = {"version": 1, "seed": index * 1000,
                "env_args": {"map_name": "10gen_protoss", "capability_config": {"n_units": 10, "n_enemies": 10}},
                "episodes": scenes}
        manifest = dict(body, sha256=check.digest(body))
        groups = [scene["sha256"] for scene in scenes]
        train, holdout = groups[:30], groups[30:]
        split = {"training_group_ids": train, "validation_group_ids": holdout}
        train_manifest = check.subset_manifest(manifest, train)
        holdout_manifest = check.subset_manifest(manifest, holdout)
        candidate = {"kind": "fixture_candidate", "round": index}
        for name, value in (("manifest.json", manifest), ("split.json", split),
                            ("train_manifest.json", train_manifest), ("holdout_manifest.json", holdout_manifest),
                            ("incumbent_policy.json", self.initial), ("candidate_policy.json", candidate)):
            save(folder / name, value)
        self.protocol["rounds"].append({"round": index, "manifest_sha256": manifest["sha256"], "split": split})
        adaptive, _ = self.run(folder / "adaptive", manifest)
        incumbent, incumbent_hashes = self.run(folder / "incumbent", manifest, self.initial, recovery=index == 1)
        job = {"episode": 1, "scene_hash": groups[0], "seed": scenes[0]["sc2_seed"], "step": 5,
               "binding": "BASE", "effect": "BASE", "family": "retarget", "effect_families": ["retarget"]}
        branch, branch_hashes = self.run(folder / "branches" / "001", check.subset_manifest(manifest, [groups[0]]),
                                       self.initial, {"step": 5, "binding": "BASE"}, win=True)
        row = {**job, "id": f"r{index}_1", "group_id": groups[0],
               "scope": "single_override_then_same_conditional_policy", "different_action": True,
               "candidate": {"return": 20.0, "outcome": "win"}, "reference": {"return": 10.0, "outcome": "loss"},
               "branch": {"return": 20.0, "outcome": "win"}, "incumbent": {"return": 10.0, "outcome": "loss"},
               "win_gain": 1, "return_gain": 10.0, "incumbent_sources": incumbent_hashes, "branch_sources": branch_hashes}
        save(folder / "branches" / "001.pair.json", row)
        save(folder / "measured.json", [row])
        save(folder / "freeze.json", {"candidate_sha256": check.digest(candidate), "training_sha256": check.digest([row]),
                                     "split_sha256": check.digest(split)})
        save(folder / "branch_plan.json", {"round_index": index, "training_groups": sorted(train),
             "holdout_groups": sorted(holdout), "jobs": [job], "planned_branch_games": 1,
             "incumbent": str(folder / "incumbent"), "source_hashes": self.sources,
             "incumbent_policy_sha256": check.digest(self.initial),
             "incumbent_manifest_sha256": incumbent_hashes["manifest.json"],
             "incumbent_events_sha256": incumbent_hashes["events.jsonl"]})
        save(folder / "run_paths.json", {"adaptive": str(folder / "adaptive"), "incumbent": str(folder / "incumbent")})
        trained, _ = self.run(folder / "candidate_train", train_manifest, candidate)
        validated, _ = self.run(folder / "candidate_holdout", holdout_manifest, candidate)
        report = {"training": check.whole_policy(trained, incumbent, train),
                  "holdout_vs_incumbent": check.whole_policy(validated, incumbent, holdout),
                  "holdout_vs_adaptive": check.whole_policy(validated, adaptive, holdout),
                  "adopted": False, "candidate_train": str(folder / "candidate_train"),
                  "candidate_holdout": str(folder / "candidate_holdout")}
        save(folder / "training_whole_policy.json", report["training"])
        save(folder / "evaluation.json", report)
        save(folder / "adopted_policy.json", self.initial)


class CompletionCheckTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.fixture = Fixture(self.temp.name)
        self.root = self.fixture.root

    def check(self):
        return check.check_campaign(self.root, project_root=self.fixture.project)

    def change(self, name, mutate):
        path = self.root / name
        data = check.read(path)
        mutate(data)
        save(path, data)

    def test_complete_preserves_audit_failure_and_does_not_read_trace(self):
        ledger = (self.root / "native_attempts.sqlite3").read_bytes()
        report = self.check()
        self.assertEqual(report["independent_scenes"], 120)
        self.assertEqual(report["native_ledger"]["reserved_episode_slots"], 363)
        self.assertEqual(report["native_ledger"]["retry_episode_slots"], 0)
        self.assertEqual(report["audit_only_recoveries"][0]["episodes_reused"], 40)
        self.assertEqual(report["audit_only_recoveries"][0]["additional_native_games"], 0)
        self.assertFalse(report["trace_audit"]["replayed"])
        self.assertFalse((self.root / "completion_checks.json").exists())
        output, saved = check.write_completion(self.root, project_root=self.fixture.project)
        self.assertEqual(check.read(output), saved)
        check.write_completion(self.root, project_root=self.fixture.project)
        self.assertEqual(ledger, (self.root / "native_attempts.sqlite3").read_bytes())

    def test_incomplete_refuses_success_file(self):
        # Move only this temporary fixture's own evidence to simulate an unfinished round.
        path = self.root / "round_3/evaluation.json"
        path.rename(path.with_suffix(".pending"))
        with self.assertRaisesRegex(ValueError, "Missing completion evidence"):
            check.write_completion(self.root, project_root=self.fixture.project)
        self.assertFalse((self.root / "completion_checks.json").exists())

    def test_active_attempt_rejected(self):
        with sqlite3.connect(self.root / "native_attempts.sqlite3") as conn:
            conn.execute("UPDATE attempts SET state='running' WHERE id=15")
        with self.assertRaisesRegex(ValueError, "running/reserved"):
            self.check()

    def test_frozen_source_change_rejected(self):
        (self.fixture.project / "frozen.py").write_text("# changed\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Frozen source changed"):
            self.check()

    def test_freeze_tamper_rejected(self):
        self.change("round_2/freeze.json", lambda value: value.update(training_sha256="wrong"))
        with self.assertRaisesRegex(ValueError, "freeze digest"):
            self.check()

    def test_wrong_measured_id_rejected(self):
        self.change("round_2/measured.json", lambda value: value[0].update(id="r1_1"))
        measured = check.read(self.root / "round_2/measured.json")
        self.change("round_2/freeze.json", lambda value: value.update(training_sha256=check.digest(measured)))
        with self.assertRaisesRegex(ValueError, "Measured ID"):
            self.check()

    def test_heldout_job_rejected(self):
        group = check.read(self.root / "round_2/split.json")["validation_group_ids"][0]
        self.change("round_2/branch_plan.json", lambda value: value["jobs"][0].update(scene_hash=group))
        self.change("round_2/measured.json", lambda value: value[0].update(scene_hash=group, group_id=group))
        measured = check.read(self.root / "round_2/measured.json")
        self.change("round_2/freeze.json", lambda value: value.update(training_sha256=check.digest(measured)))
        with self.assertRaisesRegex(ValueError, "held-out"):
            self.check()

    def test_branch_budget_rejected(self):
        def add(value):
            value["jobs"] = [deepcopy(value["jobs"][0]) for _ in range(5)]
        self.change("round_2/branch_plan.json", add)
        with self.assertRaisesRegex(ValueError, "four measured branches"):
            self.check()

    def test_native_cap_rejected(self):
        with sqlite3.connect(self.root / "native_attempts.sqlite3") as conn:
            conn.execute("UPDATE attempts SET slots=800 WHERE id=1")
        with self.assertRaisesRegex(ValueError, "episode-slot cap exceeded"):
            self.check()

    def test_unrelated_exit_one_cannot_be_recovered(self):
        console = self.root / "launch_state/attempt_00002/console.log"
        console.write_text("game crashed", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "recognized audit error"):
            self.check()

    def test_adoption_verdict_rejected(self):
        self.change("round_3/evaluation.json", lambda value: value.update(adopted=True))
        with self.assertRaisesRegex(ValueError, "win-first"):
            self.check()

    def test_duplicate_scene_across_rounds_rejected(self):
        source = check.read(self.root / "round_1/manifest.json")
        self.change("round_2/manifest.json", lambda value: value.update(source))
        with self.assertRaisesRegex(ValueError, "120 independent scenes"):
            self.check()


if __name__ == "__main__":
    unittest.main()
