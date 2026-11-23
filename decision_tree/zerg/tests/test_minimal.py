"""Minimal acceptance tests for the zerg decision-tree framework (Task 14).

Covers exactly the five required checks plus one training round-trip:

1. base+overlay source merge correctness and determinism;
2. feature schema generation from a real snapshot;
3. legal candidate -> uniform ZergCandidate conversion;
4. shadow mode never changes actions;
5. the original tactical_v1 import and step entry still work;
6. (extra) records -> ZergTrainingBackend -> scorer model -> scoring.

Run from the repository root:
    python -m unittest decision_tree.zerg.tests.test_minimal -v
No SC2 game is started anywhere in this file.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[3]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from decision_tree.zerg import candidates as zc
from decision_tree.zerg import dataset as zd
from decision_tree.zerg import objectives as zo
from decision_tree.zerg import runtime as zr
from decision_tree.zerg import schema as zs
from decision_tree.zerg import scorer as zscore
from decision_tree.zerg.features import (DEFAULT_SCHEMA, MISSING,
                                         StateFeatureExtractor)
from decision_tree.zerg.source import ZergSkillSourceResolver
from decision_tree.zerg.training import ZergTrainingBackend
from decision_tree.zerg.tests._helpers import synthetic_audit


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TestSourceResolution(unittest.TestCase):
    def test_merge_is_complete_deterministic_and_traceable(self):
        base = WORKSPACE / "skills/tactical_v1"
        overlay = (WORKSPACE / "runs/contact_all_attack_20261122_230107_662982"
                   / "marginal_near/skills/tactical_v1")
        before = {str(p): _sha(p) for p in [base / "skill_manager.py",
                                            overlay / "coordinator.py"]}
        with tempfile.TemporaryDirectory() as tmp:
            resolver = ZergSkillSourceResolver(
                base_dir=str(base), overlay_dir=str(overlay),
                output_dir=f"{tmp}/resolved")
            result = resolver.resolve()
            manifest_path = Path(result.output_dir) / "source_manifest.json"
            self.assertTrue(manifest_path.is_file())
            manifest = json.loads(manifest_path.read_text())
            self.assertEqual(result.fingerprint, manifest["fingerprint"])
            self.assertEqual(manifest["schema_version"],
                             "zerg_source_manifest_v1")
            # overlay wins on same-name files
            merged_manager = (Path(result.output_dir)
                              / "skill_manager.py").read_bytes()
            self.assertEqual(merged_manager,
                             (overlay / "skill_manager.py").read_bytes())
            # overlay-only files added, base-only files preserved
            self.assertTrue((Path(result.output_dir)
                             / "local_engagement.py").is_file())
            self.assertTrue((Path(result.output_dir)
                             / "zerg_hydra_policy.py").is_file())
            # every file carries provenance
            row = next(f for f in manifest["files"]
                       if f["path"] == "skill_manager.py")
            self.assertEqual(row["source"], "overlay")
            self.assertTrue(row["overridden"])
            self.assertEqual(len(row["sha256"]), 64)
            # deterministic repeat
            again = ZergSkillSourceResolver(
                base_dir=str(base), overlay_dir=str(overlay),
                output_dir=f"{tmp}/resolved").resolve()
            self.assertEqual(again.fingerprint, result.fingerprint)
            self.assertFalse(
                ZergSkillSourceResolver(
                    base_dir=str(base), overlay_dir=str(overlay),
                    output_dir=f"{tmp}/resolved").has_changed())
        self.assertEqual(_sha(base / "skill_manager.py"),
                         before[str(base / "skill_manager.py")])
        self.assertEqual(_sha(overlay / "coordinator.py"),
                         before[str(overlay / "coordinator.py")])


def _zerg_payload():
    from skills.tactical_v1.tests import fixtures as fx
    allies = [((17.0, 17.0), "hydralisk"), ((18.0, 17.0), "hydralisk"),
              ((17.0, 18.0), "zergling"), ((18.0, 18.0), "zergling")]
    enemies = {"e0": fx.enemy((20.0, 20.0), "marine", health=30.0),
               "e1": fx.enemy((22.0, 20.0), "marine", health=12.0)}
    state = fx.scene(allies, enemies, map_name="10gen_terran")
    return fx.with_questions(state)


def _manager():
    from skills.tactical_v1.skill_manager import SkillManager
    return SkillManager()


def _step_once(payload):
    manager = _manager()
    actions, summary = manager.step(copy.deepcopy(payload), 4)
    manager.commit_final_actions(actions)
    return manager, actions, summary


class TestFeaturesAndCandidates(unittest.TestCase):
    def test_schema_and_extraction_from_real_snapshot(self):
        from skills.tactical_v1.context import build_snapshot
        payload = _zerg_payload()
        snapshot = build_snapshot(payload, 1)
        extractor = StateFeatureExtractor(DEFAULT_SCHEMA)
        state = extractor.extract_state(snapshot)
        self.assertEqual(set(state["team"]),
                         set(DEFAULT_SCHEMA.names(layer="team")))
        self.assertEqual(set(state["skill"]),
                         set(DEFAULT_SCHEMA.names(layer="skill")))
        self.assertIn("u0", state["units"])
        self.assertEqual(state["unit_types"]["u2"], "zergling")
        hydra_features = state["units"]["u0"]
        self.assertEqual(set(hydra_features),
                         set(DEFAULT_SCHEMA.names(layer="unit",
                                                  unit_type="hydralisk")))
        # missing values are NaN, never folded into 0
        self.assertTrue(math.isnan(hydra_features["retreat_quality"]))
        # deterministic
        again = extractor.extract_state(build_snapshot(payload, 1))
        self.assertEqual(state, again)
        # digest stable
        self.assertEqual(DEFAULT_SCHEMA.digest(),
                         DEFAULT_SCHEMA.digest())

    def test_candidate_conversion_from_live_skills(self):
        from skills.tactical_v1.context import build_snapshot
        payload = _zerg_payload()
        manager = _manager()
        actions, summary = manager.step(copy.deepcopy(payload), 4)
        # The manager's own prepared snapshot/context (read-only reuse) — the
        # candidates below are the skill's real propose() output.
        snapshot, _, _, _, context = manager._pending_commit
        plans = manager.skills["focus_fire"].propose(
            context, manager.memory, manager.telemetry, None)
        self.assertTrue(plans, "fixture should yield focus_fire plans")
        internal = zc.skill_internal_candidates(plans, snapshot,
                                                context=context)
        self.assertTrue(internal)
        for candidate in internal:
            record = candidate.to_record()
            self.assertEqual(record["schema"], zc.CANDIDATE_SCHEMA)
            self.assertEqual(set(record["features"]),
                             set(DEFAULT_SCHEMA.names(layer="candidate")))
            self.assertTrue(candidate.legal)
        # summary-level joint candidates
        joint = zc.joint_candidates_from_summary(summary, snapshot,
                                                 payload=payload)
        self.assertTrue(joint)
        self.assertTrue(joint[0].legal)
        points = zc.collect_step_candidates(summary, snapshot,
                                            payload=payload)
        self.assertIn(zc.FAMILY_JOINT_PLAN, points)
        manager.commit_final_actions(actions)


class TestShadowMode(unittest.TestCase):
    def test_shadow_and_off_preserve_actions(self):
        payload = _zerg_payload()
        _, plain_actions, plain_summary = _step_once(payload)

        config = zr.default_config()
        config["mode"] = "shadow"
        with tempfile.TemporaryDirectory() as tmp:
            config["workspace"] = tmp
            config["source_root"] = str(WORKSPACE)
            config["records_path"] = f"{tmp}/shadow.jsonl"
            runtime = zr.ZergTreeRuntime(config=config,
                                         tactical_config={})
            controller = runtime.wrap(_manager())
            actions, summary = controller.step(copy.deepcopy(payload), 4)
            self.assertEqual(actions, plain_actions)
            self.assertFalse(summary["zerg_tree"]["changed"])
            controller.commit_final_actions(actions)     # pass-through
            written = runtime.finalize_episode(
                outcome="win", episode_return=1.0, ally_losses=0,
                enemy_losses=2, episode_id="1")
            self.assertEqual(written, 1)
            records = zd.read_records(f"{tmp}/shadow.jsonl")
            self.assertEqual(len(records), 1)
            record = records[0]
            self.assertEqual(record["outcome"], "win")
            self.assertTrue(record["teacher_choice"])
            self.assertTrue(record["candidates"])
            self.assertTrue(record["execution_context_hash"])
            # tree_score with no fitted models must also never change actions
            config["mode"] = "tree_score"
            config["records_path"] = f"{tmp}/tree.jsonl"
            tree_runtime = zr.ZergTreeRuntime(config=config,
                                              tactical_config={})
            controller = tree_runtime.wrap(_manager())
            actions2, _ = controller.step(copy.deepcopy(payload), 4)
            self.assertEqual(actions2, plain_actions)
            controller.commit_final_actions(actions2)
        # off mode: wrap returns the manager itself, untouched
        off = zr.ZergTreeRuntime(config=zr.default_config())
        sentinel = object()
        self.assertIs(off.wrap(sentinel), sentinel)


class TestOriginalSkillIntact(unittest.TestCase):
    def test_import_and_step_entry(self):
        from skills.tactical_v1.skill_manager import SkillManager
        manager = SkillManager()
        self.assertTrue(manager.active)
        payload = _zerg_payload()
        actions, summary = manager.step(payload, 4)
        self.assertEqual(len(actions), 4)
        self.assertTrue(all(isinstance(a, int) for a in actions))
        manager.commit_final_actions(actions)
        # idempotent repeat returns cached copies
        again, _ = manager.step(payload, 4)
        self.assertEqual(again, actions)
        manager.reset()
        self.assertEqual(manager.step(payload, 4)[0], actions)


class TestTrainingRoundTrip(unittest.TestCase):
    def _synthetic_records(self, scenes=12):
        """Build valid decision records with fabricated paired returns so the
        gain task has auditable labels (labels are for pipeline testing only,
        never evidence)."""
        records = []
        rng_state = 0

        def pseudo():
            nonlocal rng_state
            rng_state = (rng_state * 1103515245 + 12345) % (1 << 31)
            return rng_state / (1 << 31)

        for scene in range(scenes):
            candidates = []
            gains = {}
            base_return = 1.0
            for index in range(3):
                cid = f"joint_cand_{index}"
                gain = round(pseudo() * 0.4 - 0.2, 4)
                gains[cid] = gain
                features = {name: round(pseudo(), 3)
                            for name in DEFAULT_SCHEMA.names(
                                layer="candidate")}
                candidates.append({
                    "candidate_id": cid, "candidate_family": "joint_plan",
                    "candidate_skill": "focus_fire",
                    "controlled_actors": ["u0"], "target": "e0",
                    "phase": "engage", "current_actions": {},
                    "proposed_actions": {"u0": 6}, "features": features,
                    "resource_claims": {}, "dependencies": [],
                    "atomic": True, "legal": True, "rejection_reason": None,
                    "source_plan_id": cid,
                    "candidate_return": base_return + gain,
                })
            teacher = max(gains, key=lambda cid: gains[cid])
            unit_features = {
                name: 0.5 for name in DEFAULT_SCHEMA.names(
                    layer="unit", unit_type="hydralisk")}
            records.append({
                "schema": zs.DECISION_RECORD_SCHEMA,
                "run_id": "run_test",
                "episode_id": f"ep{scene:03d}", "scene_id": f"scene{scene:03d}",
                "game_loop": 100 + scene, "snapshot_id": scene + 1,
                "source_hash": "0" * 64, "execution_context_hash": "1" * 64,
                "feature_schema_version": DEFAULT_SCHEMA.version,
                "global_features": {
                    "team": {name: 1.0 for name in
                             DEFAULT_SCHEMA.names(layer="team")},
                    "skill": {name: 0.0 for name in
                              DEFAULT_SCHEMA.names(layer="skill")},
                    "units": {"u0": unit_features},
                    "unit_types": {"u0": "hydralisk"}},
                "candidates": candidates,
                "teacher_choice": teacher,
                "model_scores": [],
                "selected_candidate": teacher,
                "final_actions": {"u0": 6},
                "execution_status": "committed",
                "outcome": "win", "return": base_return,
                "baseline_return": base_return,
                "reward_provenance": {"ally_losses": 0, "enemy_losses": 1},
            })
        return records

    def test_imitation_and_gain_then_score(self):
        import tempfile
        records = self._synthetic_records()
        backend = ZergTrainingBackend()
        context_hash = "a" * 64
        with tempfile.TemporaryDirectory() as tmp:
            imitation, report = backend.train(
                "imitation", records, scorer="hydralisk_scorer",
                execution_context_hash=context_hash,
                capacity={"minimum_leaf_support": 2,
                          "minimum_leaf_groups": 2})
            zs.validate_scorer_model(imitation)
            self.assertEqual(imitation["scorer"], "hydralisk_scorer")
            self.assertEqual(imitation["capacity"]["number_of_trees"], 1)
            audited_bundle, audited_envelope = synthetic_audit(
                records, execution_context_hash=context_hash)
            gain_model, gain_report = backend.train(
                "gain", audited_bundle.joint_records, scorer="joint_scorer",
                execution_context_hash=context_hash,
                audit_envelope=audited_envelope,
                capacity={"minimum_leaf_support": 2,
                          "minimum_leaf_groups": 2})
            zs.validate_scorer_model(gain_model)
            # feature-schema mismatch must refuse to load
            tampered = copy.deepcopy(imitation)
            tampered["feature_schema"]["digest"] = "deadbeef"
            with self.assertRaises(ValueError):
                zscore.ScorerModel(tampered)
            # a matching model loads and scores with fallbacks
            model = zscore.ScorerModel(imitation)
            feature_row = {
                f"team.{name}": 1.0
                for name in DEFAULT_SCHEMA.names(layer="team")}
            feature_row.update({
                f"skill.{name}": 0.0
                for name in DEFAULT_SCHEMA.names(layer="skill")})
            candidate = zc.ZergCandidate(
                candidate_id="x", candidate_family="skill_internal",
                candidate_skill="focus_fire", controlled_actors=("u0",),
                target="e0", phase="engage", current_actions={},
                proposed_actions={"u0": 6},
                features={name: 0.5 for name in
                          DEFAULT_SCHEMA.names(layer="candidate")},
                resource_claims={}, dependencies=[], atomic=False,
                legal=True, rejection_reason=None, source_plan_id="x",
                rule_score=1.0)
            state_features = {
                "team": {name: 1.0 for name in
                         DEFAULT_SCHEMA.names(layer="team")},
                "skill": {name: 0.0 for name in
                          DEFAULT_SCHEMA.names(layer="skill")}}
            scores = zscore.score_candidates(state_features, [candidate],
                                             model=model)
            self.assertEqual(len(scores), 1)
            self.assertIn(scores[0].reason, {
                "model_score", "leaf_below_support_rule_confidence",
                "prediction_unsupported_rule_fallback"})
            # save/load round trip through JSONL-safe dumps
            path = Path(tmp) / "model.json"
            path.write_text(json.dumps(zs._sanitize(imitation), indent=2))
            loaded = zscore.ScorerModel(zs.load_json(str(path)))
            self.assertEqual(loaded.scorer, "hydralisk_scorer")

    def test_objective_guards(self):
        rows = [{
            "group_id": "sceneA", "decision_key": "1", "candidate_id": "a",
            "weight": 0.5, "teacher": 1.0, "rule_rank": 0.0,
            "candidate_return": 2.0, "base_return": 1.0, "gain": 1.0,
        }, {
            "group_id": "sceneA", "decision_key": "1", "candidate_id": "b",
            "weight": 0.5, "teacher": 0.0, "rule_rank": 1.0,
            "candidate_return": None, "base_return": 1.0, "gain": None,
        }]
        fit, problems = zo.gain_rows(rows, strict=True)
        self.assertEqual(len(fit), 1)
        self.assertEqual(fit[0]["gain"], 1.0)
        self.assertEqual(problems, [])
        fabricated = dict(rows[1], gain=0.7)
        _, problems = zo.gain_rows([fabricated], strict=True)
        self.assertTrue(problems)


if __name__ == "__main__":
    unittest.main()
