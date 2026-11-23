"""V1.3 gain-audit fix acceptance tests (Task 6).

Run from the repository root:
    python -m unittest decision_tree.zerg.tests.test_v15 -v
"""
from __future__ import annotations

import copy
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
from decision_tree.zerg import pair_dataset as pd
from decision_tree.zerg import schema as zs
from decision_tree.zerg import training as zt
from decision_tree.zerg.candidates import branch_scoring_features, \
    build_branch_candidates
from decision_tree.zerg.experiment_manifest import make_anchor_manifest
from decision_tree.zerg.features import DEFAULT_SCHEMA, MISSING
from decision_tree.zerg.schema import build_internal_model_features, \
    build_model_features
from decision_tree.zerg.training import select_active_features, \
    verify_gain_envelope

from decision_tree.zerg.tests.test_minimal import _manager
from decision_tree.zerg.tests.test_minimal import _zerg_payload as _payload
from decision_tree.zerg.tests.test_v11 import _records
from decision_tree.zerg.tests._helpers import synthetic_audit

CTX = "1" * 64
SOURCE = "0" * 64


class Test01ManifestRequired(unittest.TestCase):
    """Check 1: formal gain dataset without a manifest fails."""

    def test_missing_manifest_rejected(self):
        records = _records(4)
        pairs = [{"schema": "zerg_counterfactual_pair_v1",
                  "experiment_id": "exp", "run_id": records[0]["run_id"],
                  "episode_id": records[0]["episode_id"],
                  "anchor_id": "a:1", "pair_id": "p0",
                  "baseline_id": "BASE", "candidate_id": "x",
                  "initial_state_hash": "a", "prefix_hash": "b",
                  "execution_context_hash": CTX,
                  "source_fingerprint": SOURCE,
                  "intervention_type": "single_step",
                  "intervention_horizon": 1,
                  "completion_status": "complete", "scope": "joint"}]
        with self.assertRaises(ValueError):
            pd.build_gain_dataset(records, pairs)          # no manifest
        with self.assertRaises(ValueError):
            pd.build_gain_dataset(records, pairs, anchor_manifest={})
        # receipts are mandatory too
        manifest = make_anchor_manifest(
            experiment_id="exp", source_fingerprint=SOURCE,
            execution_context_hash=CTX, anchors={"a:1": {
                "initial_state_hash": "a", "prefix_hash": "b",
                "baseline_id": "BASE"}})
        with self.assertRaises(ValueError):
            pd.build_gain_dataset(records, pairs, anchor_manifest=manifest)

    def test_manifest_integrity(self):
        """Check 2: a tampered manifest_sha256 fails validation."""
        manifest = make_anchor_manifest(
            experiment_id="exp", source_fingerprint=SOURCE,
            execution_context_hash=CTX, anchors={"a:1": {
                "initial_state_hash": "a", "prefix_hash": "b",
                "baseline_id": "BASE"}})
        self.assertEqual(
            zs.validate_execution_receipt.__module__, "decision_tree.zerg.schema")
        problems = __import__(
            "decision_tree.zerg.experiment_manifest",
            fromlist=["validate_anchor_manifest"]).validate_anchor_manifest(
                manifest, expected_experiment_id="exp")
        self.assertEqual(problems, [])
        tampered = dict(manifest)
        tampered["anchors"] = dict(manifest["anchors"])
        tampered["anchors"]["a:1"] = dict(
            manifest["anchors"]["a:1"], prefix_hash="TAMPERED")
        problems = __import__(
            "decision_tree.zerg.experiment_manifest",
            fromlist=["validate_anchor_manifest"]).validate_anchor_manifest(
                tampered, expected_experiment_id="exp")
        self.assertTrue(any("integrity" in p for p in problems))
        # experiment_id mismatch
        problems = __import__(
            "decision_tree.zerg.experiment_manifest",
            fromlist=["validate_anchor_manifest"]).validate_anchor_manifest(
                manifest, expected_experiment_id="other")
        self.assertTrue(any("experiment_id" in p for p in problems))


class Test02IdentityAxes(unittest.TestCase):
    """Checks 3/4/12: source fingerprint, execution context and candidate
    identity must match between pair, manifest and record."""

    def setUp(self):
        self.records = _records(4)
        self.bundle, self.envelope = synthetic_audit(self.records)
        self.pairs = None
        # rebuild the synthetic pairs to probe individually
        from decision_tree.zerg.tests import _helpers
        self.helpers = _helpers

    def test_source_mismatch_against_manifest(self):
        records = self.records
        pair = {
            "schema": "zerg_counterfactual_pair_v1",
            "experiment_id": "exp_test_audit", "run_id": records[0]["run_id"],
            "episode_id": records[0]["episode_id"],
            "anchor_id": f"{records[0]['episode_id']}:"
                         f"{records[0]['snapshot_id']}",
            "pair_id": "probe", "baseline_id": "BASE",
            "candidate_id": records[0]["candidates"][1]["candidate_id"],
            "decision_key": f"{records[0]['run_id']}:"
                            f"{records[0]['episode_id']}:"
                            f"{records[0]['snapshot_id']}:"
                            f"{records[0]['game_loop']}",
            "initial_state_hash": "a" * 64, "prefix_hash": "b" * 64,
            "execution_context_hash": CTX,
            "source_fingerprint": "f" * 64,       # mismatch vs manifest
            "intervention_type": "single_step",
            "intervention_horizon": 1,
            "candidate_return": 1.2, "baseline_return": 1.0,
            "outcome": "win", "completion_status": "complete",
            "scope": "joint"}
        manifest = make_anchor_manifest(
            experiment_id="exp_test_audit", source_fingerprint=SOURCE,
            execution_context_hash=CTX, anchors={pair["anchor_id"]: {
                "initial_state_hash": "a" * 64, "prefix_hash": "b" * 64,
                "baseline_id": "BASE"}})
        joined = pd.join_counterfactual_pairs(records, [pair])
        audit = pd.audit_counterfactual_pairs(joined, records,
                                              anchor_manifest=manifest)
        self.assertEqual(audit["pairs_trainable"], 0)
        self.assertTrue(any("source_fingerprint" in p for p in
                            audit["problems_by_pair"]["probe"]))

    def test_context_mismatch_against_manifest(self):
        records = self.records
        pair = {
            "schema": "zerg_counterfactual_pair_v1",
            "experiment_id": "exp_test_audit", "run_id": records[0]["run_id"],
            "episode_id": records[0]["episode_id"],
            "anchor_id": f"{records[0]['episode_id']}:"
                         f"{records[0]['snapshot_id']}",
            "pair_id": "probe2", "baseline_id": "BASE",
            "candidate_id": records[0]["candidates"][1]["candidate_id"],
            "decision_key": f"{records[0]['run_id']}:"
                            f"{records[0]['episode_id']}:"
                            f"{records[0]['snapshot_id']}:"
                            f"{records[0]['game_loop']}",
            "initial_state_hash": "a" * 64, "prefix_hash": "b" * 64,
            "execution_context_hash": "9" * 64,   # mismatch vs manifest
            "source_fingerprint": SOURCE,
            "intervention_type": "single_step",
            "intervention_horizon": 1,
            "candidate_return": 1.2, "baseline_return": 1.0,
            "outcome": "win", "completion_status": "complete",
            "scope": "joint"}
        manifest = make_anchor_manifest(
            experiment_id="exp_test_audit", source_fingerprint=SOURCE,
            execution_context_hash=CTX, anchors={pair["anchor_id"]: {
                "initial_state_hash": "a" * 64, "prefix_hash": "b" * 64,
                "baseline_id": "BASE"}})
        joined = pd.join_counterfactual_pairs(records, [pair])
        audit = pd.audit_counterfactual_pairs(joined, records,
                                              anchor_manifest=manifest)
        self.assertEqual(audit["pairs_trainable"], 0)
        problems = audit["problems_by_pair"]["probe2"]
        self.assertTrue(any("execution_context" in p for p in problems))

    def test_candidate_scope_identity(self):
        """Check 12: a scope=candidate_id combination that does not exist in
        the decision record rejects the pair."""
        records = self.records
        pair = {
            "schema": "zerg_counterfactual_pair_v1",
            "experiment_id": "exp_test_audit", "run_id": records[0]["run_id"],
            "episode_id": records[0]["episode_id"],
            "anchor_id": f"{records[0]['episode_id']}:"
                         f"{records[0]['snapshot_id']}",
            "pair_id": "probe3", "baseline_id": "BASE",
            "candidate_id": "not_present_anywhere",
            "decision_key": f"{records[0]['run_id']}:"
                            f"{records[0]['episode_id']}:"
                            f"{records[0]['snapshot_id']}:"
                            f"{records[0]['game_loop']}",
            "initial_state_hash": "a" * 64, "prefix_hash": "b" * 64,
            "execution_context_hash": CTX, "source_fingerprint": SOURCE,
            "intervention_type": "single_step",
            "intervention_horizon": 1,
            "candidate_return": 1.2, "baseline_return": 1.0,
            "outcome": "win", "completion_status": "complete",
            "scope": "joint"}
        joined = pd.join_counterfactual_pairs(records, [pair])
        audit = pd.audit_counterfactual_pairs(joined, records)
        self.assertIn("candidate_not_in_decision_record",
                      audit["problems_by_pair"]["probe3"])


class Test03Receipts(unittest.TestCase):
    """Checks 5/6/7: missing or cross-anchor branch receipts fail."""

    def setUp(self):
        self.records = _records(4)
        self.bundle, self.envelope = synthetic_audit(self.records)

    def _receipts_from_helper(self):
        receipts = []
        source_fp = SOURCE
        for index, record in enumerate(self.records):
            anchor_id = f"{record['episode_id']}:{record['snapshot_id']}"
            for branch_id, return_value in (("BASE", 1.0),
                                            ("candidate", 1.2)):
                receipts.append(pd.make_execution_receipt(
                    experiment_id="exp_test_audit", anchor_id=anchor_id,
                    baseline_id="BASE",
                    candidate_id=record["candidates"][1]["candidate_id"],
                    branch_id=branch_id,
                    initial_state_hash="a" * 64, prefix_hash="b" * 64,
                    source_fingerprint=source_fp,
                    execution_context_hash=CTX,
                    intervention_type="single_step",
                    intervention_horizon=1,
                    execution_started=True, execution_completed=True,
                    episode_finished=True, outcome="win",
                    return_value=return_value))
        return receipts

    def test_base_receipt_missing(self):
        receipts = [r for r in self._receipts_from_helper()
                    if r["branch_id"] != "BASE"]                   # check 5
        problems = pd.audit_execution_receipts(
            [{"pair_id": "p", "experiment_id": "exp_test_audit",
              "anchor_id": f"{self.records[0]['episode_id']}:"
                           f"{self.records[0]['snapshot_id']}",
              "baseline_id": "BASE",
              "candidate_id": self.records[0]["candidates"][1]["candidate_id"],
              "initial_state_hash": "a" * 64, "prefix_hash": "b" * 64,
              "source_fingerprint": SOURCE, "execution_context_hash": CTX,
              "baseline_return": 1.0, "candidate_return": 1.2}], receipts)
        self.assertIn("baseline_execution_receipt_missing",
                      problems["p"])

    def test_candidate_receipt_missing(self):
        receipts = [r for r in self._receipts_from_helper()
                    if r["branch_id"] != "candidate"]             # check 6
        problems = pd.audit_execution_receipts(
            [{"pair_id": "p", "experiment_id": "exp_test_audit",
              "anchor_id": f"{self.records[0]['episode_id']}:"
                           f"{self.records[0]['snapshot_id']}",
              "baseline_id": "BASE",
              "candidate_id": self.records[0]["candidates"][1]["candidate_id"],
              "initial_state_hash": "a" * 64, "prefix_hash": "b" * 64,
              "source_fingerprint": SOURCE, "execution_context_hash": CTX,
              "baseline_return": 1.0, "candidate_return": 1.2}], receipts)
        self.assertIn("candidate_execution_receipt_missing",
                      problems["p"])

    def test_branch_from_different_anchor(self):
        receipts = self._receipts_from_helper()
        for receipt in receipts:                    # check 7
            if receipt["branch_id"] == "candidate":
                receipt["anchor_id"] = "other_anchor"
        problems = pd.audit_execution_receipts(
            [{"pair_id": "p", "experiment_id": "exp_test_audit",
              "anchor_id": f"{self.records[0]['episode_id']}:"
                           f"{self.records[0]['snapshot_id']}",
              "baseline_id": "BASE",
              "candidate_id": self.records[0]["candidates"][1]["candidate_id"],
              "initial_state_hash": "a" * 64, "prefix_hash": "b" * 64,
              "source_fingerprint": SOURCE, "execution_context_hash": CTX,
              "baseline_return": 1.0, "candidate_return": 1.2}], receipts)
        # a receipt bound to another anchor either disappears from the
        # pair's anchor index (missing) or is flagged as cross-anchor —
        # both reject the pair
        self.assertTrue(any("different_anchor" in p
                            or "receipt_missing" in p
                            for p in problems["p"]))

    def test_valid_pair_enters_dataset(self):
        """Check 8: the fully evidenced pair trains."""
        records = _records(8)
        bundle, envelope = synthetic_audit(records)
        self.assertTrue(envelope["audited"])
        self.assertGreater(envelope["pairs_trainable"], 0)
        backend = zt.ZergTrainingBackend()
        model, _ = backend.train(
            "gain", bundle.joint_records, scorer="joint_scorer",
            execution_context_hash=CTX, audit_envelope=envelope,
            capacity={"minimum_leaf_support": 2, "minimum_leaf_groups": 2})
        self.assertEqual(model["gain_audit"]["anchor_manifest_sha256"],
                         envelope["anchor_manifest_sha256"])
        zs.validate_scorer_model(model)


class Test04TrainingEntryEnforcement(unittest.TestCase):
    """Check 9: unaudited data cannot enter train()/train_internal()."""

    def test_unaudited_refused(self):
        records = _records(8)
        backend = zt.ZergTrainingBackend()
        for task in ("gain", "ranking", "correction"):
            with self.assertRaises(ValueError):
                if task == "correction":
                    # correction needs an old model first — build one from
                    # an audited pipeline
                    bundle, envelope = synthetic_audit(records)
                    gain, _ = backend.train(
                        "gain", bundle.joint_records,
                        scorer="joint_scorer",
                        execution_context_hash=CTX,
                        audit_envelope=envelope,
                        capacity={"minimum_leaf_support": 2,
                                  "minimum_leaf_groups": 2})
                    backend.train(
                        task, bundle.joint_records, scorer="joint_scorer",
                        execution_context_hash=CTX, old_model=gain,
                        capacity={"minimum_leaf_support": 2,
                                  "minimum_leaf_groups": 2})
                else:
                    backend.train(
                        task, records, scorer="joint_scorer",
                        execution_context_hash=CTX,
                        capacity={"minimum_leaf_support": 2,
                                  "minimum_leaf_groups": 2})
        # unaudited (require_audit=False) envelope is refused too
        pairs = self._unaudited_pairs(records)
        unaudited = pd.build_gain_dataset(records, pairs,
                                          require_audit=False)
        self.assertFalse(unaudited.audited)
        envelope = unaudited.export_envelope("joint")
        with self.assertRaises(ValueError):
            backend.train(
                "gain", unaudited.joint_records, scorer="joint_scorer",
                execution_context_hash=CTX, audit_envelope=envelope,
                capacity={"minimum_leaf_support": 2,
                          "minimum_leaf_groups": 2})
        # train_internal refuses unaudited as well
        with self.assertRaises(ValueError):
            backend.train_internal(
                "gain", records, scorer="hydralisk_scorer",
                execution_context_hash=CTX,
                capacity={"minimum_leaf_support": 2,
                          "minimum_leaf_groups": 2})

    def _unaudited_pairs(self, records):
        pairs = []
        for index, record in enumerate(records):
            pairs.append({
                "schema": "zerg_counterfactual_pair_v1",
                "experiment_id": "exp_debug", "run_id": record["run_id"],
                "episode_id": record["episode_id"],
                "anchor_id": "a", "pair_id": f"u{index}",
                "baseline_id": "BASE",
                "candidate_id": record["candidates"][1]["candidate_id"],
                "initial_state_hash": "a", "prefix_hash": "b",
                "execution_context_hash": CTX,
                "source_fingerprint": SOURCE,
                "intervention_type": "single_step",
                "intervention_horizon": 1,
                "candidate_return": 1.2, "baseline_return": 1.0,
                "completion_status": "complete", "scope": "joint"})
        return pairs


class Test05ResourceConflictMissing(unittest.TestCase):
    """Check 10: MISSING resource-conflict is never filled with 0."""

    def test_missing_not_zero(self):
        payload = _payload()
        manager = _manager()
        actions, summary = manager.step(payload, 4)
        snapshot = manager._pending_commit[0]
        branches = build_branch_candidates(manager, snapshot, summary)
        branch_scoring_features(branches, snapshot)
        for branch in branches:
            value = branch.candidate_features["candidate_resource_conflict"]
            self.assertTrue(math.isnan(value))          # MISSING, not 0
        # summary-level extractor without peer plans is also MISSING
        plan_like = zc.ZergCandidate(
            candidate_id="solo", candidate_family="skill_internal",
            candidate_skill="focus_fire", controlled_actors=("u0",),
            target=None, phase=None, current_actions={},
            proposed_actions={"u0": 6}, features={},
            resource_claims={}, dependencies=[], atomic=False,
            legal=True, rejection_reason=None, source_plan_id="solo")
        from decision_tree.zerg.features import extract_candidate_features
        features = extract_candidate_features(plan_like, snapshot)
        self.assertTrue(math.isnan(features["candidate_resource_conflict"]))
        # internal layout keeps NaN through the join (never coerced to 0)
        flat = build_internal_model_features(
            {"team": {}, "skill": {}, "units": {"u0": {"health_ratio": 1.0}},
             "unit_types": {"u0": "hydralisk"}},
            features, ["u0"])
        self.assertTrue(math.isnan(flat["candidate.candidate_resource_conflict"]))
        # training selection drops the constant-missing column instead of
        # imputing 0
        rows = [{"group_id": f"g{i}", "decision_key": f"d{i}",
                 "features": flat}
                for i in range(6)]
        active, inactive = select_active_features(rows, list(flat))
        self.assertIn("candidate.candidate_resource_conflict",
                      [row["name"] for row in inactive])
        self.assertNotIn("candidate.candidate_resource_conflict", active)
        manager.commit_final_actions(actions)


if __name__ == "__main__":
    unittest.main()
