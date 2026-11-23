"""V1.3 deployment & audit fix acceptance tests (Task 5).

Run from the repository root:
    python -m unittest decision_tree.zerg.tests.test_v14 -v
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
from decision_tree.zerg import dataset as zd
from decision_tree.zerg import factory
from decision_tree.zerg import pair_dataset as pd
from decision_tree.zerg import runtime as zr
from decision_tree.zerg import schema as zs
from decision_tree.zerg import scorer as zscore
from decision_tree.zerg import training as zt
from decision_tree.zerg.candidates import branch_scoring_features, \
    build_branch_candidates
from decision_tree.zerg.experiment_manifest import make_anchor_manifest
from decision_tree.zerg.features import DEFAULT_SCHEMA
from decision_tree.zerg.schema import build_model_features, \
    scorer_feature_names
from decision_tree.zerg.selector import select_complete_plan

from decision_tree.zerg.tests.test_minimal import _manager, _zerg_payload
from decision_tree.zerg.tests.test_v11 import _records


def _payload():
    from skills.tactical_v1.tests import fixtures as fx
    allies = [((17., 17.), "hydralisk"), ((18., 17.), "hydralisk"),
              ((17., 18.), "zergling"), ((18., 18.), "zergling")]
    enemies = {"e0": fx.enemy((20., 20.), "marine", health=30.),
               "e1": fx.enemy((22., 20.), "marine", health=12.)}
    return fx.with_questions(fx.scene(allies, enemies))


class LoaderHarness(unittest.TestCase):
    """Builds a shadow runtime without models to capture its frozen
    execution context, then writes model files bound to that context."""

    def _context(self, tmp):
        config = zr.default_config()
        config["mode"] = "shadow"
        config["workspace"] = tmp
        config["source_root"] = str(WORKSPACE)
        config["records_path"] = f"{tmp}/probe_s.jsonl"
        config["events_path"] = f"{tmp}/probe_e.jsonl"
        # the policy contract captures ENABLED scorer slots — the probe must
        # enable the same slot as the deploying runtime
        config["scorers"]["joint_scorer"] = {
            "enabled": True, "model_path": None, "min_gain": 0.0}
        probe = zr.ZergTreeRuntime(config=copy.deepcopy(config),
                                   tactical_config={})
        return config, probe.execution_context_hash, probe.source_fingerprint

    def _train(self, target, ctx_hash, source_fp, tmp, name="model.json"):
        records = _records(12)
        backend = zt.ZergTrainingBackend()
        kwargs = {}
        if target == "correction":
            gain, _ = backend.train(
                "gain", records, scorer="joint_scorer",
                execution_context_hash=ctx_hash,
                source_fingerprint=source_fp,
                capacity={"minimum_leaf_support": 2,
                          "minimum_leaf_groups": 2})
            kwargs["old_model"] = gain
        if target == "ranking":
            model, _ = backend.train(
                "ranking", records, scorer="joint_scorer",
                execution_context_hash=ctx_hash,
                source_fingerprint=source_fp,
                capacity={"minimum_leaf_support": 2,
                          "minimum_leaf_groups": 2})
        else:
            model, _ = backend.train(
                target, records, scorer="joint_scorer",
                execution_context_hash=ctx_hash,
                source_fingerprint=source_fp,
                capacity={"minimum_leaf_support": 2,
                          "minimum_leaf_groups": 2},
                **kwargs)
        path = Path(tmp) / name
        path.write_text(json.dumps(zs._sanitize(model), indent=2))
        return path

    def _runtime_with(self, tmp, model_path):
        config, _, _ = self._context(tmp)
        config["records_path"] = f"{tmp}/s.jsonl"
        config["events_path"] = f"{tmp}/e.jsonl"
        config["scorers"]["joint_scorer"] = {
            "enabled": True, "model_path": str(model_path), "min_gain": 0.0}
        return zr.ZergTreeRuntime(config=copy.deepcopy(config),
                                  tactical_config={})


class Test01ModelLoading(LoaderHarness):
    """Checks 1/2/3: standard loads, ranking loads as PairwiseRankModel,
    correction refused in a ranking slot; kind/target mismatches refused."""

    def test_standard_and_ranking_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            config, ctx_hash, source_fp = self._context(tmp)
            standard = self._train("gain", ctx_hash, source_fp, tmp,
                                   "gain.json")
            runtime = self._runtime_with(tmp, standard)
            model = runtime.scorer_model("joint_scorer")
            self.assertIsInstance(model, zscore.ScorerModel)
            # ranking record loads as PairwiseRankModel
            ranking = self._train("ranking", ctx_hash, source_fp, tmp,
                                  "ranking.json")
            runtime2 = self._runtime_with(tmp, ranking)
            pair_model = runtime2.scorer_model("joint_scorer")
            self.assertIsInstance(pair_model, zscore.PairwiseRankModel)

    def test_correction_and_kind_mismatches_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            config, ctx_hash, source_fp = self._context(tmp)
            correction = self._train("correction", ctx_hash, source_fp, tmp,
                                     "correction.json")
            with self.assertRaises(ValueError):     # check 3
                self._runtime_with(tmp, correction)
            # scorer-kind declaring target=ranking is refused
            records = _records(12)
            backend = zt.ZergTrainingBackend()
            imitation, _ = backend.train(
                "imitation", records, scorer="joint_scorer",
                execution_context_hash=ctx_hash,
                source_fingerprint=source_fp,
                capacity={"minimum_leaf_support": 2,
                          "minimum_leaf_groups": 2})
            bad = copy.deepcopy(imitation)
            bad["target"] = "ranking"
            bad_path = Path(tmp) / "bad.json"
            bad_path.write_text(json.dumps(zs._sanitize(bad)))
            with self.assertRaises(ValueError):
                self._runtime_with(tmp, bad_path)
            # ranking-kind with target!=ranking is refused
            ranking = self._train("ranking", ctx_hash, source_fp, tmp,
                                  "ranking.json")
            bad2 = json.loads(ranking.read_text())
            bad2["target"] = "gain"
            bad2_path = Path(tmp) / "bad2.json"
            bad2_path.write_text(json.dumps(zs._sanitize(bad2)))
            with self.assertRaises(ValueError):
                self._runtime_with(tmp, bad2_path)

    def test_internal_ranking_routing_refused(self):
        """Check 4: a PairwiseRankModel routed to a unit-type slot falls
        back with an explicit reason, never a bogus score."""
        with tempfile.TemporaryDirectory() as tmp:
            config, ctx_hash, source_fp = self._context(tmp)
            ranking = self._train("ranking", ctx_hash, source_fp, tmp,
                                  "ranking.json")
            model = zscore.load_zerg_model(json.loads(ranking.read_text()))
            state = {"team": {n: 1.0 for n in
                              DEFAULT_SCHEMA.names(layer="team")},
                     "skill": {n: 0.0 for n in
                               DEFAULT_SCHEMA.names(layer="skill")},
                     "unit_types": {"u0": "hydralisk"}}
            candidate = zc.ZergCandidate(
                candidate_id="h1", candidate_family="skill_internal",
                candidate_skill="focus_fire", controlled_actors=("u0",),
                target=None, phase=None, current_actions={},
                proposed_actions={"u0": 6}, features={},
                resource_claims={}, dependencies=[], atomic=False,
                legal=True, rejection_reason=None, source_plan_id="h1",
                rule_score=2.0)
            scores, counts = zscore.score_routed(
                state, [candidate],
                {"hydralisk_scorer": model, "joint_scorer": None})
            self.assertEqual(counts["hydralisk_scorer"], 1)
            self.assertEqual(scores[0].reason,
                             "internal_ranking_online_unsupported_"
                             "rule_fallback")
            self.assertFalse(scores[0].supported)
            self.assertEqual(scores[0].rule_score, 2.0)   # rule fallback kept

    def test_joint_ranking_tournament_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            config, ctx_hash, source_fp = self._context(tmp)
            ranking = self._train("ranking", ctx_hash, source_fp, tmp,
                                  "ranking.json")
            model = zscore.load_zerg_model(json.loads(ranking.read_text()))
            global_features = _records(1)[0]["global_features"]
            cands = []
            for index, candidate in enumerate(
                    _records(1)[0]["candidates"][:3]):
                cands.append(zc.ZergCandidate(
                    candidate_id=candidate["candidate_id"],
                    candidate_family="joint_plan",
                    candidate_skill="focus_fire",
                    controlled_actors=("u0",), target=None, phase=None,
                    current_actions={}, proposed_actions={"u0": 6},
                    features=dict(candidate["features"]),
                    resource_claims={}, dependencies=[], atomic=True,
                    legal=True, rejection_reason=None,
                    source_plan_id=candidate["candidate_id"],
                    rule_score=float(index)))
            state = {"team": {n: 1.0 for n in
                              DEFAULT_SCHEMA.names(layer="team")},
                     "skill": {n: 0.0 for n in
                               DEFAULT_SCHEMA.names(layer="skill")}}
            scores, counts = zscore.score_routed(
                state, cands, {"joint_scorer": model})
            self.assertEqual(counts["joint_scorer"], 3)
            for row in scores:
                self.assertEqual(row.model_kind,
                                 "zerg_pairwise_ranking_model_v1")
            ranks = sorted(row.rank for row in scores)
            self.assertEqual(ranks, [0, 1, 2])      # tie-break consistent


class Test02PairAudit(unittest.TestCase):
    """Checks 5/6/7: source fingerprint mismatch, missing context, and
    anchor-evidence mismatch all reject the pair."""

    def _base_setup(self):
        records = _records(4)
        record = records[0]
        candidate = record["candidates"][1]
        pair = {
            "schema": "zerg_counterfactual_pair_v1",
            "experiment_id": "exp", "run_id": record["run_id"],
            "episode_id": record["episode_id"],
            "anchor_id": f"{record['episode_id']}:{record['snapshot_id']}",
            "pair_id": "p0", "baseline_id": "BASE",
            "candidate_id": candidate["candidate_id"],
            "decision_key": f"{record['run_id']}:{record['episode_id']}:"
                            f"{record['snapshot_id']}:{record['game_loop']}",
            "initial_state_hash": "a" * 64, "prefix_hash": "b" * 64,
            "execution_context_hash": record["execution_context_hash"],
            "source_fingerprint": record["source_hash"],
            "intervention_type": "single_step", "intervention_horizon": 1,
            "candidate_return": candidate["candidate_return"],
            "baseline_return": record["baseline_return"],
            "outcome": "win", "completion_status": "complete",
            "scope": "joint"}
        return records, record, pair

    def test_source_fingerprint_mismatch_rejected(self):
        records, record, pair = self._base_setup()
        pair["source_fingerprint"] = "f" * 64          # != record's
        joined = pd.join_counterfactual_pairs(records, [pair])
        audit = pd.audit_counterfactual_pairs(joined, records)
        self.assertEqual(audit["pairs_trainable"], 0)
        self.assertIn("source_fingerprint_mismatch",
                      audit["problems_by_pair"]["p0"])

    def test_missing_context_and_source_rejected(self):
        records, record, pair = self._base_setup()
        broken_record = copy.deepcopy(record)
        broken_record["execution_context_hash"] = ""   # missing on record
        joined = pd.join_counterfactual_pairs([broken_record], [pair])
        audit = pd.audit_counterfactual_pairs(joined, [broken_record])
        self.assertIn("decision_record_missing_execution_context_hash",
                      audit["problems_by_pair"]["p0"])
        # pair without its own source fingerprint is also rejected
        pair2 = dict(pair)
        pair2.pop("source_fingerprint")
        joined2 = pd.join_counterfactual_pairs(records, [pair2])
        audit2 = pd.audit_counterfactual_pairs(joined2, records)
        self.assertIn("pair_missing_source_fingerprint",
                      audit2["problems_by_pair"]["p0"])

    def test_anchor_manifest_evidence_enforced(self):
        records, record, pair = self._base_setup()
        manifest = make_anchor_manifest(
            experiment_id="exp",
            source_fingerprint=record["source_hash"],
            execution_context_hash=record["execution_context_hash"],
            anchors={pair["anchor_id"]: {
                "initial_state_hash": "a" * 64,
                "prefix_hash": "b" * 64,
                "baseline_id": "BASE"}})
        joined = pd.join_counterfactual_pairs(records, [pair])
        audit = pd.audit_counterfactual_pairs(joined, records,
                                              anchor_manifest=manifest)
        self.assertEqual(audit["pairs_trainable"], 1)
        # tampered start-state evidence vs the frozen manifest -> rejected
        tampered = dict(pair, initial_state_hash="c" * 64)
        joined2 = pd.join_counterfactual_pairs(records, [tampered])
        audit2 = pd.audit_counterfactual_pairs(joined2, records,
                                               anchor_manifest=manifest)
        self.assertEqual(audit2["pairs_trainable"], 0)
        problems = audit2["problems_by_pair"]["p0"]
        self.assertTrue(any("initial_state_hash" in p for p in problems))
        # wrong baseline_id vs manifest -> rejected (check: BASE is the arm)
        wrong_base = dict(pair, baseline_id="joint_alt_1")
        joined3 = pd.join_counterfactual_pairs(records, [wrong_base])
        audit3 = pd.audit_counterfactual_pairs(joined3, records,
                                               anchor_manifest=manifest)
        self.assertEqual(audit3["pairs_trainable"], 0)


class Test03BranchFeatures(unittest.TestCase):
    """Check 8: B-class action-change features come from the snapshot's
    previous actions; unknown estimators stay MISSING."""

    def test_change_count_and_missing_semantics(self):
        payload = _payload()
        manager = _manager()
        actions, summary = manager.step(payload, 4)
        snapshot = manager._pending_commit[0]
        branches = build_branch_candidates(manager, snapshot, summary)
        # simulate a previous step where every unit executed action 1
        snapshot_prev = copy.deepcopy(snapshot)
        object.__setattr__(snapshot_prev, "previous_action_ids",
                           {f"u{i}": 1 for i in range(len(actions))})
        branch_scoring_features(branches, snapshot_prev)
        base = branches[0]
        expected_changes = sum(1 for actor, action in base.actions.items()
                               if 1 != action)
        self.assertEqual(base.candidate_features["candidate_action_change_count"],
                         float(expected_changes))
        # with NO previous actions recorded, every action counts as changed
        branch_scoring_features(branches, snapshot)
        self.assertEqual(
            branches[0].candidate_features["candidate_action_change_count"],
            float(len(branches[0].actions)))
        # unknown estimators stay MISSING (never 0)
        for branch in branches:
            features = branch.candidate_features
            self.assertTrue(math.isnan(features["candidate_support_gain"]))
        # declared damage/loss sums exist on the branch features
        self.assertIsInstance(
            branches[0].candidate_features["candidate_predicted_damage"],
            float)
        manager.commit_final_actions(actions)


class Test04Safety(unittest.TestCase):
    """Checks 9/10: shadow bit-identical; tree_score keeps full-plan
    validation and BASE fallback."""

    def test_shadow_unchanged(self):
        payload = _payload()
        plain = factory.build_zerg_skill_manager().manager
        plain_actions, _ = plain.step(copy.deepcopy(payload), 4)
        plain.commit_final_actions(plain_actions)
        with tempfile.TemporaryDirectory() as tmp:
            config = zr.default_config()
            config["mode"] = "shadow"
            config["workspace"] = tmp
            config["source_root"] = str(WORKSPACE)
            config["records_path"] = f"{tmp}/s.jsonl"
            config["events_path"] = f"{tmp}/e.jsonl"
            runtime = zr.ZergTreeRuntime(config=config, tactical_config={})
            bundle = factory.build_zerg_skill_manager()
            controller = runtime.wrap(bundle)
            actions, _ = controller.step(copy.deepcopy(payload), 4)
            self.assertEqual(actions, plain_actions)
            controller.commit_final_actions(actions)
            self.assertEqual(bundle.manager.memory.last_committed_actions,
                             plain.memory.last_committed_actions)

    def test_tree_score_gate_and_base_fallback(self):
        payload = _payload()
        plain_actions, _ = _manager().step(copy.deepcopy(payload), 4)
        with tempfile.TemporaryDirectory() as tmp:
            config = zr.default_config()
            config["mode"] = "tree_score"
            config["workspace"] = tmp
            config["source_root"] = str(WORKSPACE)
            config["records_path"] = f"{tmp}/s.jsonl"
            config["events_path"] = f"{tmp}/e.jsonl"
            runtime = zr.ZergTreeRuntime(config=config, tactical_config={})
            bundle = factory.build_zerg_skill_manager()
            controller = runtime.wrap(bundle)
            actions, summary = controller.step(copy.deepcopy(payload), 4)
            self.assertEqual(actions, plain_actions)
            self.assertIsNone(runtime.last_error)
            self.assertTrue(runtime.last_tree_score_gate)
            self.assertFalse(runtime.last_tree_score_gate["model_loaded"])
            controller.commit_final_actions(actions)
        # replacement refusal falls back to the untouched BASE
        broken = zc.CompletePlanCandidate(
            candidate_id="bad", snapshot_id=1, plans=[], actions={},
            resource_claims={}, dependencies=[], execution_duties={},
            rule_score=None, candidate_features={}, legal=True,
            source_plan_ids=[], plan_objects_available=True)
        marker = [{"base": True}]
        from decision_tree.zerg.selector import apply_complete_plan
        result = apply_complete_plan(broken, manager=None,
                                     base_plans=marker,
                                     base_actions={"u0": 6})
        self.assertFalse(result["applied"])
        self.assertTrue(result["fallback_to_base"])
        self.assertIs(result["final_plans"], marker)


if __name__ == "__main__":
    unittest.main()
