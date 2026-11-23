"""V1.3 end-to-end acceptance tests (Phase 6) — fixtures + synthetic data
only, no SC2.

Run from the repository root:
    python -m unittest decision_tree.zerg.tests.test_v13 -v
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
from decision_tree.zerg.tests._helpers import synthetic_audit
from decision_tree.zerg.candidates import branch_scoring_features, \
    build_branch_candidates
from decision_tree.zerg.experiment_manifest import make_anchor_manifest
from decision_tree.zerg.features import DEFAULT_SCHEMA
from decision_tree.zerg.schema import build_internal_model_features, \
    build_model_features
from decision_tree.zerg.selector import apply_complete_plan, \
    select_complete_plan

from decision_tree.zerg.tests.test_minimal import _manager, _zerg_payload
from decision_tree.zerg.tests.test_v11 import _records


def _payload():
    from skills.tactical_v1.tests import fixtures as fx
    allies = [((17., 17.), "hydralisk"), ((18., 17.), "hydralisk"),
              ((17., 18.), "zergling"), ((18., 18.), "zergling")]
    enemies = {"e0": fx.enemy((20., 20.), "marine", health=30.),
               "e1": fx.enemy((22., 20.), "marine", health=12.)}
    return fx.with_questions(fx.scene(allies, enemies))


class TestAInternalPipeline(unittest.TestCase):
    """流程 A: propose → collect → features → shadow score → training data
    → internal imitation training → load → online scoring."""

    def test_full_internal_chain(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = zr.default_config()
            config["mode"] = "shadow"
            config["workspace"] = tmp
            config["source_root"] = str(WORKSPACE)
            config["records_path"] = f"{tmp}/shadow.jsonl"
            config["events_path"] = f"{tmp}/events.jsonl"
            runtime = zr.ZergTreeRuntime(config=config, tactical_config={})
            bundle = factory.build_zerg_skill_manager()
            controller = runtime.wrap(bundle)
            manager = bundle.manager
            # reference (unwrapped) actions for the bit-identical check
            reference = factory.build_zerg_skill_manager().manager
            ref_actions, _ = reference.step(_payload(), 4)
            # 3 episodes of shadow data so the split has enough groups
            for outcome in ("win", "loss", "win"):
                actions, summary = controller.step(_payload(), 4)
                self.assertEqual(actions, ref_actions)   # shadow unchanged
                controller.commit_final_actions(actions)
                episode = controller.last_meta["episode_id"]
                runtime.finalize_episode(outcome=outcome,
                                         episode_return=1.0,
                                         ally_losses=0, enemy_losses=2,
                                         episode_id=episode)
                manager.reset()
            # 5. propose call count unchanged by scoring reads
            calls_before = dict(controller.collector.propose_calls)
            records = zd.read_records(f"{tmp}/shadow.jsonl")
            self.assertTrue(records)
            # 6/7: expand A-class rows from the shadow JSONL
            rows = zd.expand_skill_proposals(records)
            self.assertTrue(rows)
            hydra = zd.export_internal_training_rows(rows,
                                                     unit_type="hydralisk")
            self.assertTrue(hydra)
            # 8: unit features really present in the internal layout
            row = hydra[0]
            self.assertIn("features_internal", row)
            self.assertTrue(any(name.startswith("unit.mean.")
                                for name in row["features_internal"]))
            # 9: train a small hydralisk imitation model
            backend = zt.ZergTrainingBackend()
            model, report = backend.train_internal(
                "imitation", records, scorer="hydralisk_scorer",
                execution_context_hash="1" * 64,
                capacity={"minimum_leaf_support": 2,
                          "minimum_leaf_groups": 2},
                min_train_groups=1)
            self.assertEqual(model["feature_layout"], "internal")
            self.assertTrue(model["model_id"].startswith("mdl_"))
            # 10: load through the unified entry
            loaded = zscore.load_zerg_model(model)
            self.assertIsInstance(loaded, zscore.ScorerModel)
            # 11: online scoring on the SAME candidates (shadow path)
            global_features = records[0]["global_features"]
            cand_record = records[0]["skill_proposals"]["focus_fire"][0]
            cand = zc.ZergCandidate(
                candidate_id=cand_record["candidate_id"],
                candidate_family="skill_internal",
                candidate_skill="focus_fire",
                controlled_actors=tuple(cand_record["controlled_actors"]),
                target=None, phase=None, current_actions={},
                proposed_actions={}, features=dict(cand_record["features"]),
                resource_claims=cand_record["resource_claims"],
                dependencies=cand_record["dependencies"], atomic=False,
                legal=True, rejection_reason=None,
                source_plan_id=cand_record["source_plan_id"],
                rule_score=cand_record.get("rule_score"))
            scores = zscore.score_candidates(global_features, [cand],
                                             model=loaded)
            self.assertEqual(len(scores), 1)
            self.assertIsNotNone(scores[0].model_kind)
            self.assertEqual(calls_before,
                             controller.collector.propose_calls)


class TestBCompleteBranchPipeline(unittest.TestCase):
    """流程 B: Coordinator real branches → CompletePlanCandidate → score →
    select → apply_complete_plan → Validator → manager state → memory."""

    def test_real_branch_selection_and_memory(self):
        bundle = factory.build_zerg_skill_manager()
        manager = bundle.manager
        actions, summary = manager.step(_payload(), 4)
        snapshot = manager._pending_commit[0]
        # 1: real complete branches from the coordinator's own cache
        branches = build_branch_candidates(manager, snapshot, summary)
        self.assertGreaterEqual(len(branches), 2,
                                "fixture must yield BASE + >=1 alternative")
        base = branches[0]
        self.assertEqual(base.candidate_id, "joint_base")
        self.assertTrue(base.plan_objects_available and base.plans)
        for branch in branches:
            self.assertEqual(branch.snapshot_id, snapshot.snapshot_id)
            self.assertTrue(branch.legal)
            # plans/actions consistency: every plan's actions are in the
            # branch action dict
            for plan in branch.plans:
                for actor, action in plan.current_actions.items():
                    self.assertEqual(branch.actions.get(actor), action)
        branch_scoring_features(branches, snapshot)
        # 3: score (no model -> rule fallback keeps BASE)
        tiny = zscore.ScorerModel({
            "schema_version": 1, "kind": zs.MODEL_KIND,
            "scorer": "joint_scorer", "target": "gain",
            "feature_names": scorer_feature_names(),
            "feature_schema": {"name": DEFAULT_SCHEMA.name,
                               "version": DEFAULT_SCHEMA.version,
                               "digest": DEFAULT_SCHEMA.digest()},
            "execution_context_hash": "1" * 64,
            "trees": [{"weight": 1.0,
                       "root": {"value": 9.0, "groups": 6, "samples": 9,
                                "supported": True}}],
            "aggregation": "mean", "base_value": 0.0,
            "missing_imputation": {}, "capacity": zt.DEFAULT_CAPACITY,
            "active_feature_names": scorer_feature_names(),
        })
        state = {"team": {name: 1.0 for name in
                          DEFAULT_SCHEMA.names(layer="team")},
                 "skill": {name: 0.0 for name in
                           DEFAULT_SCHEMA.names(layer="skill")}}
        decision = select_complete_plan(branches, state, model=tiny,
                                        min_gain=0.0)
        # the tiny model scores every branch 9.0 -> no strict improvement ->
        # BASE survives (tie-break to rule rank)
        self.assertFalse(decision["changed"])
        self.assertEqual(decision["selected"].candidate_id, "joint_base")
        # 4/5: force-select the best alternative and apply the replacement
        alternative = branches[1]
        replacement = apply_complete_plan(alternative, manager,
                                          context=manager._pending_commit[4],
                                          memory=manager.memory,
                                          base_plans=list(
                                              manager._pending_commit[1]),
                                          base_actions={f"u{i}": a for i, a in enumerate(manager._cache[1])})
        self.assertTrue(replacement["applied"], replacement)
        self.assertEqual(replacement["validator_problems"], [])
        # 7: manager state update — pending commit and cache carry the new
        # branch objects
        zr.ZergTreeRuntime._update_manager_plan_state(
            manager, replacement["final_plans"], replacement["final_actions"],
            summary)
        pending_plans = manager._pending_commit[1]
        self.assertEqual([str(p.proposal_id) for p in pending_plans],
                         alternative.source_plan_ids)
        self.assertEqual(manager._cache[1],
                         [replacement["final_actions"][f"u{i}"]
                          for i in range(len(replacement["final_actions"]))])
        # 8/9: the ORIGINAL commit path still governs memory
        manager.commit_final_actions(actions)
        self.assertTrue(manager.memory.committed)
        self.assertEqual(
            manager.memory.last_committed_actions,
            {f"u{i}": action for i, action in enumerate(actions)})


def scorer_feature_names():
    from decision_tree.zerg.schema import scorer_feature_names as _s
    return _s()


class TestCCounterfactualPipeline(unittest.TestCase):
    """流程 C: decision JSONL + pair JSONL → join → audit → gain dataset →
    gain training → export → reload → prediction parity."""

    def test_gain_chain(self):
        records = _records(8)
        pairs = []
        for i, record in enumerate(records[:6]):
            candidate = record["candidates"][1]
            pairs.append({
                "schema": "zerg_counterfactual_pair_v1",
                "experiment_id": "expA", "run_id": record["run_id"],
                "episode_id": record["episode_id"],
                "anchor_id": f"{record['episode_id']}:{record['snapshot_id']}",
                "pair_id": f"p{i}", "baseline_id": "BASE",
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
                "scope": "joint"})
        # an incomplete pair (missing return) must be excluded
        pairs.append({**pairs[0], "pair_id": "p_bad",
                      "candidate_return": None,
                      "completion_status": "missing_return"})
        with tempfile.TemporaryDirectory() as tmp:
            records_path = Path(tmp) / "decision_records.jsonl"
            pairs_path = Path(tmp) / "pairs.jsonl"
            for record in records:
                with records_path.open("a") as handle:
                    handle.write(json.dumps(zs._sanitize(record)) + "\n")
            for pair in pairs:
                with pairs_path.open("a") as handle:
                    handle.write(json.dumps(pair) + "\n")
            loaded_records = zd.read_records(str(records_path))
            loaded_pairs = pd.validate_pairs(str(pairs_path))
            joined = pd.join_counterfactual_pairs(loaded_records, loaded_pairs)
            audit = pd.audit_counterfactual_pairs(joined, loaded_records)
            self.assertEqual(audit["pairs_trainable"], 6)
            self.assertIn("p_bad", audit["problems_by_pair"])
            self.assertIn("missing_return_kept_none",
                          audit["problems_by_pair"]["p_bad"])
            # formal build now requires the anchor manifest + receipts
            with self.assertRaises(ValueError):
                pd.build_gain_dataset(loaded_records, loaded_pairs)
            manifest = make_anchor_manifest(
                experiment_id="expA",
                source_fingerprint=loaded_records[0]["source_hash"],
                execution_context_hash=loaded_records[0][
                    "execution_context_hash"],
                anchors={f"{loaded_records[i]['episode_id']}:"
                         f"{loaded_records[i]['snapshot_id']}":
                         {"initial_state_hash": "a" * 64,
                          "prefix_hash": "b" * 64,
                          "baseline_id": "BASE"}
                         for i in range(6)})
            receipts = []
            for i, pair in enumerate(pairs[:6]):
                for branch, field in (("BASE", "baseline_return"),
                                      ("candidate", "candidate_return")):
                    receipts.append(pd.make_execution_receipt(
                        experiment_id="expA",
                        anchor_id=pair["anchor_id"],
                        baseline_id="BASE",
                        candidate_id=pair["candidate_id"],
                        branch_id=branch,
                        initial_state_hash="a" * 64,
                        prefix_hash="b" * 64,
                        source_fingerprint=loaded_records[0]["source_hash"],
                        execution_context_hash=loaded_records[0][
                            "execution_context_hash"],
                        intervention_type="single_step",
                        intervention_horizon=1,
                        execution_started=True,
                        execution_completed=True,
                        episode_finished=True,
                        outcome="win",
                        return_value=pair[field],
                        receipt_id=f"r{i}:{branch}",
                        receipt_source="test_driver"))
            bundle = pd.build_gain_dataset(
                loaded_records, loaded_pairs,
                anchor_manifest=manifest, execution_receipts=receipts)
            self.assertEqual(len(bundle.joint_records), 6)
            self.assertTrue(bundle.audit_report["pairs_trainable"] == 6)
            # 6: train a small gain model
            backend = zt.ZergTrainingBackend()
            model, report = backend.train(
                "gain", bundle.joint_records, scorer="joint_scorer",
                execution_context_hash="1" * 64,
                dataset_hash=bundle.dataset_hash,
                audit_envelope=bundle.export_envelope("joint"),
                capacity={"minimum_leaf_support": 2,
                          "minimum_leaf_groups": 2})
            self.assertEqual(model["dataset_hash"], bundle.dataset_hash)
            # 7: export & reload
            model_path = Path(tmp) / "gain_model.json"
            model_path.write_text(json.dumps(zs._sanitize(model), indent=2))
            loaded = zscore.load_zerg_model(str(model_path))
            # 8: prediction parity between the trained record and reload
            flat = build_model_features(records[0]["global_features"],
                                        records[0]["candidates"][0]["features"])
            trained_est = model and zscore.ScorerModel(model).estimate(flat)
            reloaded_est = loaded.estimate(flat)
            self.assertEqual(
                (trained_est[0]["value"], trained_est[0]["supported"]),
                (reloaded_est[0]["value"], reloaded_est[0]["supported"]))


class TestDDeploymentKinds(unittest.TestCase):
    """Test D: kind dispatch, unified scoring interface, context refusal."""

    def _base_model(self, target="imitation"):
        records = _records(12)
        backend = zt.ZergTrainingBackend()
        if target == "imitation":
            model, _ = backend.train(
                target, records, scorer="hydralisk_scorer",
                execution_context_hash="1" * 64,
                capacity={"minimum_leaf_support": 2,
                          "minimum_leaf_groups": 2})
            return model
        bundle, envelope = synthetic_audit(records)
        model, _ = backend.train(
            target, bundle.joint_records, scorer="hydralisk_scorer",
            execution_context_hash="1" * 64,
            audit_envelope=envelope,
            capacity={"minimum_leaf_support": 2, "minimum_leaf_groups": 2})
        return model

    def test_kind_dispatch_and_uniform_interface(self):
        records = _records(12)
        backend = zt.ZergTrainingBackend()
        imitation = self._base_model("imitation")
        gain = self._base_model("gain")
        bundle, envelope = synthetic_audit(records)
        ranking, _ = backend.train(
            "ranking", bundle.joint_records, scorer="joint_scorer",
            execution_context_hash="1" * 64,
            audit_envelope=envelope,
            capacity={"minimum_leaf_support": 2, "minimum_leaf_groups": 2})
        gain_old = gain
        correction, _ = backend.train(
            "correction", bundle.joint_records, scorer="joint_scorer",
            execution_context_hash="1" * 64, old_model=gain_old,
            audit_envelope=envelope,
            capacity={"minimum_leaf_support": 2, "minimum_leaf_groups": 2})
        # dispatch
        self.assertIsInstance(zscore.load_zerg_model(imitation),
                              zscore.ScorerModel)
        self.assertIsInstance(zscore.load_zerg_model(gain),
                              zscore.ScorerModel)
        self.assertIsInstance(zscore.load_zerg_model(correction),
                              zscore.ScorerModel)
        self.assertIsInstance(zscore.load_zerg_model(ranking),
                              zscore.PairwiseRankModel)
        with self.assertRaises(ValueError):
            zscore.load_zerg_model({"kind": "unknown_kind"})
        # unified scoring fields
        global_features = records[0]["global_features"]
        candidate = records[0]["candidates"][0]
        flat = build_model_features(global_features, candidate["features"])
        candidate_obj = zc.ZergCandidate(
            candidate_id=candidate["candidate_id"],
            candidate_family="joint_plan", candidate_skill="focus_fire",
            controlled_actors=("u0",), target=None, phase=None,
            current_actions={}, proposed_actions={"u0": 6},
            features=dict(candidate["features"]), resource_claims={},
            dependencies=[], atomic=True, legal=True, rejection_reason=None,
            source_plan_id=candidate["candidate_id"], rule_score=1.0)
        for model_record in (imitation, gain, ranking):
            model = zscore.load_zerg_model(model_record)
            if isinstance(model, zscore.PairwiseRankModel):
                rows = zscore.pairwise_uniform_scores(
                    model, {candidate_obj.candidate_id: flat})
            else:
                rows = zscore.uniform_scores(
                    zscore.score_candidates(global_features, [candidate_obj],
                                            model=model),
                    model.record["kind"])
            self.assertEqual(len(rows), 1)
            self.assertIsNotNone(rows[0].model_kind)
            self.assertIsNotNone(rows[0].rank)
            self.assertIn("fallback_reason",
                          [f for f in vars(rows[0])
                           if not f.startswith("_")] or [])
        # correction veto semantics (never a preference)
        old_model = zscore.load_zerg_model(gain_old)
        veto = zscore.correction_veto_scores(
            zscore.load_zerg_model(correction), {"c0": flat})
        self.assertIn("veto", veto["c0"])
        with self.assertRaises(ValueError):
            zscore.correction_veto_scores(old_model, {"c0": flat})

    def test_context_mismatch_refused(self):
        bundle = factory.build_zerg_skill_manager()
        model = self._base_model()
        model["execution_context_hash"] = "9" * 64     # wrong context
        config = zr.default_config()
        config["mode"] = "shadow"
        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "m.json"
            model_path.write_text(json.dumps(zs._sanitize(model)))
            config["workspace"] = tmp
            config["source_root"] = str(WORKSPACE)
            config["records_path"] = f"{tmp}/s.jsonl"
            config["events_path"] = f"{tmp}/e.jsonl"
            config["scorers"]["joint_scorer"] = {
                "enabled": True, "model_path": str(model_path),
                "min_gain": 0.0}
            with self.assertRaises(ValueError):
                zr.ZergTreeRuntime(config=config, tactical_config={})


class TestESafety(unittest.TestCase):
    """Test E: original-policy safety."""

    def test_off_shadow_and_fallback(self):
        payload = _payload()
        plain = factory.build_zerg_skill_manager().manager
        plain_actions, _ = plain.step(copy.deepcopy(payload), 4)
        plain.commit_final_actions(plain_actions)
        plain_memory = plain.memory.last_committed_actions
        # 1: off mode does not wrap
        off = zr.ZergTreeRuntime(config=zr.default_config())
        sentinel = factory.build_zerg_skill_manager()
        # off mode returns the live manager (bundle unwrapped), inert
        self.assertIs(off.wrap(sentinel), sentinel.manager)
        # 2/3: shadow keeps actions AND memory identical
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
                             plain_memory)
        # 4: tree_score without models falls back to BASE
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
            gate = runtime.last_tree_score_gate
            self.assertTrue(gate, "gate must be evaluated")
            self.assertFalse(gate["model_loaded"])
            controller.commit_final_actions(actions)
        # 5/6: failed replacement -> full BASE fallback; memory still the
        # manager's own
        empty_branch = zc.CompletePlanCandidate(
            candidate_id="bad", snapshot_id=1, plans=[], actions={},
            resource_claims={}, dependencies=[], execution_duties={},
            rule_score=None, candidate_features={}, legal=True,
            source_plan_ids=[], plan_objects_available=True)
        base_marker = [{"marker": "base_plans"}]
        result = apply_complete_plan(empty_branch, manager=None,
                                     base_plans=base_marker,
                                     base_actions={"u0": 6})
        self.assertFalse(result["applied"])
        self.assertTrue(result["fallback_to_base"])
        # the returned final state IS the untouched BASE
        self.assertIs(result["final_plans"], base_marker)
        self.assertEqual(result["final_actions"], {"u0": 6})


if __name__ == "__main__":
    unittest.main()
