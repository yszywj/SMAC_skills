"""V1.1 acceptance tests (Task 13) — targeted, offline, no SC2.

Run from the repository root:
    python -m unittest decision_tree.zerg.tests.test_v11 -v
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
from decision_tree.zerg import objectives as zo
from decision_tree.zerg import runtime as zr
from decision_tree.zerg import schema as zs
from decision_tree.zerg import scorer as zscore
from decision_tree.zerg import training as zt
from decision_tree.zerg.tests._helpers import synthetic_audit
from decision_tree.zerg.features import DEFAULT_SCHEMA
from decision_tree.zerg.schema import FeatureLayerMissing, \
    build_model_features, scorer_feature_names
from decision_tree.zerg.selector import apply_selected_plan

from decision_tree.zerg.tests.test_minimal import _zerg_payload, _manager


def _records(scenes=12, run_id="run_a"):
    """Synthetic decision records with paired returns (pipeline test only)."""
    records = []
    rng = 0

    def pseudo():
        nonlocal rng
        rng = (rng * 1103515245 + 12345) % (1 << 31)
        return rng / (1 << 31)

    for scene in range(scenes):
        candidates = []
        gains = {}
        base_return = 1.0
        for index in range(3):
            cid = f"joint_cand_{index}"
            gain = round(pseudo() * 0.4 - 0.2, 4)
            gains[cid] = gain
            features = {name: round(pseudo(), 3)
                        for name in DEFAULT_SCHEMA.names(layer="candidate")}
            candidates.append({
                "candidate_id": cid, "candidate_family": "joint_plan",
                "candidate_skill": "focus_fire",
                "controlled_actors": ["u0"], "target": "e0",
                "phase": "engage", "current_actions": {},
                "proposed_actions": {"u0": 6}, "features": features,
                "resource_claims": {}, "dependencies": [],
                "atomic": True, "legal": True, "rejection_reason": None,
                "source_plan_id": cid,
                "rule_score": round(pseudo(), 3),
                "candidate_return": base_return + gain,
            })
        teacher = max(gains, key=lambda cid: gains[cid])
        records.append({
            "schema": zs.DECISION_RECORD_SCHEMA,
            "run_id": run_id,
            "episode_id": f"ep{scene:03d}",
            "scene_config_id": f"cfg{scene % 2}",
            "seed": 5100000 + scene,
            "map_id": "10gen_terran",
            "game_loop": 100 + scene, "snapshot_id": scene + 1,
            "source_hash": "0" * 64, "execution_context_hash": "1" * 64,
            "feature_schema_version": DEFAULT_SCHEMA.version,
            "global_features": {
                "team": {name: 1.0 for name in
                         DEFAULT_SCHEMA.names(layer="team")},
                "skill": {name: 0.0 for name in
                          DEFAULT_SCHEMA.names(layer="skill")},
                "units": {}, "unit_types": {}},
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


class Test01FeatureConsistency(unittest.TestCase):
    """Task 13.1: offline rows and online scoring produce the same vector."""

    def test_train_inference_identical_vectors(self):
        payload = _zerg_payload()
        manager = _manager()
        manager.step(copy.deepcopy(payload), 4)
        snapshot = manager._pending_commit[0]
        plans = manager.skills["focus_fire"].propose(
            manager._pending_commit[4], manager.memory,
            manager.telemetry, None)
        internal = zc.skill_internal_candidates(plans, snapshot,
                                                context=manager._pending_commit[4])
        extractor = zr.StateFeatureExtractor(DEFAULT_SCHEMA)
        state = extractor.extract_state(snapshot)
        # online path (scorer)
        online = build_model_features(state, internal[0].features,
                                      DEFAULT_SCHEMA)
        # offline path (record -> candidate_rows)
        record = zd.build_decision_record(
            run_id="run_t", episode_id="ep_t", scene_config_id="cfg_t",
            seed=1, map_id="10gen_terran", snapshot=snapshot,
            candidates=internal, global_features=state,
            teacher_choice=internal[0].candidate_id)
        offline_row = zd.candidate_rows([record])[0]
        self.assertEqual(list(offline_row["features"].keys()),
                         list(online.keys()))
        for key in online:
            a, b = online[key], offline_row["features"][key]
            if isinstance(a, float) and math.isnan(a):
                self.assertTrue(math.isnan(b), key)
            else:
                self.assertEqual(a, b, key)
        # model declaration must equal the join order exactly
        self.assertEqual(scorer_feature_names(), list(online.keys()))
        manager.commit_final_actions(manager._cache[1])

    def test_missing_semantics(self):
        full_team = {name: 1.0 for name in
                     DEFAULT_SCHEMA.names(layer="team")}
        candidate = {name: 0.5 for name in
                     DEFAULT_SCHEMA.names(layer="candidate")}
        # single missing feature -> NaN (imputable)
        partial = dict(full_team)
        del partial["ally_alive_count"]
        flat = build_model_features({"team": partial, "skill": {}},
                                    candidate)
        self.assertTrue(math.isnan(flat["team.ally_alive_count"]))
        self.assertEqual(flat["team.enemy_visible_count"], 1.0)  # real 1
        # whole layer missing -> refuse
        with self.assertRaises(FeatureLayerMissing):
            build_model_features({"team": partial}, candidate)
        # model with a missing layer at scoring -> rule fallback with reason
        model = zscore.ScorerModel(self._tiny_model())
        cand = zc.ZergCandidate(
            candidate_id="x", candidate_family="skill_internal",
            candidate_skill="focus_fire", controlled_actors=("u0",),
            target=None, phase=None, current_actions={},
            proposed_actions={"u0": 6}, features=candidate,
            resource_claims={}, dependencies=[], atomic=False,
            legal=True, rejection_reason=None, source_plan_id="x",
            rule_score=1.0)
        rows = zscore.score_candidates({}, [cand], model=model)
        self.assertEqual(rows[0].reason, "layer_missing_team_rule_fallback")
        self.assertFalse(rows[0].supported)

    def test_model_feature_declaration_strict(self):
        model = self._tiny_model()
        zs.validate_scorer_model(model)          # exact order passes
        bad = copy.deepcopy(model)
        bad["feature_names"] = list(reversed(bad["feature_names"]))
        with self.assertRaises(ValueError):
            zs.validate_scorer_model(bad)

    @staticmethod
    def _tiny_model():
        names = scorer_feature_names()
        leaf = {"value": 0.5, "groups": 6, "samples": 9, "supported": True}
        return {
            "schema_version": 1, "kind": zs.MODEL_KIND,
            "scorer": "hydralisk_scorer", "target": "imitation",
            "feature_names": names,
            "feature_schema": {"name": DEFAULT_SCHEMA.name,
                               "version": DEFAULT_SCHEMA.version,
                               "digest": DEFAULT_SCHEMA.digest()},
            "execution_context_hash": "1" * 64,
            "trees": [{"weight": 1.0, "root": leaf}],
            "aggregation": "mean", "base_value": 0.0,
            "missing_imputation": {},
            "capacity": zt.DEFAULT_CAPACITY,
        }


class Test02IdentityAndSplits(unittest.TestCase):
    """Task 13.2/13.3: run_id keys and episode-disjoint splits."""

    def test_run_id_no_key_collision(self):
        record_a = _records(1, run_id="run_a")[0]
        record_b = copy.deepcopy(record_a)
        record_b["run_id"] = "run_b"
        self.assertNotEqual(zd.record_key(record_a), zd.record_key(record_b))
        with tempfile.TemporaryDirectory() as tmp:
            writer = zd.DecisionRecordWriter(f"{tmp}/r.jsonl")
            self.assertTrue(writer.write(record_a))
            self.assertTrue(writer.write(record_b))
            same = zd.DecisionRecordWriter(f"{tmp}/r.jsonl")
            self.assertFalse(same.write(copy.deepcopy(record_a)))  # dedup
            self.assertEqual(len(zd.read_records(f"{tmp}/r.jsonl")), 2)

    def test_episode_never_crosses_splits(self):
        records = _records(24)
        train, validation = zd.split_records(records, validation_fraction=.25,
                                             split_seed=3)
        train_groups = {zd.episode_group(r) for r in train}
        val_groups = {zd.episode_group(r) for r in validation}
        self.assertFalse(train_groups & val_groups)
        self.assertEqual(len(train_groups) + len(val_groups), 24)
        with tempfile.TemporaryDirectory() as tmp:
            manifest = zd.write_split_manifest(
                f"{tmp}/split.json", train, validation, split_seed=3)
            self.assertEqual(sorted(manifest["train_groups"]),
                             sorted(train_groups))
            overlapping_train = train + validation[:1]
            with self.assertRaises(ValueError):
                zd.write_split_manifest(f"{tmp}/bad.json", overlapping_train,
                                        validation)
        # scene-config holdout split also deterministic
        cfg_train, cfg_val = zd.split_records(records, group_by="scene_config",
                                              min_train_groups=1)
        self.assertFalse({zd.scene_config_group(r) for r in cfg_train}
                         & {zd.scene_config_group(r) for r in cfg_val})


class Test03CounterfactualPairs(unittest.TestCase):
    """Task 13.4: pairing arithmetic, audits and trainability."""

    def test_pair_gain_and_audit(self):
        pair = zd.build_pair_record(
            experiment_id="exp1", run_id="run_a", episode_id="ep1",
            scene_config_id="cfg1", anchor_id="ep1:3", pair_id="p1",
            baseline_id="BASE", candidate_id="c1",
            initial_state_hash="a" * 64, prefix_hash="b" * 64,
            execution_context_hash="c" * 64,
            intervention_type="single_step", intervention_horizon=1,
            candidate_return=1.4, baseline_return=1.0,
            completion_status="complete")
        self.assertAlmostEqual(pair["gain"], 0.4)
        self.assertEqual(zs.audit_pair(pair), [])
        self.assertTrue(zs.pair_is_trainable(pair))
        # fabricated gain rejected
        bad = dict(pair, gain=0.9)
        with self.assertRaises(ValueError):
            zs.validate_pair_record(bad)
        # context mismatch -> incomplete
        mismatched = dict(pair, completion_status="context_mismatch")
        self.assertTrue(zs.audit_pair(mismatched))
        # missing returns stay None, incomplete, never fabricated
        incomplete = zd.build_pair_record(
            experiment_id="exp1", run_id="run_a", episode_id="ep1",
            scene_config_id="cfg1", anchor_id="ep1:3", pair_id="p2",
            baseline_id="BASE", candidate_id="c2",
            initial_state_hash="a" * 64, prefix_hash="b" * 64,
            execution_context_hash="c" * 64,
            intervention_type="multi_step_continuous",
            intervention_horizon=4, completion_status="missing_return")
        self.assertIsNone(incomplete["gain"])
        self.assertFalse(zs.pair_is_trainable(incomplete))
        with tempfile.TemporaryDirectory() as tmp:
            writer = zd.PairWriter(f"{tmp}/pairs.jsonl")
            writer.write(pair)
            writer.write(incomplete)
            usable, incomplete_rows = zd.trainable_gain_pairs(f"{tmp}/pairs.jsonl")
            self.assertEqual(len(usable), 1)
            self.assertEqual(len(incomplete_rows), 1)
            self.assertAlmostEqual(usable[0]["gain"], 0.4)


class Test04FactorySourceBinding(unittest.TestCase):
    """Task 13.5: the live manager really comes from the resolved tree with
    the manifest fingerprint."""

    def test_factory_binds_resolved_source(self):
        bundle = factory.build_zerg_skill_manager()
        self.assertIn("decision_tree/zerg/resolved/tactical_v1",
                      bundle.module_path)
        from decision_tree.zerg.source import load_manifest
        manifest = load_manifest(str(
            Path(bundle.module_path).parent))
        self.assertEqual(bundle.source_fingerprint, manifest["fingerprint"])
        import skills.tactical_v1.skill_manager as live
        self.assertNotEqual(bundle.module_path, live.__file__)
        # Task 3.2: mismatched source fingerprint refuses model load
        with self.assertRaises(ValueError):
            zs_es = __import__("decision_tree.zerg.fingerprints",
                               fromlist=["assert_model_source"]) \
                .assert_model_source({"source_fingerprint": "deadbeef"},
                                     bundle.source_fingerprint)


class Test05ShadowAndProposals(unittest.TestCase):
    """Task 13.6/13.7/13.10: propose called once, shadow unchanged, memory
    commit controlled by the original manager."""

    def _runtime(self, tmp):
        config = zr.default_config()
        config["mode"] = "shadow"
        config["workspace"] = tmp
        config["source_root"] = str(WORKSPACE)
        config["records_path"] = f"{tmp}/shadow.jsonl"
        config["events_path"] = f"{tmp}/events.jsonl"
        return zr.ZergTreeRuntime(config=config, tactical_config={})

    def test_shadow_actions_and_propose_call_count(self):
        payload = _zerg_payload()
        plain_manager = _manager()
        plain_actions, _ = plain_manager.step(copy.deepcopy(payload), 4)
        plain_manager.commit_final_actions(plain_actions)
        plain_last = plain_manager.memory.last_committed_actions

        with tempfile.TemporaryDirectory() as tmp:
            runtime = self._runtime(tmp)
            bundle = factory.build_zerg_skill_manager()
            controller = runtime.wrap(bundle.manager)
            actions, summary = controller.step(copy.deepcopy(payload), 4)
            self.assertEqual(actions, plain_actions)          # 13.7
            controller.commit_final_actions(actions)
            self.assertEqual(bundle.manager.memory.last_committed_actions,
                             plain_last)                      # 13.10
            calls = controller.collector.propose_calls
            self.assertGreater(sum(calls.values()), 0)
            snapshot = bundle.manager._cache[4]
            # 13.6: read-only retrieval does not re-call propose
            proposals = controller.collector.candidates_for(
                snapshot.snapshot_id, snapshot.game_loop)
            self.assertGreater(sum(len(v) for v in proposals.values()), 0)
            self.assertEqual(controller.collector.propose_calls, calls)
            # A/B protocol: family preserved, decision key shared
            for skill, candidates in proposals.items():
                for candidate in candidates:
                    self.assertEqual(candidate.candidate_family,
                                     "skill_internal")
            n = runtime.finalize_episode(outcome="win", episode_return=1.0,
                                         ally_losses=0, enemy_losses=2)
            self.assertEqual(n, 1)
            record = zd.read_records(f"{tmp}/shadow.jsonl")[0]
            self.assertIn("skill_proposals", record)
            key = zd.record_key(record)
            self.assertEqual(key[0], record["run_id"])

    def test_events_logged(self):
        payload = _zerg_payload()
        with tempfile.TemporaryDirectory() as tmp:
            runtime = self._runtime(tmp)
            bundle = factory.build_zerg_skill_manager()
            controller = runtime.wrap(bundle.manager)
            actions, _ = controller.step(copy.deepcopy(payload), 4)
            controller.commit_final_actions(actions)
            events = [json.loads(line) for line
                      in Path(tmp, "events.jsonl").read_text().splitlines()]
            kinds = [event["event"] for event in events]
            for expected in ("proposed", "submitted", "committed"):
                self.assertIn(expected, kinds)
            committed = next(e for e in events if e["event"] == "committed")
            self.assertIn("NOT an SC2 engine acknowledgement",
                          committed.get("note", ""))


class Test06TreeScoreFallbackAndReplacement(unittest.TestCase):
    """Task 13.8/13.9: failed replacement -> full BASE fallback; valid
    replacement passes the Validator."""

    def test_no_plan_object_is_refused(self):
        candidate = zc.ZergCandidate(
            candidate_id="alt", candidate_family="joint_plan",
            candidate_skill="focus_fire", controlled_actors=("u0",),
            target=None, phase=None, current_actions={},
            proposed_actions={"u0": 1}, features={},
            resource_claims={}, dependencies=[], atomic=True,
            legal=True, rejection_reason=None, source_plan_id="alt",
            plan=None)
        result = apply_selected_plan(candidate, manager=None,
                                     base_plans=[object()],
                                     base_actions={"u0": 6})
        self.assertFalse(result["applied"])
        self.assertTrue(result["fallback_to_base"])
        self.assertEqual(result["reason"], "no_source_plan_object_refused")

    def test_full_replacement_passes_validator_and_fallback(self):
        payload = _zerg_payload()
        bundle = factory.build_zerg_skill_manager()
        manager = bundle.manager
        actions, summary = manager.step(copy.deepcopy(payload), 4)
        pending = manager._pending_commit
        snapshot, base_plans, _, _, context = pending
        base_actions = {f"u{i}": a for i, a in enumerate(manager._cache[1])}
        # re-select one of the manager's OWN chosen plans as the "model
        # choice": replacement should validate cleanly (same plan set minus
        # overlaps plus that plan).
        chosen = next(p for p in base_plans if p.controlled_actors)
        selected = zc.ZergCandidate(
            candidate_id=f"re-{chosen.proposal_id}",
            candidate_family="joint_plan",
            candidate_skill=chosen.primary_skill,
            controlled_actors=tuple(chosen.controlled_actors),
            target=chosen.goal, phase=chosen.commitment.phase,
            current_actions={}, proposed_actions=dict(chosen.current_actions),
            features={}, resource_claims={}, dependencies=[],
            atomic=chosen.atomic, legal=True, rejection_reason=None,
            source_plan_id=chosen.proposal_id, plan=chosen)
        result = apply_selected_plan(selected, manager, context=context,
                                     memory=manager.memory,
                                     base_plans=list(base_plans),
                                     base_actions=base_actions)
        self.assertTrue(result["applied"], result)
        self.assertEqual(result["validator_problems"], [])
        self.assertEqual(result["commit_status"], "pending_manager_commit")
        # memory commit still handled by the original manager logic
        manager.commit_final_actions(actions)
        self.assertTrue(manager.memory.committed)

    def test_tree_score_without_models_never_changes_actions(self):
        payload = _zerg_payload()
        plain_actions, _ = _manager().step(copy.deepcopy(payload), 4)
        config = zr.default_config()
        config["mode"] = "tree_score"
        with tempfile.TemporaryDirectory() as tmp:
            config["workspace"] = tmp
            config["source_root"] = str(WORKSPACE)
            config["records_path"] = f"{tmp}/t.jsonl"
            config["events_path"] = f"{tmp}/e.jsonl"
            runtime = zr.ZergTreeRuntime(config=config, tactical_config={})
            bundle = factory.build_zerg_skill_manager()
            controller = runtime.wrap(bundle.manager)
            actions, _ = controller.step(copy.deepcopy(payload), 4)
            self.assertEqual(actions, plain_actions)   # 13.8
            controller.commit_final_actions(actions)


class Test07RankingAndCorrectionLabels(unittest.TestCase):
    """Task 13.11: pairwise ranking is real; correction replays the old
    model's own winner."""

    def _trained_models(self, records):
        backend = zt.ZergTrainingBackend()
        imitation, _ = backend.train(
            "imitation", records, scorer="hydralisk_scorer",
            execution_context_hash="1" * 64,
            capacity={"minimum_leaf_support": 2, "minimum_leaf_groups": 2})
        bundle, envelope = synthetic_audit(records)
        gain_model, _ = backend.train(
            "gain", bundle.joint_records, scorer="joint_scorer",
            execution_context_hash="1" * 64,
            audit_envelope=envelope,
            capacity={"minimum_leaf_support": 2, "minimum_leaf_groups": 2})
        return backend, imitation, gain_model, records, bundle, envelope

    def test_ranking_pairwise_model(self):
        records = _records(16)
        backend, _, _, _, bundle, envelope = self._trained_models(records)
        ranking, report = backend.train(
            "ranking", bundle.joint_records, scorer="joint_scorer",
            execution_context_hash="1" * 64,
            audit_envelope=envelope,
            capacity={"minimum_leaf_support": 2, "minimum_leaf_groups": 2})
        self.assertEqual(ranking["kind"], zs.RANKING_MODEL_KIND)
        prov = report["pair_provenance"]["training"]
        self.assertGreater(prov["pairs_total"], 0)
        self.assertIn("rule_prior_ratio", prov)
        validation = report["validation"]
        self.assertIn("pairwise_accuracy", validation)
        self.assertIn("ndcg_realised_gain_points", validation)
        model = zscore.PairwiseRankModel(ranking)
        flat = build_model_features(records[0]["global_features"],
                                    records[0]["candidates"][0]["features"])
        scores = model.score_candidates({"joint_cand_0": flat,
                                         "joint_cand_1": flat})
        self.assertEqual(set(scores), {"joint_cand_0", "joint_cand_1"})

    def test_correction_uses_old_model_winner(self):
        records = _records(16)
        backend, imitation, gain_model, _, bundle, envelope = \
            self._trained_models(records)
        correction, _ = backend.train(
            "correction", bundle.joint_records, scorer="joint_scorer",
            execution_context_hash="1" * 64, old_model=gain_model,
            audit_envelope=envelope,
            capacity={"minimum_leaf_support": 2, "minimum_leaf_groups": 2})
        self.assertEqual(correction["target"], "correction")
        self.assertEqual(correction["old_model_hash"],
                         zs.digest(gain_model))
        zs.validate_scorer_model(correction)
        # label semantics: winner chosen by the OLD MODEL, veto = -its gain
        rows = zd.candidate_rows(records)
        old_scorer = zscore.ScorerModel(gain_model)
        veto_rows = zo.correction_rows(rows, old_scorer)
        self.assertTrue(veto_rows)
        for row in veto_rows:
            self.assertAlmostEqual(row["veto_target"], -row["gain"])

    def test_ranking_provenance_flagged(self):
        rows = zd.candidate_rows(_records(8))
        pairs, summary = zo.ranking_pairs(rows)
        self.assertGreater(summary["pairs_total"], 0)
        self.assertAlmostEqual(
            summary["real_gain_ratio"] + summary["rule_prior_ratio"], 1.0)


class Test08SmallOfflineFlow(unittest.TestCase):
    """Task 13.12: model records carry full provenance and reload."""

    def test_train_validate_load_report(self):
        records = _records(16)
        backend = zt.ZergTrainingBackend()
        imitation, report = backend.train(
            "imitation", records, scorer="hydralisk_scorer",
            execution_context_hash="1" * 64,
            capacity={"minimum_leaf_support": 2, "minimum_leaf_groups": 2})
        self.assertEqual(imitation["dataset_hash"], zs.digest(list(records)))
        self.assertIn("validation_report", imitation)
        self.assertIn("n_scenarios", report["validation"])
        self.assertIn("teacher_top1_agreement", report["validation"])
        validation_metrics = report["validation"]
        self.assertIsNotNone(validation_metrics["teacher_top3_agreement"])
        bundle, envelope = synthetic_audit(records)
        gain_model, gain_report = backend.train(
            "gain", bundle.joint_records, scorer="joint_scorer",
            execution_context_hash="1" * 64,
            audit_envelope=envelope,
            capacity={"minimum_leaf_support": 2, "minimum_leaf_groups": 2})
        for metric in ("mse", "positive_recognition", "negative_recognition"):
            self.assertIn(metric, gain_report["validation"])
        # JSON round trip via the NaN-safe dumps
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "imitation.json"
            path.write_text(json.dumps(zs._sanitize(imitation), indent=2))
            model = zscore.ScorerModel(zs.load_json(str(path)))
            self.assertEqual(model.scorer, "hydralisk_scorer")
        # experiment protocol smoke
        from decision_tree.zerg.experiment import (InterventionBudget,
                                                   default_exploration_config)
        from decision_tree.zerg import experiment_manifest
        budget = InterventionBudget(default_exploration_config())
        budget.record(actor="u0", family="focus_fire", scene_id="s", step=1,
                      candidate_id="c", intervention_type="single_step",
                      horizon=1)
        allowed, reason = budget.allow(actor="u0", family="focus_fire",
                                       scene_id="s", step=2)
        self.assertFalse(allowed)          # unit cooldown
        manifest = experiment_manifest.make_manifest(
            experiment_id="exp_v11", mode="counterfactual_pairs",
            base_policy="tactical_v1_rule",
            candidate_arms=[{"arm_id": "a1", "candidate_id": "c1"}],
            exploration_config=default_exploration_config(),
            source_fingerprint="f" * 64, execution_context_hash="e" * 64)
        with tempfile.TemporaryDirectory() as tmp:
            frozen = experiment_manifest.freeze(f"{tmp}/m.json", manifest)
            experiment_manifest.record_progress(
                str(frozen), [{"experiment_id": "exp_v11", "run_id": "r",
                               "episode_id": "e1", "snapshot_id": 1,
                               "game_loop": 10, "arm_id": "a1"}])
            summary = experiment_manifest.summarize(str(frozen))
            self.assertEqual(summary["per_arm"]["a1"]["unfinished"], 1)


if __name__ == "__main__":
    unittest.main()
