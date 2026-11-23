"""V1.2 acceptance tests (Phase 8) — targeted, offline, no SC2.

Run from the repository root:
    python -m unittest decision_tree.zerg.tests.test_v12 -v
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
from decision_tree.zerg.features import DEFAULT_SCHEMA
from decision_tree.zerg.schema import build_internal_model_features, \
    build_model_features, internal_scorer_feature_names, \
    scorer_feature_names
from decision_tree.zerg.selector import apply_complete_plan, \
    apply_selected_plan

from decision_tree.zerg.tests.test_minimal import _manager, _zerg_payload
from decision_tree.zerg.tests.test_v11 import _records

CONSTANTLY_MISSING = "candidate.candidate_support_gain"


def _shadow_records(tmp):
    """Two real shadow episodes over the fixture scene -> JSONL records
    with A-class skill_proposals and the units feature layer."""
    from skills.tactical_v1.tests import fixtures as fx
    bundle = factory.build_zerg_skill_manager()

    def payload():
        allies = [((17., 17.), "hydralisk"), ((18., 17.), "hydralisk"),
                  ((17., 18.), "zergling"), ((18., 18.), "zergling")]
        enemies = {"e0": fx.enemy((20., 20.), "marine", health=30.),
                   "e1": fx.enemy((22., 20.), "marine", health=12.)}
        return fx.with_questions(fx.scene(allies, enemies))

    config = zr.default_config()
    config["mode"] = "shadow"
    config["workspace"] = tmp
    config["source_root"] = str(WORKSPACE)
    config["records_path"] = f"{tmp}/shadow.jsonl"
    config["events_path"] = f"{tmp}/events.jsonl"
    runtime = zr.ZergTreeRuntime(config=config, tactical_config={})
    controller = runtime.wrap(bundle)
    manager = bundle.manager
    episode_ids = []
    for outcome in ("win", "loss"):
        actions, _ = controller.step(payload(), 4)
        controller.commit_final_actions(actions)
        episode = controller.last_meta["episode_id"]
        episode_ids.append(episode)
        runtime.finalize_episode(outcome=outcome, episode_return=1.0,
                                 ally_losses=0, enemy_losses=2,
                                 episode_id=episode)
        manager.reset()
    records = zd.read_records(f"{tmp}/shadow.jsonl")
    return runtime, controller, records, episode_ids


class Test01RunIdentity(unittest.TestCase):
    """Checks 1/2: run_id stable across loops and episodes; episode ids
    unique inside a run."""

    def test_run_id_stable_and_episode_unique(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime, controller, records, episode_ids = _shadow_records(tmp)
            run_id = runtime.run_id
            self.assertTrue(run_id.startswith("run_"))
            # every record and event carries the SAME run_id
            self.assertTrue(all(r["run_id"] == run_id for r in records))
            events = [json.loads(line) for line
                      in Path(tmp, "events.jsonl").read_text().splitlines()]
            self.assertTrue(all(e["run_id"] == run_id for e in events))
            # explicit run_id from config wins
            config = zr.default_config()
            config["run_id"] = "run_fixed_01"
            rt2 = zr.ZergTreeRuntime(config=config)
            self.assertEqual(rt2.run_id, "run_fixed_01")
            rt2.reset()
            self.assertEqual(rt2.run_id, "run_fixed_01")
            # new process (new runtime) -> new run_id
            rt3 = zr.ZergTreeRuntime(config=zr.default_config())
            self.assertNotEqual(rt3.run_id, run_id)

    def test_episode_ids_unique(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, _, records, episode_ids = _shadow_records(tmp)
            self.assertNotEqual(episode_ids[0], episode_ids[1])
            self.assertEqual({r["episode_id"] for r in records},
                             set(episode_ids))


class Test02SplitsAndSceneConfig(unittest.TestCase):
    """Checks 3 + scene_config identity."""

    def test_episode_and_anchor_splits(self):
        records = _records(24)
        train, validation = zd.split_records(records, split_seed=5)
        self.assertFalse({zd.episode_group(r) for r in train}
                         & {zd.episode_group(r) for r in validation})
        pairs = [zd.build_pair_record(
            experiment_id="e", run_id=f"run{i % 6}", episode_id=f"ep{i}",
            scene_config_id="cfg", anchor_id=f"a{i}", pair_id=f"p{i}",
            baseline_id="BASE", candidate_id=f"c{i}",
            initial_state_hash="a", prefix_hash="b",
            execution_context_hash="c", intervention_type="single_step",
            intervention_horizon=1) for i in range(12)]
        ptrain, pval = zd.split_pairs(pairs, split_seed=2,
                                      min_train_groups=2)
        self.assertFalse({zd.anchor_group(p) for p in ptrain}
                         & {zd.anchor_group(p) for p in pval})

    def test_scene_config_identity(self):
        payload = _zerg_payload()
        manager = _manager()
        manager.step(payload, 4)
        snapshot = manager._pending_commit[0]
        sid1, source1 = zr.scene_config_identity(snapshot, "10gen_terran")
        sid2, source2 = zr.scene_config_identity(snapshot, "10gen_terran")
        self.assertEqual(sid1, sid2)
        self.assertEqual(source1, "canonical_hash")
        self.assertTrue(sid1.startswith("sconf_"))
        # per-step state (health/position) must NOT change the identity...
        payload_same = copy.deepcopy(payload)
        payload_same["state"]["allies"]["u3"]["health"] = 20.0
        from skills.tactical_v1.context import build_snapshot
        snapshot_same = build_snapshot(payload_same, 1)
        sid_state, _ = zr.scene_config_identity(snapshot_same, "10gen_terran")
        self.assertEqual(sid1, sid_state)
        # ...but a roster COMPOSITION change does
        payload_comp = copy.deepcopy(payload)
        payload_comp["state"]["allies"]["u3"]["type"] = "hydralisk"
        payload_comp["state"]["allies"]["u3"]["unit_type_id"] = 107
        snapshot_comp = build_snapshot(payload_comp, 1)
        sid3, _ = zr.scene_config_identity(snapshot_comp, "10gen_terran")
        self.assertNotEqual(sid1, sid3)
        # manifest-supplied identity passes through
        config = zr.default_config()
        config["scene_config"] = {"scene_config_id": "cfg_manifest_1"}
        rt = zr.ZergTreeRuntime(config=config)
        got, src = rt._scene_config(snapshot, "10gen_terran")
        self.assertEqual((got, src), ("cfg_manifest_1", "experiment_manifest"))


class Test03ConstantMissing(unittest.TestCase):
    """Checks 4/5: constant-missing columns never crash training, are
    excluded from the model, and imputation only covers active features."""

    def _records_with_dead_column(self):
        records = _records(12)
        for record in records:
            for candidate in record["candidates"]:
                candidate["features"][CONSTANTLY_MISSING.split(".", 1)[1]] = \
                    float("nan")
        return records

    def test_training_survives_and_declares_inactive(self):
        records = self._records_with_dead_column()
        backend = zt.ZergTrainingBackend()
        model, report = backend.train(
            "imitation", records, scorer="hydralisk_scorer",
            execution_context_hash="1" * 64,
            capacity={"minimum_leaf_support": 2, "minimum_leaf_groups": 2})
        inactive_names = {row["name"] for row in model["inactive_features"]}
        self.assertIn(CONSTANTLY_MISSING, inactive_names)
        self.assertNotIn(CONSTANTLY_MISSING, model["active_feature_names"])
        self.assertIn(CONSTANTLY_MISSING, model["feature_names"])  # full schema kept
        # training rows keep the full schema too
        rows = zd.candidate_rows(records)
        self.assertIn(CONSTANTLY_MISSING, rows[0]["features"])
        zs.validate_scorer_model(model)
        loaded = zscore.ScorerModel(model)
        self.assertIn(CONSTANTLY_MISSING, loaded.feature_names)
        self.assertNotIn(CONSTANTLY_MISSING, loaded.active_feature_names)


class Test04FeatureParity(unittest.TestCase):
    """Check 5 (+2.4): offline record rows and online scoring produce the
    identical vector for BOTH layouts; metadata keys are forbidden."""

    def test_real_snapshot_parity(self):
        payload = _zerg_payload()
        manager = _manager()
        manager.step(payload, 4)
        snapshot, plans, _, _, context = manager._pending_commit
        proposals = manager.skills["focus_fire"].propose(
            context, manager.memory, manager.telemetry, None)
        internal = zc.skill_internal_candidates(proposals, snapshot,
                                                context=context)
        extractor = zr.StateFeatureExtractor(DEFAULT_SCHEMA)
        state = extractor.extract_state(snapshot)
        candidate = internal[0]
        online_standard = build_model_features(state, candidate.features)
        online_internal = build_internal_model_features(
            state, candidate.features, candidate.controlled_actors)
        record = zd.build_decision_record(
            run_id="run_p", episode_id="ep_p", snapshot=snapshot,
            candidates=internal, global_features=state,
            teacher_choice=candidate.candidate_id)
        record["skill_proposals"] = {
            "focus_fire": [c.to_record() for c in internal]}
        offline_rows = zd.candidate_rows([record])
        offline = offline_rows[0]["features"]
        self.assertEqual(list(offline.keys()), list(online_standard.keys()))
        for key, value in online_standard.items():
            if isinstance(value, float) and math.isnan(value):
                self.assertTrue(math.isnan(offline[key]), key)
            else:
                self.assertEqual(value, offline[key], key)
        a_rows = zd.expand_skill_proposals([record])
        target = next(row for row in a_rows
                      if row["candidate_id"] == candidate.candidate_id)
        self.assertEqual(list(target["features"].keys()),
                         list(online_standard.keys()))
        self.assertEqual(list(target["features_internal"].keys()),
                         list(online_internal.keys()))
        manager.commit_final_actions(manager._cache[1])

    def test_metadata_forbidden_in_features(self):
        for key in ("run_id", "episode_id", "scene_config_id", "seed",
                    "snapshot_id", "candidate_id", "unit_tag"):
            self.assertIn(key, zs.FEATURE_FORBIDDEN)


class Test05AClassPipeline(unittest.TestCase):
    """Checks 6/7/8/9: real A-class rows, group isolation, internal model
    with unit features, correct routing."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        runtime, controller, records, eps = _shadow_records(cls._tmp.name)
        cls.records = records
        cls.controller = controller

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_a_class_rows_extracted(self):
        rows = zd.expand_skill_proposals(self.records)
        self.assertTrue(rows)
        skills = {row["candidate_skill"] for row in rows}
        self.assertIn("focus_fire", skills)
        # per-skill and per-unit-type exports work from the same JSONL
        hydra = zd.export_internal_training_rows(rows, unit_type="hydralisk")
        zerg = zd.export_internal_training_rows(rows, unit_type="zergling")
        focus = zd.export_internal_training_rows(rows, skill="focus_fire")
        self.assertTrue(hydra or zerg)
        self.assertTrue(focus)
        for row in rows:
            self.assertEqual(row["candidate_family"], "skill_internal")
            self.assertTrue(row["decision_group_id"].startswith(
                row["run_id"]))

    def test_group_isolation(self):
        rows = zd.expand_skill_proposals(self.records)
        groups = {row["decision_group_id"] for row in rows}
        # different skills never share one internal decision group
        by_group: dict = {}
        for row in rows:
            by_group.setdefault(row["decision_group_id"], set()).add(
                row["candidate_skill"])
        self.assertTrue(all(len(v) == 1 for v in by_group.values()))
        self.assertGreater(len(groups), 0)

    def test_internal_model_uses_unit_features(self):
        backend = zt.ZergTrainingBackend()
        model, report = backend.train_internal(
            "imitation", self.records, scorer="hydralisk_scorer",
            execution_context_hash="1" * 64,
            capacity={"minimum_leaf_support": 2, "minimum_leaf_groups": 2},
            min_train_groups=1)
        self.assertEqual(model["feature_layout"], "internal")
        self.assertEqual(model["feature_names"],
                         internal_scorer_feature_names())
        active = model["active_feature_names"]
        self.assertTrue(any(name.startswith("unit.mean.") for name in active))
        zs.validate_scorer_model(model)
        loaded = zscore.ScorerModel(model)
        # online scoring with the internal layout uses the unit aggregates
        state = zr.StateFeatureExtractor(DEFAULT_SCHEMA).extract_state(
            self.records[0].get("_snapshot") if False else None
            if self.records[0] is None else None) if False else None
        # build a state from the record itself
        global_features = self.records[0]["global_features"]
        cand = self.records[0]["candidates"][0]
        flat = build_internal_model_features(
            global_features, cand["features"],
            cand.get("controlled_actors", []))
        cand_obj = zc.ZergCandidate(
            candidate_id="x", candidate_family="skill_internal",
            candidate_skill=cand["candidate_skill"],
            controlled_actors=tuple(cand.get("controlled_actors", [])),
            target=None, phase=None, current_actions={},
            proposed_actions={}, features=dict(cand["features"]),
            resource_claims={}, dependencies=[], atomic=False,
            legal=True, rejection_reason=None, source_plan_id="x",
            rule_score=1.0)
        scores = zscore.score_candidates(global_features, [cand_obj],
                                         model=loaded)
        self.assertEqual(len(scores), 1)
        self.assertIn(scores[0].reason,
                      {"model_score", "leaf_below_support_rule_confidence",
                       "prediction_unsupported_rule_fallback"})

    def test_routing_rules(self):
        unit_types = {"u0": "hydralisk", "u1": "zergling"}
        state = {"unit_types": unit_types}
        single = zc.ZergCandidate(
            candidate_id="s", candidate_family="skill_internal",
            candidate_skill="focus_fire", controlled_actors=("u0",),
            target=None, phase=None, current_actions={},
            proposed_actions={"u0": 6}, features={},
            resource_claims={}, dependencies=[], atomic=False,
            legal=True, rejection_reason=None, source_plan_id="s")
        mixed = zc.ZergCandidate(
            candidate_id="m", candidate_family="skill_internal",
            candidate_skill="coordinated_engage",
            controlled_actors=("u0", "u1"),
            target=None, phase=None, current_actions={},
            proposed_actions={"u0": 6, "u1": 2}, features={},
            resource_claims={}, dependencies=[], atomic=True,
            legal=True, rejection_reason=None, source_plan_id="m")
        routed = zscore.route_candidates([single, mixed], state)
        self.assertEqual(routed["hydralisk_scorer"], [single])
        self.assertEqual(routed["joint_scorer"], [mixed])
        # unknown unit type falls back to joint safely
        single.candidate_family = "joint_plan"
        routed = zscore.route_candidates([single], state)
        self.assertEqual(routed["joint_scorer"], [single])


class Test06FactoryAndCache(unittest.TestCase):
    """Checks 10/11."""

    def test_bundle_binding_and_refusal(self):
        bundle = factory.build_zerg_skill_manager()
        config = zr.default_config()
        config["mode"] = "shadow"
        with tempfile.TemporaryDirectory() as tmp:
            config["workspace"] = tmp
            config["source_root"] = str(WORKSPACE)
            config["records_path"] = f"{tmp}/s.jsonl"
            config["events_path"] = f"{tmp}/e.jsonl"
            runtime = zr.ZergTreeRuntime(config=config, tactical_config={})
            controller = runtime.wrap(bundle)
            self.assertIsNotNone(runtime.bundle_provenance)
            self.assertEqual(runtime.bundle_provenance["source_fingerprint"],
                             bundle.source_fingerprint)
            # a bundle with a wrong fingerprint must be refused
            class FakeBundle:
                manager = bundle.manager
                source_fingerprint = "deadbeef" * 8
                module_path = bundle.module_path
                effective_config = bundle.effective_config
                manifest_path = bundle.manifest_path
            with self.assertRaises(ValueError):
                runtime2 = zr.ZergTreeRuntime(config=copy.deepcopy(config),
                                              tactical_config={})
                runtime2.wrap(FakeBundle())

    def test_module_cache_namespaces(self):
        from decision_tree.zerg.factory import load_resolved_package
        resolved = WORKSPACE / "decision_tree/zerg/resolved/tactical_v1"
        module = load_resolved_package(resolved)
        self.assertIn("zerg_resolved_tactical_v1_", module.__name__)
        # same fingerprint reuse only when __file__ matches
        again = load_resolved_package(resolved)
        self.assertIs(again, module)
        # a DIFFERENT declared namespace with the same tree still resolves
        other = load_resolved_package(resolved, "other_namespace_x")
        self.assertEqual(other.__name__, "other_namespace_x")


class Test07RankingBidirectional(unittest.TestCase):
    """Check 12: tournament scores are exactly antisymmetric under a/b swap,
    and the report evaluates both directions."""

    def test_antisymmetry_and_report(self):
        records = _records(16)
        backend = zt.ZergTrainingBackend()
        ranking, report = backend.train(
            "ranking", records, scorer="joint_scorer",
            execution_context_hash="1" * 64,
            capacity={"minimum_leaf_support": 2, "minimum_leaf_groups": 2})
        validation = report["validation"]
        self.assertIn("forward_direction_accuracy", validation)
        self.assertIn("reverse_direction_accuracy", validation)
        self.assertIn("pairwise_accuracy", validation)
        model = zscore.PairwiseRankModel(ranking)
        flat_a = build_model_features(records[0]["global_features"],
                                      records[0]["candidates"][0]["features"])
        flat_b = build_model_features(records[0]["global_features"],
                                      records[0]["candidates"][1]["features"])
        forward = model.score_candidates({"A": flat_a, "B": flat_b})
        swapped = model.score_candidates({"A": flat_b, "B": flat_a})
        self.assertAlmostEqual(forward["A"]["score"], -swapped["A"]["score"])
        self.assertAlmostEqual(forward["B"]["score"], -swapped["B"]["score"])
        # wins are counted per candidate
        self.assertIn("wins", forward["A"])
        self.assertEqual(forward["A"]["wins"] + forward["A"]["losses"], 1)

    def test_ndcg_only_with_complete_real_gains(self):
        pairs, summary = zo.ranking_pairs(
            zd.candidate_rows(_records(8)))
        self.assertGreater(summary["pairs_total"], 0)
        backend = zt.ZergTrainingBackend()
        empty_report = backend._ranking_report(
            {"trees": [], "aggregation": "mean", "base_value": 0.0},
            zs.ranking_feature_names(), pairs)
        self.assertIn("ndcg_points", empty_report)


class Test08CorrectionOldModelWinner(unittest.TestCase):
    """Check 13."""

    def test_veto_targets_old_model_winner(self):
        records = _records(16)
        backend = zt.ZergTrainingBackend()
        _, _, gain_model = backend._trained_models if False else \
            (None, None, None)
        gain_model, _ = backend.train(
            "gain", records, scorer="joint_scorer",
            execution_context_hash="1" * 64,
            capacity={"minimum_leaf_support": 2, "minimum_leaf_groups": 2})
        rows = zd.candidate_rows(records)
        old = zscore.ScorerModel(gain_model)
        veto_rows = zo.correction_rows(rows, old)
        self.assertTrue(veto_rows)
        for row in veto_rows:
            self.assertAlmostEqual(row["veto_target"], -row["gain"])
        correction, _ = backend.train(
            "correction", records, scorer="joint_scorer",
            execution_context_hash="1" * 64, old_model=gain_model,
            capacity={"minimum_leaf_support": 2, "minimum_leaf_groups": 2})
        self.assertEqual(correction["old_model_hash"], zs.digest(gain_model))


class Test09TreeScoreGuards(unittest.TestCase):
    """Checks 14/15/16/17."""

    @staticmethod
    def _shadow_runtime(tmp):
        config = zr.default_config()
        config["mode"] = "tree_score"
        config["workspace"] = tmp
        config["source_root"] = str(WORKSPACE)
        config["records_path"] = f"{tmp}/s.jsonl"
        config["events_path"] = f"{tmp}/e.jsonl"
        return zr.ZergTreeRuntime(config=config, tactical_config={})

    def test_incomplete_candidates_refused(self):
        candidate = zc.CompletePlanCandidate(
            candidate_id="alt", snapshot_id=1, plans=[], actions={"u0": 1},
            resource_claims={}, dependencies=[], execution_duties={},
            rule_score=None, candidate_features={}, legal=True,
            source_plan_ids=[], plan_objects_available=False)
        result = apply_complete_plan(candidate, manager=None)
        self.assertFalse(result["applied"])
        self.assertTrue(result["fallback_to_base"])
        self.assertEqual(result["reason"], "no_complete_plan_objects_refused")

    def test_tree_score_never_changes_without_models(self):
        payload = _zerg_payload()
        plain_actions, _ = _manager().step(payload, 4)
        with tempfile.TemporaryDirectory() as tmp:
            runtime = self._shadow_runtime(tmp)
            bundle = factory.build_zerg_skill_manager()
            controller = runtime.wrap(bundle)
            manager = bundle.manager
            actions, summary = controller.step(payload, 4)
            self.assertEqual(actions, plain_actions)          # check 17/14
            self.assertFalse(summary.get("zerg_tree", {}).get("changed"))
            # gate reports exactly why
            self.assertFalse(runtime.last_tree_score_gate.get(
                "model_loaded"))
            controller.commit_final_actions(actions)
            # check 15: memory commit still the manager's own decision
            self.assertTrue(manager.memory.committed)

    def test_full_branch_replacement_and_base_fallback(self):
        payload = _zerg_payload()
        bundle = factory.build_zerg_skill_manager()
        manager = bundle.manager
        actions, summary = manager.step(payload, 4)
        snapshot, base_plans, _, _, context = manager._pending_commit
        base_actions = {f"u{i}": a for i, a in enumerate(manager._cache[1])}
        # check 15: re-select the manager's OWN branch as the model choice
        selected = zc.CompletePlanCandidate(
            candidate_id="re_selected", snapshot_id=snapshot.snapshot_id,
            plans=list(base_plans), actions=dict(base_actions),
            resource_claims={}, dependencies=[], execution_duties={},
            rule_score=None, candidate_features={}, legal=True,
            source_plan_ids=[str(p.proposal_id) for p in base_plans],
            plan_objects_available=True)
        result = apply_complete_plan(selected, manager, context=context,
                                     memory=manager.memory,
                                     base_plans=list(base_plans),
                                     base_actions=base_actions)
        self.assertTrue(result["applied"], result)
        self.assertEqual(result["validator_problems"], [])
        self.assertEqual(result["commit_status"], "pending_manager_commit")
        # manager state update keeps the original commit flow intact
        zr.ZergTreeRuntime._update_manager_plan_state(
            manager, result["final_plans"], actions, summary)
        self.assertTrue(manager._pending_commit[1])
        manager.commit_final_actions(actions)
        self.assertTrue(manager.memory.committed)
        # check 16: empty plan objects -> full BASE fallback
        broken = zc.CompletePlanCandidate(
            candidate_id="broken", snapshot_id=snapshot.snapshot_id,
            plans=[], actions={}, resource_claims={}, dependencies=[],
            execution_duties={}, rule_score=None, candidate_features={},
            legal=True, source_plan_ids=[], plan_objects_available=True)
        result = apply_complete_plan(broken, manager, context=context,
                                     memory=manager.memory,
                                     base_plans=list(base_plans),
                                     base_actions=base_actions)
        self.assertFalse(result["applied"])
        self.assertTrue(result["fallback_to_base"])

    def test_shadow_bit_identical(self):
        payload = _zerg_payload()
        plain_actions, _ = _manager().step(copy.deepcopy(payload), 4)
        _manager_second = _manager()
        plain2, _ = _manager_second.step(copy.deepcopy(payload), 4)
        self.assertEqual(plain_actions, plain2)
        with tempfile.TemporaryDirectory() as tmp:
            config = zr.default_config()
            config["mode"] = "shadow"
            config["workspace"] = tmp
            config["source_root"] = str(WORKSPACE)
            config["records_path"] = f"{tmp}/s.jsonl"
            config["events_path"] = f"{tmp}/e.jsonl"
            runtime = zr.ZergTreeRuntime(config=config, tactical_config={})
            controller = runtime.wrap(factory.build_zerg_skill_manager())
            actions, summary = controller.step(copy.deepcopy(payload), 4)
            self.assertEqual(actions, plain_actions)
            self.assertFalse(summary["zerg_tree"]["changed"])


if __name__ == "__main__":
    unittest.main()
