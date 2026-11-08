from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from decision_tree.local.runtime import digest, execution_context
from decision_tree.local import study
from decision_tree.phase2.run import carrier_policy
from decision_tree.tests.test_local_dataset import fixture, P1, replace_events
from skills.local_v2 import explore
from skills.local_v2.run import source_hashes


class ExplorationTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.base = self.root / "base"
        self.events = fixture(self.base)
        # The existing synthetic SC2 fixture records this placeholder source.
        stub = patch.object(study, "source_hashes",
                            return_value={str((study.ROOT/"code.py").resolve()): "frozen"})
        stub.start()
        self.addCleanup(stub.stop)
        self.plan = self.root / "plan"

    def test_all_steps_and_unmeasured_count_do_not_depend_on_the_old_tree(self):
        summary = explore.plan(self.base, self.plan)
        self.assertEqual(summary["counts"]["decision_steps"], 3)
        menus = [json.loads(line) for line in (self.plan/"menus.jsonl").read_bytes().splitlines()]
        self.assertEqual([row["step"] for row in menus], [0, 1, 2])
        self.assertTrue(all(row["frozen_v1_menu"] and "BASE" in row["effects"] for row in menus))
        self.assertEqual(summary["counts"]["nonbase_effects"], 6)
        report = explore.audit(self.plan, [], self.root/"empty.json")
        self.assertEqual(report["unmeasured_nonbase"], 6)
        self.assertFalse(report["complete"])

    def test_audits_legacy_forced_and_rejects_changed_prefix(self):
        explore.plan(self.base, self.plan)
        arm = self.root/"forced"
        events = fixture(arm, selected=P1, reward=3.0)
        report = explore.audit(self.plan, [arm, arm], self.root/"first.json")
        self.assertEqual(report["measured_nonbase"], 1)
        self.assertEqual(report["failed_or_rejected"], [])
        data = deepcopy(events)
        frame = next(e for e in data if e["event"]=="skill_observation")
        frame["payload"]["state"]["unexpected_observed_change"] = True
        replace_events(arm, data)
        report = explore.audit(self.plan, [arm], self.root/"second.json")
        self.assertEqual(report["measured_nonbase"], 0)
        self.assertIn("prefix", report["failed_or_rejected"][0]["reason"])

    def test_v2_event_receipts_and_explicit_binding(self):
        policy = {"mode":"base"}
        carrier = carrier_policy(policy)
        base_meta = study.read(self.base/"run.json")
        base_meta["local_policy"]["policy"] = carrier
        base_meta["local_policy"]["context_hash"] = execution_context(base_meta, carrier)
        (self.base/"run.json").write_text(json.dumps(base_meta), encoding="utf-8")
        explore.plan(self.base, self.plan)
        arm = self.root/"forced"
        events = fixture(arm, selected=P1, reward=3.0)
        version = {"implementation":"v2","forced":[1,P1],"source_hashes":source_hashes(),
                   "selector_kind":"phase2","selector":policy,"selector_sha256":digest(policy)}
        for name in ("run.json","summary.json"):
            data = study.read(arm/name)
            data["local_v2"] = version
            if name == "run.json":
                data["local_policy"]["policy"] = carrier
                data["local_policy"]["context_hash"] = execution_context(data, carrier)
            (arm/name).write_text(json.dumps(data),encoding="utf-8")
        for e in events:
            if e["event"] == "local_decision":
                forced = e["step"] == 1
                e.update(event="local_v2_decision",local_v2={"forced":forced,"actor":"u0",
                          "family":"retarget","requested_action":7})
        replace_events(arm,events)
        report = explore.audit(self.plan,[arm],self.root/"v2.json")
        self.assertEqual(report["measured_nonbase"],1)
        bad_step = next(e for e in events if e["event"] == "step" and e["step"]==2)
        bad_step["feedback"]["actors"]["u0"]["submission"] = "rejected"
        replace_events(arm,events)
        report = explore.audit(self.plan,[arm],self.root/"v2bad.json")
        self.assertEqual(report["measured_nonbase"],0)
        self.assertEqual(len(report["failed_or_rejected"]),1)

    def test_incomplete_or_mutated_plan_fails_closed(self):
        explore.plan(self.base,self.plan)
        (self.plan/"menus.jsonl").write_text("",encoding="utf-8")
        with self.assertRaisesRegex(ValueError,"menu changed"):
            explore.audit(self.plan,[],self.root/"bad.json")
        frame = next(e for e in self.events if e["event"]=="local_decision")
        frame["selected_id"] = P1
        replace_events(self.base,self.events)
        other = self.root/"incomplete"
        with self.assertRaisesRegex(ValueError,"BASE modified"):
            explore.plan(self.base,other)
        self.assertFalse((other/"summary.json").exists())
