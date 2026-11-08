from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from decision_tree.conditional.runtime import ConditionalController, normalize_forced
from decision_tree.phase2.policy import REGROUP_POLICY
from decision_tree.tests.test_phase2_runtime import obs
from sc2_skill_adapter.bridge.skill_runtime import load_manager
from skills.local_v3.counterfactual import audit_counterfactual, read_audited_run
from skills.local_v3.run import check_completion
from skills.local_v3.tests.test_audit import synthetic_run


def events_file(path, events):
    (path/"events.jsonl").write_text("\n".join(json.dumps(event) for event in events), encoding="utf-8")


class ForcedContinuousTests(unittest.TestCase):
    def test_base_override_does_not_consume_and_later_policy_resumes(self):
        owner = ConditionalController(load_manager()(), forced={"step": 0, "binding": "BASE"},
                                      context_hash=REGROUP_POLICY["model"]["context_hash"])
        records = []
        for clock in (0, 8, 16, 24):
            actions, _, events = owner.step(obs(10, clock), 6)
            owner.manager.commit_final_actions(actions)
            records.append(events[0])
        self.assertEqual([row["forced"] for row in records], [True, False, False, False])
        self.assertEqual([row["selected_id"] for row in records],
                         ["BASE", "regroup_step:u0:4", "BASE", "regroup_step:u0:4"])
        self.assertEqual([row["interventions"] for row in records], [0, 1, 1, 2])

    def test_forced_catalogue_direction_and_base_alias(self):
        for binding, expected, count in (("regroup_step:u0:3", 3, 1), ("regroup_step:u0:2", 2, 0)):
            owner = ConditionalController(load_manager()(), forced={"step": 0, "binding": binding},
                                          context_hash=REGROUP_POLICY["model"]["context_hash"])
            actions, _, events = owner.step(obs(10), 6)
            self.assertEqual(actions[0], expected)
            self.assertEqual(events[0]["interventions"], count)
            self.assertEqual(events[0]["proposed_id"], binding)
            self.assertTrue(events[0]["forced"])
        for invalid in ({"step": -1, "binding": "BASE"}, {"step": 0, "binding": "bad"}):
            with self.assertRaises(ValueError):
                normalize_forced(invalid)

    def test_audit_multiscene_incumbent_and_changed_prefix_rejection(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            old, new = root/"old", root/"new"
            old.mkdir()
            new.mkdir()
            _, context, previous = synthetic_run(old)
            forced = {"step": 1, "binding": "BASE"}
            metadata, other_context, altered = synthetic_run(new, forced, scene_index=1)
            self.assertEqual(context, other_context)
            events_file(old, previous)
            events_file(new, altered)
            with patch.dict(REGROUP_POLICY["model"], {"context_hash": context}):
                cached = read_audited_run(old)
                pair = audit_counterfactual(cached, new, 2, 1, "BASE")
                self.assertTrue(pair["different_action"])
                self.assertEqual(pair["seed"], 1)
                self.assertEqual(pair["return_gain"], 0)
                mutated = deepcopy(altered)
                observation = next(event for event in mutated if event["event"] == "skill_observation")
                observation["payload"]["state"]["marker"] = "changed_prefix"
                events_file(new, mutated)
                with self.assertRaisesRegex(ValueError, "Observable prefix"):
                    audit_counterfactual(cached, new, 2, 1, "BASE")
                with self.assertRaisesRegex(ValueError, "Wrong declared override"):
                    audit_counterfactual(cached, new, 2, 2, "BASE")
                events_file(new, altered)
                check_completion(new, metadata, 1)

    def test_unreachable_anchor_fails_complete_audit(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary)
            metadata, context, events = synthetic_run(path, {"step": 5, "binding": "BASE"}, scene_index=0)
            events_file(path, events)
            with patch.dict(REGROUP_POLICY["model"], {"context_hash": context}):
                with self.assertRaisesRegex(ValueError, "exactly once"):
                    check_completion(path, metadata, 1)
