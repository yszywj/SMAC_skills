"""Regression checks for masked supervision and trajectory provenance."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

from tree_skills.data.build_shard import (analyze, build_shard,
                                         native_teacher_diagnostics,
                                         original_assignments)
from tree_skills.data.collect_real import PairCollector
from tree_skills.scripts.relabel_collection import compare_labels, replay_rows
from tree_skills.teacher.action_mapping import SmacActionMap


def sample_row(loop=8):
    # Native 0..8, with keep extension 9. Both illegal native slot 6 and
    # keep/padding slots deliberately dominate the unmasked logits.
    logits = [0, 4, 0, 0, 0, 0, 100, 3, 2, 200] + [300] * 10
    return {"episode": "ep0001", "game_loop": loop,
            "teacher_raw": {"agents": [{"agent": 0, "obs": [loop],
                                         "avail": [0, 1, 0, 0, 0, 0, 0, 1, 1, 1]}],
                            "logits": {"0": logits}},
            "healers": {"u0": {"candidates": {
                "u1": {"features": [1.0], "score": 3},
                "u2": {"features": [2.0], "score": 2}}}},
            "pair_scoring": {"proposed_assignments": {"u0": "u1"},
                             "changed": {"u0": {"from": "u2", "to": "u1"}}},
            "final_actions": [7, 1, 1], "healing": {}}


class CollectionLabelsTest(unittest.TestCase):
    def setUp(self):
        self.mapping = SmacActionMap(n_agents=3, n_enemies=3)

    def test_native_argmax_masks_illegal_keep_and_padding(self):
        diag = native_teacher_diagnostics(sample_row(), "u0", self.mapping)
        self.assertEqual(diag["teacher_native_argmax"], 1)
        self.assertEqual(diag["teacher_native_legal_slots"], [1, 7, 8])
        self.assertAlmostEqual(diag["teacher_stop_probability"] +
                               diag["teacher_legal_heal_probability"], 1.0)

    def test_shard_preserves_original_proposal_and_rejects_illegal_labels(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "shard.jsonl"
            written, skipped = build_shard([sample_row()], self.mapping, str(path))
            self.assertEqual(written, 1)
            self.assertEqual(sum(skipped.values()), 0)
            group = json.loads(path.read_text())
            self.assertEqual(group["labels"], {"u1": 0.5, "u2": -0.5})
            self.assertEqual(group["diagnostics"]["original_assignments"], {"u0": "u2"})
            self.assertEqual(group["diagnostics"]["proposed_assignments"], {"u0": "u1"})
            with self.assertRaises(FileExistsError):
                build_shard([sample_row()], self.mapping, str(path))
            bad = sample_row()
            bad["teacher_raw"]["agents"][0]["avail"][7] = 0
            written, skipped = build_shard([bad], self.mapping, str(Path(temp) / "bad.jsonl"))
            self.assertEqual(written, 0)
            self.assertEqual(skipped["invalid_mask_or_illegal_candidate"], 1)

    def test_changes_count_actions_not_episodes(self):
        kept, other = sample_row(16), sample_row(24)
        kept["final_actions"][0] = 9
        other["final_actions"][0] = 4
        stats = analyze([sample_row(), kept, other], self.mapping)
        self.assertEqual(stats["recommended_changes"], 3)
        self.assertEqual(stats["recommendations_submitted"], 1)
        self.assertEqual(stats["recommendations_kept"], 1)
        self.assertEqual(stats["recommendations_other_action"], 1)

    def test_collector_receives_actual_keep_decisions(self):
        with tempfile.TemporaryDirectory() as temp:
            collector = PairCollector(temp, "test")
            collector.record_teacher_step("ep0001", 8, [], {})
            keep = {"u0": {"selected": True, "reason": "preserve_observed_order"}}
            collector.finish_step("ep0001", 8, sample_row()["pair_scoring"], {}, [9],
                                  keep_decisions=keep)
            row = json.loads(Path(collector.path).read_text())
            self.assertEqual(row["healing"]["keep_decisions"], keep)
            self.assertEqual(row["pair_scoring"]["original_assignments"], {"u0": "u2"})
            self.assertIsNone(original_assignments({}))

    def test_replay_preserves_history_actions_and_source(self):
        class FakeClient:
            def __init__(self):
                self.loops, self.resets = [], []
            def reset_episode(self, episode):
                self.resets.append(episode)
            def logits(self, episode, loop, agents):
                self.loops.append(loop)
                self_seen = [a["agent"] for a in agents]
                assert self_seen == [0]
                logits = [0.0] * 20
                logits[7], logits[8] = 6.0, 9.0
                return {0: logits}
        rows = [sample_row(8), sample_row(16)]
        rows[0]["healers"] = {}  # Still required in the teacher's history.
        original = copy.deepcopy(rows)
        client = FakeClient()
        replayed = list(replay_rows(rows, client, self.mapping, {"kind": "test"}))
        self.assertEqual(rows, original)
        self.assertEqual(client.loops, [8, 16])
        self.assertEqual(client.resets, ["ep0001"])
        self.assertEqual(replayed[1]["final_actions"], rows[1]["final_actions"])
        self.assertEqual(replayed[1]["pair_scoring"], rows[1]["pair_scoring"])
        candidate = replayed[1]["healers"]["u0"]["candidates"]["u2"]
        self.assertEqual(candidate["trajectory_score"], 2)
        self.assertEqual(candidate["score"], 9)
        self.assertEqual(compare_labels(rows, replayed, self.mapping)["multi_candidate_top1_changed"], 1)

    def test_replay_rejects_missing_history_observation(self):
        rows = [sample_row(8), sample_row(16)]
        rows[0]["teacher_raw"]["agents"] = []
        class FakeClient:
            def reset_episode(self, episode):
                pass
        with self.assertRaisesRegex(ValueError, "missing replay observations"):
            list(replay_rows(rows, FakeClient(), self.mapping, {}))


if __name__ == "__main__":
    unittest.main()
