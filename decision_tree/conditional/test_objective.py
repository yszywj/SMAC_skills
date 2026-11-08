import unittest

from decision_tree.conditional.objective import (
    adoption_verdict, pair_labels, partition_rows, summarize_pairs, training_preference,
)


def pair(group, *, identity="p0", candidate_return=20.0, reference_return=20.0,
         candidate_outcome="win", reference_outcome="win", scope="single_intervention"):
    return {
        "group_id": group, "pair_id": identity, "scope": scope,
        "candidate": {"return": candidate_return, "outcome": candidate_outcome},
        "reference": {"return": reference_return, "outcome": reference_outcome},
    }


class WinFirstObjectiveTests(unittest.TestCase):
    def test_large_return_cannot_compensate_for_lost_win(self):
        rows = [pair(str(i), candidate_return=1000, candidate_outcome="loss") for i in range(4)]
        summary = summarize_pairs(rows, scope="single_intervention")
        self.assertEqual(training_preference(summary)["choice"], "reference")
        self.assertEqual(summary["mean_win_gain"], -1)
        self.assertEqual(summary["mean_return_gain"], 980)

    def test_equal_scene_mass_with_repeated_pairs_and_missing_arm(self):
        rows = [pair("a", identity=str(i), candidate_return=24) for i in range(10)]
        rows += [pair("b", candidate_return=18), pair("c", candidate_return=20)]
        missing = pair("d")
        missing["candidate"] = None
        rows.append(missing)
        summary = summarize_pairs(rows, scope="single_intervention")
        self.assertAlmostEqual(summary["mean_return_gain"], 2 / 3)
        self.assertEqual(summary["matched_scenes"], 3)
        self.assertEqual(summary["missing_pairs"], [["d", "p0"]])
        self.assertEqual(training_preference(summary)["choice"], "insufficient")
        self.assertIsNone(pair_labels(missing))

    def test_filter_scenes_before_examining_holdout_labels(self):
        heldout = {"group_id": "heldout", "candidate": "malformed label must not be read"}
        rows = partition_rows([pair("train"), heldout], training_groups={"train"},
                              validation_groups={"heldout"}, partition="train")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["group_id"], "train")
        with self.assertRaisesRegex(ValueError, "disjoint"):
            partition_rows(rows, training_groups={"train"}, validation_groups={"train"}, partition="train")
        with self.assertRaisesRegex(ValueError, "unregistered"):
            partition_rows([pair("other")], training_groups={"train"},
                           validation_groups={"heldout"}, partition="train")

    def test_whole_policy_requires_complete_registered_cohort(self):
        rows = [pair(str(i), scope="whole_policy", reference_outcome="loss", candidate_return=1)
                for i in range(4)]
        summary = summarize_pairs(rows, scope="whole_policy", expected_groups={str(i) for i in range(4)})
        self.assertEqual(adoption_verdict(summary), {"adopt": True, "reason": "higher_win_rate"})
        summary = summarize_pairs(rows, scope="whole_policy", expected_groups={str(i) for i in range(5)})
        self.assertFalse(adoption_verdict(summary)["adopt"])
        self.assertEqual(summary["missing_scene_rows"], ["4"])
        summary = summarize_pairs(rows, scope="whole_policy")
        self.assertFalse(adoption_verdict(summary)["adopt"])

    def test_single_step_and_repeated_scene_cannot_approve_continuous_policy(self):
        rows = [pair(str(i), candidate_return=21) for i in range(4)]
        summary = summarize_pairs(rows, scope="single_intervention", expected_groups={str(i) for i in range(4)})
        self.assertEqual(training_preference(summary)["choice"], "candidate")
        self.assertFalse(adoption_verdict(summary)["adopt"])
        with self.assertRaisesRegex(ValueError, "cannot be pooled"):
            summarize_pairs(rows, scope="whole_policy")
        repeat = [pair("one", scope="whole_policy"), pair("one", identity="other", scope="whole_policy")]
        with self.assertRaisesRegex(ValueError, "multiple whole-policy"):
            summarize_pairs(repeat, scope="whole_policy")

    def test_unknown_or_nonfinite_terminals_and_duplicate_pairs_are_errors(self):
        row = pair("a")
        row["candidate"]["outcome"] = "failed"
        with self.assertRaisesRegex(ValueError, "known outcome"):
            pair_labels(row)
        row["candidate"] = {"return": float("nan"), "outcome": "win"}
        with self.assertRaisesRegex(ValueError, "finite return"):
            pair_labels(row)
        with self.assertRaisesRegex(ValueError, "Duplicate pair"):
            summarize_pairs([pair("a"), pair("a")], scope="single_intervention")


if __name__ == "__main__":
    unittest.main()
