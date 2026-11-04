"""Teacher-environment regression tests against the actual official API.

Run with torch and the archived MARL-GPT dependencies available. Environments
without torch skip this module; a skip is not teacher validation evidence.
"""

import unittest

try:
    from tree_skills.teacher.obs_builder import SmacObsBuilder
    from tree_skills.teacher.reference_check import (
        ACT, BLOCK, DIMS, HISTORY_LEN, N_ATTR, VOCAB, OfficialReference)
    import numpy as np
except ImportError as exc:
    raise unittest.SkipTest("teacher dependencies unavailable: " + str(exc))


class TestTeacherToken(unittest.TestCase):
    def builder(self, last_token=True, **kwargs):
        return SmacObsBuilder(history_len=HISTORY_LEN, block_size=BLOCK,
                              vocab_size=VOCAB, action_size=ACT, n_attr=N_ATTR,
                              last_token=last_token, **DIMS, **kwargs)

    def inputs(self, builder):
        # Distinct frames expose accidental history/position changes too.
        frames = [np.linspace(-0.1, 0.9, 162, dtype=np.float32) * (i + 1) / 4
                  for i in range(HISTORY_LEN)]
        for loop, frame in enumerate(frames):
            builder.observe("episode", 2, loop, frame)
        return builder.build_input("episode", 2), frames

    def test_default_matches_official_constructor_and_method(self):
        ours, frames = self.inputs(self.builder())
        reference = OfficialReference(True)
        expected = reference.build(frames)
        self.assertFalse(reference.inference.cfg.env_specific)
        self.assertEqual(reference.inference._env_indicator, 0)
        self.assertEqual(float(ours["obs"][0, -1]), 0)
        self.assertEqual(int(ours["attr_pos"][0, -1]), N_ATTR - 1)
        for key in ("obs", "group_pos", "agent_pos", "time_pos", "attr_pos"):
            np.testing.assert_array_equal(ours[key], expected[key])

    def test_old_indicator_one_is_detected_as_different(self):
        old, frames = self.inputs(self.builder(env_indicator=1))
        expected = OfficialReference(True).build(frames)
        self.assertEqual(float(old["obs"][0, -1]), 1)
        self.assertNotEqual(float(old["obs"][0, -1]),
                            float(expected["obs"][0, -1]))
        np.testing.assert_array_equal(old["obs"][..., :-1], expected["obs"][..., :-1])

    def test_last_token_disabled_matches_official_reference(self):
        ours, frames = self.inputs(self.builder(last_token=False))
        expected = OfficialReference(False).build(frames)
        for key in ("obs", "group_pos", "agent_pos", "time_pos", "attr_pos"):
            np.testing.assert_array_equal(ours[key], expected[key])
        self.assertNotEqual(int(ours["attr_pos"][0, -1]), N_ATTR - 1)


if __name__ == "__main__":
    unittest.main()
