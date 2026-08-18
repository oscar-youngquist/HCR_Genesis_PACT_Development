"""Focused CPU-fallback checks for shared B1Z1 rollout timing tags."""

import time
import unittest

from rsl_rl.utils.rollout_timing import ROLLOUT_TIMING_TAGS, RolloutPhaseTimer


class RolloutTimingTests(unittest.TestCase):
    def test_cpu_fallback_emits_every_consistent_tag(self):
        timer = RolloutPhaseTimer("cpu")
        for phase in ROLLOUT_TIMING_TAGS:
            if phase == "total":
                continue
            start = timer.start(phase)
            time.sleep(0.0001)
            timer.stop(phase, start)
        metrics = timer.finish()
        self.assertEqual(set(metrics), set(ROLLOUT_TIMING_TAGS.values()))
        self.assertTrue(all(value >= 0.0 for value in metrics.values()))


if __name__ == "__main__":
    unittest.main()
