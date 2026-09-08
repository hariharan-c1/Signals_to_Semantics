import unittest

import numpy as np

from signals_to_semantics.detection.braking_events import detect_brakes


class BrakingEventTests(unittest.TestCase):
    def test_constant_speed_has_no_braking_event(self):
        timestamps = np.arange(0.0, 3.1, 0.1)
        speed = np.full_like(timestamps, 10.0)

        segments, acceleration, jerk = detect_brakes(
            speed,
            timestamps,
            a_min=-1.0,
            j_min=-5.0,
            smooth_window=1,
            mode="OR",
        )

        self.assertEqual(segments, [])
        self.assertTrue(np.allclose(acceleration, 0.0))
        self.assertTrue(np.allclose(jerk, 0.0))

    def test_linear_speed_drop_is_detected(self):
        timestamps = np.arange(0.0, 3.1, 0.1)
        speed = np.where(
            timestamps < 1.0,
            10.0,
            np.where(timestamps <= 2.0, 12.0 - 2.0 * timestamps, 8.0),
        )

        segments, _, _ = detect_brakes(
            speed,
            timestamps,
            a_min=-1.0,
            j_min=-1000.0,
            min_dur_s=0.2,
            max_gap_s=0.15,
            smooth_window=1,
            mode="OR",
            min_delta_v=0.5,
        )

        self.assertEqual(len(segments), 1)
        start, end = segments[0]
        self.assertGreaterEqual(timestamps[start], 0.9)
        self.assertLessEqual(timestamps[end], 2.2)
        self.assertGreaterEqual(speed[start] - speed[end], 0.5)


if __name__ == "__main__":
    unittest.main()
