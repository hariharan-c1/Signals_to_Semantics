import contextlib
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from signals_to_semantics.cli import main
from signals_to_semantics.demo import (
    DEFAULT_HERO_CASE,
    DemoValidationError,
    run_synthetic_signal_demo,
    validate_hero_trace,
)


class DemoTests(unittest.TestCase):
    def test_real_hero_trace_validates(self):
        summary = validate_hero_trace()
        self.assertEqual(summary.checksums_verified, 10)
        self.assertEqual(summary.top_actor, "ACTOR1")
        self.assertEqual(summary.model_label, "obj_crossing")
        self.assertEqual(summary.human_label, "cut_in")
        self.assertEqual(summary.human_primary_actor, "ACTOR1")
        self.assertEqual(summary.retrieval_rank, 3)

    def test_real_demo_json_output(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exit_code = main(["demo", "--json"])
        payload = json.loads(output.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["case_id"], "hero-val50-crossing-vehicle")
        self.assertAlmostEqual(payload["s1_hmm_posterior"], 0.9986784334)

    def test_changed_hero_artifact_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            copied = Path(temp_dir) / "hero_scenario"
            shutil.copytree(DEFAULT_HERO_CASE, copied)
            target = copied / "s1_verification.json"
            target.write_text(target.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaisesRegex(DemoValidationError, "Checksum mismatch"):
                validate_hero_trace(copied)

    def test_synthetic_signal_demo_executes_detector(self):
        result = run_synthetic_signal_demo()
        self.assertEqual(result["demo"], "synthetic-signal-detection")
        self.assertEqual(result["sampling_hz"], 20.0)
        self.assertGreaterEqual(len(result["events"]), 1)
        event = result["events"][0]
        self.assertGreaterEqual(event["duration_s"], 0.25)
        self.assertGreaterEqual(event["speed_drop_mps"], 0.5)
        self.assertLessEqual(event["minimum_acceleration_mps2"], -1.0)
        self.assertTrue(event["window_key"].startswith("synthetic-braking-demo|"))

    def test_synthetic_cli_json_output(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exit_code = main(["demo-synthetic", "--json"])
        payload = json.loads(output.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["sample_count"], 161)
        self.assertIn("S0/S1A", payload["scope"])


if __name__ == "__main__":
    unittest.main()
