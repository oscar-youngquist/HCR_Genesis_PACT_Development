"""Synthetic event tests; no simulator, training imports, or GPU needed."""

import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary
from tensorboard.summary.writer.event_file_writer import EventFileWriter
from tensorboard.util import tensor_util

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/extract_training_metrics.py"
spec = importlib.util.spec_from_file_location("extract_training_metrics", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ExtractionTests(unittest.TestCase):
    def test_sampling_endpoints_and_keep_all(self):
        self.assertEqual(module.retained_indices(10, 3), {0, 4, 9})
        self.assertEqual(list(module.retained_indices(4, 0)), [0, 1, 2, 3])

    def test_filters(self):
        for tag in ("Episode/rew_tracking_lin_vel", "tracking/linear_velocity",
                    "PINN/inverse/mae", "physics/loss/rollout", "Loss/pinn_loss"):
            self.assertTrue(module.selected(tag, module.DEFAULT_PATTERNS, []))
        self.assertFalse(module.selected("debug/weights", module.DEFAULT_PATTERNS, []))
        self.assertFalse(module.selected("PINN/timing", module.DEFAULT_PATTERNS, ["*/timing"]))

    def test_round_trip_and_source_preservation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            writer = EventFileWriter(str(root / "source"))
            for step in range(10):
                writer.add_event(Event(step=step, wall_time=100 + step, summary=Summary(value=[
                    Summary.Value(tag="Episode/reward", simple_value=step),
                    Summary.Value(tag="physics/loss/rollout", tensor=tensor_util.make_tensor_proto(
                        np.array(float("nan") if step == 4 else step / 10., dtype=np.float64))),
                    Summary.Value(tag="debug/noise", simple_value=step),
                    Summary.Value(tag="Episode/image", image=Summary.Image(encoded_image_string=b"image")),
                ])))
            writer.close()
            source = next((root / "source").glob("events.*"))
            before = source.read_bytes()
            output = root / "summary"
            module.extract(source, output, module.DEFAULT_PATTERNS, [], 3)
            self.assertEqual(source.read_bytes(), before)
            with (output / "metrics.csv").open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 6)
            reward = [row for row in rows if row["tag"] == "Episode/reward"]
            self.assertEqual([int(row["step"]) for row in reward], [0, 4, 9])
            self.assertEqual([float(row["wall_time"]) for row in reward], [100., 104., 109.])
            event_file = next((output / "tensorboard").glob("events.*"))
            self.assertEqual(len(list(module.scalar_values(event_file))), 6)
            manifest = json.loads((output / "summary.json").read_text())
            self.assertEqual(manifest["kept_tags"]["physics/loss/rollout"]["nonfinite_count"], 1)
            self.assertEqual(manifest["kept_tags"]["Episode/reward"]["maximum"], 9.)
            self.assertIn("debug/noise", manifest["dropped_scalar_tags"])
            with self.assertRaises(FileExistsError):
                module.extract(source, output, module.DEFAULT_PATTERNS, [], 3)


if __name__ == "__main__":
    unittest.main()
