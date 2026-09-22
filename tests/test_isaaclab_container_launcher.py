"""Launcher tests use a fake Apptainer binary; no simulator or GPU is started."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


LAUNCHER = Path(__file__).resolve().parents[1] / "containers/isaaclab/run.sh"


class LauncherTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        repo = root / "repo"
        scripts = repo / "legged_gym/scripts"
        scripts.mkdir(parents=True)
        for name in ("train.py", "train_hard_pact.py", "b1z1_unifp_unity.sh", "go2_hard_pact_unity.sh"):
            (scripts / name).touch()
        image = root / "runtime.sif"
        image.touch()
        fake = root / "apptainer"
        fake.write_text('#!/bin/bash\nprintf "CVD=%s SIM=%s\\n" "$APPTAINERENV_CUDA_VISIBLE_DEVICES" "$APPTAINERENV_SIMULATOR"\nprintf "%s\\n" "$@"\n')
        fake.chmod(0o755)
        self.env = dict(os.environ, IMAGE=str(image), REPO=str(repo),
                        RUN_DIR=str(root / "run"), CUDA_VISIBLE_DEVICES="1",
                        OMNI_KIT_ACCEPT_EULA="YES", PATH=f"{root}:{os.environ['PATH']}")

    def run_launcher(self, *args):
        return subprocess.run(["bash", str(LAUNCHER), *args], env=self.env,
                              text=True, capture_output=True)

    def test_b1z1_gpu_mapping_and_smoke_limits(self):
        result = self.run_launcher("smoke", "b1z1_pact")
        self.assertEqual(result.returncode, 0, result.stderr)
        for token in ("CVD=1 SIM=isaaclab_b1z1_pact", "train.py", "--gpu=cuda:0",
                      "--num_envs=2", "--max_iterations=1", "--cleanenv", "--nv"):
            self.assertIn(token, result.stdout)

    def test_hardpact_entrypoint_and_solver_forwarding(self):
        result = self.run_launcher("train", "go2_hard_pact_full_isaaclab", "--qp_solver", "cupiqp")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("SIM=isaaclab", result.stdout)
        self.assertIn("train_hard_pact.py", result.stdout)
        self.assertIn("--qp_solver\ncupiqp", result.stdout)

    def test_reject_device_override_and_multiple_gpus(self):
        self.assertNotEqual(self.run_launcher("train", "b1z1_unifp", "--gpu=cuda:1").returncode, 0)
        self.assertNotEqual(self.run_launcher("smoke", "b1z1_unifp", "--num_envs=4096").returncode, 0)
        self.env["CUDA_VISIBLE_DEVICES"] = "0,1"
        self.assertNotEqual(self.run_launcher("train", "b1z1_unifp").returncode, 0)

    def test_unity_shell_script_forwarding(self):
        result = self.run_launcher("script", "b1z1_unifp_unity.sh", "--seed=3")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("/bin/bash\nb1z1_unifp_unity.sh\n--seed=3", result.stdout)
        result = self.run_launcher("script", "go2_hard_pact_unity.sh",
                                   "--task=go2_hard_pact_baseline_isaaclab")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("SIM=isaaclab", result.stdout)

    def test_b1z1_unity_launchers_preserve_existing_conventions(self):
        fake_python = Path(self.tmp.name) / "python"
        fake_python.write_text('#!/bin/bash\nprintf "CVD=%s SIM=%s\\n" "$CUDA_VISIBLE_DEVICES" "$SIMULATOR"\nprintf "%s\\n" "$@"\n')
        fake_python.chmod(0o755)
        scripts = LAUNCHER.parents[2] / "legged_gym/scripts"
        for task in ("b1z1_unifp", "b1z1_pact", "b1z1_pact_pos",
                     "b1z1_unifp_original", "b1z1_unifp_reject"):
            result = subprocess.run(["bash", str(scripts / f"{task}_unity.sh"), "--seed=3"],
                                    env=self.env, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("CVD=1", result.stdout)
            self.assertIn(f"--task={task}", result.stdout)
            self.assertIn("--gpu=cuda:0", result.stdout)
            self.assertTrue(result.stdout.endswith("--seed=3\n"))


if __name__ == "__main__":
    unittest.main()
