"""Opt-in five-step Isaac Lab checks with real checkpoints; never starts learning.

Run from the repository root in the Isaac Lab environment. Each case gets a fresh
Kit process, and COMPLETE is checked because Kit shutdown can mask exit status.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pact-run", required=True)
    p.add_argument("--pact-ckpt", default="-1")
    p.add_argument("--pos-run", required=True)
    p.add_argument("--pos-ckpt", default="-1")
    p.add_argument("--unifp-run", required=True)
    p.add_argument("--unifp-ckpt", default="-1")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--gpu", default="cuda:0")
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    a.output_dir.mkdir(parents=True, exist_ok=True)
    cases = [
        ("pact_nominal", "b1z1_pact", a.pact_run, a.pact_ckpt, "nominal"),
        ("pact_combined", "b1z1_pact", a.pact_run, a.pact_ckpt, "combined-disturbance"),
        ("pos_ee", "b1z1_ppo_pos", a.pos_run, a.pos_ckpt, "ee-disturbance"),
        ("original_reject", "b1z1_unifp_original", a.unifp_run, a.unifp_ckpt, "combined-disturbance"),
        # These reuse architecture-compatible weights only to exercise dispatch;
        # they are NOT trained ablation checkpoints or performance comparisons.
        ("ab6_position", "b1z1_pact_ab6_position_film", a.pos_run, a.pos_ckpt, "ee-disturbance"),
        ("ab8_coupled", "b1z1_pact_ab8_coupled_film", a.pact_run, a.pact_ckpt, "combined-disturbance"),
        ("forced_failure", "b1z1_pact", a.pact_run, a.pact_ckpt, "nominal"),
    ]
    results = []
    for name, task, run, ckpt, condition in cases:
        output = a.output_dir / name
        code = ("import runpy; m=runpy.run_path('legged_gym/scripts/eval_b1z1_exp1.py'); "
                "m['EVAL_CONFIG'].update(event_times=[0.02,0.04,0.06],transition_seconds=0.02,"
                "recovery_window_seconds=0.02,recovery_dwell_seconds=0.02); ")
        if name == "forced_failure":
            code += "m['EVAL_CONFIG'].update(height_min_m=5.,height_max_m=6.,failure_patience_seconds=0.02); "
        if name == "ab6_position":
            code += ("m['EVAL_CONFIG']['overrides']['noise']['add_noise']=True; "
                     "m['EVAL_CONFIG']['overrides']['domain_rand'].update(randomize_pd_gain=True,"
                     "randomize_motor_strength=True,randomize_ctrl_delay=True); ")
        code += "m['main']()"
        command = [sys.executable, "-u", "-c", code, "--task", task, "--run", run, "--ckpt", ckpt,
                   "--num-envs", "2", "--episodes", "3", "--horizon-seconds", ".02" if name == "forced_failure" else ".10",
                   "--condition", condition, "--load-range", "out", "--headless", "--gpu", a.gpu,
                   "--output-dir", str(output)]
        with (a.output_dir / (name + ".log")).open("w") as stream:
            result = subprocess.run(command, cwd=root, stdout=stream, stderr=subprocess.STDOUT,
                                    env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
        if result.returncode or not (output / "COMPLETE").exists():
            raise RuntimeError(f"{name} failed; inspect {a.output_dir / (name + '.log')}")
        manifest = json.loads((output / "records.json").read_text())
        parts = [np.load(output / item["file"], allow_pickle=False) for item in manifest["chunks"]]
        data = {key: np.concatenate([part[key] for part in parts]) for key in parts[0].files}
        assert all(np.isfinite(value).all() for value in data.values()), name
        assert set(data["trial_id"]) == {0, 1, 2}, name
        for trial in (0, 1, 2):
            rows = data["trial_id"] == trial
            np.testing.assert_allclose(data["joint_position"][rows][:-1], data["joint_position_t"][rows][1:])
        max_loads = [float(np.abs(data[key]).max()) for key in ("ee_force_world", "base_force_world", "base_moment_world")]
        if condition == "nominal":
            assert max_loads == [0., 0., 0.]
        elif condition == "ee-disturbance":
            assert max_loads[0] > 50 and max_loads[1:] == [0., 0.]
        else:
            assert all(value > bound for value, bound in zip(max_loads, (50, 35, 10)))
        metadata = json.loads((output / "metadata.json").read_text())
        assert metadata["weights_unchanged"]
        if name == "forced_failure":
            assert data["physical_failure"].all() and data["horizon_complete"].all()
        if "position" in name or name == "pos_ee":
            assert not np.any(data["ff_torque_interval_mean"])
        results.append(dict(case=name, rows=len(data["trial_id"]), max_loads=max_loads, status="passed"))
        print(results[-1], flush=True)
    (a.output_dir / "smoke_results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
