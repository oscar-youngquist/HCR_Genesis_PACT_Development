"""Opt-in 14-control-step GPU checks, compressed schedule; NOT performance data."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for method in ("pact", "pos", "unifp"):
        p.add_argument(f"--{method}-run", required=True)
        p.add_argument(f"--{method}-ckpt", default="-1")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--gpu", default="cuda:0")
    p.add_argument("--case", action="append", help="Optional named case(s); otherwise all cases")
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    methods = [("pact", "b1z1_pact", a.pact_run, a.pact_ckpt),
               ("pos", "b1z1_ppo_pos", a.pos_run, a.pos_ckpt),
               ("unifp", "b1z1_unifp_original", a.unifp_run, a.unifp_ckpt)]
    cases = [(f"{name}_{profile}", task, run, ckpt, profile, "loaded")
             for name, task, run, ckpt in methods for profile in ("pushing", "pulling", "carrying")]
    # Architecture-compatible weights test dispatch only, not trained ablation performance.
    cases += [("ab6", "b1z1_pact_ab6_position_film", a.pos_run, a.pos_ckpt, "pushing", "loaded"),
              ("ab8", "b1z1_pact_ab8_coupled_film", a.pact_run, a.pact_ckpt, "pulling", "loaded"),
              ("ab9", "b1z1_pact_ab9_representation", a.pact_run, a.pact_ckpt, "carrying", "loaded"),
              ("pact_unloaded", "b1z1_pact", a.pact_run, a.pact_ckpt, "pushing", "unloaded"),
              ("failure", "b1z1_pact", a.pact_run, a.pact_ckpt, "pushing", "loaded")]
    if a.case and set(a.case)-{x[0] for x in cases}:
        p.error("Unknown case")
    a.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for name, task, run, ckpt, profile, loading in cases:
        if a.case and name not in a.case:
            continue
        output = a.output_dir / name
        code = '''import importlib.util
spec = importlib.util.spec_from_file_location("exp2_smoke", "legged_gym/scripts/eval_b1z1_exp2.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
m.EVAL_CONFIG.update(move_start_s=.02, velocity_ramp_s=.02, load_times_s=[.06,.10,.14,.24],
                     stop_s=.18, finish_s=.28, recovery_window_seconds=.04, recovery_dwell_seconds=.02)
# Confirm deterministic inference is unchanged after additional decoder calls.
base_inference = m.exp1.inference
def inference(actor, obs, history):
    import torch
    action = base_inference(actor, obs, history)
    state = torch.random.get_rng_state().clone()
    if hasattr(actor, "decode_context"):
        actor.decode_context(actor.context_encoder.forward_inf(history))
    torch.testing.assert_close(action, base_inference(actor, obs, history), rtol=0, atol=0)
    assert torch.equal(state, torch.random.get_rng_state())
    return action
m.exp1.inference = inference
'''
        if name == "failure":
            code += 'm.EVAL_CONFIG.update(height_min_m=5., height_max_m=6.)\n'
        code += "m.main()\n"
        command = [sys.executable, "-u", "-c", code, "--task", task, "--run", run, "--ckpt", ckpt,
                   "--profile", profile, "--loading", loading, "--num-envs", "2", "--episodes", "3",
                   "--horizon-seconds", ".02" if name == "failure" else ".28", "--headless", "--gpu", a.gpu,
                   "--output-dir", str(output)]
        print("Starting", name, flush=True)
        with (a.output_dir / (name+".log")).open("w") as stream:
            result = subprocess.run(command, cwd=root, stdout=stream, stderr=subprocess.STDOUT,
                                    env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
        if result.returncode or not (output / "COMPLETE").exists():
            raise RuntimeError(f"{name} failed; inspect {a.output_dir / (name+'.log')}")
        manifest = json.loads((output / "records.json").read_text())
        chunks = []
        for chunk in manifest["chunks"]:
            with np.load(output / chunk["file"]) as archive:
                chunks.append({k: archive[k] for k in archive.files})
        data = {k: np.concatenate([c[k] for c in chunks]) for k in chunks[0]}
        assert set(data["trial_id"]) == {0, 1, 2}
        assert all(np.isfinite(v).all() for v in data.values())
        assert not any(data[k].any() for k in ("base_force_world", "base_moment_world", "ee_moment_world"))
        norms = np.linalg.norm(data["ee_force_world"], axis=-1)
        if loading == "unloaded" or name == "failure":
            assert norms.max() == 0.
        else:
            assert np.isclose(norms.max(), 65.)
        if loading == "loaded" and name != "failure":
            for value in (25., 45., 65.):
                assert np.isclose(norms, value).any()
            assert data["applied_load_exceeds_training_bounds"].any()
        np.testing.assert_allclose(data["pd_weighted_motor_interval_mean"]+data["ff_weighted_motor_interval_mean"],
                                   data["torque_request_motor_scaled_interval_mean"], rtol=1e-5, atol=1e-4)
        assert (data["executed_torque_mean_square"] >= data["executed_torque_interval_mean"]**2-1e-3).all()
        for trial in (0, 1, 2):
            mask = data["trial_id"] == trial
            np.testing.assert_allclose(data["joint_position"][mask][:-1], data["joint_position_t"][mask][1:])
        if name == "failure":
            assert data["physical_failure"].all() and data["horizon_complete"].all()
            assert not data["alive_for_metrics"].any()
        metadata = json.loads((output / "metadata.json").read_text())
        assert metadata["weights_unchanged"] and metadata["experiment"] == 2
        results.append(dict(case=name, rows=len(norms), force_peak_n=float(norms.max()), status="passed"))
        print(results[-1], flush=True)
        (a.output_dir / "smoke_results.json").write_text(json.dumps(results, indent=2))
    # Loaded/unloaded runs with identical seed/IDs have identical reset mechanics.
    if all((a.output_dir / name / "COMPLETE").exists() for name in ("pact_pushing", "pact_unloaded")):
        initial = [(a.output_dir / name / "initial_states.jsonl").read_text().splitlines() for name in ("pact_pushing", "pact_unloaded")]
        assert [json.loads(x)["initial_state_hash"] for x in initial[0]] == [json.loads(x)["initial_state_hash"] for x in initial[1]]


if __name__ == "__main__":
    main()
