"""Process-isolation and failure-continuation tests for the QP benchmark."""

import importlib.util
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest import mock


SCRIPT = Path(__file__).parents[1] / "scripts" / "benchmark_hard_pact_qp.py"
SPEC = importlib.util.spec_from_file_location("hard_pact_qp_benchmark", SCRIPT)
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


def test_oom_classification():
    assert benchmark.classify_failure("CUDA out of memory") == "OOM"
    assert benchmark.classify_failure("cuPIQP backend is unavailable") == "unsupported_dependency"
    assert benchmark.classify_failure("bad residual") == "numerical_failure"


def test_parent_continues_after_simulated_oom(tmp_path):
    args = SimpleNamespace(
        solvers="qpth,cupiqp", modes="rollout", batch_sizes="256,512",
        dtype="float32", device="cuda:0", warmup=1, iterations=1,
        timeout=1.0, output_dir=tmp_path,
    )
    outcomes = iter((
        {"solver": "qpth", "mode": "rollout", "batch_size": 256,
         "dtype": "float32", "status": "OOM"},
        {"solver": "cupiqp", "mode": "rollout", "batch_size": 256,
         "dtype": "float32", "status": "unsupported_dependency"},
        {"solver": "cupiqp", "mode": "rollout", "batch_size": 512,
         "dtype": "float32", "status": "unsupported_dependency"},
    ))
    with mock.patch.object(benchmark, "run_cell", side_effect=lambda *_: next(outcomes)) as call:
        rows = benchmark.run_matrix(args, SCRIPT)
    assert call.call_count == 3
    assert [row["status"] for row in rows] == [
        "OOM", "skipped_after_oom",
        "unsupported_dependency", "unsupported_dependency",
    ]


def test_real_training_benchmark_forwards_backend_and_chunk_sizes():
    script = SCRIPT.with_name("benchmark_hard_pact_training.py")
    completed = subprocess.run(
        [
            sys.executable, str(script), "--dry-run", "--solvers", "cupiqp",
            "--task", "go2_hard_pact_full_isaaclab", "--num-envs", "4096",
            "--iterations", "5",
        ],
        check=True, capture_output=True, text=True,
    )
    command = completed.stdout.strip()
    assert "--task go2_hard_pact_full_isaaclab" in command
    assert "--num_envs 4096" in command
    assert "--max_iterations 5" in command
    assert "--qp_solver cupiqp" in command
    assert "--qp_rollout_chunk_size 4096" in command
    assert "--qp_ppo_chunk_size 4096" in command
