"""Exercise launcher dispatch without loading a simulator or checkpoint tensors."""
import os
from pathlib import Path
import shutil
import subprocess

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "legged_gym" / "scripts"
TASKS = (
    "b1z1_ppo_pos", "b1z1_pact_ab4_coupled_none",
    "b1z1_pact_ab5_coupled_concat", "b1z1_pact_ab8_coupled_film",
    "b1z1_pact_ab6_position_film",
)


@pytest.mark.parametrize("task", TASKS)
def test_collection_arguments(tmp_path, task):
    launcher = tmp_path / f"eval_{task}_three_seeds.sh"
    shutil.copyfile(SCRIPTS / launcher.name, launcher)
    # Capture each invocation as a NUL-delimited argv vector, including paths with spaces.
    (tmp_path / "eval_b1z1_exp1.sh").write_text(
        '#!/bin/bash\nprintf "%s\\0" "$@" >> "$CAPTURE"\nprintf "\\n" >> "$CAPTURE"\n'
    )
    run = tmp_path / "run folder"
    run.mkdir()
    checkpoint = run / "model_28000.pt"
    checkpoint.touch()
    capture = tmp_path / "argv"
    env = dict(os.environ, CAPTURE=str(capture))
    output = tmp_path / "output folder"
    result = subprocess.run(
        ["bash", str(launcher), str(run), "28000", str(output)],
        env=env, capture_output=True, text=True, check=True,
    )
    rows = capture.read_bytes().splitlines()
    assert len(rows) == 9
    for index, row in enumerate(rows):
        args = row.decode().rstrip("\0").split("\0")
        assert args.count("--headless") == 1
        args.remove("--headless")
        flags = dict(zip(args[::2], args[1::2]))
        condition = ("nominal", "ee-disturbance", "combined-disturbance")[index // 3]
        folder = ("nominal", "ee_only_in", "combine_in")[index // 3]
        assert flags == {
            "--task": task, "--run": str(run), "--ckpt": "28000",
            "--seed": str(index % 3), "--num-envs": "10", "--episodes": "100",
            "--horizon-seconds": "12", "--condition": condition,
            "--load-range": "in", "--motion-mode": "mixed", "--gpu": "cuda:0",
            "--output-dir": str(output / folder / f"seed_{index % 3}"),
        }
    assert "[Collection 9/9]" in result.stdout
    checkpoint.unlink()
    failed = subprocess.run(
        ["bash", str(launcher), str(run), "28000"], env=env, capture_output=True, text=True,
    )
    assert failed.returncode == 2
    assert "Checkpoint not found" in failed.stderr
    assert capture.read_bytes().splitlines() == rows


@pytest.mark.parametrize("fail_index", [None, 2])
def test_sequential_approaches(tmp_path, fail_index):
    launcher = tmp_path / "eval_b1z1_approaches_three_seeds.sh"
    shutil.copyfile(SCRIPTS / launcher.name, launcher)
    capture = tmp_path / "order"
    for index, task in enumerate(TASKS):
        (tmp_path / f"eval_{task}_three_seeds.sh").write_text(
            f'echo {task} >> "$CAPTURE"\nexit {1 if index == fail_index else 0}\n'
        )
    result = subprocess.run(
        ["bash", str(launcher)], env=dict(os.environ, CAPTURE=str(capture)),
        capture_output=True, text=True,
    )
    count = len(TASKS) if fail_index is None else fail_index + 1
    assert capture.read_text().splitlines() == list(TASKS[:count])
    assert result.returncode == (0 if fail_index is None else 1)
    assert f"[Approach {count}/5]" in result.stdout


def test_shell_syntax():
    for name in [f"eval_{task}_three_seeds.sh" for task in TASKS] + [
        "eval_b1z1_approaches_three_seeds.sh"
    ]:
        subprocess.run(["bash", "-n", str(SCRIPTS / name)], check=True)
