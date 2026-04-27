"""
Loop over ALL_LIQUID_CONFIGS, run play_test_water.py for each.
go2, 100 envs, headless. Skips configs whose output dir already has HDF5 files.

    python legged_gym/scripts/collect_all_water.py
"""
import subprocess
import sys
from pathlib import Path

from legged_gym.scripts.liquid_payload_configs import ALL_LIQUID_CONFIGS

TASK = "go2_pact_water"
NUM_ENVS = 100
REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = REPO_ROOT / "exp_data" / "water_collect"


for i, (liquid_type, volume, tank) in enumerate(ALL_LIQUID_CONFIGS, start=1):
    out_dir = OUT_ROOT / f"{TASK}_{int(volume)}L{liquid_type}_{tank}"
    tag = f"[{i}/{len(ALL_LIQUID_CONFIGS)}] {liquid_type}-{volume}L-{tank}"

    if out_dir.exists() and any(p.suffix in (".h5", ".hdf5") for p in out_dir.iterdir()):
        print(f"{tag} skip")
        continue

    print(f"{tag} run")
    subprocess.run([
        sys.executable, "legged_gym/scripts/play_test_water.py",
        "--task", TASK,
        "--num_envs", str(NUM_ENVS),
        "--liquid_type", liquid_type,
        "--liquid_volume", str(volume),
        "--liquid_tank", tank,
        "--headless",
    ], cwd=REPO_ROOT)
