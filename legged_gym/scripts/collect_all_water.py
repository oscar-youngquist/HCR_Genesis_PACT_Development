"""
Loop over ALL_LIQUID_CONFIGS, run play_test_water.py for each.
go2, 100 envs, headless. Skips configs that already have HDF5 files in any
prior timestamped run dir under exp_data/water_collect/go2/*_<vol>L<liq>_<tank>/.

    SIMULATOR=genesis_pact_water python legged_gym/scripts/collect_all_water.py
"""
import os
import subprocess
import sys
import time
from pathlib import Path

from legged_gym.scripts.liquid_payload_configs import ALL_LIQUID_CONFIGS

TASK = "go2_pact_water"
NUM_ENVS = 100
REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = REPO_ROOT / "exp_data" / "water_collect"
ROBOT_DIR = OUT_ROOT / "go2"


def _config_existing_files(liquid_type, volume, tank):
    pattern = f"*_{int(volume)}L{liquid_type}_{tank}"
    return [p for d in ROBOT_DIR.glob(pattern) if d.is_dir() for p in d.glob("*.h5")]


def _fmt_hms(seconds):
    s = max(int(seconds), 0)
    return f"{s//3600}h{(s%3600)//60:02d}m"


def _slurm_wall_remaining_s():
    job_id = os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOBID")
    if not job_id:
        return None
    try:
        out = subprocess.check_output(
            ["scontrol", "show", "job", str(job_id), "-o"],
            stderr=subprocess.DEVNULL, text=True, timeout=5)
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    fields = dict(p.split("=", 1) for p in out.split() if "=" in p)
    def _to_s(t):
        if "-" in t: d, t = t.split("-"); d = int(d) * 86400
        else: d = 0
        parts = [int(x) for x in t.split(":")]
        while len(parts) < 3: parts.insert(0, 0)
        return d + parts[0]*3600 + parts[1]*60 + parts[2]
    try:
        return _to_s(fields["TimeLimit"]) - _to_s(fields["RunTime"])
    except (KeyError, ValueError):
        return None


configs = list(ALL_LIQUID_CONFIGS)
total_volume = sum(v for _, v, _ in configs)
print(f"=== SWEEP START: {len(configs)} configs, total volume sum = {total_volume:g}L ===", flush=True)
for i, (lt, v, tk) in enumerate(configs, start=1):
    print(f"  [{i:2d}] {lt:5s} {v:g}L {tk}", flush=True)
wall_init = _slurm_wall_remaining_s()
if wall_init is not None:
    print(f"=== SLURM wall budget at start: {_fmt_hms(wall_init)} ===", flush=True)

sweep_t0 = time.time()
results = []
done_volume = 0.0
done_elapsed = 0.0
for i, (liquid_type, volume, tank) in enumerate(configs, start=1):
    tag = f"[{i}/{len(configs)}] {liquid_type}-{volume}L-{tank}"

    pre_existing = _config_existing_files(liquid_type, volume, tank)
    if pre_existing:
        print(f"{tag} skip ({len(pre_existing)} existing files)", flush=True)
        results.append((tag, "skip", 0, len(pre_existing)))
        continue

    print(f"{tag} START  cumulative_sweep_elapsed={_fmt_hms(time.time()-sweep_t0)}", flush=True)
    cfg_t0 = time.time()
    result = subprocess.run([
        sys.executable, "-u", "legged_gym/scripts/play_test_water.py",
        "--task", TASK,
        "--num_envs", str(NUM_ENVS),
        "--liquid_type", liquid_type,
        "--liquid_volume", str(volume),
        "--liquid_tank", tank,
        "--headless",
    ], cwd=REPO_ROOT)
    cfg_elapsed = time.time() - cfg_t0
    n_files = len(_config_existing_files(liquid_type, volume, tank))
    status = "OK" if result.returncode == 0 else f"FAIL(rc={result.returncode})"
    print(f"{tag} END {status}  config_elapsed={_fmt_hms(cfg_elapsed)} ({cfg_elapsed:.0f}s)  files_written={n_files}", flush=True)
    results.append((tag, status, cfg_elapsed, n_files))

    if status == "OK":
        done_volume += volume
        done_elapsed += cfg_elapsed
        remaining_volume = sum(v for _, v, _ in configs[i:])
        if done_volume > 0 and remaining_volume > 0:
            per_L = done_elapsed / done_volume
            eta_remaining = remaining_volume * per_L
            line = (f"   >> ETA: {per_L:.0f}s/L  remaining_volume={remaining_volume:g}L  "
                    f"projected_remaining={_fmt_hms(eta_remaining)}")
            wall = _slurm_wall_remaining_s()
            if wall is not None:
                line += f"  |  SLURM wall remaining={_fmt_hms(wall)}"
                if eta_remaining > wall:
                    line += "  *** WILL NOT FIT ***"
            print(line, flush=True)

print(f"\n=== SWEEP DONE  total_elapsed={_fmt_hms(time.time()-sweep_t0)} ===", flush=True)
for tag, status, elapsed, n in results:
    print(f"  {status:18s}  {_fmt_hms(elapsed):>7s}  files={n:3d}  {tag}", flush=True)
