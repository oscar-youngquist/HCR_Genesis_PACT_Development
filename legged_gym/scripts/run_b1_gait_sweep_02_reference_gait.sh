#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SWEEP_SCRIPT="${SWEEP_SCRIPT:-${SCRIPT_DIR}/b1_scripted_gait_parameter_sweep.py}"
RESULTS_ROOT="${RESULTS_ROOT:-${SCRIPT_DIR}/b1_gait_sweep_results}"
STAGE1_DIR="${STAGE1_DIR:-${RESULTS_ROOT}/01_pd_gain_phase_lead_grid}"
OUTPUT_DIR="${OUTPUT_DIR:-${RESULTS_ROOT}/02_reference_gait_grid_from_stage1_best}"
BEST_JSON="${STAGE1_DIR}/b1_gait_sweep_best.json"

CONDA_SH="${CONDA_SH:-/home/oyoungquist/anaconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-/home/oyoungquist/.conda/envs/genesis_lr}"
GPU="${GPU:-cuda:0}"
SEED="${SEED:-0}"

if [[ ! -f "${SWEEP_SCRIPT}" ]]; then
    echo "Sweep script not found: ${SWEEP_SCRIPT}" >&2
    echo "Set SWEEP_SCRIPT=/path/to/b1_scripted_gait_parameter_sweep.py" >&2
    exit 1
fi

if [[ ! -f "${BEST_JSON}" ]]; then
    echo "Stage-1 best-result file not found: ${BEST_JSON}" >&2
    echo "Run run_b1_gait_sweep_01_pd_phase.sh first, or set STAGE1_DIR." >&2
    exit 1
fi

if [[ ! -f "${CONDA_SH}" ]]; then
    echo "Conda setup not found: ${CONDA_SH}" >&2
    exit 1
fi

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
export SIMULATOR="${SIMULATOR:-genesis_b1_unifp}"

read -r BEST_GAIN_PROFILE BEST_PHASE_LEAD < <(
    python - "${BEST_JSON}" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as file:
    best = json.load(file)["best_parameters"]

gain_keys = (
    "hip_kp",
    "thigh_kp",
    "calf_kp",
    "hip_kd",
    "thigh_kd",
    "calf_kd",
)
gain_profile = ":".join(str(best[key]) for key in gain_keys)
print(gain_profile, best["sweep_phase_lead"])
PY
)

mkdir -p "${OUTPUT_DIR}"

python "${SWEEP_SCRIPT}" \
    --strategy grid \
    --max-trials 0 \
    --seed "${SEED}" \
    --resume \
    --settling-time 1.5 \
    --evaluation-time 4.0 \
    --command 0.5,0.0,0.0 \
    --gain-profiles "${BEST_GAIN_PROFILE}" \
    --phase-leads "${BEST_PHASE_LEAD}" \
    --sweep-amplitudes 0.08,0.10,0.12,0.14 \
    --cycle-times 0.56,0.64,0.72,0.80 \
    --target-joint-pos-scales 0.17,0.20,0.23 \
    --target-joint-pos-thds 0.50 \
    --output-dir "${OUTPUT_DIR}" \
    --headless \
    --gpu="${GPU}"
