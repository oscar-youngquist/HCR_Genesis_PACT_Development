#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SWEEP_SCRIPT="${SWEEP_SCRIPT:-${SCRIPT_DIR}/b1_scripted_gait_parameter_sweep.py}"
RESULTS_ROOT="${RESULTS_ROOT:-${SCRIPT_DIR}/b1_gait_sweep_results}"
OUTPUT_DIR="${OUTPUT_DIR:-${RESULTS_ROOT}/01_pd_gain_phase_lead_grid}"

CONDA_SH="${CONDA_SH:-/home/oyoungquist/anaconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-/home/oyoungquist/.conda/envs/genesis_lr}"
GPU="${GPU:-cuda:0}"
SEED="${SEED:-0}"

if [[ ! -f "${SWEEP_SCRIPT}" ]]; then
    echo "Sweep script not found: ${SWEEP_SCRIPT}" >&2
    echo "Set SWEEP_SCRIPT=/path/to/b1_scripted_gait_parameter_sweep.py" >&2
    exit 1
fi

if [[ ! -f "${CONDA_SH}" ]]; then
    echo "Conda setup not found: ${CONDA_SH}" >&2
    exit 1
fi

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
export SIMULATOR="${SIMULATOR:-genesis_b1_unifp}"

mkdir -p "${OUTPUT_DIR}"

python "${SWEEP_SCRIPT}" \
    --strategy grid \
    --max-trials 0 \
    --seed "${SEED}" \
    --resume \
    --settling-time 1.5 \
    --evaluation-time 8.0 \
    --command 0.5,0.0,0.0 \
    --gain-profiles '100:100:200:2.5:2.5:5;150:150:300:4:4:8;200:200:350:5:5:9;250:250:400:6.25:6.25:10;300:300:500:7.5:7.5:12.5' \
    --phase-leads 0.0,0.025,0.05,0.075,0.10 \
    --sweep-amplitudes 0.10 \
    --cycle-times 0.64 \
    --target-joint-pos-scales 0.20 \
    --target-joint-pos-thds 0.50 \
    --output-dir "${OUTPUT_DIR}" \
    --headless \
    --gpu="${GPU}"
