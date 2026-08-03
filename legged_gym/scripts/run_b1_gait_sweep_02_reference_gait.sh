#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SWEEP_SCRIPT="${SWEEP_SCRIPT:-${SCRIPT_DIR}/b1_scripted_gait_parameter_sweep.py}"
RESULTS_ROOT="${RESULTS_ROOT:-${SCRIPT_DIR}/b1_gait_sweep_results}"
OUTPUT_DIR="${OUTPUT_DIR:-${RESULTS_ROOT}/02_optuna_bound_discovery_random_screen}"

CONDA_SH="${CONDA_SH:-/home/oyoungquist/anaconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-/home/oyoungquist/.conda/envs/genesis_lr}"
GPU="${GPU:-cuda:0}"
SEED="${SEED:-2}"
MAX_TRIALS="${MAX_TRIALS:-500}"

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

# This is a coarse joint screen for identifying useful Optuna/TPE bounds.
# It includes deliberately weaker/stronger guard points around the Phase-1
# winners and randomly samples the large Cartesian space without replacement.
python "${SWEEP_SCRIPT}" \
    --strategy random \
    --max-trials "${MAX_TRIALS}" \
    --seed "${SEED}" \
    --resume \
    --settling-time 2.0 \
    --evaluation-time 5.0 \
    --command 0.5,0.0,0.0 \
    --gain-profiles '175:175:325:4.5:4.5:8.5;200:200:350:5:5:9;225:225:375:5.625:5.625:9.5;250:250:400:6.25:6.25:10;275:275:450:6.875:6.875:11.25;300:300:500:7.5:7.5:12.5;325:325:550:8.125:8.125:13.75' \
    --phase-leads 0.05,0.075,0.10,0.125,0.15,0.175,0.20 \
    --sweep-amplitudes 0.06,0.08,0.10,0.12,0.14,0.16,0.18 \
    --cycle-times 0.48,0.56,0.64,0.72,0.80,0.88 \
    --target-joint-pos-scales 0.14,0.17,0.20,0.23,0.26,0.29 \
    --target-joint-pos-thds 0.35,0.40,0.45,0.50,0.55,0.60,0.65 \
    --output-dir "${OUTPUT_DIR}" \
    --headless \
    --gpu="${GPU}"