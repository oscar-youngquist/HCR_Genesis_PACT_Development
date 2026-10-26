#!/usr/bin/env bash
set -euo pipefail

# Activate your Isaac Lab environment before running this launcher.
# Edit these defaults, or supply RUN_DIRECTORY CHECKPOINT [OUTPUT_ROOT].
RUN=""  # Trained run directory or registered run name.
CKPT=""  # Fixed checkpoint number or model file (not -1).
if [[ $# -gt 3 ]]; then
    echo "Usage: bash $0 RUN_DIRECTORY CHECKPOINT [OUTPUT_ROOT]" >&2
    exit 2
fi

HERE="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
RUN="${1:-$RUN}"
CKPT="${2:-$CKPT}"
OUTPUT_ROOT="${3:-evaluation/exp1/pact/combined_in}"

# Explicit evaluation seeds; all use the same trained checkpoint.
SEEDS=(0 1 2)

# Require a fixed checkpoint to avoid changes between seed runs.
if [[ -z "$RUN" || -z "$CKPT" || "$CKPT" == "-1" ]]; then
    echo "Please specify a fixed checkpoint number or model file, not -1." >&2
    exit 2
fi

for SEED in "${SEEDS[@]}"; do
    echo "Evaluating B1Z1 PACT: seed=$SEED, episodes=100, robots=10"
    bash "$HERE/eval_b1z1_exp1.sh" \
        --task b1z1_pact \
        --run "$RUN" \
        --ckpt "$CKPT" \
        --seed "$SEED" \
        --num-envs 10 \
        --episodes 100 \
        --horizon-seconds 12 \
        --condition combined-disturbance \
        --load-range in \
        --motion-mode mixed \
        --gpu cuda:0 \
        --headless \
        --output-dir "$OUTPUT_ROOT/seed_${SEED}"
done
