#!/usr/bin/env bash
set -euo pipefail

# Activate Isaac Lab first. Edit these paths before running.
RUN=""   # Trained run directory or registered run name.
CKPT=""  # Fixed checkpoint number or model file (not -1).
OUTPUT_ROOT="evaluation/exp1/b1z1_pact_ab6_position_film/combined_in"

# Explicit evaluation seeds; 100 episodes per seed across 10 robots on cuda:0.
SEEDS=(0 1 2)
HERE="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
if [[ -z "$RUN" || -z "$CKPT" || "$CKPT" == "-1" ]]; then
    echo "Set RUN and a fixed CKPT at the top of this script." >&2
    exit 2
fi

for SEED in "${SEEDS[@]}"; do
    echo "Evaluating b1z1_pact_ab6_position_film: seed=$SEED, episodes=100, robots=10"
    bash "$HERE/eval_b1z1_exp1.sh" \
        --task b1z1_pact_ab6_position_film \
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
