#!/usr/bin/env bash
set -euo pipefail

# Run the five additional approaches sequentially on cuda:0 using their pinned checkpoints.
# Each child collects 3 conditions x 3 seeds x 100 trials with 10 parallel robots.
# Existing PACT and UniFP collections are intentionally not repeated here.
HERE="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
SCRIPTS=(
    eval_b1z1_ppo_pos_three_seeds.sh
    eval_b1z1_pact_ab4_coupled_none_three_seeds.sh
    eval_b1z1_pact_ab5_coupled_concat_three_seeds.sh
    eval_b1z1_pact_ab8_coupled_film_three_seeds.sh
    eval_b1z1_pact_ab6_position_film_three_seeds.sh
)

for INDEX in "${!SCRIPTS[@]}"; do
    echo "[Approach $((INDEX + 1))/${#SCRIPTS[@]}] Starting ${SCRIPTS[$INDEX]}"
    if ! bash "$HERE/${SCRIPTS[$INDEX]}"; then
        echo "Collection failed for ${SCRIPTS[$INDEX]}; stopping before subsequent approaches." >&2
        exit 1
    fi
    echo "[Approaches completed: $((INDEX + 1))/${#SCRIPTS[@]}] ${SCRIPTS[$INDEX]}"
done

# From the repository root (activate the Isaac Lab environment first):
# conda activate lr_lab_cupiqp
# bash legged_gym/scripts/eval_b1z1_approaches_three_seeds.sh
