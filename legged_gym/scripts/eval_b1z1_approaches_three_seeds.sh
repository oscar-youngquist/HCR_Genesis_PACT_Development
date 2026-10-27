#!/usr/bin/env bash
set -euo pipefail

# Shared device for every child evaluation (Exp1 and Exp2). Edit cuda:0 here,
# or override at launch with EVAL_GPU=cuda:1 bash <this script>.
export EVAL_GPU="${EVAL_GPU:-cuda:1}"

# Run the selected approaches sequentially using their pinned checkpoints.
# Each child collects 3 conditions x 3 seeds x 100 trials with 10 parallel robots.
# Existing PACT and UniFP collections are intentionally not repeated here.
HERE="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
# SCRIPTS=(
#     eval_b1z1_ppo_pos_three_seeds.sh
#     eval_b1z1_pact_ab4_coupled_none_three_seeds.sh
#     eval_b1z1_pact_ab5_coupled_concat_three_seeds.sh
#     eval_b1z1_pact_ab8_coupled_film_three_seeds.sh
#     eval_b1z1_pact_ab6_position_film_three_seeds.sh
# )

SCRIPTS=(
    eval_b1z1_pact_three_seeds.sh
    eval_b1z1_unifp_original_three_seeds.sh
    eval_b1z1_ppo_pos_three_seeds.sh
    eval_b1z1_pact_ab4_coupled_none_three_seeds.sh
    eval_b1z1_pact_ab5_coupled_concat_three_seeds.sh
    eval_b1z1_pact_ab8_coupled_film_three_seeds.sh
    eval_b1z1_pact_ab6_position_film_three_seeds.sh
)

for INDEX in "${!SCRIPTS[@]}"; do
    echo "[Approach $((INDEX + 1))/${#SCRIPTS[@]}] Starting ${SCRIPTS[$INDEX]} on $EVAL_GPU"
    if ! bash "$HERE/${SCRIPTS[$INDEX]}"; then
        echo "Collection failed for ${SCRIPTS[$INDEX]}; stopping before subsequent approaches." >&2
        exit 1
    fi
    echo "[Approaches completed: $((INDEX + 1))/${#SCRIPTS[@]}] ${SCRIPTS[$INDEX]}"
done

# From the repository root (activate the Isaac Lab environment first):
# conda activate lr_lab_cupiqp
# bash legged_gym/scripts/eval_b1z1_approaches_three_seeds.sh
