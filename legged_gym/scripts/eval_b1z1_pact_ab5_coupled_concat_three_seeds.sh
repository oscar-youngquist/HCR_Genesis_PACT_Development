#!/usr/bin/env bash
set -euo pipefail

# Activate Isaac Lab first. Defaults pin the latest local checkpoint as of 2026-10-27.
# Optional arguments: RUN_DIRECTORY CHECKPOINT_NUMBER OUTPUT_BASE
HERE="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO="$(CDPATH= cd -- "$HERE/../.." && pwd)"
RUN="${1:-$REPO/logs/b1z1/b1z1_pact_ab5_coupled_concat/Oct24_06-31-57_b1z1_pact_ab5_coupled_concat}"
CKPT="${2:-28000}"
OUTPUT_BASE="${3:-$REPO/evaluation/exp1/b1z1_pact_ab5_coupled_concat}"
if [[ $# -gt 3 || ! "$CKPT" =~ ^[0-9]+$ ]]; then
    echo "Usage: bash $0 [RUN_DIRECTORY [CHECKPOINT_NUMBER [OUTPUT_BASE]]]" >&2
    exit 2
fi
if [[ ! -f "$RUN/model_${CKPT}.pt" ]]; then
    echo "Checkpoint not found: $RUN/model_${CKPT}.pt" >&2
    exit 2
fi

SEEDS=(0 1 2)
CONDITIONS=(nominal ee-disturbance combined-disturbance)
FOLDERS=(nominal ee_only_in combine_in)
COLLECTION=0
for INDEX in "${!CONDITIONS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        COLLECTION=$((COLLECTION + 1))
        echo "[Collection $COLLECTION/9] b1z1_pact_ab5_coupled_concat: ${CONDITIONS[$INDEX]}, seed=$SEED; 100 episodes, 10 robots"
        bash "$HERE/eval_b1z1_exp1.sh" \
            --task b1z1_pact_ab5_coupled_concat \
            --run "$RUN" \
            --ckpt "$CKPT" \
            --seed "$SEED" \
            --num-envs 10 \
            --episodes 100 \
            --horizon-seconds 15 \
            --condition "${CONDITIONS[$INDEX]}" \
            --load-range in \
            --motion-mode mixed \
            --gpu cuda:0 \
            --headless \
            --output-dir "$OUTPUT_BASE/${FOLDERS[$INDEX]}/seed_${SEED}"
    done
done

# From the repository root:
# conda activate lr_lab_cupiqp
# bash legged_gym/scripts/eval_b1z1_pact_ab5_coupled_concat_three_seeds.sh
