#!/usr/bin/env bash
set -euo pipefail

# Activate Isaac Lab first. Defaults pin the latest local checkpoint as of 2026-10-27.
# Optional arguments: RUN_DIRECTORY CHECKPOINT_NUMBER OUTPUT_BASE
HERE="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO="$(CDPATH= cd -- "$HERE/../.." && pwd)"
RUN="${1:-$REPO/logs/b1z1/b1z1_ppo_pos/Oct23_16-58-40_position_only}"
CKPT="${2:-28000}"
OUTPUT_BASE="${3:-$REPO/evaluation/exp1/b1z1_ppo_pos}"
if [[ $# -gt 3 || ! "$CKPT" =~ ^[0-9]+$ ]]; then
    echo "Usage: bash $0 [RUN_DIRECTORY [CHECKPOINT_NUMBER [OUTPUT_BASE]]]" >&2
    exit 2
fi
if [[ ! -f "$RUN/model_${CKPT}.pt" ]]; then
    echo "Checkpoint not found: $RUN/model_${CKPT}.pt" >&2
    exit 2
fi

SEEDS=(0 1 2)

# CONDITIONS=(nominal ee-disturbance combined-disturbance)
# FOLDERS=(nominal ee_only_in combine_in)
# COLLECTION=0
# for INDEX in "${!CONDITIONS[@]}"; do
#     for SEED in "${SEEDS[@]}"; do
#         COLLECTION=$((COLLECTION + 1))
#         echo "[Collection $COLLECTION/9] b1z1_ppo_pos: ${CONDITIONS[$INDEX]}, seed=$SEED; 100 episodes, 10 robots"
#         bash "$HERE/eval_b1z1_exp1.sh" \
#             --task b1z1_ppo_pos \
#             --run "$RUN" \
#             --ckpt "$CKPT" \
#             --seed "$SEED" \
#             --num-envs 10 \
#             --episodes 100 \
#             --horizon-seconds 15 \
#             --condition "${CONDITIONS[$INDEX]}" \
#             --load-range in \
#             --motion-mode mixed \
#             --gpu cuda:0 \
#             --headless \
#             --output-dir "$OUTPUT_BASE/${FOLDERS[$INDEX]}/seed_${SEED}"
#     done
# done

# From the repository root:
# conda activate lr_lab_cupiqp
# bash legged_gym/scripts/eval_b1z1_ppo_pos_three_seeds.sh

# Experiment 2 follows all existing Experiment 1 collections.
# Override EXP2_OUTPUT_BASE in the shell to choose a separate recording directory.
EXP2_OUTPUT_BASE="${EXP2_OUTPUT_BASE:-evaluation/exp2/b1z1_ppo_pos}"
EXP2_PROFILES=(pushing pulling carrying)
EXP2_COLLECTION=0
for PROFILE in "${EXP2_PROFILES[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        EXP2_COLLECTION=$((EXP2_COLLECTION + 1))
        echo "[Experiment 2 $EXP2_COLLECTION/9] b1z1_ppo_pos: $PROFILE, loaded, seed=$SEED; 50 episodes, 10 robots"
        bash "$HERE/eval_b1z1_exp2.sh" \
            --task b1z1_ppo_pos \
            --run "$RUN" \
            --ckpt "$CKPT" \
            --seed "$SEED" \
            --profile "$PROFILE" \
            --loading loaded \
            --num-envs 10 \
            --episodes 50 \
            --horizon-seconds 28 \
            --gpu cuda:0 \
            --headless \
            --output-dir "$EXP2_OUTPUT_BASE/$PROFILE/loaded/seed_${SEED}"
    done
done
