#!/usr/bin/env bash
set -euo pipefail

# Activate Isaac Lab first. Edit these paths before running.
RUN="Oct23_17-51-33_b1z1_pact_ab10_actor"   # Trained run directory or registered run name.
CKPT="27000"  # Fixed checkpoint number or model file (not -1).
OUTPUT_BASE="evaluation/exp1/b1z1_pact_ab10_actor"

# Explicit evaluation seeds; 100 episodes per seed across 25 robots on cuda:0.
SEEDS=(0 1 2)
HERE="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
if [[ -z "$RUN" || -z "$CKPT" || "$CKPT" == "-1" ]]; then
    echo "Set RUN and a fixed CKPT at the top of this script." >&2
    exit 2
fi

CONDITIONS=(nominal ee-disturbance combined-disturbance)
FOLDERS=(nominal ee_only_in combine_in)
COLLECTION=0
for INDEX in "${!CONDITIONS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        COLLECTION=$((COLLECTION + 1))
        echo "[Collection $COLLECTION/9] b1z1_pact_ab10_actor: ${CONDITIONS[$INDEX]}, seed=$SEED; 100 episodes, 25 robots"
        bash "$HERE/eval_b1z1_exp1.sh" \
            --task b1z1_pact_ab10_actor \
            --run "$RUN" \
            --ckpt "$CKPT" \
            --seed "$SEED" \
            --num-envs 25 \
            --episodes 100 \
            --horizon-seconds 15 \
            --condition "${CONDITIONS[$INDEX]}" \
            --load-range in \
            --motion-mode mixed \
            --gpu "${EVAL_GPU:-cuda:0}" \
            --headless \
            --output-dir "$OUTPUT_BASE/${FOLDERS[$INDEX]}/seed_${SEED}"
    done
done

# From the repository root:
# conda activate lr_lab_cupiqp
# bash legged_gym/scripts/eval_b1z1_pact_ab10_actor_three_seeds.sh

# Experiment 2 follows all existing Experiment 1 collections.
# Override EXP2_OUTPUT_BASE in the shell to choose a separate recording directory.
EXP2_OUTPUT_BASE="${EXP2_OUTPUT_BASE:-$HERE/../../evaluation/exp2/b1z1_pact_ab10_actor}"
EXP2_PROFILES=(pushing pulling carrying)
EXP2_COLLECTION=0
for PROFILE in "${EXP2_PROFILES[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        EXP2_COLLECTION=$((EXP2_COLLECTION + 1))
        echo "[Experiment 2 $EXP2_COLLECTION/9] b1z1_pact_ab10_actor: $PROFILE, loaded, seed=$SEED; 50 episodes, 25 robots"
        bash "$HERE/eval_b1z1_exp2.sh" \
            --task b1z1_pact_ab10_actor \
            --run "$RUN" \
            --ckpt "$CKPT" \
            --seed "$SEED" \
            --profile "$PROFILE" \
            --loading loaded \
            --num-envs 25 \
            --episodes 50 \
            --horizon-seconds 28 \
            --gpu "${EVAL_GPU:-cuda:0}" \
            --headless \
            --output-dir "$EXP2_OUTPUT_BASE/$PROFILE/loaded/seed_${SEED}"
    done
done
