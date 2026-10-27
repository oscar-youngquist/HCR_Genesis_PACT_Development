#!/usr/bin/env bash
set -euo pipefail

# Activate your Isaac Lab environment before running this launcher.
# Edit these defaults, or supply RUN_DIRECTORY CHECKPOINT [OUTPUT_ROOT].
RUN="Oct21_00-46-21_b1z1_pact_improved"  # Trained run directory or registered run name.
CKPT="-1"  # Fixed checkpoint number or model file (not -1).
if [[ $# -gt 3 ]]; then
    echo "Usage: bash $0 RUN_DIRECTORY CHECKPOINT [OUTPUT_ROOT]" >&2
    exit 2
fi

HERE="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
RUN="${1:-$RUN}"
CKPT="${2:-$CKPT}"
# The optional third argument is a parent directory, never a condition directory.
OUTPUT_BASE="${3:-evaluation/exp1/pact}"
OUTPUT_ROOT="$OUTPUT_BASE/nominal"

# Explicit evaluation seeds; all use the same trained checkpoint.
SEEDS=(0 1 2)

# Require a fixed checkpoint to avoid changes between seed runs.
if [[ -z "$RUN" || -z "$CKPT" || "$CKPT" == "-1" ]]; then
    echo "Please specify a fixed checkpoint number or model file, not -1." >&2
    exit 2
fi

for SEED in "${SEEDS[@]}"; do
    echo "Evaluating B1Z1 PACT Nominal: seed=$SEED, episodes=100, robots=10"
    bash "$HERE/eval_b1z1_exp1.sh" \
        --task b1z1_pact \
        --run "$RUN" \
        --ckpt "$CKPT" \
        --seed "$SEED" \
        --num-envs 10 \
        --episodes 100 \
        --horizon-seconds 12 \
        --condition nominal \
        --load-range in \
        --motion-mode mixed \
        --gpu cuda:0 \
        --headless \
        --output-dir "$OUTPUT_ROOT/seed_${SEED}"
done

OUTPUT_ROOT="$OUTPUT_BASE/ee_only_in"
for SEED in "${SEEDS[@]}"; do
    echo "Evaluating B1Z1 PACT EE-Only-In: seed=$SEED, episodes=100, robots=10"
    bash "$HERE/eval_b1z1_exp1.sh" \
        --task b1z1_pact \
        --run "$RUN" \
        --ckpt "$CKPT" \
        --seed "$SEED" \
        --num-envs 10 \
        --episodes 100 \
        --horizon-seconds 12 \
        --condition ee-disturbance \
        --load-range in \
        --motion-mode mixed \
        --gpu cuda:0 \
        --headless \
        --output-dir "$OUTPUT_ROOT/seed_${SEED}"
done

OUTPUT_ROOT="$OUTPUT_BASE/combine_in"
for SEED in "${SEEDS[@]}"; do
    echo "Evaluating B1Z1 PACT Combined-In: seed=$SEED, episodes=100, robots=10"
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


# conda activate lr_lab_cupiqp
# bash legged_gym/scripts/eval_b1z1_ppo_pos_three_seeds.sh

# conda activate lr_lab_cupiqp
# bash legged_gym/scripts/eval_b1z1_pact_three_seeds.sh \
#     "logs/b1z1/b1z1_pact_lab/Oct21_00-46-21_b1z1_pact_improved" \
#     -1
