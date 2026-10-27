#!/usr/bin/env bash
set -euo pipefail

# Activate Isaac Lab first. Pin one checkpoint for every seed and experiment.
# Optional arguments: RUN_DIRECTORY CHECKPOINT_NUMBER OUTPUT_BASE
HERE="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO="$(CDPATH= cd -- "$HERE/../.." && pwd)"
RUN="${1:-$REPO/logs/b1z1/b1z1_unifp_original/Oct23_15-25-20_upstream_architecture}"
CKPT="${2:-70000}"
OUTPUT_BASE="${3:-$REPO/evaluation/exp1/b1z1_unifp_original}"
if [[ $# -gt 3 || ! "$CKPT" =~ ^[0-9]+$ ]]; then
    echo "Usage: bash $0 [RUN_DIRECTORY [CHECKPOINT_NUMBER [OUTPUT_BASE]]]" >&2
    exit 2
fi
if [[ ! -f "$RUN/model_${CKPT}.pt" ]]; then
    echo "Checkpoint not found: $RUN/model_${CKPT}.pt" >&2
    exit 2
fi
OUTPUT_ROOT="$OUTPUT_BASE/nominal"


# Explicit evaluation seeds; 100 episodes per seed across 10 robots on cuda:0.
SEEDS=(0 1 2)

for SEED in "${SEEDS[@]}"; do
    echo "Evaluating b1z1_unifp_original: seed=$SEED, episodes=100, robots=10"
    bash "$HERE/eval_b1z1_exp1.sh" \
        --task b1z1_unifp_original \
        --run "$RUN" \
        --ckpt "$CKPT" \
        --seed "$SEED" \
        --num-envs 10 \
        --episodes 100 \
        --horizon-seconds 15 \
        --condition nominal \
        --load-range in \
        --motion-mode mixed \
        --gpu "${EVAL_GPU:-cuda:0}" \
        --headless \
        --output-dir "$OUTPUT_ROOT/seed_${SEED}"
done

OUTPUT_ROOT="$OUTPUT_BASE/ee_only_in"
for SEED in "${SEEDS[@]}"; do
    echo "Evaluating b1z1_unifp_original: seed=$SEED, episodes=100, robots=10"
    bash "$HERE/eval_b1z1_exp1.sh" \
        --task b1z1_unifp_original \
        --run "$RUN" \
        --ckpt "$CKPT" \
        --seed "$SEED" \
        --num-envs 10 \
        --episodes 100 \
        --horizon-seconds 15 \
        --condition ee-disturbance \
        --load-range in \
        --motion-mode mixed \
        --gpu "${EVAL_GPU:-cuda:0}" \
        --headless \
        --output-dir "$OUTPUT_ROOT/seed_${SEED}"
done

OUTPUT_ROOT="$OUTPUT_BASE/combine_in"
for SEED in "${SEEDS[@]}"; do
    echo "Evaluating b1z1_unifp_original: seed=$SEED, episodes=100, robots=10"
    bash "$HERE/eval_b1z1_exp1.sh" \
        --task b1z1_unifp_original \
        --run "$RUN" \
        --ckpt "$CKPT" \
        --seed "$SEED" \
        --num-envs 10 \
        --episodes 100 \
        --horizon-seconds 15 \
        --condition combined-disturbance \
        --load-range in \
        --motion-mode mixed \
        --gpu "${EVAL_GPU:-cuda:0}" \
        --headless \
        --output-dir "$OUTPUT_ROOT/seed_${SEED}"
done

# conda activate lr_lab_cupiqp
# bash legged_gym/scripts/eval_b1z1_unifp_original_three_seeds.sh \
#     "logs/b1z1/b1z1_unifp_original/Oct23_15-25-20_upstream_architecture" \
#     70000

# Experiment 2 follows all existing Experiment 1 collections.
# Override EXP2_OUTPUT_BASE in the shell to choose a separate recording directory.
EXP2_OUTPUT_BASE="${EXP2_OUTPUT_BASE:-$REPO/evaluation/exp2/b1z1_unifp_original}"
EXP2_PROFILES=(pushing pulling carrying)
EXP2_COLLECTION=0
for PROFILE in "${EXP2_PROFILES[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        EXP2_COLLECTION=$((EXP2_COLLECTION + 1))
        echo "[Experiment 2 $EXP2_COLLECTION/9] b1z1_unifp_original: $PROFILE, loaded, seed=$SEED; 50 episodes, 10 robots"
        bash "$HERE/eval_b1z1_exp2.sh" \
            --task b1z1_unifp_original \
            --run "$RUN" \
            --ckpt "$CKPT" \
            --seed "$SEED" \
            --profile "$PROFILE" \
            --loading loaded \
            --num-envs 10 \
            --episodes 50 \
            --horizon-seconds 28 \
            --gpu "${EVAL_GPU:-cuda:0}" \
            --headless \
            --output-dir "$EXP2_OUTPUT_BASE/$PROFILE/loaded/seed_${SEED}"
    done
done
