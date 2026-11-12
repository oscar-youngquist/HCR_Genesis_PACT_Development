#!/usr/bin/env bash
set -euo pipefail
# Shared collection settings for new ablations; no default old-policy checkpoint.
HERE="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO="$(CDPATH= cd -- "$HERE/../.." && pwd)"
TASK="${1:?Expected task}"
shift
case "$TASK" in
    b1z1_pact_ab13_workspace|b1z1_pact_ab14_no_ee_recruitment|b1z1_pact_ab15_no_channel_physics) ;;
    *) echo "Unsupported new ablation: $TASK" >&2; exit 2 ;;
esac
if [[ $# -lt 2 || $# -gt 3 || ! "$2" =~ ^[0-9]+$ ]]; then
    echo "Usage: bash $0 TASK RUN_DIRECTORY CHECKPOINT_NUMBER [OUTPUT_BASE]" >&2
    exit 2
fi
RUN="$1"
CKPT="$2"
OUTPUT_BASE="${3:-$REPO/evaluation/exp1/$TASK}"
if [[ ! -f "$RUN/model_${CKPT}.pt" ]]; then
    echo "Checkpoint not found: $RUN/model_${CKPT}.pt" >&2; exit 2
fi
SEEDS=(0 1 2)
CONDITIONS=(nominal ee-disturbance combined-disturbance)
FOLDERS=(nominal ee_only_in combine_in)
COLLECTION=0
for INDEX in "${!CONDITIONS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        COLLECTION=$((COLLECTION + 1))
        echo "[Collection $COLLECTION/18] $TASK: ${CONDITIONS[$INDEX]}, seed=$SEED"
        bash "$HERE/eval_b1z1_exp1.sh" --task "$TASK" --run "$RUN" --ckpt "$CKPT" \
            --seed "$SEED" --num-envs 25 --episodes 100 --horizon-seconds 15 \
            --condition "${CONDITIONS[$INDEX]}" --load-range in --motion-mode mixed \
            --gpu "${EVAL_GPU:-cuda:0}" --headless \
            --output-dir "$OUTPUT_BASE/${FOLDERS[$INDEX]}/seed_${SEED}"
    done
done
EXP2_OUTPUT_BASE="${EXP2_OUTPUT_BASE:-$REPO/evaluation/exp2/$TASK}"
for PROFILE in pushing pulling carrying; do
    for SEED in "${SEEDS[@]}"; do
        COLLECTION=$((COLLECTION + 1))
        echo "[Collection $COLLECTION/18] $TASK: $PROFILE, loaded, seed=$SEED"
        bash "$HERE/eval_b1z1_exp2.sh" --task "$TASK" --run "$RUN" --ckpt "$CKPT" \
            --seed "$SEED" --profile "$PROFILE" --loading loaded --num-envs 25 \
            --episodes 50 --horizon-seconds 28 --gpu "${EVAL_GPU:-cuda:0}" --headless \
            --output-dir "$EXP2_OUTPUT_BASE/$PROFILE/loaded/seed_${SEED}"
    done
done
