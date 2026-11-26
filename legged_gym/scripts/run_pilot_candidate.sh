#!/usr/bin/env bash
# Run from the active Isaac Lab Python environment; no checkpoints overwritten.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
GPU="${GPU:-cuda:1}"
OUT="${OUT:-analysis/pilot_candidate/new_experiment}"
ITERATIONS="${ITERATIONS:-1000}"
EPISODES="${EPISODES:-20}"
export SIMULATOR=isaaclab_b1z1_pact PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-1}"
ARGS=(--candidate --gpu "$GPU" --headless --num-envs 64 --seeds 0 1 2
      --episodes-per-condition "$EPISODES" --calibration-episodes "$EPISODES"
      --conditions nominal up_yield_up_force up_yield_down_force
      down_yield_up_force down_yield_down_force ample_base_on ample_base_off
      restricted_base_on restricted_base_off)
python legged_gym/scripts/eval_b1z1_pilot_compliance.py "${ARGS[@]}" --output-dir "$OUT/start"
test -f "$OUT/start/COMPLETE"
python legged_gym/scripts/train_pilot_continuation.py --gpu "$GPU" --num-envs 4096 --iterations "$ITERATIONS" --output-dir "$OUT/training"
test -f "$OUT/training/COMPLETE"
python legged_gym/scripts/eval_b1z1_pilot_compliance.py "${ARGS[@]}" --run "$OUT/training" --output-dir "$OUT/final"
test -f "$OUT/final/COMPLETE"
for PHASE in start final; do
    python legged_gym/scripts/analyze_pilot_candidate.py "$OUT/$PHASE"
    python legged_gym/scripts/plot_b1z1_pilot_compliance.py "$OUT/$PHASE"
    python legged_gym/scripts/analyze_pilot_reference_stages.py "$OUT/$PHASE"
done
