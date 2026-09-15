#!/usr/bin/env bash
# Offline CSV generation for the paired collections from play_reconstruction_eval.sh.
set -euo pipefail

if [[ ${1:-} == --help ]]; then
  cat <<'EOF'
Usage: bash legged_gym/scripts/analyze_reconstruction_eval.sh [RUN_DIR] [OUTPUT_DIR]

Defaults: RUN_DIR=eval/reconstruction_comparison, OUTPUT_DIR=RUN_DIR/analysis
Uses genesis_lr on CPU. Generates controlled reconstruction, GRF, dynamics and
momentum CSVs plus plots. Re-running replaces reports in OUTPUT_DIR.

Optional environment settings:
  CONDA_ENV=genesis_lr BOOTSTRAP=1000 ANALYSIS_SEED=1
  RUN_DYNAMICS=1 RUN_PROBES=0 PLOTS=1       Toggle with 0 or 1
  CONTACT_THRESHOLD=1 TRANSITION_WINDOW=.03 RECOVERY_WINDOW=.75
  MOMENTUM_WINDOWS=".05 .1"
  PROGRESS_INTERVAL=15         Heartbeat interval, seconds

RUN_PROBES=1 adds frozen ridge-probe CSVs; at least five independent scenario
families are required. Relative paths are interpreted from the repository root.
EOF
  exit 0
fi
if (( $# > 2 )); then
  echo "Expected RUN_DIR and optional OUTPUT_DIR; see --help." >&2
  exit 2
fi

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO_ROOT"
RUN_DIR=${1:-eval/reconstruction_comparison}
OUTPUT_DIR=${2:-$RUN_DIR/analysis}
for source in pact abl3; do
  if [[ ! -f "$RUN_DIR/$source/manifest.json" ]]; then
    echo "Missing $RUN_DIR/$source/manifest.json. Run the collection script first." >&2
    exit 1
  fi
done
options=()
for setting in "${RUN_DYNAMICS:-1}" "${RUN_PROBES:-0}" "${PLOTS:-1}"; do
  if [[ "$setting" != 0 && "$setting" != 1 ]]; then
    echo "RUN_DYNAMICS, RUN_PROBES and PLOTS must be 0 or 1." >&2
    exit 2
  fi
done
if [[ ${RUN_DYNAMICS:-1} == 1 ]]; then options+=(--dynamics); fi
if [[ ${RUN_PROBES:-0} == 1 ]]; then options+=(--probes); fi
if [[ ${PLOTS:-1} == 1 ]]; then options+=(--plots); fi
read -r -a momentum_windows <<< "${MOMENTUM_WINDOWS:-.05 .1}"
export CUDA_VISIBLE_DEVICES=""
export MPLBACKEND=Agg
conda run -n "${CONDA_ENV:-genesis_lr}" --no-capture-output python -m reconstruction_eval.analysis \
  "$RUN_DIR/pact" "$RUN_DIR/abl3" --output "$OUTPUT_DIR" --controlled \
  --bootstrap "${BOOTSTRAP:-1000}" --seed "${ANALYSIS_SEED:-1}" \
  --progress-interval "${PROGRESS_INTERVAL:-15}" \
  --contact-threshold "${CONTACT_THRESHOLD:-1}" --transition-window "${TRANSITION_WINDOW:-.03}" \
  --recovery-window "${RECOVERY_WINDOW:-.75}" --momentum-windows "${momentum_windows[@]}" \
  "${options[@]}"
printf 'Analysis CSVs saved to %s\n' "$OUTPUT_DIR"
