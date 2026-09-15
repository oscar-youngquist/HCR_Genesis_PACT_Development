#!/usr/bin/env bash
# Collect both rollout sources sequentially; score both frozen models on each.
set -euo pipefail

CONDITION=${CONDITION:-impulse_com}
IMPULSE_MODE=${IMPULSE_MODE:-bounded_velocity}
PLANAR_VELOCITY_BOUND=${PLANAR_VELOCITY_BOUND:-2.0}
DOWNWARD_VELOCITY_BOUND=${DOWNWARD_VELOCITY_BOUND:-1.0}
ANGULAR_VELOCITY_BOUND=${ANGULAR_VELOCITY_BOUND:-2.0}
INJECTION_TIME=${INJECTION_TIME:-4}


if [[ ${1:-} == --help ]]; then
  cat <<'EOF'
Usage: bash legged_gym/scripts/play_reconstruction_eval.sh [RUN_DIR]

Default RUN_DIR: eval/reconstruction_comparison (must not already exist).
Runs PACT, then ABL3, on physical GPU 1 using conda environment genesis_lr.

Optional environment settings:
  CONDITION=impulse_com        nominal | payload | impulse | impulse_com
  TERRAIN=rough              Any collector terrain name
  NUM_EPS=10 NUM_ENVS=10 SCENARIO_SEED=42
  PAYLOAD_BOUNDS="4 10"        Added mass, kg
  COM_BOUNDS="-.25 .25 -.25 .25 -.15 .15"   x/y/z bounds, metres
  INJECTION_TIME=4 VELOCITY_DELTA="1.5 1.5 1.0 1.5 1.5 1.5"
  IMPULSE_MODE=bounded_velocity   fixed | bounded_velocity (requires impulse or impulse_com)
  PLANAR_VELOCITY_BOUND=2      Maximum planar increment magnitude, m/s
  DOWNWARD_VELOCITY_BOUND=1    Vertical increment sampled from [-bound, 0], m/s
  ANGULAR_VELOCITY_BOUND=2     Each angular increment sampled from [-bound, bound], rad/s
  PACT_CHECKPOINT / ABL3_CHECKPOINT         Override trained checkpoint paths
  PACT_CONFIG / ABL3_CONFIG                 Override original task config paths
  CONFIG_ROOT                 Original base-config checkout; defaults to repo
  CONDA_ENV=genesis_lr
  PROGRESS_INTERVAL=15         Heartbeat interval, seconds

Relative paths are interpreted from the repository root. Each rollout uses the
same settings and scenario seed. No analysis or training runs in this script.
impulse_com samples CoM independently each episode and holds it fixed; added
payload is zero. Plain impulse retains zero CoM offset.
EOF
  exit 0
fi
if (( $# > 1 )); then
  echo "Expected at most one RUN_DIR argument; see --help." >&2
  exit 2
fi

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO_ROOT"
RUN_DIR=${1:-eval/reconstruction_comparison}
CONDA_ENV=${CONDA_ENV:-genesis_lr}
PACT_CHECKPOINT=${PACT_CHECKPOINT:-logs/pact_corl/go1_pact_rough/Sep14_18-03-36_pact_100hz_spec_smartcurr/model_8000.pt}
ABL3_CHECKPOINT=${ABL3_CHECKPOINT:-logs/pact_corl/go1_abl3_rough/Sep11_00-04-14_hybrid_100hz_spec_materr/model_6000.pt}
PACT_CONFIG=${PACT_CONFIG:-$(dirname -- "$PACT_CHECKPOINT")/go1_pact_config.py}
ABL3_CONFIG=${ABL3_CONFIG:-$(dirname -- "$ABL3_CHECKPOINT")/go1_abl3_config.py}

if [[ -e "$RUN_DIR" ]]; then
  echo "Output already exists: $RUN_DIR. Choose a new RUN_DIR." >&2
  exit 1
fi
for input in "$PACT_CHECKPOINT" "$ABL3_CHECKPOINT" "$PACT_CONFIG" "$ABL3_CONFIG"; do
  if [[ ! -f "$input" ]]; then
    echo "A required model/config file is missing; check the configured input paths." >&2
    exit 1
  fi
done
command -v conda >/dev/null
mkdir -p -- "$RUN_DIR"
RUN_DIR=$(cd -- "$RUN_DIR" && pwd)
# Physical GPU 1 becomes local cuda:0; never expose the training GPU 0.
export CUDA_VISIBLE_DEVICES=1
PYTHON=(conda run -n "$CONDA_ENV" --no-capture-output python)
# Loading settings are temporary inputs, not part of the evaluation logs.
MODEL_SETTINGS=$(mktemp)
trap 'rm -f -- "$MODEL_SETTINGS"' EXIT

"${PYTHON[@]}" -m reconstruction_eval.collect metadata --task go1_pact \
  --config-root "${CONFIG_ROOT:-$REPO_ROOT}" --training-config "$PACT_CONFIG" --output "$RUN_DIR/pact.json"
"${PYTHON[@]}" -m reconstruction_eval.collect metadata --task go1_abl3 \
  --config-root "${CONFIG_ROOT:-$REPO_ROOT}" --training-config "$ABL3_CONFIG" --output "$RUN_DIR/abl3.json"
"${PYTHON[@]}" -c '
import json, sys, uuid
from pathlib import Path
from reconstruction_eval.io import sha256
root = Path(sys.argv[1])
comparison_id = str(uuid.uuid4())
models = [dict(method=m, checkpoint=str(Path(p).resolve()), metadata=str(root / (m + ".json")),
               comparison_id=comparison_id, expected_sha256=sha256(p))
          for m, p in zip(("pact", "abl3"), sys.argv[2:4])]
Path(sys.argv[4]).write_text(json.dumps(models, indent=2) + "\n")
' "$RUN_DIR" "$PACT_CHECKPOINT" "$ABL3_CHECKPOINT" "$MODEL_SETTINGS"

read -r -a payload_bounds <<< "${PAYLOAD_BOUNDS:-4 10}"
read -r -a com_bounds <<< "${COM_BOUNDS:--.25 .25 -.25 .25 -.15 .15}"
read -r -a velocity_delta <<< "${VELOCITY_DELTA:-1.5 1.5 1.0 1.5 1.5 1.5}"
case "${IMPULSE_MODE:-fixed}" in
  fixed)
    disturbance_options=(--velocity-delta "${velocity_delta[@]}")
    ;;
  bounded_velocity)
    disturbance_options=(--planar-velocity-bound "${PLANAR_VELOCITY_BOUND:-2}"
      --downward-velocity-bound "${DOWNWARD_VELOCITY_BOUND:-1}"
      --angular-velocity-bound "${ANGULAR_VELOCITY_BOUND:-2}")
    ;;
  *)
    echo "IMPULSE_MODE must be fixed or bounded_velocity." >&2
    exit 2
    ;;
esac
for source in pact abl3; do
  backend=genesis_pact
  if [[ "$source" == abl3 ]]; then backend=genesis_pact_nopinn; fi
  echo "Collecting $source on physical GPU 1 -> $RUN_DIR/$source"
  SIMULATOR="$backend" "${PYTHON[@]}" legged_gym/scripts/play_exp_decoder_eval.py collect \
    --models "$MODEL_SETTINGS" --rollout-source "$source" \
    --condition "${CONDITION:-payload}" --terrain-type "${TERRAIN:-rough}" \
    --payload-bounds "${payload_bounds[@]}" --com-bounds "${com_bounds[@]}" \
    --injection-time "${INJECTION_TIME:-2}" "${disturbance_options[@]}" \
    --num-eps "${NUM_EPS:-10}" --num-envs "${NUM_ENVS:-10}" --seed "${SCENARIO_SEED:-42}" \
    --progress-interval "${PROGRESS_INTERVAL:-15}" \
    --headless --gpu cuda:0 --output "$RUN_DIR/$source" \
    2>&1 | tee "$RUN_DIR/$source.log"
done
printf 'Both collections finished. Generate CSVs with:\n  bash legged_gym/scripts/analyze_reconstruction_eval.sh %q\n' "$RUN_DIR"
