#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")"
export SIMULATOR="${SIMULATOR:-isaaclab_b1z1_unifp}"
exec conda run --no-capture-output -n "${CONDA_ENV:-lr_lab_cupiqp}" python -u train.py --task=b1z1_unifp_original --seed=1 --gpu=cuda:1 --headless "$@"
