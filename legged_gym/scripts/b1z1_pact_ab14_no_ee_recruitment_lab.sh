#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
# Use the active Isaac Lab Python environment; retain scheduler GPU visibility.
export SIMULATOR=isaaclab_b1z1_pact
python -u train.py --task=b1z1_pact_ab14_no_ee_recruitment --seed=1 --gpu=cuda:0 --headless "$@"
