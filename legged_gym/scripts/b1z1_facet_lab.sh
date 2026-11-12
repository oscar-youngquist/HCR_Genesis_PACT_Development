#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
. "${FACET_CONDA_SH:-/home/oyoungquist/anaconda3/etc/profile.d/conda.sh}"
conda activate "${FACET_CONDA_ENV:-/home/oyoungquist/.conda/envs/lr_lab_cupiqp}"
unset CUDA_VISIBLE_DEVICES
export SIMULATOR=isaaclab_b1z1_pact
# Match B1Z1 launchers: forwarded --seed overrides this default (also on resume).
exec python -u train_b1z1_facet.py --seed=1 --gpu=cuda:0 --headless "$@"

# bash legged_gym/scripts/b1z1_facet_lab.sh --seed 1
