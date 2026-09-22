#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
# The container supplies Python; preserve Slurm's assigned GPU visibility.
export SIMULATOR=isaaclab_b1z1_unifp
python -u train.py --task=b1z1_unifp --seed=1 --gpu=cuda:0 --headless "$@"
