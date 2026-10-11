#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
. /home/oyoungquist/anaconda3/etc/profile.d/conda.sh
conda activate /home/oyoungquist/.conda/envs/lr_lab_cupiqp
unset CUDA_VISIBLE_DEVICES
export SIMULATOR=isaaclab_b1z1_pact
python -u train.py --task=b1z1_pact --seed=1 --gpu=cuda:1 --headless "$@"
