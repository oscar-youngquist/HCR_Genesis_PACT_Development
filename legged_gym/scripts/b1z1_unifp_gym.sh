#!/bin/bash

. /home/oyoungquist/anaconda3/etc/profile.d/conda.sh

conda activate /home/oyoungquist/.conda/envs/lr_gym

export SIMULATOR=isaacgym_b1z1_unifp

python train.py --task=b1z1_unifp --seed=1 --gpu=cuda:1 --headless "$@"
