#!/bin/bash
# set -e

. /home/oyoungquist/anaconda3/etc/profile.d/conda.sh
conda activate /home/oyoungquist/.conda/envs/genesis_lr

export SIMULATOR=genesis_b1_unifp
python train.py --task=b1_unifp --seed=1 --gpu=cuda:0 --headless
