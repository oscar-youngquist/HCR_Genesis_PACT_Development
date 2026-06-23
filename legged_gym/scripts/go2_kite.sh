#!/bin/bash

. /home/oyoungquist/anaconda3/etc/profile.d/conda.sh

conda activate /home/oyoungquist/.conda/envs/genesis_lr

export SIMULATOR=genesis_kite_depth

python train.py --task=go2_kite --seed=1 --gpu=cuda:0 --headless --num_envs=3000
