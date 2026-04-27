#!/bin/bash

. /home/oyoungquist/anaconda3/etc/profile.d/conda.sh

conda activate /home/oyoungquist/.conda/envs/genesis_lr

export SIMULATOR=genesis_pact_rl2ac

python train.py --task=go1_rl2ac --headless --seed=1 --gpu=cuda:1
