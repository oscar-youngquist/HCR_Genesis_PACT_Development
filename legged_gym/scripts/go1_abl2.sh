#!/bin/bash

. /home/oyoungquist/anaconda3/etc/profile.d/conda.sh

conda activate /home/oyoungquist/.conda/envs/genesis_lr

export SIMULATOR=genesis_pact_nopinn

python train.py --task=go1_abl2 --headless --gpu=cuda:0 --seed=1
