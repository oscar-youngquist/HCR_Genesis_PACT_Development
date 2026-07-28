#!/bin/bash

# Standalone Genesis B1/Z1 PACT training entrypoint.
. /home/oyoungquist/anaconda3/etc/profile.d/conda.sh
conda activate /home/oyoungquist/.conda/envs/genesis_lr

export SIMULATOR=genesis_b1z1_pact
python train.py --task=b1z1_pact --seed=1 --gpu=cuda:0
