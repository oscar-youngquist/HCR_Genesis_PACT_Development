#!/bin/bash

. /home/oscaryoungquist/anaconda3/etc/profile.d/conda.sh

conda activate /home/oscaryoungquist/.conda/envs/genesis_lr

export SIMULATOR=genesis_pact_postau

python train.py --task=go1_pos --headless --seed=1
