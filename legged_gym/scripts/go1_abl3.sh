#!/bin/bash

###
#   Bizon
###
# . /home/oyoungquist/anaconda3/etc/profile.d/conda.sh

# conda activate /home/oyoungquist/.conda/envs/genesis_lr

###
#   Omen
###
. /home/oscaryoungquist/anaconda3/etc/profile.d/conda.sh

conda activate /home/oscaryoungquist/anaconda3/envs/genesis_lr

export SIMULATOR=genesis_pact_nopinn

python train.py --task=go1_abl3 --headless --seed=1
