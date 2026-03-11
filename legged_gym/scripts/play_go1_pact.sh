#!/bin/bash

. /home/oyoungquist/anaconda3/etc/profile.d/conda.sh

conda activate /home/oyoungquist/.conda/envs/genesis_lr

export SIMULATOR=genesis_pact
python play_exp.py --task=go1_pact

# export SIMULATOR=genesis
# python play.py --task=go2_dreamwaq

