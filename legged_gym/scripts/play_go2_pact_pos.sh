#!/bin/bash

. /home/oyoungquist/anaconda3/etc/profile.d/conda.sh

conda activate /home/oyoungquist/.conda/envs/genesis_lr

export SIMULATOR=genesis_pact_pos
python play_exp.py --task=go2_pact_pos --headless

# export SIMULATOR=genesis
# python play.py --task=go2_dreamwaq
