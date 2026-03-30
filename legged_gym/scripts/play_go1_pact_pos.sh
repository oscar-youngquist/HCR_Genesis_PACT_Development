#!/bin/bash

. /home/oyoungquist/anaconda3/etc/profile.d/conda.sh

conda activate /home/oyoungquist/.conda/envs/genesis_lr

export SIMULATOR=genesis_pact_pos
python play.py --task=go1_pact_pos --use_joystick

# export SIMULATOR=genesis
# python play.py --task=go2_dreamwaq
