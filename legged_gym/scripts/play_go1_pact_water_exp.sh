#!/bin/bash

. /home/oyoungquist/anaconda3/etc/profile.d/conda.sh

conda activate /home/oyoungquist/.conda/envs/genesis_lr

export SIMULATOR=genesis_pact_water
python play_exp_water.py --task=go1_pact_water --use_joystick --follow_robot --record_frames --gpu=cuda:1

# export SIMULATOR=genesis
# python play.py --task=go2_dreamwaq