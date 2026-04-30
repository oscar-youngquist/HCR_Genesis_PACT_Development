#!/bin/bash

. /home/oyoungquist/anaconda3/etc/profile.d/conda.sh

conda activate /home/oyoungquist/.conda/envs/genesis_lr

export SIMULATOR=genesis_pact_postau
python play_exp.py --task=go1_tau --headless --gpu=cuda:1

# export SIMULATOR=genesis
# python play.py --task=go2_dreamwaq --use_joystick

