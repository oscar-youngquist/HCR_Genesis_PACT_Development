#!/bin/bash

. /home/oyoungquist/anaconda3/etc/profile.d/conda.sh

conda activate /home/oyoungquist/.conda/envs/genesis_lr

export SIMULATOR=genesis_pact_rl2ac
python play_exp.py --task=go1_rl2ac --gpu=cuda:0 --use_joystick --follow_robot

# export SIMULATOR=genesis
# python play.py --task=go2_dreamwaq
