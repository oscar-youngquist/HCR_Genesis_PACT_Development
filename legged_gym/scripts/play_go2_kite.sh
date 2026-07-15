#!/bin/bash

. /home/oyoungquist/anaconda3/etc/profile.d/conda.sh

conda activate /home/oyoungquist/.conda/envs/genesis_lr

export SIMULATOR=genesis_kite_depth
# python play_exp_kite.py --task=go2_kite --use_joystick --follow_robot --gpu=cuda:0

# Visualization/debug example:
python play_exp_kite.py --task=go2_kite --use_joystick --follow_robot --gpu=cuda:0 \
  --render-depth-image --render-height-field --render-surface-normals --debug-robot-id=0 \
  --normal-length=0.12 --normal-refresh-steps=5 --viz-height-offset=0.02

# export SIMULATOR=genesis
# python play.py --task=go2_dreamwaq
