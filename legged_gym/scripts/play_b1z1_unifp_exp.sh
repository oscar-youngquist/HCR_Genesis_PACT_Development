#!/bin/bash

. /home/oyoungquist/anaconda3/etc/profile.d/conda.sh

conda activate /home/oyoungquist/.conda/envs/genesis_lr

export SIMULATOR=genesis_b1z1_unifp

python play_exp_unifp.py --task=b1z1_unifp --seed=1 --gpu=cuda:0 \
  --follow_robot --render_ee_goal_debug --render_ee_frame_debug \
  --no-apply_ee_external_forces --no-apply_base_external_forces --no-use_unifp_impedance_controller \
  --ee_eval_mode=random_sphere "$@"

# Examples:
#   Disable physical external disturbances:
#   ./play_b1z1_unifp_exp.sh --no-apply_ee_external_forces --no-apply_base_external_forces
#
#   Disable estimator-fed UniFP impedance compensation:
#   ./play_b1z1_unifp_exp.sh --no-use_unifp_impedance_controller
