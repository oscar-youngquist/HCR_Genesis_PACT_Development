#!/bin/bash
set -euo pipefail

. /home/oyoungquist/anaconda3/etc/profile.d/conda.sh
conda activate /home/oyoungquist/.conda/envs/genesis_lr

export SIMULATOR=genesis_b1z1_unifp

# Nominal evaluation defaults:
#   - fixed +0.4 m/s forward command
#   - smoothly interpolated random EE goals
#   - flat terrain
#   - no force tasks, physical disturbances, impedance compensation, or DR
python play_exp_unifp_nominal.py \
  --task=b1z1_unifp \
  --seed=1 \
  --gpu=cuda:0 \
  --follow_robot \
  --base_command_mode=fixed \
  --cmd_x=0.4 \
  --cmd_y=0.0 \
  --cmd_yaw=0.0 \
  --ee_eval_mode=random_sphere \
  --ee_transition_s=2.0 \
  --ee_command_hold_s=5.0 \
  "$@"

# Examples:
#   Test a specific checkpoint:
#     ./play_b1z1_unifp_nominal.sh --load_run=RUN_NAME --ckpt=CHECKPOINT
#
#   Test standing with a fixed EE target:
#     ./play_b1z1_unifp_nominal.sh --cmd_x=0.0 \
#       --ee_eval_mode=fixed_sphere --fixed_ee_sphere 0.55 0.0 0.0
#
#   Run the base-command sequence:
#     ./play_b1z1_unifp_nominal.sh --base_command_mode=scripted
