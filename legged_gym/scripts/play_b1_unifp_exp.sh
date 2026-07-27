#!/bin/bash

. /home/oyoungquist/anaconda3/etc/profile.d/conda.sh

conda activate /home/oyoungquist/.conda/envs/genesis_lr

export SIMULATOR=genesis_b1_unifp

python play_exp_b1_unifp.py --task=b1_unifp --seed=1 --gpu=cuda:0 \
  --follow_robot --terrain=rough --command_mode=random \
  --no-enable_torso_force_streams --no-apply_base_external_forces "$@"

# Examples:
#   Select a run and checkpoint:
#   ./play_b1_unifp_exp.sh --load_run=RUN_NAME --ckpt=CHECKPOINT
#
#   Evaluate a fixed forward command on flat ground:
#   ./play_b1_unifp_exp.sh --terrain=plane --command_mode=fixed --cmd_x=0.4
#
#   Enable UniFP torso force-command and external-disturbance streams:
#   ./play_b1_unifp_exp.sh --enable_torso_force_streams --apply_base_external_forces
#
#   Record the evaluation:
#   ./play_b1_unifp_exp.sh --record_frames --video_filename=b1_unifp_eval.mp4
