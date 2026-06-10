#!/bin/bash

. /home/oyoungquist/anaconda3/etc/profile.d/conda.sh

conda activate /home/oyoungquist/.conda/envs/genesis_lr

export SIMULATOR=genesis_pact_nopinn

# # No disturbance conditions
# python play_exp_largescale.py --task=go1_abl1 --gpu=cuda:1 --headless --seed=1  --num_eps=2.00 --num_envs=100 --terrain_type=plane --disturbance_type=none --log --log_path=exp_data_corl_10/no_disturbance 

# python play_exp_largescale.py --task=go1_abl1 --gpu=cuda:1 --headless --seed=1  --num_eps=2.00 --num_envs=100 --terrain_type=slope --disturbance_type=none --log --log_path=exp_data_corl_10/no_disturbance  

# python play_exp_largescale.py --task=go1_abl1 --gpu=cuda:1 --headless --seed=1  --num_eps=2.00 --num_envs=100 --terrain_type=stairs --disturbance_type=none --log --log_path=exp_data_corl_10/no_disturbance  

# python play_exp_largescale.py --task=go1_abl1 --gpu=cuda:1 --headless --seed=1  --num_eps=2.00 --num_envs=100 --terrain_type=rough --disturbance_type=none --log --log_path=exp_data_corl_10/no_disturbance  

# python play_exp_largescale.py --task=go1_abl1 --gpu=cuda:1 --headless --seed=1  --num_eps=2.00 --num_envs=100 --terrain_type=discrete --disturbance_type=none --log --log_path=exp_data_corl_10/no_disturbance  

# python play_exp_largescale.py --task=go1_abl1 --gpu=cuda:1 --headless --seed=1  --num_eps=2.00 --num_envs=100 --terrain_type=wave --disturbance_type=none --log --log_path=exp_data_corl_10/no_disturbance 

# # payload with COM shift
# python play_exp_largescale.py --task=go1_abl1 --gpu=cuda:1 --headless --seed=1  --shift_com --num_eps=2.00 --num_envs=100 --terrain_type=plane --disturbance_type=payload --log --log_path=exp_data_corl_10/payload_disturbance 

# python play_exp_largescale.py --task=go1_abl1 --gpu=cuda:1 --headless --seed=1  --shift_com --num_eps=2.00 --num_envs=100 --terrain_type=slope --disturbance_type=payload --log --log_path=exp_data_corl_10/payload_disturbance 

# python play_exp_largescale.py --task=go1_abl1 --gpu=cuda:1 --headless --seed=1  --shift_com --num_eps=2.00 --num_envs=100 --terrain_type=stairs --disturbance_type=payload --log --log_path=exp_data_corl_10/payload_disturbance 

# python play_exp_largescale.py --task=go1_abl1 --gpu=cuda:1 --headless --seed=1  --shift_com --num_eps=2.00 --num_envs=100 --terrain_type=rough --disturbance_type=payload --log --log_path=exp_data_corl_10/payload_disturbance 

# python play_exp_largescale.py --task=go1_abl1 --gpu=cuda:1 --headless --seed=1  --shift_com --num_eps=2.00 --num_envs=100 --terrain_type=discrete --disturbance_type=payload --log --log_path=exp_data_corl_10/payload_disturbance 

# python play_exp_largescale.py --task=go1_abl1 --gpu=cuda:1 --headless --seed=1  --shift_com --num_eps=2.00 --num_envs=100 --terrain_type=wave --disturbance_type=payload --log --log_path=exp_data_corl_10/payload_disturbance 

# # pushes
# python play_exp_largescale.py --task=go1_abl1 --gpu=cuda:1 --headless --seed=1  --num_eps=2.00 --num_envs=100 --terrain_type=plane --disturbance_type=push --log --log_path=exp_data_corl_10/push_disturbance 

# python play_exp_largescale.py --task=go1_abl1 --gpu=cuda:1 --headless --seed=1  --num_eps=2.00 --num_envs=100 --terrain_type=slope --disturbance_type=push --log --log_path=exp_data_corl_10/push_disturbance 

# python play_exp_largescale.py --task=go1_abl1 --gpu=cuda:1 --headless --seed=1  --num_eps=2.00 --num_envs=100 --terrain_type=stairs --disturbance_type=push --log --log_path=exp_data_corl_10/push_disturbance 

# python play_exp_largescale.py --task=go1_abl1 --gpu=cuda:1 --headless --seed=1  --num_eps=2.00 --num_envs=100 --terrain_type=rough --disturbance_type=push --log --log_path=exp_data_corl_10/push_disturbance 

# python play_exp_largescale.py --task=go1_abl1 --gpu=cuda:1 --headless --seed=1  --num_eps=2.00 --num_envs=100 --terrain_type=discrete --disturbance_type=push --log --log_path=exp_data_corl_10/push_disturbance 

# python play_exp_largescale.py --task=go1_abl1 --gpu=cuda:1 --headless --seed=1  --num_eps=2.00 --num_envs=100 --terrain_type=wave --disturbance_type=push --log --log_path=exp_data_corl_10/push_disturbance 


###
#   Max payload test
###
# python play_eval_fixed_episodes.py --task=go1_abl1 --gpu=cuda:1  --headless --seed=1 --num_eps=100 --num_envs=10 --terrain_type=plane --disturbance_type=payload --payload_bounds 4.0 4.0 --log --log_path=exp_data_corl_10/payload_max

# python play_eval_fixed_episodes.py --task=go1_abl1 --gpu=cuda:1  --headless --seed=1 --num_eps=100 --num_envs=10 --terrain_type=plane --disturbance_type=payload --payload_bounds 5.0 5.0 --log --log_path=exp_data_corl_10/payload_max

# python play_eval_fixed_episodes.py --task=go1_abl1 --gpu=cuda:1  --headless --seed=1 --num_eps=100 --num_envs=10 --terrain_type=plane --disturbance_type=payload --payload_bounds 6.0 6.0 --log --log_path=exp_data_corl_10/payload_max

# python play_eval_fixed_episodes.py --task=go1_abl1 --gpu=cuda:1  --headless --seed=1 --num_eps=100 --num_envs=10 --terrain_type=plane --disturbance_type=payload --payload_bounds 7.0 7.0 --log --log_path=exp_data_corl_10/payload_max

# python play_eval_fixed_episodes.py --task=go1_abl1 --gpu=cuda:1  --headless --seed=1 --num_eps=100 --num_envs=10 --terrain_type=plane --disturbance_type=payload --payload_bounds 8.0 8.0 --log --log_path=exp_data_corl_10/payload_max

# python play_eval_fixed_episodes.py --task=go1_abl1 --gpu=cuda:1  --headless --seed=1 --num_eps=100 --num_envs=10 --terrain_type=plane --disturbance_type=payload --payload_bounds 9.0 9.0 --log --log_path=exp_data_corl_10/payload_max

# python play_eval_fixed_episodes.py --task=go1_abl1 --gpu=cuda:1  --headless --seed=1 --num_eps=100 --num_envs=10 --terrain_type=plane --disturbance_type=payload --payload_bounds 10.0 10.0 --log --log_path=exp_data_corl_10/payload_max

# python play_eval_fixed_episodes.py --task=go1_abl1 --gpu=cuda:1  --headless --seed=1 --num_eps=100 --num_envs=10 --terrain_type=plane --disturbance_type=payload --payload_bounds 11.0 11.0 --log --log_path=exp_data_corl_10/payload_max

# python play_eval_fixed_episodes.py --task=go1_abl1 --gpu=cuda:1  --headless --seed=1 --num_eps=100 --num_envs=10 --terrain_type=plane --disturbance_type=payload --payload_bounds 12.0 12.0 --log --log_path=exp_data_corl_10/payload_max

# python play_eval_fixed_episodes.py --task=go1_abl1 --gpu=cuda:1  --headless --seed=1 --num_eps=100 --num_envs=10 --terrain_type=plane --disturbance_type=payload --payload_bounds 13.0 13.0 --log --log_path=exp_data_corl_10/payload_max

# python play_eval_fixed_episodes.py --task=go1_abl1 --gpu=cuda:1  --headless --seed=1 --num_eps=100 --num_envs=10 --terrain_type=plane --disturbance_type=payload --payload_bounds 14.0 14.0 --log --log_path=exp_data_corl_10/payload_max

# python play_eval_fixed_episodes.py --task=go1_abl1 --gpu=cuda:1  --headless --seed=1 --num_eps=100 --num_envs=10 --terrain_type=plane --disturbance_type=payload --payload_bounds 15.0 15.0 --log --log_path=exp_data_corl_10/payload_max

# python play_eval_fixed_episodes.py --task=go1_abl1 --gpu=cuda:1  --headless --seed=1 --num_eps=100 --num_envs=10 --terrain_type=plane --disturbance_type=payload --payload_bounds 16.0 16.0 --log --log_path=exp_data_corl_10/payload_max

# python play_eval_fixed_episodes.py --task=go1_abl1 --gpu=cuda:1  --headless --seed=1 --num_eps=100 --num_envs=10 --terrain_type=plane --disturbance_type=payload --payload_bounds 17.0 17.0 --log --log_path=exp_data_corl_10/payload_max

# python play_eval_fixed_episodes.py --task=go1_abl1 --gpu=cuda:1  --headless --seed=1 --num_eps=100 --num_envs=10 --terrain_type=plane --disturbance_type=payload --payload_bounds 18.0 18.0 --log --log_path=exp_data_corl_10/payload_max

# python play_eval_fixed_episodes.py --task=go1_abl1 --gpu=cuda:1  --headless --seed=1 --num_eps=100 --num_envs=10 --terrain_type=plane --disturbance_type=payload --payload_bounds 19.0 19.0 --log --log_path=exp_data_corl_10/payload_max

# python play_eval_fixed_episodes.py --task=go1_abl1 --gpu=cuda:1  --headless --seed=1 --num_eps=100 --num_envs=10 --terrain_type=plane --disturbance_type=payload --payload_bounds 20.0 20.0 --log --log_path=exp_data_corl_10/payload_max

# ###
# #   Max push test
# ###
# python play_eval_fixed_episodes.py --task=go1_abl1 --gpu=cuda:1  --headless --seed=1 --num_eps=100 --num_envs=10 --terrain_type=plane --disturbance_type=push --push_bounds 0.5000 0.1000 0.5000 --log --log_path=exp_data_corl_10/wrench_max

# python play_eval_fixed_episodes.py --task=go1_abl1 --gpu=cuda:1  --headless --seed=1 --num_eps=100 --num_envs=10 --terrain_type=plane --disturbance_type=push --push_bounds 0.6042 0.1792 0.6042 --log --log_path=exp_data_corl_10/wrench_max

# python play_eval_fixed_episodes.py --task=go1_abl1 --gpu=cuda:1  --headless --seed=1 --num_eps=100 --num_envs=10 --terrain_type=plane --disturbance_type=push --push_bounds 0.7083 0.2583 0.7083 --log --log_path=exp_data_corl_10/wrench_max

# python play_eval_fixed_episodes.py --task=go1_abl1 --gpu=cuda:1  --headless --seed=1 --num_eps=100 --num_envs=10 --terrain_type=plane --disturbance_type=push --push_bounds 0.8125 0.3375 0.8125 --log --log_path=exp_data_corl_10/wrench_max

# python play_eval_fixed_episodes.py --task=go1_abl1 --gpu=cuda:1  --headless --seed=1 --num_eps=100 --num_envs=10 --terrain_type=plane --disturbance_type=push --push_bounds 0.9167 0.4167 0.9167 --log --log_path=exp_data_corl_10/wrench_max

# python play_eval_fixed_episodes.py --task=go1_abl1 --gpu=cuda:1  --headless --seed=1 --num_eps=100 --num_envs=10 --terrain_type=plane --disturbance_type=push --push_bounds 1.0208 0.4958 1.0208 --log --log_path=exp_data_corl_10/wrench_max

# python play_eval_fixed_episodes.py --task=go1_abl1 --gpu=cuda:1  --headless --seed=1 --num_eps=100 --num_envs=10 --terrain_type=plane --disturbance_type=push --push_bounds 1.1250 0.5750 1.1250 --log --log_path=exp_data_corl_10/wrench_max

# python play_eval_fixed_episodes.py --task=go1_abl1 --gpu=cuda:1  --headless --seed=1 --num_eps=100 --num_envs=10 --terrain_type=plane --disturbance_type=push --push_bounds 1.2292 0.6542 1.2292 --log --log_path=exp_data_corl_10/wrench_max

# python play_eval_fixed_episodes.py --task=go1_abl1 --gpu=cuda:1  --headless --seed=1 --num_eps=100 --num_envs=10 --terrain_type=plane --disturbance_type=push --push_bounds 1.3333 0.7333 1.3333 --log --log_path=exp_data_corl_10/wrench_max

# python play_eval_fixed_episodes.py --task=go1_abl1 --gpu=cuda:1  --headless --seed=1 --num_eps=100 --num_envs=10 --terrain_type=plane --disturbance_type=push --push_bounds 1.4375 0.8125 1.4375 --log --log_path=exp_data_corl_10/wrench_max

# python play_eval_fixed_episodes.py --task=go1_abl1 --gpu=cuda:1  --headless --seed=1 --num_eps=100 --num_envs=10 --terrain_type=plane --disturbance_type=push --push_bounds 1.5417 0.8917 1.5417 --log --log_path=exp_data_corl_10/wrench_max

# python play_eval_fixed_episodes.py --task=go1_abl1 --gpu=cuda:1  --headless --seed=1 --num_eps=100 --num_envs=10 --terrain_type=plane --disturbance_type=push --push_bounds 1.6458 0.9708 1.6458 --log --log_path=exp_data_corl_10/wrench_max

# python play_eval_fixed_episodes.py --task=go1_abl1 --gpu=cuda:1  --headless --seed=1 --num_eps=100 --num_envs=10 --terrain_type=plane --disturbance_type=push --push_bounds 1.7500 1.0500 1.7500 --log --log_path=exp_data_corl_10/wrench_max

# python play_eval_fixed_episodes.py --task=go1_abl1 --gpu=cuda:1  --headless --seed=1 --num_eps=100 --num_envs=10 --terrain_type=plane --disturbance_type=push --push_bounds 1.8542 1.1292 1.8542 --log --log_path=exp_data_corl_10/wrench_max

# python play_eval_fixed_episodes.py --task=go1_abl1 --gpu=cuda:1  --headless --seed=1 --num_eps=100 --num_envs=10 --terrain_type=plane --disturbance_type=push --push_bounds 1.9583 1.2083 1.9583 --log --log_path=exp_data_corl_10/wrench_max

# python play_eval_fixed_episodes.py --task=go1_abl1 --gpu=cuda:1  --headless --seed=1 --num_eps=100 --num_envs=10 --terrain_type=plane --disturbance_type=push --push_bounds 2.0625 1.2875 2.0625 --log --log_path=exp_data_corl_10/wrench_max

# python play_eval_fixed_episodes.py --task=go1_abl1 --gpu=cuda:1  --headless --seed=1 --num_eps=100 --num_envs=10 --terrain_type=plane --disturbance_type=push --push_bounds 2.1667 1.3667 2.1667 --log --log_path=exp_data_corl_10/wrench_max

# python play_eval_fixed_episodes.py --task=go1_abl1 --gpu=cuda:1  --headless --seed=1 --num_eps=100 --num_envs=10 --terrain_type=plane --disturbance_type=push --push_bounds 2.2708 1.4458 2.2708 --log --log_path=exp_data_corl_10/wrench_max

# python play_eval_fixed_episodes.py --task=go1_abl1 --gpu=cuda:1  --headless --seed=1 --num_eps=100 --num_envs=10 --terrain_type=plane --disturbance_type=push --push_bounds 2.3750 1.5250 2.3750 --log --log_path=exp_data_corl_10/wrench_max

# python play_eval_fixed_episodes.py --task=go1_abl1 --gpu=cuda:1  --headless --seed=1 --num_eps=100 --num_envs=10 --terrain_type=plane --disturbance_type=push --push_bounds 2.4792 1.6042 2.4792 --log --log_path=exp_data_corl_10/wrench_max

# python play_eval_fixed_episodes.py --task=go1_abl1 --gpu=cuda:1  --headless --seed=1 --num_eps=100 --num_envs=10 --terrain_type=plane --disturbance_type=push --push_bounds 2.5833 1.6833 2.5833 --log --log_path=exp_data_corl_10/wrench_max

# python play_eval_fixed_episodes.py --task=go1_abl1 --gpu=cuda:1  --headless --seed=1 --num_eps=100 --num_envs=10 --terrain_type=plane --disturbance_type=push --push_bounds 2.6875 1.7625 2.6875 --log --log_path=exp_data_corl_10/wrench_max

# python play_eval_fixed_episodes.py --task=go1_abl1 --gpu=cuda:1  --headless --seed=1 --num_eps=100 --num_envs=10 --terrain_type=plane --disturbance_type=push --push_bounds 2.7917 1.8417 2.7917 --log --log_path=exp_data_corl_10/wrench_max

# python play_eval_fixed_episodes.py --task=go1_abl1 --gpu=cuda:1  --headless --seed=1 --num_eps=100 --num_envs=10 --terrain_type=plane --disturbance_type=push --push_bounds 2.8958 1.9208 2.8958 --log --log_path=exp_data_corl_10/wrench_max

# python play_eval_fixed_episodes.py --task=go1_abl1 --gpu=cuda:1  --headless --seed=1 --num_eps=100 --num_envs=10 --terrain_type=plane --disturbance_type=push --push_bounds 3.0000 2.0000 3.0000 --log --log_path=exp_data_corl_10/wrench_max


###
#   Terrain completion tests
###

python play_eval_fixed_episodes_terrain_reset.py --task=go1_abl1 --gpu=cuda:1 --headless --seed=1 --num_eps=100 --num_envs=10 --terrain_type=stairs_up --disturbance_type push payload --payload_bounds 10.0 10.0 --push_bounds 1.5 0.0 1.5 --shift_com --com_bounds 0.20 0.15 0.15 --fixed_cmd 0.65 0.0 0.0 0.0 --log --log_path=exp_data_corl_10/terrain_03

python play_eval_fixed_episodes_terrain_reset.py --task=go1_abl1 --gpu=cuda:1 --headless --seed=1 --num_eps=100 --num_envs=10 --terrain_type=stairs_down --disturbance_type push payload --payload_bounds 10.0 10.0 --push_bounds 1.5 0.0 1.5 --shift_com --com_bounds 0.20 0.15 0.15 --fixed_cmd 0.65 0.0 0.0 0.0 --log --log_path=exp_data_corl_10/terrain_03

python play_eval_fixed_episodes_terrain_reset.py --task=go1_abl1 --gpu=cuda:1 --headless --seed=1 --num_eps=100 --num_envs=10 --terrain_type=slope_up --disturbance_type push payload --payload_bounds 10.0 10.0 --push_bounds 1.5 0.0 1.5 --shift_com --com_bounds 0.20 0.15 0.15 --fixed_cmd 0.65 0.0 0.0 0.0 --log --log_path=exp_data_corl_10/terrain_03

python play_eval_fixed_episodes_terrain_reset.py --task=go1_abl1 --gpu=cuda:1 --headless --seed=1 --num_eps=100 --num_envs=10 --terrain_type=slope_down --disturbance_type push payload --payload_bounds 10.0 10.0 --push_bounds 1.5 0.0 1.5 --shift_com --com_bounds 0.20 0.15 0.15 --fixed_cmd 0.65 0.0 0.0 0.0 --log --log_path=exp_data_corl_10/terrain_03

python play_eval_fixed_episodes_terrain_reset.py --task=go1_abl1 --gpu=cuda:1 --headless --seed=1 --num_eps=100 --num_envs=10 --terrain_type=rough --disturbance_type push payload --payload_bounds 10.0 10.0 --push_bounds 1.5 0.0 1.5 --shift_com --com_bounds 0.20 0.15 0.15 --fixed_cmd 0.65 0.0 0.0 0.0 --log --log_path=exp_data_corl_10/terrain_03

python play_eval_fixed_episodes_terrain_reset.py --task=go1_abl1 --gpu=cuda:1 --headless --seed=1 --num_eps=100 --num_envs=10 --terrain_type=discrete --disturbance_type push payload --payload_bounds 10.0 10.0 --push_bounds 1.5 0.0 1.5 --shift_com --com_bounds 0.20 0.15 0.15 --fixed_cmd 0.65 0.0 0.0 0.0 --log --log_path=exp_data_corl_10/terrain_03

python play_eval_fixed_episodes_terrain_reset.py --task=go1_abl1 --gpu=cuda:1 --headless --seed=1 --num_eps=100 --num_envs=10 --terrain_type=wave --disturbance_type push payload --payload_bounds 10.0 10.0 --push_bounds 1.5 0.0 1.5 --shift_com --com_bounds 0.20 0.15 0.15 --fixed_cmd 0.65 0.0 0.0 0.0 --log --log_path=exp_data_corl_10/terrain_03