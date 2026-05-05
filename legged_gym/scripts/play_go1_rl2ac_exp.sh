#!/bin/bash

. /home/oyoungquist/anaconda3/etc/profile.d/conda.sh

conda activate /home/oyoungquist/.conda/envs/genesis_lr

export SIMULATOR=genesis_pact_rl2ac

# No disturbance conditions
python play_exp_largescale.py --task=go1_rl2ac --gpu=cuda:0 --headless --seed=1 --num_eps=2.00 --num_envs=100 --terrain_type=plane --disturbance_type=none --log --log_path=exp_data_corl/no_disturbance

python play_exp_largescale.py --task=go1_rl2ac --gpu=cuda:0 --headless --seed=1 --num_eps=2.00 --num_envs=100 --terrain_type=slope --disturbance_type=none --log --log_path=exp_data_corl/no_disturbance

python play_exp_largescale.py --task=go1_rl2ac --gpu=cuda:0 --headless --seed=1 --num_eps=2.00 --num_envs=100 --terrain_type=stairs --disturbance_type=none --log --log_path=exp_data_corl/no_disturbance

python play_exp_largescale.py --task=go1_rl2ac --gpu=cuda:0 --headless --seed=1 --num_eps=2.00 --num_envs=100 --terrain_type=rough --disturbance_type=none --log --log_path=exp_data_corl/no_disturbance

python play_exp_largescale.py --task=go1_rl2ac --gpu=cuda:0 --headless --seed=1 --num_eps=2.00 --num_envs=100 --terrain_type=discrete --disturbance_type=none --log --log_path=exp_data_corl/no_disturbance

python play_exp_largescale.py --task=go1_rl2ac --gpu=cuda:0 --headless --seed=1 --num_eps=2.00 --num_envs=100 --terrain_type=wave --disturbance_type=none --log --log_path=exp_data_corl/no_disturbance