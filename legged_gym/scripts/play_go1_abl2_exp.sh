#!/bin/bash

. /home/oyoungquist/anaconda3/etc/profile.d/conda.sh

conda activate /home/oyoungquist/.conda/envs/genesis_lr

export SIMULATOR=genesis_pact_nopinn

# No disturbance conditions
python play_exp_largescale.py --task=go1_abl2 --gpu=cuda:1 --headless --seed=1 --num_eps=2.00 --num_envs=100 --terrain_type=plane --disturbance_type=none --log --log_path=exp_data_corl_03/no_disturbance  

python play_exp_largescale.py --task=go1_abl2 --gpu=cuda:1 --headless --seed=1 --num_eps=2.00 --num_envs=100 --terrain_type=slope --disturbance_type=none --log --log_path=exp_data_corl_03/no_disturbance  

python play_exp_largescale.py --task=go1_abl2 --gpu=cuda:1 --headless --seed=1 --num_eps=2.00 --num_envs=100 --terrain_type=stairs --disturbance_type=none --log --log_path=exp_data_corl_03/no_disturbance  

python play_exp_largescale.py --task=go1_abl2 --gpu=cuda:1 --headless --seed=1 --num_eps=2.00 --num_envs=100 --terrain_type=rough --disturbance_type=none --log --log_path=exp_data_corl_03/no_disturbance  

python play_exp_largescale.py --task=go1_abl2 --gpu=cuda:1 --headless --seed=1 --num_eps=2.00 --num_envs=100 --terrain_type=discrete --disturbance_type=none --log --log_path=exp_data_corl_03/no_disturbance  

python play_exp_largescale.py --task=go1_abl2 --gpu=cuda:1 --headless --seed=1 --num_eps=2.00 --num_envs=100 --terrain_type=wave --disturbance_type=none --log --log_path=exp_data_corl_03/no_disturbance  

# payload with COM shift
python play_exp_largescale.py --task=go1_abl2 --gpu=cuda:1 --shift_com --headless --seed=1 --num_eps=2.00 --num_envs=100 --terrain_type=plane --disturbance_type=payload --log --log_path=exp_data_corl_03/payload_disturbance  

python play_exp_largescale.py --task=go1_abl2 --gpu=cuda:1 --shift_com --headless --seed=1 --num_eps=2.00 --num_envs=100 --terrain_type=slope --disturbance_type=payload --log --log_path=exp_data_corl_03/payload_disturbance  

python play_exp_largescale.py --task=go1_abl2 --gpu=cuda:1 --shift_com --headless --seed=1 --num_eps=2.00 --num_envs=100 --terrain_type=stairs --disturbance_type=payload --log --log_path=exp_data_corl_03/payload_disturbance  

python play_exp_largescale.py --task=go1_abl2 --gpu=cuda:1 --shift_com --headless --seed=1 --num_eps=2.00 --num_envs=100 --terrain_type=rough --disturbance_type=payload --log --log_path=exp_data_corl_03/payload_disturbance  

python play_exp_largescale.py --task=go1_abl2 --gpu=cuda:1 --shift_com --headless --seed=1 --num_eps=2.00 --num_envs=100 --terrain_type=discrete --disturbance_type=payload --log --log_path=exp_data_corl_03/payload_disturbance  

python play_exp_largescale.py --task=go1_abl2 --gpu=cuda:1 --shift_com --headless --seed=1 --num_eps=2.00 --num_envs=100 --terrain_type=wave --disturbance_type=payload --log --log_path=exp_data_corl_03/payload_disturbance  

# pushes
python play_exp_largescale.py --task=go1_abl2 --gpu=cuda:1 --headless --seed=1 --num_eps=2.00 --num_envs=100 --terrain_type=plane --disturbance_type=push --log --log_path=exp_data_corl_03/push_disturbance  

python play_exp_largescale.py --task=go1_abl2 --gpu=cuda:1 --headless --seed=1 --num_eps=2.00 --num_envs=100 --terrain_type=slope --disturbance_type=push --log --log_path=exp_data_corl_03/push_disturbance  

python play_exp_largescale.py --task=go1_abl2 --gpu=cuda:1 --headless --seed=1 --num_eps=2.00 --num_envs=100 --terrain_type=stairs --disturbance_type=push --log --log_path=exp_data_corl_03/push_disturbance  

python play_exp_largescale.py --task=go1_abl2 --gpu=cuda:1 --headless --seed=1 --num_eps=2.00 --num_envs=100 --terrain_type=rough --disturbance_type=push --log --log_path=exp_data_corl_03/push_disturbance  

python play_exp_largescale.py --task=go1_abl2 --gpu=cuda:1 --headless --seed=1 --num_eps=2.00 --num_envs=100 --terrain_type=discrete --disturbance_type=push --log --log_path=exp_data_corl_03/push_disturbance  

python play_exp_largescale.py --task=go1_abl2 --gpu=cuda:1 --headless --seed=1 --num_eps=2.00 --num_envs=100 --terrain_type=wave --disturbance_type=push --log --log_path=exp_data_corl_03/push_disturbance  