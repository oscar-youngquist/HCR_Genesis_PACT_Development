#!/bin/bash

. /home/oyoungquist/anaconda3/etc/profile.d/conda.sh

conda activate /home/oyoungquist/.conda/envs/genesis_lr

export SIMULATOR=genesis_pact_rl2ac

# # Sanity checks w/ paper parameters
# python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:0 --headless --seed=1 --num_eps=1.00 --num_envs=50 --terrain_type=rough --disturbance_type=payload --log --log_path=exp_data_corl_rl2ac/param_search_02 --alpha=50.0 --kappa=1.2 --lambda_0=3.0 --k_0=20.0

# python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:0 --headless --seed=1 --num_eps=1.00 --num_envs=50 --terrain_type=rough --disturbance_type=payload --log --log_path=exp_data_corl_rl2ac/param_search_02 --kappa=0.0 

# python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:0 --headless --seed=1 --num_eps=1.00 --num_envs=50 --terrain_type=rough --disturbance_type=payload --log --log_path=exp_data_corl_rl2ac/param_search_02 --lambda_0=0.0

# more conservative benchmark
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:0 --headless --seed=1 --num_eps=1.00 --num_envs=50 --terrain_type=rough --disturbance_type=payload --log --log_path=exp_data_corl_rl2ac/param_search_02 --alpha=10.0 --kappa=0.1 --lambda_0=0.3 --k_0=5.0

# Alpha sweep
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:0 --headless --seed=1 --num_eps=1.00 --num_envs=50 --terrain_type=rough --disturbance_type=payload --log --log_path=exp_data_corl_rl2ac/param_search_02 --alpha=2.0 --kappa=0.1 --lambda_0=0.3 --k_0=5.0

python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:0 --headless --seed=1 --num_eps=1.00 --num_envs=50 --terrain_type=rough --disturbance_type=payload --log --log_path=exp_data_corl_rl2ac/param_search_02 --alpha=5.0 --kappa=0.1 --lambda_0=0.3 --k_0=5.0

python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:0 --headless --seed=1 --num_eps=1.00 --num_envs=50 --terrain_type=rough --disturbance_type=payload --log --log_path=exp_data_corl_rl2ac/param_search_02 --alpha=20.0 --kappa=0.1 --lambda_0=0.3 --k_0=5.0

python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:0 --headless --seed=1 --num_eps=1.00 --num_envs=50 --terrain_type=rough --disturbance_type=payload --log --log_path=exp_data_corl_rl2ac/param_search_02 --alpha=35.0 --kappa=0.1 --lambda_0=0.3 --k_0=5.0

python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:0 --headless --seed=1 --num_eps=1.00 --num_envs=50 --terrain_type=rough --disturbance_type=payload --log --log_path=exp_data_corl_rl2ac/param_search_02 --alpha=50.0 --kappa=0.1 --lambda_0=0.3 --k_0=5.0

# kappa sweep
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:0 --headless --seed=1 --num_eps=1.00 --num_envs=50 --terrain_type=rough --disturbance_type=payload --log --log_path=exp_data_corl_rl2ac/param_search_02 --alpha=10.0 --kappa=0.0 --lambda_0=0.3 --k_0=5.0

python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:0 --headless --seed=1 --num_eps=1.00 --num_envs=50 --terrain_type=rough --disturbance_type=payload --log --log_path=exp_data_corl_rl2ac/param_search_02 --alpha=10.0 --kappa=0.05 --lambda_0=0.3 --k_0=5.0

python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:0 --headless --seed=1 --num_eps=1.00 --num_envs=50 --terrain_type=rough --disturbance_type=payload --log --log_path=exp_data_corl_rl2ac/param_search_02 --alpha=10.0 --kappa=0.25 --lambda_0=0.3 --k_0=5.0

python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:0 --headless --seed=1 --num_eps=1.00 --num_envs=50 --terrain_type=rough --disturbance_type=payload --log --log_path=exp_data_corl_rl2ac/param_search_02 --alpha=10.0 --kappa=0.5 --lambda_0=0.3 --k_0=5.0

python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:0 --headless --seed=1 --num_eps=1.00 --num_envs=50 --terrain_type=rough --disturbance_type=payload --log --log_path=exp_data_corl_rl2ac/param_search_02 --alpha=10.0 --kappa=0.8 --lambda_0=0.3 --k_0=5.0

python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:0 --headless --seed=1 --num_eps=1.00 --num_envs=50 --terrain_type=rough --disturbance_type=payload --log --log_path=exp_data_corl_rl2ac/param_search_02 --alpha=10.0 --kappa=1.2 --lambda_0=0.3 --k_0=5.0


# lambda_0 sweep
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:0 --headless --seed=1 --num_eps=1.00 --num_envs=50 --terrain_type=rough --disturbance_type=payload --log --log_path=exp_data_corl_rl2ac/param_search_02 --alpha=10.0 --kappa=0.1 --lambda_0=0.0 --k_0=5.0

python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:0 --headless --seed=1 --num_eps=1.00 --num_envs=50 --terrain_type=rough --disturbance_type=payload --log --log_path=exp_data_corl_rl2ac/param_search_02 --alpha=10.0 --kappa=0.1 --lambda_0=0.05 --k_0=5.0

python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:0 --headless --seed=1 --num_eps=1.00 --num_envs=50 --terrain_type=rough --disturbance_type=payload --log --log_path=exp_data_corl_rl2ac/param_search_02 --alpha=10.0 --kappa=0.1 --lambda_0=0.1 --k_0=5.0

python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:0 --headless --seed=1 --num_eps=1.00 --num_envs=50 --terrain_type=rough --disturbance_type=payload --log --log_path=exp_data_corl_rl2ac/param_search_02 --alpha=10.0 --kappa=0.1 --lambda_0=1.0 --k_0=5.0

python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:0 --headless --seed=1 --num_eps=1.00 --num_envs=50 --terrain_type=rough --disturbance_type=payload --log --log_path=exp_data_corl_rl2ac/param_search_02 --alpha=10.0 --kappa=0.1 --lambda_0=3.0 --k_0=5.0

# k_0 sweep
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:0 --headless --seed=1 --num_eps=1.00 --num_envs=50 --terrain_type=rough --disturbance_type=payload --log --log_path=exp_data_corl_rl2ac/param_search_02 --alpha=10.0 --kappa=0.1 --lambda_0=0.3 --k_0=1.0

python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:0 --headless --seed=1 --num_eps=1.00 --num_envs=50 --terrain_type=rough --disturbance_type=payload --log --log_path=exp_data_corl_rl2ac/param_search_02 --alpha=10.0 --kappa=0.1 --lambda_0=0.3 --k_0=2.0

python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:0 --headless --seed=1 --num_eps=1.00 --num_envs=50 --terrain_type=rough --disturbance_type=payload --log --log_path=exp_data_corl_rl2ac/param_search_02 --alpha=10.0 --kappa=0.1 --lambda_0=0.3 --k_0=10.0

python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:0 --headless --seed=1 --num_eps=1.00 --num_envs=50 --terrain_type=rough --disturbance_type=payload --log --log_path=exp_data_corl_rl2ac/param_search_02 --alpha=10.0 --kappa=0.1 --lambda_0=0.3 --k_0=20.0


