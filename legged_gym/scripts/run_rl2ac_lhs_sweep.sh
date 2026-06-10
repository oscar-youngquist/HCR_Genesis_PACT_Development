#!/bin/bash

. /home/oyoungquist/anaconda3/etc/profile.d/conda.sh

conda activate /home/oyoungquist/.conda/envs/genesis_lr

export SIMULATOR=genesis_pact_rl2ac

# # Sanity checks w/ paper parameters
# python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:1 --headless --seed=1 --num_eps=1.00 --num_envs=100 --terrain_type=rough --disturbance_type=payload --shift_com --log --log_path=exp_data_corl_rl2ac/sample_param_search_full_02 --alpha=50 --kappa=1.2 --lambda_0=3 --k_0=20

# # Conservative benchmark
# python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:1 --headless --seed=1 --num_eps=1.00 --num_envs=100 --terrain_type=rough --disturbance_type=payload --shift_com --log --log_path=exp_data_corl_rl2ac/sample_param_search_full_02 --alpha=10 --kappa=0.1 --lambda_0=0.3 --k_0=5
# Latin-hypercube sweep: n_samples=100, seed=1
# LHS sample 000
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:1 --headless --seed=1 --num_eps=1.00 --num_envs=100 --terrain_type=rough --disturbance_type=payload --shift_com --log --log_path=exp_data_corl_rl2ac/sample_param_search_full_02 --alpha=21.988 --kappa=18.496 --lambda_0=2.7624 --k_0=15.295

# LHS sample 001
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:1 --headless --seed=1 --num_eps=1.00 --num_envs=100 --terrain_type=rough --disturbance_type=payload --shift_com --log --log_path=exp_data_corl_rl2ac/sample_param_search_full_02 --alpha=25.384 --kappa=14.451 --lambda_0=4.5276 --k_0=16.475

# LHS sample 002
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:1 --headless --seed=1 --num_eps=1.00 --num_envs=100 --terrain_type=rough --disturbance_type=payload --shift_com --log --log_path=exp_data_corl_rl2ac/sample_param_search_full_02 --alpha=15.515 --kappa=1.5588 --lambda_0=2.359 --k_0=8.1449

# LHS sample 003
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:1 --headless --seed=1 --num_eps=1.00 --num_envs=100 --terrain_type=rough --disturbance_type=payload --shift_com --log --log_path=exp_data_corl_rl2ac/sample_param_search_full_02 --alpha=23.049 --kappa=7.2942 --lambda_0=4.4976 --k_0=15.775

# LHS sample 004
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:1 --headless --seed=1 --num_eps=1.00 --num_envs=100 --terrain_type=rough --disturbance_type=payload --shift_com --log --log_path=exp_data_corl_rl2ac/sample_param_search_full_02 --alpha=38.593 --kappa=12.536 --lambda_0=0.92837 --k_0=6.8621

# LHS sample 005
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:1 --headless --seed=1 --num_eps=1.00 --num_envs=100 --terrain_type=rough --disturbance_type=payload --shift_com --log --log_path=exp_data_corl_rl2ac/sample_param_search_full_02 --alpha=23.511 --kappa=7.9137 --lambda_0=4.4428 --k_0=19.84

# LHS sample 006
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:1 --headless --seed=1 --num_eps=1.00 --num_envs=100 --terrain_type=rough --disturbance_type=payload --shift_com --log --log_path=exp_data_corl_rl2ac/sample_param_search_full_02 --alpha=20.29 --kappa=12.237 --lambda_0=2.1052 --k_0=11.326

# LHS sample 007
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:1 --headless --seed=1 --num_eps=1.00 --num_envs=100 --terrain_type=rough --disturbance_type=payload --shift_com --log --log_path=exp_data_corl_rl2ac/sample_param_search_full_02 --alpha=12.964 --kappa=13.193 --lambda_0=0.2399 --k_0=12.731

# LHS sample 008
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:1 --headless --seed=1 --num_eps=1.00 --num_envs=100 --terrain_type=rough --disturbance_type=payload --shift_com --log --log_path=exp_data_corl_rl2ac/sample_param_search_full_02 --alpha=33.789 --kappa=16.25 --lambda_0=3.4315 --k_0=5.5227

# LHS sample 009
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:1 --headless --seed=1 --num_eps=1.00 --num_envs=100 --terrain_type=rough --disturbance_type=payload --shift_com --log --log_path=exp_data_corl_rl2ac/sample_param_search_full_02 --alpha=16.961 --kappa=3.3738 --lambda_0=1.4799 --k_0=7.341

# LHS sample 010
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:1 --headless --seed=1 --num_eps=1.00 --num_envs=100 --terrain_type=rough --disturbance_type=payload --shift_com --log --log_path=exp_data_corl_rl2ac/sample_param_search_full_02 --alpha=35.488 --kappa=7.5023 --lambda_0=3.255 --k_0=17.635

# LHS sample 011
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:1 --headless --seed=1 --num_eps=1.00 --num_envs=100 --terrain_type=rough --disturbance_type=payload --shift_com --log --log_path=exp_data_corl_rl2ac/sample_param_search_full_02 --alpha=47.705 --kappa=10.616 --lambda_0=3.5768 --k_0=12.981

# LHS sample 012
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:1 --headless --seed=1 --num_eps=1.00 --num_envs=100 --terrain_type=rough --disturbance_type=payload --shift_com --log --log_path=exp_data_corl_rl2ac/sample_param_search_full_02 --alpha=24.367 --kappa=8.5736 --lambda_0=2.564 --k_0=11.182

# LHS sample 013
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:1 --headless --seed=1 --num_eps=1.00 --num_envs=100 --terrain_type=rough --disturbance_type=payload --shift_com --log --log_path=exp_data_corl_rl2ac/sample_param_search_full_02 --alpha=27.304 --kappa=17.084 --lambda_0=4.7909 --k_0=9.3079

# LHS sample 014
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:1 --headless --seed=1 --num_eps=1.00 --num_envs=100 --terrain_type=rough --disturbance_type=payload --shift_com --log --log_path=exp_data_corl_rl2ac/sample_param_search_full_02 --alpha=16.454 --kappa=19.967 --lambda_0=2.0704 --k_0=19.688

# LHS sample 015
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:1 --headless --seed=1 --num_eps=1.00 --num_envs=100 --terrain_type=rough --disturbance_type=payload --shift_com --log --log_path=exp_data_corl_rl2ac/sample_param_search_full_02 --alpha=18.512 --kappa=3.9922 --lambda_0=3.3784 --k_0=7.749

# LHS sample 016
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:1 --headless --seed=1 --num_eps=1.00 --num_envs=100 --terrain_type=rough --disturbance_type=payload --shift_com --log --log_path=exp_data_corl_rl2ac/sample_param_search_full_02 --alpha=37.985 --kappa=6.515 --lambda_0=4.0337 --k_0=15.187

# LHS sample 017
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:1 --headless --seed=1 --num_eps=1.00 --num_envs=100 --terrain_type=rough --disturbance_type=payload --shift_com --log --log_path=exp_data_corl_rl2ac/sample_param_search_full_02 --alpha=48.442 --kappa=10.033 --lambda_0=0.32609 --k_0=13.84

# LHS sample 018
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:1 --headless --seed=1 --num_eps=1.00 --num_envs=100 --terrain_type=rough --disturbance_type=payload --shift_com --log --log_path=exp_data_corl_rl2ac/sample_param_search_full_02 --alpha=25.011 --kappa=8.7391 --lambda_0=4.266 --k_0=11.053

# LHS sample 019
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:1 --headless --seed=1 --num_eps=1.00 --num_envs=100 --terrain_type=rough --disturbance_type=payload --shift_com --log --log_path=exp_data_corl_rl2ac/sample_param_search_full_02 --alpha=49.352 --kappa=5.1798 --lambda_0=4.835 --k_0=19.938

# LHS sample 020
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:1 --headless --seed=1 --num_eps=1.00 --num_envs=100 --terrain_type=rough --disturbance_type=payload --shift_com --log --log_path=exp_data_corl_rl2ac/sample_param_search_full_02 --alpha=19.592 --kappa=13.642 --lambda_0=4.1207 --k_0=9.7253

# LHS sample 021
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:1 --headless --seed=1 --num_eps=1.00 --num_envs=100 --terrain_type=rough --disturbance_type=payload --shift_com --log --log_path=exp_data_corl_rl2ac/sample_param_search_full_02 --alpha=31.521 --kappa=16.059 --lambda_0=3.5313 --k_0=13.458

# LHS sample 022
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:1 --headless --seed=1 --num_eps=1.00 --num_envs=100 --terrain_type=rough --disturbance_type=payload --shift_com --log --log_path=exp_data_corl_rl2ac/sample_param_search_full_02 --alpha=15.721 --kappa=18.294 --lambda_0=1.7063 --k_0=12.175

# LHS sample 023
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:1 --headless --seed=1 --num_eps=1.00 --num_envs=100 --terrain_type=rough --disturbance_type=payload --shift_com --log --log_path=exp_data_corl_rl2ac/sample_param_search_full_02 --alpha=46.471 --kappa=5.9031 --lambda_0=4.5813 --k_0=19.005

# LHS sample 024
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:1 --headless --seed=1 --num_eps=1.00 --num_envs=100 --terrain_type=rough --disturbance_type=payload --shift_com --log --log_path=exp_data_corl_rl2ac/sample_param_search_full_02 --alpha=12.731 --kappa=1.6082 --lambda_0=4.3526 --k_0=8.7403

# LHS sample 025
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:1 --headless --seed=1 --num_eps=1.00 --num_envs=100 --terrain_type=rough --disturbance_type=payload --shift_com --log --log_path=exp_data_corl_rl2ac/sample_param_search_full_02 --alpha=43.933 --kappa=7.7568 --lambda_0=2.4309 --k_0=18.016

# LHS sample 026
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:1 --headless --seed=1 --num_eps=1.00 --num_envs=100 --terrain_type=rough --disturbance_type=payload --shift_com --log --log_path=exp_data_corl_rl2ac/sample_param_search_full_02 --alpha=10.78 --kappa=2.9957 --lambda_0=4.957 --k_0=17.353

# LHS sample 027
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:1 --headless --seed=1 --num_eps=1.00 --num_envs=100 --terrain_type=rough --disturbance_type=payload --shift_com --log --log_path=exp_data_corl_rl2ac/sample_param_search_full_02 --alpha=18.994 --kappa=15.648 --lambda_0=1.6704 --k_0=7.9917

# LHS sample 028
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:1 --headless --seed=1 --num_eps=1.00 --num_envs=100 --terrain_type=rough --disturbance_type=payload --shift_com --log --log_path=exp_data_corl_rl2ac/sample_param_search_full_02 --alpha=43.107 --kappa=5.2298 --lambda_0=0.51307 --k_0=12.278

# LHS sample 029
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:1 --headless --seed=1 --num_eps=1.00 --num_envs=100 --terrain_type=rough --disturbance_type=payload --shift_com --log --log_path=exp_data_corl_rl2ac/sample_param_search_full_02 --alpha=44.025 --kappa=4.592 --lambda_0=1.248 --k_0=10.363

# LHS sample 030
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:1 --headless --seed=1 --num_eps=1.00 --num_envs=100 --terrain_type=rough --disturbance_type=payload --shift_com --log --log_path=exp_data_corl_rl2ac/sample_param_search_full_02 --alpha=29.259 --kappa=13.386 --lambda_0=2.7267 --k_0=14.223

# LHS sample 031
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:1 --headless --seed=1 --num_eps=1.00 --num_envs=100 --terrain_type=rough --disturbance_type=payload --shift_com --log --log_path=exp_data_corl_rl2ac/sample_param_search_full_02 --alpha=22.206 --kappa=14.102 --lambda_0=3.8581 --k_0=10.673

# LHS sample 032
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:1 --headless --seed=1 --num_eps=1.00 --num_envs=100 --terrain_type=rough --disturbance_type=payload --shift_com --log --log_path=exp_data_corl_rl2ac/sample_param_search_full_02 --alpha=36.113 --kappa=2.7825 --lambda_0=1.849 --k_0=6.5897

# LHS sample 033
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:1 --headless --seed=1 --num_eps=1.00 --num_envs=100 --terrain_type=rough --disturbance_type=payload --shift_com --log --log_path=exp_data_corl_rl2ac/sample_param_search_full_02 --alpha=14.301 --kappa=15.272 --lambda_0=0.71737 --k_0=19.386

# LHS sample 034
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:1 --headless --seed=1 --num_eps=1.00 --num_envs=100 --terrain_type=rough --disturbance_type=payload --shift_com --log --log_path=exp_data_corl_rl2ac/sample_param_search_full_02 --alpha=13.42 --kappa=17.94 --lambda_0=2.2614 --k_0=10.745

# LHS sample 035
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:1 --headless --seed=1 --num_eps=1.00 --num_envs=100 --terrain_type=rough --disturbance_type=payload --shift_com --log --log_path=exp_data_corl_rl2ac/sample_param_search_full_02 --alpha=49.89 --kappa=3.0932 --lambda_0=0.88658 --k_0=19.529

# LHS sample 036
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:1 --headless --seed=1 --num_eps=1.00 --num_envs=100 --terrain_type=rough --disturbance_type=payload --shift_com --log --log_path=exp_data_corl_rl2ac/sample_param_search_full_02 --alpha=10.858 --kappa=19.138 --lambda_0=4.3885 --k_0=13.165

# LHS sample 037
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:1 --headless --seed=1 --num_eps=1.00 --num_envs=100 --terrain_type=rough --disturbance_type=payload --shift_com --log --log_path=exp_data_corl_rl2ac/sample_param_search_full_02 --alpha=24.416 --kappa=12.642 --lambda_0=1.6131 --k_0=18.897

# LHS sample 038
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:1 --headless --seed=1 --num_eps=1.00 --num_envs=100 --terrain_type=rough --disturbance_type=payload --shift_com --log --log_path=exp_data_corl_rl2ac/sample_param_search_full_02 --alpha=31.677 --kappa=10.267 --lambda_0=3.49 --k_0=16.823

# LHS sample 039
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:1 --headless --seed=1 --num_eps=1.00 --num_envs=100 --terrain_type=rough --disturbance_type=payload --shift_com --log --log_path=exp_data_corl_rl2ac/sample_param_search_full_02 --alpha=12.169 --kappa=19.408 --lambda_0=3.2854 --k_0=9.9642

# LHS sample 040
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:1 --headless --seed=1 --num_eps=1.00 --num_envs=100 --terrain_type=rough --disturbance_type=payload --shift_com --log --log_path=exp_data_corl_rl2ac/sample_param_search_full_02 --alpha=37.056 --kappa=8.9043 --lambda_0=2.8394 --k_0=17.755

# LHS sample 041
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:1 --headless --seed=1 --num_eps=1.00 --num_envs=100 --terrain_type=rough --disturbance_type=payload --shift_com --log --log_path=exp_data_corl_rl2ac/sample_param_search_full_02 --alpha=42.264 --kappa=15.508 --lambda_0=0.60997 --k_0=7.053

# LHS sample 042
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:1 --headless --seed=1 --num_eps=1.00 --num_envs=100 --terrain_type=rough --disturbance_type=payload --shift_com --log --log_path=exp_data_corl_rl2ac/sample_param_search_full_02 --alpha=19.985 --kappa=2.2084 --lambda_0=1.9061 --k_0=6.1136

# LHS sample 043
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:1 --headless --seed=1 --num_eps=1.00 --num_envs=100 --terrain_type=rough --disturbance_type=payload --shift_com --log --log_path=exp_data_corl_rl2ac/sample_param_search_full_02 --alpha=17.705 --kappa=11.821 --lambda_0=1.9819 --k_0=15.497

# LHS sample 044
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:1 --headless --seed=1 --num_eps=1.00 --num_envs=100 --terrain_type=rough --disturbance_type=payload --shift_com --log --log_path=exp_data_corl_rl2ac/sample_param_search_full_02 --alpha=49.053 --kappa=14.291 --lambda_0=3.6642 --k_0=8.574

# LHS sample 045
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:1 --headless --seed=1 --num_eps=1.00 --num_envs=100 --terrain_type=rough --disturbance_type=payload --shift_com --log --log_path=exp_data_corl_rl2ac/sample_param_search_full_02 --alpha=45.35 --kappa=10.411 --lambda_0=2.16 --k_0=9.4362

# LHS sample 046
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:1 --headless --seed=1 --num_eps=1.00 --num_envs=100 --terrain_type=rough --disturbance_type=payload --shift_com --log --log_path=exp_data_corl_rl2ac/sample_param_search_full_02 --alpha=36.486 --kappa=3.7142 --lambda_0=3.6548 --k_0=15.991

# LHS sample 047
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:1 --headless --seed=1 --num_eps=1.00 --num_envs=100 --terrain_type=rough --disturbance_type=payload --shift_com --log --log_path=exp_data_corl_rl2ac/sample_param_search_full_02 --alpha=39.836 --kappa=1.2652 --lambda_0=2.5335 --k_0=5.26

# LHS sample 048
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:1 --headless --seed=1 --num_eps=1.00 --num_envs=100 --terrain_type=rough --disturbance_type=payload --shift_com --log --log_path=exp_data_corl_rl2ac/sample_param_search_full_02 --alpha=22.446 --kappa=12.805 --lambda_0=4.6362 --k_0=14.805

# LHS sample 049
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:1 --headless --seed=1 --num_eps=1.00 --num_envs=100 --terrain_type=rough --disturbance_type=payload --shift_com --log --log_path=exp_data_corl_rl2ac/sample_param_search_full_02 --alpha=30.273 --kappa=6.4645 --lambda_0=0.99129 --k_0=18.234

# LHS sample 050
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:1 --headless --seed=1 --num_eps=1.00 --num_envs=100 --terrain_type=rough --disturbance_type=payload --shift_com --log --log_path=exp_data_corl_rl2ac/sample_param_search_full_02 --alpha=23.845 --kappa=14.618 --lambda_0=1.5652 --k_0=16.309

# LHS sample 051
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:1 --headless --seed=1 --num_eps=1.00 --num_envs=100 --terrain_type=rough --disturbance_type=payload --shift_com --log --log_path=exp_data_corl_rl2ac/sample_param_search_full_02 --alpha=44.866 --kappa=19.624 --lambda_0=4.204 --k_0=7.1494

# LHS sample 052
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:1 --headless --seed=1 --num_eps=1.00 --num_envs=100 --terrain_type=rough --disturbance_type=payload --shift_com --log --log_path=exp_data_corl_rl2ac/sample_param_search_full_02 --alpha=11.579 --kappa=17.265 --lambda_0=3.2222 --k_0=8.3064

# LHS sample 053
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:1 --headless --seed=1 --num_eps=1.00 --num_envs=100 --terrain_type=rough --disturbance_type=payload --shift_com --log --log_path=exp_data_corl_rl2ac/sample_param_search_full_02 --alpha=41.531 --kappa=13.816 --lambda_0=0.46371 --k_0=18.374

# LHS sample 054
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:1 --headless --seed=1 --num_eps=1.00 --num_envs=100 --terrain_type=rough --disturbance_type=payload --shift_com --log --log_path=exp_data_corl_rl2ac/sample_param_search_full_02 --alpha=18.3 --kappa=4.1241 --lambda_0=4.6867 --k_0=14.418

# LHS sample 055
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:1 --headless --seed=1 --num_eps=1.00 --num_envs=100 --terrain_type=rough --disturbance_type=payload --shift_com --log --log_path=exp_data_corl_rl2ac/sample_param_search_full_02 --alpha=37.522 --kappa=18.711 --lambda_0=3.0064 --k_0=11.76

# LHS sample 056
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:1 --headless --seed=1 --num_eps=1.00 --num_envs=100 --terrain_type=rough --disturbance_type=payload --shift_com --log --log_path=exp_data_corl_rl2ac/sample_param_search_full_02 --alpha=14.932 --kappa=2.0941 --lambda_0=3.1735 --k_0=7.5093

# LHS sample 057
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:1 --headless --seed=1 --num_eps=1.00 --num_envs=100 --terrain_type=rough --disturbance_type=payload --shift_com --log --log_path=exp_data_corl_rl2ac/sample_param_search_full_02 --alpha=14.615 --kappa=16.911 --lambda_0=3.9789 --k_0=16.577

# LHS sample 058
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:1 --headless --seed=1 --num_eps=1.00 --num_envs=100 --terrain_type=rough --disturbance_type=payload --shift_com --log --log_path=exp_data_corl_rl2ac/sample_param_search_full_02 --alpha=30.877 --kappa=19.49 --lambda_0=2.4745 --k_0=6.2802

# LHS sample 059
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:1 --headless --seed=1 --num_eps=1.00 --num_envs=100 --terrain_type=rough --disturbance_type=payload --shift_com --log --log_path=exp_data_corl_rl2ac/sample_param_search_full_02 --alpha=32.742 --kappa=16.445 --lambda_0=3.7335 --k_0=13.885

# LHS sample 060
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:1 --headless --seed=1 --num_eps=1.00 --num_envs=100 --terrain_type=rough --disturbance_type=payload --shift_com --log --log_path=exp_data_corl_rl2ac/sample_param_search_full_02 --alpha=35.058 --kappa=5.5313 --lambda_0=0.74134 --k_0=14.109

# LHS sample 061
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:1 --headless --seed=1 --num_eps=1.00 --num_envs=100 --terrain_type=rough --disturbance_type=payload --shift_com --log --log_path=exp_data_corl_rl2ac/sample_param_search_full_02 --alpha=45.727 --kappa=9.4077 --lambda_0=0.84873 --k_0=15.61

# LHS sample 062
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:1 --headless --seed=1 --num_eps=1.00 --num_envs=100 --terrain_type=rough --disturbance_type=payload --shift_com --log --log_path=exp_data_corl_rl2ac/sample_param_search_full_02 --alpha=16.181 --kappa=4.6132 --lambda_0=1.3437 --k_0=18.662

# LHS sample 063
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:1 --headless --seed=1 --num_eps=1.00 --num_envs=100 --terrain_type=rough --disturbance_type=payload --shift_com --log --log_path=exp_data_corl_rl2ac/sample_param_search_full_02 --alpha=44.73 --kappa=14.841 --lambda_0=2.9024 --k_0=8.2989

# LHS sample 064
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:1 --headless --seed=1 --num_eps=1.00 --num_envs=100 --terrain_type=rough --disturbance_type=payload --shift_com --log --log_path=exp_data_corl_rl2ac/sample_param_search_full_02 --alpha=46.277 --kappa=12.117 --lambda_0=4.8978 --k_0=10.427

# LHS sample 065
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:1 --headless --seed=1 --num_eps=1.00 --num_envs=100 --terrain_type=rough --disturbance_type=payload --shift_com --log --log_path=exp_data_corl_rl2ac/sample_param_search_full_02 --alpha=26.741 --kappa=15.934 --lambda_0=4.2948 --k_0=6.4154

# LHS sample 066
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:1 --headless --seed=1 --num_eps=1.00 --num_envs=100 --terrain_type=rough --disturbance_type=payload --shift_com --log --log_path=exp_data_corl_rl2ac/sample_param_search_full_02 --alpha=46.959 --kappa=9.3187 --lambda_0=3.0451 --k_0=18.193

# LHS sample 067
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:1 --headless --seed=1 --num_eps=1.00 --num_envs=100 --terrain_type=rough --disturbance_type=payload --shift_com --log --log_path=exp_data_corl_rl2ac/sample_param_search_full_02 --alpha=34.11 --kappa=18.173 --lambda_0=3.9436 --k_0=11.611

# LHS sample 068
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:1 --headless --seed=1 --num_eps=1.00 --num_envs=100 --terrain_type=rough --disturbance_type=payload --shift_com --log --log_path=exp_data_corl_rl2ac/sample_param_search_full_02 --alpha=33.551 --kappa=6.0598 --lambda_0=0.35344 --k_0=9.6183

# LHS sample 069
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:1 --headless --seed=1 --num_eps=1.00 --num_envs=100 --terrain_type=rough --disturbance_type=payload --shift_com --log --log_path=exp_data_corl_rl2ac/sample_param_search_full_02 --alpha=30.715 --kappa=4.8577 --lambda_0=1.5037 --k_0=12.579

# LHS sample 070
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:1 --headless --seed=1 --num_eps=1.00 --num_envs=100 --terrain_type=rough --disturbance_type=payload --shift_com --log --log_path=exp_data_corl_rl2ac/sample_param_search_full_02 --alpha=28.204 --kappa=2.4519 --lambda_0=2.6048 --k_0=8.8853

# LHS sample 071
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:1 --headless --seed=1 --num_eps=1.00 --num_envs=100 --terrain_type=rough --disturbance_type=payload --shift_com --log --log_path=exp_data_corl_rl2ac/sample_param_search_full_02 --alpha=39.369 --kappa=8.1804 --lambda_0=0.55899 --k_0=11.913

# LHS sample 072
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:1 --headless --seed=1 --num_eps=1.00 --num_envs=100 --terrain_type=rough --disturbance_type=payload --shift_com --log --log_path=exp_data_corl_rl2ac/sample_param_search_full_02 --alpha=40.669 --kappa=8.335 --lambda_0=4.0424 --k_0=12.874

# LHS sample 073
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:1 --headless --seed=1 --num_eps=1.00 --num_envs=100 --terrain_type=rough --disturbance_type=payload --shift_com --log --log_path=exp_data_corl_rl2ac/sample_param_search_full_02 --alpha=29.101 --kappa=9.5643 --lambda_0=1.0341 --k_0=16.248

# LHS sample 074
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:1 --headless --seed=1 --num_eps=1.00 --num_envs=100 --terrain_type=rough --disturbance_type=payload --shift_com --log --log_path=exp_data_corl_rl2ac/sample_param_search_full_02 --alpha=29.928 --kappa=11.48 --lambda_0=4.7504 --k_0=14.681

# LHS sample 075
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:1 --headless --seed=1 --num_eps=1.00 --num_envs=100 --terrain_type=rough --disturbance_type=payload --shift_com --log --log_path=exp_data_corl_rl2ac/sample_param_search_full_02 --alpha=13.611 --kappa=1.0587 --lambda_0=4.1571 --k_0=9.0072

# LHS sample 076
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:1 --headless --seed=1 --num_eps=1.00 --num_envs=100 --terrain_type=rough --disturbance_type=payload --shift_com --log --log_path=exp_data_corl_rl2ac/sample_param_search_full_02 --alpha=48.168 --kappa=10.696 --lambda_0=2.8595 --k_0=5.9207

# LHS sample 077
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:1 --headless --seed=1 --num_eps=1.00 --num_envs=100 --terrain_type=rough --disturbance_type=payload --shift_com --log --log_path=exp_data_corl_rl2ac/sample_param_search_full_02 --alpha=32.033 --kappa=5.6751 --lambda_0=2.1852 --k_0=11.461

# LHS sample 078
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:1 --headless --seed=1 --num_eps=1.00 --num_envs=100 --terrain_type=rough --disturbance_type=payload --shift_com --log --log_path=exp_data_corl_rl2ac/sample_param_search_full_02 --alpha=20.911 --kappa=15.064 --lambda_0=0.81981 --k_0=10.236

# LHS sample 079
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:1 --headless --seed=1 --num_eps=1.00 --num_envs=100 --terrain_type=rough --disturbance_type=payload --shift_com --log --log_path=exp_data_corl_rl2ac/sample_param_search_full_02 --alpha=38.06 --kappa=7.1706 --lambda_0=1.4444 --k_0=5.1332

# LHS sample 080
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:1 --headless --seed=1 --num_eps=1.00 --num_envs=100 --terrain_type=rough --disturbance_type=payload --shift_com --log --log_path=exp_data_corl_rl2ac/sample_param_search_full_02 --alpha=39.158 --kappa=10.915 --lambda_0=1.1618 --k_0=12.369

# LHS sample 081
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:1 --headless --seed=1 --num_eps=1.00 --num_envs=100 --terrain_type=rough --disturbance_type=payload --shift_com --log --log_path=exp_data_corl_rl2ac/sample_param_search_full_02 --alpha=28.604 --kappa=6.1989 --lambda_0=3.7538 --k_0=10.945

# LHS sample 082
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:1 --headless --seed=1 --num_eps=1.00 --num_envs=100 --terrain_type=rough --disturbance_type=payload --shift_com --log --log_path=exp_data_corl_rl2ac/sample_param_search_full_02 --alpha=33.145 --kappa=11.859 --lambda_0=1.1359 --k_0=18.552

# LHS sample 083
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:1 --headless --seed=1 --num_eps=1.00 --num_envs=100 --terrain_type=rough --disturbance_type=payload --shift_com --log --log_path=exp_data_corl_rl2ac/sample_param_search_full_02 --alpha=26.257 --kappa=4.2698 --lambda_0=1.7938 --k_0=17.103

# LHS sample 084
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:1 --headless --seed=1 --num_eps=1.00 --num_envs=100 --terrain_type=rough --disturbance_type=payload --shift_com --log --log_path=exp_data_corl_rl2ac/sample_param_search_full_02 --alpha=21.264 --kappa=6.7849 --lambda_0=1.391 --k_0=9.8339

# LHS sample 085
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:1 --headless --seed=1 --num_eps=1.00 --num_envs=100 --terrain_type=rough --disturbance_type=payload --shift_com --log --log_path=exp_data_corl_rl2ac/sample_param_search_full_02 --alpha=25.625 --kappa=15.048 --lambda_0=2.3695 --k_0=19.184

# LHS sample 086
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:1 --headless --seed=1 --num_eps=1.00 --num_envs=100 --terrain_type=rough --disturbance_type=payload --shift_com --log --log_path=exp_data_corl_rl2ac/sample_param_search_full_02 --alpha=40.01 --kappa=9.9229 --lambda_0=1.0939 --k_0=15.934

# LHS sample 087
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:1 --headless --seed=1 --num_eps=1.00 --num_envs=100 --terrain_type=rough --disturbance_type=payload --shift_com --log --log_path=exp_data_corl_rl2ac/sample_param_search_full_02 --alpha=47.202 --kappa=16.599 --lambda_0=0.25654 --k_0=5.8808

# LHS sample 088
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:1 --headless --seed=1 --num_eps=1.00 --num_envs=100 --terrain_type=rough --disturbance_type=payload --shift_com --log --log_path=exp_data_corl_rl2ac/sample_param_search_full_02 --alpha=27.936 --kappa=1.9478 --lambda_0=3.3541 --k_0=16.901

# LHS sample 089
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:1 --headless --seed=1 --num_eps=1.00 --num_envs=100 --terrain_type=rough --disturbance_type=payload --shift_com --log --log_path=exp_data_corl_rl2ac/sample_param_search_full_02 --alpha=11.725 --kappa=9.0346 --lambda_0=1.9485 --k_0=6.7248

# LHS sample 090
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:1 --headless --seed=1 --num_eps=1.00 --num_envs=100 --terrain_type=rough --disturbance_type=payload --shift_com --log --log_path=exp_data_corl_rl2ac/sample_param_search_full_02 --alpha=34.403 --kappa=17.612 --lambda_0=4.9135 --k_0=7.5696

# LHS sample 091
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:1 --headless --seed=1 --num_eps=1.00 --num_envs=100 --terrain_type=rough --disturbance_type=payload --shift_com --log --log_path=exp_data_corl_rl2ac/sample_param_search_full_02 --alpha=10.205 --kappa=3.6546 --lambda_0=2.9751 --k_0=14.504

# LHS sample 092
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:1 --headless --seed=1 --num_eps=1.00 --num_envs=100 --terrain_type=rough --disturbance_type=payload --shift_com --log --log_path=exp_data_corl_rl2ac/sample_param_search_full_02 --alpha=20.616 --kappa=2.5924 --lambda_0=1.7684 --k_0=17.183

# LHS sample 093
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:1 --headless --seed=1 --num_eps=1.00 --num_envs=100 --terrain_type=rough --disturbance_type=payload --shift_com --log --log_path=exp_data_corl_rl2ac/sample_param_search_full_02 --alpha=35.934 --kappa=11.368 --lambda_0=0.64223 --k_0=5.6707

# LHS sample 094
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:1 --headless --seed=1 --num_eps=1.00 --num_envs=100 --terrain_type=rough --disturbance_type=payload --shift_com --log --log_path=exp_data_corl_rl2ac/sample_param_search_full_02 --alpha=17.281 --kappa=17.85 --lambda_0=2.303 --k_0=13.626

# LHS sample 095
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:1 --headless --seed=1 --num_eps=1.00 --num_envs=100 --terrain_type=rough --disturbance_type=payload --shift_com --log --log_path=exp_data_corl_rl2ac/sample_param_search_full_02 --alpha=43.285 --kappa=11.198 --lambda_0=1.2585 --k_0=5.3612

# LHS sample 096
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:1 --headless --seed=1 --num_eps=1.00 --num_envs=100 --terrain_type=rough --disturbance_type=payload --shift_com --log --log_path=exp_data_corl_rl2ac/sample_param_search_full_02 --alpha=42.498 --kappa=12.979 --lambda_0=2.6709 --k_0=9.1513

# LHS sample 097
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:1 --headless --seed=1 --num_eps=1.00 --num_envs=100 --terrain_type=rough --disturbance_type=payload --shift_com --log --log_path=exp_data_corl_rl2ac/sample_param_search_full_02 --alpha=27.037 --kappa=18.939 --lambda_0=0.40369 --k_0=15.03

# LHS sample 098
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:1 --headless --seed=1 --num_eps=1.00 --num_envs=100 --terrain_type=rough --disturbance_type=payload --shift_com --log --log_path=exp_data_corl_rl2ac/sample_param_search_full_02 --alpha=41.954 --kappa=17.346 --lambda_0=3.0938 --k_0=13.381

# LHS sample 099
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:1 --headless --seed=1 --num_eps=1.00 --num_envs=100 --terrain_type=rough --disturbance_type=payload --shift_com --log --log_path=exp_data_corl_rl2ac/sample_param_search_full_02 --alpha=41.168 --kappa=6.9607 --lambda_0=3.8044 --k_0=17.508
