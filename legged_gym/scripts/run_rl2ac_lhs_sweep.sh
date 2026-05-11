#!/bin/bash

. /home/oyoungquist/anaconda3/etc/profile.d/conda.sh

conda activate /home/oyoungquist/.conda/envs/genesis_lr

export SIMULATOR=genesis_pact_rl2ac

# Sanity checks w/ paper parameters
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:0 --headless --seed=1 --num_eps=2.00 --num_envs=50 --terrain_type=rough --disturbance_type=payload --log --log_path=exp_data_corl_rl2ac/sample_param_search --alpha=50 --kappa=1.2 --lambda_0=3 --k_0=20

# Conservative benchmark
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:0 --headless --seed=1 --num_eps=2.00 --num_envs=50 --terrain_type=rough --disturbance_type=payload --log --log_path=exp_data_corl_rl2ac/sample_param_search --alpha=10 --kappa=0.1 --lambda_0=0.3 --k_0=5

# Best zero-failure result from initial coarse grid sweep
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:0 --headless --seed=1 --num_eps=2.00 --num_envs=50 --terrain_type=rough --disturbance_type=payload --log --log_path=exp_data_corl_rl2ac/sample_param_search --alpha=35 --kappa=0.1 --lambda_0=0.3 --k_0=5

# Best low-kappa linear tracking result from initial coarse grid sweep
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:0 --headless --seed=1 --num_eps=2.00 --num_envs=50 --terrain_type=rough --disturbance_type=payload --log --log_path=exp_data_corl_rl2ac/sample_param_search --alpha=50 --kappa=0.1 --lambda_0=0.3 --k_0=5

# Latin-hypercube sweep: n_samples=100, seed=1
# LHS sample 000
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:0 --headless --seed=1 --num_eps=2.00 --num_envs=50 --terrain_type=rough --disturbance_type=payload --log --log_path=exp_data_corl_rl2ac/sample_param_search --alpha=30.489 --kappa=0.23179 --lambda_0=0.84738 --k_0=14.354

# LHS sample 001
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:0 --headless --seed=1 --num_eps=2.00 --num_envs=50 --terrain_type=rough --disturbance_type=payload --log --log_path=exp_data_corl_rl2ac/sample_param_search --alpha=33.461 --kappa=0.18283 --lambda_0=1.3622 --k_0=15.769

# LHS sample 002
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:0 --headless --seed=1 --num_eps=2.00 --num_envs=50 --terrain_type=rough --disturbance_type=payload --log --log_path=exp_data_corl_rl2ac/sample_param_search --alpha=24.826 --kappa=0.026764 --lambda_0=0.72971 --k_0=5.7739

# LHS sample 003
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:0 --headless --seed=1 --num_eps=2.00 --num_envs=50 --terrain_type=rough --disturbance_type=payload --log --log_path=exp_data_corl_rl2ac/sample_param_search --alpha=31.418 --kappa=0.096194 --lambda_0=1.3535 --k_0=14.929

# LHS sample 004
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:0 --headless --seed=1 --num_eps=2.00 --num_envs=50 --terrain_type=rough --disturbance_type=payload --log --log_path=exp_data_corl_rl2ac/sample_param_search --alpha=45.019 --kappa=0.15965 --lambda_0=0.31244 --k_0=4.2345

# LHS sample 005
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:0 --headless --seed=1 --num_eps=2.00 --num_envs=50 --terrain_type=rough --disturbance_type=payload --log --log_path=exp_data_corl_rl2ac/sample_param_search --alpha=31.822 --kappa=0.10369 --lambda_0=1.3375 --k_0=19.808

# LHS sample 006
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:0 --headless --seed=1 --num_eps=2.00 --num_envs=50 --terrain_type=rough --disturbance_type=payload --log --log_path=exp_data_corl_rl2ac/sample_param_search --alpha=29.004 --kappa=0.15603 --lambda_0=0.65568 --k_0=9.5916

# LHS sample 007
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:0 --headless --seed=1 --num_eps=2.00 --num_envs=50 --terrain_type=rough --disturbance_type=payload --log --log_path=exp_data_corl_rl2ac/sample_param_search --alpha=22.593 --kappa=0.1676 --lambda_0=0.11164 --k_0=11.278

# LHS sample 008
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:0 --headless --seed=1 --num_eps=2.00 --num_envs=50 --terrain_type=rough --disturbance_type=payload --log --log_path=exp_data_corl_rl2ac/sample_param_search --alpha=40.815 --kappa=0.20461 --lambda_0=1.0425 --k_0=2.6273

# LHS sample 009
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:0 --headless --seed=1 --num_eps=2.00 --num_envs=50 --terrain_type=rough --disturbance_type=payload --log --log_path=exp_data_corl_rl2ac/sample_param_search --alpha=26.091 --kappa=0.048735 --lambda_0=0.4733 --k_0=4.8092

# LHS sample 010
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:0 --headless --seed=1 --num_eps=2.00 --num_envs=50 --terrain_type=rough --disturbance_type=payload --log --log_path=exp_data_corl_rl2ac/sample_param_search --alpha=42.302 --kappa=0.098712 --lambda_0=0.99105 --k_0=17.163

# LHS sample 011
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:0 --headless --seed=1 --num_eps=2.00 --num_envs=50 --terrain_type=rough --disturbance_type=payload --log --log_path=exp_data_corl_rl2ac/sample_param_search --alpha=52.992 --kappa=0.13641 --lambda_0=1.0849 --k_0=11.577

# LHS sample 012
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:0 --headless --seed=1 --num_eps=2.00 --num_envs=50 --terrain_type=rough --disturbance_type=payload --log --log_path=exp_data_corl_rl2ac/sample_param_search --alpha=32.571 --kappa=0.11168 --lambda_0=0.7895 --k_0=9.4185

# LHS sample 013
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:0 --headless --seed=1 --num_eps=2.00 --num_envs=50 --terrain_type=rough --disturbance_type=payload --log --log_path=exp_data_corl_rl2ac/sample_param_search --alpha=35.141 --kappa=0.2147 --lambda_0=1.439 --k_0=7.1695

# LHS sample 014
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:0 --headless --seed=1 --num_eps=2.00 --num_envs=50 --terrain_type=rough --disturbance_type=payload --log --log_path=exp_data_corl_rl2ac/sample_param_search --alpha=25.647 --kappa=0.2496 --lambda_0=0.64552 --k_0=19.626

# LHS sample 015
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:0 --headless --seed=1 --num_eps=2.00 --num_envs=50 --terrain_type=rough --disturbance_type=payload --log --log_path=exp_data_corl_rl2ac/sample_param_search --alpha=27.448 --kappa=0.056221 --lambda_0=1.027 --k_0=5.2988

# LHS sample 016
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:0 --headless --seed=1 --num_eps=2.00 --num_envs=50 --terrain_type=rough --disturbance_type=payload --log --log_path=exp_data_corl_rl2ac/sample_param_search --alpha=44.487 --kappa=0.086761 --lambda_0=1.2182 --k_0=14.225

# LHS sample 017
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:0 --headless --seed=1 --num_eps=2.00 --num_envs=50 --terrain_type=rough --disturbance_type=payload --log --log_path=exp_data_corl_rl2ac/sample_param_search --alpha=53.637 --kappa=0.12934 --lambda_0=0.13678 --k_0=12.607

# LHS sample 018
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:0 --headless --seed=1 --num_eps=2.00 --num_envs=50 --terrain_type=rough --disturbance_type=payload --log --log_path=exp_data_corl_rl2ac/sample_param_search --alpha=33.135 --kappa=0.11368 --lambda_0=1.2859 --k_0=9.2641

# LHS sample 019
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:0 --headless --seed=1 --num_eps=2.00 --num_envs=50 --terrain_type=rough --disturbance_type=payload --log --log_path=exp_data_corl_rl2ac/sample_param_search --alpha=54.433 --kappa=0.070598 --lambda_0=1.4519 --k_0=19.926

# LHS sample 020
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:0 --headless --seed=1 --num_eps=2.00 --num_envs=50 --terrain_type=rough --disturbance_type=payload --log --log_path=exp_data_corl_rl2ac/sample_param_search --alpha=28.393 --kappa=0.17304 --lambda_0=1.2435 --k_0=7.6703

# LHS sample 021
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:0 --headless --seed=1 --num_eps=2.00 --num_envs=50 --terrain_type=rough --disturbance_type=payload --log --log_path=exp_data_corl_rl2ac/sample_param_search --alpha=38.831 --kappa=0.2023 --lambda_0=1.0716 --k_0=12.149

# LHS sample 022
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:0 --headless --seed=1 --num_eps=2.00 --num_envs=50 --terrain_type=rough --disturbance_type=payload --log --log_path=exp_data_corl_rl2ac/sample_param_search --alpha=25.006 --kappa=0.22935 --lambda_0=0.53934 --k_0=10.61

# LHS sample 023
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:0 --headless --seed=1 --num_eps=2.00 --num_envs=50 --terrain_type=rough --disturbance_type=payload --log --log_path=exp_data_corl_rl2ac/sample_param_search --alpha=51.913 --kappa=0.079354 --lambda_0=1.3779 --k_0=18.806

# LHS sample 024
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:0 --headless --seed=1 --num_eps=2.00 --num_envs=50 --terrain_type=rough --disturbance_type=payload --log --log_path=exp_data_corl_rl2ac/sample_param_search --alpha=22.39 --kappa=0.027363 --lambda_0=1.3112 --k_0=6.4883

# LHS sample 025
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:0 --headless --seed=1 --num_eps=2.00 --num_envs=50 --terrain_type=rough --disturbance_type=payload --log --log_path=exp_data_corl_rl2ac/sample_param_search --alpha=49.691 --kappa=0.10179 --lambda_0=0.75068 --k_0=17.619

# LHS sample 026
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:0 --headless --seed=1 --num_eps=2.00 --num_envs=50 --terrain_type=rough --disturbance_type=payload --log --log_path=exp_data_corl_rl2ac/sample_param_search --alpha=20.683 --kappa=0.044159 --lambda_0=1.4875 --k_0=16.823

# LHS sample 027
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:0 --headless --seed=1 --num_eps=2.00 --num_envs=50 --terrain_type=rough --disturbance_type=payload --log --log_path=exp_data_corl_rl2ac/sample_param_search --alpha=27.87 --kappa=0.19732 --lambda_0=0.52885 --k_0=5.59

# LHS sample 028
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:0 --headless --seed=1 --num_eps=2.00 --num_envs=50 --terrain_type=rough --disturbance_type=payload --log --log_path=exp_data_corl_rl2ac/sample_param_search --alpha=48.969 --kappa=0.071203 --lambda_0=0.19131 --k_0=10.733

# LHS sample 029
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:0 --headless --seed=1 --num_eps=2.00 --num_envs=50 --terrain_type=rough --disturbance_type=payload --log --log_path=exp_data_corl_rl2ac/sample_param_search --alpha=49.772 --kappa=0.063482 --lambda_0=0.40566 --k_0=8.4359

# LHS sample 030
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:0 --headless --seed=1 --num_eps=2.00 --num_envs=50 --terrain_type=rough --disturbance_type=payload --log --log_path=exp_data_corl_rl2ac/sample_param_search --alpha=36.852 --kappa=0.16994 --lambda_0=0.83695 --k_0=13.067

# LHS sample 031
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:0 --headless --seed=1 --num_eps=2.00 --num_envs=50 --terrain_type=rough --disturbance_type=payload --log --log_path=exp_data_corl_rl2ac/sample_param_search --alpha=30.681 --kappa=0.1786 --lambda_0=1.1669 --k_0=8.808

# LHS sample 032
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:0 --headless --seed=1 --num_eps=2.00 --num_envs=50 --terrain_type=rough --disturbance_type=payload --log --log_path=exp_data_corl_rl2ac/sample_param_search --alpha=42.849 --kappa=0.041577 --lambda_0=0.58095 --k_0=3.9076

# LHS sample 033
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:0 --headless --seed=1 --num_eps=2.00 --num_envs=50 --terrain_type=rough --disturbance_type=payload --log --log_path=exp_data_corl_rl2ac/sample_param_search --alpha=23.764 --kappa=0.19277 --lambda_0=0.2509 --k_0=19.263

# LHS sample 034
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:0 --headless --seed=1 --num_eps=2.00 --num_envs=50 --terrain_type=rough --disturbance_type=payload --log --log_path=exp_data_corl_rl2ac/sample_param_search --alpha=22.992 --kappa=0.22506 --lambda_0=0.70123 --k_0=8.8941

# LHS sample 035
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:0 --headless --seed=1 --num_eps=2.00 --num_envs=50 --terrain_type=rough --disturbance_type=payload --log --log_path=exp_data_corl_rl2ac/sample_param_search --alpha=54.904 --kappa=0.045338 --lambda_0=0.30025 --k_0=19.435

# LHS sample 036
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:0 --headless --seed=1 --num_eps=2.00 --num_envs=50 --terrain_type=rough --disturbance_type=payload --log --log_path=exp_data_corl_rl2ac/sample_param_search --alpha=20.75 --kappa=0.23957 --lambda_0=1.3217 --k_0=11.798

# LHS sample 037
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:0 --headless --seed=1 --num_eps=2.00 --num_envs=50 --terrain_type=rough --disturbance_type=payload --log --log_path=exp_data_corl_rl2ac/sample_param_search --alpha=32.614 --kappa=0.16094 --lambda_0=0.51215 --k_0=18.677

# LHS sample 038
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:0 --headless --seed=1 --num_eps=2.00 --num_envs=50 --terrain_type=rough --disturbance_type=payload --log --log_path=exp_data_corl_rl2ac/sample_param_search --alpha=38.967 --kappa=0.13218 --lambda_0=1.0596 --k_0=16.188

# LHS sample 039
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:0 --headless --seed=1 --num_eps=2.00 --num_envs=50 --terrain_type=rough --disturbance_type=payload --log --log_path=exp_data_corl_rl2ac/sample_param_search --alpha=21.898 --kappa=0.24283 --lambda_0=0.9999 --k_0=7.957

# LHS sample 040
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:0 --headless --seed=1 --num_eps=2.00 --num_envs=50 --terrain_type=rough --disturbance_type=payload --log --log_path=exp_data_corl_rl2ac/sample_param_search --alpha=43.674 --kappa=0.11568 --lambda_0=0.86982 --k_0=17.306

# LHS sample 041
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:0 --headless --seed=1 --num_eps=2.00 --num_envs=50 --terrain_type=rough --disturbance_type=payload --log --log_path=exp_data_corl_rl2ac/sample_param_search --alpha=48.231 --kappa=0.19563 --lambda_0=0.21958 --k_0=4.4636

# LHS sample 042
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:0 --headless --seed=1 --num_eps=2.00 --num_envs=50 --terrain_type=rough --disturbance_type=payload --log --log_path=exp_data_corl_rl2ac/sample_param_search --alpha=28.737 --kappa=0.034628 --lambda_0=0.59761 --k_0=3.3364

# LHS sample 043
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:0 --headless --seed=1 --num_eps=2.00 --num_envs=50 --terrain_type=rough --disturbance_type=payload --log --log_path=exp_data_corl_rl2ac/sample_param_search --alpha=26.742 --kappa=0.15099 --lambda_0=0.61972 --k_0=14.597

# LHS sample 044
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:0 --headless --seed=1 --num_eps=2.00 --num_envs=50 --terrain_type=rough --disturbance_type=payload --log --log_path=exp_data_corl_rl2ac/sample_param_search --alpha=54.172 --kappa=0.18089 --lambda_0=1.1104 --k_0=6.2888

# LHS sample 045
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:0 --headless --seed=1 --num_eps=2.00 --num_envs=50 --terrain_type=rough --disturbance_type=payload --log --log_path=exp_data_corl_rl2ac/sample_param_search --alpha=50.931 --kappa=0.13392 --lambda_0=0.67166 --k_0=7.3235

# LHS sample 046
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:0 --headless --seed=1 --num_eps=2.00 --num_envs=50 --terrain_type=rough --disturbance_type=payload --log --log_path=exp_data_corl_rl2ac/sample_param_search --alpha=43.175 --kappa=0.052857 --lambda_0=1.1076 --k_0=15.189

# LHS sample 047
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:0 --headless --seed=1 --num_eps=2.00 --num_envs=50 --terrain_type=rough --disturbance_type=payload --log --log_path=exp_data_corl_rl2ac/sample_param_search --alpha=46.106 --kappa=0.02321 --lambda_0=0.78059 --k_0=2.312

# LHS sample 048
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:0 --headless --seed=1 --num_eps=2.00 --num_envs=50 --terrain_type=rough --disturbance_type=payload --log --log_path=exp_data_corl_rl2ac/sample_param_search --alpha=30.891 --kappa=0.16291 --lambda_0=1.3939 --k_0=13.766

# LHS sample 049
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:0 --headless --seed=1 --num_eps=2.00 --num_envs=50 --terrain_type=rough --disturbance_type=payload --log --log_path=exp_data_corl_rl2ac/sample_param_search --alpha=37.739 --kappa=0.08615 --lambda_0=0.33079 --k_0=17.881

# LHS sample 050
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:0 --headless --seed=1 --num_eps=2.00 --num_envs=50 --terrain_type=rough --disturbance_type=payload --log --log_path=exp_data_corl_rl2ac/sample_param_search --alpha=32.115 --kappa=0.18484 --lambda_0=0.49819 --k_0=15.57

# LHS sample 051
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:0 --headless --seed=1 --num_eps=2.00 --num_envs=50 --terrain_type=rough --disturbance_type=payload --log --log_path=exp_data_corl_rl2ac/sample_param_search --alpha=50.508 --kappa=0.24545 --lambda_0=1.2678 --k_0=4.5793

# LHS sample 052
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:0 --headless --seed=1 --num_eps=2.00 --num_envs=50 --terrain_type=rough --disturbance_type=payload --log --log_path=exp_data_corl_rl2ac/sample_param_search --alpha=21.382 --kappa=0.21689 --lambda_0=0.98146 --k_0=5.9677

# LHS sample 053
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:0 --headless --seed=1 --num_eps=2.00 --num_envs=50 --terrain_type=rough --disturbance_type=payload --log --log_path=exp_data_corl_rl2ac/sample_param_search --alpha=47.589 --kappa=0.17514 --lambda_0=0.17692 --k_0=18.049

# LHS sample 054
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:0 --headless --seed=1 --num_eps=2.00 --num_envs=50 --terrain_type=rough --disturbance_type=payload --log --log_path=exp_data_corl_rl2ac/sample_param_search --alpha=27.263 --kappa=0.057818 --lambda_0=1.4086 --k_0=13.302

# LHS sample 055
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:0 --headless --seed=1 --num_eps=2.00 --num_envs=50 --terrain_type=rough --disturbance_type=payload --log --log_path=exp_data_corl_rl2ac/sample_param_search --alpha=44.082 --kappa=0.2344 --lambda_0=0.91853 --k_0=10.112

# LHS sample 056
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:0 --headless --seed=1 --num_eps=2.00 --num_envs=50 --terrain_type=rough --disturbance_type=payload --log --log_path=exp_data_corl_rl2ac/sample_param_search --alpha=24.315 --kappa=0.033244 --lambda_0=0.96726 --k_0=5.0111

# LHS sample 057
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:0 --headless --seed=1 --num_eps=2.00 --num_envs=50 --terrain_type=rough --disturbance_type=payload --log --log_path=exp_data_corl_rl2ac/sample_param_search --alpha=24.038 --kappa=0.2126 --lambda_0=1.2022 --k_0=15.892

# LHS sample 058
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:0 --headless --seed=1 --num_eps=2.00 --num_envs=50 --terrain_type=rough --disturbance_type=payload --log --log_path=exp_data_corl_rl2ac/sample_param_search --alpha=38.267 --kappa=0.24383 --lambda_0=0.7634 --k_0=3.5363

# LHS sample 059
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:0 --headless --seed=1 --num_eps=2.00 --num_envs=50 --terrain_type=rough --disturbance_type=payload --log --log_path=exp_data_corl_rl2ac/sample_param_search --alpha=39.899 --kappa=0.20696 --lambda_0=1.1306 --k_0=12.662

# LHS sample 060
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:0 --headless --seed=1 --num_eps=2.00 --num_envs=50 --terrain_type=rough --disturbance_type=payload --log --log_path=exp_data_corl_rl2ac/sample_param_search --alpha=41.926 --kappa=0.074853 --lambda_0=0.25789 --k_0=12.931

# LHS sample 061
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:0 --headless --seed=1 --num_eps=2.00 --num_envs=50 --terrain_type=rough --disturbance_type=payload --log --log_path=exp_data_corl_rl2ac/sample_param_search --alpha=51.261 --kappa=0.12178 --lambda_0=0.28921 --k_0=14.732

# LHS sample 062
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:0 --headless --seed=1 --num_eps=2.00 --num_envs=50 --terrain_type=rough --disturbance_type=payload --log --log_path=exp_data_corl_rl2ac/sample_param_search --alpha=25.409 --kappa=0.063739 --lambda_0=0.43359 --k_0=18.395

# LHS sample 063
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:0 --headless --seed=1 --num_eps=2.00 --num_envs=50 --terrain_type=rough --disturbance_type=payload --log --log_path=exp_data_corl_rl2ac/sample_param_search --alpha=50.389 --kappa=0.18754 --lambda_0=0.8882 --k_0=5.9586

# LHS sample 064
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:0 --headless --seed=1 --num_eps=2.00 --num_envs=50 --terrain_type=rough --disturbance_type=payload --log --log_path=exp_data_corl_rl2ac/sample_param_search --alpha=51.742 --kappa=0.15457 --lambda_0=1.4702 --k_0=8.5119

# LHS sample 065
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:0 --headless --seed=1 --num_eps=2.00 --num_envs=50 --terrain_type=rough --disturbance_type=payload --log --log_path=exp_data_corl_rl2ac/sample_param_search --alpha=34.648 --kappa=0.20078 --lambda_0=1.2943 --k_0=3.6985

# LHS sample 066
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:0 --headless --seed=1 --num_eps=2.00 --num_envs=50 --terrain_type=rough --disturbance_type=payload --log --log_path=exp_data_corl_rl2ac/sample_param_search --alpha=52.339 --kappa=0.1207 --lambda_0=0.92983 --k_0=17.831

# LHS sample 067
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:0 --headless --seed=1 --num_eps=2.00 --num_envs=50 --terrain_type=rough --disturbance_type=payload --log --log_path=exp_data_corl_rl2ac/sample_param_search --alpha=41.096 --kappa=0.22789 --lambda_0=1.1919 --k_0=9.9334

# LHS sample 068
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:0 --headless --seed=1 --num_eps=2.00 --num_envs=50 --terrain_type=rough --disturbance_type=payload --log --log_path=exp_data_corl_rl2ac/sample_param_search --alpha=40.607 --kappa=0.08125 --lambda_0=0.14475 --k_0=7.5419

# LHS sample 069
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:0 --headless --seed=1 --num_eps=2.00 --num_envs=50 --terrain_type=rough --disturbance_type=payload --log --log_path=exp_data_corl_rl2ac/sample_param_search --alpha=38.125 --kappa=0.066698 --lambda_0=0.48025 --k_0=11.094

# LHS sample 070
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:0 --headless --seed=1 --num_eps=2.00 --num_envs=50 --terrain_type=rough --disturbance_type=payload --log --log_path=exp_data_corl_rl2ac/sample_param_search --alpha=35.928 --kappa=0.037575 --lambda_0=0.80141 --k_0=6.6623

# LHS sample 071
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:0 --headless --seed=1 --num_eps=2.00 --num_envs=50 --terrain_type=rough --disturbance_type=payload --log --log_path=exp_data_corl_rl2ac/sample_param_search --alpha=45.698 --kappa=0.10692 --lambda_0=0.20471 --k_0=10.296

# LHS sample 072
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:0 --headless --seed=1 --num_eps=2.00 --num_envs=50 --terrain_type=rough --disturbance_type=payload --log --log_path=exp_data_corl_rl2ac/sample_param_search --alpha=46.836 --kappa=0.10879 --lambda_0=1.2207 --k_0=11.449

# LHS sample 073
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:0 --headless --seed=1 --num_eps=2.00 --num_envs=50 --terrain_type=rough --disturbance_type=payload --log --log_path=exp_data_corl_rl2ac/sample_param_search --alpha=36.714 --kappa=0.12367 --lambda_0=0.34327 --k_0=15.498

# LHS sample 074
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:0 --headless --seed=1 --num_eps=2.00 --num_envs=50 --terrain_type=rough --disturbance_type=payload --log --log_path=exp_data_corl_rl2ac/sample_param_search --alpha=37.437 --kappa=0.14686 --lambda_0=1.4272 --k_0=13.618

# LHS sample 075
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:0 --headless --seed=1 --num_eps=2.00 --num_envs=50 --terrain_type=rough --disturbance_type=payload --log --log_path=exp_data_corl_rl2ac/sample_param_search --alpha=23.16 --kappa=0.020711 --lambda_0=1.2542 --k_0=6.8087

# LHS sample 076
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:0 --headless --seed=1 --num_eps=2.00 --num_envs=50 --terrain_type=rough --disturbance_type=payload --log --log_path=exp_data_corl_rl2ac/sample_param_search --alpha=53.397 --kappa=0.13738 --lambda_0=0.87568 --k_0=3.1048

# LHS sample 077
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:0 --headless --seed=1 --num_eps=2.00 --num_envs=50 --terrain_type=rough --disturbance_type=payload --log --log_path=exp_data_corl_rl2ac/sample_param_search --alpha=39.279 --kappa=0.076593 --lambda_0=0.679 --k_0=9.7528

# LHS sample 078
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:0 --headless --seed=1 --num_eps=2.00 --num_envs=50 --terrain_type=rough --disturbance_type=payload --log --log_path=exp_data_corl_rl2ac/sample_param_search --alpha=29.547 --kappa=0.19025 --lambda_0=0.28078 --k_0=8.2835

# LHS sample 079
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:0 --headless --seed=1 --num_eps=2.00 --num_envs=50 --terrain_type=rough --disturbance_type=payload --log --log_path=exp_data_corl_rl2ac/sample_param_search --alpha=44.553 --kappa=0.094697 --lambda_0=0.46294 --k_0=2.1598

# LHS sample 080
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:0 --headless --seed=1 --num_eps=2.00 --num_envs=50 --terrain_type=rough --disturbance_type=payload --log --log_path=exp_data_corl_rl2ac/sample_param_search --alpha=45.513 --kappa=0.14003 --lambda_0=0.38052 --k_0=10.843

# LHS sample 081
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:0 --headless --seed=1 --num_eps=2.00 --num_envs=50 --terrain_type=rough --disturbance_type=payload --log --log_path=exp_data_corl_rl2ac/sample_param_search --alpha=36.279 --kappa=0.082934 --lambda_0=1.1365 --k_0=9.1344

# LHS sample 082
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:0 --headless --seed=1 --num_eps=2.00 --num_envs=50 --terrain_type=rough --disturbance_type=payload --log --log_path=exp_data_corl_rl2ac/sample_param_search --alpha=40.251 --kappa=0.15146 --lambda_0=0.37298 --k_0=18.263

# LHS sample 083
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:0 --headless --seed=1 --num_eps=2.00 --num_envs=50 --terrain_type=rough --disturbance_type=payload --log --log_path=exp_data_corl_rl2ac/sample_param_search --alpha=34.224 --kappa=0.059581 --lambda_0=0.56485 --k_0=16.524

# LHS sample 084
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:0 --headless --seed=1 --num_eps=2.00 --num_envs=50 --terrain_type=rough --disturbance_type=payload --log --log_path=exp_data_corl_rl2ac/sample_param_search --alpha=29.856 --kappa=0.090028 --lambda_0=0.44738 --k_0=7.8007

# LHS sample 085
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:0 --headless --seed=1 --num_eps=2.00 --num_envs=50 --terrain_type=rough --disturbance_type=payload --log --log_path=exp_data_corl_rl2ac/sample_param_search --alpha=33.672 --kappa=0.19006 --lambda_0=0.73276 --k_0=19.021

# LHS sample 086
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:0 --headless --seed=1 --num_eps=2.00 --num_envs=50 --terrain_type=rough --disturbance_type=payload --log --log_path=exp_data_corl_rl2ac/sample_param_search --alpha=46.259 --kappa=0.12801 --lambda_0=0.36072 --k_0=15.121

# LHS sample 087
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:0 --headless --seed=1 --num_eps=2.00 --num_envs=50 --terrain_type=rough --disturbance_type=payload --log --log_path=exp_data_corl_rl2ac/sample_param_search --alpha=52.552 --kappa=0.20882 --lambda_0=0.11649 --k_0=3.0569

# LHS sample 088
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:0 --headless --seed=1 --num_eps=2.00 --num_envs=50 --terrain_type=rough --disturbance_type=payload --log --log_path=exp_data_corl_rl2ac/sample_param_search --alpha=35.694 --kappa=0.031473 --lambda_0=1.02 --k_0=16.281

# LHS sample 089
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:0 --headless --seed=1 --num_eps=2.00 --num_envs=50 --terrain_type=rough --disturbance_type=payload --log --log_path=exp_data_corl_rl2ac/sample_param_search --alpha=21.509 --kappa=0.11726 --lambda_0=0.60999 --k_0=4.0698

# LHS sample 090
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:0 --headless --seed=1 --num_eps=2.00 --num_envs=50 --terrain_type=rough --disturbance_type=payload --log --log_path=exp_data_corl_rl2ac/sample_param_search --alpha=41.352 --kappa=0.22109 --lambda_0=1.4748 --k_0=5.0835

# LHS sample 091
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:0 --headless --seed=1 --num_eps=2.00 --num_envs=50 --terrain_type=rough --disturbance_type=payload --log --log_path=exp_data_corl_rl2ac/sample_param_search --alpha=20.179 --kappa=0.052135 --lambda_0=0.90939 --k_0=13.405

# LHS sample 092
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:0 --headless --seed=1 --num_eps=2.00 --num_envs=50 --terrain_type=rough --disturbance_type=payload --log --log_path=exp_data_corl_rl2ac/sample_param_search --alpha=29.289 --kappa=0.039276 --lambda_0=0.55746 --k_0=16.619

# LHS sample 093
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:0 --headless --seed=1 --num_eps=2.00 --num_envs=50 --terrain_type=rough --disturbance_type=payload --log --log_path=exp_data_corl_rl2ac/sample_param_search --alpha=42.692 --kappa=0.14551 --lambda_0=0.22898 --k_0=2.8048

# LHS sample 094
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:0 --headless --seed=1 --num_eps=2.00 --num_envs=50 --terrain_type=rough --disturbance_type=payload --log --log_path=exp_data_corl_rl2ac/sample_param_search --alpha=26.371 --kappa=0.22398 --lambda_0=0.71337 --k_0=12.351

# LHS sample 095
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:0 --headless --seed=1 --num_eps=2.00 --num_envs=50 --terrain_type=rough --disturbance_type=payload --log --log_path=exp_data_corl_rl2ac/sample_param_search --alpha=49.124 --kappa=0.14345 --lambda_0=0.40872 --k_0=2.4334

# LHS sample 096
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:0 --headless --seed=1 --num_eps=2.00 --num_envs=50 --terrain_type=rough --disturbance_type=payload --log --log_path=exp_data_corl_rl2ac/sample_param_search --alpha=48.436 --kappa=0.16501 --lambda_0=0.82067 --k_0=6.9816

# LHS sample 097
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:0 --headless --seed=1 --num_eps=2.00 --num_envs=50 --terrain_type=rough --disturbance_type=payload --log --log_path=exp_data_corl_rl2ac/sample_param_search --alpha=34.908 --kappa=0.23715 --lambda_0=0.15941 --k_0=14.036

# LHS sample 098
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:0 --headless --seed=1 --num_eps=2.00 --num_envs=50 --terrain_type=rough --disturbance_type=payload --log --log_path=exp_data_corl_rl2ac/sample_param_search --alpha=47.96 --kappa=0.21788 --lambda_0=0.94401 --k_0=12.057

# LHS sample 099
python play_exp_rl2ac_tune.py --task=go1_rl2ac --gpu=cuda:0 --headless --seed=1 --num_eps=2.00 --num_envs=50 --terrain_type=rough --disturbance_type=payload --log --log_path=exp_data_corl_rl2ac/sample_param_search --alpha=47.272 --kappa=0.092155 --lambda_0=1.1513 --k_0=17.009
