#!/bin/bash

. /home/oyoungquist/anaconda3/etc/profile.d/conda.sh

conda activate /home/oyoungquist/.conda/envs/genesis_lr

python build_contained_seaborn_eval_csvs.py --exp_folder exp_data_corl --approaches go1_pact go1_tau go1_pos go1_rl2ac go1_abl1 go1_abl2 go1_abl3 --disturbances none payload push --output_dir plot_data