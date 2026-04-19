#!/bin/bash

. /home/oyoungquist/anaconda3/etc/profile.d/conda.sh

conda activate /home/oyoungquist/.conda/envs/genesis_lr

export SIMULATOR=genesis_pact

python train.py --task=go1_pact --headless --gpu=cuda:1 --seed=1 --pinn_loss_weight=-1.0
