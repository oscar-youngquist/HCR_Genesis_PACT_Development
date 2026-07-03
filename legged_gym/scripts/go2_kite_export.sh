#!/bin/bash

. /home/oyoungquist/anaconda3/etc/profile.d/conda.sh

conda activate /home/oyoungquist/.conda/envs/genesis_lr

export SIMULATOR=genesis_kite_depth

python export_kite_async_jit.py --task go2_kite --debug_export_untrained