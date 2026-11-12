#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
# Run inside the Isaac Lab container; preserve Slurm's assigned GPU visibility.
export SIMULATOR=isaaclab_b1z1_pact
# Match B1Z1 launchers: forwarded --seed overrides this default (also on resume).
exec python -u train_b1z1_facet.py --seed=1 --gpu=cuda:0 --headless "$@"

# bash legged_gym/scripts/b1z1_facet_unity.sh --seed 1
