#!/usr/bin/env bash
set -euo pipefail
# Activate the IsaacLab conda environment before launching.
# GPU selection is set directly by --gpu below.
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"
SIMULATOR="${SIMULATOR:-isaaclab}" python -m legged_gym.scripts.train --task go2_dreamwaq --headless --gpu cuda:1 "$@"
