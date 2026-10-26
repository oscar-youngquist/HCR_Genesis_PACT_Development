#!/usr/bin/env bash
set -euo pipefail
# Use the caller's activated Isaac Lab Python environment, on workstation or HPC.
HERE="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
exec python -u "$HERE/eval_b1z1_exp1.py" "$@"
