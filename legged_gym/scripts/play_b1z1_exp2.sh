#!/usr/bin/env bash
set -euo pipefail
HERE="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
exec python -u "$HERE/play_b1z1_exp2.py" "$@"
