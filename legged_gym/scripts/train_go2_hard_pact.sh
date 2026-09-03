#!/usr/bin/env bash
# Compatibility name retained for callers of the original HardPACT launcher.
set -eu
script_dir="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
exec "$script_dir/go2_hard_pact.sh" "$@"
