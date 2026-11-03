#!/bin/bash
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$script_dir/play_pact_water.sh" --robot go2 "$@"
