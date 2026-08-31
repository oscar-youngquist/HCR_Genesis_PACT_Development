#!/bin/bash

set -euo pipefail

backend="${1:-genesis}"
if [[ $# -ge 1 ]]; then shift; fi
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec bash "$script_dir/go2_hard_pact.sh" "$backend" go2_hard_pact_pos "$@"
