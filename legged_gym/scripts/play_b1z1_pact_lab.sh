#!/usr/bin/env sh
# Clear backend-specific alias; keep existing invocations working too.
exec sh "$(dirname "$0")/play_b1z1_pact_gym.sh" "$@"


# sh legged_gym/scripts/play_b1z1_pact_lab.sh \
#   --load_run Oct21_00-46-21_b1z1_pact_improved \
#   --ckpt -1 \
#   --gpu cuda:0 \
#   --follow_robot