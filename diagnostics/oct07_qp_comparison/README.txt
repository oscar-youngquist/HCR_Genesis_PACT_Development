OCTOBER 7 CHECKPOINT 20000: NO QP VS EVERY-SUBSTEP HISTORICAL QP

Run: Oct07_10-22-08_hard_pact_full_isaaclab/model_20000.pt
Protocol: seed 1, ten environments, 3000 control steps (60 simulated seconds)
per condition, sequential headless execution on physical GPU 0/logical cuda:0.
Conditions: off, every_held. There is NO saved-schedule condition in this test.

The same Observer/report helpers as diagnostics/matched5500 collect tracking,
survival/reset causes, paired pre-reset windows, correction distributions,
primary/recovery/fallback outcomes, predicted/measured joint and torque
violations, model velocity changes, contact errors/lags, bounded pre-failure
traces, timing and memory. Definitions/caveats are in that harness's README.

Historical QP implementation is imported in an isolated process-local namespace
from deployments/oct07_20000/archive/rsl_rl/algorithms, including historical
assembly, recovery, acceptance, diagnostics and cuPIQP backend code. The source
hashes are verified against the exported bundle. The archived environment/config
byte-match historical commit 41d5b7b7b5ebbcef3d760bf975d1275ebb7410e4. The complete
original dirty-tree identity is unavailable. Runtime Isaac Lab/BARD and installed
cuPIQP dependencies are current and recorded, not claimed to be original binaries.

QP is 24-variable total torque/force, 92 canonical inequalities including the
5ms endpoint-torque checks; joint-only recovery is 36-variable/104 inequalities.
Use checkpoint-final objective weights (curriculum progress 1), original solver
profiles, full correction alpha=1, no QP warmup and every-substep execution.
These evaluation schedule overrides do not alter objective/constraints. The
original training schedule was random_one_substep. The added compatibility
field use_external_wrench=True only exposes the historical always-on wrench
behavior to the current environment; it changes no QP mathematics.

Physics heads/latent/contact predictions are fixed at policy rate. GRF torque
conditioning uses interval-start bounded nominal torque. At each 5ms substep,
use fresh state/mechanics/live orientation and fresh nominal PD, without
accumulating corrections. Physical limits are read from the simulator.

Fixed joint dynamics use saved range midpoints; Kp/Kd/motor multipliers=1.
Saved configuration otherwise restored before play-specific overrides: 2x5
mixed terrain, ten environments, fixed play command ranges and disabled
command/terrain curricula. This differs from the training terrain distribution.
Command/wrench/push tape and initial hashes are recorded; QP sampling RNG is
isolated. Trajectories/reset randomization diverge after condition-specific
failures. One seed is descriptive, not evidence of statistical significance.

Run/resume (completed trials skipped, interrupted trial restarted from seed):

  /home/oyoungquist/.conda/envs/lr_lab_cupiqp/bin/python diagnostics/matched5500/evaluate.py \
    --gpu 0 --steps 3000 --seeds 1 --conditions off every_held \
    --run /home/oyoungquist/Research/Genesis_Development/HCR_Genesis_PACT_Development/logs/hardpact_iclr/go2_pact_rough/Oct07_10-22-08_hard_pact_full_isaaclab \
    --checkpoint 20000 --historical-qp-bundle deployments/oct07_20000 \
    --output diagnostics/oct07_qp_comparison/results

The bounded preliminary smoke uses --steps 20 and a separate smoke output.
No training or production source modifications. GPU timing includes simulator,
Python and collection overhead/interference; it is not hardware latency.
