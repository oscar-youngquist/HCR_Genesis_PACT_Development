MATCHED CHECKPOINT-5500 EVALUATION (evaluation-only)

Production checkout: /home/oyoungquist/Research/Genesis_Development/HCR_HardPact
Checkpoint: Nov13_18-50-32_hard_pact_rss_full_isaaclab/model_5500.pt
No production source edits, optimizer steps, or network training.

Run/resume sequential sweep (physical GPU 0, updated at user's request):
  /home/oyoungquist/.conda/envs/lr_lab_cupiqp/bin/python diagnostics/matched5500/evaluate.py \
    --gpu 0 --steps 3000 --seeds 1 2 3 --output diagnostics/matched5500/results

The parent process selects CUDA_VISIBLE_DEVICES=0; each child uses logical cuda:0.
It loads the saved resolved config and checkpoint-effective QP weights. Explicit
play overrides: 10 environments, 2x5 mixed terrain, terrain/command curricula
disabled, commands vx +/-1, vy +/-0.5, yaw +/-1, resampling every 10 seconds,
heading command disabled. Play terrain difficulty distribution differs from
training's level curriculum. No common pre-QP prefix is inserted.

Fixed joint friction/damping/stiffness use saved final-range midpoints; armature
uses its range midpoint. Kp/Kd/motor multipliers are exactly one. Range endpoints
are made equal so resets cannot re-randomize these values. Assertions repeat
every 250 steps. Other domain randomization is retained.

Conditions:
  off: QP disabled.
  saved_held: saved execution schedule; control-rate GRF predictions.
  every_held: QP every physics substep; control-rate GRF predictions.
  every_fresh_nominal: fresh PD + original nominal physical FF conditions GRF.
  every_fresh_previous: fresh PD + previous actually executed torque minus its
    PD component conditions GRF. This FF is an input to the decoder only, NOT
    a held/accumulating controller correction. Substep zero uses nominal FF.
Both fresh paths clip conditioning to existing physical torque limits. Latent,
explicit/contact and wrench remain fixed at control rate. The QP reference stays
the fresh bounded nominal policy torque. All QP conditions use alpha=1.

RNG and pairing:
The existing task samplers create a common absolute-control-time tape of velocity
commands, persistent wrenches, and velocity impulses. QP sampling uses an isolated
forked RNG; observation noise has a separate per-control-step stream. Initial
state/randomized-property and scenario hashes are saved and compared. This is a
matched-tape extension of the old smoke, not bit-for-bit reproduction of its RNG
trajectory. After condition-specific resets, physical trajectories, episode-local
filters, and property/reset randomization diverge. Do not interpret later samples
as identical-state counterfactuals.

Metrics:
  Ordinary RMSE uses the previous experiment's post-step non-reset samples.
  Shared RMSE uses nonoverlapping complete 1-second windows before either member
  of each condition pair first resets. Both use the same physical commands;
  shared values use owned pre-reset terminal velocities. Coverage is explicit.
  Resets include per-predicate overlapping causes; they need not sum to failures.
  Physical failure takes priority for diagnostic timeout classification.
  Episode KM curves censor timeouts and the observation horizon. Episodes and
  environments are correlated; curves are descriptive, not independent trials.
  Contact Brier error uses the existing simulator contact processor. Detection
  lag matches same-type transitions one-to-one within +/-100ms within an episode;
  positive lag means prediction is late. Unmatched events are counted, not zero lag.
  Model velocity changes are 5ms constant-derivative body-frame predictions,
  compared with live body/root-link velocities before/after actual physics.
  Raw candidate and executed-torque model predictions retain optimized forces;
  nominal predictions use gated GRF references. Torque-only acceleration changes
  hold mechanics, wrench, and optimized force fixed. Command-direction summaries
  exclude commands with planar speed <=0.1m/s. They are model decompositions, not
  causal attribution of the measured robot's motion.
  Position/velocity violations use original limits (count thresholds 1e-4 rad,
  1e-3 rad/s); torque violations use 1e-4 Nm. Raw magnitudes are retained. Candidate
  position/velocity diagnostics use the saved one-step integration convention;
  they do not replace the solver's production certification.

Artifacts:
  protocol.json, evaluation_source/: exact protocol and harness source snapshot.
  condition_seed*/manifest.json: checkpoint hash, code SHA/diff, saved/resolved
    configs, effective QP, dependencies, initial/scenario hashes.
  trace.npz: bounded 60-second compact control/substep data for offline analysis.
  prefailure.npz: up to 20 owned windows of <=0.5 seconds, clipped at episode start;
    the report states how many failure windows were omitted by this cap.
  progress.json: atomic partial progress, timing, allocator memory and GPU status.
  result.json: completion marker plus per-seed aggregates.
  comparison.csv, paired.json and plots: regenerated after each completed trial.

Resume skips completed trials. An interrupted trial restarts from its seed; the
partial trace is not claimed to restore PhysX state. Do not mix modified harness
versions in one results directory. A nonzero child exit OR missing completion
marker stops the sweep. Existing failure logs are retained on retry.

Timing:
100 control steps are warmup for GPU-event timing only; all samples still enter
behavior metrics. GPU events span end-to-end collection including CPU gaps and
artifact writes, NOT isolated solver kernels. Torch allocated/reserved peaks
exclude PhysX/Warp allocations; nvidia-smi status supplies shared-GPU context.
Simulator throughput is not hardware deployment latency.

Interpretation plan (only after results exist):
  (a) saved_held vs every_held probes execution-frequency mismatch;
  (b) every_held vs every_fresh_nominal probes stale conditioning;
  (c) every_fresh_nominal vs every_fresh_previous isolates prior-corrected FF;
  (d) residual lateral prediction errors/contact lag and pre-failure corrections
      can motivate a contact/model/objective hypothesis, not prove its cause.
No significance claims from three seeds. A next experiment should isolate the
largest remaining discrepancy rather than assuming faster decoders cure it.
