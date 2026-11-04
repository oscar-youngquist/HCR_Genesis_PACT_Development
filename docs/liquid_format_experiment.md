# Opt-in liquid-format experiment

Run from the repository root using the existing patched environment and CUDA 0:

```bash
NUMBA_CACHE_DIR=/tmp/liquid-numba MPLCONFIGDIR=/tmp/liquid-mpl \
  /home/oyoungquist/.conda/envs/genesis_lr/bin/python -m liquid_collection.validation.run \
  --output runs/liquid_format_experiment
MPLCONFIGDIR=/tmp/liquid-mpl \
  /home/oyoungquist/.conda/envs/genesis_lr/bin/python -m liquid_collection.validation.analyze \
  runs/liquid_format_experiment
```

The runner freezes `criteria.json` before launching workers. `--criteria FILE`
accepts a complete custom criteria object; do not change it after inspecting data.
Defaults are two environments, one episode capped at six seconds per environment,
three deterministic payloads shared across Go1/Go2, 450 seconds per worker and
2700 seconds total. `--prepare-only` writes the plan without simulation.
`--platforms go1-dact` or `go2-dact` limits an explicitly requested diagnostic run.
There are no automatic retries. Failed workers retain durable chunks; the analyzer
skips running workers and reports missing/insufficient data as inconclusive.
Particle snapshots are disabled. The production launcher remains at 200 Hz.

The three payloads use midpoint dimensions/properties, with contrasting fill
clearance offsets 0.032/0.082 m and viscosity 0.001/0.025. All use the existing
expanded bounds, unexpanded mounting bounds, and actual spawn validation.
Selected stairs of -0.04, +0.04, -0.08 m exercise supported terrain variation,
including negative world height. This tests a limited terrain subset, not general
rough-terrain coverage. The central spawn platform is flat; terrain provenance
retains the actual heights and origins so traversal is not assumed.

Validation-only commands enter immediately before the normal observation builder,
so its history deque advances once per policy step. The consumed command channels
are asserted against the recorded commands before inference. The episode schedule
is stand until 1.6 s, forward at 0.4 m/s until 3.0 s, stop until 4.2 s, then forward
and turn. Values are clipped to the checkpoint's saved bounds. Heading mode uses
a wrapped target heading and the player's 0.5 heading-error gain and yaw limits.
Termination and midpoint joint dynamics now share the play-script helper. The
explicit collection episode cap overrides only the player episode duration.
Controller clocks, box contacts, particles, patched Genesis and displays are unchanged.

## Same-trajectory reduction

Native mode saves only aggregate states and coupling impulses. Offline replay uses
`GridAccumulator`, retaining the primary 5 ms endpoints, exact sparse events,
terminal/reset records and off-grid partial intervals. Wrenches sum in a common
world frame and a fixed world-origin moment reference. The result is shifted to
the *endpoint* base origin, then rotated. A separate NumPy audit checks each
interval's sums, endpoints, durations, masks, shifts, totals and event contents;
it does not run the reducer a second time. Streams, source indices, replay code and reducer code are hashed. The frozen
absolute reduction tolerance (2e-5) is in impulse units; average labels are
multiplied by their durations before checking it.

## Reconstruction contract

`validation/reconstruction.py` accepts only one isolated episode's saved states,
events and intervals. Compact and native reference objects are built separately.
Position uses cubic Hermite interpolation with saved position/linear velocity.
Orientation uses sign-continuous quaternion Hermite interpolation with endpoint
quaternion derivatives formed from saved angular velocity, followed by analytic
normalization. Differentiating these interpolants defines reconstructed velocity,
angular velocity and accelerations consistently. This is **offline interpolation
using the next saved endpoint**; it is not an online estimator. It does not change
the causal TCN prediction contract.

Recorded r, v and internal S each use their own C2 cubic spline fitted over the
entire valid saved episode prefix. These aggregate fits therefore also use future
**compact** endpoints offline; no native intermediate samples enter them. In particular,
v is not defined as r-dot. Held acceleration estimates remain right-continuous
piecewise-constant *world* vectors at their exact event times. Their rotating-frame
derivatives are `-omega cross held_B`; no future held update is interpolated.
The other channel derivatives are `-omega cross g_B`,
`a_B - omega cross u_B`, and `alpha_B`.

A wrench label is used only on its own `(start,end]` interval. Its world force and
fixed-world-origin moment averages are constant there, then shifted and rotated
at the query pose. The value exactly at an endpoint belongs to the interval that
just ended, including for trapezoidal quadrature. This convention is explicit:
it is not a claim that an interval average is instantaneous truth. The native
reference uses the same convention at 0.25 ms, with a separate midpoint-linear
sensitivity check. Sensitivity failure makes instantaneous adequacy inconclusive.
Kinematic-reference sensitivity also compares native Hermite derivatives against
recorded native endpoint velocities and completed-interval velocity differences.
This exposes amplification of float32 world-position quantization by second
derivatives at 0.25 ms spacing; neither convention is silently treated as exact
instantaneous acceleration.

JVP collocation uses the exact connected complement of all 249 shifted knot/event
exclusion intervals, with 0.1 ms margins. Intervals are sampled proportional to
length and points uniformly within them. All taps share an eligible episode and
valid held inputs. Analytical channel/history derivatives are compared with
vector automatic differentiation and a deterministic nonlinear test map; centered
finite differences at 10, 20 and 40 microseconds stay inside the exclusion margin.
No network is trained.

## Physics and decisions

Local residuals use the requested Newton–Euler equations with independently
recorded coupling force/moment and independent aggregate interpolants. Integrated
residuals use exactly 41 endpoints over 40 5-ms intervals and trapezoidal quadrature.
They are compared both with native-resolution trapezoids and native quantities on
the same coarse grid, separating reference quadrature from reconstruction effects.
Independent angular closure subtracts moving-CoM orbital angular momentum at
**both** endpoints; a fixed endpoint base-origin moment is not used as a CoM torque.

L is the Euclidean diagonal of the nominal interior dimensions. Scales are mg,
mgL, sqrt(gL), sqrt(g/L), g and g/L; derivative scales divide channel scales by
sqrt(L/g). Integrated residual scales are L, mg*0.2 and mgL*0.2. Provisional limits
are 5% normalized RMS and 10% p95 reconstruction error, 10% conservation RMS,
5% excess over native floor, and 1e-5 JVP plumbing error. Native-reference
sensitivity RMS is 5%. Matched interior metrics sample a deterministic random phase in each 5-ms
interval rather than always sampling the midpoint (which can alias native coupling).
Metrics also report bias, maxima, peak magnitude/timing,
and error divided by the changing wrench RMS rather than gravity alone.

Each environment must supply at least 100 histories, 100 valid sampled collocation
points, 40 integrated windows, and 20 policy observations in each maneuver phase.
Windows are evaluated every 20 ms; overlapping windows/points are correlated and
are not independent statistical trials. Failed coverage cannot become a pooled
pass. Observed threshold failures are retained even when the case-level verdict
is inconclusive. Production readiness is never automatically promoted.

The experiment is an engineering diagnostic, not a proof over the payload range.
Native SPH conservation errors remain measured floors. On insufficient coverage,
fix the demonstrated configuration/contact blocker before collecting more data.
If reconstruction fails with adequate coverage, test a better interpolant or
selected additional native pose/velocity/wrench channels; keep the original JVP
and integrated objectives and do not switch the whole production format blindly.
