# Compliance recovery failures

If the previous reference leaves the current workspace, recovery tests the
nominal target, previous reference, measured EE position, and workspace midpoint.
Failure of this finite candidate set is not proof that no feasible point exists.

When all four fail, only affected environments terminate at the end of the
current control interval. The issued action and actual disturbances still execute
once through the normal physical path; no mid-transition reset occurs. A finite,
workspace-valid midpoint is used solely to finish reference bookkeeping. It is
not certified collision-free and must not be interpreted as a successful target.
Existing terminal-row masks exclude these transitions from actor physics losses.
Ordinary terminal reward handling is retained. These are failures, not timeouts,
even if the episode clock expires simultaneously; timeout bootstrapping is disabled.

`infos["compliance_recovery_failed"]` contains a detached per-environment mask.
`infos["compliance_recovery_rejections"]` retains the pre-reset boolean checks,
with shape `[environment, candidate, check]`. Candidate order is the four points
above; check order is workspace, EE geometry, forbidden box, ground. True means
rejected. Non-recovering rows have false entries. Partial resets clear only the
corresponding live buffers, not these transition snapshots.

Episode TensorBoard statistics under `compliance_recovery/` report failure and
per-check rejection fractions among resetting environments (not all training
steps). No bounds, geometry exclusions, physical loads, or gains are relaxed.
Applicable PACT-derived baselines/ablations inherit the shared termination path.
