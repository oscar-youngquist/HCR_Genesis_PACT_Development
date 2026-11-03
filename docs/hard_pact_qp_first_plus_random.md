# First Plus Random Substep QP

Set the existing `algorithm.hard_pact_qp` dictionary entry:

```python
'qp_update_mode': 'first_plus_random_substep',
```

The default remains `random_one_substep`. With four physics substeps per
policy interval, the new mode solves every environment at k=0 and again at
K sampled from {1,2,3}. Later-index counts differ by at most one across
environments; random offsets/permutations refresh every policy interval.
Only selected rows build mechanics and enter each later batched solve.

Before execution, an independent fair coin per environment selects k=0 or K
as the single PPO replay anchor. Selection never depends on solver success;
failed/recovery solves retain their existing masks. Existing sampled-state,
previous executed torque, controller and delayed-action records are reused.
Mechanics matrices are reconstructed during replay as before. There is no
second stored transition, differentiated solve, loss multiplier, or change to
the existing PPO row/epoch budget or normalization.

Unselected substeps recompute fresh nominal torque with the existing bounds
and optional rate clamp. No correction is held or accumulated. Blending,
recovery/fallback and feedforward reconstruction are unchanged.

Existing rollout solve-count diagnostics report two solves per environment;
`sampled_substep/*_count` reports replay-anchor coverage (approximately half
at zero and one sixth at each later phase). Deployment metadata records the
new selection semantics. Existing modes retain their selection and RNG use.
