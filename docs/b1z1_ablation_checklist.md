# B1Z1 Legged-Manipulation Ablation Checklist

All run scripts below are in `legged_gym/scripts/` and use IsaacLab on Unity.
Set `TRAIN_SCRIPT` in the Job Composer batch script to the listed filename.
Mark each stage independently: change `[ ]` to `[x]` after training or experimental
validation is complete. Launcher checks and smoke tests do not count as experimental
validation.

| ID | Experiment | Action mode | Conditioning | Representation PINN | Actor PINN | Run script (`TRAIN_SCRIPT`) | Training Running | Training Done | Experiments Running | Experimental Validation Done |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | Vanilla position-only PPO | Position | Current observation only; no context encoder | Off | Off | `b1z1_ppo_pos_unity.sh` | [x] | [ ] | [ ] | [ ] |
| Baseline | Original-architecture UniFP tracking | Position | UniFP context; no FiLM | Off | Off | `b1z1_unifp_original_unity.sh` | [x] | [ ] | [ ] | [ ] |
| Baseline | UniFP-Reject | Position | UniFP context with estimated-force compensation; no FiLM | Off | Off | `b1z1_unifp_reject_unity.sh` | [x] | [ ] | [ ] | [ ] |
| 4 | Coupled PACT without force/error conditioning | Coupled | Common context only; no FiLM or appended condition | Off | Off | `b1z1_pact_ab4_coupled_none_unity.sh` | [ ] | [ ] | [ ] | [ ] |
| 5 | Coupled PACT with concatenated conditioning | Coupled | Concat | Off | Off | `b1z1_pact_ab5_coupled_concat_unity.sh` | [ ] | [ ] | [ ] | [ ] |
| 6 | Position PACT without PINNs | Position | FiLM | Off | Off | `b1z1_pact_ab6_position_film_unity.sh` | [ ] | [ ] | [ ] | [ ] |
| 7 | Position PACT with both PINNs | Position | FiLM | On | On | `b1z1_pact_ab7_position_full_unity.sh` | [ ] | [ ] | [ ] | [ ] |
| 8 | Coupled PACT without PINNs | Coupled | FiLM | Off | Off | `b1z1_pact_ab8_coupled_film_unity.sh` | [ ] | [ ] | [ ] | [ ] |
| 9 | Coupled PACT with representation PINN only | Coupled | FiLM | On | Off | `b1z1_pact_ab9_representation_unity.sh` | [ ] | [ ] | [ ] | [ ] |
| 10 | Coupled PACT with actor PINN only | Coupled | FiLM | Off | On | `b1z1_pact_ab10_actor_unity.sh` | [ ] | [ ] | [ ] | [ ] |
| 11 | Full coupled PACT | Coupled | FiLM | On | On | `b1z1_pact_unity.sh` | [x] | [ ] | [ ] | [ ] |

## Notes

- Numeric IDs follow `rsl_rl/b1z1_pact_ablations.py`. That matrix only encodes
  IDs 4-11, so the three baseline rows are named rather than assigned an
  unverified numeric ordering.
- Coupled control uses desired-position and feedforward-torque actions.
  Position control executes PD control without learned feedforward torque.
- Concat appends the same detached force/tracking-error condition used by FiLM.
  Variant 4 still retains the shared PACT context input and supervised decoders.
- Enabled PINNs retain their configured weights, gates, and warmup schedules;
  "On" does not mean active at full strength from iteration zero.
- Variants 6/7 use the shared PACT training stack, not the legacy
  `b1z1_pact_pos` task. The legacy `b1z1_pact_pos_unity.sh` and retained
  `b1z1_unifp_unity.sh` remain available but are not rows in this experiment set.
- Launchers inherit task settings and forward extra training arguments. Submit
  each experiment as a separate job with its own output directory.
