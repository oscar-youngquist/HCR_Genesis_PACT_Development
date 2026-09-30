# Run from Unity Job Composer

## One-time setup

1. Build/upload `isaaclab-hcr.sif` to your project storage. An environment YAML or
   conda archive alone is not the container image.
2. Upload the updated repository (including `containers/isaaclab` and Unity scripts).
3. For Go2, keep a separate `aligned_iclr_2027_qp_pinn` checkout. Copy only
   `legged_gym/scripts/go2_hard_pact_unity.sh` into its scripts directory; the
   branch's existing `go2_hard_pact.sh` and `train_hard_pact.py` remain unchanged.

## Submit a job

Open **Jobs > Job Composer** in [Unity OnDemand](https://docs.unity.rc.umass.edu/documentation/jobs/ondemand/).
Use `job_composer.sbatch` as the job script. Edit the image/repository paths and
choose `TRAIN_SCRIPT` below, then submit. Add your Slurm account/QOS if your
allocation requires it; the template deliberately does not assume `-q long`.

| Training | TRAIN_SCRIPT |
| --- | --- |
| Retained UniFP | `b1z1_unifp_unity.sh` |
| Original-architecture UniFP | `b1z1_unifp_original_unity.sh` |
| Force-rejection UniFP | `b1z1_unifp_reject_unity.sh` |
| Coupled PACT | `b1z1_pact_unity.sh` |
| Position-only PACT | `b1z1_pact_pos_unity.sh` |
| Position-only PPO | `b1z1_ppo_pos_unity.sh` |
| Ablation 4: coupled, no conditioning, no PINNs | `b1z1_pact_ab4_coupled_none_unity.sh` |
| Ablation 5: coupled, concat, no PINNs | `b1z1_pact_ab5_coupled_concat_unity.sh` |
| Ablation 6: position, FiLM, no PINNs | `b1z1_pact_ab6_position_film_unity.sh` |
| Ablation 7: position, FiLM, both PINNs | `b1z1_pact_ab7_position_full_unity.sh` |
| Ablation 8: coupled, FiLM, no PINNs | `b1z1_pact_ab8_coupled_film_unity.sh` |
| Ablation 9: coupled, FiLM, representation PINN only | `b1z1_pact_ab9_representation_unity.sh` |
| Ablation 10: coupled, FiLM, actor PINN only | `b1z1_pact_ab10_actor_unity.sh` |
| Go2 HardPACT | `go2_hard_pact_unity.sh` |

Ablation 11 (full PACT) uses the existing `b1z1_pact_unity.sh`.
All B1Z1 launchers inherit task configuration, including PINN weights and schedules;
they do not inject a PINN-weight override. Select one `TRAIN_SCRIPT` per job and
submit separate jobs for concurrent variants. No container rebuild is needed:
the launcher uses the code in the mounted checkout.

For Go2, change `REPO` to your aligned-branch checkout. Keep `TOOLS` pointing to
the checkout containing this container helper. Default Go2 task is
`go2_hard_pact_full_isaaclab`; add `--qp_solver=cupiqp` to TRAIN_ARGS for cuPIQP.
For an ablation, also add e.g. `--task=go2_hard_pact_baseline_isaaclab`.

The first submission defaults to two environments and one PPO iteration. Once
that passes, change only TRAIN_ARGS, for example:

```bash
TRAIN_ARGS=(--seed=1 --num_envs=4096 --max_iterations=50000)
```

The job loads `apptainer/latest`, starts the container, then runs the selected
Unity `.sh` script. No host conda module, conda activation, or manual GPU selection
is needed. Keep `l40s`, not `vram32` (which also matches unsuitable GPU types).
Do not unset CUDA_VISIBLE_DEVICES. Logs/checkpoints appear under
`$PROJECT/training_runs/<job-id>/logs`; scheduler output is `isaaclab-<job-id>.out`
in the job's working directory. No trailing `wait` is needed.

Workstation launchers and training behavior are unchanged. These scripts target
the supplied container layout; a successful Unity GPU/simulator smoke test is
still required before production training.
