# Go2 HardPACT full Isaac Lab training benchmark: QP and BARD PINN

## Experiment

- Script: `scripts/benchmark_hard_pact_training.py`.
- Training command: `scripts/go2_hard_pact.sh --task go2_hard_pact_full_isaaclab --headless --num_envs 4096 --max_iterations 5 --seed 1 --gpu cuda:0 --qp_solver cupiqp --qp_rollout_chunk_size 4096 --qp_ppo_chunk_size 4096 --profile_bard_timing --benchmark_bard_active --pinn_loss_weight -1`.
- Conditions: real Isaac Lab, 4096 environments, per-substep cuPIQP, differentiable replay, and inverse/rollout BARD losses active from iteration zero on an RTX 4090.
- Purpose: test whether the maximum standalone cuPIQP batch remains viable in the complete HardPACT training pipeline. It does not: the retained BARD ABA graph exhausts 24 GiB before iteration 0 completes.

## Results

| Solver | Rollout chunk | PPO chunk | Status | Iteration mean s | Iteration std s | Steps/s | BARD inverse ms/update | BARD rollout ms/update |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| cupiqp | 4096 | 4096 | OOM | nan | nan | nan | nan | nan |
