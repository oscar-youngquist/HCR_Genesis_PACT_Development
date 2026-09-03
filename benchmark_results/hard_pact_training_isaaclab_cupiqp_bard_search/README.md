# Go2 HardPACT full Isaac Lab training benchmark: QP and BARD PINN

## Experiment

- Script: `scripts/benchmark_hard_pact_training.py`.
- Training command: `/home/oyoungquist/Research/Genesis_Development/HCR_Genesis_PACT_Development/scripts/go2_hard_pact.sh --task go2_hard_pact_full_isaaclab --headless --num_envs 4096 --max_iterations 5 --seed 1 --gpu cuda:0 --qp_solver cupiqp --qp_rollout_chunk_size 2048 --qp_ppo_chunk_size 2048 --profile_bard_timing --benchmark_bard_active --pinn_loss_weight -1`.
- Conditions: registered full HardPACT task, real simulator, 4096 environments by default, five PPO iterations, per-substep QP execution, differentiable QP replay, and both BARD PINN objectives when requested. GPU memory is sampled externally.
- Purpose: measure end-to-end suitability for the intended training workload. Unlike the standalone benchmark, this includes simulation, rollout QPs, BARD dynamics, PPO, PCGrad, and optimizer work.

## Results

| Solver | Rollout chunk | PPO chunk | Status | Iteration mean s | Iteration std s | Steps/s | BARD inverse ms/update | BARD rollout ms/update |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| cupiqp | 2048 | 2048 | OOM | nan | nan | nan | nan | nan |
| cupiqp | 1024 | 1024 | OOM | nan | nan | nan | nan | nan |
| cupiqp | 512 | 512 | OOM | nan | nan | nan | nan | nan |
