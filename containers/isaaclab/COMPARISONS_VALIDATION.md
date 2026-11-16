# Local container validation — 2026-11-15

Reused `/home/oyoungquist/isaaclab-apptainer-build/isaaclab-hcr-v2.sif`
(not a new Docker image). Apptainer 1.5.4; Torch 2.7.0+cu128, NumPy 1.26.0,
Isaac Sim 5.1.0.0, Isaac Lab 0.54.2. Current `rescaled_qp_exps` working source
was mounted, including uncommitted changes; the transfer bundle records hashes.

Executed sequentially with `CUDA_VISIBLE_DEVICES=1`, approved NVIDIA EULA
acceptance, and the actual `comparisons.sh` → `run.sh` production path:

| Check | Result |
|---|---|
| BR `check` | CUDA/autograd and production CLI configuration validation passed |
| BR `smoke --seed 1` | teacher_student, stage 1, one update, 8 interactions, final model_1.pt |
| MULE `smoke --seed 1` | nominal_only, one update, 8 interactions, final model_1.pt |

Both GPU smokes used two environments and four rollout steps, production
rewards and smoke-local optimizer/terrain sizing. Losses were finite. Before
updates, policy ratio=1 and KL=0. BR load-estimation loss=9.6086 and latent
matching=1.1831. MULE velocity MSE=.07782, reconstruction=.54039, KL=.084453.

BR measured training-loop wall time 2.370 s (3.375 transitions/s), peak Torch
allocated 51,578,880 bytes; MULE 1.660 s (4.819 transitions/s), peak Torch
allocated 41,850,368 bytes. **These are cold, tiny smokes on a shared GPU, not
throughput benchmarks.** Peaks exclude PhysX, Kit, driver and other processes;
do not use them to size production VRAM. Scene startup is outside these loop
times. GPU 1 already had ~15 GiB used by another workload before testing.

Logs/checkpoints on the source machine:

* `/tmp/hcr_comparisons_br_check.log`
* `/tmp/hcr_comparisons_br_smoke.log`
* `/tmp/hcr_comparisons_br_smoke/logs/beyond_robustness/Nov15_18-27-31_br/`
* `/tmp/hcr_comparisons_mule_smoke.log`
* `/tmp/hcr_comparisons_mule_smoke/logs/mule/Nov15_18-28-36_mule/`

Kit printed file-watch `errno=28` warnings while both updates completed. The
host had ample disk space: these messages concern change-watch registration,
consistent with an exhausted host inotify watch budget under concurrent Kit
workloads. No host sysctl or production simulator setting was changed.

Not exercised here: remote hosts, Unity execution, Docker build/run, full-sized
4096-environment training, payload curriculum advancement, adaptive/student
reinforcement phases, container checkpoint resume, or offline asset availability.
The two short runs cannot validate those cases. Existing method tests cover
many learning/resume semantics independently; run `check` and `smoke` on every
destination before a full job.
