# HardPACT solver environment record

Recorded 2026-09-02 on `CICSUNIX-US-003` (two NVIDIA GeForce RTX 4090 GPUs).
The original simulator environments were treated as immutable; solver-specific
experiments use clones or the isolated solver environment.

## Original environments

- `genesis_lr`: Python 3.10.19, Genesis 0.3.11, Torch 2.8.0+cu126.
  Import/CUDA, a real Genesis smoke, a cuPIQP canonical solve, and a HardPACT
  PPO minibatch were verified. No dependency repair was performed in this pass.
- `lr_lab`: Python 3.11.16, NumPy 1.26.0, Isaac Sim 5.1.0, Torch
  2.7.0+cu128. A post-clone check again reported those exact Python, NumPy, and
  Torch versions and CUDA availability. The environment was not migrated.

## cuPIQP clones

The official cuPIQP CUDA extra initially installed NumPy 2.4.6/CuPy 14, which
is outside Isaac Sim 5.1's NumPy range. Only in this clone:

```text
python -m pip install --force-reinstall --no-deps numpy==1.26.0
python -m pip install --force-reinstall --no-deps cupy-cuda12x==13.6.0
```

That broad-extra clone is retained only as an installation experiment and is
not used to make simulator-support claims.

`lr_lab_cupiqp_clean` was then cloned afresh from the untouched `lr_lab` and
kept on Python 3.11.16 / NumPy 1.26.0 / Torch 2.7.0+cu128 (CUDA 12.8). It has
CuPy 13.6.0. cuPIQP 0.1.0 and
its exact source dependency were installed without dependency resolution;
only the CUDA packages demanded by observed import/link errors were added.
The final sparse requirement was `nvidia-cudss-cu12==0.7.1.6`, also installed
with `--no-deps`. Isaac Sim imports still pass. On a real RTX 4090, dense
forward/implicit backward (including retained PCGrad VJPs and rollout/PPO
interleaving), zero-copy Torch/CuPy DLPack, and repeated sparse setup/update
solves all pass. The original `lr_lab` was not modified.

## Moreau environments

- `hard_pact_moreau`: isolated Python 3.12.14, Moreau 0.3.3, Torch
  2.10.0+cu128. CPU float64 forward/backward and qpth parity tests pass. cuDSS
  was pinned from 0.8.0.10 to 0.7.1.6 to distinguish the earlier cuDSS setup
  error from licensing; the CUDA solve then reached Moreau and returned the
  authoritative error `No license key found`.
- `lr_lab_moreau`: created with
  `conda create -y -n lr_lab_moreau --clone lr_lab`, then migrated **only the
  clone** to Python 3.12.14 and installed `moreau[cuda12,torch]==0.3.3`.
  Python/Torch/Moreau/CUDA imports pass. Isaac Sim 5.1's Python-3.11 wheels do
  not survive the Python-3.12 migration (`isaacsim`/`isaaclab` unavailable),
  and Moreau CUDA additionally requires a license key. Consequently this is a
  truthful unsupported capability, not a numerical fallback or an OOM result.
