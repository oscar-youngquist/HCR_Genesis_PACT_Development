"""Repository-local warm-start extension for qpth 0.0.18.

qpth's public :func:`QPFunction` does not expose the primal/dual initializer,
although its batched PDIPM kernel keeps the complete ``(x, y, z, s)`` state.
This module derives a local copy of that kernel, changing only its initializer;
the Newton loop and qpth's implicit KKT backward remain byte-for-byte upstream.
No installed file is patched.
"""

from __future__ import annotations

import inspect
from threading import Lock

import torch
from qpth.qp import QPFunction
from qpth.solvers.pdipm import batch as pdipm_batch


_PATCH_LOCK = Lock()
_WARM_FORWARD = None


def _make_warm_forward():
    """Compile the installed PDIPM loop with an optional detached initializer."""
    source = inspect.getsource(pdipm_batch.forward)
    signature = "            maxIter=20, solver=KKTSolvers.LU_PARTIAL):"
    if signature not in source:
        raise RuntimeError(
            "Unsupported qpth PDIPM source; warm start is verified for qpth 0.0.18"
        )
    source = source.replace(
        signature,
        "            maxIter=20, solver=KKTSolvers.LU_PARTIAL, warm_start=None, warm_mask=None):",
        1,
    ).replace("def forward(", "def forward_warm(", 1)
    begin = source.index("    # Find initial values")
    end = source.index("    # Make all of the slack variables >= 1.")
    cold = source[begin:end]
    # With no cache, retain upstream's exact cold initializer. With a cache,
    # avoid paying that full batched KKT solve merely because a few rows reset:
    # invalid rows receive finite neutral seeds (x=y=0,z=s=1), while valid rows
    # retain their own terminal state. Shared post-solve certification rejects
    # a neutral seed that does not converge and cold-resolves only that row.
    # y is the equality dual, z the inequality dual, and s=h-Gx. All cached
    # tensors are detached by the caller.
    nested_cold = "\n".join(
        ("    " + line if line.strip() else line) for line in cold.splitlines()
    ) + "\n"
    replacement = (
        "    if warm_start is not None:\n"
        "        warm_x, warm_y, warm_z, warm_s = warm_start\n"
        "        row = warm_mask.reshape(-1, 1)\n"
        "        x = torch.where(row, warm_x, torch.zeros_like(warm_x))\n"
        "        y = torch.where(row, warm_y, torch.zeros_like(warm_y))\n"
        "        z = torch.where(row, warm_z, torch.ones_like(warm_z))\n"
        "        s = torch.where(row, warm_s, torch.ones_like(warm_s))\n"
        "    else:\n" + nested_cold
    )
    source = source[:begin] + replacement + source[end:]
    namespace = dict(pdipm_batch.__dict__)
    exec(compile(source, __file__, "exec"), namespace)
    return namespace["forward_warm"]


def solve_qpth_warm(Q, p, G, h, A, b, *, warm_start, warm_mask=None, eps, verbose,
                    not_improved_limit, max_iter, check_q_spd):
    """Solve through qpth and return its detached terminal PDIPM state.

    qpth resolves ``pdipm_b.forward`` during the custom autograd forward call.
    A short process-local replacement lets its unchanged ``QPFunction`` use
    the local kernel. The original symbol is restored before returning, and a
    lock prevents concurrent callers from observing the temporary adapter.
    """
    global _WARM_FORWARD
    if _WARM_FORWARD is None:
        _WARM_FORWARD = _make_warm_forward()
    captured = {}
    if warm_start is not None:
        expected = (
            (Q.shape[0], Q.shape[1]),
            (Q.shape[0], A.shape[1]),
            (Q.shape[0], G.shape[1]),
            (Q.shape[0], G.shape[1]),
        )
        if any(
            not isinstance(value, torch.Tensor)
            or tuple(value.shape) != shape
            or not torch.isfinite(value).all()
            for value, shape in zip(warm_start, expected)
        ):
            raise RuntimeError("invalid or incompatible qpth warm-start state")
        if warm_mask is None:
            warm_mask = torch.ones(
                Q.shape[0], device=Q.device, dtype=torch.bool
            )
        if tuple(warm_mask.shape) != (Q.shape[0],):
            raise RuntimeError("invalid qpth warm-start row mask")

    def run(*args, **kwargs):
        values = _WARM_FORWARD(
            *args, **kwargs, warm_start=warm_start, warm_mask=warm_mask
        )
        captured["state"] = tuple(
            value.detach() if isinstance(value, torch.Tensor) else value
            for value in (values[0], values[1], values[2], values[3])
        )
        return values

    with _PATCH_LOCK:
        original = pdipm_batch.forward
        pdipm_batch.forward = run
        try:
            solution = QPFunction(
                eps=eps, verbose=verbose,
                notImprovedLim=not_improved_limit, maxIter=max_iter,
                check_Q_spd=check_q_spd,
            )(Q, p, G, h, A, b)
        finally:
            pdipm_batch.forward = original
    return solution, captured["state"]
