"""Optional dynamic-batch tensor kernels. No solver state or CUDA graphs.

Only pure numerical blocks compile; recovery dispatch and solver ownership stay
eager. Scalar schedule weights are tensors, avoiding guards on curriculum values.
"""
from functools import lru_cache
import torch


def row_scale(matrix, rhs):
    scale = torch.maximum(matrix.detach().square().sum(-1).sqrt(),
                          rhs.detach().abs()).clamp_min(1.)
    return matrix / scale.unsqueeze(-1), rhs / scale, scale


def add_residual(Q, p, C, e, weight):
    return (Q + 2 * weight * C.transpose(1,2) @ C,
            p + 2 * weight * (C.transpose(1,2) @ e[...,None]).squeeze(-1))


def certify(A, b, G, h, z, tolerance):
    eq = (A @ z[...,None]).squeeze(-1)-b
    iq = (G @ z[...,None]).squeeze(-1)-h
    er = eq.abs().amax(-1) if eq.shape[-1] else z.new_zeros(z.shape[0])
    ir = iq.clamp_min(0).amax(-1) if iq.shape[-1] else z.new_zeros(z.shape[0])
    return torch.isfinite(z).all(-1) & (er<=tolerance) & (ir<=tolerance), er, ir


@lru_cache(None)
def compiled(name):
    # Inductor's default mode does not enable CUDA graphs. Dynamic batch sizes
    # share kernels; structurally different primary/recovery matrices specialize.
    fn = torch.compile(globals()[name], dynamic=True, fullgraph=True,
                       options={"triton.cudagraphs": False})
    def call(*args):
        # Torch 2.7 AOT's donation check is evaluated again at backward time;
        # changing its process-global flag would affect unrelated training.
        # Compile rollout/certification only; keep retained PPO VJPs eager.
        if torch.is_grad_enabled() and any(x.requires_grad for x in args if isinstance(x,torch.Tensor)):
            return globals()[name](*args)
        return fn(*args)
    return call


def kernel(name, enabled):
    return compiled(name) if enabled else globals()[name]
