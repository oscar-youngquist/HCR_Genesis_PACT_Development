import torch


def _skew(v: torch.Tensor) -> torch.Tensor:
    """v: (..., 3) -> (..., 3, 3)"""
    O = torch.zeros_like(v[..., 0])
    vx, vy, vz = v[..., 0], v[..., 1], v[..., 2]
    return torch.stack([
        torch.stack([ O, -vz,  vy], dim=-1),
        torch.stack([ vz,  O, -vx], dim=-1),
        torch.stack([-vy, vx,  O], dim=-1),
    ], dim=-2)


def _safe_inv(M: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    n = M.shape[-1]
    I = torch.eye(n, device=M.device, dtype=M.dtype)
    return torch.linalg.inv(M + eps * I)


def _safe_pinv(M: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    n = M.shape[-1]
    I = torch.eye(n, device=M.device, dtype=M.dtype)
    return torch.linalg.pinv(M + eps * I)


def _eig_desc(S: torch.Tensor):
    """Symmetric eigendecomposition with descending eigenvalues."""
    evals, evecs = torch.linalg.eigh(S)  # ascending
    idx = torch.argsort(evals, dim=-1, descending=True)
    evals = torch.gather(evals, 1, idx)
    evecs = torch.gather(evecs, 2, idx.unsqueeze(1).expand(-1, evecs.shape[1], -1))
    return evals, evecs


def _geom_mean(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    return torch.exp(torch.mean(torch.log(torch.clamp(x, min=eps)), dim=dim))


def _interval_reward(x: torch.Tensor, lo: float, hi: float, sharpness: float = 2.0):
    low = torch.relu(lo - x)
    high = torch.relu(x - hi)
    return torch.exp(-sharpness * (low**2 + high**2))


def _upper_reward(x: torch.Tensor, hi: float, sharpness: float = 2.0):
    viol = torch.relu(x - hi)
    return torch.exp(-sharpness * viol**2)