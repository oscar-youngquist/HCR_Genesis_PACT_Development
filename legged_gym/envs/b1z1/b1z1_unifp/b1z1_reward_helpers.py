import torch


def _sanitize_tensor(x: torch.Tensor, clamp: float = 1e6) -> torch.Tensor:
    return torch.nan_to_num(x, nan=0.0, posinf=clamp, neginf=-clamp)


def _safe_normalize(v: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    v = _sanitize_tensor(v)
    denom = torch.linalg.norm(v, dim=dim, keepdim=True)
    denom = torch.clamp(_sanitize_tensor(denom), min=eps)
    return _sanitize_tensor(v / denom)


def _safe_symmetrize(M: torch.Tensor) -> torch.Tensor:
    M = _sanitize_tensor(M)
    return 0.5 * (M + M.transpose(-1, -2))


def _skew(v: torch.Tensor) -> torch.Tensor:
    """v: (..., 3) -> (..., 3, 3)"""
    v = _sanitize_tensor(v)
    O = torch.zeros_like(v[..., 0])
    vx, vy, vz = v[..., 0], v[..., 1], v[..., 2]
    return torch.stack([
        torch.stack([ O, -vz,  vy], dim=-1),
        torch.stack([ vz,  O, -vx], dim=-1),
        torch.stack([-vy, vx,  O], dim=-1),
    ], dim=-2)


def _safe_inv(M: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    n = M.shape[-1]
    I = torch.eye(n, device=M.device, dtype=M.dtype).expand_as(M)
    M_reg = _safe_symmetrize(M) + eps * I
    return _sanitize_tensor(torch.linalg.inv(M_reg))


def _safe_pinv(M: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    n = M.shape[-1]
    I = torch.eye(n, device=M.device, dtype=M.dtype).expand_as(M)
    M_reg = _safe_symmetrize(M) + eps * I
    return _sanitize_tensor(torch.linalg.pinv(M_reg))


def _eig_desc(S: torch.Tensor, eps: float = 1e-8):
    """Symmetric eigendecomposition with descending eigenvalues."""
    evals, evecs = torch.linalg.eigh(_safe_symmetrize(S))  # ascending
    evals = torch.clamp(_sanitize_tensor(evals), min=eps)
    evecs = _safe_normalize(evecs, dim=-2, eps=eps)
    idx = torch.argsort(evals, dim=-1, descending=True)
    evals = torch.gather(evals, 1, idx)
    evecs = torch.gather(evecs, 2, idx.unsqueeze(1).expand(-1, evecs.shape[1], -1))
    return _sanitize_tensor(evals), _sanitize_tensor(evecs)


def _geom_mean(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    x = torch.clamp(_sanitize_tensor(x), min=eps)
    return _sanitize_tensor(torch.exp(torch.mean(torch.log(x), dim=dim)))


def _interval_reward(x: torch.Tensor, lo: float, hi: float, sharpness: float = 2.0):
    x = _sanitize_tensor(x)
    low = torch.relu(lo - x)
    high = torch.relu(x - hi)
    return _sanitize_tensor(torch.exp(-sharpness * (low**2 + high**2)))


def _upper_reward(x: torch.Tensor, hi: float, sharpness: float = 2.0):
    x = _sanitize_tensor(x)
    viol = torch.relu(x - hi)
    return _sanitize_tensor(torch.exp(-sharpness * viol**2))


def _rot_x(self, theta: torch.Tensor) -> torch.Tensor:
    """Batched rotation matrix about x-axis.

    Args:
        theta: Shape (N,)

    Returns:
        R: Shape (N, 3, 3)
    """
    c = torch.cos(theta)
    s = torch.sin(theta)

    N = theta.shape[0]
    R = torch.zeros(N, 3, 3, device=theta.device, dtype=theta.dtype)

    R[:, 0, 0] = 1.0
    R[:, 1, 1] = c
    R[:, 1, 2] = -s
    R[:, 2, 1] = s
    R[:, 2, 2] = c

    return R


def _rot_y(self, theta: torch.Tensor) -> torch.Tensor:
    """Batched rotation matrix about y-axis.

    Args:
        theta: Shape (N,)

    Returns:
        R: Shape (N, 3, 3)
    """
    c = torch.cos(theta)
    s = torch.sin(theta)

    N = theta.shape[0]
    R = torch.zeros(N, 3, 3, device=theta.device, dtype=theta.dtype)

    R[:, 0, 0] = c
    R[:, 0, 2] = s
    R[:, 1, 1] = 1.0
    R[:, 2, 0] = -s
    R[:, 2, 2] = c

    return R


def _rot_z(self, theta: torch.Tensor) -> torch.Tensor:
    """Batched rotation matrix about z-axis.

    Args:
        theta: Shape (N,)

    Returns:
        R: Shape (N, 3, 3)
    """
    c = torch.cos(theta)
    s = torch.sin(theta)

    N = theta.shape[0]
    R = torch.zeros(N, 3, 3, device=theta.device, dtype=theta.dtype)

    R[:, 0, 0] = c
    R[:, 0, 1] = -s
    R[:, 1, 0] = s
    R[:, 1, 1] = c
    R[:, 2, 2] = 1.0

    return R