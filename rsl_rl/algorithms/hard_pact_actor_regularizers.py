"""Optional actor-only geometry/load objectives; no changes to QP or execution.

Canonical joints/feet are FR, FL, RR, RL. All measured/estimated quantities
are constants. Shared actor weights can affect both output channels even when
the loss VJP enters only one channel. Nominal FF is supervised against detached
gravity/external-load stance support, without a PD/FF ratio or QP traversal.
"""
from dataclasses import dataclass
import math
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ActorRegularizerConfig:
    lambda_fk: float = 0.0
    fk_position_scale_m: float = 0.10
    fk_deadband_m: float = 0.0
    lambda_load: float = 0.0
    # Accepted only for old resolved configs; no longer affect the support loss.
    load_preferred_fraction: float = 0.75
    load_min_normalized_demand: float = 0.02
    load_allocation_regularization: float = 1e-6
    load_balance_relative_tolerance: float = 1e-5
    load_balance_absolute_tolerance: float = 1e-6
    load_force_scale_n: float = 100.0
    load_moment_scale_nm: float = 25.0
    load_epsilon: float = 1e-8

    def __post_init__(self):
        for name, value in vars(self).items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
        for name in ('fk_position_scale_m','load_allocation_regularization',
                     'load_force_scale_n','load_moment_scale_nm','load_epsilon'):
            if getattr(self,name) <= 0:
                raise ValueError(f'{name} must be positive')


def fk_consistency_loss(predicted, target, stance, valid, cfg):
    """Mean over valid stance feet and XYZ; discard invalid data BEFORE arithmetic."""
    mask = stance.detach().bool() & valid.detach().bool().reshape(-1,1)
    mask &= torch.isfinite(predicted.detach()).all(-1) & torch.isfinite(target.detach()).all(-1)
    error = predicted[mask] - target.detach()[mask]
    residual = error.sign() * (error.abs()-cfg.fk_deadband_m).clamp_min(0)
    loss = F.smooth_l1_loss(residual/cfg.fk_position_scale_m,
                            torch.zeros_like(residual), reduction='sum') / max(error.numel(),1)
    return loss, {'loss':loss.detach(), 'valid_fraction':mask.float().mean(),
        'error_m':error.detach().norm(dim=-1).sum()/mask.sum().clamp_min(1)}


@torch.no_grad()
def nominal_support_gravity(dynamics, q):
    """Nominal g(q)=RNEA(q,0,0), never h(q,v), payload, or passive springs.

    Use the existing backend conversion/model, bounded by its batch capacity.
    Pinocchio's existing worker boundary synchronizes; BARD stays on device.
    Results are owned because backend workspaces are mutable.
    """
    values=[]
    for first in range(0,q.shape[0],dynamics.batch_capacity):
        state=q[first:first+dynamics.batch_capacity].detach()
        valid=torch.isfinite(state).all(-1)&(state[:,3:7].square().sum(-1)>0)
        safe=torch.zeros_like(state);safe[:,3]=1  # simulator WXYZ identity
        state=torch.where(valid[:,None],state,safe)
        zero=state.new_zeros(state.shape[0],18)
        context=dynamics.build_context(state,zero,parameters={},need_jacobians=False)
        result=context.rnea(zero).detach().clone()
        values.append(torch.where(valid[:,None],result,float('nan')))
    return torch.cat(values)


@torch.no_grad()
def compensation_demand(foot_jacobian, base_jacobian, wrench, gravity, stance, cfg, mu,
                        diagnostics=None):
    """Minimum-norm EXACT stance support, not incremental load allocation.

    M a + g = S.T tau + J.T f + B.T W. World-aligned forces use the
    same foot/base origins as the QP Jacobians; generalized order is canonical.
    Row scaling changes neither the exact solution nor its minimum-force norm.
    No ridge: singular geometry uses a rank-aware Moore-Penrose solution and
    is accepted only if the requested balance is achievable and cone-feasible.
    """
    # Double precision only for this small detached allocation: forming C C^T
    # squares its condition number, and float32 CUDA can reject achievable
    # two-contact balance. QP/training precision remains unchanged.
    output_dtype=gravity.dtype
    J=foot_jacobian.detach().double(); B=base_jacobian.detach().double()
    W=wrench.detach().double(); g=gravity.detach().double(); stance=stance.detach().bool()
    finite=(torch.isfinite(J).flatten(1).all(1)&torch.isfinite(B).flatten(1).all(1)
            &torch.isfinite(W).all(1)&torch.isfinite(g).all(1))
    J=torch.where(finite[:,None,None,None],J,0.)*stance[:,:,None,None]
    B=torch.where(finite[:,None,None],B,0.); W=torch.where(finite[:,None],W,0.)
    g=torch.where(finite[:,None],g,0.)
    rhs_full=g-torch.einsum('bkn,bk->bn',B,W)
    C=J.flatten(1,2)[:,:,:6].transpose(1,2)
    scale=W.new_tensor([cfg.load_force_scale_n]*3+[cfg.load_moment_scale_nm]*3)
    Cn=C/scale[None,:,None]; b=rhs_full[:,:6]/scale
    gram=Cn@Cn.transpose(1,2)
    finite &= (torch.isfinite(gram).flatten(1).all(1)&torch.isfinite(b).all(1)
               &torch.isfinite(rhs_full).all(1))
    Cn=torch.where(finite[:,None,None],Cn,0.)
    b=torch.where(finite[:,None],b,0.)
    gram=torch.where(finite[:,None,None],gram,0.)
    # Batched spectral rank detection avoids per-row dispatch/synchronization.
    # eigh on CUDA may synchronize in PyTorch; no explicit host scalar reads.
    eig,U=torch.linalg.eigh(gram)
    cutoff=12*torch.finfo(Cn.dtype).eps*eig[:,-1:].clamp_min(0)
    keep=eig>cutoff
    full=keep.all(-1)&finite
    eye=torch.eye(6,device=W.device,dtype=W.dtype)
    dual,info=torch.linalg.solve_ex(torch.where(full[:,None,None],gram,eye),
        torch.where(full[:,None],b,0.)[...,None],check_errors=False)
    inv=torch.where(keep,1/torch.where(keep,eig,1.),0.)
    pseudo=U@(inv[...,None]*(U.transpose(1,2)@b[...,None]))
    solved=(~full)|(info==0)
    dual=torch.where(full[:,None,None],dual,pseudo)
    force=(Cn.transpose(1,2)@dual).squeeze(-1).reshape(-1,4,3)
    residual=(Cn@force.flatten(1)[...,None]).squeeze(-1)-b
    error=residual.norm(dim=-1)
    balanced=error <= cfg.load_balance_absolute_tolerance+cfg.load_balance_relative_tolerance*b.norm(dim=-1)
    normal=force[:,:,2]
    cone=((normal>=0)&(force[:,:,:2].abs()<=mu*normal[...,None]+1e-6).all(-1))
    friction=((~stance)|cone).all(-1)
    target=(rhs_full-torch.einsum('bfkn,bfk->bn',J,force))[:,6:].to(output_dtype)
    finite &= torch.isfinite(target).all(1)&torch.isfinite(force).flatten(1).all(1)&torch.isfinite(error)
    valid=finite&solved&balanced&friction&stance.any(-1)
    if diagnostics is not None:
        diagnostics.update(balance_residual=torch.where(finite,error,0.).mean(),
            invalid_nonfinite_fraction=(~finite).float().mean(),
            invalid_solver_fraction=(finite&~solved).float().mean(),
            invalid_balance_fraction=(finite&~balanced).float().mean(),
            invalid_friction_fraction=(finite&~friction).float().mean(),
            empty_stance_fraction=(~stance.any(-1)).float().mean(),
            rank_deficient_fraction=(~full).float().mean(),
            valid_target_fraction=valid.float().mean())
    return torch.where(valid[:,None],target,0.).to(output_dtype),valid


def load_compensation_loss(nominal_ff, demand, torque_scales, stance, valid, feasible, cfg):
    """Huber on physical nominal FF minus detached FULL support torque.

    Mean over valid stance joint coordinates (not rows); no PD subtraction,
    QP path, preferred channel ratio, or minimum-disturbance threshold.
    """
    joints=stance.detach().bool().repeat_interleave(3,dim=-1)
    rows=valid.detach().bool().reshape(-1)&feasible.detach().bool()
    rows &= torch.isfinite(nominal_ff.detach()).all(-1)&torch.isfinite(demand).all(-1)
    mask=rows[:,None]&joints
    scales=torch.as_tensor(torque_scales,device=nominal_ff.device,dtype=nominal_ff.dtype).detach()
    error=nominal_ff[mask]-demand.detach()[mask]
    normalized=error/scales.expand_as(nominal_ff)[mask]
    count=mask.sum()
    loss=F.smooth_l1_loss(normalized,torch.zeros_like(normalized),reduction='sum')/count.clamp_min(1)
    return loss,dict(loss=loss.detach(),valid_fraction=rows.float().mean(),
        valid_joint_count=count.detach(),
        error_nm=error.detach().abs().sum()/count.clamp_min(1),
        target_magnitude_nm=demand.detach()[mask].abs().sum()/count.clamp_min(1))
