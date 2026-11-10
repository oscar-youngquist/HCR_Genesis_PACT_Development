"""Optional actor-only geometry/load objectives; no changes to QP or execution.

Canonical joints/feet are FR, FL, RR, RL. All measured/estimated quantities
are constants. Shared actor weights can affect both output channels even when
the loss VJP enters only one channel. Load compensation encourages sufficient
effort, not an identifiable disturbance-specific incremental torque or PD ratio.
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
    load_preferred_fraction: float = 0.75
    load_min_normalized_demand: float = 0.02
    load_allocation_regularization: float = 1e-6
    load_balance_relative_tolerance: float = 0.05
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
def compensation_demand(foot_jacobian, base_jacobian, wrench, reference_force, stance, cfg, mu):
    """Allocate df with C df = -w_base, then d = -(Jc.T df + Jb.T W)_joint.

    Jacobians use world-aligned frame axes at their existing origins, and
    generalized columns use canonical body-base/joint velocities. No torso
    joint-block shortcut: the floating-base balance must pass certification.
    Test the resulting TOTAL reference+df forces against the existing cone.
    This small regularized 6x6 solve is detached, not another contact QP.
    """
    J=foot_jacobian.detach(); B=base_jacobian.detach(); W=wrench.detach()
    stance=stance.detach().bool(); f=reference_force.detach()
    finite=(torch.isfinite(J).flatten(1).all(1)&torch.isfinite(B).flatten(1).all(1)
            &torch.isfinite(W).all(1)&torch.isfinite(f).flatten(1).all(1))
    J=torch.where(finite[:,None,None,None],J,0.)*stance[:,:,None,None]
    B=torch.where(finite[:,None,None],B,0.);W=torch.where(finite[:,None],W,0.)
    f=torch.where(finite[:,None,None],f,0.)
    generalized=torch.einsum('bkn,bk->bn',B,W)
    C=J.flatten(1,2)[:,:,:6].transpose(1,2)
    scale=W.new_tensor([cfg.load_force_scale_n]*3+[cfg.load_moment_scale_nm]*3)
    Cn=C/scale[None,:,None]; rhs=-generalized[:,:6]/scale
    eye=torch.eye(6,device=W.device,dtype=W.dtype)
    gram=Cn@Cn.transpose(1,2)+cfg.load_allocation_regularization*eye
    # Finite inputs can still overflow during products/scaling. Invalid rows
    # enter the factorization as I * 0, never NaN/Inf; they remain ineligible.
    finite &= (torch.isfinite(generalized).all(-1)
               &torch.isfinite(Cn).flatten(1).all(-1)
               &torch.isfinite(gram).flatten(1).all(-1)
               &torch.isfinite(rhs).all(-1))
    gram=torch.where(finite[:,None,None],gram,eye)
    rhs=torch.where(finite[:,None],rhs,0.)
    Cn=torch.where(finite[:,None,None],Cn,0.)
    # Cn Cn^T + ridge*I is SPD mathematically. Use its structure instead of
    # batched LU (cublasSgetrfBatched). Reject numerical factorization failures
    # without changing the ridge or silently accepting an approximate solve.
    factor,info=torch.linalg.cholesky_ex(gram,check_errors=False)
    solved=finite&(info==0)&torch.isfinite(factor).flatten(1).all(-1)
    factor=torch.where(solved[:,None,None],factor,eye)
    dual=torch.cholesky_solve(torch.where(solved[:,None],rhs,0.)[...,None],factor)
    solved &= torch.isfinite(dual).flatten(1).all(-1)
    dual=torch.where(solved[:,None,None],dual,0.)
    df=(Cn.transpose(1,2)@dual).squeeze(-1).reshape(-1,4,3)
    balance=(Cn@df.flatten(1)[...,None]).squeeze(-1)-rhs
    balance_norm=balance.norm(dim=-1);rhs_norm=rhs.norm(dim=-1)
    balanced=(torch.isfinite(balance_norm)&torch.isfinite(rhs_norm)
              &(balance_norm<=cfg.load_balance_relative_tolerance*rhs_norm.clamp_min(cfg.load_epsilon)))
    total=torch.where(stance[...,None],f,0.)+df
    normal=total[:,:,2]
    cone=(normal>=0)&(total[:,:,:2].abs()<=mu*normal[...,None]+1e-6).all(-1)
    feasible=(((~stance)|cone).all(-1)&stance.any(-1)&balanced&solved
              &torch.isfinite(total).flatten(1).all(-1))
    d=-(torch.einsum('bfkn,bfk->bn',J,df)+generalized)[:,6:]
    feasible &= torch.isfinite(d).all(-1)
    return torch.where(feasible[:,None],d,0.), feasible


def load_compensation_loss(post_ff, demand, torque_scales, stance, valid, feasible, cfg):
    """Directional hinge, normalized by sqrt(number of stance joints).

    No PD subtraction or channel-ratio target occurs in this objective.
    Feedforward is the existing physical post-QP command, not a raw action.
    """
    joints=stance.detach().bool().repeat_interleave(3,dim=-1)
    mask=valid.detach().bool().reshape(-1)&feasible.detach().bool()
    mask &= torch.isfinite(post_ff.detach()).all(-1)&torch.isfinite(demand).all(-1)
    rows=mask.nonzero(as_tuple=True)[0]
    scales=torch.as_tensor(torque_scales,device=post_ff.device,dtype=post_ff.dtype).detach()
    # Configuration/controller guarantees positive physical scales.
    d=(demand.detach()[rows]/scales)*joints[rows]
    tau=(post_ff[rows]/scales)*joints[rows]
    norm=d.norm(dim=-1); divisor=joints[rows].sum(-1).clamp_min(1).sqrt()
    supported=norm/divisor>cfg.load_min_normalized_demand
    d=d[supported];tau=tau[supported];norm=norm[supported];divisor=divisor[supported]
    direction=d/(norm[:,None]+cfg.load_epsilon)
    contribution=(direction*tau).sum(-1)/divisor
    required=norm/divisor
    hinge=(cfg.load_preferred_fraction*required-contribution).clamp_min(0)
    loss=hinge.square().sum()/max(hinge.numel(),1)
    count=hinge.new_tensor(hinge.numel())
    # Same normalized hinge direction expressed as a physical torque unit
    # vector. Retain the loss's sqrt(stance-joint count) normalization. With
    # uniform torque scale T this is approximately hinge*T (epsilon aside).
    physical_direction_gain=(direction.detach()/scales).norm(dim=-1).clamp_min(cfg.load_epsilon)
    shortfall_nm=(hinge.detach()/physical_direction_gain).sum()/count.clamp_min(1)
    return loss, {'loss':loss.detach(),'valid_fraction':count/max(post_ff.shape[0],1),
        'shortfall_nm':shortfall_nm,
        'demand':required.sum()/count.clamp_min(1),
        'contribution':contribution.detach().sum()/count.clamp_min(1),
        'active_hinge_fraction':(hinge.detach()>0).float().sum()/count.clamp_min(1)}
