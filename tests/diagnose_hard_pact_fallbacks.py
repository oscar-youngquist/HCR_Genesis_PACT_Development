"""Read-only fallback classification around the bounded real training smoke.

No constraint/solver setting changes. At most eight primal-failed rows per
mode/phase are checked independently for LP feasibility after training.
"""
import json
import runpy
import sys
from pathlib import Path
import torch
from legged_gym.scripts.train_hard_pact import prepare_solver_runtime
prepare_solver_runtime(sys.argv[1:])
from rsl_rl.algorithms.hard_pact_qp import HardPACTDifferentiableQP

stats={};packets={}
def bucket(qp):
    key=qp.cfg.qp_update_mode+'/'+qp._diagnostics_phase
    return key,stats.setdefault(key,{})
def add(s,key,value):s[key]=s.get(key,0)+int(value)

solve=HardPACTDifferentiableQP.solve
def checked_solve(self,*a,**kw):
    result=solve(self,*a,**kw)
    key,s=bucket(self);d=result.diagnostics
    failed=~result.differentiated_mask
    add(s,'rows',failed.numel());add(s,'certified',(~failed).sum())
    remaining=failed.clone()
    for name in ('nonfinite_input','empty_torque_intersection','empty_qdd_intersection','mechanics'):
        flag=d['failure/'+name]
        add(s,'reason/'+name,(remaining&flag).sum());remaining &= ~flag
    for name,flag in (
        ('solver_exception',d['full/solver_exception']),
        ('nonfinite_output',d['full/attempted']&~d['full/output_finite']),
        ('primal_certification',~torch.isfinite(d['selected/inequality_max']) |
            (d['selected/inequality_max']>self._profile(kw.get('differentiable',False))['feasibility']) |
            (d['selected/equality_max']>self._profile(kw.get('differentiable',False))['feasibility'])),
    ):
        add(s,'reason/'+name,(remaining&flag).sum());remaining &= ~flag
    p=self._profile(kw.get('differentiable',False))
    gap_bad=(~torch.isfinite(d['full/duality_gap'])|~torch.isfinite(d['full/duality_gap_rel'])|
        ((d['full/duality_gap']>p['gap_abs'])&(d['full/duality_gap_rel']>p['gap_rel'])))
    add(s,'reason/gap_requirement',(remaining&gap_bad).sum() if p['gap_policy']=='require' else 0)
    if p['gap_policy']=='require':remaining &= ~gap_bad
    add(s,'reason/unclassified',remaining.sum())
    return result
HardPACTDifferentiableQP.solve=checked_solve

build=HardPACTDifferentiableQP._build
def checked_build(self,data):
    m=build(self,data);key,s=bucket(self)
    empty=m.qdd_lower>m.qdd_upper
    add(s,'empty_joint_coordinates',empty.sum())
    if empty.any():
        q=data['joint_position'].detach();v=data['joint_velocity'].detach()
        _,qmin,qmax,vmax=self._limits(data['tau_nom'])
        dt=data['dt'].reshape(-1,1);beta=self.cfg.position_integration_coefficient
        amax=q.new_tensor(self.cfg.joint_acceleration_limits_rad_s2)
        lowers=torch.stack((-amax.expand_as(q),(-vmax-v)/dt,(qmin-q-dt*v)/(beta*dt.square())),0)
        uppers=torch.stack((amax.expand_as(q),(vmax-v)/dt,(qmax-q-dt*v)/(beta*dt.square())),0)
        li=lowers.argmax(0);ui=uppers.argmin(0)
        for i,name in enumerate(('acceleration','velocity','position')):
            add(s,'empty_lower_source/'+name,(empty&(li==i)).sum())
            add(s,'empty_upper_source/'+name,(empty&(ui==i)).sum())
        add(s,'empty_joint_current_position_outside',(empty&((q<qmin)|(q>qmax))).sum())
        add(s,'empty_joint_current_velocity_outside',(empty&(v.abs()>vmax)).sum())
    return m
HardPACTDifferentiableQP._build=checked_build

certificate=HardPACTDifferentiableQP._certificate
@torch.no_grad()
def checked_certificate(self,m,z,tolerance):
    result=certificate(self,m,z,tolerance);key,s=bucket(self)
    residual=(m.G@z[...,None]).squeeze(-1)-m.h
    for name,sl in (('torque_rate',slice(0,24)),('joint',slice(24,48)),('friction',slice(48,68))):
        maxima=residual[:,sl].clamp_min(0).amax(-1)
        add(s,'primal_group/'+name,(maxima>tolerance).sum())
        finite=maxima[torch.isfinite(maxima)]
        if finite.numel():s['normalized_max/'+name]=max(s.get('normalized_max/'+name,0),float(finite.max()))
    stored=packets.setdefault(key,[])
    ids=(~result[0]).nonzero().flatten()[:max(0,8-len(stored))]
    for i in ids.tolist():stored.append((m.G[i].detach().double().cpu().numpy(),m.h[i].detach().double().cpu().numpy()))
    return result
HardPACTDifferentiableQP._certificate=checked_certificate

finished=False
def finish():
    global finished
    if finished:return
    finished=True
    from scipy.optimize import linprog
    for key,rows in packets.items():
        s=stats[key]
        for G,h in rows:
            if not torch.as_tensor(G).isfinite().all() or not torch.as_tensor(h).isfinite().all():
                add(s,'lp/nonfinite_problem',1);continue
            r=linprog([0.]*24,A_ub=G,b_ub=h,bounds=[(None,None)]*24,method='highs',
                options={'time_limit':5.,'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
            if r.success and (G@r.x-h).max()<=1e-7:add(s,'lp/verified_feasible',1)
            elif r.status==2:add(s,'lp/infeasible',1)
            else:add(s,'lp/undetermined',1)
    print('FALLBACK_DIAGNOSIS '+json.dumps(stats,sort_keys=True),flush=True)

# Isaac Sim app.close may terminate the process rather than unwind Python.
# Emit the read-only report before close, without modifying the shared smoke.
import builtins
original_print=builtins.print
def report_print(*args,**kwargs):
    original_print(*args,**kwargs)
    if args and str(args[0])=='QP_STALL_SMOKE_PASS':finish()
builtins.print=report_print
try:
    runpy.run_path(str(Path(__file__).with_name('smoke_hard_pact_qp_stall.py')),run_name='__main__')
finally:
    finish()
    builtins.print=original_print
