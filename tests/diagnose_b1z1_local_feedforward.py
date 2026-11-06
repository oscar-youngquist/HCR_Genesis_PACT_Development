"""Capture one tiny smoke update, or replay teacher inputs without learning.

Capture: SIMULATOR=isaaclab_b1z1_pact PYTHONPATH=. python -u tests/diagnose_b1z1_local_feedforward.py --task b1z1_pact --gpu cuda:0 --headless
Replay: SIMULATOR=isaaclab PYTHONPATH=. python tests/diagnose_b1z1_local_feedforward.py --replay
Files live in /tmp/b1z1_teacher_diagnosis; production settings are untouched.
"""
import os
os.environ.setdefault('TORCHINDUCTOR_COMPILE_THREADS','1')
import sys
import json
import csv
from pathlib import Path
from types import SimpleNamespace
import torch
import legged_gym.envs  # Register config modules before the PPO/task-registry imports.
from rsl_rl.algorithms import b1z1_local_feedforward as local
from rsl_rl.algorithms.ppo_b1z1_pact import PPO_B1Z1PACT

OUT=Path('/tmp/b1z1_teacher_diagnosis')


@torch.no_grad()
def describe(a, actions, state, fixed, acceleration, ids, dofs, data):
    target,bias,coupling,external,valid,_=local.target(a,actions,state,fixed,acceleration,ids,dofs,data)
    n=a.actor_critic.num_actions;g=ids+6;cfg=a.cfg
    pd=a._coupled_torque(actions[:,:n].clamp(-cfg['clip_actions'],cfg['clip_actions']),state)[:,ids]
    mass=fixed.mass_matrix[:,g][:,:,g];ha=fixed.local_h[:,:,g]
    desired_arm=torch.linalg.solve(mass,(target+pd-bias-coupling+external)[...,None]).squeeze(-1)
    a0=torch.linalg.solve(mass,pd[...,None]).squeeze(-1)
    au=acceleration.index_fill(1,g,0.)
    err=data['local_ref_position_start']-fixed.local_s
    vel=data['local_ref_velocity_start']-(fixed.local_h@state[:,26:51,None]).squeeze(-1)
    desired=data['local_ref_acceleration']+cfg['actor_phys_compliance_omega']**2*err+2*cfg['actor_phys_compliance_zeta']*cfg['actor_phys_compliance_omega']*vel
    residual=(ha@desired_arm[...,None]).squeeze(-1)+(fixed.local_h@au[...,None]).squeeze(-1)+fixed.local_bias-desired
    rhs_parts=torch.stack(((ha@a0[...,None]).squeeze(-1),desired,
        (fixed.local_h@au[...,None]).squeeze(-1)+fixed.local_bias),-1)
    corrections=ha.transpose(1,2)@torch.linalg.solve(ha@ha.transpose(1,2)
        +cfg['actor_phys_compliance_dls_damping']**2*torch.eye(3,device=state.device,dtype=state.dtype),rhs_parts)
    retained_pd=a0-corrections[:,:,0]
    ref_correction=corrections[:,:,1]
    motion_correction=-corrections[:,:,2]
    q=state[:,7:26][:,ids];v=state[:,32:51][:,ids];dt=cfg['dt'];margin=cfg['actor_phys_compliance_joint_margin']
    nextq=q+dt*v+.5*dt**2*desired_arm;nextv=v+dt*desired_arm
    pos_bad=(nextq<data['q_min'][:,ids]+margin)|(nextq>data['q_max'][:,ids]-margin)
    vel_bad=nextv.abs()>data['qd_max'][:,ids]
    limits=state[:,97:116][:,ids].abs()*state[:,155:156].abs()*actions.new_tensor(cfg['torque_action_scale']).expand(n)[ids].abs()*cfg['clip_actions']
    ff_bad=target.abs()>limits;total_bad=(target+pd).abs()>data['torque_max'][:,ids]
    def norm(x):return float(x.norm(dim=-1).mean())
    result=dict(rows=len(actions),valid_fraction=float(valid.float().mean()),position_failure=float(pos_bad.any(-1).float().mean()),
        velocity_failure=float(vel_bad.any(-1).float().mean()),ff_failure=float(ff_bad.any(-1).float().mean()),total_torque_failure=float(total_bad.any(-1).float().mean()),
        position_error_m=norm(err),velocity_error_mps=norm(vel),reference_accel_mps2=norm(data['local_ref_acceleration']),
        desired_task_accel=norm(desired),task_acceleration_residual=norm(residual),nonarm_task_accel=norm((fixed.local_h@au[...,None]).squeeze(-1)),
        a0_norm=norm(a0),retained_pd_acceleration_norm=norm(retained_pd),reference_correction_norm=norm(ref_correction),
        nonarm_bias_correction_norm=norm(motion_correction),arm_acceleration_norm=norm(desired_arm),pd_nm=norm(pd),teacher_nm=norm(target),bias_nm=norm(bias),coupling_nm=norm(coupling),
        per_joint_ff_failure=ff_bad.float().mean(0).tolist(),per_joint_velocity_failure=vel_bad.float().mean(0).tolist(),
        per_joint_position_failure=pos_bad.float().mean(0).tolist(),per_joint_total_failure=total_bad.float().mean(0).tolist(),
        per_joint_teacher_abs_nm=target.abs().mean(0).tolist(),per_joint_ff_limit_nm=limits.mean(0).tolist(),
        per_joint_execution_limit_nm=data['torque_max'][:,ids].mean(0).tolist(),
        mean_abs_generalized_acceleration=acceleration.abs().mean(0).tolist(),
        mass_condition=float(torch.linalg.cond(fixed.mass_matrix).median()))
    return result


def replay():
    p=torch.load(OUT/'inputs.pt',map_location='cuda:0',weights_only=False)
    a=SimpleNamespace(cfg=p['cfg'],actor_critic=SimpleNamespace(num_actions=p['count'],action_mode='coupled'))
    a._coupled_torque=lambda act,s:PPO_B1Z1PACT._coupled_torque(a,act,s)
    actions,state,fixed,acc,ids,dofs,data=(p[k] for k in ('actions','state','fixed','acceleration','ids','dofs','data'))
    fixed=SimpleNamespace(**fixed);basecfg=dict(a.cfg);results={}
    def evaluate(name,ac=actions,s=state,m=fixed,au=acc,d=data,omega=None,damping=None):
        a.cfg=dict(basecfg)
        if omega is not None:a.cfg['actor_phys_compliance_omega']=omega
        if damping is not None:a.cfg['actor_phys_compliance_dls_damping']=damping
        results[name]=describe(a,ac,s,m,au,ids,dofs,d)
    evaluate('captured')
    # Primary experiment: hold every recorded input/model quantity fixed.
    # Damping's residual cost matters: improving coverage alone is not enough.
    for omega in (0.,.5,1.,2.,3.,4.,6.,8.,12.):
        for damping in (.001,.005,.01,.025,.05,.1,.2,.5):
            evaluate(f'omega{omega}_dls{damping}',omega=omega,damping=damping)
    # Freeze a recorded configuration, zero its velocities, and request a hold
    # exactly at the model EE pose. Recompute bias/kinematics, never reuse h(v).
    from legged_gym.dynamics import BardB1Z1DynamicsBackend
    from rsl_rl.algorithms.b1z1_bard_pinn import mechanics
    meta=p['model'];backend=BardB1Z1DynamicsBackend(meta['urdf'],meta['dofs'],meta['feet'],meta['ee'],meta['base'],device='cuda:0',batch_capacity=len(state))
    s=state.clone();s[:,26:51]=0
    m=mechanics(backend,dict(rollout_initial_state=s[:,:51],dynamics_state=s))
    for k,v in local.kinematics(backend,s,m.ee_jacobian).items():setattr(m,k,v)
    ac=torch.zeros_like(actions);ac[:,:p['count']]=(s[:,7:7+p['count']]-s[:,156:156+p['count']])/a.cfg['position_action_scale']
    d={**data,'local_ref_position_start':m.local_s.clone(),'local_ref_velocity_start':torch.zeros_like(m.local_s),
       'local_ref_acceleration':torch.zeros_like(m.local_s),'feedforward_applied_force':torch.zeros_like(m.local_s)}
    evaluate('static_hold',ac=ac,s=s,m=m,au=torch.zeros_like(acc),d=d)
    for omega in (1.,3.,6.):
        ref=d['local_ref_position_start'].clone();ref[:,0]+=.1
        evaluate('static_10cm_x_omega'+str(omega),ac=ac,s=s,m=m,au=torch.zeros_like(acc),d={**d,'local_ref_position_start':ref},omega=omega)
    output=dict(metadata=meta,arm_indices=ids.tolist(),
        original_omega=basecfg['actor_phys_compliance_omega'],original_dls=basecfg['actor_phys_compliance_dls_damping'],results=results)
    (OUT/'results.json').write_text(json.dumps(output,indent=2,allow_nan=False))
    scalar_keys=[k for k,v in results['captured'].items() if not isinstance(v,list)]
    with (OUT/'sweep.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['scenario']+scalar_keys);writer.writeheader()
        for name,values in results.items():writer.writerow({'scenario':name,**{k:values[k] for k in scalar_keys}})
    for name,v in results.items():print(name,json.dumps({k:v[k] for k in ('valid_fraction','position_failure','velocity_failure','ff_failure','total_torque_failure','a0_norm','arm_acceleration_norm','coupling_nm','desired_task_accel')}))


def capture():
    import smoke_b1z1_local_feedforward as smoke
    from legged_gym import LEGGED_GYM_ROOT_DIR
    original=local.target;make=smoke.task_registry.make_alg_runner;metadata={};calls=0
    def runner(*args,**kwargs):
        result=make(*args,**kwargs);env=result[0].env;c=env.cfg
        metadata.update(urdf=c.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR),dofs=c.asset.dof_names,feet=c.asset.foot_name,ee=c.asset.gripper_name,base=c.asset.base_name)
        metadata['sim_armature']=env.simulator._joint_armature.cpu().tolist()
        metadata['sim_passive_damping']=env.simulator._joint_damping.cpu().tolist()
        return result
    def record(a,actions,state,fixed,acceleration,ids,dofs,data):
        nonlocal calls
        if calls==0:
            payload=dict(cfg=dict(a.cfg),count=a.actor_critic.num_actions,actions=actions.detach().cpu(),state=state.detach().cpu(),
                fixed={k:v.detach().cpu() for k,v in vars(fixed).items()},acceleration=acceleration.detach().cpu(),ids=ids.cpu(),dofs=dofs,
                data={k:v.detach().cpu() for k,v in data.items()},model=metadata)
            OUT.mkdir(parents=True,exist_ok=True);torch.save(payload,OUT/'inputs.pt')
            print('DIAGNOSTIC_CAPTURE '+str(OUT/'inputs.pt'),flush=True)
        calls+=1
        return original(a,actions,state,fixed,acceleration,ids,dofs,data)
    local.target=record;smoke.task_registry.make_alg_runner=runner
    smoke.main(num_envs=8,steps=8)


if __name__=='__main__':
    if '--replay' in sys.argv:replay()
    else:capture()
