"""Pure, CPU-testable Figure 1 scenario math (SI units, world-frame forces)."""
import re
from pathlib import Path
import numpy as np

RUN = 'Nov13_09-58-29_b1z1_pact_improved'
CONFIG = dict(mass_kg=1.5, handle_spacing_m=.6, spring_n_m=150., damping_ns_m=10.,
              force_limit_n=45., attach_s=4., lift_s=6., release_s=8., guide_s=10.,
              hold_s=16., lift_m=.15, guidance_m=.65, horizon_s=20.,
              nominal_ee_tolerance_m=.15, base_drift_tolerance_m=.15,
              upward_yield_min_m=.015, recovery_tolerance_m=.06,
              recruitment_displacement_min_m=.06, capability_gate_min=.2,
              arm_motion_limit_m=.4, finite_force_tolerance_n=1e-5)
PROFILES = {
    'nominal': [0.]*12,
    'support-follow': [.6,.6,.6,.6,.05,.8,.8,.8,.8,.8,0.,0.],
    'adjust-hold': [.6,.6,.6,.6,.05,.8,0.,0.,0.,0.,0.,0.],
}


def checkpoint_path(run, checkpoint=None):
    run=Path(run).expanduser().resolve()
    if run.name != RUN or not run.is_dir():
        raise ValueError(f'Expected the exact run directory {RUN}: {run}')
    files={int(m.group(1)):p for p in run.iterdir()
           if p.is_file() and (m:=re.fullmatch(r'model_(\d+)\.pt',p.name))}
    if not files: raise FileNotFoundError(f'No numeric model checkpoints in {run}')
    number=max(files) if checkpoint is None else int(checkpoint)
    if number not in files: raise FileNotFoundError(f'No model_{number}.pt in {run}')
    return files[number],number


def smooth(t, start, end):
    u=np.clip((t-start)/(end-start),0.,1.)
    return u*u*u*(10+u*(-15+6*u)), 30*u*u*(1-u)**2/(end-start)


def partner_motion(t, scenario, config=CONFIG):
    """Offsets in initial-heading coordinates; vertical is world z."""
    p=np.zeros(3);v=np.zeros(3)
    if scenario=='support-follow':
        up,du=smooth(t,config['lift_s'],config['lift_s']+.8)
        down,dd=smooth(t,config['release_s'],config['release_s']+.8)
        p[2]=config['lift_m']*(up-down);v[2]=config['lift_m']*(du-dd)
    u,du=smooth(t,config['guide_s'],config['hold_s'])
    axis=0 if scenario=='support-follow' else 1
    p[axis]=config['guidance_m']*u;v[axis]=config['guidance_m']*du
    return p,v


def spring_force(point, velocity, handle, handle_velocity, k, damping, limit):
    """Force ON the endpoint; the payload receives exactly its negative."""
    force=k*(handle-point)+damping*(handle_velocity-velocity)
    return force*np.minimum(1.,limit/np.maximum(np.linalg.norm(force,axis=-1,keepdims=True),1e-12))


def audit_coupling(rows):
    """Check recorded world-frame handle wrenches, not an independent force sensor."""
    x=rows['substep_interaction'].reshape(-1,21)
    x=x[np.isfinite(x).all(1)]
    if not len(x):
        return dict(active_substeps=0,force_balance_max_n=None,moment_balance_max_nm=None)
    handles=x[:,3:9].reshape(-1,2,3)
    offsets=handles-handles.mean(1,keepdims=True)  # Box origin equals its CoM.
    expected_moment=np.cross(offsets,-np.stack((x[:,15:18],x[:,18:21]),1)).sum(1)
    return dict(active_substeps=len(x),
                force_balance_max_n=float(np.abs(x[:,9:12]+x[:,15:18]+x[:,18:21]).max()),
                moment_balance_max_nm=float(np.abs(x[:,12:15]-expected_moment).max()))


def summarize(rows, scenario, config=CONFIG):
    """One trial, no post-failure padding. Unobserved phases stay unavailable."""
    t=rows['time']; xyz=rows['ee_world']; nominal=rows['nominal_world']
    base=rows['base_world']; error=np.linalg.norm(xyz-nominal,axis=-1)
    def avg(value, mask):
        selected=value[mask]; selected=selected[np.isfinite(selected)]
        return float(selected.mean()) if len(selected) else None
    baseline=(t>=config['attach_s']+1)&(t<config['lift_s'])
    lift=(t>=config['lift_s']+.8)&(t<config['release_s'])
    recovery=(t>=config['release_s']+1)&(t<config['guide_s'])
    late=t>=config['hold_s']
    z0=avg(xyz[:,2]-nominal[:,2],baseline)
    zup=avg(xyz[:,2]-nominal[:,2],lift)
    zr=avg(xyz[:,2]-nominal[:,2],recovery)
    drift=np.linalg.norm(base[:,:2]-base[0,:2],axis=-1)
    joint_gate=avg(rows['joint_gate'].max(-1),late)
    # Guidance-relative movement excludes settling drift. A gate already active
    # before guidance is not evidence of increasing capability restriction.
    before_guidance=(t>=config['guide_s']-1)&(t<config['guide_s'])
    gate_before=avg(rows['joint_gate'].max(-1),before_guidance)
    guidance_progress=None
    if before_guidance.any() and late.any():
        q=rows['base_quat_xyzw'][0]
        yaw=np.arctan2(2*(q[3]*q[2]+q[0]*q[1]),1-2*(q[1]**2+q[2]**2))
        direction=np.array([np.cos(yaw),np.sin(yaw)]) if scenario=='support-follow' else np.array([-np.sin(yaw),np.cos(yaw)])
        guidance_progress=float((base[late,:2].mean(0)-base[before_guidance,:2].mean(0))@direction)
    result=dict(samples=len(t), observed_seconds=float(t[-1]) if len(t) else 0.,
        mean_nominal_ee_error_m=avg(error,np.ones(len(t),dtype=bool)),
        mean_compliant_ee_error_m=avg(np.linalg.norm(xyz-rows['reference_world'],axis=-1),np.ones(len(t),dtype=bool)),
        max_base_drift_m=float(np.max(drift)),
        support_force_z_n=avg(rows['robot_force'][:,2],baseline),
        upward_yield_m=None if z0 is None or zup is None else zup-z0,
        recovery_error_m=None if z0 is None or zr is None else abs(zr-z0),
        late_joint_gate=joint_gate, late_base_drift_m=avg(drift,late),
        guidance_base_progress_m=guidance_progress,
        joint_gate_increase=None if gate_before is None or joint_gate is None else joint_gate-gate_before,
        max_arm_relative_displacement_m=float(np.linalg.norm(rows['ee_task']-rows['ee_task'][0],axis=-1).max()),
        clipping_fraction=float(rows['clipped'].mean()))
    result['behavior_checks']={
        'nominal_tracking': result['mean_nominal_ee_error_m'] is not None and result['mean_nominal_ee_error_m']<config['nominal_ee_tolerance_m'],
        'downward_support': None if result['support_force_z_n'] is None else result['support_force_z_n'] < -1.,
        'upward_yield': None if result['upward_yield_m'] is None else result['upward_yield_m']>config['upward_yield_min_m'],
        'release_recovery': None if result['recovery_error_m'] is None else result['recovery_error_m']<config['recovery_tolerance_m'],
        'capability_and_base_motion': None if joint_gate is None else joint_gate>config['capability_gate_min'] and result['late_base_drift_m']>config['recruitment_displacement_min_m'],
        'hold_station': result['max_base_drift_m']<config['base_drift_tolerance_m'],
        'bounded_arm': result['max_arm_relative_displacement_m']<config['arm_motion_limit_m']}
    if scenario=='nominal':
        for key in ('downward_support','upward_yield','release_recovery','capability_and_base_motion'):
            result['behavior_checks'][key]=None
    return result
