"""Opt-in frozen-evaluation diagnostics. Never imported by production training."""
import json
from pathlib import Path
import numpy as np
import torch


def failure_evidence(path, iteration):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    event=EventAccumulator(str(path),size_guidance={'scalars':0});event.Reload()
    counts={}
    for tag in event.Tags()['scalars']:
        if tag.startswith(('EpisodeDiagnostics/terrain/','EpisodeDiagnostics/level/')) and tag.endswith('/failure_count'):
            rows=[x for x in event.Scalars(tag) if iteration-500<x.step<=iteration and np.isfinite(x.value)]
            counts[tag]=dict(count=sum(x.value for x in rows),report_intervals=len(rows))
    return dict(iteration_range=[iteration-499,iteration],counts=counts,
        limitation='Training logs contain separate family and level marginals, NOT joint family/level failure counts. Target highest-failure represented levels, balanced by family.')


def targeted_indices(terrain,saved,evidence,seed):
    cumulative=np.cumsum(saved['environment']['resolved']['terrain']['terrain_proportions'])
    columns=saved['environment']['resolved']['terrain']['num_cols']
    levels=terrain['terrain_levels'].cpu().numpy();types=terrain['terrain_types'].cpu().numpy()
    families=np.searchsorted(cumulative,types/columns+.001,side='right')
    counts=evidence['counts'];rng=np.random.default_rng(seed);ids=[];plan=[]
    for family,budget in ((2,3),(3,3),(4,4)):
        available=np.unique(levels[families==family])
        score=lambda level:counts.get(f'EpisodeDiagnostics/level/{level}/failure_count',{}).get('count',0.)
        ordered=sorted(available,key=lambda l:(-score(l),-l))[:3]
        if not ordered or not score(ordered[0]):raise ValueError('No recorded failure evidence for target levels')
        for j in range(budget):
            level=ordered[j%len(ordered)]
            index=int(rng.choice(np.flatnonzero((families==family)&(levels==level))))
            ids.append(index);plan.append(dict(family={2:'stairs_down',3:'stairs_up',4:'discrete_obstacles'}[family],
                level=int(level),column=int(types[index]),marginal_training_failures=score(level)))
    return torch.tensor(ids),plan


def pre(o):
    """Fresh command timestamp; cached affine maps replace diagnostic dynamics solves."""
    o.solve_data=o.result=None
    o.velocity=o.body_velocity().clone()
    o.record('velocity_before',o.velocity[:,[0,1,5]])
    o.record('command',o.command)
    o.record('orientation_xyzw',o.s.hard_pact_base_quat_xyzw())
    o.pre_original()
    d,r=o.solve_data,o.result
    if r is None:raise RuntimeError('Four-solve experiment missed a substep')
    ids=d['environment_ids']
    if len(ids)!=o.e.num_envs:raise RuntimeError('Expected all real environments at each physics substep')
    o.nominal=o.e._hard_pact_bounded_nominal_torque.clone()
    executed=o.s.hard_pact_executed_torque().clone()
    baseline=(o.e._qp_baseline_selection & (1<<o.k))!=0
    strength=torch.where(baseline,executed.new_tensor(1.),executed.new_tensor(o.e._qp_extra_strength))
    from rsl_rl.algorithms.hard_pact_qp_curriculum import execution_torque
    accepted=r.differentiated_mask|r.recovery_mask
    reconstructed=execution_torque(d['execution_base'],r.tau_safe,accepted,d['execution_alpha'],
        o.qp.execution_correction_limits(executed),o.qp.torque_limits.to(executed))
    # Device-side invariance evidence; checked once when saving, not sync per step.
    o.record('execution_reconstruction_error',(reconstructed-executed[ids]).abs().amax(-1))
    o.record('alpha_error',(d['execution_alpha'].flatten()-strength[ids]).abs())
    requested=d['execution_alpha']*(r.tau_safe-d['execution_base'])
    clipped=(requested.abs()>o.qp.execution_correction_limits(executed)+1e-5).any(-1)&accepted
    for name,value in [('baseline',baseline),('execution_strength',strength),('nominal_torque',o.nominal),
        ('execution_base',d['execution_base']),('optimized_torque',r.tau_safe),('executed_torque',executed),
        ('correction',executed-d['execution_base']),('requested_correction',requested),('clipped',clipped),
        ('stage',r.stage),('constraint_residual',r.diagnostics['selected/inequality_max']),
        ('torque_violation',(executed.abs()-o.s.torque_limits).clamp_min(0)),
        ('contact_probability',o.actor.cenet_torso_velo[:,3:7]),
        ('qp_stance_mask',d['contact_probability']>=o.qp.cfg.contact_threshold)]:o.record(name,value)
    o.costs(d,r,executed)
    o.record('height_estimate',o.actor.cenet_torso_velo[:,11])
    o.record('velocity_estimate_error',o.actor.cenet_torso_velo[:,:3]/o.e.obs_scales.lin_vel-o.velocity[:,:3])
    o.pred_force=o.e._yaw_local_to_world(o.e._qp_control_grf,o.e._current_base_quat_xyzw())
    o.record('pred_force',o.pred_force)
    label=o.s.hard_pact_wrench_at_base_origin_world(o.e._current_sustained_wrench_world)
    wrench=o.e._qp_control_wrench
    prediction=torch.cat([o.e._yaw_local_to_world(wrench[:,None,sl],o.e._current_base_quat_xyzw())[:,0]
        for sl in (slice(0,3),slice(3,6))],-1)
    o.record('wrench_error',prediction-label)


def trajectory_metrics(o,d,r,m,x0,x1,xe,a0,a1,ae):
    dt=d['dt'].reshape(-1,1);c=o.qp.cfg;accepted=r.differentiated_mask|r.recovery_mask
    for label,x,a in [('nominal',x0,a0),('candidate',x1,a1),('execution',xe,ae)]:
        qnext=d['joint_position']+dt*d['joint_velocity']+c.position_integration_coefficient*dt.square()*a[:,6:]
        vnext=d['joint_velocity']+dt*a[:,6:]
        qlim=o.s.dof_pos_limits_hard
        o.record(label+'_joint_position_violation',torch.where(accepted[:,None],
            (qlim[:,0]-qnext).clamp_min(0)+(qnext-qlim[:,1]).clamp_min(0),float('nan')))
        o.record(label+'_joint_velocity_violation',torch.where(accepted[:,None],
            (vnext.abs()-o.s.dof_vel_limits).clamp_min(0),float('nan')))
        residual=(m.physical_G@x[...,None]).squeeze(-1)-m.physical_h
        o.record(label+'_constraint_max',torch.where(accepted,residual.clamp_min(0).amax(-1),float('nan')))
        if c.endpoint_torque_constraints:
            o.record(label+'_endpoint_torque_violation',torch.where(accepted,
                residual[:,68:92].clamp_min(0).amax(-1),float('nan')))
    for name,a in [('candidate_delta',a1),('executed_model_delta',ae),('nominal_model_delta',a0)]:
        o.record(name,torch.where(accepted[:,None],dt*a[:,[0,1,5]],float('nan')))
    torque_change=(m.acceleration_map[:,:,:12]@(r.tau_safe-d['tau_nom'])[...,None]).squeeze(-1)
    o.record('torque_correction_delta',torch.where(accepted[:,None],dt*torque_change[:,[0,1,5]],float('nan')))
    o.record('recovery_joint_slack',r.recovery_slack)
    o.record('objective_candidate_joint_recovery',c.soft_joint_recovery_weight*(r.recovery_slack/c.soft_joint_recovery_scale_rad_s2).square().sum(-1))


def capture_ring(o,failures):
    if not hasattr(o,'failure_ring'):o.failure_ring=[]
    remaining=o.args.capture_cap-len(o.failure_ring)
    if remaining<=0:return
    ids=failures.nonzero().flatten()[:remaining].tolist()
    end=(o.t+1)*4;start=max(0,end-100)
    indices=torch.arange(start,end,device=o.e.device)%100
    for env in ids:
        o.failure_ring.append({name:value[indices,env].clone() for name,value in o.history_ring.items()})


def save(o,a,result):
    from scripts.eval_hard_pact_tradeoff import report,write_json
    reset=a['control_reset'].astype(bool);failure=reset&a['control_failure'].astype(bool)
    clean=~reset & ~a['control_reasons'][...,6].astype(bool);mask=np.repeat(clean,4,axis=0);dt=o.e.dt
    if np.max(a['execution_reconstruction_error'])>2e-4 or np.max(a['alpha_error'])>1e-6:
        raise RuntimeError('Execution override/reconstruction did not reach actual torque')
    assert (a['stage']>=0).all() and o.calls==len(reset)*4
    assert (a['baseline'].reshape(-1,4,o.e.num_envs).sum(1)==2).all()
    result['execution_checks']=dict(solves_per_env_control=4,baseline_per_env_control=2,
        max_reconstruction_error_nm=float(np.max(a['execution_reconstruction_error'])),
        max_alpha_error=float(np.max(a['alpha_error'])),
        baseline_alpha=np.unique(a['execution_strength'][a['baseline']]).tolist(),
        extra_alpha=np.unique(a['execution_strength'][~a['baseline']]).tolist())
    result['intervention_groups']={}
    for baseline,name in ((True,'baseline'),(False,'additional')):
        rows=a['baseline']==baseline
        accepted=rows&((a['stage']==0)|(a['stage']==1))
        result['intervention_groups'][name]=dict(rows=int(rows.sum()),accepted=int(accepted.sum()),
            primary=int((rows&(a['stage']==0)).sum()),recovery=int((rows&(a['stage']==1)).sum()),
            fallback=int((rows&(a['stage']==2)).sum()),clipped_fraction=float(a['clipped'][accepted].mean()) if accepted.any() else None,
            requested_nm=report.finite_stats(abs(a['requested_correction'][accepted])),
            applied_nm=report.finite_stats(abs(a['correction'][rows])))
    result['terrain_metrics']=[]
    for family in sorted({x['family'] for x in o.manifest['target_plan']}):
        envs=np.array([x['family']==family for x in o.manifest['target_plan']])
        for level in np.unique(a['control_terrain_level'][:,envs]):
            rows=envs[None,:]&(a['control_terrain_level']==level);safe=rows&clean
            physical=np.repeat(rows,4,axis=0);events=rows&failure
            result['terrain_metrics'].append(dict(family=family,level=int(level),env_s=float(rows.sum()*dt),
                failures=int(events.sum()),failure_per_env_s=float(events.sum()/(rows.sum()*dt)),
                tracking_rmse=[report.finite_stats(a['control_nonreset_error'][...,j][safe])['rms'] for j in range(3)],
                tracking_samples=int(safe.sum()),episode_lengths_s=report.finite_stats(a['control_episode_steps'][rows&reset]*dt),
                overlapping_causes={n:int((events&a['control_reasons'][...,i]).sum()) for i,n in enumerate(('contact','roll','pitch','height_low','height_high'))},
                violations={n:report.finite_stats(a[n][physical]) for n in ('joint_position_violation','joint_velocity_violation','torque_violation')}))
    for prefix in ('objective_execution_','execution_joint_','execution_endpoint_','execution_constraint_'):
        for name,value in a.items():
            if name.startswith(prefix):result['diagnostics'][name]=report.finite_stats(value[mask])
    for name in ('torque','grf','stance','attitude','height','planar','yaw','planar_error','yaw_error'):
        result['diagnostics']['objective_execution_change_'+name]=report.finite_stats((a['objective_execution_'+name]-a['objective_nominal_'+name])[mask])
    # Replace the older report's unmasked derivative summaries, not failure counts.
    for stage,label in ((0,'primary'),(1,'recovery')):
        valid=mask&(a['stage']==stage)
        for name in ('candidate_delta','executed_model_delta','nominal_model_delta'):
            err=a[name]-a['realized_delta']
            result['one_substep_model'][label][name]=[report.finite_stats(err[...,j][valid]) for j in range(3)]
    # Bounded 0.5-second histories, episode-safe, first 24 events, plus six
    # deterministic healthy controls. Compact full traces are bounded to 1000 steps.
    captures={};events=list(map(tuple,np.argwhere(failure)));chosen=[(t,e,'failure') for t,e in events[:o.args.capture_cap]]
    for e in range(min(6,o.e.num_envs)):
        eligible=[t for t in range(24,len(reset),25) if not reset[t-24:t+1,e].any()]
        if eligible:chosen.append((eligible[len(eligible)//2],e,'healthy'))
    identities=[]
    for j,(end,env,kind) in enumerate(chosen):
        previous=np.flatnonzero(reset[:end,env]);start=max(0,end-24,int(previous[-1])+1 if len(previous) else 0)
        identities.append(dict(id=j,kind=kind,env=int(env),start=int(start),end=int(end)))
        for name,value in a.items():
            if name.startswith('control_'):segment=value[start:end+1,env]
            elif kind=='failure':segment=o.failure_ring[j][name][-4*(end-start+1):].cpu().numpy()
            else:segment=value[start*4:(end+1)*4,env]
            captures[f'{j:02d}_{name}']=segment
    np.savez_compressed(o.out/'bounded_histories.npz',**captures)
    write_json(o.out/'history_index.json',identities)
    result['failure_captures']=min(len(events),o.args.capture_cap)
    result['failure_capture_drops']=max(0,len(events)-o.args.capture_cap)
    result['diagnostic_assumptions']=['Execution predictions retain the optimized candidate GRF; physical GRF may differ.',
        'Canonical qdd predicts the derivative of body-frame velocity; each measured endpoint uses its live body frame.',
        'All reset/teleport control intervals excluded from model differences; failures still counted.',
        'Failure causes overlap; terrain family/level targeting uses marginal training evidence.',
        'Nominal reference may be infeasible. Full candidate acceptance does not certify clipped/blended execution.']
    write_json(o.out/'result.json',result)
