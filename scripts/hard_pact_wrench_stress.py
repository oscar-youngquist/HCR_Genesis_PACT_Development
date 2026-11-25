"""Evaluation-only absolute-time tapes. No training imports or configuration edits."""
import numpy as np
import torch


def heavier_waveform(w,target,caps):
    """Scale each event's amplitude, preserving its temporal envelope and XY direction."""
    maxima=torch.stack((target[:,:2].norm(dim=-1),target[:,2].abs(),target[:,3:5].abs().amax(-1),target[:,5].abs()),-1)
    scale=torch.minimum(torch.full_like(maxima,2.),caps/maxima.clamp_min(1e-12))
    h=w.clone();h[:,:2]*=scale[:,0,None];h[:,2]*=scale[:,1];h[:,3:5]*=scale[:,2,None];h[:,5]*=scale[:,3]
    return h


class Tapes:
    def __init__(self,o):
        self.o=o;e=o.e;s=o.s;n=e.num_envs;steps=o.args.steps
        from scripts.eval_hard_pact_frozen import tensor_hash
        fields={k:v.clone() for k,v in vars(e).items() if k.startswith('_persistent_') and torch.is_tensor(v)}
        fields.update({k:getattr(e,k).clone() for k in ('_current_sustained_wrench_world','_current_sustained_active_mask')})
        commands=e.commands.clone();self.commands=[];self.light=[];self.heavy=[];self.active=[]
        ids=torch.arange(n,device=e.device)
        final=e.domain_rand_curriculum.schema
        caps=commands.new_tensor([final['persistent_force'].range_at(1.)[1],
            -final['persistent_vertical_force'].range_at(1.)[0],final['persistent_torque'].range_at(1.)[1],
            final['persistent_yaw_torque'].range_at(1.)[1]])
        with torch.random.fork_rng(devices=[0]):
            torch.manual_seed(o.args.seed*1000003+77123)
            e._reset_persistent_wrench_state(ids)
            start=e.common_step_counter
            for t in range(steps+1):
                if t%max(1,round(e.cfg.commands.resampling_time/e.dt))==0:e._resample_commands(ids)
                self.commands.append(e.commands.clone())
                e._update_persistent_wrench(start+t)
                w=e._current_sustained_wrench_world.clone();target=e._persistent_wrench_target_world
                # Bound event amplitudes, not instantaneous waveform values: preserve ramps.
                h=heavier_waveform(w,target,caps)
                self.light.append(w);self.heavy.append(h);self.active.append(e._current_sustained_active_mask.clone())
        for k,v in fields.items():getattr(e,k).copy_(v)
        e.commands.copy_(commands)
        self.commands=torch.stack(self.commands);self.light=torch.stack(self.light);self.heavy=torch.stack(self.heavy);self.active=torch.stack(self.active)
        self.wrench=self.heavy if o.args.load=='heavier' else self.light
        self.mechanics=tensor_hash(e._canonical_randomized_parameters())
        digest=lambda v:tensor_hash({'value':v})
        o.manifest['tapes']=dict(commands=digest(self.commands),light_wrench=digest(self.light),
            heavy_wrench=digest(self.heavy),applied=digest(self.wrench),mechanics=self.mechanics,
            load=o.args.load,final_caps=caps.tolist(),
            semantics='Absolute control-time native command/heading targets and native persistent waveform; event-amplitude caps. Heading feedback remains trajectory-dependent. Noise/push RNG keyed by absolute step; action delay fixed at initial realized sample.')
        np.savez_compressed(o.out/'tapes.npz',commands=self.commands.cpu().numpy(),light=self.light.cpu().numpy(),heavy=self.heavy.cpu().numpy())
        e._resample_commands=lambda ids:None
        e._reset_persistent_wrench_state=lambda ids:None
        e._update_persistent_wrench=self.wrench_update
        s._reset_domain_randomization=lambda ids:None
        s._reset_sample_ids=lambda name,ids:ids[:0]
        delay=e.action_delay.clone() if hasattr(e,'action_delay') else None
        reset=e.reset_idx
        def reset_fixed(ids):
            timers={k:getattr(s,k).clone() for k in ('push_timeouts','vert_timeouts','wrench_timeouts')}
            with torch.random.fork_rng(devices=[0]):
                torch.manual_seed(o.args.seed*1000003+81123+max(o.t,0));reset(ids)
            for k,v in timers.items():getattr(s,k).copy_(v)
            if delay is not None:e.action_delay.copy_(delay)
            self.set_commands(min(o.t+1,steps));self.wrench_update(None)
        e.reset_idx=reset_fixed
        push=s.push_robots
        def push_fixed():
            with torch.random.fork_rng(devices=[0]):
                torch.manual_seed(o.args.seed*1000003+82123+max(o.t,0));push()
        s.push_robots=push_fixed
        compute=e.compute_observations
        def observations():
            self.set_commands(min(max(o.t+1,0),steps))
            with torch.random.fork_rng(devices=[0]):
                torch.manual_seed(o.args.seed*1000003+83123+max(o.t,0));compute()
        e.compute_observations=observations
        self.set_commands(0);e.compute_observations()

    def set_commands(self,t):
        e=self.o.e;e.commands.copy_(self.commands[t])
        if e.cfg.commands.heading_command:
            from legged_gym.utils.math_utils import quat_apply,wrap_to_pi
            forward=quat_apply(e.simulator.base_quat,e.forward_vec)
            heading=torch.atan2(forward[:,1],forward[:,0])
            e.commands[:,2]=(0.5*wrap_to_pi(e.commands[:,3]-heading)).clamp(*e.command_ranges['ang_vel_yaw'])

    def wrench_update(self,_):
        e=self.o.e;t=max(self.o.t,0)
        e._current_sustained_wrench_world.copy_(self.wrench[t]);e._current_sustained_active_mask.copy_(self.active[t])

    def before(self,t):
        self.set_commands(t)


def install(o):
    expected={'torque_tracking_weight':20.,'force_tracking_weight':1.,'contact_acceleration_weight':.1,
        'height_weight':.5,'planar_velocity_weight':20.,'yaw_rate_weight':10.}
    if o.args.worker in ('B_planar10','B_yaw5','B_both_lower'):
        from scripts.eval_hard_pact_tradeoff import INTERVENTION_VARIANTS
        overrides,strength=INTERVENTION_VARIANTS[o.args.worker]
        assert set(overrides)<= {'planar_velocity_weight','yaw_rate_weight'} and strength==.5
        expected.update(overrides,attitude_weight=.1)
    if o.args.worker=='C_torque30':expected['torque_tracking_weight']=30.
    for k,v in expected.items():
        if getattr(o.qp.cfg,k)!=v:raise ValueError(f'Unexpected saved objective {k}: {getattr(o.qp.cfg,k)} != {v}')
    o.tapes=Tapes(o)
    if o.args.worker=='off':
        # Retain actor/estimator inference and original controller, skip intervention only.
        o.e._solve_hard_pact_rollout_qp_substep=lambda *a:o.e._apply_non_qp_torque_rate_clip()
    o.manifest['notes']=['Reset-independent tapes replace native reset-triggered exogenous sampling; realized mechanics frozen.',
        'Identical initial states/terrain within seed. State, heading-feedback yaw commands, and delay-queue contents diverge after resets.',
        'Same absolute-step noise and push RNG. Reset sampling is isolated but reset subsets differ.',
        'Wrench-load stress test, not moving-liquid validation. Learned wrench predictions retained.']
    if o.args.worker=='D_true_wrench':
        o.manifest['notes'].append('Oracle: current applied external wrench about base-link origin sampled at first substep, held yaw-local over control interval. Normal live-yaw conversion; no inertial/gravity mismatch added. Learned decoder outputs unchanged and separately logged. Diagnostic only.')


def oracle_input(o,data,to_local=None):
    """Only QP input changes; first-substep pose precedes any interval integration."""
    if to_local is None:
        from legged_gym.envs.go2.go2_hard_pact.transition import wrench_world_to_yaw_local
        to_local=wrench_world_to_yaw_local
    q=o.e._current_base_quat_xyzw()
    if o.k==0:
        label=o.s.hard_pact_wrench_at_base_origin_world(o.e._current_sustained_wrench_world)
        o.oracle_yaw=to_local(label,q).detach().clone()
        o.oracle_world_at_sample=label.detach().clone()
    held=o.oracle_yaw
    world=torch.cat([o.e._yaw_local_to_world(held[:,None,sl],q)[:,0] for sl in (slice(0,3),slice(3,6))],-1)
    result=dict(data,wrench_pred_world=world[data['environment_ids']])
    o.record('qp_wrench_input_world',world)
    o.record('oracle_world_at_sample',o.oracle_world_at_sample)
    o.record('oracle_initial_frame_error',(world-o.oracle_world_at_sample).abs().amax(-1) if o.k==0 else torch.zeros_like(world[:,0]))
    return result


def save_metrics(o,a,result):
    from scripts.eval_hard_pact_tradeoff import report,write_json
    from scripts.eval_hard_pact_frozen import tensor_hash
    if tensor_hash(o.e._canonical_randomized_parameters())!=o.tapes.mechanics:raise RuntimeError('Realized mechanics changed')
    clean=~a['control_reset'].astype(bool)&~a['control_reasons'][...,6].astype(bool);mask=np.repeat(clean,4,axis=0)
    failures=a['control_reset'].astype(bool)&a['control_failure'].astype(bool)
    result['failure_free_fraction']=float((~failures.any(0)).mean())
    result['observation_duration_s']=o.args.steps*o.e.dt
    result['stability']={
        'roll_rad':report.finite_stats(abs(a['control_tilt'][...,0])),
        'pitch_rad':report.finite_stats(abs(a['control_tilt'][...,1])),
        'height_error_m':report.finite_stats(a['control_base_height']-o.e.cfg.rewards.base_height_target),
        'vertical_velocity_m_s':report.finite_stats(a['body_velocity_after'][...,2][mask]),
        'angular_velocity_rad_s':report.finite_stats(a['body_velocity_after'][...,3:][mask])}
    result['command_distribution']=[report.finite_stats(a['control_command'][...,i]) for i in range(3)]
    result['applied_wrench_axes']=[report.finite_stats(a['applied_wrench'][...,i]) for i in range(6)]
    result['tapes']=o.manifest['tapes'];result['mechanics_unchanged']=True
    if o.args.worker=='off':
        assert o.calls==0
        result['execution_checks']={'solves_per_env_control':0}
        # Reuse bounded event/control captures without requiring fictitious QP data.
        events=list(map(tuple,np.argwhere(failures)));chosen=[(t,e,'failure') for t,e in events[:o.args.capture_cap]]
        for e in range(min(6,o.e.num_envs)):
            valid=[t for t in range(24,len(clean),25) if clean[t-24:t+1,e].all()]
            if valid:chosen.append((valid[len(valid)//2],e,'healthy'))
        captures={};identities=[]
        for j,(end,e,kind) in enumerate(chosen):
            resets=np.flatnonzero(a['control_reset'][:end,e]);start=max(0,end-24,int(resets[-1])+1 if len(resets) else 0)
            identities.append(dict(id=j,kind=kind,env=int(e),start=int(start),end=int(end)))
            for k,v in a.items():captures[f'{j:02d}_{k}']=v[start:end+1,e] if k.startswith('control_') else v[start*4:(end+1)*4,e]
        np.savez_compressed(o.out/'bounded_histories.npz',**captures);write_json(o.out/'history_index.json',identities)
        result['failure_captures']=min(len(events),o.args.capture_cap);result['failure_capture_drops']=max(0,len(events)-o.args.capture_cap)
    write_json(o.out/'result.json',result)
