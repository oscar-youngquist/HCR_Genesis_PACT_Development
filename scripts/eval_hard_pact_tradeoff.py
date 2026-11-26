#!/usr/bin/env python3
"""Opt-in frozen objective sweep. No runner, optimizer, backward or curriculum updates."""
import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from dataclasses import asdict, replace

ROOT = Path(__file__).resolve().parents[1]
EVENT = 'events.out.tfevents.1795231251.CICSUNIX-US-003.2659464.0'
VARIANTS = {'saved': None, 'stance0_torque20': (0.,20.),
            'stance010_torque20': (.1,20.), 'stance010_torque30': (.1,30.)}
ATTITUDE_VARIANTS = {'attitude050_normal':(.5,False),'attitude010_normal':(.1,False),
                     'attitude050_probe':(.5,True),'attitude010_probe':(.1,True)}
INTERVENTION_VARIANTS = {'A_reference':({},1.), 'B_extra050':({},.5),
    'C_attitude005':({'attitude_weight':.05},1.), 'D_planar10':({'planar_velocity_weight':10.},1.),
    'B_extra075':({},.75), 'E_extra075':({},.75),
    'F_extra075_attitude005':({'attitude_weight':.05},.75), 'off':({},1.),
    'B_planar10':({'planar_velocity_weight':10.},.5),
    'B_yaw5':({'yaw_rate_weight':5.},.5),
    'B_both_lower':({'planar_velocity_weight':10.,'yaw_rate_weight':5.},.5),
    'C_torque30':({'torque_tracking_weight':30.},1.), 'D_true_wrench':({},1.),
    'E_rebalanced':({'torque_tracking_weight':15.,'force_tracking_weight':.5,
        'contact_acceleration_weight':.05,'attitude_weight':.05,'height_weight':.25,
        'planar_velocity_weight':5.,'yaw_rate_weight':5.},1.),
    'F_no_torso':({'torque_tracking_weight':15.,'force_tracking_weight':.5,
        'contact_acceleration_weight':.05,'attitude_weight':0.,'height_weight':0.,
        'planar_velocity_weight':5.,'yaw_rate_weight':5.},1.),
    'H_weak_torso':({'torque_tracking_weight':15.,'force_tracking_weight':.5,
        'contact_acceleration_weight':.05,'attitude_weight':.01,'height_weight':.01,
        'planar_velocity_weight':5.,'yaw_rate_weight':5.},1.),
    'G_equal_stability':({'torque_tracking_weight':15.,'force_tracking_weight':.5,
        'contact_acceleration_weight':.1,'attitude_weight':.1,'height_weight':.1,
        'planar_velocity_weight':5.,'yaw_rate_weight':5.},1.)}
ALL_VARIANTS = {**VARIANTS,**ATTITUDE_VARIANTS,**INTERVENTION_VARIANTS}
sys.path.insert(0, str(ROOT))
from scripts.eval_hard_pact_frozen import apply_config, sha_file, tensor_hash, dependency_versions


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


matched = module('matched_eval_reference', ROOT/'diagnostics/matched5500/evaluate.py')
report = module('matched_report_reference', ROOT/'diagnostics/matched5500/report.py')
write_json = matched.write_json


def freeze_settings(training, checkpoint, variant):
    from rsl_rl.algorithms.hard_pact_qp import HardPACTQPConfig
    from rsl_rl.algorithms.hard_pact_qp_curriculum import QPCurriculum
    from rsl_rl.algorithms.hard_pact_intervention_curriculum import intervention_probabilities
    cfg = HardPACTQPConfig.from_dict(training['algorithm']['hard_pact_qp'])
    state = checkpoint['training_state']
    schedule = QPCurriculum(cfg)
    schedule.load_state_dict(state['qp_curriculum'])
    cfg, alpha = schedule.begin(state['next_iteration'])
    intervention = state['qp_curriculum']['intervention']
    progress = intervention['snapshot_progress']
    probabilities = intervention_probabilities(progress)
    strength = cfg.intervention_extra_strength_start+(1-cfg.intervention_extra_strength_start)*progress
    if variant in INTERVENTION_VARIANTS:
        if cfg.qp_update_mode!='curriculum_substeps' or not cfg.intervention_extra_strength_enabled:
            raise ValueError('Intervention sweep requires saved curriculum_substeps with baseline masks')
        if cfg.contact_acceleration_weight!=.1:raise ValueError('Expected saved stance=0.1; never use attitude-probe preset')
        overrides,strength=INTERVENTION_VARIANTS[variant]
        before=asdict(cfg);cfg=replace(cfg,**overrides)
        assert {k for k,v in asdict(cfg).items() if v!=before[k]} <= set(overrides)
        alpha,probabilities=1.,(0.,0.,1.)
    elif variant in ATTITUDE_VARIANTS:
        cfg = replace(cfg,contact_acceleration_weight=.25,torque_tracking_weight=20.,
                      attitude_weight=ATTITUDE_VARIANTS[variant][0])
    elif VARIANTS[variant] is not None:
        stance, torque = VARIANTS[variant]
        cfg = replace(cfg, contact_acceleration_weight=stance, torque_tracking_weight=torque)
    return cfg, alpha, probabilities, strength


class Observer(matched.Observer):
    """Reuse matched evaluator frame/trace helpers, not its play/tape overrides."""
    def __init__(self, env, actor, qp, args, manifest):
        import torch
        self.e, self.s, self.actor, self.qp = env, env.simulator, actor, qp
        self.args, self.manifest = args, manifest
        self.t, self.k, self.calls = -1, 0, 0
        self.arrays, self.controls = {}, {}
        self.out = args.output/f'{args.worker}_seed{args.seed}'
        self.out.mkdir(parents=True, exist_ok=True)
        self.solve_data = self.result = None
        self.before_hash = tensor_hash(actor.state_dict())
        self.probe=None
        self.video=None
        if getattr(args,'recovery_diagnostic',False):
            from scripts.hard_pact_recovery_probe import RecoveryProbe
            self.probe=RecoveryProbe(env.num_envs,env.device,env.dt,True,
                args.probe_duration,args.probe_confirmation,max_duration=args.probe_duration,cap=args.probe_cap)
        if getattr(args,'attitude_probe',False):
            from scripts.hard_pact_recovery_probe import RecoveryProbe
            self.probe=RecoveryProbe(env.num_envs,env.device,env.dt,ATTITUDE_VARIANTS[args.worker][1],
                                     args.probe_extension,args.probe_confirmation)
        self.pre_original = self.s._hard_pact_pre_physics_substep
        self.post_original = self.s._hard_pact_grf_post_physics_substep
        self.s._hard_pact_pre_physics_substep = self.pre
        self.s._hard_pact_grf_post_physics_substep = self.post
        original = qp.solve
        def solve(**data):
            if args.worker=='D_true_wrench':
                from scripts.hard_pact_wrench_stress import oracle_input
                data=oracle_input(self,data)
            result = original(**data)
            self.solve_data, self.result = data, result
            self.calls += 1
            if not getattr(args,'intervention_sweep',False):self.costs(data, result)
            return result
        qp.solve = solve
        build = qp._build
        def assembly(data):
            self.build = build(data)
            self.assembly_weights = (qp.cfg.contact_acceleration_weight, qp.cfg.torque_tracking_weight)
            if getattr(args,'intervention_sweep',False):
                self.assembly_weights+=(qp.cfg.attitude_weight,qp.cfg.planar_velocity_weight,qp.cfg.yaw_rate_weight)
            if self.probe is not None:self.assembly_weights+= (qp.cfg.attitude_weight,)
            if getattr(args,'preservation_visual',False):
                self.record('assembled_seven_weights',data['tau_nom'].new_tensor([
                    getattr(qp.cfg,k) for k in ('torque_tracking_weight','force_tracking_weight',
                    'contact_acceleration_weight','attitude_weight','height_weight','planar_velocity_weight','yaw_rate_weight')
                ]).expand(env.num_envs,-1))
            return self.build
        qp._build = assembly
        termination = env.check_termination
        def terminated():
            termination()
            if self.t < 0: return
            s,e = self.s,self.e
            angle = (s._base_euler[:,:2]+torch.pi)%(2*torch.pi)-torch.pi
            height = (s.base_pos[:,2:3]-s.measured_heights).mean(-1)
            reasons = torch.stack(((s.link_contact_forces[:,s.termination_contact_indices].norm(dim=-1)>10).any(-1),
                angle[:,0].abs()>e.cfg.termination.roll_threshold, angle[:,1].abs()>e.cfg.termination.pitch_threshold,
                height<e.cfg.termination.height_min, height>e.cfg.termination.height_max,
                e.time_out_buf.bool(), e.non_failure_reset_buf.bool()),-1)
            for name,value in [('reasons',reasons),('reset',e.reset_buf.bool()),
                ('failure',e.fail_buf>e.cfg.env.fail_to_terminal_time_s/e.dt),
                ('episode_steps',e.episode_length_buf),('terrain_level',s.terrain_levels),
                ('terrain_type',s.terrain_types),('base_height',height),('tilt',angle),
                ('nonreset_error',self.body_velocity()[:,[0,1,5]]-self.command)]:
                self.record(name,value,control=True)
            if getattr(args,'intervention_sweep',False):
                from scripts.hard_pact_intervention_eval import capture_ring
                capture_ring(self,e.reset_buf.bool() & (e.fail_buf>e.cfg.env.fail_to_terminal_time_s/e.dt))
            if self.probe is not None:
                self.record('original_reset',e.reset_buf.bool(),control=True)
                self.record('original_failure',e.fail_buf>e.cfg.env.fail_to_terminal_time_s/e.dt,control=True)
                enabled=e.cfg.termination.termination_terms
                geometry=torch.zeros(e.num_envs,device=e.device,dtype=torch.bool)
                for index,name in ((1,'roll'),(2,'pitch'),(3,'height_min'),(4,'height_max')):
                    if name in enabled:geometry |= reasons[:,index]
                q,v=e._canonical_joint_state()
                hard=reasons[:,0] | ~torch.isfinite(torch.cat((q,v,self.body_velocity()),-1)).all(-1)
                healthy=None
                if getattr(args,'recovery_diagnostic',False):
                    # Only roll/low-height are deferred. Other original safety bounds stay hard.
                    geometry=reasons[:,1]|reasons[:,3]
                    hard |= reasons[:,2]|reasons[:,4]
                    healthy=(angle[:,0].abs()<=e.cfg.termination.roll_threshold-args.probe_hysteresis_rad)&(
                        height>=e.cfg.termination.height_min+args.probe_hysteresis_m)&~hard
                decision=self.probe.update(e.reset_buf.bool(),e.fail_buf>e.cfg.env.fail_to_terminal_time_s/e.dt,
                    geometry,hard,reasons[:,5]|reasons[:,6],self.t,healthy=healthy)
                for name,value in decision.items():self.record(name,value,control=True)
                e.reset_buf.copy_(decision['actual_reset'])
                self.record('reset',e.reset_buf.bool(),control=True)
            if getattr(args,'preservation_visual',False):
                self.record('base_position_world',s._robot.data.root_link_pos_w,control=True)
                self.record('terrain_height_reference',s.measured_heights.mean(-1),control=True)
                if self.video is not None:self.video.capture()
        env.check_termination = terminated
        # QP choices cannot consume command/randomization RNG. No extra draws
        # inside the production sampler; only isolate its existing RNG stream.
        import legged_gym.envs.go2.go2_hard_pact.go2_hard_pact as implementation
        selection = implementation.qp_interval_selection
        def isolated(*a, **kw):
            with torch.random.fork_rng(devices=[0]):
                torch.manual_seed(args.seed*1000003+40001+self.t)
                return selection(*a, **kw)
        implementation.qp_interval_selection = isolated

    def record(self,name,value,control=False):
        import torch
        if self.t < 0:return
        store=self.controls if control else self.arrays
        if name not in store:
            shape=(self.args.steps if control else self.args.steps*4,)+tuple(value.shape)
            store[name]=torch.full(shape,float('nan') if value.is_floating_point() else 0,
                                  dtype=value.dtype,device=value.device)
        store[name][self.t if control else self.t*4+self.k].copy_(value.detach())
        if not control and getattr(self.args,'intervention_sweep',False):
            if not hasattr(self,'history_ring'):self.history_ring={}
            if name not in self.history_ring:self.history_ring[name]=torch.empty((100,)+tuple(value.shape),dtype=value.dtype,device=value.device)
            self.history_ring[name][(self.t*4+self.k)%100].copy_(value.detach())

    def costs(self, d, r, executed=None):
        """Detached residuals with production scales; no new solver calls."""
        import torch
        m,c = self.build,self.qp.cfg
        ids=d['environment_ids']; accepted=r.differentiated_mask|r.recovery_mask
        stance=d['contact_probability']>=c.contact_threshold
        force=torch.where(stance[...,None],d['force_pred_world'],0.).flatten(1)
        x0=torch.cat((d['tau_nom'],force),-1)
        x1=torch.cat((r.tau_safe,r.force_world.flatten(1)),-1)
        A,b=m.acceleration_map[:,:,:24],m.acceleration_offset
        a0=(A@x0[...,None]).squeeze(-1)+b
        a1=(A@x1[...,None]).squeeze(-1)+b
        q=d['base_quaternion']; yaw=torch.atan2(2*(q[:,3]*q[:,2]+q[:,0]*q[:,1]),1-2*(q[:,1]**2+q[:,2]**2))
        tilt=torch.stack((-2*(q[:,1]*q[:,2]-q[:,3]*q[:,0]),2*(q[:,0]*q[:,2]+q[:,3]*q[:,1]),torch.zeros_like(yaw)),-1)
        C,e,_=self.qp._velocity_tracking_affine(d,A,b,horizon_s=c.qp_velocity_objective_horizon_s)
        height_data=d
        if 'estimated_height' not in d:
            # Zero-weight production assembly intentionally omits height inputs.
            # Restore held estimates only for detached diagnostics, not the solve.
            from rsl_rl.modules.hard_pact_physics import estimated_qp_height_inputs
            height_data=dict(d,**estimated_qp_height_inputs(
                self.e._hard_pact_policy_explicit[ids],d['base_quaternion'],c.height_velocity_obs_scale))
        H,h,desired=self.qp._height_affine(height_data,A,b)
        def values(x,a):
            contact=torch.einsum('bfkn,bn->bfk',d['foot_jacobians'],a)+d['foot_acceleration_bias']
            angular=(d['base_jacobian'][:,3:6]@a[...,None]).squeeze(-1)+c.attitude_kp*tilt+c.attitude_kd*d['base_angular_velocity_world']
            rp=torch.stack((yaw.cos()*angular[:,0]+yaw.sin()*angular[:,1],-yaw.sin()*angular[:,0]+yaw.cos()*angular[:,1]),-1)
            vel=(C@x[...,None]).squeeze(-1)+e
            height=(H@x[...,None]).flatten()+h.flatten()
            return dict(torque=c.torque_tracking_weight*((x[:,:12]-d['tau_nom'])/c.torque_scale_nm).square().sum(-1),
                grf=c.force_tracking_weight*((x[:,12:]-force)/c.force_scale_n).square().sum(-1),
                stance=c.contact_acceleration_weight*(contact/c.contact_acceleration_scale_m_s2).square().sum(-1).mul(stance).sum(-1),
                attitude=c.attitude_weight*(rp/c.attitude_acceleration_scale_rad_s2).square().sum(-1),
                height=c.height_weight*(height/c.height_acceleration_scale).square(),
                planar=c.planar_velocity_weight*(vel[:,:2]/c.planar_velocity_scale_m_s).square().sum(-1),
                yaw=c.yaw_rate_weight*(vel[:,2]/c.yaw_rate_scale_rad_s).square(),
                torque_residual=(x[:,:12]-d['tau_nom']).norm(dim=-1),
                grf_residual=(x[:,12:]-force).norm(dim=-1),
                stance_residual=(contact.square().sum(-1)*stance).sum(-1).sqrt(),
                attitude_residual=rp.norm(dim=-1),height_residual=height,
                planar_error=vel[:,:2].norm(dim=-1),yaw_error=vel[:,2].abs())
        samples=[('nominal',values(x0,a0)),('candidate',values(x1,a1))]
        if executed is not None:
            xe=torch.cat((executed[ids],r.force_world.flatten(1)),-1)
            ae=(A@xe[...,None]).squeeze(-1)+b
            samples.append(('execution',values(xe,ae)))
            from scripts.hard_pact_intervention_eval import trajectory_metrics
            trajectory_metrics(self,d,r,m,x0,x1,xe,a0,a1,ae)
        for label,values_ in samples:
            for name,v in values_.items():
                out=v.new_full((self.e.num_envs,),float('nan'))
                out[ids]=torch.where(accepted,v,float('nan'))
                self.record('objective_'+label+'_'+name,out)
        self.record('assembly_weights',x0.new_tensor(self.assembly_weights).expand(self.e.num_envs,-1))
        if self.probe is not None:
            def compact(name,value):
                out=value.new_full((self.e.num_envs,)+tuple(value.shape[1:]),float('nan'))
                keep=accepted.reshape((-1,)+(1,)*(value.ndim-1))
                out[ids]=torch.where(keep,value,float('nan'));self.record(name,out)
            desired_world=-c.attitude_kp*tilt-c.attitude_kd*d['base_angular_velocity_world']
            compact('desired_angular_acceleration_world',desired_world)
            for label,a in [('nominal',a0),('optimized',a1)]:
                contact=torch.einsum('bfkn,bn->bfk',d['foot_jacobians'],a)+d['foot_acceleration_bias']
                compact(label+'_stance_acceleration_world',torch.where(stance[...,None],contact,float('nan')))
                compact(label+'_angular_acceleration_world',(d['base_jacobian'][:,3:6]@a[...,None]).squeeze(-1))
            compact('optimized_torque',r.tau_safe)

    def pre(self):
        self._pre()
        if getattr(self.args,'audit_wrench_application',False):
            import torch
            from scripts.hard_pact_shadow_estimation import rotate
            q=self.s.hard_pact_base_quat_xyzw()
            c=self.s._robot.permanent_wrench_composer
            requested=self.e._current_sustained_wrench_world
            expected=self.s.hard_pact_wrench_at_base_origin_world(requested)
            delivered=torch.cat((rotate(q,c.composed_force_as_torch[:,self.s._base_link_index]),
                                 rotate(q,c.composed_torque_as_torch[:,self.s._base_link_index])),-1)
            self.record('requested_wrench_world',requested)
            self.record('expected_wrench_base_origin_world',expected)
            self.record('delivered_wrench_base_origin_world',delivered)

    def _pre(self):
        import torch
        if getattr(self.args,'wrench_stress',False) and self.args.worker=='off':
            super().pre()
            self.record('baseline',torch.zeros(self.e.num_envs,device=self.e.device,dtype=torch.bool))
            self.pred_force=self.e._yaw_local_to_world(self.e._qp_control_grf,self.e._current_base_quat_xyzw())
            return
        if getattr(self.args,'intervention_sweep',False):
            from scripts.hard_pact_intervention_eval import pre
            pre(self)
            if getattr(self.args,'shadow_estimation',False):
                from scripts.hard_pact_shadow_estimation import capture_pre
                capture_pre(self)
            return
        super().pre()
        baseline=((self.e._qp_baseline_selection & (1<<self.k))!=0)
        self.record('baseline',baseline)
        if self.probe is not None:
            self.record('qp_stance_mask',self.actor.cenet_torso_velo[:,3:7]>=self.qp.cfg.contact_threshold)
            strength=torch.where(baseline,self.nominal.new_tensor(self.e._qp_execution_alpha),
                self.nominal.new_tensor(self.e._qp_execution_alpha*self.e._qp_extra_strength))
            self.record('execution_strength',strength)
        self.record('height_estimate',self.actor.cenet_torso_velo[:,11])
        self.record('velocity_estimate_error',self.actor.cenet_torso_velo[:,:3]/self.e.obs_scales.lin_vel-self.velocity[:,:3])
        self.pred_force=self.e._yaw_local_to_world(self.e._qp_control_grf,self.e._current_base_quat_xyzw())
        # The control-rate prediction is transformed once into the live world frame.
        self.record('pred_force',self.pred_force)
        label=self.s.hard_pact_wrench_at_base_origin_world(self.e._current_sustained_wrench_world)
        prediction=self.e._yaw_local_to_world(self.e._qp_control_wrench[:,None,:3],self.e._current_base_quat_xyzw())[:,0]
        moment=self.e._yaw_local_to_world(self.e._qp_control_wrench[:,None,3:],self.e._current_base_quat_xyzw())[:,0]
        self.record('wrench_error',torch.cat((prediction,moment),-1)-label)

    def post(self):
        import torch
        super().post()
        self.k-=1  # inherited post advances after recording; keep matching slot
        if getattr(self.args,'shadow_estimation',False):
            from scripts.hard_pact_shadow_estimation import capture_post
            capture_post(self)
        if getattr(self.args,'wrench_stress',False):
            self.record('body_velocity_after',self.body_velocity())
            self.record('applied_wrench',self.s.hard_pact_wrench_at_base_origin_world(self.e._current_sustained_wrench_world))
            q=self.e._current_base_quat_xyzw();w=self.e._qp_control_wrench
            self.record('predicted_wrench',torch.cat([self.e._yaw_local_to_world(w[:,None,sl],q)[:,0] for sl in (slice(0,3),slice(3,6))],-1))
        self.record('grf_error',self.pred_force-self.e.grf_processor.raw)
        if self.probe is not None:
            self.record('measured_force_world',self.e.grf_processor.raw)
            self.record('velocity_after',self.body_velocity())
            self.record('orientation_after_xyzw',self.s.hard_pact_base_quat_xyzw())
            feet=self.s._robot.data.body_link_pos_w[:,self.s._feet_indices,:]
            self.record('foot_clearance',feet[:,:,2]-self.s.height_around_feet.mean(-1)-self.e.cfg.rewards.foot_height_offset)
            self.record('height_after',self.s._robot.data.root_link_pos_w[:,2]-self.s.measured_heights.mean(-1))
        height=(self.s.base_pos[:,2:3]-self.s.measured_heights).mean(-1)
        self.record('height_estimate_error',self.actor.cenet_torso_velo[:,11]-height)
        if hasattr(self.s,'feet_vel'):
            self.record('stance_slip',self.s.feet_vel[:,:,:2].norm(dim=-1)*self.e.grf_processor.contacts)
        self.k+=1

    def save(self, count, complete=False):
        import numpy as np
        import torch
        arrays={k:v[:count*4].cpu().numpy() for k,v in self.arrays.items()}
        arrays.update({'control_'+k:v[:count].cpu().numpy() for k,v in self.controls.items()})
        np.savez_compressed(self.out/'trace.npz',**arrays)
        progress=dict(completed_steps=count,complete=complete,condition=self.args.worker,seed=self.args.seed,
            qp_batch_dispatches=self.calls,wall_s=time.monotonic()-self.start,
            weights_unchanged=tensor_hash(self.actor.state_dict())==self.before_hash,
            torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),torch_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            steady_wall_s=time.monotonic()-getattr(self,'warm_start',self.start),
            steady_control_steps=max(count-100,0),physical_gpu=self.args.physical_gpu)
        write_json(self.out/'progress.json',progress)
        if getattr(self.args,'shadow_estimation',False):
            pairs=getattr(self,'shadow_timing_events',[])
            write_json(self.out/'shadow_capture.json',dict(
                cuda_capture_ms=sum(start.elapsed_time(end) for start,end in pairs),
                timed_substeps=len(pairs)//2,ground_contact_semantics=self.s.grf_semantics,
                extra_trace_bytes=sum(v.numel()*v.element_size() for k,v in self.arrays.items() if k.startswith('shadow_')),
                note='Warmed device-event copy timing; excludes archive compression/offline solves. Shared GPU interference possible.'))
        if complete:
            result=report.summarize(self.out)
            if getattr(self.args,'audit_wrench_application',False):
                error=arrays['delivered_wrench_base_origin_world']-arrays['expected_wrench_base_origin_world']
                maximum=float(np.abs(error).max())
                # Permissive diagnostic-only check, not a QP/physics tolerance.
                # Preserve raw discrepancies so a passing check cannot hide them.
                tolerance=.01+1e-3*np.abs(arrays['expected_wrench_base_origin_world'])
                result['wrench_delivery_audit']=dict(max_abs_error=maximum,
                    per_axis_max_abs_error=np.abs(error).max(axis=(0,1)).tolist(),
                    atol=.01,rtol=1e-3,max_tolerance_ratio=float((np.abs(error)/tolerance).max()),samples=int(np.prod(error.shape[:-1])),
                    convention='Pre-physics world axes, moment about live base-link origin; raw request retained separately.')
                assert np.isfinite(error).all() and (np.abs(error)<=tolerance).all(), result['wrench_delivery_audit']
            reset=arrays['control_reset'].astype(bool)
            failure=arrays['control_failure'].astype(bool)&reset
            result['failure_per_env_s']=float(failure.sum()/(count*self.e.num_envs*self.e.dt))
            result['completed_episode_length_s']=report.finite_stats(arrays['control_episode_steps'][reset]*self.e.dt)
            result['failure_by_terrain_level']={str(i):int((failure&(arrays['control_terrain_level']==i)).sum()) for i in range(self.e.cfg.terrain.num_rows)}
            result['failure_by_terrain_type']={str(i):int((failure&(arrays['control_terrain_type']==i)).sum()) for i in range(self.e.cfg.terrain.num_cols)}
            # All physical samples occur before reset; exclude entire reset
            # intervals from aligned model comparisons to be conservative.
            mask=np.repeat(~reset,4,axis=0)
            result['diagnostics']={}
            for name,a in arrays.items():
                if name.startswith('objective_') or name in ('velocity_estimate_error','height_estimate_error','wrench_error','grf_error','stance_slip'):
                    result['diagnostics'][name]=report.finite_stats(a[mask])
            for name in ('torque','grf','stance','attitude','height','planar','yaw','planar_error','yaw_error'):
                if 'objective_candidate_'+name not in arrays:continue
                delta=arrays['objective_candidate_'+name]-arrays['objective_nominal_'+name]
                result['diagnostics']['objective_change_'+name]=report.finite_stats(delta[mask])
                if name.endswith('error'):
                    v=delta[mask];v=v[np.isfinite(v)]
                    result['diagnostics']['predicted_worsened_fraction_'+name]=float((v>1e-4).mean()) if len(v) else None
            for k in range(4):
                for group in ('baseline','extra'):
                    selected=(arrays['stage'][k::4]>=0)&(arrays['baseline'][k::4]==(group=='baseline'))
                    result['diagnostics'][f'correction_substep{k}_{group}_nm']=report.finite_stats(abs(arrays['correction'][k::4][selected]))
            delta=arrays['executed_model_delta']-arrays['realized_delta']
            result['aligned_next_substep_error_xyz']= [report.finite_stats(delta[...,i][mask]) for i in range(3)]
            result['unavailable']=['Unfiltered simulator contact impulse timing is not a ground-truth instantaneous force.',
                'No exact physical continuation: fresh episodes sampled from saved terrain distribution.',
                'No hardware latency measurement; collection timing includes diagnostics and shared host interference.']
            write_json(self.out/'result.json',result)
            if getattr(self.args,'wrench_stress',False):
                from scripts.hard_pact_wrench_stress import save_metrics
                save_metrics(self,arrays,result)
                if self.probe is not None:
                    from scripts.hard_pact_recovery_probe import save_probe
                    save_probe(self,arrays,result)
                if self.args.worker=='off':return
            if getattr(self.args,'intervention_sweep',False):
                from scripts.hard_pact_intervention_eval import save
                save(self,arrays,result)
                return
            # Bounded 1-second histories, at most 12 failures and 6 controls.
            captures={}; chosen=list(zip(*np.where(failure)))[:12]
            chosen += [(min(count-1,100+200*i),i%self.e.num_envs) for i in range(6)]
            for j,(end,env_id) in enumerate(chosen):
                start=max(0,end-49)
                previous=np.where(reset[:end,env_id])[0]
                if len(previous):start=max(start,int(previous[-1])+1)
                for name,a in arrays.items():
                    captures[f'{j:02d}_{name}']=a[start:end+1,env_id] if name.startswith('control_') else a[start*4:(end+1)*4,env_id]
            np.savez_compressed(self.out/'bounded_histories.npz',**captures)
            if self.probe is not None:
                from scripts.hard_pact_recovery_probe import save_probe
                save_probe(self,arrays,result)


def worker(args, protocol):
    args.attitude_probe=protocol.get('attitude_probe',False)
    args.intervention_sweep=protocol.get('intervention_sweep',False)
    args.wrench_stress=protocol.get('wrench_stress',False)
    args.load=protocol.get('load','light')
    for key in ('probe_extension','probe_confirmation','capture_cap','probe_source',
                'preservation_visual','recovery_diagnostic','probe_duration','probe_cap',
                'probe_hysteresis_rad','probe_hysteresis_m','video','discard_video','shadow_estimation','momentum_sanity','momentum_sanity_correct_wrench','audit_wrench_application'):
        if key in protocol:setattr(args,key,protocol[key])
    import torch
    from legged_gym.scripts.train_hard_pact import prepare_solver_runtime
    prepare_solver_runtime(['--qp_solver','cupiqp'])
    if getattr(args,'video',False):
        # Isaac Sim 5.1 Replicator annotations use Warp's old module alias.
        # Evaluation process only; same Kernel class, no numerical changes.
        import warp
        if not hasattr(warp,'context'):
            from warp._src import context
            warp.context=context
    torch.cuda.set_device(0)
    import legged_gym.envs
    from legged_gym.utils import task_registry
    from legged_gym.utils.helpers import set_seed,class_to_dict
    from legged_gym.dynamics import create_go2_dynamics
    from scripts.eval_hard_pact_frozen import build_actor, frozen_policy_actions
    from rsl_rl.algorithms.hard_pact_qp import HardPACTDifferentiableQP
    checkpoint=torch.load(protocol['checkpoint'],map_location='cpu',weights_only=False)
    if sha_file(Path(protocol['checkpoint']))!=protocol['checkpoint_sha256']:raise RuntimeError('Checkpoint changed')
    if 'config_sha256' in protocol and sha_file(Path(protocol['config']))!=protocol['config_sha256']:raise RuntimeError('Saved config changed')
    saved=json.loads(Path(protocol['config']).read_text())
    training=saved['training']['resolved']
    task='go2_hard_pact_rss_full_isaaclab'
    cfg=copy.deepcopy(task_registry.env_cfgs[task]);apply_config(cfg,saved['environment']['resolved'])
    cfg.env.num_envs=10;cfg.seed=args.seed
    if args.momentum_sanity:
        from scripts.hard_pact_momentum_sanity import configure
        configure(cfg)
    # Keep saved terrain GENERATION settings; freeze promotions via hooks only.
    set_seed(args.seed)
    env=task_registry.get_task_class(task)(cfg,class_to_dict(cfg.sim),'cuda:0',True)
    app=env.simulator._app_launcher.app
    try:
        with torch.inference_mode():
            state=checkpoint['training_state'];progress=state['transferable_progress'];s=env.simulator
            env.load_domain_rand_curriculum_state_dict(progress['domain'])
            env.load_command_curriculum_state_dict(progress['commands'])
            env._update_terrain_curriculum=lambda ids:None
            env._update_command_curriculum=lambda ids:None
            env.use_tradeoff=False
            cfg.rewards.use_reward_curriculum=False
            env.common_step_counter=progress['common_step_counter']
            env.iteration=progress['iteration']
            # Sample saved JOINT level/type distribution, not play's 2x5 grid
            # or a uniform level distribution. Owned deterministic CPU generator.
            terrain=progress['terrain'];g=torch.Generator().manual_seed(args.seed)
            indices=torch.randperm(len(terrain['terrain_levels']),generator=g)[:10]
            target_plan=None
            if args.attitude_probe:
                from scripts.hard_pact_recovery_probe import targeted_indices
                indices,target_plan=targeted_indices(args.probe_source,terrain,args.seed)
            if args.intervention_sweep:
                from scripts.hard_pact_intervention_eval import targeted_indices
                indices,target_plan=targeted_indices(terrain,saved,protocol['failure_evidence'],args.seed)
            s._terrain_levels.copy_(terrain['terrain_levels'][indices].to(env.device))
            s._terrain_types.copy_(terrain['terrain_types'][indices].to(env.device))
            s._env_origins.copy_(s._terrain_origins[s._terrain_levels,s._terrain_types])
            env.reset()
            if args.momentum_sanity:
                from scripts.hard_pact_momentum_sanity import nominal_properties
                nominal_properties(env)
            actor=build_actor(env,training,checkpoint)
            actor.hard_pact_features=env.hard_pact_features
            from rsl_rl.modules.hard_pact_physics import GRFSwingConfig
            actor.physics_estimator.grf_swing=GRFSwingConfig.from_task(cfg)
            qp_cfg,alpha,probabilities,strength=freeze_settings(training,checkpoint,args.worker)
            qp=HardPACTDifferentiableQP(qp_cfg,s.torque_limits,s.dof_pos_limits_hard[:,0],s.dof_pos_limits_hard[:,1],s.dof_vel_limits)
            dynamics=create_go2_dynamics('bard',str(ROOT/training['algorithm'].get('bard_urdf_path','resources/robots/go2/urdf/go2.urdf')),device=env.device,
                batch_capacity=10,default_joint_position=s.default_dof_pos.reshape(-1,12)[0])
            env.configure_hard_pact_substep_qp(actor,dynamics,qp)
            env._cache_qp_replay_mechanics=False
            env._qp_execution_alpha=alpha;env._qp_intervention_probabilities=probabilities;env._qp_extra_strength=strength
            env.set_hard_pact_qp_enabled(True)
            initial=tensor_hash({'q':env._canonical_configuration(),'v':env._canonical_velocity_world(),**env._canonical_randomized_parameters()})
            manifest=dict(protocol,condition=args.worker,seed=args.seed,effective_qp=asdict(qp.cfg),execution_alpha=alpha,
                extra_strength=strength,substep_probabilities=probabilities,initial_state_hash=initial,scenario_sha256=None,
                terrain_source_rows=indices.tolist(),terrain_levels=s.terrain_levels.tolist(),terrain_types=s.terrain_types.tolist(),
                target_plan=target_plan,
                environment=class_to_dict(cfg),saved_config=saved,dependencies=dependency_versions(),
                wrench_application_semantics=getattr(s,'hard_pact_wrench_application_semantics','legacy'),
                restored_curriculum=dict(next_iteration=state['next_iteration'],
                    progress=dict(env.domain_rand_curriculum.progress),
                    current_wrench_bounds={k:list(env.domain_rand_curriculum.schema[k].range_at(env.domain_rand_curriculum.progress['disturbance']))
                        for k in ('persistent_force','persistent_vertical_force','persistent_torque','persistent_yaw_torque')}),
                notes=['Native command/heading/disturbance/reset protocol retained. QP sampler RNG isolated.',
                    'Initial states matched by seed; commands/randomization diverge after asynchronous resets.',
                    'Training terrain geometry and saved empirical joint level/type distribution; no play terrain overrides.',
                    'Physics-step differences compare derivatives in evolving body frames, matching canonical qdd.',
                    'Bounded numeric trace: <=2000 control steps, 10 envs; no solver matrices/graphs retained.'])
            matched.ARGS=args
            observer=Observer(env,actor,qp,args,manifest)
            if args.audit_wrench_application:
                from scripts.hard_pact_momentum_sanity import audit_application
                assert s.hard_pact_wrench_application_semantics=='base_origin_explicit_moment_v2'
                manifest['production_wrench_audit']=audit_application(s)
            if args.wrench_stress:
                from scripts.hard_pact_wrench_stress import install
                install(observer)
            if args.momentum_sanity:
                from scripts.hard_pact_momentum_sanity import install as install_sanity
                install_sanity(observer,args.momentum_sanity)
            if getattr(args,'video',False):
                from scripts.hard_pact_eval_video import Recorder
                observer.video=Recorder(observer)
            if args.intervention_sweep:
                manifest['execution_diagnostics_convention']='Execution costs/constraints hold optimized GRFs fixed; not an executed force measurement or certificate.'
            write_json(observer.out/'manifest.json',manifest)
            observer.start=time.monotonic();torch.cuda.reset_peak_memory_stats()
            for t in range(args.steps):
                observer.t,observer.k=t,0
                if args.wrench_stress:observer.tapes.before(t)
                observer.command=env.commands[:,:3].clone()
                observer.record('command',observer.command,control=True)
                obs,history,_,_=env.get_observations()
                action=frozen_policy_actions(actor,obs,history)
                env.step(action)
                if (t+1)%250==0 or t==args.steps-1:
                    if not torch.isfinite(action).all() or not torch.isfinite(env.obs_buf).all():
                        raise RuntimeError('Nonfinite policy/observation; failed trial retained')
                if t==99:
                    torch.cuda.synchronize();observer.warm_start=time.monotonic()
                if (t+1)%250==0:
                    write_json(observer.out/'live.json',dict(steps=t+1,wall_s=time.monotonic()-observer.start))
                    print('PROGRESS',args.worker,args.seed,t+1,args.steps,flush=True)
            torch.cuda.synchronize()
            observer.save(args.steps,True)
            if observer.video is not None:observer.video.close()
            assert tensor_hash(actor.state_dict())==observer.before_hash
            print('COMPLETE',args.worker,args.seed,flush=True)
    except BaseException:
        import traceback
        text=traceback.format_exc()
        print(text,flush=True)
        write_json(args.output/f'{args.worker}_seed{args.seed}'/'failure.json',dict(traceback=text))
        raise
    finally:
        env.simulator._sim.stop()
        env.simulator._sim.clear_instance()
        app.close(wait_for_replicator=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--physical-gpu',type=int,default=1)
    p.add_argument('--steps',type=int,default=2000)
    p.add_argument('--seeds',type=int,nargs='+',default=[1,2,3])
    p.add_argument('--worker',choices=ALL_VARIANTS)
    p.add_argument('--seed',type=int,default=1)
    p.add_argument('--variants',nargs='+',choices=ALL_VARIANTS)
    p.add_argument('--attitude-probe',action='store_true',help='Four targeted seed-1 attitude/probe conditions')
    p.add_argument('--intervention-sweep',action='store_true',help='Four-solve A/B/C/D sweep; no termination changes')
    p.add_argument('--wrench-stress',action='store_true',help='Reset-independent evaluation tapes and frozen mechanics')
    p.add_argument('--load',choices=['light','heavier'],default='light')
    p.add_argument('--event',default=EVENT,help='Exact event filename identifying the run')
    p.add_argument('--probe-source',type=Path,default=ROOT/'diagnostics/tradeoff11000/full')
    p.add_argument('--probe-extension',type=float,default=.2)
    p.add_argument('--probe-confirmation',type=float,default=.5)
    p.add_argument('--capture-cap',type=int,default=24)
    p.add_argument('--preservation-visual',action='store_true')
    p.add_argument('--recovery-diagnostic',action='store_true')
    p.add_argument('--probe-duration',type=float,default=1.)
    p.add_argument('--probe-cap',type=int,default=3)
    p.add_argument('--probe-hysteresis-rad',type=float,default=.05)
    p.add_argument('--probe-hysteresis-m',type=float,default=.02)
    p.add_argument('--video',action='store_true')
    p.add_argument('--discard-video',action='store_true',help='Match video runtime/render calls, discard frames without encoding')
    p.add_argument('--shadow-estimation',action='store_true',help='Eval-only owned nominal mechanics/velocity capture for causal offline inverse dynamics')
    p.add_argument('--audit-wrench-application',action='store_true',help='Lightweight pre-physics request/label/composer capture; supports QP-off')
    p.add_argument('--momentum-sanity',choices=['standing','walking'],help='Eval-only nominal mechanics and flat terrain with known load pulses')
    p.add_argument('--momentum-sanity-correct-wrench',action='store_true',help='Sanity ONLY: compose point-force moment explicitly to bypass installed composer torque-overwrite bug')
    p.add_argument('--checkpoint',type=Path,help='Explicit immutable checkpoint from the exact event-file run')
    args=p.parse_args();args.output=args.output.resolve()
    if args.discard_video:args.video=True
    if args.wrench_stress:args.intervention_sweep=True
    if args.recovery_diagnostic or args.video:args.preservation_visual=True
    if args.recovery_diagnostic and not args.wrench_stress:p.error('Recovery diagnostic requires matched wrench tapes')
    if not 1<=args.steps<=2000: p.error('bounded evaluation requires 1..2000 steps')
    if args.worker:
        protocol=json.loads((args.output/'protocol.json').read_text())
        args.physical_gpu=protocol.get('physical_gpu',1)
        if os.environ.get('CUDA_VISIBLE_DEVICES')!=str(args.physical_gpu):raise RuntimeError('Physical GPU visibility mismatch')
        return worker(args,protocol)
    if args.physical_gpu<0:p.error('physical GPU must be nonnegative')
    if args.attitude_probe and args.seeds!=[1]:p.error('Attitude/probe experiment requires --seeds 1')
    if args.capture_cap<0:p.error('capture cap must be nonnegative')
    if args.intervention_sweep and args.attitude_probe:p.error('Distinct experiments; do not combine presets')
    if args.variants is None:args.variants=list(INTERVENTION_VARIANTS)[:4] if args.intervention_sweep else list(ATTITUDE_VARIANTS if args.attitude_probe else VARIANTS)
    if args.intervention_sweep and not set(args.variants)<=set(INTERVENTION_VARIANTS):p.error('Invalid intervention variant')
    if args.shadow_estimation and (not args.wrench_stress or not args.intervention_sweep or 'off' in args.variants):
        p.error('Shadow capture requires a QP-enabled intervention/wrench-stress evaluation')
    if args.momentum_sanity and not args.shadow_estimation:p.error('Momentum sanity requires shadow capture')
    if args.momentum_sanity_correct_wrench and not args.momentum_sanity:p.error('Corrected load delivery is restricted to nominal sanity checks')
    args.output.mkdir(parents=True,exist_ok=True)
    protocol_path=args.output/'protocol.json'
    if protocol_path.exists():
        protocol=json.loads(protocol_path.read_text())
        if protocol['steps']!=args.steps or protocol['evaluator_sha256']!=sha_file(Path(__file__)):
            raise RuntimeError('Use new output directory after code/protocol changes')
    else:
        matches=list((ROOT/'logs').rglob(args.event))
        if len(matches)!=1:raise RuntimeError(f'Expected exact event file once; found {matches}')
        run=matches[0].parent
        checkpoint=(args.checkpoint.resolve() if args.checkpoint else
                    max(run.glob('model_*.pt'),key=lambda f:int(f.stem.split('_')[-1])))
        if args.attitude_probe:
            prior=json.loads((args.probe_source/'protocol.json').read_text())
            checkpoint=Path(prior['checkpoint'])
            if sha_file(checkpoint)!=prior['checkpoint_sha256']:raise RuntimeError('Source checkpoint changed')
        if checkpoint.parent!=run.resolve():raise RuntimeError('Checkpoint is not from the specified event-file run')
        protocol=dict(schema=1,event=str(matches[0]),checkpoint=str(checkpoint),checkpoint_sha256=sha_file(checkpoint),
            checkpoint_iteration=int(checkpoint.stem.split('_')[-1]),config=str(run/'hard_pact_resolved_config.json'),
            code_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            evaluator_sha256=sha_file(Path(__file__)),steps=args.steps,physical_gpu=args.physical_gpu,seeds=args.seeds,variants=args.variants,
            attitude_probe=args.attitude_probe,probe_source=str(args.probe_source.resolve()),
            intervention_sweep=args.intervention_sweep,
            wrench_stress=args.wrench_stress,load=args.load,
            probe_extension=args.probe_extension,probe_confirmation=args.probe_confirmation,capture_cap=args.capture_cap,
            preservation_visual=args.preservation_visual,recovery_diagnostic=args.recovery_diagnostic,
            probe_duration=args.probe_duration,probe_cap=args.probe_cap,
            probe_hysteresis_rad=args.probe_hysteresis_rad,probe_hysteresis_m=args.probe_hysteresis_m,video=args.video,
            discard_video=args.discard_video,shadow_estimation=args.shadow_estimation,momentum_sanity=args.momentum_sanity,
            momentum_sanity_correct_wrench=args.momentum_sanity_correct_wrench,
            audit_wrench_application=args.audit_wrench_application,
            argv=sys.argv,restart='Completed trials skipped; interrupted trials restart from seed, not exact PhysX.')
        protocol['config_sha256']=sha_file(Path(protocol['config']))
        protocol['diagnostic_helper_sha256']=sha_file(ROOT/'scripts/hard_pact_intervention_eval.py')
        if args.intervention_sweep:
            from scripts.hard_pact_intervention_eval import failure_evidence
            protocol['failure_evidence']=failure_evidence(matches[0],protocol['checkpoint_iteration'])
            protocol['followup_rule']='Preliminary promising: fewer failures, each tracking RMSE <= reference*1.10+0.01, no >0.001 increase in joint violation frequency, no >0.01 Nm max torque excess. Not a significance test.'
        if args.wrench_stress:
            protocol['followup_rule']='Rank Stage 1 by failure count, then roll/pitch RMS, then height error RMS, then summed tracking RMSE. No tracking-regression veto. Only seed2 selected-versus-off at light/heavier.'
            if any(v in args.variants for v in ('C_torque30','D_true_wrench')):
                protocol['followup_rule']='Seed 3 B/C confirmation only if C offers promising survival/stability with useful tracking. Oracle is diagnostic only; use restored checkpoint envelope.'
            protocol['tape_helper_sha256']=sha_file(ROOT/'scripts/hard_pact_wrench_stress.py')
        write_json(protocol_path,protocol)
        import shutil
        archive=args.output/'sources';archive.mkdir(exist_ok=True)
        for source in (Path(__file__),ROOT/'scripts/hard_pact_intervention_eval.py',ROOT/'scripts/report_hard_pact_tradeoff.py',Path(protocol['config'])):
            shutil.copy2(source,archive/source.name)
        if args.shadow_estimation:
            shutil.copy2(ROOT/'scripts/hard_pact_shadow_estimation.py',archive/'hard_pact_shadow_estimation.py')
        if args.wrench_stress:shutil.copy2(ROOT/'scripts/hard_pact_wrench_stress.py',archive/'hard_pact_wrench_stress.py')
    for seed in args.seeds:
        for variant in args.variants:
            out=args.output/f'{variant}_seed{seed}';out.mkdir(exist_ok=True)
            if (out/'result.json').exists():continue
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(args.physical_gpu),SIMULATOR='isaaclab',PYTHONPATH=str(ROOT),
                PYTHONUNBUFFERED='1',OMP_NUM_THREADS='1',LD_PRELOAD=str(Path(sys.executable).parents[1]/'lib/libstdc++.so.6'))
            if args.video:
                env['ENABLE_CAMERAS']='1'
                if os.environ.get('HARDPACT_EVAL_NUMPY_PATH'):
                    env['PYTHONPATH']=os.environ['HARDPACT_EVAL_NUMPY_PATH']+os.pathsep+str(ROOT)
            with (out/'run.log').open('w') as log:
                status=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker',variant,'--seed',str(seed),
                    '--steps',str(args.steps),'--output',str(args.output)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            if status.returncode or not (out/'result.json').exists():raise RuntimeError(f'{variant}/{seed} failed: {out}/run.log')
            print('COMPLETE',variant,seed,flush=True)
    results=[json.loads(f.read_text()) for f in args.output.glob('*/result.json')]
    write_json(args.output/'summary.json',results)
    with (args.output/'summary.csv').open('w') as f:
        writer=csv.writer(f);writer.writerow(['variant','seed','failure/env_s','failures','timeouts','rmse_x','rmse_y','rmse_yaw','primary','recovery','fallback'])
        for r in results:writer.writerow([r['condition'],r['seed'],r['failure_per_env_s'],r['physical_resets'],r['timeout_resets'],*r['tracking_rmse'],r['primary_rows'],r['recovery_rows'],r['fallback_rows']])


if __name__=='__main__':main()
