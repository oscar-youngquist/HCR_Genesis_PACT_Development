"""Frozen PILOT-only RSS compliance evaluation; native termination and resets."""
import argparse
import ast
import copy
import csv
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
# Load numerical utilities without selecting a simulator through utils/__init__.
spec=importlib.util.spec_from_file_location('pilot_compliance_eval',ROOT/'legged_gym/utils/pilot_compliance_eval.py')
util=importlib.util.module_from_spec(spec);spec.loader.exec_module(util)
EVAL_CONFIG=copy.deepcopy(util.EVAL_CONFIG)  # Editable protocol; no training defaults modified.


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,default=ROOT/'logs/b1z1/b1z1_pact_lab'/util.DEFAULT_RUN)
    p.add_argument('--checkpoint');p.add_argument('--seeds',type=int,nargs='+',default=[0,1,2])
    p.add_argument('--episodes-per-condition',type=int,default=100)
    p.add_argument('--calibration-episodes',type=int,default=10,help='Per force level per seed; 0 disables the separate calibration')
    p.add_argument('--conditions',nargs='+',choices=['nominal',*util.CONDITIONS,*util.CALIBRATIONS],default=list(util.CONDITIONS))
    p.add_argument('--config',type=Path);p.add_argument('--gpu',default='cuda:1')
    p.add_argument('--num-envs',type=int,default=64);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--headless',action='store_true');p.add_argument('--validate-only',action='store_true')
    return p


def dump(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False))


def restore_sections(obj,saved):
    """Restore resolved scalars/sections without converting gain dictionaries to objects."""
    for k,v in saved.items():
        current=getattr(obj,k,None)
        if isinstance(v,dict) and current is not None and not isinstance(current,dict):
            restore_sections(current,v)
        else: setattr(obj,k,copy.deepcopy(v))


def diagnostic_free_ast(source):
    """Audit that saved reference dynamics differ ONLY by opt-in snapshot statements."""
    class Strip(ast.NodeTransformer):
        def visit_If(self,node):
            if isinstance(node.test,ast.Attribute) and ast.unparse(node.test)=='self.collect_diagnostics': return None
            return self.generic_visit(node)
        def visit_Assign(self,node):
            if any(ast.unparse(t) in ('self.collect_diagnostics','self.diagnostics') for t in node.targets): return None
            return self.generic_visit(node)
    return ast.dump(Strip().visit(ast.parse(source)),include_attributes=False)


def witness_poses(cfg,c):
    """Finite FK grid certifies candidate geometry, not policy reachability or full-arm IK."""
    import torch
    import xml.etree.ElementTree as ET
    from types import SimpleNamespace
    from legged_gym.envs.b1z1.compliance_geometry import RobotGeometry
    from legged_gym.envs.b1z1.directional_compliance import spherical,directional_probe
    names=cfg.asset.dof_names;ids=list(range(12,18))
    cc=SimpleNamespace(**vars(cfg.compliance));cc.compile_geometry=False
    path=cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=str(ROOT))
    root=ET.parse(path).getroot()
    limits=torch.tensor([[float(root.find(f"joint[@name='{n}']/limit").get(k)) for k in ('lower','upper')] for n in names])
    geometry=RobotGeometry(path,names,ids,cfg.asset.base_name,cfg.asset.gripper_name,cfg.asset.foot_name,cc,'cpu')
    q=torch.tensor([[cfg.init_state.default_joint_angles[n] for n in names]])
    grid=torch.cartesian_prod(*[torch.linspace(float(limits[i,0])+.06,float(limits[i,1])-.06,c['witness_grid']) for i in (13,14,15)])
    q=q.repeat(len(grid),1);q[:,13:16]=grid
    pos,j,arm,legs=geometry.forward(q)
    center=torch.tensor([cfg.goal_ee.sphere_center.x_offset,cfg.goal_ee.sphere_center.y_offset,
                        cfg.goal_ee.sphere_center.z_invariant_offset-cfg.init_state.pos[2]])
    sphere=spherical(pos-center)
    bounds=torch.tensor([cfg.goal_ee.ranges.pos_l,cfg.goal_ee.ranges.pos_p,cfg.goal_ee.ranges.pos_y])
    clearance=geometry.point_clearance(pos,arm,legs).amin(-1)
    valid=((sphere>bounds[:,0])&(sphere<bounds[:,1])).all(1)&(clearance>cc.collision_margin)
    local=pos-center
    valid &= ~((local>torch.tensor(cfg.goal_ee.collision_lower_limits))&(local<torch.tensor(cfg.goal_ee.collision_upper_limits))).all(1)
    valid &= local[:,2]>cfg.goal_ee.underground_limit
    if not valid.any(): raise ValueError('No feasible finite-grid pose witnesses')
    report={}
    for key in ('vertical_sphere','ample_sphere','restricted_sphere'):
        distances=(sphere-torch.tensor(c[key])).square().sum(1);distances[~valid]=float('inf')
        i=int(distances.argmin());selected=sphere[i]
        if distances[i]>.08**2: raise ValueError(f'No nearby feasible FK witness for {key}')
        direction=torch.tensor([[cc.probe_distance,0.,0.]])
        _,gate,progress=directional_probe(j[i:i+1],q[i:i+1,ids],limits[ids][None],direction,cc)
        c[key]=selected.tolist()
        report[key]=dict(sphere=c[key],joint_witness=q[i,ids].tolist(),ee_body_m=pos[i].tolist(),
            clearance_m=float(clearance[i]),forward_gate=float(gate[0]),forward_progress=float(progress[0]),
            spherical_margins=torch.minimum(selected-bounds[:,0],bounds[:,1]-selected).tolist())
    # Explicit vertical geometric travel checks for the maximum requested yield.
    r=report['vertical_sphere'];p=torch.tensor(r['ee_body_m'])-center
    extent=c['force_n']*max(c['amplitude_fraction'])*c['yielding_coefficient']*cc.ee_gain[2]
    for sign in (-1,1):
        s=spherical((p+torch.tensor([0,0,sign*extent]))[None])[0]
        if not ((s>bounds[:,0])&(s<bounds[:,1])).all(): raise ValueError('Vertical yielding envelope exceeds spherical bounds')
    report['vertical_requested_equilibrium_extent_m']=extent
    return report


def make_environment_class(base,c):
    import torch
    from legged_gym.envs.b1z1.directional_compliance import heading,rotate_yaw,task_position,spherical
    from legged_gym.utils.math_utils import quat_from_euler_xyz

    class RSSComplianceEnv(base):
        ready=False;capture=False;eval_time=0.

        def _resample_commands(self,ids):
            self.commands[ids,:3]=0

        def _resample_ee_goal(self,ids,is_init=False):
            if not self.ready: return super()._resample_ee_goal(ids,is_init)
            if not is_init: return
            self.ee_start_sphere[ids]=self.init_start_ee_sphere
            self.ee_goal_sphere[ids]=self.eval_spheres[ids]
            self.curr_ee_goal_sphere[ids]=self.init_start_ee_sphere
            self.commands[ids,3:6]=self.init_start_ee_sphere
            self.goal_timer[ids]=0;self.traj_timesteps[ids]=c['trajectory_s']/self.dt
            self.traj_total_timesteps[ids]=1e9

        def _reset_dofs(self,ids):
            if not self.ready: return super()._reset_dofs(ids)
            q=self.simulator.default_dof_pos.repeat(len(ids),1)
            q[:,:17]+=self.eval_joint_delta[ids]
            self.simulator.reset_dofs(ids,q,torch.zeros_like(q))

        def _reset_root_states(self,ids):
            if not self.ready: return super()._reset_root_states(ids)
            s=self.simulator;p=s.base_init_pos.reshape(1,3)+s._env_origins[ids]
            p[:,:2]+=self.eval_xy[ids];yaw=self.eval_yaw[ids];zero=torch.zeros_like(yaw)
            quat=quat_from_euler_xyz(zero,zero,yaw)
            s.reset_root_states(ids,p,quat,torch.zeros_like(p),torch.zeros_like(p))

        def reset_idx(self,ids):
            super().reset_idx(ids)
            if self.ready and len(ids):
                # Override only newly sampled coefficients, before first observations.
                # Reference/filter/action/history clearing remains the native reset.
                self.compliance.reference.profile[ids]=self.eval_profiles[ids]
                self.compliance.reference.family_ids[ids]=6

        def _post_physics_step_callback(self):
            self._step_contact_targets();self.update_curr_ee_goal()

        def _push_gripper(self,ids):
            if not self.ready:
                self.ee_force_ext_world.zero_()
            else:
                yaw=heading(self.simulator.base_quat)
                local=self.eval_force_direction*self.eval_amplitude[:,None]*float(util.force_scale(self.eval_time,c))
                local*=self.eval_active[:,None]
                self.ee_force_ext_world.copy_(rotate_yaw(local,yaw))
            self.simulator.apply_ee_force(self.ee_force_ext_world)
            if self.capture:
                self.before_reference_count=self.compliance.wrench_count.clone()
                self.force_start=self.ee_force_ext_world.clone()
                self.ee_start=self.simulator.ee_pos.clone()
                self.base_start=self.simulator.base_pos.clone()

        def check_termination(self):
            super().check_termination()
            if not self.capture: return
            s=self.simulator;ref=self.compliance.target;yaw=heading(s.base_quat)
            finite=torch.isfinite(s.dof_pos).all(1)&torch.isfinite(s.dof_vel).all(1)&torch.isfinite(s.ee_pos).all(1)&torch.isfinite(ref.position).all(1)
            # Only nonfinite infrastructure failures add a terminal condition;
            # all native physical and compliance-recovery checks remain intact.
            self.fail_buf |= ~finite;self.reset_buf |= ~finite
            nominal=self.compliance.nominal(0)[0]
            measured=task_position(s.ee_pos,s.base_pos,yaw)
            sphere=spherical(measured-self.compliance.center());bounds=self.compliance.bounds
            margin=torch.minimum(sphere-bounds[:,0],bounds[:,1]-sphere)
            limits=getattr(s,'dof_pos_limits_hard',s.dof_pos_limits)
            if limits.ndim==2: limits=limits[None]
            joint_margin=torch.minimum(s.dof_pos[:,12:18]-limits[:,12:18,0],limits[:,12:18,1]-s.dof_pos[:,12:18]).amin(1)
            reasons=torch.stack((self.contact_fail_buf,self.roll_fail_buf,self.pitch_fail_buf,
                self.height_min_fail_buf,self.height_max_fail_buf,self.compliance.recovery_failed,~finite,self.time_out_buf),-1)
            applied=s._external_force_world[:,s._gripper_index]
            # Capture all rows BEFORE native reset_idx mutates terminal state/targets.
            self.transition={k:v.detach().clone() for k,v in dict(
                valid_transition=self.eval_active,alive_for_metrics=self.eval_active&~self.reset_buf,
                failure=self.reset_buf,failure_reasons=reasons,base_world=s.base_pos,base_quat_xyzw=s.base_quat,
                ee_world=s.ee_pos,ee_world_start=self.ee_start,base_world_start=self.base_start,
                ee_task=measured,ee_velocity_task=self.compliance.measured_velocity(),
                reference_task=ref.position,reference_velocity_task=ref.velocity,nominal_task=nominal,
                base_measured=self.compliance.measured_base(),base_reference=ref.base,
                nominal_commands=self.commands,profile=self.compliance.profile,
                applied_force_world=applied,force_start_world=self.force_start,
                base_force_world=s._external_force_world[:,s._base_link_index],
                base_moment_world=s._external_torque_world[:,s._base_link_index],
                gates=ref.gates,workspace_gate=ref.workspace,joint_gate=ref.joint,collision_gate=ref.collision,
                workspace_margin=margin,joint_margin=joint_margin,q=s.dof_pos,qd=s.dof_vel,
                executed_torque=s.torques,clipped=ref.clipped,recovering=self.compliance.reference.recovering,
                reference_advances=self.compliance.wrench_count-self.before_reference_count).items()}
            if self.compliance.reference.collect_diagnostics:
                self.transition.update({'reference_stage_'+k:v.detach().clone()
                    for k,v in self.compliance.reference.diagnostics.items()})
    return RSSComplianceEnv


def main(argv=None):
    args=parser().parse_args(argv)
    if min(args.num_envs,args.episodes_per_condition)<1 or args.calibration_episodes<0 or len(set(args.seeds))!=len(args.seeds) or len(set(args.conditions))!=len(args.conditions):
        raise ValueError('Positive counts and unique seeds/conditions required')
    if args.output_dir.exists() and any(args.output_dir.iterdir()): raise FileExistsError('Use a new output directory')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    os.environ['SIMULATOR']='isaaclab_b1z1_pact'
    os.environ.setdefault('TORCHINDUCTOR_COMPILE_THREADS','1')
    import numpy as np
    import torch
    torch.manual_seed(args.seeds[0]);np.random.seed(args.seeds[0])
    import legged_gym.envs
    from legged_gym.utils.helpers import class_to_dict
    from legged_gym.scripts.eval_b1z1_exp1 import build_actor,inference,model_hash,util as shared
    from legged_gym.envs.b1z1.b1z1_pact.b1z1_pact import B1Z1PACT
    path,iteration=util.resolve_checkpoint(args.run,args.checkpoint)
    state=torch.load(path,map_location='cpu',weights_only=False)  # Trusted local checkpoint; never resume its training state.
    contract=state.get('training_state',{}).get('contract')
    if not contract: raise ValueError('Resolved checkpoint contract required; do not infer semantic compatibility from tensor dimensions')
    archive_path=path.parent/'b1z1_pact_config.py'
    spec=importlib.util.spec_from_file_location('rss_archive',archive_path);archive=importlib.util.module_from_spec(spec);spec.loader.exec_module(archive)
    cfg=shared.materialize(archive.B1Z1PACTCfg());training=shared.materialize(archive.B1Z1PACTCfgPPO())
    restore_sections(cfg,contract['environment']);restore_sections(training,contract['training'])
    cfg._compliance_layout_added=True  # Saved dimensions already include the 12 coefficients.
    if cfg.env.num_observations!=93 or cfg.env.num_policy_actions!=34 or training.runner_class_name!='B1Z1PACTRunner' or not cfg.compliance.enabled:
        raise ValueError('Unsupported saved PILOT observation/action/compliance contract')
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    saved_revision=state['training_state'].get('source_revision')
    critical=['legged_gym/envs/b1z1/b1z1_pact/b1z1_pact.py','legged_gym/envs/b1z1/compliance_task.py',
        'rsl_rl/modules/actor_critic_b1z1_pact.py']
    if not saved_revision or subprocess.check_output(['git','diff',saved_revision,'--',*critical],cwd=ROOT):
        raise ValueError('Policy/observation/compliance source differs from the saved revision; explicit semantic audit required')
    reference_path='legged_gym/envs/b1z1/directional_compliance.py'
    original=subprocess.check_output(['git','show',f'{saved_revision}:{reference_path}'],cwd=ROOT,text=True)
    if diagnostic_free_ast(original)!=diagnostic_free_ast((ROOT/reference_path).read_text()):
        raise ValueError('Reference dynamics changed beyond the audited opt-in diagnostics')
    c=copy.deepcopy(EVAL_CONFIG)
    if args.config:
        override=json.loads(args.config.read_text())
        if override.keys()-c.keys(): raise ValueError('Unknown evaluation configuration fields')
        c.update(override)
    util.validate_config(c)
    pose_report=witness_poses(cfg,c)
    if c['yielding_coefficient']<=cfg.compliance.stiffness_threshold: raise ValueError('EE coefficient must exceed stiffness recruitment threshold')
    if c['force_n']*max(c['amplitude_fraction'])>min(abs(x) for x in cfg.commands.max_push_force_xyz_gripper_ext):
        raise ValueError('Requested load exceeds trained per-axis range')
    calibration_conditions=list(util.CALIBRATIONS) if args.calibration_episodes and 'nominal' not in args.conditions else []
    calibration_conditions=[n for n in calibration_conditions if n not in args.conditions]
    for name in [*args.conditions,*calibration_conditions]:
        p=util.condition(name,c)['profile']
        lo=cfg.compliance.ee_profile_low+cfg.compliance.base_profile_low
        hi=cfg.compliance.ee_profile_high+cfg.compliance.base_profile_high
        if not all(a<=b<=d for a,b,d in zip(lo,p,hi)): raise ValueError('Profile outside saved training domain')
    cfg.env.num_envs=args.num_envs;cfg.env.env_spacing=3.
    cfg.env.episode_length_s=c['horizon_s']+2
    cfg.env.render_ee_goal_debug=False;cfg.env.render_ee_frame_debug=False
    cfg.terrain.mesh_type='plane';cfg.terrain.curriculum=False
    cfg.commands.curriculum=False;cfg.commands.heading_command=False
    cfg.commands.push_gripper_stators=True;cfg.commands.apply_ee_external_forces=True
    cfg.commands.push_robot_base=False;cfg.commands.apply_base_external_forces=False;cfg.commands.apply_base_external_torques=False
    cfg.commands.compliance_disturbance_mode_weights=None
    cfg.commands.use_shared_rejection_curriculum=False
    cfg.control.use_tradeoff_curriculum=False;cfg.control.randomize_pact_weights=False
    cfg.noise.add_noise=False  # Controlled mechanics/observations; only manifest initialization varies.
    cfg.domain_rand.use_domainrand_curriculum=False;cfg.domain_rand.push_robots=False
    for key in vars(cfg.domain_rand):
        if key.startswith('randomize_'): setattr(cfg.domain_rand,key,False)
    # Fixed midpoint passive mechanics, inside saved training domains. PD/motor gains remain nominal.
    for flag,key,value in [('friction','friction_range',1.),('joint_armature','joint_armature_range',.015),
                            ('joint_friction','joint_friction_range_end',.1),('joint_damping','joint_damping_range_end',.4)]:
        setattr(cfg.domain_rand,'randomize_'+flag,True);setattr(cfg.domain_rand,key,[value,value])
    cfg.domain_rand.isaacgym_use_final_domain_rand_ranges=True
    cfg.domain_rand.randomize_ctrl_delay=True;cfg.domain_rand.ctrl_delay_step_range=[0,0]
    cfg.compliance.compile_geometry=c['compile_geometry']
    cfg.compliance.ee_profile_low=cfg.compliance.ee_profile_high=[0.]*6
    cfg.compliance.base_profile_low=cfg.compliance.base_profile_high=[0.]*6
    # No optimizer, runner, BARD or training storage is constructed.
    from types import SimpleNamespace
    stub=SimpleNamespace(device='cpu',num_obs=cfg.env.num_observations,num_privileged_obs=cfg.env.num_privileged_obs,
        num_crit_obs_stack=cfg.env.num_priv_stack,num_actions=cfg.env.num_actions,
        num_obs_hist=cfg.env.num_obs_hist,num_exp_labels=cfg.env.num_explicit_recon_obs)
    actor=build_actor(stub,training,state)
    actor.requires_grad_(False);before=model_hash(actor)
    with torch.inference_mode():
        action=inference(actor,torch.zeros(2,stub.num_obs),torch.zeros(2,stub.num_obs*stub.num_obs_hist))
    if action.shape!=(2,34) or not torch.isfinite(action).all(): raise ValueError('Invalid strict native inference')
    actor.reset()
    metadata=dict(task='b1z1_pact',checkpoint=str(path),checkpoint_iteration=iteration,checkpoint_sha256=shared.file_hash(path),
        command=[sys.executable,*sys.argv],schema_version=2,
        evaluator_sha256=shared.file_hash(Path(__file__)),utility_sha256=shared.file_hash(ROOT/'legged_gym/utils/pilot_compliance_eval.py'),
        urdf_sha256=shared.file_hash(Path(cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=str(ROOT)))),
        git_revision=revision,saved_revision=saved_revision,dirty_status=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True),
        saved_contract=contract,resolved_environment=class_to_dict(cfg),resolved_training=class_to_dict(training),
        evaluation_config=c,pose_witnesses=pose_report,training_seed=contract['training'].get('seed'),
        evaluation_seeds=args.seeds,conditions=args.conditions,episodes_per_condition=args.episodes_per_condition,
        calibration_conditions=calibration_conditions,calibration_episodes=args.calibration_episodes,
        reference_diagnostics=c['reference_diagnostics'],reference_ast_parity=True,
        policy_hash_before=before,strict_weights_load=True,semantic_source_match=True,
        frames=dict(ee_task='base-heading-relative xy, world absolute z; m',ee_velocity_task='mixed-coordinate derivative including base/yaw transport; m/s',
            base_measured='heading vx/vy [m/s], Euler yaw rate [rad/s]',force='world N, ON robot, gripper link origin; held over exact interval',
            time='successor seconds; interval_start is force/action time',workspace_margin='radius m, elevation/azimuth rad',
            profile_order='EE [-x,+x,-y,+y,-z,+z], base [-x,+x,-y,+y,-yaw,+yaw]',phase=['settling','ramp_up','sustained','ramp_down','recovery']),
        limitations=['Fixed nominal mechanics, zero observation noise/delay for controlled force experiments; not broad domain-randomization robustness.',
                    'FK witnesses certify only sampled EE geometry, not learned reachability or full articulated collision freedom.',
                    'Exactly one trained checkpoint; seeds are evaluation repetitions.'])
    dump(args.output_dir/'metadata.json',metadata)
    print(f'PINNED {path} sha256={metadata["checkpoint_sha256"]}',flush=True)
    if args.validate_only: return
    all_manifests={seed:util.manifests(seed,args.episodes_per_condition,args.conditions,c)+
        util.manifests(seed,args.calibration_episodes,calibration_conditions,c) for seed in args.seeds}
    dump(args.output_dir/'trial_manifest.json',all_manifests)
    env=None;episodes=[];started=time.perf_counter()
    try:
        env=make_environment_class(B1Z1PACT,c)(cfg,class_to_dict(cfg.sim),args.gpu,args.headless)
        env.compliance.reference.collect_diagnostics=c['reference_diagnostics']
        actor=actor.to(env.device);env.ready=True
        n=env.num_envs;steps=int(round(c['horizon_s']/env.dt))
        metadata['control_dt']=env.dt;metadata['physics_dt']=env.simulator._sim_params['dt']
        metadata['gpu']=args.gpu;metadata['gpu_name']=torch.cuda.get_device_name(env.device)
        dump(args.output_dir/'metadata.json',metadata)
        for seed in args.seeds:
            for batch_index,batch in enumerate(util.batches(all_manifests[seed],n)):
                padded=batch+[batch[-1]]*(n-len(batch));device=env.device
                tensor=lambda key:torch.tensor([x[key] for x in padded],device=device,dtype=torch.float32)
                env.eval_profiles=tensor('profile');env.eval_spheres=tensor('nominal_sphere')
                env.eval_joint_delta=tensor('joint_delta_rad');env.eval_xy=tensor('initial_xy_m')
                env.eval_yaw=tensor('initial_yaw_rad');env.eval_amplitude=tensor('force_amplitude_n')
                env.eval_force_direction=tensor('force_heading_direction')
                env.eval_active=torch.arange(n,device=device)<len(batch)
                env.capture=False;env.eval_time=0.
                # Native stepping creates inference tensors: manual inter-batch resets
                # must use the same scope as automatic resets inside env.step().
                with torch.inference_mode():
                    env.reset_idx(env.all_env_ids);actor.reset()
                    # Refresh kinematics without advancing physics or compliance.
                    sim=env.simulator;sim._sim.forward();sim._robot.update(0.);sim.post_physics_step()
                    env._refresh_curr_ee_goal_world();env.compute_observations()
                    obs,hist,_,_=env.get_observations()
                assert torch.equal(env.compliance.profile,env.eval_profiles)
                assert torch.equal(obs[:,-18:-6],env.eval_profiles)
                assert actor._film_history is None
                env.capture=True;records=[]
                with torch.inference_mode():
                    for step in range(steps):
                        env.eval_time=step*env.dt
                        action=inference(actor,obs,hist)
                        if not torch.isfinite(action).all(): raise RuntimeError('Nonfinite native inference; infrastructure failure')
                        action=torch.where(env.eval_active[:,None],action,torch.zeros_like(action))
                        obs,_,hist,_,_,dones,_,_=env.step(action)
                        row=env.transition;row['raw_action']=action.clone();records.append(row)
                        env.eval_active &= ~row['failure']
                        actor.reset(dones)
                        if (step+1)%200==0:
                            print(f'seed={seed} batch={batch_index+1} time={(step+1)*env.dt:.1f}s alive={int(env.eval_active.sum())}/{len(batch)}',flush=True)
                # One transfer per channel/batch, not per robot or physics substep.
                arrays={k:torch.stack([r[k] for r in records]).cpu().numpy() for k in records[0]}
                env.capture=False
                if not np.all(arrays['reference_advances']==1): raise RuntimeError('Reference advanced other than once per interval')
                if not np.allclose(arrays['applied_force_world'],arrays['force_start_world'],rtol=0,atol=1e-6):
                    raise RuntimeError('Reference input and actual simulator EE force differ')
                if np.any(arrays['base_force_world']) or np.any(arrays['base_moment_world']): raise RuntimeError('Forbidden torso load')
                for key,value in arrays.items():
                    if value.dtype.kind=='f' and not np.isfinite(value[arrays['valid_transition']]).all():
                        raise RuntimeError(f'Nonfinite recorded field: {key}')
                for i,trial in enumerate(batch):
                    mask=arrays['valid_transition'][:,i]
                    data={k:v[mask,i] for k,v in arrays.items()}
                    data['time']=(np.flatnonzero(mask)+1)*env.dt;data['interval_start']=np.flatnonzero(mask)*env.dt
                    data['phase']=util.phase(data['interval_start'],c)
                    summary=util.summarize(data,trial,c,env.dt)
                    folder=args.output_dir/f'seed_{seed}'/trial['condition'];folder.mkdir(parents=True,exist_ok=True)
                    np.savez_compressed(folder/f'trial_{trial["trial_id"]:04d}.npz',**data)
                    dump(folder/f'trial_{trial["trial_id"]:04d}.json',summary);episodes.append(summary)
                dump(args.output_dir/'episodes.json',episodes)
                print(f'seed={seed} batch={batch_index+1} completed={len(episodes)}/{sum(map(len,all_manifests.values()))} elapsed={time.perf_counter()-started:.1f}s failures={sum(x["failed"] for x in episodes)}',flush=True)
        expected=sum(map(len,all_manifests.values()))
        if len(episodes)!=expected: raise RuntimeError('Episode accounting mismatch')
        if model_hash(actor)!=before: raise RuntimeError('Policy weights changed')
        metadata.update(policy_hash_after=model_hash(actor),elapsed_seconds=time.perf_counter()-started,
            reference_advances_verified=True,applied_force_alignment_verified=True,completed_episodes=len(episodes))
        dump(args.output_dir/'metadata.json',metadata)
        dump(args.output_dir/'summary.json',util.aggregate([e for e in episodes if e['condition'] in args.conditions]))
        if calibration_conditions: dump(args.output_dir/'calibration_summary.json',util.aggregate([e for e in episodes if e['condition'] in calibration_conditions]))
        dump(args.output_dir/'paired_contrasts.json',util.paired_contrasts(episodes,c['bootstrap_samples']))
        with (args.output_dir/'episodes.csv').open('w') as f:
            keys=[k for k,v in episodes[0].items() if not isinstance(v,(dict,list))]
            writer=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');writer.writeheader();writer.writerows(episodes)
        (args.output_dir/'COMPLETE').write_text('All scheduled trials counted; inspect setup/reference/policy outcomes.\n')
    except BaseException:
        import traceback
        (args.output_dir/'ERROR.txt').write_text(traceback.format_exc())
        traceback.print_exc();sys.stderr.flush();raise
    finally:
        if env is not None: env.simulator._app_launcher.app.close()


if __name__=='__main__': main()
