"""Figure 1 physical-interaction milestone. Frozen policy, one dynamic robot."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import random
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,default=ROOT/'logs/b1z1/b1z1_pact_lab/Nov13_09-58-29_b1z1_pact_improved')
    p.add_argument('--checkpoint',type=int)
    p.add_argument('--scenario',choices=['nominal','support-follow','adjust-hold','all'],default='all')
    p.add_argument('--gpu',default='cuda:0');p.add_argument('--seed',type=int,default=0)
    p.add_argument('--trials',type=int,default=3);p.add_argument('--horizon',type=float,default=20.)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--config',type=Path)
    p.add_argument('--headless',action='store_true');p.add_argument('--video',action='store_true')
    p.add_argument('--video-fps',type=int,default=10)
    p.add_argument('--validate-only',action='store_true',help='Strict CPU checkpoint/layout check, no simulator')
    return p


def load_archive(run):
    path=run/'b1z1_pact_config.py'
    if not path.is_file(): raise FileNotFoundError(f'Archived configuration required: {path}')
    spec=importlib.util.spec_from_file_location('figure1_archived_config',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def clean(value):
    import numpy as np
    if isinstance(value,dict): return {k:clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [clean(v) for v in value]
    if isinstance(value,np.ndarray): return clean(value.tolist())
    if isinstance(value,np.generic): value=value.item()
    if isinstance(value,float) and not np.isfinite(value): return None
    return value


def main(argv=None):
    args=parser().parse_args(argv)
    if args.trials<1 or args.horizon<=0 or args.video_fps<1: raise ValueError('Positive trials/horizon/fps required')
    if args.output.exists() and any(args.output.iterdir()): raise FileExistsError('Use an empty output directory')
    os.environ['SIMULATOR']='isaaclab_b1z1_pact'
    import numpy as np
    import torch
    import legged_gym.envs
    from legged_gym.utils.helpers import class_to_dict
    from legged_gym.scripts.eval_b1z1_exp1 import build_actor,inference,model_hash,util
    from legged_gym.envs.b1z1.b1z1_pact.b1z1_pact import B1Z1PACT
    from legged_gym.envs.b1z1.directional_compliance import task_position,world_position,heading
    from legged_gym.utils.pilot_figure1 import CONFIG,PROFILES,checkpoint_path,summarize,audit_coupling
    from legged_gym.scripts.pilot_figure1_scene import InteractionScene,NoPropScene
    path,number=checkpoint_path(args.run,args.checkpoint)
    config=dict(CONFIG)
    if args.config:
        overrides=json.loads(args.config.read_text())
        if overrides.keys()-config.keys(): raise ValueError('Unknown scenario configuration keys')
        config.update(overrides)
    for key,value in config.items():
        if not isinstance(value,(int,float)) or not np.isfinite(value) or value<=0:
            raise ValueError(f'Invalid scenario value: {key}')
    if not config['attach_s']<config['lift_s']<config['release_s']<config['guide_s']<config['hold_s']:
        raise ValueError('Scenario event times must be strictly ordered')
    archived=load_archive(path.parent)
    cfg=util.materialize(archived.B1Z1PACTCfg());training=util.materialize(archived.B1Z1PACTCfgPPO())
    archived_env=class_to_dict(cfg);archived_train=class_to_dict(training)
    if not cfg.compliance.enabled or training.policy.conditioning_mode!='film' or not training.policy.film_v02_enabled:
        raise ValueError('This milestone requires the archived differential-compliance FiLM V02 PILOT architecture')
    cfg.env.num_envs=1;cfg.env.episode_length_s=max(args.horizon+2,20.)
    cfg.env.render_ee_goal_debug=False;cfg.env.render_ee_frame_debug=False
    cfg.terrain.mesh_type='plane';cfg.terrain.curriculum=False
    cfg.commands.curriculum=False;cfg.commands.heading_command=False
    cfg.commands.push_gripper_stators=True;cfg.commands.apply_ee_external_forces=True
    cfg.commands.push_robot_base=False;cfg.commands.apply_base_external_forces=False
    cfg.commands.apply_base_external_torques=False
    cfg.control.use_tradeoff_curriculum=False;cfg.control.randomize_pact_weights=False
    cfg.domain_rand.push_robots=False;cfg.domain_rand.use_domainrand_curriculum=False
    for name in vars(cfg.domain_rand):
        if name.startswith('randomize_'): setattr(cfg.domain_rand,name,False)
    # Same numerical geometry/gates; eager avoids compiling kernels for one robot.
    cfg.compliance.compile_geometry=False
    cfg.compliance.ee_profile_low=cfg.compliance.ee_profile_high=PROFILES['nominal'][:6]
    cfg.compliance.base_profile_low=cfg.compliance.base_profile_high=PROFILES['nominal'][6:]
    cfg.sim.evaluation_cameras=args.video
    created_sim=[]
    def scene_factory(sim):
        created_sim.append(sim)
        return (NoPropScene if args.scenario=='nominal' else InteractionScene)(sim,config,args.video)
    cfg.sim.evaluation_scene_factory=scene_factory
    torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
    checkpoint=torch.load(path,map_location='cpu',weights_only=False)
    args.output.mkdir(parents=True,exist_ok=True)
    print(f'Exact checkpoint: {path}\nFilename iteration: {number}; saved next iteration: {checkpoint.get("iteration")}\nSHA256: {util.file_hash(path)}',flush=True)
    from types import SimpleNamespace
    from legged_gym.envs.b1z1.compliance_task import configure as configure_compliance
    configure_compliance(cfg)
    contract_env=SimpleNamespace(device='cpu',num_obs=cfg.env.num_observations,
        num_privileged_obs=cfg.env.num_privileged_obs,num_crit_obs_stack=cfg.env.num_priv_stack,
        num_actions=cfg.env.num_actions,num_obs_hist=cfg.env.num_obs_hist,
        num_exp_labels=cfg.env.num_explicit_recon_obs)
    check_actor=build_actor(contract_env,training,checkpoint)
    check_hash=model_hash(check_actor)
    with torch.inference_mode():
        a=inference(check_actor,torch.zeros(1,contract_env.num_obs),
                    torch.zeros(1,contract_env.num_obs*contract_env.num_obs_hist))
    if a.shape!=(1,34) or not torch.isfinite(a).all(): raise ValueError('Invalid native policy output')
    if model_hash(check_actor)!=check_hash: raise ValueError('Inference mutated weights')
    preflight=dict(checkpoint=str(path),checkpoint_sha256=util.file_hash(path),iteration=number,
        observation_width=contract_env.num_obs,history_frames=contract_env.num_obs_hist,
        action_width=34,film_history_steps=check_actor.film_history_steps,
        strict_load=True,finite_cpu_inference=True,policy_hash=check_hash,
        control=archived_env['control'],normalization=archived_env['normalization'],policy=archived_train['policy'])
    (args.output/'checkpoint_validation.json').write_text(json.dumps(clean(preflight),indent=2,allow_nan=False))
    del check_actor
    if args.validate_only: return

    def snapshot(env):
        s=env.simulator;scene=s.evaluation_scene;target=env.compliance.target
        yaw=heading(s.base_quat)
        result=dict(base_world=s.base_pos,base_quat_xyzw=s.base_quat,base_velocity_world=s._base_world_lin_vel,
            ee_world=s.ee_pos,ee_velocity_world=s._ee_vel,ee_task=task_position(s.ee_pos,s.base_pos,yaw),
            nominal_world=env.curr_ee_goal_cart_world,nominal_commands=env.commands,
            q=s.dof_pos,qd=s.dof_vel,torques=s.torques,
            interval_torques=s.interval_executed_torque,
            profile=env.compliance.profile,robot_force=env.ee_force_ext_world,
            payload_state=(scene.payload.data.root_state_w if scene.payload is not None else s.base_pos.new_full((1,13),float('nan'))),partner_position=scene.partner_position,
            partner_velocity=scene.partner_velocity,partner_force=scene.partner_force,
            payload_handles=scene.handles,reference_world=world_position(target.position,s.base_pos,yaw),
            reference_task=target.position,reference_velocity_task=target.velocity,base_reference=target.base,
            gates=target.gates,workspace_gate=target.workspace,joint_gate=target.joint,collision_gate=target.collision,
            clipped=target.clipped,recovery_failed=env.compliance.recovery_failed,
            substep_interaction=(torch.stack(scene.substep_records,1) if scene.substep_records
                else s.base_pos.new_full((1,cfg.control.decimation,21),float('nan'))),
            failure_flags=torch.stack((env.contact_fail_buf,env.roll_fail_buf,env.pitch_fail_buf,
                env.height_min_fail_buf,env.height_max_fail_buf),-1))
        return {k:v[0].detach().cpu().numpy().copy() for k,v in result.items()}

    class ScenarioEnv(B1Z1PACT):
        demo_time=0.;recording=False;terminal_snapshot=None
        def _resample_commands(self,ids):
            self.commands[ids,:3]=0
        def _resample_ee_goal(self,ids,is_init=False):
            if not len(ids): return
            if is_init:
                super()._resample_ee_goal(ids,True)
                self.traj_total_timesteps[ids]=1e9
        def _push_gripper(self,ids):
            scene=self.simulator.evaluation_scene
            scene.begin_interval(self,self.demo_time)
            self.ee_force_ext_world.copy_(scene.robot_force)
            self.simulator.apply_ee_force(self.ee_force_ext_world)
            # Copied before physics/reset; these are the actual held inputs.
            self.interaction_start=dict(robot_force_start=scene.robot_force[0].cpu().numpy().copy(),
                robot_application_point_start=self.simulator.ee_pos[0].cpu().numpy().copy(),
                payload_handles_start=scene.handles[0].cpu().numpy().copy(),
                payload_force_start=(-scene.robot_force-scene.partner_force)[0].cpu().numpy().copy())
        def reset_idx(self,ids):
            if self.recording and len(ids): self.terminal_snapshot=snapshot(self)
            super().reset_idx(ids)
        def _post_physics_step_callback(self):
            self._step_contact_targets();self.update_curr_ee_goal()

    env=None;actor=None;results=[]
    try:
        env=ScenarioEnv(cfg,class_to_dict(cfg.sim),args.gpu,args.headless)
        actor=build_actor(env,training,checkpoint)
        before=model_hash(actor)
        scene=env.simulator.evaluation_scene
        resolved=class_to_dict(cfg);resolved['sim'].pop('evaluation_scene_factory',None)
        asset_path=Path(cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=str(ROOT)))
        metadata=dict(checkpoint=str(path),checkpoint_sha256=util.file_hash(path),filename_iteration=number,
            source_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            evaluator_sha256=util.file_hash(Path(__file__)),
            scene_sha256=util.file_hash(ROOT/'legged_gym/scripts/pilot_figure1_scene.py'),
            robot_asset=str(asset_path),robot_asset_sha256=util.file_hash(asset_path) if asset_path.is_file() else None,
            arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
            saved_next_iteration=checkpoint.get('iteration'),training_seed=archived_env.get('seed'),evaluation_seed=args.seed,
            archived_config_sha256=util.file_hash(path.parent/'b1z1_pact_config.py'),
            archived_environment=archived_env,archived_training=archived_train,resolved_environment=resolved,
            scenario_config=config,profiles=PROFILES,observation_width=env.num_obs,history_frames=env.num_obs_hist,
            policy_hash_before=before,control_dt=env.dt,physics_dt=cfg.control.dt/cfg.control.decimation,
            joint_order=cfg.asset.dof_names,
            conventions={'forces':'N, world, acting ON robot; box receives equal/opposite at its handle',
                'grasp':'force-only at configured gripper link origin/TCP; no orientation constraint',
                'timing':'force computed at interval start, held through substeps; successor logged before reset',
                'moments':'Nm; payload lever moments recomputed at each substep about its box origin/CoM',
                'payload_state':'world position xyz [m], quaternion wxyz, world linear velocity [m/s], world angular velocity [rad/s]',
                'robot_state':'base quaternion xyzw; q rad, qd rad/s; measured world positions m and velocities m/s',
                'actions':'native coupled actor output, position then feedforward channels; scaling and joint mapping in archived/resolved configuration',
                'torques':'Nm; torques last executed substep, interval_torques executed control-interval average',
                'substep_interaction':'[TCP world(3), payload handles world(6), payload force world(3), payload moment world(3), robot force world(3), partner force world(3)]; NaN before attachment',
                'references':'mixed task x/y heading-relative to base, z world; one advance per policy interval'},
            limitations=['Archived Python config inherits current ComplianceCfg defaults; old checkpoint has no resolved full contract.',
                'Corrected Z1 collisions/inertia may differ from training; strict weight loading alone cannot validate dynamics.',
                'Explicit springs evaluated at control rate, held per substep; robot/payload collision filtered to prevent duplicate forces.'])
        (args.output/'metadata.json').write_text(json.dumps(clean(metadata),indent=2,allow_nan=False))
        scenarios=['nominal','support-follow','adjust-hold'] if args.scenario=='all' else [args.scenario]
        with torch.inference_mode():
            for scenario in scenarios:
                for trial in range(args.trials):
                    seed=args.seed+trial;torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
                    profile=PROFILES[scenario]
                    # Existing reset sampler, exact fixed bounds; no profile-ID input.
                    env.compliance.reference.sampler.low[:]=torch.tensor(profile,device=env.device)
                    env.compliance.reference.sampler.high[:]=torch.tensor(profile,device=env.device)
                    env.recording=False;env.demo_time=0;env.terminal_snapshot=None
                    scene.enabled=False;env.reset();actor.reset()
                    scene.enabled=scenario!='nominal';scene.scenario=scenario
                    env.recording=True
                    obs,hist,_,_=env.get_observations();rows=[];failure=None
                    output=args.output/f'{scenario}_trial{trial:02d}';output.mkdir()
                    video=None
                    if args.video:
                        import imageio.v2 as imageio
                        video=imageio.get_writer(str(output/'rollout.mp4'),fps=args.video_fps)
                    try:
                        for step in range(int(np.ceil(args.horizon/env.dt))):
                            env.demo_time=step*env.dt
                            action=inference(actor,obs,hist)
                            if not torch.isfinite(action).all(): failure='nonfinite_action';break
                            nominal_start=env.curr_ee_goal_cart_world.clone()
                            observation_start=obs[0].clone()
                            obs,_,hist,_,_,dones,info,_=env.step(action)
                            row=env.terminal_snapshot if dones.any() else snapshot(env)
                            row.update(time=(step+1)*env.dt,interval_start=step*env.dt,
                                raw_action=action[0].cpu().numpy(),observation_start=observation_start.cpu().numpy(),
                                nominal_start_world=nominal_start[0].cpu().numpy(),**env.interaction_start)
                            rows.append(row)
                            if args.video and step%max(1,round(1/(env.dt*args.video_fps)))==0 and not dones.any():
                                counter=env.common_step_counter
                                film_history=actor._film_history.clone() if actor._film_history is not None else None
                                image=scene.image(env);video.append_data(image)
                                assert env.common_step_counter==counter
                                if film_history is not None: assert torch.equal(actor._film_history,film_history)
                                if step==0 or step%max(1,round(2/env.dt))==0:
                                    imageio.imwrite(output/f'frame_{step:05d}.png',image)
                            finite=all(np.isfinite(row[k]).all() for k in ('q','qd','ee_world','base_world','robot_force'))
                            if scenario!='nominal':
                                finite=finite and all(np.isfinite(row[k]).all() for k in
                                    ('payload_state','partner_position','partner_force'))
                            if dones.any() or not finite:
                                reasons=[name for name,flag in zip(('torso_contact','roll','pitch','height_min','height_max'),row['failure_flags']) if flag]
                                failure=('nonfinite_state' if not finite else 'compliance_recovery' if bool(row['recovery_failed']) else '+'.join(reasons) or 'environment_termination')
                                break
                    finally:
                        if video is not None: video.close()
                        env.recording=False
                    data={k:np.asarray([r[k] for r in rows]) for k in rows[0]} if rows else {}
                    np.savez_compressed(output/'trajectory.npz',**data)
                    summary=summarize(data,scenario,config) if rows else {}
                    if rows: summary['coupling_audit']=audit_coupling(data)
                    summary.update(scenario=scenario,trial=trial,seed=seed,failure=failure,
                        horizon_complete=failure is None and len(rows)>=int(np.ceil(args.horizon/env.dt)),
                        policy_unchanged=model_hash(actor)==before)
                    (output/'summary.json').write_text(json.dumps(clean(summary),indent=2,allow_nan=False))
                    print(json.dumps(clean(summary)),flush=True);results.append(summary)
        metadata['policy_hash_after']=model_hash(actor)
        if metadata['policy_hash_after']!=before: raise RuntimeError('Frozen policy weights changed')
        (args.output/'metadata.json').write_text(json.dumps(clean(metadata),indent=2,allow_nan=False))
        (args.output/'summary.json').write_text(json.dumps(clean(results),indent=2,allow_nan=False))
        (args.output/'COMPLETE').write_text('Evaluation finished; inspect failures and behavior checks.\n')
    except BaseException:
        import traceback
        error=traceback.format_exc()
        print(error,file=sys.stderr,flush=True)
        (args.output/'ERROR.txt').write_text(error)
        raise
    finally:
        if created_sim: created_sim[0]._app_launcher.app.close()


if __name__=='__main__': main()
