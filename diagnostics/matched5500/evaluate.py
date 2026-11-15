#!/usr/bin/env python3
"""Evaluation-only checkpoint-5500 comparison. No training or source mutations."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/home/oyoungquist/Research/Genesis_Development/HCR_HardPact')
RUN = ROOT / 'logs/hardpact_iclr/go2_hard_pact_rss_full/Nov13_18-50-32_hard_pact_rss_full_isaaclab'
CONDITIONS = ('off', 'saved_held', 'every_held', 'every_fresh_nominal', 'every_fresh_previous')


def historical_qp(env, train_cfg, bundle):
    """Load the archived solver/assembly in an isolated evaluation namespace."""
    import dataclasses
    import importlib
    import types
    manifest = json.loads((bundle/'resolved_deployment.json').read_text())
    hashes = json.loads((bundle/'SHA256.json').read_text())
    archive = bundle/'archive/rsl_rl/algorithms'
    for path in archive.glob('*.py'):
        if digest(path) != hashes[str(path.relative_to(bundle))]:
            raise RuntimeError('Historical QP source checksum mismatch: '+str(path))
    package = types.ModuleType('_historical_oct07')
    package.__path__ = [str(archive),str(ROOT/'rsl_rl/algorithms')]
    sys.modules[package.__name__] = package
    module = importlib.import_module('_historical_oct07.hard_pact_qp')
    # Current environment's explicit wrench gate; this is not a change to the
    # historical QP math. Its historical path always supplied estimated wrench.
    config_type = dataclasses.make_dataclass('HistoricalExecutionConfig',
        [('use_external_wrench',bool,dataclasses.field(default=True))],
        bases=(module.HardPACTQPConfig,),frozen=True)
    config = config_type(**manifest['qp_config'])
    config = dataclasses.replace(config,enabled=True,warmup_iterations=0,
        qp_update_mode='every_substep',correction_ramp_enabled=False,
        objective_curriculum_enabled=False,torso_stability_curriculum_enabled=False)
    s=env.simulator
    qp=module.HardPACTDifferentiableQP(config,s.torque_limits,
        s.dof_pos_limits_hard[:,0],s.dof_pos_limits_hard[:,1],s.dof_vel_limits)
    print('HISTORICAL_QP',manifest['historical_source_commit'],str(archive),flush=True)
    return qp,manifest


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    def clean(x):
        import math
        if isinstance(x, dict): return {str(k): clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)): return [clean(v) for v in x]
        if hasattr(x, 'tolist'): return clean(x.tolist())
        if isinstance(x, float) and not math.isfinite(x): return None
        return x
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(clean(value), indent=2, allow_nan=False, default=str))
    tmp.replace(path)


def configure(env, runner, train_cfg, saved_config, checkpoint):
    import copy
    import torch
    from dataclasses import asdict, replace
    from legged_gym.scripts.hard_pact_play_qp import configure_play_qp
    from rsl_rl.algorithms.hard_pact_qp import HardPACTQPConfig
    from rsl_rl.algorithms.hard_pact_qp_curriculum import QPCurriculum
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if ARGS.historical_qp_bundle:
        archived=json.loads((ARGS.historical_qp_bundle/'resolved_deployment.json').read_text())
        if digest(checkpoint)!=archived['checkpoint_sha256']:
            raise RuntimeError('Historical bundle does not match requested checkpoint')
        if state['hard_pact_qp_curriculum']['progress']!=1.:
            raise RuntimeError('Historical evaluation requires resolved final objective weights')
        alpha=1.
        effective=HardPACTQPConfig.from_dict(archived['qp_config'])
    else:
        schedule = QPCurriculum(HardPACTQPConfig.from_dict(train_cfg.algorithm.hard_pact_qp))
        schedule.load_state_dict(state['training_state']['qp_curriculum'])
        effective, alpha = schedule.begin(state['training_state']['next_iteration'])
    train_cfg.algorithm.hard_pact_qp = asdict(effective)
    qp = configure_play_qp(env, runner, train_cfg, True)
    historical_manifest=None
    if ARGS.historical_qp_bundle:
        qp,historical_manifest=historical_qp(env,train_cfg,ARGS.historical_qp_bundle)
        env.configure_hard_pact_substep_qp(runner.alg.actor_critic,env._hard_pact_bard_dynamics,qp)
    mode = effective.qp_update_mode if ARGS.worker == 'saved_held' else 'every_substep'
    qp.cfg = replace(qp.cfg, qp_update_mode=mode)
    env.set_hard_pact_qp_enabled(ARGS.worker != 'off')
    runner.alg.actor_critic.eval().requires_grad_(False)
    from legged_gym.utils.helpers import class_to_dict
    from scripts.eval_hard_pact_frozen import dependency_versions, tensor_hash
    manifest = dict(schema=1, condition=ARGS.worker, seed=ARGS.seed, steps=ARGS.steps,
        checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint),
        source_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        source_diff=subprocess.check_output(['git','diff'],cwd=ROOT,text=True),
        evaluation_sha256=digest(__file__), dependencies=dependency_versions(),
        saved_config=saved_config, environment=class_to_dict(env.cfg),
        training=class_to_dict(train_cfg), effective_qp=asdict(qp.cfg),
        saved_alpha=alpha, execution_alpha=1., physical_gpu=ARGS.gpu,
        historical_qp=historical_manifest,
        midpoint_values=env._evaluation_midpoints,
        initial_state_hash=tensor_hash({'q':env._canonical_configuration(),
            'v':env._canonical_velocity_world(), **env._canonical_randomized_parameters()}),
        notes=['Play 2x5 mixed terrain and fixed play command bounds retained from prior experiment; not training terrain distribution.',
            'Common absolute-time command/wrench/push tape, independent QP RNG. Physical trajectories and reset/property sampling diverge after failures.',
            'Control 20ms, physics 5ms; contact truth is processed simulator contact, not actor input.',
            'Full candidate predictions differ from executed-torque predictions. Neither is measured motion.',
            'Timeouts right-censor episode survival; physical failure takes precedence for diagnostic categorization.',
            'No claim of exact PhysX process reproducibility or deployment latency.'])
    obs = Observer(env, runner.alg.actor_critic, qp, manifest)
    return obs


class Observer:
    def __init__(self, env, actor, qp, manifest):
        import torch
        from scripts.eval_hard_pact_frozen import make_scenario, tensor_hash
        self.e, self.s, self.actor, self.qp = env, env.simulator, actor, qp
        self.manifest = manifest
        self.t = -1
        self.k = 0
        self.out = ARGS.output / f'{ARGS.worker}_seed{ARGS.seed}'
        self.out.mkdir(parents=True, exist_ok=True)
        self.arrays = {}
        self.controls = {}
        self.calls = 0
        self.refreshes = 0
        self.save_seconds = 0.
        self.previous_ff = None
        self.solve_data = None
        self.result = None
        self.before_hash = tensor_hash(actor.state_dict())
        self.scenario = make_scenario(env, ARGS.steps, ARGS.seed)
        manifest['scenario_sha256'] = tensor_hash(self.scenario)
        write_json(self.out/'manifest.json', manifest)
        # Stream isolation: existing QP sampler unchanged, but its RNG cannot
        # alter environment draws. This hook is process-local and restored on exit.
        import legged_gym.envs.go2.go2_hard_pact.go2_hard_pact as module
        original_select = module.qp_interval_selection
        def select(*args, **kwargs):
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                torch.manual_seed(ARGS.seed*1000003 + 40001 + self.t)
                return original_select(*args, **kwargs)
        module.qp_interval_selection = select
        def wrench(_):
            t = max(self.t, 0)
            env._current_sustained_wrench_world.copy_(self.scenario['wrench_world'][t])
            env._current_sustained_active_mask.copy_(self.scenario['wrench_active'][t])
        env._update_persistent_wrench = wrench
        def push():
            delta = self.scenario['push_delta_world'][max(self.t,0)]
            ids = delta.any(-1).nonzero().flatten()
            self.s._rand_push_vels.copy_(delta[:,:3])
            self.s._rand_wrench_vels.copy_(delta[:,3:])
            if ids.numel():
                self.s._robot.write_root_link_velocity_to_sim(
                    self.s._robot.data.root_link_vel_w[ids,:6]+delta[ids],env_ids=ids)
        self.s.push_robots = push
        env._resample_commands = lambda ids: env.commands.__setitem__(ids,
            self.scenario['commands'][max(self.t+1,0),ids])
        callback = env._post_physics_step_callback
        def callback_wrapped():
            callback()
            env.commands.copy_(self.scenario['commands'][self.t+1])
        env._post_physics_step_callback = callback_wrapped
        compute = env.compute_observations
        def compute_wrapped():
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                torch.manual_seed(ARGS.seed*1000003+90001+self.t)
                compute()
        env.compute_observations = compute_wrapped
        self.pre_original = self.s._hard_pact_pre_physics_substep
        self.post_original = self.s._hard_pact_grf_post_physics_substep
        self.s._hard_pact_pre_physics_substep = self.pre
        self.s._hard_pact_grf_post_physics_substep = self.post
        original_solve = qp.solve
        def solve(**data):
            result = original_solve(**data)
            self.solve_data, self.result = data, result
            self.calls += 1
            return result
        qp.solve = solve
        original_substep = env._solve_hard_pact_rollout_qp_substep
        def substep(quat, mass_com_wrench):
            from rsl_rl.modules.hard_pact_control import requested_torque_components
            q,v = env._canonical_joint_state()
            p = env._hard_pact_control_parameters
            pd = requested_torque_components(env._hard_pact_q_d,
                torch.zeros_like(env._hard_pact_tau_ff),q,v,p)[1]
            if self.k and ARGS.worker.startswith('every_fresh'):
                ff = (self.previous_ff if ARGS.worker == 'every_fresh_previous' else
                    requested_torque_components(env._hard_pact_q_d,env._hard_pact_tau_ff,q,v,p)[2])
                lim = p['control_torque_limits']
                condition = (pd+ff).clamp(-lim,lim)
                heads = actor.physics_estimator
                env._qp_control_grf = heads.grf_to_physical(heads.predict_grf(
                    env._hard_pact_policy_latent,env._hard_pact_policy_explicit,condition)).reshape(-1,4,3).detach()
                for name,value in (('q',q),('v',v),('torque',condition)):
                    getattr(env,'_qp_grf_conditioning_'+name).copy_(value)
                self.refreshes += 1
            original_substep(quat,mass_com_wrench)
            self.previous_ff = (self.s.hard_pact_executed_torque()-pd).clone()
        env._solve_hard_pact_rollout_qp_substep = substep
        terminate = env.check_termination
        def termination():
            terminate()
            if self.t < 0: return
            e,s = self.e,self.s
            angle = (s._base_euler[:,:2]+torch.pi)%(2*torch.pi)-torch.pi
            height = (s.base_pos[:,2:3]-s.measured_heights).mean(-1)
            reasons = torch.stack((
                (s.link_contact_forces[:,s.termination_contact_indices].norm(dim=-1)>10).any(-1),
                angle[:,0].abs()>e.cfg.termination.roll_threshold,
                angle[:,1].abs()>e.cfg.termination.pitch_threshold,
                height<e.cfg.termination.height_min,height>e.cfg.termination.height_max,
                e.time_out_buf.bool(),e.non_failure_reset_buf.bool()),-1)
            self.record('reasons',reasons,control=True)
            self.record('reset',e.reset_buf.bool(),control=True)
            self.record('failure',e.fail_buf > e.cfg.env.fail_to_terminal_time_s/e.dt,control=True)
            self.record('episode_steps',e.episode_length_buf,control=True)
            self.record('terminal_velocity',torch.cat((s.base_lin_vel[:,:2],s.base_ang_vel[:,2:3]),-1),control=True)
        env.check_termination = termination
        env.commands.copy_(self.scenario['commands'][0])
        with torch.inference_mode(): env.compute_observations()

    def record(self,name,value,control=False):
        import torch
        if self.t < 0: return
        store = self.controls if control else self.arrays
        i = self.t if control else self.t*4+self.k
        if name not in store:
            store[name] = torch.zeros((ARGS.steps if control else ARGS.steps*4,)+tuple(value.shape),
                dtype=value.dtype,device=value.device)
        store[name][i].copy_(value.detach())

    def body_velocity(self):
        import torch
        q = self.s.hard_pact_base_quat_xyzw()
        world = self.s._robot.data.root_link_vel_w
        def inverse(v):
            t = 2*torch.cross(q[:,:3],v,dim=-1)
            return v-q[:,3:]*t+torch.cross(q[:,:3],t,dim=-1)
        return torch.cat((inverse(world[:,:3]),inverse(world[:,3:6])),-1)

    def pre(self):
        import torch
        self.solve_data = self.result = None
        self.velocity = self.body_velocity().clone()
        self.record('velocity_before',self.velocity[:,[0,1,5]])
        self.record('command',self.command)
        self.record('orientation_xyzw',self.s.hard_pact_base_quat_xyzw())
        self.nominal = self.e._hard_pact_bounded_nominal_torque.clone()
        self.record('nominal_torque',self.nominal)
        self.pre_original()
        executed = self.s.hard_pact_executed_torque().clone()
        self.record('executed_torque',executed)
        self.record('correction',executed-self.nominal)
        self.record('torque_violation',(executed.abs()-self.s.torque_limits).clamp_min(0))
        n = self.e.num_envs
        stage = torch.full((n,),-1,dtype=torch.int16,device=self.e.device)
        raw = executed.new_full((n,3),float('nan'))
        applied = raw.clone(); nominal = raw.clone(); torque_delta = raw.clone()
        residual = executed.new_full((n,),float('nan'))
        position_violation = executed.new_full((n,12),float('nan'))
        velocity_violation = position_violation.clone()
        if self.result is not None:
            d,r = self.solve_data,self.result
            ids = d['environment_ids']; stage[ids] = r.stage.to(stage.dtype)
            accepted = r.differentiated_mask | r.recovery_mask
            dt = float(self.e.cfg.sim.dt)
            # One solve for diagnostic torque changes, reusing captured mechanics.
            rhs = executed.new_zeros(ids.numel(),18,2)
            rhs[:,6:,0] = executed[ids]-r.tau_safe
            rhs[:,6:,1] = r.tau_safe-d['tau_nom']
            delta,info = torch.linalg.solve_ex(d['mass_matrix'],rhs,check_errors=False)
            delta = torch.where((info==0)[:,None,None],delta,float('nan'))
            raw[ids] = torch.where(accepted[:,None],dt*r.qdd[:,[0,1,5]],float('nan'))
            applied[ids] = torch.where(accepted[:,None],dt*(r.qdd+delta[:,:,0])[:,[0,1,5]],float('nan'))
            torque_delta[ids] = torch.where(accepted[:,None],dt*delta[:,[0,1,5],1],float('nan'))
            # Nominal reference forces, gated only by the same stance mask.
            f = d['force_pred_world'].reshape(-1,4,3)
            f = torch.where((d['contact_probability']>=self.qp.cfg.contact_threshold)[...,None],f,0).flatten(1)
            generalized = torch.einsum('bij,bi->bj',d['foot_jacobians'].reshape(-1,12,18),f)
            generalized += torch.einsum('bij,bi->bj',d['base_jacobian'],d['wrench_pred_world'])-d['bias']
            generalized[:,6:] += d['tau_nom']
            a_nom,info = torch.linalg.solve_ex(d['mass_matrix'],generalized.unsqueeze(-1),check_errors=False)
            a_nom = torch.where((info==0)[:,None],a_nom.squeeze(-1),float('nan'))
            nominal[ids] = dt*a_nom[:,[0,1,5]]
            residual[ids] = r.diagnostics['selected/inequality_max']
            qnext = d['joint_position']+dt*d['joint_velocity']+self.qp.cfg.position_integration_coefficient*dt*dt*r.qdd[:,6:]
            vnext = d['joint_velocity']+dt*r.qdd[:,6:]
            plim = self.s.dof_pos_limits_hard
            position_violation[ids] = torch.where(accepted[:,None],
                (plim[:,0]-qnext).clamp_min(0)+(qnext-plim[:,1]).clamp_min(0),float('nan'))
            velocity_violation[ids] = torch.where(accepted[:,None],
                (vnext.abs()-self.s.dof_vel_limits).clamp_min(0),float('nan'))
        for name,value in [('stage',stage),('candidate_delta',raw),('executed_model_delta',applied),
            ('nominal_model_delta',nominal),('torque_correction_delta',torque_delta),('constraint_residual',residual)]:
            self.record(name,value)
        self.record('candidate_joint_position_violation',position_violation)
        self.record('candidate_joint_velocity_violation',velocity_violation)
        self.record('contact_probability',self.actor.cenet_torso_velo[:,3:7])

    def post(self):
        self.post_original()
        after = self.body_velocity()
        self.record('realized_delta',(after-self.velocity)[:,[0,1,5]])
        self.record('contact_truth',self.e.grf_processor.contacts)
        q,v = self.e._canonical_joint_state()
        self.record('joint_position_violation',
            (self.s.dof_pos_limits_hard[:,0]-q).clamp_min(0)+(q-self.s.dof_pos_limits_hard[:,1]).clamp_min(0))
        self.record('joint_velocity_violation',(v.abs()-self.s.dof_vel_limits).clamp_min(0))
        self.k += 1

    def save(self,count,complete=False):
        import numpy as np
        import torch
        from scripts.eval_hard_pact_frozen import tensor_hash
        save_start = time.monotonic()
        output = {k:v[:count*4].cpu().numpy() for k,v in self.arrays.items()}
        output.update({'control_'+k:v[:count].cpu().numpy() for k,v in self.controls.items()})
        tmp = self.out/'trace.tmp.npz'
        np.savez_compressed(tmp,**output)
        tmp.replace(self.out/'trace.npz')
        self.save_seconds += time.monotonic()-save_start
        gpu_status = subprocess.run(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,memory.total',
            '--format=csv,noheader'],capture_output=True,text=True).stdout
        write_json(self.out/'progress.json',dict(completed_steps=count,complete=complete,
            qp_batch_dispatches=self.calls,extra_grf_refreshes=self.refreshes,
            wall_s=time.monotonic()-self.start,seed=ARGS.seed,condition=ARGS.worker,
            weights_unchanged=tensor_hash(self.actor.state_dict())==self.before_hash,
            artifact_save_seconds=self.save_seconds,gpu_status=gpu_status,
            torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            torch_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            gpu_event_ms=self.start_event.elapsed_time(self.end_event) if complete and count>100 else None))
        if complete:
            from report import summarize
            write_json(self.out/'result.json',summarize(self.out))


def rollout(observer,env,policy,steps):
    import torch
    observer.start = time.monotonic()
    observer.start_event = torch.cuda.Event(enable_timing=True)
    observer.end_event = torch.cuda.Event(enable_timing=True)
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for t in range(steps):
            observer.t,observer.k = t,0
            torch.manual_seed(ARGS.seed*1000003+t)
            observer.command = env.commands[:,:3].clone()
            observer.record('command',observer.command,control=True)
            obs,history,_,_ = env.get_observations()
            action = policy(obs,history)
            obs,_,history,_,reward,done,infos,_ = env.step(action)
            observer.record('nonreset_error',torch.cat((env.simulator.base_lin_vel[:,:2],
                env.simulator.base_ang_vel[:,2:3]),-1)-observer.command,control=True)
            for name,value in [('action',action),('obs',obs),('history',history),('reward',reward)]:
                if not torch.isfinite(value).all(): raise RuntimeError('nonfinite '+name)
            if t == 99:
                observer.start_event.record()
                observer.manifest['warmup_wall_s'] = time.monotonic()-observer.start
                write_json(observer.out/'manifest.json',observer.manifest)
            if (t+1)%250 == 0:
                fields={'joint_armature':'_joint_armature','joint_friction':'_joint_friction',
                    'joint_damping':'_joint_damping','joint_stiffness':'_joint_stiffness',
                    'kp':'_kp_scale','kd':'_kd_scale','motor_strength':'_motor_strength'}
                for name,field in fields.items():
                    actual=getattr(env.simulator,field)
                    torch.testing.assert_close(actual,torch.full_like(actual,env._evaluation_midpoints[name]))
                observer.save(t+1)
                print('PROGRESS',ARGS.worker,ARGS.seed,t+1,steps,flush=True)
        if steps <= 100: observer.start_event.record()
        observer.end_event.record()
        torch.cuda.synchronize()
        observer.save(steps,True)
        print('COMPLETE',ARGS.worker,ARGS.seed,flush=True)


def worker():
    protocol=json.loads((ARGS.output/'protocol.json').read_text())
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    diff=subprocess.check_output(['git','diff'],cwd=ROOT)
    if revision!=protocol['production_sha'] or hashlib.sha256(diff).hexdigest()!=protocol['production_diff_sha256']:
        raise RuntimeError('Production checkout changed during sweep; refusing a mixed-code comparison')
    sys.path[:0] = [str(ROOT),str(Path(__file__).parent)]
    sys.modules['matched_eval'] = sys.modules[__name__]
    from legged_gym.scripts.train_hard_pact import prepare_solver_runtime
    prepare_solver_runtime(['--task','go2_hard_pact_isaaclab'])
    path = ROOT/'legged_gym/scripts/play_go2_hard_pact_pos.py'
    source = path.read_text()
    def replace(old,new):
        nonlocal source
        assert source.count(old)==1,old
        source = source.replace(old,new)
    replace('    env_cfg.env.num_envs =', '''    import json
    from scripts.eval_hard_pact_frozen import apply_config
    saved_config = json.loads((__import__('pathlib').Path(args.load_run)/'hard_pact_resolved_config.json').read_text())
    apply_config(env_cfg,saved_config['environment']['resolved'])
    apply_config(train_cfg,saved_config['training']['resolved'])
    env_cfg.domain_rand.persistent_wrench_at_com = saved_config['environment']['resolved']['domain_rand'].get('persistent_wrench_at_com',False)
    midpoint = {}
    d = env_cfg.domain_rand
    for name in ('joint_friction','joint_damping','joint_stiffness','joint_armature','kp','kd','motor_strength'):
        suffix = '_range_end' if name in ('joint_friction','joint_damping','joint_stiffness') else '_range'
        mid = sum(getattr(d,name+suffix))/2
        if name in ('kp','kd','motor_strength'): mid = 1.
        midpoint[name] = mid
        for suffix in ('_range','_range_start','_range_end'):
            if hasattr(d,name+suffix): setattr(d,name+suffix,[mid,mid])
    env_cfg.env.num_envs =''')
    replace("    play_qp = configure_play_qp(env, runner, train_cfg, getattr(args, 'play_qp', None))",'''    from matched_eval import configure, rollout, ARGS
    env._evaluation_midpoints = midpoint
    with torch.inference_mode():
        observer = configure(env,runner,train_cfg,saved_config,checkpoint)''')
    start = source.index('    # Roughly ten episodes;')
    end = source.index('\n\nif __name__',start)
    source = source[:start]+'''    rollout(observer,env,policy,ARGS.steps)
'''+source[end:]
    sys.argv = [str(path),'--task',('go2_hard_pact_isaaclab' if ARGS.historical_qp_bundle else 'go2_hard_pact_rss_full_isaaclab'),'--load_run',str(RUN),
        '--ckpt',str(ARGS.checkpoint),'--gpu','cuda:0','--num_envs','10','--seed',str(ARGS.seed),'--headless']
    exec(compile(source,str(path),'exec'),{'__name__':'__main__','__file__':str(path)})


def main():
    global ARGS, RUN
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--gpu',type=int,default=0)
    p.add_argument('--steps',type=int,default=3000)
    p.add_argument('--seeds',nargs='+',type=int,default=[1,2,3])
    p.add_argument('--conditions',nargs='+',choices=CONDITIONS,default=list(CONDITIONS))
    p.add_argument('--worker',choices=CONDITIONS)
    p.add_argument('--seed',type=int,default=1)
    p.add_argument('--run',type=Path,default=RUN)
    p.add_argument('--checkpoint',type=int,default=5500)
    p.add_argument('--historical-qp-bundle',type=Path)
    ARGS = p.parse_args()
    RUN=ARGS.run.resolve()
    if ARGS.historical_qp_bundle: ARGS.historical_qp_bundle=ARGS.historical_qp_bundle.resolve()
    ARGS.output = ARGS.output.resolve()
    if ARGS.worker: return worker()
    ARGS.output.mkdir(parents=True,exist_ok=True)
    import shutil
    archive=ARGS.output/'evaluation_source'
    archive.mkdir(exist_ok=True)
    for name in ('evaluate.py','report.py'):
        destination=archive/name
        if destination.exists() and digest(destination)!=digest(Path(__file__).with_name(name)):
            raise RuntimeError('Evaluation code changed: use a new output directory to avoid mixing protocols')
        shutil.copy2(Path(__file__).with_name(name),destination)
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    diff=subprocess.check_output(['git','diff'],cwd=ROOT)
    if (ARGS.output/'protocol.json').exists():
        old=json.loads((ARGS.output/'protocol.json').read_text())
        if old.get('production_sha',revision)!=revision or old.get('production_diff_sha256',hashlib.sha256(diff).hexdigest())!=hashlib.sha256(diff).hexdigest():
            raise RuntimeError('Production source changed since previous trials; choose a new output directory')
    write_json(ARGS.output/'protocol.json',dict(steps=ARGS.steps,seeds=ARGS.seeds,conditions=ARGS.conditions,
        gpu=ARGS.gpu,argv=sys.argv,checkpoint_sha256=digest(RUN/f'model_{ARGS.checkpoint}.pt'),
        run=str(RUN),checkpoint=ARGS.checkpoint,historical_qp_bundle=ARGS.historical_qp_bundle,
        parent_pid=os.getpid(),production_sha=revision,production_diff_sha256=hashlib.sha256(diff).hexdigest(),
        production_root=str(ROOT),restart_semantics='Completed trials skipped; interrupted trials restart from seed, not exact PhysX continuation.'))
    for seed in ARGS.seeds:
        for condition in ARGS.conditions:
            dest = ARGS.output/f'{condition}_seed{seed}'
            if (dest/'result.json').exists():
                progress = json.loads((dest/'progress.json').read_text())
                if progress['complete'] and progress['completed_steps']==ARGS.steps: continue
                raise RuntimeError('Existing run has incompatible step budget')
            dest.mkdir(exist_ok=True)
            if (dest/'run.log').exists():
                (dest/'run.log').rename(dest/f'run_previous_{time.time_ns()}.log')
            env = dict(os.environ,CUDA_VISIBLE_DEVICES=str(ARGS.gpu),SIMULATOR='isaaclab',
                PYTHONPATH=str(ROOT)+':'+str(Path(__file__).parent),OMP_NUM_THREADS='1',PYTHONUNBUFFERED='1',
                LD_PRELOAD='/home/oyoungquist/.conda/envs/lr_lab_cupiqp/lib/libstdc++.so.6')
            extra=['--run',str(RUN),'--checkpoint',str(ARGS.checkpoint)]
            if ARGS.historical_qp_bundle: extra+=['--historical-qp-bundle',str(ARGS.historical_qp_bundle)]
            with (dest/'run.log').open('w') as log:
                result = subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker',condition,
                    '--seed',str(seed),'--steps',str(ARGS.steps),'--gpu',str(ARGS.gpu),
                    '--output',str(ARGS.output),*extra],env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode or not (dest/'result.json').exists():
                raise RuntimeError(f'{condition}/{seed} failed; inspect {dest}/run.log')
            from report import report
            report(ARGS.output)


if __name__ == '__main__': main()
