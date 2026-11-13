"""Bounded REAL Isaac Lab validation, merged BR + shared rewards by default.

SIMULATOR=isaaclab PYTHONPATH=. python tests/smoke_beyond_robustness_payload.py --gpu cuda:0 --num-envs 4
No production launcher or exact author-reproduction claim. Prints explicit PASS/FAIL
because Kit shutdown can mask a Python exception in the process exit code.
"""
import argparse
import json
import time
import traceback
import tempfile
import importlib.metadata
from pathlib import Path
import torch
from legged_gym.envs.go2.go2_beyond_robustness import IsaacLabBeyondRobustnessEnv
from rsl_rl.beyond_robustness_config import resolve_beyond_robustness
from rsl_rl.beyond_robustness_payload import cuboid_inertia
from rsl_rl.runners.beyond_robustness_runner import BeyondRobustnessRunner

parser=argparse.ArgumentParser()
parser.add_argument('--gpu',default='cuda:0'); parser.add_argument('--num-envs',type=int,default=4)
parser.add_argument('--steps',type=int,default=8)
parser.add_argument('--reward-mode',choices=('merged','smoke_tracking'),default='merged')
parser.add_argument('--physical-checks',action='store_true')
parser.add_argument('--production-rollout',action='store_true',help='Use live terrain layout, rollout length, PPO/aux epochs and minibatches')
parser.add_argument('--output-dir',default='/tmp/br_stage4')
args=parser.parse_args()
overrides={'env.num_envs':args.num_envs,'sim.console_debug':False,'viewer.debug_vis':False}
if not args.production_rollout:
    overrides.update({'terrain.num_rows':2,'terrain.num_cols':2,'terrain.max_init_terrain_level':0})
training={} if args.production_rollout else dict(num_steps_per_env=args.steps,
    num_learning_epochs=1,num_mini_batches=1,auxiliary_epochs=1)
resolved=resolve_beyond_robustness(environment_overrides=overrides,training_overrides=training)
cfg=resolved.environment
if args.physical_checks:
    # Flat level-zero trimesh and nominal robot properties isolate the physical
    # checks. Production adapter leaves the supplied terrain/DR config intact.
    overrides.update({'terrain.terrain_proportions':[1.]+[0.]*10,
        'terrain.curriculum':True,'terrain.num_rows':1,
        'domain_rand.push_robots':False,'domain_rand.persistent_disturbance':False})
    for name in dir(cfg.domain_rand):
        if name.startswith('randomize_'): overrides['domain_rand.'+name]=False
    resolved=resolve_beyond_robustness(environment_overrides=overrides,training_overrides=training)
    cfg=resolved.environment
# Isolate contact tests from randomized pushes; restore flags for runner smoke.
env=None
try:
    env=IsaacLabBeyondRobustnessEnv(resolved,args.gpu,True,reward_mode=args.reward_mode)
    packet=env.reset(); packet.validate(env.input_schema,args.num_envs,args.gpu)
    sim=env.simulator
    # Real simulator property readbacks, reset labels, and noise isolation.
    torch.testing.assert_close(sim.payload.root_physx_view.get_inertias().to(args.gpu).reshape(-1,3,3),
        cuboid_inertia(sim.payload_mass,sim.box_size),rtol=1e-5,atol=1e-7)
    torch.testing.assert_close(packet.load_labels[:,3:6],sim.payload_relative_velocity,rtol=1e-4,atol=2e-6)
    torch.testing.assert_close(packet.state[:,:45],env._clean_proprioception(),rtol=0,atol=0)
    rng=env.actor_observation_cache.generator.get_state().clone()
    torch.testing.assert_close(packet.observations,env.observe().observations,rtol=0,atol=0)
    assert torch.equal(rng,env.actor_observation_cache.generator.get_state())
    before=env.observe().clone(); root=sim._robot.data.root_link_state_w.clone()
    env._reset(torch.tensor([0],device=args.gpu))
    torch.testing.assert_close(before.observations[1:],env.observe().observations[1:],rtol=0,atol=0)
    torch.testing.assert_close(root[1:],sim._robot.data.root_link_state_w[1:],rtol=0,atol=0)
    assert (sim.dof_pos-sim.default_dof_pos).abs().max()>.01
    checks=dict(per_env_geometry_inertia=True,rotating_reset_labels=True,clean_privileged=True,
        repeated_observe_rng_pure=True,partial_reset_noise_isolation=True,randomized_joint_reset=True)
    if env.rewards is not None:
        owned=env._reward_snapshot(); old_position=owned.dof_pos.clone()
        env._reset(torch.tensor([0],device=args.gpu))
        torch.testing.assert_close(owned.dof_pos,old_position,rtol=0,atol=0)
        checks['reward_snapshot_owned']=True
    if args.physical_checks:
        from br_payload_physical_checks import physics_checks
        checks=physics_checks(env)
        print('BR_PHYSICS_CHECKS_PASS',json.dumps(checks),flush=True)
    for _ in range(args.steps):
        result=env.step(torch.zeros(args.num_envs,12,device=args.gpu))
        assert torch.isfinite(torch.cat(result.observation.inputs(),-1)).all()
        assert torch.isfinite(result.rewards).all()
    # Force one instantaneous push through the real adapter step, not an extra
    # physics step. Check owned terminal inputs and raw reward/gating evidence.
    original_push=sim.push_robots
    adapter_push_enabled=env.cfg.domain_rand.push_robots
    push_evidence={}
    def fixed_push():
        data=sim._robot.data
        velocity=torch.zeros(args.num_envs,6,device=args.gpu)
        velocity[:,0]=.2; velocity[:,1]=-.1; velocity[:,5]=.2
        expected_feet=sim.feet_vel+(velocity[:,:3]-data.root_link_lin_vel_w)[:,None]+torch.cross(
            (velocity[:,3:]-data.root_link_ang_vel_w)[:,None].expand(-1,4,-1),
            sim.feet_pos-data.root_link_pos_w[:,None],dim=-1)
        contact=sim.hard_pact_foot_forces_world()[:,:,2]>1.
        push_evidence['slip']=(contact*expected_feet[:,:,:2].sum(-1)).square().sum()
        sim._robot.write_root_link_velocity_to_sim(velocity)
    sim.push_robots=fixed_push; env.cfg.domain_rand.push_robots=True
    if env.rewards is not None: env.rewards.begin_iteration(0)
    result=env.step(torch.zeros(args.num_envs,12,device=args.gpu))
    terminal=result.terminal
    expected=torch.exp(-4*(terminal.state[:,6:8]-terminal.state[:,45:47]).square().sum(-1))
    torch.testing.assert_close(result.curriculum_tracking,expected)
    if env.rewards is not None:
        expected_reward=expected  # Fixed BR exp(-4*error_xy^2), not HardPACT dynamic sigma.
        torch.testing.assert_close(env.rewards.last_raw['tracking_lin_vel'],expected_reward)
        torch.testing.assert_close(env.rewards.last_raw['foot_slip'].sum(),push_evidence['slip'])
        assert env.rewards.age.ge(0).all()
    sim.push_robots=original_push; env.cfg.domain_rand.push_robots=adapter_push_enabled
    checks['push_timestamp_reward_observation_tracking_parity']=True
    # Complete contract comes from the factory; no hand-written schema override.
    runner=BeyondRobustnessRunner(env,resolved.training,log_dir=args.output_dir,device=args.gpu)
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); start=time.perf_counter()
    logs=runner.learn(1)
    runner.set_training_phase('student_reinforcement'); logs+=runner.learn(1)
    torch.cuda.synchronize(); elapsed=time.perf_counter()-start
    assert all(torch.isfinite(p).all() for p in runner.alg.actor_critic.parameters())
    assert all(torch.isfinite(p.grad).all() for p in runner.alg.actor_critic.parameters() if p.grad is not None)
    if env.rewards is not None:
        assert all('reward/total' in log and 'reward/original_br' in log for log in logs)
    with tempfile.TemporaryDirectory(prefix='br_stage3_') as directory:
        path=Path(directory)/'checkpoint.pt'
        runner.save(path)
        masses=sim.payload_mass.clone()
        phase=runner.alg.training_phase; iteration=runner.alg.iteration
        runner.load(path,resume_mode='fresh')
        assert runner.alg.iteration==iteration and runner.alg.training_phase==phase
        torch.testing.assert_close(sim.payload_mass,masses,rtol=0,atol=0)
        assert runner.alg.history.frames.eq(0).all()
        if env.rewards is not None:
            assert env.rewards.age.eq(0).all() and env.rewards.actions.eq(0).all()
            assert env.rewards.iteration==iteration-1
        snapshot=env.observe().clone(); root=sim._robot.data.root_link_state_w.clone()
        runner.load(path,resume_mode='fresh')
        torch.testing.assert_close(env.observe().observations,snapshot.observations,rtol=0,atol=0)
        torch.testing.assert_close(sim._robot.data.root_link_state_w,root,rtol=0,atol=0)
    print('BR_PAYLOAD_SMOKE_PASS',json.dumps(dict(num_envs=args.num_envs,seconds=elapsed,reward_mode=args.reward_mode,
        metadata=dict(isaaclab=importlib.metadata.version('isaaclab'),isaacsim=importlib.metadata.version('isaacsim'),
            torch=torch.__version__,gpu=torch.cuda.get_device_name(),cuda=torch.version.cuda),
        transitions_per_second=2*resolved.training.num_steps_per_env*args.num_envs/elapsed,
        torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        load_mass_range=[float(sim.payload_mass.min()),float(sim.payload_mass.max())],box_size_bank=sim.box_size_bank,
        noise_enabled=cfg.noise.add_noise,physics_dt=cfg.sim.dt,control_dt=cfg.control.dt,
        physical_checks=checks,schema=env.input_schema.contract(),logs=logs)),flush=True)
except BaseException:
    traceback.print_exc(); print('BR_PAYLOAD_SMOKE_FAIL',flush=True); raise
finally:
    if env is not None: env.close()
