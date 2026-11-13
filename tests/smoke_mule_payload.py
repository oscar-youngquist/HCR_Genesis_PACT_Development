"""Bounded actual Isaac Lab/MULERunner test. NOT a production reward or launcher."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import time
import traceback
from copy import deepcopy
import torch
from legged_gym.envs.go2.go2_mule.go2_mule_config import resolve_go2_mule
from legged_gym.envs.go2.go2_mule.go2_mule_isaaclab import IsaacLabMULEEnv
from legged_gym.envs.go2.go2_mule.rolling_payload import sphere_inertia
from rsl_rl.runners.mule_runner import MULERunner


def test_only_mule_tracking_reward(env,packet,failure,timeout):
    # Deliberately NOT the paper's two production objectives. Separate streams
    # exercise the real dual learner without claiming reward reproduction.
    error=(packet.velocity_b[:,:2]-env.commands[:,:2]).square().sum(-1)
    nominal=env.dt*torch.nan_to_num(torch.exp(-error)-failure.float())
    adaptive=env.dt*torch.nan_to_num(.5*torch.exp(-2*error)-failure.float())
    return nominal,adaptive


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--gpu',default='cuda:0'); p.add_argument('--num-envs',type=int,default=4)
    p.add_argument('--steps',type=int,default=4); p.add_argument('--mode',choices=('rolling','nominal'),default='rolling')
    p.add_argument('--physical-checks',action='store_true'); p.add_argument('--output-dir',default='/tmp/mule_payload')
    args=p.parse_args(); out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    overrides={'env.num_envs':args.num_envs,'terrain.num_rows':1,'terrain.num_cols':2,
        'terrain.max_init_terrain_level':0,'terrain.terrain_proportions':[1.]+[0.]*10,
        'sim.console_debug':False,'viewer.debug_vis':False}
    if args.physical_checks:
        probe=resolve_go2_mule()
        overrides['domain_rand.push_robots']=False
        for key in vars(probe.shared.domain_rand):
            if key.startswith('randomize_'): overrides['domain_rand.'+key]=False
    resolved=resolve_go2_mule(environment_overrides=overrides,
        training_overrides=dict(num_steps_per_env=args.steps,num_learning_epochs=1,auxiliary_epochs=1,num_mini_batches=1),
        payload_overrides=dict(mode=args.mode))
    env=None; status=1
    try:
        env=IsaacLabMULEEnv(resolved,args.gpu,True,reward_fn=test_only_mule_tracking_reward,reward_contract='test_only_tracking_v1')
        packet=env.reset(); s=env.simulator; checks={}
        labels=env.physical_labels()
        assert (labels['realized_robot']['tray_mass']>0).all()
        assert torch.equal(labels['total_ball_mass_kg'],s.ball_mass.sum(-1))
        if args.mode=='nominal': assert not s.balls and labels['total_ball_mass_kg'].count_nonzero()==0
        labels['realized_robot']['motor_strength'].zero_()
        assert (s._motor_strength>0).all()  # owned labels, no property aliases
        rng=env.actor_observation_cache.generator.get_state().clone()
        torch.testing.assert_close(packet.actor,env.observe().actor,rtol=0,atol=0)
        assert torch.equal(rng,env.actor_observation_cache.generator.get_state())
        before=env.observe(); roots=s._robot.data.root_link_state_w.clone()
        env.reset(torch.tensor([0],device=args.gpu))
        torch.testing.assert_close(before.actor[1:],env.observe().actor[1:],rtol=0,atol=0)
        torch.testing.assert_close(roots[1:],s._robot.data.root_link_state_w[1:],rtol=0,atol=0)
        checks.update(observe_pure=True,partial_reset=True)
        for k,ball in enumerate(s.balls):
            torch.testing.assert_close(ball.root_physx_view.get_inertias().to(args.gpu).reshape(-1,3,3),
                sphere_inertia(s.ball_mass[:,k],s.ball_radius[:,k]),rtol=1e-5,atol=1e-8)
        if s.balls:
            positions=[b.data.root_state_w.clone() for b in s.balls]
            s.mass_elapsed.fill_(3.99); s.periodic_mass_update(.02)
            assert (s.mass_change_count==1).all() and (s.ball_mass<=2.).all()
            for k,ball in enumerate(s.balls):
                torch.testing.assert_close(ball.data.root_state_w,positions[k])
                torch.testing.assert_close(ball.root_physx_view.get_masses().to(args.gpu)[:,0],s.ball_mass[:,k])
                torch.testing.assert_close(ball.root_physx_view.get_inertias().to(args.gpu).reshape(-1,3,3),
                    sphere_inertia(s.ball_mass[:,k],s.ball_radius[:,k]),rtol=1e-5,atol=1e-8)
            checks['periodic_mass_inertia_pose_velocity']=True
        if args.physical_checks:
            from mule_payload_physical_checks import physics_checks
            checks.update(physics_checks(env))
        runner=MULERunner(env,resolved.training,args.gpu,console=False)
        torch.cuda.reset_peak_memory_stats(); started=time.perf_counter()
        logs=runner.learn(1); runner.save(out/'nominal.pt')
        saved=deepcopy(runner.alg.model.state_dict())
        runner.load(out/'nominal.pt',resume_mode='fresh')
        for k,v in runner.alg.model.state_dict().items():
            if isinstance(v,torch.Tensor): torch.testing.assert_close(v,saved[k],rtol=0,atol=0)
            else: assert v==saved[k]
        assert runner.history.observations.count_nonzero()==0
        runner.transition_to_joint_adaptation(); logs+=runner.learn(1); runner.save(out/'joint.pt')
        sampling=s.sampling_state_dict(); runner.load(out/'joint.pt',resume_mode='fresh')
        assert runner.alg.model.phase=='joint_adaptation' and runner.alg.iteration==2
        torch.testing.assert_close(s.ball_mass,sampling['tensors']['ball_mass'])
        torch.testing.assert_close(s.mass_elapsed,sampling['tensors']['mass_elapsed'])
        assert torch.equal(s.payload_rng.get_state(),sampling['rng'])
        assert all(torch.isfinite(torch.tensor(v)) for log in logs for v in log.values())
        torch.cuda.synchronize(); elapsed=time.perf_counter()-started
        versions={}
        for name in ('isaaclab','isaacsim','torch','warp-lang'):
            try: versions[name]=importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError: versions[name]='unavailable'
        import warp
        versions['warp_runtime']=warp.__version__
        report=dict(checks=checks,mode=args.mode,num_envs=args.num_envs,steps=args.steps,
            phases=['nominal_only','joint_adaptation'],logs=logs,elapsed_s=elapsed,
            interactions_per_s=2*args.steps*args.num_envs/elapsed,
            peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(),
            versions=versions,resume=runner.resume_report,reward='test_only_tracking_v1')
        (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False))
        print('MULE_PAYLOAD_SMOKE_PASS',json.dumps(report),flush=True); status=0
    except Exception:
        traceback.print_exc(); print('MULE_PAYLOAD_SMOKE_FAIL',flush=True)
    finally:
        if env is not None:
            try: env.close()
            except SystemExit: pass
    return status


if __name__=='__main__': raise SystemExit(main())
