"""Two-environment reference/controller integration smoke; no learning or checkpoint.

SIMULATOR=isaaclab_b1z1_pact python -m tests.smoke_b1z1_compliance
is equivalent to running this file with the repository on PYTHONPATH.
"""
import torch
import argparse
import json
import sys
import legged_gym.envs
from legged_gym.utils import get_args, task_registry
from legged_gym.envs.b1z1.directional_compliance import chart, heading, rotate_yaw, cartesian, task_position


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--profile-scenarios',action='store_true')
    parser.add_argument('--review-collision',action='store_true')
    options,remaining=parser.parse_known_args()
    sys.argv=[sys.argv[0]]+remaining
    args=get_args()
    options.profile_scenarios |= options.review_collision
    args.num_envs=8 if options.profile_scenarios else 2;args.headless=True
    cfg,_=task_registry.get_cfgs(args.task,args)
    cfg.compliance.enabled=True
    cfg.terrain.mesh_type='plane';cfg.terrain.curriculum=False
    env,_=task_registry.make_env(args.task,args=args,env_cfg=cfg)
    try:
        env.reset()
        # Deterministic load/removal in this test only; exercise the original
        # force-application path and verify no duplicate torso force is added.
        env.cfg.commands.push_gripper_stators=True
        env.cfg.commands.apply_ee_external_forces=True
        env.cfg.commands.apply_base_external_forces=False
        env.cfg.commands.apply_base_external_torques=False
        load=torch.zeros(args.num_envs,3,device=env.device)
        if options.profile_scenarios:
            # Matched diagnostic loads, not a modification to production sampling.
            names=['rejection','arm_only','base_first','handoff','lateral_no_yaw',
                   'yaw_only','payload','reverse_payload']
            profiles=torch.zeros(8,12,device=env.device)
            profiles[1,3]=.7
            profiles[2,9]=.7
            profiles[3,3]=.7;profiles[3,9]=.7
            if options.review_collision:
                profiles[3,:10]=1  # EE yielding, independent yaw disabled.
            profiles[4,9]=.7
            profiles[5,10:]=.7
            profiles[6,4:6]=torch.tensor([.1,.5],device=env.device)
            profiles[7,4:6]=torch.tensor([.5,.1],device=env.device)
            env.compliance.reference.profile.copy_(profiles)
            env.commands[:,:3]=0;env.commands[4,0]=.2
            traces=[]
        def apply_load(ids):
            env.ee_force_ext_world.copy_(load)
            env.simulator.apply_ee_force(load)
        env._push_gripper=apply_load
        for step in range(6):
            load.zero_()
            load[:,1]=10. if step<3 else 0.
            if options.profile_scenarios:
                load[6:,1]=0;load[6:,2]=-10. if step<3 else 0.
            if options.review_collision and step<3:
                # Diagnostic only: request motion toward a real nonadjacent
                # arm segment. Use current simulator configuration and frames;
                # do not alter production force distributions or geometry.
                from legged_gym.utils.math_utils import quat_apply
                ee,_,arm,_=env.compliance.geometry.forward(env.simulator.dof_pos)
                direction=(arm[:,3]+arm[:,4])/2-ee
                world=quat_apply(env.simulator.base_quat,direction)
                task=rotate_yaw(world,heading(env.simulator.base_quat),True)
                force=task/task.new_tensor(cfg.compliance.ee_gain)
                load[3]=rotate_yaw(force,heading(env.simulator.base_quat))[3]
                if step==1:
                    # Initialize a near-obstacle accepted reference to exercise
                    # handoff without running an untrained robot into contact.
                    # This is a reference-state diagnostic, not a measured move.
                    from legged_gym.envs.b1z1.ee_collision import clip_prefix
                    geometry=env.compliance.geometry
                    _,_,points,legs=geometry.forward(env.simulator.dof_pos)
                    end=ee+direction
                    prefix=clip_prefix(end,ee,lambda candidate,start:
                        geometry.path_clearance(start,candidate,points,legs).amin(-1)>=0)
                    near=ee+.85*(prefix-ee)
                    world_near=quat_apply(env.simulator.base_quat,near)+env.simulator.base_pos
                    accepted=task_position(world_near,env.simulator.base_pos,heading(env.simulator.base_quat))
                    state=env.compliance.reference
                    state.accepted[3]=accepted[3]
                    state.delta[3]=accepted[3]-env.compliance.nominal(0)[0][3]
                    state.rate[3]=0
            if options.review_collision and step==4:
                # Test-only terrain-height measurement change, not a terrain
                # teleport or claim of trained-policy terrain performance.
                sphere=load.new_tensor([.9,-.5,0.])
                env.ee_start_sphere[0]=sphere;env.ee_goal_sphere[0]=sphere
                env.compliance.reference.accepted[0]=env.compliance.center()[0]+cartesian(sphere)
                env.compliance.reference.delta[0]=0
                env.compliance.reference.rate[0]=.3
                env.simulator.measured_heights[0]+=.01
            action=torch.zeros(args.num_envs,cfg.env.num_policy_actions,device=env.device)
            result=env.step(action)
            assert all(torch.isfinite(x).all() for x in result[:6])
            assert env.obs_buf.shape[-1]==cfg.env.num_observations
            assert env.privileged_obs_buf.shape[-1]==cfg.env.num_privileged_obs*cfg.env.num_priv_stack
            ref=env.compliance.target
            sphere=chart(ref.position-env.compliance.center(),env.compliance.bounds)
            assert (sphere>=env.compliance.bounds[:,0]-1e-5).all()
            assert (sphere<=env.compliance.bounds[:,1]+1e-5).all()
            assert not any(x.requires_grad for x in env.compliance.interval.values())
            torch.testing.assert_close(env.simulator._external_force_world.sum(1),load)
            if options.profile_scenarios:
                assert not result[5].any(), 'Diagnostic window unexpectedly reset; do not mix episodes'
                torch.testing.assert_close(env.compliance.profile,profiles)
                assert ref.base[[0,1,6,7]].eq(0).all()
                assert ref.base[[2,3,4],2].eq(0).all()
                assert ref.base[5,:2].eq(0).all()
                if step<3:
                    assert ref.base[2,1]>0 and ref.base[4,1]>0
                    assert ref.base[5,2].abs()>0
                    assert env.compliance.reference.delta[1,1]>0, 'EE-only clearance must permit yielding'
                    if options.review_collision:
                        if step==0:
                            assert ref.collision[3].max()<.1, 'Distant obstacle must not immediately recruit base'
                        if step==1:
                            assert ref.collision[3].max()>.5, 'Nearby reference must recruit base'
                            assert ref.base[3,:2].abs().sum()>0
                        assert env.compliance.reference.delta[3].norm()>0
                if options.review_collision and step==4:
                    assert env.compliance.reference.recovering[0]
                    assert env.compliance.reference.delta[0].abs().max()<1e-6
                    assert env.compliance.reference.rate[0].eq(0).all()
                traces.append(dict(step=step,ee_offset=env.compliance.reference.delta.cpu().tolist(),
                    base_target=ref.base.cpu().tolist(),gates=ref.gates.cpu().tolist()))
        if options.profile_scenarios:
            print('PROFILE_SCENARIOS '+json.dumps(dict(names=names,traces=traces)),flush=True)
        print(f'COMPLIANCE SMOKE PASSED: {args.task}, device={env.device}, {args.num_envs} environments, 6 decisions; no learning',flush=True)
    except BaseException:
        # Kit shutdown can terminate Python before its pending exception prints.
        import traceback
        traceback.print_exc()
        raise
    finally:
        env.simulator._app_launcher.app.close()


if __name__=='__main__':main()
