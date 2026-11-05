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
from legged_gym.envs.b1z1.directional_compliance import chart


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--profile-scenarios',action='store_true')
    options,remaining=parser.parse_known_args()
    sys.argv=[sys.argv[0]]+remaining
    args=get_args()
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
            load[:,1]=10. if step<3 else 0.
            if options.profile_scenarios:
                load[6:,1]=0;load[6:,2]=-10. if step<3 else 0.
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
