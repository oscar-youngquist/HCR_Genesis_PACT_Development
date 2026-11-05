"""Eight matched diagnostic wrench cases, six decisions, no policy or learning.

SIMULATOR=isaaclab_b1z1_pact PYTHONPATH=. python tests/smoke_b1z1_torso_compliance.py --task b1z1_pact --gpu cuda:0 --headless
"""
import json
import torch

import legged_gym.envs
from legged_gym.utils import get_args, task_registry
from legged_gym.envs.b1z1.directional_compliance import heading, rotate_yaw


def main():
    args=get_args();args.num_envs=8;args.headless=True
    cfg,_=task_registry.get_cfgs(args.task,args)
    cfg.compliance.enabled=True;cfg.compliance.include_torso_disturbances=True
    cfg.terrain.mesh_type='plane';cfg.terrain.curriculum=False
    cfg.domain_rand.push_robots=False
    cfg.commands.push_gripper_stators=True;cfg.commands.push_robot_base=True
    cfg.commands.apply_ee_external_forces=True
    cfg.commands.apply_base_external_forces=True;cfg.commands.apply_base_external_torques=True
    cfg.commands.compliance_disturbance_mode_weights=None
    env,_=task_registry.make_env(args.task,args=args,env_cfg=cfg)
    try:
        env.reset()
        sim=env.simulator;state=env.compliance
        state.reference.profile.zero_();state.reference.profile[:,6:]=1
        state.reference.profile[5,:6]=1  # Arm gates closed with zero EE load.
        state.reference.profile[7,6:]=0  # All base axes disabled despite loading.
        env.commands[:,:3]=0
        ee=torch.zeros(8,3,device=env.device);base=ee.clone();couple=ee.clone()
        def apply_ee(ids):
            env.ee_force_ext_world.copy_(ee);sim.apply_ee_force(ee)
        def apply_torso(ids):
            env.base_force_ext_world.copy_(base);env.base_torque_ext_world.copy_(couple)
            sim.apply_base_force(base);sim.apply_base_torque(couple)
        env._push_gripper=apply_ee;env._push_robot_base=apply_torso
        names=['torso_translation','torso_yaw','ee_only','reinforcing','opposing',
               'torso_closed_ee_gates','vertical_roll_pitch','base_disabled']
        traces=[]
        for step in range(6):
            f=ee.new_zeros(8,3);b=f.clone();t=f.clone()
            if step<3:
                b[0,0]=10;t[1,2]=3
                f[2:5,0]=10;b[3,0]=5;b[4,0]=-10
                b[5,1]=10;b[6,2]=30;t[6,:2]=3
                f[7,0]=10;b[7,1]=10;t[7,2]=3
            yaw=heading(sim.base_quat)
            ee.copy_(rotate_yaw(f,yaw));base.copy_(rotate_yaw(b,yaw));couple.copy_(rotate_yaw(t,yaw))
            # Compare interval-start wrench snapshots, not successor geometry.
            point=sim._robot.data.body_link_pos_w[:,sim._base_link_index]
            point_lever=rotate_yaw(point-sim.base_pos,yaw,True)
            expected_direct=torch.cat((b[:,:2],(t+torch.cross(point_lever,b,dim=-1))[:,2:]),-1)
            lever=rotate_yaw(sim.ee_pos-sim.base_pos,yaw,True)
            result=env.step(torch.zeros(8,cfg.env.num_policy_actions,device=env.device))
            assert not result[5].any(), 'Unexpected reset in diagnostic window'
            assert all(torch.isfinite(value).all() for value in result[:6])
            ref=state.target
            gated=f*ref.gates
            expected_ee=torch.cat((gated[:,:2],torch.cross(lever,gated,dim=-1)[:,2:]),-1)
            torch.testing.assert_close(ref.torso_wrench,expected_direct,atol=1e-4,rtol=1e-4)
            torch.testing.assert_close(ref.ee_wrench,expected_ee,atol=1e-4,rtol=1e-4)
            torch.testing.assert_close(ref.combined_wrench,expected_direct+expected_ee,atol=1e-4,rtol=1e-4)
            # Only original physical forces/couples enter the simulator. There
            # is no duplicate EE lever force/torque in the torso buffers.
            torch.testing.assert_close(sim._external_force_world.sum(1),ee+base)
            torch.testing.assert_close(sim._external_torque_world.sum(1),couple)
            torch.testing.assert_close(state.interval['command'],ref.base)
            assert not any(value.requires_grad for value in state.interval.values())
            assert ref.base[7].eq(0).all()
            if step<3:
                assert ref.base[0,0]>0 and ref.base[1,2]>0 and ref.base[5,1]>0
                assert ref.base[3,0]>ref.base[2,0]
                assert ref.base[4,0].abs()<1e-5
                assert ref.base[6].abs().max()<1e-5
                assert ref.gates[5].eq(0).all()
            traces.append(dict(step=step,direct=ref.torso_wrench.cpu().tolist(),
                ee_gated=ref.ee_wrench.cpu().tolist(),combined=ref.combined_wrench.cpu().tolist(),
                base_target=ref.base.cpu().tolist()))
        assert traces[5]['base_target'][0][0]<traces[2]['base_target'][0][0]
        log=state.wrench_statistics(torch.arange(8,device=env.device))
        assert log['compliance_wrench/interval_count']>=48
        print('TORSO_COMPLIANCE_SMOKE '+json.dumps(dict(task=args.task,device=env.device,names=names,traces=traces)),flush=True)
        print('TORSO COMPLIANCE SMOKE PASSED: 8 environments, 6 decisions; no learning',flush=True)
    except BaseException:
        import traceback
        traceback.print_exc();raise
    finally:
        env.simulator._app_launcher.app.close()


if __name__=='__main__':main()
