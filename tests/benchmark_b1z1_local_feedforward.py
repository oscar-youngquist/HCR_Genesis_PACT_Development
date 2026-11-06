"""GPU-0 teacher microbenchmark; real randomized BARD, no simulator/training.

Reports shared mechanics, extra local cache, and selected-row loss/backward
separately. This is NOT whole-training throughput. Run with PYTHONPATH=.
"""
import argparse
import json
import time
from types import SimpleNamespace
import torch
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.b1z1.b1z1_pact.b1z1_pact_config import B1Z1PACTCfg
from legged_gym.dynamics import BardB1Z1DynamicsBackend
from rsl_rl.algorithms import b1z1_local_feedforward as local, b1z1_actor_feedforward as ff
from rsl_rl.algorithms.b1z1_bard_pinn import mechanics
from rsl_rl.algorithms.ppo_b1z1_pact import PPO_B1Z1PACT


def main():
    p=argparse.ArgumentParser();p.add_argument('--count',type=int,default=4096);p.add_argument('--repeats',type=int,default=10)
    args=p.parse_args();torch.manual_seed(5);c=B1Z1PACTCfg();n=args.count;device='cuda:0'
    names=c.asset.dof_names;dofs=len(names);count=c.env.num_actions
    ids=[i for i,name in enumerate(names) if name.startswith('z1_') and i<count]
    s=torch.zeros(n,180,device=device);s[:,6]=1
    s[:,7:26]=torch.tensor([c.init_state.default_joint_angles[name] for name in names],device=device)
    s[:,156:175]=s[:,7:26];s[:,97:116]=1;s[:,116:135]=30;s[:,135:154]=1;s[:,154:156]=1
    s[:,175]=1.;s[:,176:179]=.01;s[:,179]=.1
    backend=BardB1Z1DynamicsBackend(c.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR),names,
        c.asset.foot_name,c.asset.gripper_name,c.asset.base_name,device=device,batch_capacity=n)
    def timed(fn):
        for _ in range(3):fn()
        torch.cuda.synchronize();start=time.perf_counter()
        for _ in range(args.repeats):fn()
        torch.cuda.synchronize();return 1000*(time.perf_counter()-start)/args.repeats
    def common():return mechanics(backend,dict(rollout_initial_state=s[:,:51],dynamics_state=s))
    common_ms=timed(common);fixed=common()
    prep=lambda:local.kinematics(backend,s,fixed.ee_jacobian)
    local_ms=timed(prep)
    for k,v in prep().items():setattr(fixed,k,v)
    cfg=dict(actor_phys_enabled=True,actor_phys_coef=1.,actor_phys_arm_feedforward_enabled=True,
        actor_phys_arm_feedforward_weight=.05,actor_phys_compliance_enabled=True,
        actor_phys_arm_feedforward_compliant_local_enabled=True,actor_phys_arm_indices=ids,
        actor_phys_compliance_omega=c.compliance.omega,actor_phys_compliance_zeta=c.compliance.zeta,
        actor_phys_compliance_dls_damping=c.compliance.dls_damping,actor_phys_compliance_joint_margin=c.compliance.joint_margin,
        clip_actions=1.,position_action_scale=c.control.action_scale,torque_action_scale=c.control.torque_scale,
        actor_phys_huber_delta=1.,dt=c.sim.dt*c.control.decimation)
    a=SimpleNamespace(cfg=cfg,actor_critic=SimpleNamespace(num_actions=count,action_mode='coupled'),pinn_weight=1.)
    a._coupled_torque=lambda act,state:PPO_B1Z1PACT._coupled_torque(a,act,state)
    actions=torch.zeros(n,2*count,device=device,requires_grad=True)
    acc=torch.zeros(n,25,device=device)  # Cached rollout acceleration supplied identically to both teachers.
    data=dict(local_ref_position_start=fixed.local_s,local_ref_velocity_start=torch.zeros(n,3,device=device),
        local_ref_acceleration=torch.zeros(n,3,device=device),local_ref_valid=torch.ones(n,1,device=device,dtype=torch.bool),
        feedforward_applied_force=torch.zeros(n,3,device=device),q_min=s.new_full((n,dofs),-4),q_max=s.new_full((n,dofs),4),
        qd_max=s.new_full((n,dofs),100),torque_max=s.new_full((n,dofs),100))
    results={}
    for name in ('disabled','original','local'):
        cfg['actor_phys_arm_feedforward_enabled']=name!='disabled'
        cfg['actor_phys_arm_feedforward_compliant_local_enabled']=name=='local'
        def objective():
            actions.grad=None
            loss,metrics=ff.objective(a,actions,s,fixed,acc,dofs,data);loss.backward()
            assert torch.isfinite(loss)
            return metrics
        torch.cuda.reset_peak_memory_stats()
        elapsed=timed(objective);metrics=objective()
        results[name]=dict(loss_backward_ms=elapsed,extra_cache_ms=local_ms if name=='local' else 0.,
            valid_fraction=float(metrics.get('arm_feedforward_valid_fraction',0.)),
            peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
    print(json.dumps(dict(count=n,repeats=args.repeats,shared_mechanics_ms=common_ms,teachers=results),indent=2))


if __name__=='__main__':main()
