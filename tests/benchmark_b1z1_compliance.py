"""Reference-only CUDA microbenchmark: URDF FK, all gates and feasibility probes.

No simulator, policy, or learning. Timing includes bounded collision batches;
it is overhead per control interval, not end-to-end simulator throughput.
"""
import argparse
import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path
import torch
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.b1z1.b1z1_pact.b1z1_pact_config import B1Z1PACTCfg
from legged_gym.envs.b1z1.compliance_geometry import RobotGeometry
from legged_gym.envs.b1z1.directional_compliance import ComplianceReference,cartesian,directional_probe


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--counts',type=int,nargs='+',default=[128,4096])
    parser.add_argument('--repeats',type=int,default=10)
    parser.add_argument('--collision-batch-size',type=int)
    parser.add_argument('--compile-geometry',action='store_true')
    parser.add_argument('--compile-full-fk',action='store_true')
    args=parser.parse_args()
    cfg=B1Z1PACTCfg();device='cuda:0';names=cfg.asset.dof_names
    cfg.compliance.compile_geometry=False  # benchmark variants explicitly below
    if args.collision_batch_size is not None:cfg.compliance.collision_batch_size=args.collision_batch_size
    arm=[i for i,n in enumerate(names) if n.startswith('z1_') and n!='z1_jointGripper']
    path=cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
    model=RobotGeometry(path,names,arm,cfg.asset.base_name,cfg.asset.gripper_name,cfg.asset.foot_name,cfg.compliance,device)
    if args.compile_geometry:model._probe_clearance=torch.compile(model._probe_clearance,dynamic=True,fullgraph=True)
    if args.compile_full_fk:model.forward=torch.compile(model.forward,dynamic=True,fullgraph=True)
    joints={j.get('name'):j.find('limit') for j in ET.parse(path).getroot().findall('joint')}
    limits=torch.tensor([[float(joints[names[i]].get(k)) for k in ('lower','upper')] for i in arm],device=device)
    bounds=torch.tensor([cfg.goal_ee.ranges.pos_l,cfg.goal_ee.ranges.pos_p,cfg.goal_ee.ranges.pos_y],device=device)
    for n in args.counts:
        full=torch.tensor([cfg.init_state.default_joint_angles[k] for k in names],device=device).expand(n,-1)
        q=full[:,arm];ref=ComplianceReference(cfg.compliance,n,device)
        center=torch.zeros(n,3,device=device)
        nominal=cartesian(bounds.mean(-1).expand(n,-1))
        forces=nominal.new_tensor([30.,20.,-25.]).expand_as(nominal)
        zero=torch.zeros_like(nominal)
        def step():
            measured,j,_,legs=model.forward(full)
            clear=lambda x:model.clearance(x,full,legs)
            baseline=clear(q)
            def safe(candidate,previous):
                dq,_,_=directional_probe(j,q,limits.expand(n,-1,-1),candidate-measured,cfg.compliance)
                fractions=torch.linspace(1/3,1,3,device=device)
                margin=clear(q[:,None]+dq[:,None]*fractions[None,:,None]).amin(-1)
                return torch.where(((margin>=0)|(margin>baseline+1e-6))[:,None],candidate,previous)
            return ref.advance(nominal=nominal,nominal_velocity=zero,center=center,bounds=bounds,
                force=forces,lever=measured,command=zero,command_limits=nominal.new_tensor([[-1.,1.]]*3),
                q=q,limits=limits.expand(n,-1,-1),jacobian=j,collision_clearance=clear,dt=.02,
                target_safe=safe,measured_position=measured)
        warmup=time.perf_counter()
        for _ in range(3):step()
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        warmup_seconds=time.perf_counter()-warmup
        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(args.repeats):result=step()
        end.record();torch.cuda.synchronize()
        assert torch.isfinite(result.position).all()
        print(json.dumps(dict(environments=n,repeats=args.repeats,ms_per_interval=start.elapsed_time(end)/args.repeats,
            compile_geometry=args.compile_geometry,compile_full_fk=args.compile_full_fk,warmup_seconds=warmup_seconds,
            peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,device=torch.cuda.get_device_name(0))),flush=True)


if __name__=='__main__':main()
