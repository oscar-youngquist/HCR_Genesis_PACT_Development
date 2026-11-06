"""Reference-only CUDA microbenchmark: URDF FK, all gates and feasibility probes.

No simulator, policy, or learning. Timing includes bounded collision batches;
it is overhead per control interval, not end-to-end simulator throughput.
"""
import argparse
import json
import statistics
import time
import xml.etree.ElementTree as ET
from pathlib import Path
import torch
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.b1z1.b1z1_pact.b1z1_pact_config import B1Z1PACTCfg
from legged_gym.envs.b1z1.compliance_geometry import RobotGeometry
from legged_gym.envs.b1z1.ee_collision import clip_prefix
from legged_gym.envs.b1z1.directional_compliance import ComplianceReference,cartesian,directional_probe,collision_approach


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--counts',type=int,nargs='+',default=[128,4096])
    parser.add_argument('--repeats',type=int,default=10)
    parser.add_argument('--collision-batch-size',type=int)
    parser.add_argument('--compile-geometry',action='store_true')
    parser.add_argument('--compile-full-fk',action='store_true')
    parser.add_argument('--collision-limited',action='store_true',help='Stress safe-prefix clipping for every row')
    parser.add_argument('--blocked-fraction',type=float,default=1.,help='Fraction driven toward obstacles in the collision-limited case')
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--full-batch-refinement',action='store_true',help='Fixed full-batch refinement (now the default)')
    mode.add_argument('--compact-refinement',action='store_true',help='Opt-in compact/bucketed CUDA-graph path')
    parser.add_argument('--vary-blocked-fraction',action='store_true',help='Cycle zero/one/1%%/10%%/50%%/all obstructed rows after warmup')
    parser.add_argument('--legacy-kernels',action='store_true',help='Unfused target/probe path for matched runtime and numerical comparison')
    parser.add_argument('--profile-output',help='Optional warmed CPU/CUDA trace; profiled timings include instrumentation overhead')
    args=parser.parse_args()
    if not 0 <= args.blocked_fraction <= 1:parser.error('blocked-fraction must be in [0,1]')
    cfg=B1Z1PACTCfg();device='cuda:0';names=cfg.asset.dof_names
    cfg.compliance.compile_geometry=False  # benchmark variants explicitly below
    cfg.compliance.collision_compact_refinement=args.compact_refinement
    if args.collision_batch_size is not None:cfg.compliance.collision_batch_size=args.collision_batch_size
    arm=[i for i,n in enumerate(names) if n.startswith('z1_') and n!='z1_jointGripper']
    path=cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
    model=RobotGeometry(path,names,arm,cfg.asset.base_name,cfg.asset.gripper_name,cfg.asset.foot_name,cfg.compliance,device)
    model.graph_target_checks=args.compile_geometry and args.compact_refinement and not args.legacy_kernels
    if args.compile_geometry:
        model._probe_clearance=torch.compile(model._probe_clearance,dynamic=True,fullgraph=True)
        model.path_clearance=torch.compile(model.path_clearance,dynamic=True,fullgraph=True)
        model.point_clearance=torch.compile(model.point_clearance,dynamic=True,fullgraph=True)
        if not args.legacy_kernels:
            mode='reduce-overhead' if model.graph_target_checks else 'default'
            model._target_path_allowed=torch.compile(model._target_path_allowed,dynamic=True,fullgraph=True,mode=mode)
            model.handoff_paths=torch.compile(model.handoff_paths,dynamic=True,fullgraph=True)
    cfg.compliance.compile_geometry=args.compile_geometry
    if args.compile_full_fk:model.forward=torch.compile(model.forward,dynamic=True,fullgraph=True)
    joints={j.get('name'):j.find('limit') for j in ET.parse(path).getroot().findall('joint')}
    limits=torch.tensor([[float(joints[names[i]].get(k)) for k in ('lower','upper')] for i in arm],device=device)
    bounds=torch.tensor([cfg.goal_ee.ranges.pos_l,cfg.goal_ee.ranges.pos_p,cfg.goal_ee.ranges.pos_y],device=device)
    for n in args.counts:
        full=torch.tensor([cfg.init_state.default_joint_angles[k] for k in names],device=device).expand(n,-1)
        q=full[:,arm];ref=ComplianceReference(cfg.compliance,n,device)
        if args.legacy_kernels:
            from legged_gym.envs.b1z1.directional_compliance import workspace_gate
            ref.directional_probe=directional_probe;ref.workspace_gate=workspace_gate
        center=torch.zeros(n,3,device=device)
        nominal=cartesian(bounds.mean(-1).expand(n,-1))
        forces=nominal.new_tensor([30.,20.,-25.]).expand_as(nominal)
        if args.collision_limited:
            nominal,_,points,_=model.forward(full)
            forces=((points[:,3]+points[:,4])/2-nominal)/nominal.new_tensor(cfg.compliance.ee_gain)
            all_forces=forces.clone()
            forces[round(n*args.blocked_fraction):]=0
            ref.profile.fill_(1)
        zero=torch.zeros_like(nominal)
        identity=nominal.new_tensor([0.,0.,0.,1.]).expand(n,-1)
        yaw=nominal.new_zeros(n)
        lower,upper=nominal.new_full((3,),-100.),nominal.new_full((3,),-99.)
        def step():
            measured,j,arm_points,legs=model.forward(full)
            clear=lambda x:model.clearance_pairs(x,full,legs)
            baseline=clear(q)
            def path_gate(directions,previous):
                if not args.legacy_kernels:
                    return model.handoff_paths(directions,previous,measured,arm_points,legs,
                        zero,identity,yaw,cfg.compliance.collision_margin)
                result=torch.zeros(directions.shape[:2],device=device)
                for origin in (measured,previous):
                    start=origin[:,None].expand_as(directions)
                    baseline=model.point_clearance(start,arm_points[:,None],legs[:,None])
                    path=model.path_clearance(start,start+directions,arm_points[:,None],legs[:,None])
                    result=torch.maximum(result,collision_approach(baseline,path,cfg.compliance.collision_margin))
                return result
            def make_path(ids=None):
                take=lambda x:x if ids is None else x.index_select(0,ids)
                j0,q0,full0,measured0,arm0,legs0,baseline0=map(take,(j,q,full,measured,arm_points,legs,baseline))
                zero0,identity0,yaw0=map(take,(zero,identity,yaw))
                fractions=torch.linspace(1/3,1,3,device=device)
                def allowed_path(candidate,previous):
                    if not args.legacy_kernels:
                        return model.target_path_allowed(candidate,previous,q0,j0,limits.expand(len(q0),-1,-1),
                            measured0,baseline0,arm0,legs0,zero0,identity0,yaw0,zero0,
                            lower=lower,upper=upper,underground=-100.)
                    dq,_,_=directional_probe(j0,q0,limits.expand(len(q0),-1,-1),candidate-measured0,cfg.compliance)
                    margin=model.clearance_pairs(q0[:,None]+dq[:,None]*fractions[None,:,None],full0,legs0).amin(1)
                    allowed=((margin>=0)|(margin>baseline0+1e-6)).all(-1)
                    start=torch.stack((measured0,previous),1)
                    clearance=model.point_clearance(start,arm0[:,None],legs0[:,None])
                    path=model.path_clearance(start,candidate[:,None].expand_as(start),arm0[:,None],legs0[:,None])
                    end=model.point_clearance(candidate,arm0,legs0)[:,None]
                    return allowed & ((path>=0)|((clearance<0)&(path>=clearance-1e-6)&(end>clearance+1e-6))).all(-1).all(-1)
                return allowed_path
            safe=lambda candidate,previous:clip_prefix(candidate,previous,make_path(),
                select_path=make_path if args.compact_refinement else None)
            return ref.advance(nominal=nominal,nominal_velocity=zero,center=center,bounds=bounds,
                force=forces,lever=measured,command=zero,command_limits=nominal.new_tensor([[-1.,1.]]*3),
                q=q,limits=limits.expand(n,-1,-1),jacobian=j,collision_clearance=clear,dt=.02,
                target_safe=safe,measured_position=measured,collision_path_gate=path_gate,
                collision_start=None if args.legacy_kernels else baseline)
        warmup=time.perf_counter()
        for _ in range(3):step()
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        warmup_seconds=time.perf_counter()-warmup
        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        intervals=[(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)) for _ in range(args.repeats)]
        from torch._dynamo.utils import counters
        graphs_before=counters['stats']['unique_graphs']
        profiler=None
        if args.profile_output:
            profiler=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA])
            profiler.__enter__()
        start.record()
        for i,(a,b) in enumerate(intervals):
            if args.vary_blocked_fraction and args.collision_limited:
                count=(0,1,round(.01*n),round(.1*n),round(.5*n),n)[i%6]
                forces.zero_();forces[:count]=all_forces[:count]
            a.record();result=step();b.record()
        end.record();torch.cuda.synchronize()
        durations=sorted(a.elapsed_time(b) for a,b in intervals)
        if profiler is not None:
            profiler.__exit__(None,None,None)
            profiler.export_chrome_trace(args.profile_output)
            print(profiler.key_averages().table(sort_by='self_cpu_time_total',row_limit=15))
        assert torch.isfinite(result.position).all()
        print(json.dumps(dict(environments=n,repeats=args.repeats,collision_limited=args.collision_limited,ms_per_interval=start.elapsed_time(end)/args.repeats,
            blocked_fraction=args.blocked_fraction,full_batch_refinement=not args.compact_refinement,
            varied_blocked_fraction=args.vary_blocked_fraction,
            interval_median_ms=statistics.median(durations),interval_p95_ms=durations[int(.95*(len(durations)-1))],
            interval_max_ms=max(durations),new_compile_graphs=counters['stats']['unique_graphs']-graphs_before,
            legacy_kernels=args.legacy_kernels,
            compile_geometry=args.compile_geometry,compile_full_fk=args.compile_full_fk,warmup_seconds=warmup_seconds,
            peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,device=torch.cuda.get_device_name(0))),flush=True)


if __name__=='__main__':main()
