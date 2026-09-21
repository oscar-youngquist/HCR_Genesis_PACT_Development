"""Bounded real collection/PPO diagnostic; phase markers and timeout stacks."""
import faulthandler
import time
import sys
faulthandler.enable();faulthandler.dump_traceback_later(60,repeat=True)
print('PHASE startup begin',flush=True)
started=time.perf_counter()
from legged_gym.scripts.train_hard_pact import prepare_solver_runtime
prepare_solver_runtime(sys.argv[1:])
import torch
import tempfile
import json
from dataclasses import replace
from legged_gym.envs import *
from legged_gym.utils import get_args,task_registry

args=get_args();cfg,train=task_registry.get_cfgs(args.task,args)
cfg.terrain.mesh_type='trimesh';cfg.terrain.num_rows=2;cfg.terrain.num_cols=2
cfg.terrain.max_init_terrain_level=1
train.runner.resume=False;train.policy.pretrained_path=None
train.runner.num_steps_per_env=4
train.algorithm.num_learning_epochs=1;train.algorithm.num_mini_batches=1
train.algorithm.num_encoder_epochs=1;train.algorithm.ppo_qp_shard_percentage=100.
train.policy.pinn_init_steps=-1;train.policy.pinn_warmup=1
train.algorithm.hard_pact_qp.update(warmup_iterations=0,diagnostics_level='minimal',
    cuda_event_profiling=False,tensorboard_diagnostics_interval=1)
print('PHASE environment begin',flush=True)
env,_=task_registry.make_env(args.task,args,env_cfg=cfg)
print('PHASE runner begin',flush=True)
runner,_=task_registry.make_alg_runner(env,args.task,args,train_cfg=train,
    log_root=tempfile.mkdtemp(prefix='hard_pact_stall_'))
torch.cuda.synchronize()
print('PHASE startup end seconds',time.perf_counter()-started,flush=True)
qp=runner.alg.hard_pact_qp
timings={};counts={};observed={}
gradient_sums={}
actor=runner.alg.actor_critic
for label,weight in (('actor_tau',actor.act_tau_out.weight),('actor_position',actor.act_pos_out.weight),
    ('encoder',actor.context_encoder.ce_out_mean.weight),
    ('grf',actor.physics_estimator.grf_head[-1].weight),('wrench',actor.physics_estimator.wrench_head[-1].weight)):
    def record(gradient,label=label):
        gradient_sums[label]=gradient_sums.get(label,0)+gradient.detach().abs().sum()
    weight.register_hook(record)
def wrap(owner,name,label):
    original=getattr(owner,name)
    def call(*a,**kw):
        counts[label]=counts.get(label,0)+1
        first=counts[label]<=2
        if first:print('PHASE',label,'begin',counts[label],flush=True)
        torch.cuda.synchronize();t=time.perf_counter()
        result=original(*a,**kw)
        torch.cuda.synchronize();elapsed=time.perf_counter()-t
        timings.setdefault(label,[]).append(elapsed)
        if first:print('PHASE',label,'end seconds',elapsed,flush=True)
        return result
    setattr(owner,name,call)
for owner,name,label in ((qp,'_build','assembly'),(qp,'_backend_solve','backend'),
        (runner.alg.physics_dynamics,'build_context','mechanics'),
        (runner.alg,'update','ppo_update_backward'),(env,'step','control')):
    wrap(owner,name,label)
from cupiqp import DenseSolver
for name in ('setup','update','solve'):wrap(DenseSolver,name,'cupiqp_'+name)
original=qp.solve
def checked(**kw):
    result=original(**kw)
    phase='ppo' if kw.get('differentiable',False) else 'rollout'
    observed[phase]=observed.get(phase,0)+result.stage.numel()
    observed[phase+'_certified']=observed.get(phase+'_certified',0)+int(result.differentiated_mask.sum())
    assert result.tau_safe.isfinite().all()
    assert (result.tau_safe.abs()<=qp.torque_limits+1e-5).all()
    assert ((result.tau_safe-kw['previous_torque']).abs()<=qp.cfg.torque_rate_limit_nm_s*kw['dt']+1e-5).all()
    return result
qp.solve=checked
report={}
try:
    for mode in ('every_substep','random_one_substep'):
        qp.cfg=replace(qp.cfg,qp_update_mode=mode)
        env.set_hard_pact_qp_enabled(False)
        with torch.inference_mode():env.reset()
        observed.clear();timings.clear();counts.clear();gradient_sums.clear()
        torch.cuda.reset_peak_memory_stats();t=time.perf_counter()
        print('PHASE learn',mode,'begin',flush=True)
        runner.learn(2)
        torch.cuda.synchronize()
        assert observed.get('rollout',0)==100*4*2*(4 if mode=='every_substep' else 1)
        assert observed.get('ppo',0)>0
        assert all(p.grad is None or p.grad.isfinite().all() for p in runner.alg.actor_critic.parameters())
        assert len(gradient_sums)==5 and all(v.isfinite() and v>0 for v in gradient_sums.values())
        report[mode]={'seconds':time.perf_counter()-t,'rows':dict(observed),
            'phases':{k:{'cold_s':v[0],'warm_mean_s':sum(v[1:])/max(len(v)-1,1),'calls':len(v)} for k,v in timings.items()},
            'gradient_abs_sums':{k:float(v) for k,v in gradient_sums.items()},
            'torch_peak_allocated_bytes':torch.cuda.max_memory_allocated()}
        print('MODE_PASS',mode,json.dumps(report[mode]),flush=True)
    print('QP_STALL_SMOKE_PASS',json.dumps(report),flush=True)
finally:
    faulthandler.cancel_dump_traceback_later()
    env.simulator._app_launcher.app.close()
