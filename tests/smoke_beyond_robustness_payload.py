"""Bounded REAL Isaac Lab Stage-3 validation, explicitly temporary tracking reward.

SIMULATOR=isaaclab PYTHONPATH=. python tests/smoke_beyond_robustness_payload.py --gpu cuda:0 --num-envs 4
No production launcher/checkpoint or paper-reward claim. Prints explicit PASS/FAIL
because Kit shutdown can mask a Python exception in the process exit code.
"""
import argparse
import json
import time
import traceback
import tempfile
import importlib.metadata
from pathlib import Path
from dataclasses import replace
import torch
from legged_gym.envs.go2.go2_hard_pact.go2_hard_pact_config import GO2HardPACTCfg, GO2HardPACTCfgPPO
from legged_gym.envs.go2.go2_hard_pact.ablation_configs import make_hard_pact_backend_configs
from legged_gym.utils.helpers import class_to_dict
from legged_gym.envs.go2.go2_beyond_robustness import IsaacLabBeyondRobustnessEnv
from rsl_rl.beyond_robustness_config import BeyondRobustnessConfig
from rsl_rl.beyond_robustness_payload import SlidingPayloadConfig
from rsl_rl.runners.beyond_robustness_runner import BeyondRobustnessRunner

parser=argparse.ArgumentParser()
parser.add_argument('--gpu',default='cuda:0'); parser.add_argument('--num-envs',type=int,default=4)
parser.add_argument('--steps',type=int,default=8)
parser.add_argument('--physical-checks',action='store_true')
args=parser.parse_args()
torch.manual_seed(8)
EnvCfg,TrainCfg=make_hard_pact_backend_configs('isaaclab')
cfg=EnvCfg(); cfg.env.num_envs=args.num_envs
cfg.terrain.mesh_type='trimesh'; cfg.terrain.num_rows=cfg.terrain.num_cols=2
cfg.terrain.max_init_terrain_level=0
cfg.sim.console_debug=False; cfg.sim.suppress_backend_warnings=False
cfg.viewer.debug_vis=False
if args.physical_checks:
    # Flat level-zero trimesh and nominal robot properties isolate the physical
    # checks. Production adapter leaves the supplied terrain/DR config intact.
    cfg.terrain.terrain_proportions=[1.]+[0.]*10
    cfg.terrain.curriculum=True; cfg.terrain.num_rows=1
    for name in dir(cfg.domain_rand):
        if name.startswith('randomize_'): setattr(cfg.domain_rand,name,False)
    cfg.domain_rand.push_robots=False; cfg.domain_rand.persistent_disturbance=False
# Isolate contact tests from randomized pushes; restore flags for runner smoke.
env=None
try:
    env=IsaacLabBeyondRobustnessEnv(cfg,class_to_dict(cfg.sim),args.gpu,True,reward_mode='smoke_tracking')
    packet=env.reset(); packet.validate(env.input_schema,args.num_envs,args.gpu)
    sim=env.simulator
    checks={}
    if args.physical_checks:
        from br_payload_physical_checks import physics_checks
        checks=physics_checks(env)
        print('BR_PHYSICS_CHECKS_PASS',json.dumps(checks),flush=True)
    for _ in range(args.steps):
        result=env.step(torch.zeros(args.num_envs,12,device=args.gpu))
        assert torch.isfinite(torch.cat(result.observation.inputs(),-1)).all()
    train=BeyondRobustnessConfig.from_hardpact(cfg,TrainCfg(),input_schema=env.input_schema,
        num_steps_per_env=args.steps,num_learning_epochs=1,num_mini_batches=1,auxiliary_epochs=1)
    runner=BeyondRobustnessRunner(env,train,device=args.gpu)
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); start=time.perf_counter()
    logs=runner.learn(1)
    runner.set_training_phase('student_reinforcement'); logs+=runner.learn(1)
    torch.cuda.synchronize(); elapsed=time.perf_counter()-start
    assert all(torch.isfinite(p).all() for p in runner.alg.actor_critic.parameters())
    assert all(torch.isfinite(p.grad).all() for p in runner.alg.actor_critic.parameters() if p.grad is not None)
    with tempfile.TemporaryDirectory(prefix='br_stage3_') as directory:
        path=Path(directory)/'checkpoint.pt'
        runner.save(path)
        masses=sim.payload_mass.clone()
        phase=runner.alg.training_phase; iteration=runner.alg.iteration
        runner.load(path,resume_mode='fresh')
        assert runner.alg.iteration==iteration and runner.alg.training_phase==phase
        torch.testing.assert_close(sim.payload_mass,masses,rtol=0,atol=0)
        assert runner.alg.history.frames.eq(0).all()
    print('BR_PAYLOAD_SMOKE_PASS',json.dumps(dict(num_envs=args.num_envs,seconds=elapsed,
        metadata=dict(isaaclab=importlib.metadata.version('isaaclab'),isaacsim=importlib.metadata.version('isaacsim'),
            torch=torch.__version__,gpu=torch.cuda.get_device_name(),cuda=torch.version.cuda),
        transitions_per_second=2*args.steps*args.num_envs/elapsed,
        torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        load_mass=sim.payload_mass.tolist(),box_size=sim.box_size,
        physical_checks=checks,schema=env.input_schema.contract(),logs=logs)),flush=True)
except BaseException:
    traceback.print_exc(); print('BR_PAYLOAD_SMOKE_FAIL',flush=True); raise
finally:
    if env is not None: env.close()
