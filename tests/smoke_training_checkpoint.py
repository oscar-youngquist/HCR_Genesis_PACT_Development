"""Bounded registered HardPACT/Pos save, fresh-episode load, next-update smoke.

SIMULATOR=isaaclab PYTHONPATH=. python tests/smoke_training_checkpoint.py \
    --task go2_hard_pact_rss_full_isaaclab --gpu cuda:0 --num_envs 2 --headless
Only smoke-local sizing/activation overrides; no production config edits.
"""
import sys
from legged_gym.scripts.train_hard_pact import prepare_solver_runtime
prepare_solver_runtime(sys.argv[1:])
import json
from pathlib import Path
import tempfile
import torch
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
from rsl_rl.utils import training_checkpoint as tc

args = get_args()
cfg, train = task_registry.get_cfgs(args.task, args)
cfg.terrain.num_rows = cfg.terrain.num_cols = 2
cfg.terrain.max_init_terrain_level = 1
train.runner.resume = False
train.policy.pretrained_path = None
train.runner.num_steps_per_env = 4
train.algorithm.num_learning_epochs = train.algorithm.num_mini_batches = train.algorithm.num_encoder_epochs = 1
if hasattr(train.algorithm, 'ppo_qp_shard_percentage'):
    train.algorithm.ppo_qp_shard_percentage = 100.
if hasattr(train.algorithm, 'bard_batch_capacity'):
    train.algorithm.bard_batch_capacity = 32
if hasattr(train.algorithm, 'hard_pact_qp'):
    train.algorithm.hard_pact_qp.update(warmup_iterations=0)
root = Path(tempfile.mkdtemp(prefix='training_checkpoint_gpu_'))
env, _ = task_registry.make_env(args.task, args, env_cfg=cfg)
failed = False

def equal(a, b):
    if torch.is_tensor(a): assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a: equal(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b): equal(x, y)
    else: assert a == b

try:
    first, _ = task_registry.make_alg_runner(env, args.task, args, train_cfg=train, log_root=str(root/'source'))
    torch.cuda.reset_peak_memory_stats(args.gpu)
    first.learn(1)
    path = Path(first.log_dir)/'model_1.pt'
    saved = torch.load(path, map_location=first.device, weights_only=False)[tc.KEY]
    assert saved['next_iteration'] == 1 and saved['stage_end']
    if first.writer is not None: first.writer.close()
    # Reuse the physical scene, whose lazy Isaac Lab buffers were allocated in
    # inference-mode collection. Only its reset needs that context; NEVER
    # construct trainable networks/optimizers under inference_mode.
    reset = env.reset
    env.reset = torch.inference_mode()(reset)
    try:
        second, _ = task_registry.make_alg_runner(env, args.task, args, train_cfg=train, log_root=str(root/'resume'))
    finally:
        env.reset = reset
    second.load(path)
    for name, module in tc.modules(second).items(): equal(module.state_dict(), saved['models'][name])
    for name, optimizer in tc.optimizers(second).items(): equal(optimizer.state_dict(), saved['optimizers'][name])
    assert second.current_learning_iteration == 1 and second.alg.storage.step == 0
    assert not env.episode_length_buf.any() and torch.isfinite(env.obs_buf).all()
    second.learn(1)
    assert second.current_learning_iteration == 2
    assert (Path(second.log_dir)/'model_2.pt').is_file()
    assert all(torch.isfinite(p).all() for module in tc.modules(second).values() for p in module.parameters())
    print('TRAINING_CHECKPOINT_SMOKE_PASS '+json.dumps(dict(task=args.task,
        next_iteration=2, log_root=str(root),
        peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(args.gpu),
        memory_scope='PyTorch allocator only; two runner instances, excludes PhysX/Kit')), flush=True)
    if second.writer is not None: second.writer.close()
except BaseException:
    failed = True
    import traceback
    traceback.print_exc()
    print('TRAINING_CHECKPOINT_SMOKE_FAIL '+str(root), flush=True)
    raise
finally:
    try:
        env.simulator._app_launcher.app.close()
    except SystemExit:
        if failed: raise SystemExit(1)
