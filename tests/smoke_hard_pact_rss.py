"""Real registered RSS task: reset, four control steps, one owned PPO update.

Only smoke sizing/warmup overrides; production task configuration is unchanged.
Use SIMULATOR=isaaclab ... --task go2_hard_pact_rss_<variant>_isaaclab --gpu cuda:0 --headless --num_envs 4.
"""
import sys
from legged_gym.scripts.train_hard_pact import prepare_solver_runtime
prepare_solver_runtime(sys.argv[1:])
import json
import tempfile
import torch
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry

args = get_args()
cfg, train = task_registry.get_cfgs(args.task, args)
cfg.terrain.num_rows = cfg.terrain.num_cols = 2
cfg.terrain.max_init_terrain_level = 1
train.runner.resume = False
train.policy.pretrained_path = None
train.runner.num_steps_per_env = 4
train.algorithm.num_learning_epochs = train.algorithm.num_mini_batches = train.algorithm.num_encoder_epochs = 1
train.algorithm.ppo_qp_shard_percentage = 100.
train.algorithm.bard_batch_capacity = 32
train.algorithm.actor_physics_warmup_iterations = 0
train.algorithm.hard_pact_qp.update(warmup_iterations=0)
env, _ = task_registry.make_env(args.task, args, env_cfg=cfg)
try:
    runner, _ = task_registry.make_alg_runner(env, args.task, args, train_cfg=train,
        log_root=tempfile.mkdtemp(prefix='hard_pact_rss_smoke_'))
    actor, alg = runner.alg.actor_critic, runner.alg
    counts = {'controls': 0, 'rollout_rows': 0, 'ppo_rows': 0}
    grads = {}
    for label, weight in (('position', actor.act_pos_out.weight), ('ff', actor.act_tau_out.weight),
        ('encoder', actor.context_encoder.ce_out_mean.weight),
        ('grf', actor.physics_estimator.grf_head[-1].weight),
        ('wrench', actor.physics_estimator.wrench_head[-1].weight)):
        def record(g, label=label):
            assert torch.isfinite(g).all(), label
            grads[label] = grads.get(label, 0) + g.detach().abs().sum()
        weight.register_hook(record)
    step = env.step
    def checked_step(action):
        assert action.shape[-1] == actor.std.numel()
        result = step(action)
        counts['controls'] += 1
        for value in (action, result[0], result[4], result[-1]):
            assert torch.isfinite(value).all()
        assert (env._hard_pact_executed_torque.abs() <= env.simulator.torque_limits + 1e-5).all()
        if actor.std.numel() == 12:
            assert env._hard_pact_tau_ff.eq(0).all() and env.actions[:,12:].eq(0).all()
        return result
    env.step = checked_step
    if alg.hard_pact_qp is not None:
        solve = alg.hard_pact_qp.solve
        def checked_solve(**kw):
            phase = kw.get('diagnostics_phase', 'rollout')
            counts[phase + '_rows'] += kw['tau_nom'].shape[0]
            result = solve(**kw)
            assert torch.isfinite(result.tau_safe).all()
            return result
        alg.hard_pact_qp.solve = checked_solve
    runner.learn(1)
    assert not alg.bard_inverse_enabled and not alg.bard_rollout_enabled
    assert counts['controls'] == 4
    assert grads['position'] > 0 and grads['grf'] > 0 and grads['wrench'] > 0
    if actor.std.numel() == 12: assert grads.get('ff', 0) == 0
    if alg.hard_pact_features.execution_qp: assert counts['rollout_rows'] > 0 and counts['ppo_rows'] > 0
    else: assert counts['rollout_rows'] == counts['ppo_rows'] == 0
    print('RSS_SMOKE_PASS', json.dumps(dict(task=args.task, counts=counts,
        gradients={k: float(v) for k,v in grads.items()},
        actor_losses={k:float(v) for k,v in alg.last_actor_loss_metrics.items()},
        log_dir=runner.log_dir)), flush=True)
except BaseException:
    import traceback
    traceback.print_exc()
    print('RSS_SMOKE_FAIL', args.task, flush=True)
    raise
finally:
    env.simulator._app_launcher.app.close()
