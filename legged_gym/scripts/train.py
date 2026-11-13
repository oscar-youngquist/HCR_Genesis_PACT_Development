import os

from legged_gym import *
from legged_gym.envs import *
from legged_gym.utils import (
    get_args, init_genesis, save_hard_pact_resolved_config, task_registry,
)
import shutil
import json

def train(args):
    if "genesis" in SIMULATOR:
        init_genesis(args, gs)
    # Make environment and algorithm runner
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    
    # Copy env.py and env_config.py to log_dir for backup
    log_dir = ppo_runner.log_dir
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if args.task.startswith("go2_hard_pact_"):
        # Ablation task names intentionally have no copied environment files;
        # archive the one shared implementation/config instead.
        robot_file_path = os.path.join(LEGGED_GYM_ROOT_DIR, "legged_gym", "envs", "go2", "go2_hard_pact", "go2_hard_pact.py")
        robot_config_path = os.path.join(LEGGED_GYM_ROOT_DIR, "legged_gym", "envs", "go2", "go2_hard_pact", "go2_hard_pact_config.py")
    elif env_cfg.asset.name == args.task:
        robot_file_path = os.path.join(LEGGED_GYM_ROOT_DIR, "legged_gym", "envs", env_cfg.asset.name, args.task+".py")
        robot_config_path = os.path.join(LEGGED_GYM_ROOT_DIR, "legged_gym", "envs", env_cfg.asset.name, args.task+"_config.py")
    else:
        robot_file_path = os.path.join(LEGGED_GYM_ROOT_DIR, "legged_gym", "envs", env_cfg.asset.name, args.task, args.task+".py")
        robot_config_path = os.path.join(LEGGED_GYM_ROOT_DIR, "legged_gym", "envs", env_cfg.asset.name, args.task, args.task+"_config.py")
    
    shutil.copy(robot_file_path, log_dir)
    shutil.copy(robot_config_path, log_dir)
    # HardPACT registrations use thin dynamic backend/ablation subclasses.
    # Persist one authoritative resolved snapshot so inherited PACT values,
    # command-line overrides, and the selected feature matrix are all present
    # without copying a chain of partially overlapping Python source files.
    save_hard_pact_resolved_config(
        log_dir, args.task, env_cfg, train_cfg,
        effective_train_cfg=getattr(ppo_runner, "train_cfg", None),
    )
    if hasattr(env.simulator, "grf_semantics"):
        with open(os.path.join(log_dir, "grf_semantics.json"), "w") as stream:
            json.dump(env.simulator.grf_semantics, stream, indent=2)
    
    # Start training session
    iterations = train_cfg.runner.max_iterations
    if args.task.startswith('go2_hard_pact'):
        additional = getattr(args, 'additional_iterations', None)
        if additional is not None:
            if additional < 0 or not (getattr(args,'state_checkpoint',None) or args.resume):
                raise ValueError('--additional-iterations must be nonnegative and requires a checkpoint')
            if args.max_iterations is not None: raise ValueError('Choose total --max_iterations OR --additional-iterations')
            iterations = additional
        else:
            iterations = max(0, iterations - ppo_runner.current_learning_iteration)
    ppo_runner.learn(num_learning_iterations=iterations, init_at_random_ep_len=True)

if __name__ == '__main__':
    args = get_args()
    if args.debug:
        args.num_envs = 1
    train(args)
