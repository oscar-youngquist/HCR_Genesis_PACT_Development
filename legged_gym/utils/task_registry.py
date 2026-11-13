import os
from datetime import datetime
from typing import Tuple
import torch
import numpy as np

from rsl_rl.env import VecEnv
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.utils.runner_registry import runner_registry

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from .helpers import get_args, update_cfg_from_args, class_to_dict, get_load_path, get_load_path_ee, set_seed, configure_runtime_device

class TaskRegistry():
    def __init__(self):
        self.task_classes = {}
        self.env_cfgs = {}
        self.train_cfgs = {}
        self.resolved_factories = {}

    def register_resolved(self, name, factory, task_class, runner_class):
        """Opt-in flat-config adapters; legacy task construction remains unchanged."""
        self.resolved_factories[name]=(factory,runner_class)
        self.task_classes[name]=task_class
    
    def register(self, name: str, task_class: VecEnv, env_cfg, train_cfg):
        self.task_classes[name] = task_class
        self.env_cfgs[name] = env_cfg
        self.train_cfgs[name] = train_cfg
    
    def get_task_class(self, name: str) -> VecEnv:
        return self.task_classes[name]
    
    def get_cfgs(self, name, args):
        if name in self.resolved_factories:
            resolved=self.resolved_factories[name][0]()
            return resolved,resolved.training
        train_cfg = self.train_cfgs[name]
        env_cfg = self.env_cfgs[name]
        
        # copy seed from args
        train_cfg.seed = args.seed
        env_cfg.seed = args.seed

        return env_cfg, train_cfg
    
    def make_env(self, name, args=None, env_cfg=None):
        """ Creates an environment either from a registered namme or from the provided config file.

        Args:
            name (string): Name of a registered env.
            args (Args, optional): Isaac Gym comand line arguments. If None get_args() will be called. Defaults to None.
            env_cfg (Dict, optional): Environment config file used to override the registered config. Defaults to None.

        Raises:
            ValueError: Error if no registered env corresponds to 'name' 

        Returns:
            isaacgym.VecTaskPython: The created environment
            Dict: the corresponding config file
        """
        if getattr(args, 'load_operation', None) == 'stage_transfer':
            hooks = ('validate_stage_transfer','import_training_progress','initialize_transferred_payload')
            if any(not hasattr(self.task_classes[name], h) for h in hooks):
                raise ValueError('Selected task has no validated cross-payload stage-transfer adapter; liquid integration is not implemented')
        if name in self.resolved_factories:
            if args is None or not hasattr(args,'device'):
                raise ValueError('Resolved BR tasks require train_beyond_robustness.py arguments')
            resolved=env_cfg if env_cfg is not None else self.resolved_factories[name][0]()
            env=self.task_classes[name](resolved,device=args.device,headless=args.headless)
            return env,resolved
        # if no args passed get command line arguments
        if args is None:
            args = get_args()
        else:
            configure_runtime_device(args)
        # check if there is a registered env with that name
        if name in self.task_classes:
            task_class = self.get_task_class(name)
        else:
            raise ValueError(f"Task with name: {name} was not registered")
        if env_cfg is None:
            # load config files
            env_cfg, _ = self.get_cfgs(name, args)
        # override cfg from args (if specified)
        env_cfg, _ = update_cfg_from_args(env_cfg, None, args)
        set_seed(env_cfg.seed)
        # parse sim params (convert to dict first)
        sim_device = "cpu" if args.cpu else args.gpu
        # sim_params
        sim_params = class_to_dict(env_cfg.sim)
        env = task_class(   cfg=env_cfg,
                            sim_params=sim_params,
                            sim_device=sim_device,
                            headless=args.headless)
        return env, env_cfg

    def make_alg_runner(self, env, name=None, args=None, train_cfg=None, log_root="default"):
        """ Creates the training algorithm  either from a registered namme or from the provided config file.

        Args:
            env (isaacgym.VecTaskPython): The environment to train (TODO: remove from within the algorithm)
            name (string, optional): Name of a registered env. If None, the config file will be used instead. Defaults to None.
            args (Args, optional): Isaac Gym comand line arguments. If None get_args() will be called. Defaults to None.
            train_cfg (Dict, optional): Training config file. If None 'name' will be used to get the config file. Defaults to None.
            log_root (str, optional): Logging directory for Tensorboard. Set to 'None' to avoid logging (at test time for example). 
                                      Logs will be saved in <log_root>/<date_time>_<run_name>. Defaults to "default"=<path_to_LEGGED_GYM>/logs/<experiment_name>.

        Raises:
            ValueError: Error if neither 'name' or 'train_cfg' are provided
            Warning: If both 'name' or 'train_cfg' are provided 'name' is ignored

        Returns:
            PPO: The created algorithm
            Dict: the corresponding config file
        """
        if name in self.resolved_factories:
            cfg=train_cfg if train_cfg is not None else env.resolved_config.training
            if log_root=='default': raise ValueError('Resolved tasks require an explicit log directory')
            runner=self.resolved_factories[name][1](env,cfg,log_dir=log_root,device=env.device)
            return runner,cfg
        # if no args passed get command line arguments
        if args is None:
            args = get_args()
        else:
            configure_runtime_device(args)
        # if config files are passed use them, otherwise load from the name
        if train_cfg is None:
            if name is None:
                raise ValueError("Either 'name' or 'train_cfg' must be not None")
            # load config files
            _, train_cfg = self.get_cfgs(name, args)
        else:
            if name is not None:
                print(f"'train_cfg' provided -> Ignoring 'name={name}'")
        # override cfg from args (if specified)
        _, train_cfg = update_cfg_from_args(None, train_cfg, args)

        state_path = getattr(args, 'state_checkpoint', None)
        if not state_path and (getattr(args,'load_operation','resume')!='resume'
                or getattr(args,'target_stage',None) or getattr(args,'finetune_learning_rate',None) is not None):
            raise ValueError('Explicit load operations and overrides require --state-checkpoint')
        state_options = state_path or getattr(args, 'additional_iterations', None) is not None
        if state_options and not (name or '').startswith('go2_hard_pact'):
            raise ValueError('Complete-state arguments require a HardPACT task (BR uses its dedicated entrypoint)')
        if state_path:
            if args.resume: raise ValueError('Choose --state-checkpoint or legacy --resume, not both')
            train_cfg.runner.resume = False
            # Explicit checkpoint selection takes precedence over bootstrap paths.
            train_cfg.policy.pretrained_path = None
        if getattr(args, 'checkpoint_log_root', None): log_root = args.checkpoint_log_root

        if log_root=="default":
            log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', 'hardpact_iclr', train_cfg.runner.experiment_name)
            # log_root = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name)
            log_dir = os.path.join(log_root, datetime.now().strftime('%b%d_%H-%M-%S') + '_' + train_cfg.runner.run_name)
        elif log_root is None:
            log_dir = None
        else:
            log_dir = os.path.join(log_root, datetime.now().strftime('%b%d_%H-%M-%S') + '_' + train_cfg.runner.run_name)
        if state_path and log_dir is not None and os.path.realpath(log_dir)==os.path.dirname(os.path.realpath(state_path)):
            raise ValueError('Destination log directory must differ from source run')
        
        train_cfg_dict = class_to_dict(train_cfg)
        sim_device = "cpu" if args.cpu else args.gpu

        # Pos retains its existing config-only behavior. HardPACT's parser
        # returns None when omitted, preserving the resolved config (including
        # benchmark overrides); explicit CLI values, including zero, still win.
        if "pact" in name and "pos" not in name and args.pinn_loss_weight is not None:
            train_cfg_dict["policy"]["pinn_loss_weight"] = args.pinn_loss_weight

        # select runner according to runner_class_name
        runner_class = runner_registry.get_runner_class(train_cfg.runner_class_name)
        runner = runner_class(env, train_cfg_dict, log_dir, device=sim_device)
        if state_path:
            runner.load(state_path, operation=args.load_operation,
                learning_rate=args.finetune_learning_rate, allow_legacy=args.allow_legacy_checkpoint,
                target_stage=args.target_stage, stage_local_initial=args.stage_local_initial)
        #save resume path before creating a new log_dir
        resume = train_cfg.runner.resume
        if resume:
            if train_cfg.runner_class_name == "OnPolicyRunnerEE":
                model_resume_path, estimator_resume_path = get_load_path_ee(log_root, load_run=train_cfg.runner.load_run, checkpoint=train_cfg.runner.checkpoint)
                print(f"Loading model from: {model_resume_path}")
                print(f'Loading estimator from: {estimator_resume_path}')
                runner.load(model_resume_path, estimator_resume_path)
            else:
                # load previously trained model
                resume_path = get_load_path(log_root, load_run=train_cfg.runner.load_run, checkpoint=train_cfg.runner.checkpoint)
                print(f"Loading model from: {resume_path}")
                if (name or '').startswith('go2_hard_pact'):
                    runner.load(resume_path, allow_legacy=getattr(args,'allow_legacy_checkpoint',False))
                else:
                    runner.load(resume_path)
        return runner, train_cfg

# make global task registry
task_registry = TaskRegistry()
