import os
import copy
import torch
import numpy as np
import random
import argparse

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR


def _normalize_gpu_arg(gpu):
    gpu = str(gpu).strip().lower()
    if gpu.isdigit():
        return f"cuda:{gpu}"
    if gpu == "cuda":
        return gpu
    if gpu.startswith("cuda:"):
        index = gpu.split(":", 1)[1]
        if index.isdigit():
            return f"cuda:{index}"
    raise ValueError(
        f"Unsupported GPU specifier '{gpu}'. Use values like 'cuda', 'cuda:0', or '1'."
    )


def configure_runtime_device(args):
    """Normalize GPU selection and, when needed, mask visibility to the requested physical GPU.

    Genesis and some CUDA codepaths may still resolve work onto the process-local `cuda:0`.
    When a specific physical GPU is requested, we remap visibility so that local `cuda:0`
    corresponds to the requested GPU.
    """
    if getattr(args, "cpu", False):
        if hasattr(args, "gpu"):
            args.gpu = "cpu"
        if hasattr(args, "device"):
            args.device = "cpu"
        if hasattr(args, "requested_gpu"):
            args.requested_gpu = "cpu"
        return args

    requested_gpu = getattr(args, "requested_gpu", None)
    if requested_gpu is None:
        requested_gpu = getattr(args, "gpu", None)
    if requested_gpu is None:
        requested_gpu = getattr(args, "device", "cuda:0")

    requested_gpu = _normalize_gpu_arg(requested_gpu)
    runtime_gpu = requested_gpu

    if requested_gpu.startswith("cuda:"):
        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible_devices:
            visible_gpu_ids = [gpu_id.strip() for gpu_id in visible_devices.split(",") if gpu_id.strip()]
            local_index = int(requested_gpu.split(":", 1)[1])
            # If CUDA_VISIBLE_DEVICES is already set before Python starts, treat cuda:N
            # as a process-local device index.
            if 0 <= local_index < len(visible_gpu_ids):
                runtime_gpu = f"cuda:{local_index}"
            else:
                physical_index = str(local_index)
                if physical_index not in visible_gpu_ids:
                    raise ValueError(
                        f"Requested GPU '{requested_gpu}' is not available under "
                        f"CUDA_VISIBLE_DEVICES={visible_devices}."
                    )
                runtime_gpu = f"cuda:{visible_gpu_ids.index(physical_index)}"
        else:
            physical_index = requested_gpu.split(":", 1)[1]
            os.environ["CUDA_VISIBLE_DEVICES"] = physical_index
            runtime_gpu = "cuda:0"

    elif requested_gpu == "cuda" and os.environ.get("CUDA_VISIBLE_DEVICES"):
        runtime_gpu = "cuda:0"

    args.requested_gpu = requested_gpu
    args.gpu = runtime_gpu
    if hasattr(args, "device"):
        args.device = runtime_gpu
    return args


def init_genesis(args, gs):
    """Initialize Genesis after device selection has been normalized."""
    configure_runtime_device(args)
    gs.init(backend=gs.cpu if args.cpu else gs.gpu, logging_level="warning")
    if not args.cpu and args.gpu.startswith("cuda"):
        torch.cuda.set_device(torch.device(args.gpu))

def class_to_dict(obj) -> dict:
    if not hasattr(obj,"__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result

def update_class_from_dict(obj, dict):
    for key, val in dict.items():
        attr = getattr(obj, key, None)
        if isinstance(attr, type):
            update_class_from_dict(attr, val)
        else:
            setattr(obj, key, val)
    return

def set_seed(seed):
    if seed == -1:
        seed = np.random.randint(0, 10000)
    print("Setting seed: {}".format(seed))
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_load_path(root, load_run=-1, checkpoint=-1):
    try:
        runs = os.listdir(root)
        #TODO sort by date to handle change of month
        runs.sort()
        if 'exported' in runs: runs.remove('exported')
        last_run = os.path.join(root, runs[-1])
    except:
        raise ValueError("No runs in this directory: " + root)
    if load_run==-1:
        load_run = last_run
    else:
        load_run = os.path.join(root, load_run)

    if checkpoint==-1:
        models = [file for file in os.listdir(load_run) if 'model' in file]
        models.sort(key=lambda m: '{0:0>15}'.format(m))
        model = models[-1]
    else:
        model = "model_{}.pt".format(checkpoint) 

    load_path = os.path.join(load_run, model)
    return load_path

def get_load_path_ee(root, load_run=-1, checkpoint=-1):
    try:
        runs = os.listdir(root)
        #TODO sort by date to handle change of month
        runs.sort()
        if 'exported' in runs: runs.remove('exported')
        last_run = os.path.join(root, runs[-1])
    except:
        raise ValueError("No runs in this directory: " + root)
    if load_run==-1:
        load_run = last_run
    else:
        load_run = os.path.join(root, load_run)

    if checkpoint==-1:
        models = [file for file in os.listdir(load_run) if 'model' in file]
        models.sort(key=lambda m: '{0:0>15}'.format(m))
        model = models[-1]
        # estimator
        estimators = [file for file in os.listdir(load_run) if 'estimator' in file]
        estimators.sort(key=lambda m: '{0:0>15}'.format(m))
        estimator = estimators[-1]
    else:
        model = "model_{}.pt".format(checkpoint)
        estimator = "estimator_{}.pt".format(checkpoint)

    actor_load_path = os.path.join(load_run, model)
    estimator_load_path = os.path.join(load_run, estimator)
    return actor_load_path, estimator_load_path

def update_cfg_from_args(env_cfg, cfg_train, args):
    """Override some configuration parameters from command line arguments
       Called in task_registry.py/make_env()

    Args:
        env_cfg : environment configuration
        cfg_train : training configuration
        args : command line arguments

    Returns:
        env_cfg : updated environment configuration
        cfg_train : updated training configuration
    """
    # environment parameters
    if env_cfg is not None:
        # num envs
        if args.num_envs is not None:
            env_cfg.env.num_envs = args.num_envs
        if args.debug:
            env_cfg.env.debug = args.debug
    # training parameters
    if cfg_train is not None:
        # alg runner parameters
        if args.max_iterations is not None:
            cfg_train.runner.max_iterations = args.max_iterations
        if args.resume:
            cfg_train.runner.resume = args.resume
        if args.sync_wandb:
            cfg_train.runner.sync_wandb = args.sync_wandb
        # if args.ckpt is not None:
        #     cfg_train.runner.checkpoint = args.ckpt
        if args.load_run is not None:
            cfg_train.runner.load_run = args.load_run
        # Optional HardPACT benchmark overrides are deliberately applied only
        # when the selected algorithm exposes the shared QP dictionary. They
        # therefore have no effect on legacy tasks or their configuration.
        qp_cfg = getattr(cfg_train.algorithm, "hard_pact_qp", None)
        if qp_cfg is not None:
            if getattr(args, "profile_bard_timing", False):
                cfg_train.algorithm.profile_bard_timing = True
            if getattr(args, "bard_batch_capacity", None) is not None:
                cfg_train.algorithm.bard_batch_capacity = args.bard_batch_capacity
            if getattr(args, "benchmark_bard_active", False):
                # Benchmark-only: execute both configured BARD losses from
                # iteration zero at full weight without changing defaults.
                cfg_train.policy.pinn_init_steps = -1
                cfg_train.policy.pinn_loss_weight = -1.0
            if getattr(args, "qp_solver", None) is not None:
                qp_cfg["qp_solver"] = args.qp_solver
                qp_cfg["rollout_qp_solver"] = None
                qp_cfg["ppo_qp_solver"] = None
            if getattr(args, "qp_solver_dtype", None) is not None:
                qp_cfg["solver_dtype"] = args.qp_solver_dtype
            if getattr(args, "qp_rollout_chunk_size", None) is not None:
                qp_cfg["rollout_chunk_size"] = args.qp_rollout_chunk_size
            if getattr(args, "qp_ppo_chunk_size", None) is not None:
                qp_cfg["ppo_chunk_size"] = args.qp_ppo_chunk_size

    return env_cfg, cfg_train

def get_args():
    """Parse command line arguments

    Returns:
        args: parsed command line arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--task',           type=str, default='go2', help="task name")
    parser.add_argument('--headless',       action='store_true', default=False, help="enable visualization by default")
    parser.add_argument('--cpu',            action='store_true', default=False, help="use CPU instead of CUDA")
    parser.add_argument('--gpu',            type=str, default='cuda:0', help="which GPU to use (default: cuda:0)")
    parser.add_argument('--num_envs',       type=int, default=None, help="number of parallel environments")
    parser.add_argument('--max_iterations', type=int, default=None, help="max number of training iterations")
    parser.add_argument('--resume',         action='store_true', default=False, help="resume training from specified checkpoint")
    parser.add_argument('--sync_wandb',     action='store_true', default=False, help="synchronize training log with wandb")
    parser.add_argument('--export_onnx',    action='store_true', default=False, help="export policy as onnx (besides jit)")
    parser.add_argument('--debug',          action='store_true', default=False, help="enable debug mode")
    parser.add_argument('--load_run',       type=str, default=None, help="run to load, default: last run")
    parser.add_argument('--ckpt',           type=int, default=-1, help="checkpoint to load, -1 means latest")
    parser.add_argument('--use_joystick',   action='store_true', default=False, help="use joystick to provide commands")
    parser.add_argument('--joystick_type',  type=str, default='xbox', help="type of joystick: xbox, switch")
    parser.add_argument('--follow_robot',   action='store_true', default=False, help="whether the camera follows the robot during play")
    parser.add_argument('--record_frames',   action='store_true', default=False, help="whether to record the camera")

    parser.add_argument('--seed',       type=int, default=1, help="int seed for random sampling (default 1)")

    # PACT PINN specific thing.
    parser.add_argument('--pinn_loss_weight',       type=float, default=0.01, help="float for weight of PINN loss (default 0.01)")
    parser.add_argument(
        '--qp_solver', choices=('qpth', 'cupiqp', 'moreau'), default=None,
        help='HardPACT-only QP backend override (legacy tasks ignore it)',
    )
    parser.add_argument(
        '--qp_solver_dtype', choices=('auto', 'float32', 'float64'), default=None,
        help='HardPACT-only solver precision override',
    )
    parser.add_argument(
        '--qp_rollout_chunk_size', type=int, default=None,
        help='HardPACT-only rollout QP chunk size override',
    )
    parser.add_argument(
        '--qp_ppo_chunk_size', type=int, default=None,
        help='HardPACT-only differentiable PPO QP chunk size override',
    )
    parser.add_argument(
        '--profile_bard_timing', action='store_true', default=False,
        help='HardPACT-only CUDA-event timing for inverse/rollout PINN losses',
    )
    parser.add_argument(
        '--benchmark_bard_active', action='store_true', default=False,
        help='HardPACT benchmark-only: activate BARD losses from iteration zero',
    )
    parser.add_argument(
        '--bard_batch_capacity', type=int, default=None,
        help='HardPACT-only BARD streaming workspace capacity',
    )

    return configure_runtime_device(parser.parse_args())

# def export_policy_as_jit(actor_critic, path, prefix=None):
#     if hasattr(actor_critic, 'memory_a'):
#         exporter = PolicyExporterLSTM(actor_critic)
#         exporter.export(path)
#     else: 
#         os.makedirs(path, exist_ok=True)
#         filename = prefix + "_policy.pt" if prefix != None else "policy.pt"
#         path = os.path.join(path, filename)
#         model = copy.deepcopy(actor_critic.actor).to('cpu')
#         traced_script_module = torch.jit.script(model)
#         traced_script_module.save(path)

class PolicyExporter(torch.nn.Module):
    def __init__(self, actor_critic):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor)
    
    def forward(self, obs):
        return self.actor(obs)
    
    def export(self, path, env_cfg, export_onnx=False, train_cfg=None):
        os.makedirs(path, exist_ok=True)
        filename = train_cfg.runner.load_run + "_ite" + str(train_cfg.runner.checkpoint) + ".pt"
        path_pt = os.path.join(path, filename)
        self.to('cpu')
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path_pt)
        
        # export onnx model if needed
        if export_onnx:
            filename = train_cfg.runner.load_run + "_ite" + str(train_cfg.runner.checkpoint) + ".onnx"
            path_onnx = os.path.join(path, filename)
            input_names = ["nn_input"]
            output_names = ["nn_output"]
            dummy_input = torch.randn(1, env_cfg.env.num_observations)
            torch.onnx.export(self, dummy_input, path_onnx, 
                              verbose=True, 
                              export_params=True,
                              input_names=input_names,
                              output_names=output_names,
                              opset_version=11)

class PolicyExporterTS(torch.nn.Module):
    """Policy exporter for teacher student policies

    Attention: This module is consistent with ActorCriticTS in rsl_rl/modules/actor_critic_ts.py
               When ActorCriticTS is updated, please remember to update this module accordingly.
    """
    def __init__(self, actor_critic):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor)
        self.encoder = copy.deepcopy(actor_critic.history_encoder)
    
    def forward(self, obs, history):
        latent = self.encoder(history)
        x = torch.cat([obs, latent], dim=-1)
        return self.actor(x)
 
    def export(self, path, env_cfg, export_onnx=False, train_cfg=None):
        os.makedirs(path, exist_ok=True)
        filename = train_cfg.runner.load_run + "_ite" + str(train_cfg.runner.checkpoint) + ".pt"
        path = os.path.join(path, filename)
        self.to('cpu')
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)
        
        # export onnx model if needed
        if export_onnx:
            filename = train_cfg.runner.load_run + "_ite" + str(train_cfg.runner.checkpoint) + ".onnx"
            path_onnx = os.path.join(path, filename)
            input_names = ["obs_input", "obs_history_input"]
            output_names = ["nn_output"]
            dummy_obs = torch.randn(1, env_cfg.env.num_observations)
            dummy_history = torch.randn(1, env_cfg.env.num_history_obs)
            torch.onnx.export(self, (dummy_obs, dummy_history), path_onnx, 
                              verbose=True, 
                              export_params=True,
                              input_names=input_names,
                              output_names=output_names,
                              opset_version=11)

class PolicyExporterEE(torch.nn.Module):
    """Policy exporter for explicit estimator policies

    Attention: This module is consistent with ActorCriticEE in rsl_rl/modules/actor_critic_ee.py
               When ActorCriticEE is updated, please remember to update this module accordingly.
    """
    def __init__(self, actor_critic):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor)
        self.estimator = copy.deepcopy(actor_critic.estimator)
    
    def forward(self, obs_history):
        estimated_state = self.estimator(obs_history)
        x = torch.cat([obs_history, estimated_state], dim=-1)
        return self.actor(x)
 
    def export(self, path, env_cfg, export_onnx=False, train_cfg=None):
        os.makedirs(path, exist_ok=True)
        filename = train_cfg.runner.load_run + "_ite" + str(train_cfg.runner.checkpoint) + ".pt"
        pt_path = os.path.join(path, filename)
        self.to('cpu')
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(pt_path)
        
        # export onnx model if needed
        if export_onnx:
            filename = train_cfg.runner.load_run + "_ite" + str(train_cfg.runner.checkpoint) + ".onnx"
            onnx_path = os.path.join(path, filename)
            input_names = ["nn_input"]
            output_names = ["nn_output"]
            dummy_input = torch.randn(1, env_cfg.env.num_estimator_features)
            torch.onnx.export(self, dummy_input, onnx_path, 
                              verbose=True, 
                              export_params=True,
                              input_names=input_names,
                              output_names=output_names,
                              opset_version=11)

class PolicyExporterWaQ(torch.nn.Module):
    """Policy exporter for DreamWaQ policies
    
    Attention: This module is consistent with ActorCriticDreamWaQ in rsl_rl/modules/actor_critic_dreamwaq.py
               When ActorCriticDreamWaQ is updated, please remember to update this module accordingly.
    """
    def __init__(self, actor_critic):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor)
        self.vae = copy.deepcopy(actor_critic.vae)
    
    def forward(self, obs, obs_history):
        vae_out = self.vae.inference(obs_history)
        x = torch.cat([obs, vae_out], dim=-1)
        return self.actor(x)
 
    def export(self, path, env_cfg, export_onnx=False, train_cfg=None):
        os.makedirs(path, exist_ok=True)
        filename = train_cfg.runner.load_run + "_ite" + str(train_cfg.runner.checkpoint) + ".pt"
        path = os.path.join(path, filename)
        self.to('cpu')
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)
        
        # export onnx model if needed
        if export_onnx:
            filename = train_cfg.runner.load_run + "_ite" + str(train_cfg.runner.checkpoint) + ".onnx"
            path_onnx = os.path.join(path, filename)
            input_names = ["obs_input", "obs_history_input"]
            output_names = ["nn_output"]
            dummy_obs = torch.randn(1, env_cfg.env.num_observations)
            dummy_history = torch.randn(1, env_cfg.env.num_history_obs)
            torch.onnx.export(self, (dummy_obs, dummy_history), path_onnx, 
                              verbose=True, 
                              export_params=True,
                              input_names=input_names,
                              output_names=output_names,
                              opset_version=11)
            
class PolicyExporterPACT():
    def __init__(self, actor_critic):
        self.actor = actor_critic

    def export(self, path, env_cfg, train_cfg=None):
        os.makedirs(path, exist_ok=True)
        filename = train_cfg.runner.load_run + "_ite" + str(train_cfg.runner.checkpoint) + ".pt"
        path = os.path.join(path, filename)
        self.actor.to('cpu')
        traced_script_module = torch.jit.script(self.actor)
        traced_script_module.save(path)


class PolicyExporterLSTM(torch.nn.Module):
    def __init__(self, actor_critic):
        super().__init__()
        self.actor = copy.deepcopy(actor_critic.actor)
        self.is_recurrent = actor_critic.is_recurrent
        self.memory = copy.deepcopy(actor_critic.memory_a.rnn)
        self.memory.cpu()
        self.register_buffer(f'hidden_state', torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size))
        self.register_buffer(f'cell_state', torch.zeros(self.memory.num_layers, 1, self.memory.hidden_size))

    def forward(self, x):
        out, (h, c) = self.memory(x.unsqueeze(0), (self.hidden_state, self.cell_state))
        self.hidden_state[:] = h
        self.cell_state[:] = c
        return self.actor(out.squeeze(0))

    @torch.jit.export
    def reset_memory(self):
        self.hidden_state[:] = 0.
        self.cell_state[:] = 0.
 
    def export(self, path):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, 'policy_lstm_1.pt')
        self.to('cpu')
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)

    
