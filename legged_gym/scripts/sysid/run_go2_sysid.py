from legged_gym.envs import *
from legged_gym.utils import get_args, init_genesis, task_registry, class_to_dict
import genesis as gs


args = get_args() # sysid in air
args.headless = True
init_genesis(args, gs)
env_cfg, train_cfg = task_registry.get_cfgs(name="go2_sysid")
env, env_cfg = task_registry.make_env(name="go2_sysid", args=args, env_cfg=env_cfg)
env.system_id_in_air(env_cfg)
