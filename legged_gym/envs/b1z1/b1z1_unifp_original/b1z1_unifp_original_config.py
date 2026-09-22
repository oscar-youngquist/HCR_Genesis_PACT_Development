"""Copy nested sections so the retained task's configuration cannot be mutated."""
from copy import deepcopy
from types import SimpleNamespace
from legged_gym.envs.b1z1.b1z1_unifp.b1z1_unifp_config import B1Z1UniFPCfg, B1Z1UniFPCfgPPO


def copy_sections(source):
    values = {}
    for name in dir(source):
        if name.startswith("_"):
            continue
        value = getattr(source, name)
        if isinstance(value, type):
            values[name] = copy_sections(value)
        elif not callable(value):
            values[name] = deepcopy(value)
    return SimpleNamespace(**values)


class B1Z1UniFPOriginalCfg:
    def __init__(self):
        self.__dict__.update(vars(copy_sections(B1Z1UniFPCfg)))
        self.env.num_obs_hist = 32
        self.env.num_priv_stack = 3
        self.env.num_pred_obs = 12
        self.commands.use_external_impedance_compensation = False


class B1Z1UniFPOriginalCfgPPO:
    def __init__(self):
        self.__dict__.update(vars(copy_sections(B1Z1UniFPCfgPPO)))
        self.runner_class_name = "UniFPOriginalRunner"
        self.policy = SimpleNamespace()  # Deliberately fixed upstream architecture.
        allowed = {"learning_rate", "clip_param", "gamma", "lam", "value_loss_coef",
                   "entropy_coef", "max_grad_norm", "use_clipped_value_loss", "desired_kl",
                   "schedule", "num_learning_epochs", "num_mini_batches", "num_encoder_epochs",
                   "use_adaptive_entropy"}
        self.algorithm = SimpleNamespace(**{k: v for k, v in vars(self.algorithm).items()
                                           if k in allowed or k.startswith("adaptive_ent_")})
        self.runner.policy_class_name = "ActorCriticUniFPOriginal"
        self.runner.algorithm_class_name = "PPO_UniFPOriginal"
        self.runner.experiment_name = "b1z1_unifp_original"
        self.runner.run_name = "upstream_architecture"
        self.runner.resume = False
        self.runner.load_run = -1
        self.runner.enable_additional_diagnostics = False
