"""Simulator-neutral BR Stage-1 contract; no HardPACT objectives are imported.

Integration (the caller supplies its already resolved HardPACT configs)::

    cfg = BeyondRobustnessConfig.from_hardpact(env_cfg, train_cfg)
    model = ActorCriticBeyondRobustness(cfg)
    alg = PPOBeyondRobustness(model, cfg, device="cpu")

Use alg.act(o_t, s_t, p_t, load_t) before stepping, then process_env_step,
compute_returns and update. Switch training_phase explicitly at an update boundary;
it is not a simulator curriculum stage. Defaults below match f79eb0f; the factory
inherits live values. No task registration, reward, box physics or launcher is added.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass


PHASES = ("teacher_student", "student_reinforcement")


@dataclass(frozen=True)
class BeyondRobustnessConfig:
    # H means exactly H frames, oldest first, including the current observation.
    history_length: int = 20
    training_phase: str = "teacher_student"
    num_envs: int = 4096
    num_steps_per_env: int = 24
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    auxiliary_epochs: int = 5
    auxiliary_learning_rate: float = 2e-4
    learning_rate: float = 3e-4
    clip_param: float = 0.2
    gamma: float = 0.99
    lam: float = 0.95
    value_loss_coef: float = 1.0
    entropy_coef: float = 0.01
    max_grad_norm: float = 1.0
    use_clipped_value_loss: bool = True
    schedule: str = "adaptive"
    desired_kl: float = 0.01
    init_noise_std: tuple = (0.8, 1.0, 1.0) * 4
    min_noise_std: tuple = (0.15, 0.25, 0.25) * 4
    max_noise_std: float = 1.10

    def __post_init__(self):
        if self.training_phase not in PHASES:
            raise ValueError(f"training_phase must be one of {PHASES}")
        for key in ("history_length", "num_envs", "num_steps_per_env",
                    "num_learning_epochs", "num_mini_batches", "auxiliary_epochs"):
            if not isinstance(getattr(self, key), int) or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be a positive integer")

    @classmethod
    def from_hardpact(cls, env_cfg, train_cfg, **overrides):
        """Explicit allowlist: inherit current settings, not HardPACT architectures/losses.

        Accept config objects or resolved dictionaries; no simulator import is needed.
        New tasks should call this factory rather than rely on snapshot defaults.
        """
        def get(obj, key):
            return obj[key] if isinstance(obj, dict) else getattr(obj, key)
        env, alg = get(env_cfg, "env"), get(train_cfg, "algorithm")
        policy, runner = get(train_cfg, "policy"), get(train_cfg, "runner")
        keys = ("num_learning_epochs", "num_mini_batches", "auxiliary_learning_rate",
                "learning_rate", "clip_param", "gamma", "lam", "value_loss_coef",
                "entropy_coef", "max_grad_norm", "use_clipped_value_loss", "schedule", "desired_kl")
        values = {key: deepcopy(get(alg, key)) for key in keys}
        values.update(history_length=get(env, "num_obs_hist"), num_envs=get(env, "num_envs"),
                      num_steps_per_env=get(runner, "num_steps_per_env"),
                      auxiliary_epochs=get(alg, "num_learning_epochs"))
        for key in ("init_noise_std", "min_noise_std", "max_noise_std"):
            value = deepcopy(get(policy, key))
            if isinstance(value, (tuple, list)):
                if len(value) not in (12, 24):
                    raise ValueError(f"{key}: expected scalar, 12 or 24 action values")
                value = tuple(value[:12])
            values[key] = value
        values.update(overrides)
        return cls(**values)

    def resolved(self):
        return asdict(self)
