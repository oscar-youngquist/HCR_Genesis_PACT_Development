"""Simulator-neutral BR learning contract; no HardPACT objectives are imported.

Integration (the caller supplies its already resolved HardPACT configs)::

    cfg = BeyondRobustnessConfig.from_hardpact(env_cfg, train_cfg)
    runner = BeyondRobustnessRunner(br_env_adapter, cfg, log_dir="logs/br")
    runner.learn(2)  # bounded validation, not a production launcher

The Stage-1 algorithm API remains usable directly. The runner adds collection,
phase scheduling and environment-aware resume. Switch phase explicitly at an
update boundary or set student_reinforcement_start_iteration; neither controls a
simulator curriculum stage. The factory inherits live optimization/terrain settings.
Stage-3 physics is opt-in via IsaacLabBeyondRobustnessEnv; this learning
configuration remains simulator-neutral. Production rewards/launchers are separate.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass, field
import math
from rsl_rl.beyond_robustness_schema import BeyondRobustnessInputSchema


PHASES = ("teacher_student", "student_reinforcement")


@dataclass(frozen=True)
class BeyondRobustnessConfig:
    input_schema: BeyondRobustnessInputSchema = field(default_factory=BeyondRobustnessInputSchema)
    # H means exactly H frames, oldest first, including the current observation.
    history_length: int = 20
    training_phase: str = "teacher_student"
    # Switch BEFORE collecting this absolute iteration; None never switches automatically.
    student_reinforcement_start_iteration: int | None = None
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
    use_adaptive_entropy: bool = True
    adaptive_ent_bounds: tuple = (0.005, 0.01)
    adaptive_ent_lin_threshold: float = 0.75
    adaptive_ent_ang_threshold: float = 0.35
    adaptive_ent_ter_threshold: float = 6.0
    adaptive_ent_softmax_temp: float = 2.0
    max_grad_norm: float = 1.0
    use_clipped_value_loss: bool = True
    schedule: str = "adaptive"
    desired_kl: float = 0.01
    init_noise_std: tuple = (0.8, 1.0, 1.0) * 4
    min_noise_std: tuple = (0.15, 0.25, 0.25) * 4
    max_noise_std: float = 1.10

    def __post_init__(self):
        if isinstance(self.input_schema, dict):
            object.__setattr__(self, "input_schema", BeyondRobustnessInputSchema(**self.input_schema))
        object.__setattr__(self, "adaptive_ent_bounds", tuple(self.adaptive_ent_bounds))
        for name in ("init_noise_std", "min_noise_std", "max_noise_std"):
            if isinstance(getattr(self, name), list):
                object.__setattr__(self, name, tuple(getattr(self, name)))
        start = self.student_reinforcement_start_iteration
        if start is not None and (not isinstance(start, int) or start < 0):
            raise ValueError("student_reinforcement_start_iteration must be None or a nonnegative integer")
        if (len(self.adaptive_ent_bounds) != 2 or
            not all(math.isfinite(x) and x >= 0 for x in self.adaptive_ent_bounds) or
            not math.isfinite(self.adaptive_ent_softmax_temp) or self.adaptive_ent_softmax_temp <= 0):
            raise ValueError("Invalid adaptive entropy bounds/temperature")
        for name in ("adaptive_ent_lin_threshold", "adaptive_ent_ang_threshold", "adaptive_ent_ter_threshold"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
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
                "entropy_coef", "max_grad_norm", "use_clipped_value_loss", "schedule", "desired_kl",
                "use_adaptive_entropy", "adaptive_ent_bounds", "adaptive_ent_lin_threshold",
                "adaptive_ent_ang_threshold", "adaptive_ent_ter_threshold", "adaptive_ent_softmax_temp")
        values = {key: deepcopy(get(alg, key)) for key in keys}
        values.update(history_length=get(env, "num_obs_hist"), num_envs=get(env, "num_envs"),
                      num_steps_per_env=get(runner, "num_steps_per_env"),
                      auxiliary_epochs=get(alg, "num_learning_epochs"))
        # An adapter may override parameter field declarations, but not silently pad them.
        values["input_schema"] = overrides.pop("input_schema", None)
        if values["input_schema"] is None:
            values["input_schema"] = BeyondRobustnessInputSchema.from_terrain(get(env_cfg, "terrain"))
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
