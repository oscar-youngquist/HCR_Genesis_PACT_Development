"""Selective experimental inheritance; MULE architecture/losses are independent.

Go2 MULE's environment and learning configurations live together here.
Resolving configuration does not construct a simulator or register a task.
See docs/mule_stage1.rst for source evidence and explicit reproduction choices.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass, field
import math
from .mule_schema import MULESchema

PHASES = ('nominal_only', 'joint_adaptation')
SHARED_OPTIMIZATION = ('learning_rate', 'auxiliary_learning_rate', 'num_learning_epochs',
    'num_mini_batches', 'clip_param', 'gamma', 'lam', 'value_loss_coef', 'entropy_coef',
    'max_grad_norm', 'use_clipped_value_loss', 'schedule', 'desired_kl',
    'use_adaptive_entropy', 'adaptive_ent_bounds', 'adaptive_ent_lin_threshold',
    'adaptive_ent_ang_threshold', 'adaptive_ent_ter_threshold', 'adaptive_ent_softmax_temp')


def get(obj, key):
    return obj[key] if isinstance(obj, dict) else getattr(obj, key)


@dataclass(frozen=True)
class Go2MULECfgPPO:
    schema: MULESchema = field(default_factory=MULESchema)
    optimization_phase: str = 'nominal_only'
    latent_dim: int = 16
    beta: float = 1.  # Reproduction choice, NOT copied from HardPACT's VAE schedule.
    logvar_bounds: tuple = (-10., 10.)  # Numerical safeguard; explicit assumption.
    action_scale: float = .25
    clip_actions: float = 100.
    init_noise_std: tuple = (1.,)*12
    min_noise_std: tuple = (.01,)*12
    max_noise_std: tuple = (2.,)*12
    learning_rate: float = 3e-4
    auxiliary_learning_rate: float = 2e-4
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    auxiliary_epochs: int = 5
    num_steps_per_env: int = 24
    num_envs: int = 4096
    control_dt: float = .02
    clip_param: float = .2
    gamma: float = .99
    lam: float = .95
    value_loss_coef: float = 1.
    entropy_coef: float = .01
    max_grad_norm: float = 1.
    use_clipped_value_loss: bool = True
    schedule: str = 'adaptive'
    desired_kl: float = .01
    use_adaptive_entropy: bool = True
    adaptive_ent_bounds: tuple = (.005, .01)
    adaptive_ent_lin_threshold: float = .75
    adaptive_ent_ang_threshold: float = .35
    adaptive_ent_ter_threshold: float = 6.
    adaptive_ent_softmax_temp: float = 2.
    seed: int = 1
    max_iterations: int = 20000
    save_interval: int = 500

    def __post_init__(self):
        if isinstance(self.schema, dict): object.__setattr__(self, 'schema', MULESchema(**self.schema))
        if self.optimization_phase not in PHASES: raise ValueError('Unknown MULE optimization phase')
        if self.latent_dim != 16: raise ValueError('MULE v1 uses the audited 16-D context')
        for name in ('init_noise_std', 'min_noise_std', 'max_noise_std'):
            x = getattr(self, name)
            x = (float(x),)*12 if isinstance(x, (float, int)) else tuple(x)
            if len(x) != 12 or not all(math.isfinite(v) and v > 0 for v in x):
                raise ValueError(f'{name} requires 12 positive values or a scalar')
            object.__setattr__(self, name, x)
        if any(not lo <= init <= hi for lo, init, hi in zip(self.min_noise_std, self.init_noise_std, self.max_noise_std)):
            raise ValueError('Exploration must satisfy min <= initial <= max')
        for key in ('action_scale', 'clip_actions', 'learning_rate', 'auxiliary_learning_rate', 'control_dt'):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) <= 0: raise ValueError(f'Invalid {key}')
        if not math.isfinite(self.beta) or self.beta < 0: raise ValueError('beta must be nonnegative')
        object.__setattr__(self, 'logvar_bounds', tuple(self.logvar_bounds))
        object.__setattr__(self, 'adaptive_ent_bounds', tuple(self.adaptive_ent_bounds))
        if len(self.logvar_bounds) != 2 or not all(math.isfinite(x) for x in self.logvar_bounds) or self.logvar_bounds[0] >= self.logvar_bounds[1]:
            raise ValueError('Invalid logvar bounds')
        for key in ('num_learning_epochs', 'num_mini_batches', 'auxiliary_epochs', 'num_steps_per_env', 'num_envs'):
            if not isinstance(getattr(self, key), int) or getattr(self, key) < 1: raise ValueError(f'Invalid {key}')

    @classmethod
    def from_shared(cls, env_cfg, train_cfg, **overrides):
        env, terrain, control = (get(env_cfg, k) for k in ('env', 'terrain', 'control'))
        alg, policy, runner = (get(train_cfg, k) for k in ('algorithm', 'policy', 'runner'))
        normal = get(env_cfg, 'normalization'); scales = get(normal, 'obs_scales')
        heights = get(terrain, 'measure_heights')
        schema = MULESchema(history_length=get(env, 'num_obs_hist'),
            terrain_points_x=get(terrain, 'measured_points_x') if heights else (),
            terrain_points_y=get(terrain, 'measured_points_y') if heights else (),
            joint_order=get(get(env_cfg, 'asset'), 'dof_names'),
            angular_velocity_scale=get(scales, 'ang_vel'), linear_velocity_scale=get(scales, 'lin_vel'),
            joint_position_scale=get(scales, 'dof_pos'), joint_velocity_scale=get(scales, 'dof_vel'),
            terrain_height_scale=get(scales, 'height_measurements'))
        values = {k: deepcopy(get(alg, k)) for k in SHARED_OPTIMIZATION}
        values.update(schema=schema, num_envs=get(env, 'num_envs'), action_scale=get(control, 'action_scale'),
            clip_actions=get(normal, 'clip_actions'), control_dt=get(control, 'dt'),
            num_steps_per_env=get(runner, 'num_steps_per_env'), auxiliary_epochs=get(alg, 'num_learning_epochs'),
            seed=get(train_cfg, 'seed'), max_iterations=get(runner, 'max_iterations'), save_interval=get(runner, 'save_interval'))
        for key in ('init_noise_std', 'min_noise_std', 'max_noise_std'):
            x = deepcopy(get(policy, key))
            if isinstance(x, (tuple, list)):
                if len(x) not in (12, 24): raise ValueError(f'{key}: expected 12 position or 24 coupled channels')
                x = tuple(x[:12])
            values[key] = x
        values.update(overrides)
        return cls(**values)

    def resolved(self): return asdict(self)

    def contract(self):
        result = self.resolved()
        result.pop('optimization_phase')  # Runtime phase restored independently of initial default.
        return dict(version=1, config=result, schema=self.schema.contract(),
                    actor_critic_widths=(512,256,128), encoder_widths=(128,64), decoder_widths=(64,128),
                    activation='ELU', decoder_input='estimated velocity_B * linear_velocity_scale, sampled z',
                    routing='auxiliary-only context; detached shared rollout context for both actors')


@dataclass(frozen=True)
class Go2MULECfg:
    schema: MULESchema
    num_envs: int
    control_dt: float
    decimation: int
    action_scale: float
    clip_actions: float
    standing_position: tuple

    @property
    def num_observations(self): return self.schema.observation_dim
    @property
    def num_actions(self): return 12
    @property
    def num_privileged_obs(self): return self.schema.critic_dim

    def __post_init__(self):
        if not isinstance(self.decimation,int) or self.decimation<1: raise ValueError('Invalid decimation')
        if len(self.standing_position)!=12 or not all(math.isfinite(x) for x in self.standing_position):
            raise ValueError('Expected twelve finite named standing joint positions')


def resolve_go2_mule_configs(env_cfg=None, train_cfg=None, **learning_overrides):
    """Owned env/learning contracts, not a runnable physical task yet."""
    if (env_cfg is None)!=(train_cfg is None): raise ValueError('Supply both shared configs or neither')
    if env_cfg is None:
        from legged_gym.envs.go2.go2_hard_pact.ablation_configs import make_hard_pact_backend_configs
        E,T=make_hard_pact_backend_configs('isaaclab'); env_cfg,train_cfg=E(),T()
    training=Go2MULECfgPPO.from_shared(env_cfg,train_cfg,**learning_overrides)
    angles=get(get(env_cfg,'init_state'),'default_joint_angles')
    environment=Go2MULECfg(training.schema, training.num_envs, training.control_dt,
        get(get(env_cfg,'control'),'decimation'), training.action_scale, training.clip_actions,
        tuple(angles[name] for name in training.schema.joint_order))
    return environment,training
