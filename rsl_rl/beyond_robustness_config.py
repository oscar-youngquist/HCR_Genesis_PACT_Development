"""Simulator-neutral BR learning contract; no HardPACT objectives are imported.

Physical integration (resolved from live shared settings)::

    resolved = resolve_beyond_robustness()
    env = IsaacLabBeyondRobustnessEnv(resolved)  # merged BR + shared additions
    runner = BeyondRobustnessRunner(env, resolved.training, log_dir="logs/br")
    runner.learn(2)  # bounded validation, not a production launcher

The Stage-1 algorithm API remains usable directly. The runner adds collection,
phase scheduling and environment-aware resume. Switch phase explicitly at an
update boundary or set student_reinforcement_start_iteration; neither controls a
simulator curriculum stage. The factory inherits live optimization/terrain settings.
Stage-3 physics is opt-in via IsaacLabBeyondRobustnessEnv; this learning
configuration remains simulator-neutral. Production launchers remain separate.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass, field
import math
from rsl_rl.beyond_robustness_schema import BeyondRobustnessInputSchema
from types import SimpleNamespace


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
    seed: int = 1
    max_iterations: int = 20000
    save_interval: int = 500

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
        for key in ('seed','max_iterations','save_interval'):
            if not isinstance(getattr(self,key),int) or getattr(self,key)<0:
                raise ValueError(f'{key} must be a nonnegative integer')

    @classmethod
    def from_hardpact(cls, env_cfg, train_cfg, **overrides):
        """Explicit allowlist: inherit current settings, not HardPACT architectures/losses.

        Accept config objects or resolved dictionaries; no simulator import is needed.
        Physical tasks use resolve_beyond_robustness(), which also supplies the
        tray-aware schema and validates the environment/controller contract.
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
        for key, source in (("seed", train_cfg), ("max_iterations", runner), ("save_interval", runner)):
            if key not in overrides:
                value = source.get(key) if isinstance(source, dict) else getattr(source, key, None)
                if value is not None: values[key] = value
        return cls(**values)

    def resolved(self):
        return asdict(self)


def _owned_settings(value):
    """Materialize inherited class attributes too; never share class-level lists."""
    if isinstance(value, dict): return {k: _owned_settings(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return type(value)(_owned_settings(v) for v in value)
    if hasattr(value, '__dict__'):
        return SimpleNamespace(**{k: _owned_settings(getattr(value, k)) for k in dir(value)
                                  if not k.startswith('_') and not callable(getattr(value, k))})
    return deepcopy(value)


def settings_dict(value):
    if isinstance(value, SimpleNamespace): return {k: settings_dict(v) for k,v in vars(value).items()}
    if isinstance(value, dict): return {k: settings_dict(v) for k,v in value.items()}
    if isinstance(value, (tuple,list)): return [settings_dict(v) for v in value]
    return value


@dataclass
class BeyondRobustnessResolvedConfig:
    environment: object
    training: BeyondRobustnessConfig
    payload: object
    manifest: dict

    @property
    def sim_params(self): return settings_dict(self.environment.sim)


def resolve_beyond_robustness(env_cfg=None, train_cfg=None, *, environment_overrides=None,
                             training_overrides=None, payload_overrides=None, reward_overrides=None):
    """Authoritative physical BR factory. Overrides use explicit dotted field paths.

    Whole physical sections below are shared; rewards are NOT inherited. No
    HardPACT network/physics/QP dictionaries enter BR's learning configuration.
    Source configurations are never mutated. Default sources are the live Go2
    Isaac Lab configs, not a snapshot of their numerical values.
    """
    from rsl_rl.beyond_robustness_schema import PARAMETER_FIELDS, InputField
    from rsl_rl.beyond_robustness_payload import SlidingPayloadConfig
    from rsl_rl.beyond_robustness_rewards import PARAMETERS, resolve_reward_spec
    if (env_cfg is None) != (train_cfg is None):
        raise ValueError('Supply both shared configurations or neither')
    if env_cfg is None:
        from legged_gym.envs.go2.go2_hard_pact.ablation_configs import make_hard_pact_backend_configs
        E,T = make_hard_pact_backend_configs('isaaclab')
        env_cfg,train_cfg = E(),T()
    sections = ('terrain','commands','init_state','normalization','noise','domain_rand',
                'termination','asset','sim','viewer','sensor')
    env_fields = ('num_envs','num_obs_hist','episode_length_s','env_spacing',
                  'fail_to_terminal_time_s','send_timeouts','lateral_push_only','debug','debug_viz')
    control_fields = ('stiffness','damping','action_scale','dt','decimation')
    reward_fields = ('soft_dof_pos_limit','soft_dof_vel_limit','soft_torque_limit',
                     'edge_clearance_forward_cells','edge_clearance_lateral_cells','feet_edge_threshold') + PARAMETERS
    cfg = SimpleNamespace(**{k:_owned_settings(getattr(env_cfg,k)) for k in sections})
    for section, names in (('env',env_fields),('control',control_fields),('rewards',reward_fields)):
        source = getattr(env_cfg,section)
        setattr(cfg,section,SimpleNamespace(**{k:deepcopy(getattr(source,k)) for k in names}))
    cfg.env.num_actions=12; cfg.env.num_observations=45
    cfg.sensor.add_depth=False  # BR actor is proprioceptive; no depth/TCN path.
    cfg.control.control_type='P'; cfg.control.use_tradeoff_curriculum=False
    cfg.control.randomize_pact_weights=False
    # Independent payload bodies need more broadphase memory at 4096 envs.
    # Capacity only: solver iterations/tolerances remain inherited unchanged.
    cfg.sim.physx.gpu_total_aggregate_pairs_capacity=getattr(cfg.sim.physx,'gpu_total_aggregate_pairs_capacity',2**22)
    cfg.seed=int((training_overrides or {}).get('seed',getattr(train_cfg,'seed',1)))
    overrides=deepcopy(environment_overrides or {})
    for path,value in overrides.items():
        parts=path.split('.'); target=cfg
        for name in parts[:-1]: target=getattr(target,name)
        if not hasattr(target,parts[-1]): raise ValueError(f'Unknown BR environment override: {path}')
        setattr(target,parts[-1],_owned_settings(value))
    dt=cfg.control.dt; decimation=cfg.control.decimation
    if not math.isfinite(dt) or dt<=0 or not isinstance(decimation,int) or decimation<1:
        raise ValueError('Invalid control dt/decimation')
    if 'sim.dt' in overrides and not math.isclose(cfg.sim.dt,dt/decimation):
        raise ValueError('sim.dt must equal control.dt/decimation')
    cfg.sim.dt=dt/decimation
    cfg.br_reward=resolve_reward_spec(env_cfg.rewards,dt,reward_overrides)
    cfg.br_reward['parameters']=settings_dict(cfg.rewards)
    cfg.br_reward['action_scale']=cfg.control.action_scale
    cfg.br_reward['joint_order']=list(cfg.asset.dof_names)
    cfg.br_reward['foot_order']=list(cfg.asset.foot_name)
    cfg.terrain.num_subterrains=cfg.terrain.num_rows*cfg.terrain.num_cols
    schema=BeyondRobustnessInputSchema.from_terrain(cfg.terrain,
        parameter_fields=PARAMETER_FIELDS+(InputField('tray_mass',1,'kg'),),
        joint_order=tuple(cfg.asset.dof_names), actor_clip=float(cfg.normalization.clip_observations))
    canonical=tuple(f'{leg}_{joint}_joint' for leg in schema.foot_order for joint in ('hip','thigh','calf'))
    if schema.joint_order!=canonical: raise ValueError('BR requires canonical FR/FL/RR/RL joint ordering')
    if set(cfg.init_state.default_joint_angles)!=set(canonical): raise ValueError('Joint references do not match named joints')
    if tuple(cfg.asset.foot_name)!=tuple(f'{leg}_foot' for leg in schema.foot_order):
        raise ValueError('Feet must be named in canonical FR/FL/RR/RL order')
    if set(cfg.control.stiffness)!=set(cfg.control.damping):
        raise ValueError('Stiffness/damping named gain groups must match')
    for gains in (cfg.control.stiffness,cfg.control.damping):
        if any(not math.isfinite(v) or v<0 for v in gains.values()) or any(not any(k in j for k in gains) for j in canonical):
            raise ValueError('PD gains must cover every named joint with finite nonnegative values')
    scale=cfg.control.action_scale
    if not isinstance(scale,(int,float)) or not math.isfinite(scale) or scale<=0:
        raise ValueError('Position action_scale must be a positive scalar')
    if cfg.env.num_actions!=12 or cfg.env.num_observations!=45 or cfg.control.control_type!='P':
        raise ValueError('BR requires 45 proprioception values and 12 position actions')
    if cfg.commands.curriculum_patience_iterations<0 or cfg.terrain.curriculum_upward_delay_iterations<0:
        raise ValueError('Curriculum patience/delay must be nonnegative')
    for value in (cfg.env.episode_length_s,cfg.commands.resampling_time,cfg.normalization.clip_actions):
        if not math.isfinite(value) or value<=0: raise ValueError('Episode/command duration and action clip must be positive')
    if cfg.sensor.add_depth: raise ValueError('BR Stage 4 uses proprioception, not depth/TCN observations')
    capacity=cfg.sim.physx.gpu_total_aggregate_pairs_capacity
    if not isinstance(capacity,int) or capacity<1: raise ValueError('Invalid PhysX aggregate-pair capacity')
    for value in (cfg.noise.noise_level,*(getattr(cfg.noise.noise_scales,k) for k in ('ang_vel','gravity','dof_pos','dof_vel'))):
        if not math.isfinite(value) or value<0: raise ValueError('Noise scales must be finite and nonnegative')
    delay=cfg.domain_rand.ctrl_delay_step_range
    if len(delay)!=2 or any(not isinstance(x,int) or x<0 for x in delay) or delay[1]<delay[0]:
        raise ValueError('Control delay requires ordered nonnegative integer steps')
    training=BeyondRobustnessConfig.from_hardpact(cfg,train_cfg,input_schema=schema,**(training_overrides or {}))
    if cfg.seed!=training.seed: raise ValueError('Set the shared run seed through training_overrides')
    if training.num_envs!=cfg.env.num_envs or training.history_length!=cfg.env.num_obs_hist:
        raise ValueError('Override environment sizing/history, not the derived training values')
    payload=SlidingPayloadConfig(**dict({'seed':cfg.seed},**(payload_overrides or {})))
    if payload.seed!=cfg.seed: raise ValueError('Payload seed must match resolved run seed')
    manifest=dict(version=2,source='live Go2 HardPACT Isaac Lab',inherited_sections=sections,
        inherited_fields=dict(env=env_fields,control=control_fields,physical_limits=reward_fields),
        br_specific=dict(actor='45 SI channels; absolute joint angles; cached sensor noise',
            privileged='clean unclipped simulator truth',actions='12 position offsets, clipped before delay',
            depth=False,controller='position PD, no branch weighting or QP',
            latent=32,load_labels='7 physical values; supervised-only estimator',reward=deepcopy(cfg.br_reward),
            resets='shared Go2PACT joint/root reset methods',geometry='seeded size bank; creation-time only',
            gpu_total_aggregate_pairs_capacity=capacity),
        explicit_overrides=dict(environment=overrides,training=deepcopy(training_overrides or {}),payload=deepcopy(payload_overrides or {}),reward=deepcopy(reward_overrides or {})),
        effective_environment=settings_dict(cfg),effective_training=training.resolved(),payload=asdict(payload))
    return BeyondRobustnessResolvedConfig(cfg,training,payload,manifest)
