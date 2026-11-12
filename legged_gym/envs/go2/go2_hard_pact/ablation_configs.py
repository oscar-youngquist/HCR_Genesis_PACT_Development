"""Extremely thin configuration selectors for the shared HardPACT task."""

from copy import deepcopy
import inspect
from rsl_rl.hard_pact_ablations import RSS_HARD_PACT_ABLATIONS

from .ablations import HARD_PACT_VARIANTS, resolve_hard_pact_features
from .go2_hard_pact_config import GO2HardPACTCfg, GO2HardPACTCfgPPO
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg
from legged_gym.envs.go2.go2_hard_pact_pos.go2_hard_pact_pos_config import (
    GO2HardPACTPosCfg, GO2HardPACTPosCfgPPO,
)


def make_hard_pact_variant_configs(variant_id: str | None, backend: str):
    """Backend adaptation plus optional flag-only ablation, like B1Z1 PACT.

    None retains the source configuration; named variants explicitly override
    feature switches without copying task logic, architecture, or loss weights.
    """
    if variant_id is not None and variant_id not in HARD_PACT_VARIANTS:
        raise ValueError(variant_id)
    features = resolve_hard_pact_features(variant_id) if variant_id is not None else None

    class EnvCfg(GO2HardPACTCfg):
        ablation_variant = variant_id
        task_backend = backend

        if backend == "isaaclab":
            class sim(GO2HardPACTCfg.sim):
                use_pact_adapter = True

                class physx(LeggedRobotCfg.sim.physx):
                    pass
            class terrain(GO2HardPACTCfg.terrain):
                # Isaac Lab's tested rough-terrain path consumes the shared
                # terrain as a triangle mesh.  Genesis keeps its reference
                # heightfield setting unchanged.
                mesh_type = "trimesh"
            class asset(GO2HardPACTCfg.asset):
                base_link_name = "base"

    class PPOCfg(GO2HardPACTCfgPPO):
        ablation_variant = variant_id
        task_backend = backend

        class algorithm(GO2HardPACTCfgPPO.algorithm):
            ablation_variant = variant_id
            if features is not None:
                bard_inverse_enabled = features.inverse_loss
                bard_rollout_enabled = features.rollout_loss

        class runner(GO2HardPACTCfgPPO.runner):
            run_name = f"hard_pact_{variant_id}_{backend}" if variant_id is not None else f"hard_pact_{backend}"
            task_backend = backend

    label = variant_id.title().replace('_', '') if variant_id is not None else ''
    EnvCfg.__name__ = f"GO2HardPACT{label}{backend.title()}Cfg"
    PPOCfg.__name__ = EnvCfg.__name__ + "PPO"
    return EnvCfg, PPOCfg


def make_hard_pact_backend_configs(backend: str):
    """Normal training task: backend identity only, no ablation overrides."""
    return make_hard_pact_variant_configs(None, backend)


def make_rss_hard_pact_configs(variant):
    """RSS changes only features; never inherit the Pos pretraining schedule."""
    features = RSS_HARD_PACT_ABLATIONS[variant]
    BaseEnv, BasePPO = make_hard_pact_backend_configs("isaaclab")
    # Config classes contain inherited mutable lists/dicts. Own these per RSS
    # selector so CLI/curriculum overrides cannot leak into siblings or legacy.
    def isolated(base):
        attrs = {}
        for name in dir(base):
            if name.startswith('_'):
                continue
            value = getattr(base, name)
            if inspect.isclass(value):
                attrs[name] = isolated(value)
            elif isinstance(value, (dict, list, set)):
                attrs[name] = deepcopy(value)
        return type(base.__name__, (base,), attrs)
    BaseEnv, BasePPO = isolated(BaseEnv), isolated(BasePPO)

    class EnvCfg(BaseEnv):
        ablation_variant = features.variant_id
        action_mode = features.action_mode

    class PPOCfg(BasePPO):
        ablation_variant = features.variant_id

        class algorithm(BasePPO.algorithm):
            ablation_variant = features.variant_id
            bard_inverse_enabled = False
            bard_rollout_enabled = False
            actor_regularizers = deepcopy(BasePPO.algorithm.actor_regularizers)
            if not features.fk:
                actor_regularizers['lambda_fk'] = 0.0
            if not features.ff_support:
                actor_regularizers['lambda_load'] = 0.0
            hard_pact_qp = deepcopy(BasePPO.algorithm.hard_pact_qp)
            hard_pact_qp['use_external_wrench'] = features.qp_wrench

        class policy(BasePPO.policy):
            # Keep the same actor/estimator capacity. Only the distribution is
            # narrowed; environment history retains zero FF placeholders.
            init_noise_std = deepcopy(BasePPO.policy.init_noise_std)
            min_noise_std = deepcopy(BasePPO.policy.min_noise_std)
            max_noise_std = deepcopy(BasePPO.policy.max_noise_std)
            if features.action_mode == "position":
                if isinstance(init_noise_std, (list, tuple)):
                    init_noise_std = init_noise_std[:12]
                if isinstance(min_noise_std, (list, tuple)):
                    min_noise_std = min_noise_std[:12]
                if isinstance(max_noise_std, (list, tuple)):
                    max_noise_std = max_noise_std[:12]

        class runner(BasePPO.runner):
            run_name = f"hard_pact_rss_{variant}_isaaclab"
            experiment_name = f"go2_hard_pact_rss_{variant}"

    EnvCfg.__name__ = f"GO2HardPACTRSS{variant}Cfg"
    PPOCfg.__name__ = EnvCfg.__name__ + "PPO"
    return EnvCfg, PPOCfg


def make_hard_pact_pos_backend_configs(backend: str):
    """Thin backend identity for the one shared neutral pretraining task."""
    class EnvCfg(GO2HardPACTPosCfg):
        task_backend = backend

        if backend == "isaaclab":
            class sim(GO2HardPACTPosCfg.sim):
                use_pact_adapter = True

                class physx(LeggedRobotCfg.sim.physx):
                    pass
            class terrain(GO2HardPACTPosCfg.terrain):
                mesh_type = "trimesh"
            class asset(GO2HardPACTPosCfg.asset):
                base_link_name = "base"

    class PPOCfg(GO2HardPACTPosCfgPPO):
        task_backend = backend

        class runner(GO2HardPACTPosCfgPPO.runner):
            run_name = f"hard_pact_pos_{backend}"
            task_backend = backend

    return EnvCfg, PPOCfg
