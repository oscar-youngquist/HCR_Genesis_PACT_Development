"""Extremely thin configuration selectors for the shared HardPACT task."""

from .ablations import HARD_PACT_VARIANTS
from .go2_hard_pact_config import GO2HardPACTCfg, GO2HardPACTCfgPPO
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg
from legged_gym.envs.go2.go2_hard_pact_pos.go2_hard_pact_pos_config import (
    GO2HardPACTPosCfg, GO2HardPACTPosCfgPPO,
)


def make_hard_pact_variant_configs(variant_id: str, backend: str):
    """Create config subclasses which select metadata and no task logic."""
    if variant_id not in HARD_PACT_VARIANTS:
        raise ValueError(variant_id)

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

        class runner(GO2HardPACTCfgPPO.runner):
            run_name = f"hard_pact_{variant_id}_{backend}"
            task_backend = backend

    EnvCfg.__name__ = f"GO2HardPACT{variant_id.title().replace('_', '')}{backend.title()}Cfg"
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
