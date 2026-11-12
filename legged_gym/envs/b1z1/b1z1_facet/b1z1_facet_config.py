"""Independent Pos configuration snapshots; never mutate the baseline classes."""
from legged_gym.envs.b1z1.b1z1_ppo_pos.b1z1_ppo_pos_config import (
    B1Z1PPOPosCfg, B1Z1PPOPosCfgPPO,
)
from legged_gym.envs.b1z1.b1z1_unifp_original.b1z1_unifp_original_config import copy_sections
from .reference import FACETReferenceCfg
from .tracking import FACETTrackingCfg
from .observations import FACETObservationCfg


class B1Z1FACETCfg:
    def __init__(self):
        self.__dict__.update(vars(copy_sections(B1Z1PPOPosCfg)))
        # Nominal references only. Disabling compliance also avoids its derived
        # +12 profile observation entries; keep native Pos history/controller slots.
        self.compliance.enabled = False
        self.use_force_shifted_target = False
        self.action_mode = "position"  # 17 learned positions, no learned FF head.
        self.facet_reference = FACETReferenceCfg()
        self.facet_tracking = FACETTrackingCfg()  # FACET-only registration; physical settings unchanged.
        self.facet_observations = FACETObservationCfg()  # Separate API, not legacy runner input.


class B1Z1FACETCfgPPO:
    def __init__(self):
        self.__dict__.update(vars(copy_sections(B1Z1PPOPosCfgPPO)))
        # Reuse the currently registered Pos runner/model, not the legacy PPO
        # adapter. Master switches disable every representation/actor PINN term.
        self.algorithm.representation_pinn_enabled = False
        self.algorithm.actor_phys_enabled = False
        self.algorithm.actor_phys_arm_feedforward_enabled = False
        self.policy.action_mode = "position"
        self.runner.experiment_name = "b1z1_facet"
        self.runner.run_name = "facet_stage1"
