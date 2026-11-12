"""Independent Pos configuration snapshots; never mutate the baseline classes."""
from legged_gym.envs.b1z1.b1z1_ppo_pos.b1z1_ppo_pos_config import B1Z1PPOPosCfg
from legged_gym.envs.b1z1.b1z1_unifp_original.b1z1_unifp_original_config import copy_sections
from .reference import FACETReferenceCfg
from .tracking import FACETTrackingCfg
from .observations import FACETObservationCfg
from types import SimpleNamespace
from legged_gym.envs.b1z1.b1z1_pact.b1z1_pact_config import B1Z1PACTCfgPPO


COMMON_PPO = ('value_loss_coef', 'use_clipped_value_loss', 'clip_param', 'entropy_coef',
    'learning_rate', 'schedule', 'gamma', 'lam', 'desired_kl', 'max_grad_norm',
    'num_learning_epochs', 'num_mini_batches', 'use_adaptive_entropy')


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
        # FACET upstream quadruped.py, pinned c0d6f5a: only import gains, not
        # its effort/velocity limits, action scales, actuator model or reset pose.
        self.control.stiffness = {name: 40. if name.startswith('z1_') else 100.
                                  for name in self.asset.dof_names}
        self.control.damping = {name: 1. if name.startswith('z1_') else 2.
                                for name in self.asset.dof_names}
        self.terrain.curriculum = False


class B1Z1FACETCfgPPO:
    def __init__(self):
        pilot = copy_sections(B1Z1PACTCfgPPO)
        self.seed = pilot.seed
        self.runner_class_name = 'B1Z1FACETRunner'
        self.runner = pilot.runner
        self.algorithm = SimpleNamespace(**{key: value for key, value in vars(pilot.algorithm).items()
            if key in COMMON_PPO or key.startswith('adaptive_ent_')})
        # AdamW matches PILOT ownership groups: all weights/biases decay 1e-6,
        # Gaussian std has no decay. Estimator uses upstream's separate Adam.
        self.algorithm.weight_decay = 1e-6
        self.algorithm.estimator_learning_rate = 5e-4
        self.algorithm.estimator_epochs = 2
        self.algorithm.estimator_teacher_interval = 2
        self.algorithm.estimator_ema_tau = .04
        self.algorithm.student_start_iteration = 60000  # Completed PPO updates, never simulator steps.
        self.policy = SimpleNamespace(action_mode='position')
        self.runner.policy_class_name = 'FACETModel'
        self.runner.algorithm_class_name = 'FACETPPO'
        self.runner.experiment_name = "b1z1_facet"
        self.runner.run_name = "facet_teacher_student"
