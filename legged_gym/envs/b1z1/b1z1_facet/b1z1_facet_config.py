"""Independent Pos configuration snapshots; never mutate the baseline classes."""
from legged_gym.envs.b1z1.b1z1_ppo_pos.b1z1_ppo_pos_config import B1Z1PPOPosCfg
from legged_gym.envs.b1z1.b1z1_unifp_original.b1z1_unifp_original_config import copy_sections
from .reference import FACETReferenceCfg
from .tracking import FACETTrackingCfg
from .observations import FACETObservationCfg
from .forces import FACETForceCfg
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
        self.pilot_interactions.enabled = False  # FACET keeps its own reference/force formulation.
        self.use_force_shifted_target = False
        self.action_mode = "position"  # 17 learned positions, no learned FF head.
        self.facet_reference = FACETReferenceCfg()
        self.facet_reference.sampling_mode = 'bounded_b1z1'
        self.facet_reference.formulation_version = 3
        self.facet_reference.bounded = True
        self.facet_reference.yaw_reaction = True
        self.facet_reference.ee_stiffness = 90.  # Paper calibration when sampling_mode='fixed'.
        self.facet_reference.ee_damping = 2*(2*90.)**.5
        self.facet_reference.base_damping = 2*(4*16.)**.5
        self.facet_reference.yaw_damping = 2*(1.5*16.)**.5
        r = self.facet_reference
        r.radius, r.pitch, r.azimuth = (tuple(getattr(self.goal_ee.ranges, key)) for key in ('pos_l', 'pos_p', 'pos_y'))
        r.center_xy = (self.goal_ee.sphere_center.x_offset, self.goal_ee.sphere_center.y_offset)
        r.box_low, r.box_high = tuple(self.goal_ee.collision_lower_limits), tuple(self.goal_ee.collision_upper_limits)
        r.ground = self.goal_ee.underground_limit
        self.facet_forces = FACETForceCfg()
        self.domain_rand.push_robots = False
        self.commands.push_gripper_stators = self.commands.push_robot_base = True
        self.commands.apply_ee_external_forces = self.commands.apply_base_external_forces = True
        self.commands.apply_base_external_torques = False
        self.commands.zero_vel_cmd_prob_after_force = self.commands.zero_vel_cmd_prob
        # Reward/domain curricula keep their old control-transition clock in
        # the FACET runner, rather than moving milestones 32/24 later.
        self.env.num_steps_per_env = 32
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
        # Retain legacy threshold metadata for old diagnostic readers; the new
        # formulation disables adaptive entropy and rejection-based force gates.
        tracking = FACETTrackingCfg()
        self.algorithm.facet_reward_schedule_version = 1
        self.algorithm.adaptive_ent_lin_threshold *= tracking.weights['base_velocity'] / B1Z1PPOPosCfg.rewards.scales.tracking_lin_vel_force_world
        self.algorithm.adaptive_ent_ang_threshold *= tracking.weights['yaw_rate'] / B1Z1PPOPosCfg.rewards.scales.tracking_ang_vel
        # Pinned FACET optimizer defaults, not PILOT's AdamW/adaptive schedule.
        self.algorithm.weight_decay = 0.
        self.algorithm.optimizer = 'Adam'
        self.algorithm.learning_rate = 5e-4
        self.algorithm.schedule = 'fixed'
        self.algorithm.entropy_coef = .002
        self.algorithm.use_adaptive_entropy = False
        self.algorithm.num_learning_epochs = 4
        self.algorithm.num_mini_batches = 4
        self.algorithm.use_clipped_value_loss = False
        self.algorithm.formulation_version = 3
        self.algorithm.curriculum_steps_per_iteration = 24
        self.algorithm.estimator_learning_rate = 5e-4
        self.algorithm.estimator_epochs = 2
        self.algorithm.estimator_teacher_interval = 2
        self.algorithm.estimator_ema_tau = .04
        self.algorithm.student_start_iteration = 16500  # 80% of the current 20,625-update budget; 4,125 student updates.
        self.algorithm.adaptation_iterations = 0  # Optional adaptation consumes the remaining budget, never adds interactions.
        self.policy = SimpleNamespace(action_mode='position')
        self.runner.policy_class_name = 'FACETModel'
        self.runner.algorithm_class_name = 'FACETPPO'
        self.runner.experiment_name = "b1z1_facet"
        self.runner.run_name = "facet_teacher_student"
        self.runner.num_steps_per_env = 32
        self.runner.max_iterations = pilot.runner.max_iterations*24//32
        self.runner.save_interval = max(1, pilot.runner.save_interval*24//32)
