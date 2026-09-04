"""Go2 HardPACT environment with GRF and persistent-wrench extensions."""

from __future__ import annotations

from collections import deque
import time

import torch

from rsl_rl.modules.hard_pact_physics import compose_explicit_estimator_target
from rsl_rl.algorithms.hard_pact_qp import (
    balanced_anchor_indices,
    balanced_substep_indices,
)
from legged_gym.dynamics import wrench_at_point
from legged_gym.envs.go2.go2_pact.go2_pact import Go2PACT
from legged_gym.utils.math_utils import quat_rotate_inverse

from .grf import GRFProcessingConfig, IntervalGRFProcessor, world_to_yaw_local
from .ablations import resolve_hard_pact_features
from .domain_rand_curriculum import HardPACTDomainRandCurriculum

from .transition import (
    DISTURBANCE_CRITIC_DIM,
    added_mass_gravity_wrench_world,
    pack_disturbance_fields,
    physics_transition_mask,
    wrench_world_to_scaled_yaw_local,
)


class Go2HardPACT(Go2PACT):
    """Legacy Go2 PACT with GRF and persistent external-wrench targets."""

    _legacy_task_class = Go2PACT

    def _reward_torque_cancellation(self):
        r"""Penalize opposing effective PD/feed-forward joint torques.

        This is the B1Z1 PACT torque-conflict reward. For each joint,

        ``c_j = |tau_fb| + |tau_ff| - |tau_fb + tau_ff|``

        is zero for aligned contributions and twice the smaller magnitude for
        opposing contributions.  Normalizing by the actuator limit makes the
        configured deadband comparable across Go2 joints.
        """
        simulator = self.simulator
        feedback = (
            simulator.feedback_tau_weight * simulator.feedback_torques
        )
        feedforward = (
            simulator.feedforward_tau_weight * simulator.feedforward_torques
        )
        # Go2 PACT applies motor-strength scaling after combining the two
        # weighted branches, so include it in both effective contributions.
        motor_strength = getattr(simulator, "_motor_strength", None)
        if motor_strength is not None:
            feedback = feedback * motor_strength
            feedforward = feedforward * motor_strength
        limits = simulator.torque_limits[:feedback.shape[-1]].clamp_min(1.0e-6)
        cancellation = (
            feedback.abs() + feedforward.abs()
            - (feedback + feedforward).abs()
        ).clamp_min(0.0)
        excess = torch.relu(
            cancellation / limits
            - float(self.cfg.rewards.torque_cancellation_deadband)
        )
        return excess.square().mean(dim=-1)

    def _reward_foot_clearance_terrain_aware(self):
        """Track local-terrain clearance on swing feet, as in B1Z1 PACT."""
        feet_z = self.simulator.feet_pos[:, :, 2]
        # HardPACT's conditioned contact state is backend-neutral and already
        # uses the configured GRF deadband/threshold at every physics substep.
        contacts = self.grf_processor.contacts
        swing = ~contacts

        height_patch = self.simulator._height_around_feet
        if height_patch.ndim == 4:
            height_patch = height_patch.reshape(
                height_patch.shape[0], height_patch.shape[1], -1
            )
        local_terrain_height = height_patch.amax(dim=-1)
        desired_height = (
            float(self.cfg.rewards.foot_clearance_target)
            + float(self.cfg.rewards.foot_height_offset)
            + local_terrain_height
        )
        tracking_error = (feet_z - desired_height).square()
        excess = torch.relu(
            feet_z
            - desired_height
            - float(self.cfg.rewards.foot_clearance_excess_margin)
        )
        total_error = (
            swing
            * (
                tracking_error
                + float(self.cfg.rewards.foot_clearance_excess_weight)
                * excess.square()
            )
        ).sum(dim=-1)
        return torch.exp(
            -total_error
            / float(self.cfg.rewards.foot_clearance_tracking_sigma)
        )


    def _reward_front_foot_overreach(self):
        # Assumed order is FR/L, RL/R....
        front_1 = quat_rotate_inverse(
            self.simulator.base_quat,
            self.simulator.feet_pos[:, 0, :] - self.simulator.base_pos
        )  # (N,3)

        front_2 = quat_rotate_inverse(
            self.simulator.base_quat,
            self.simulator.feet_pos[:, 1, :] - self.simulator.base_pos
        )  # (N,3)

        front_x_1 = front_1[:, 0]   # (N,)
        front_x_2 = front_2[:, 0]   # (N,)

        front_x = torch.stack([front_x_1, front_x_2], dim=1)   # (N,2)

        # Penalize stance feet on either side of the optimized nominal x
        # location, rather than only limiting excessive forward extension.
        front_x_nominal = self.cfg.rewards.front_foot_x_nominal
        front_x_margin = self.cfg.rewards.foot_x_margin
        x_error = torch.abs(front_x - front_x_nominal)
        overreach = torch.relu(x_error - front_x_margin)

        # stance/contact gating
        contact = (
            self.simulator.link_contact_forces[
                :, self.simulator.feet_contact_indices[:2], 2
            ] > 5.0
        ).float()

        penalty = torch.sum(contact * overreach ** 2, dim=1)

        return penalty

    def _reward_rear_foot_overreach(self):
        """
        Penalize rear feet for being too far from their nominal rear-foot x location
        in the base frame, in either direction.

        Penalizes:
        - rear foot too far forward
        - rear foot too far backward

        Assumed foot order: FR, FL, RR, RL
        """

        # Rear feet in base frame
        rear_1 = quat_rotate_inverse(
            self.simulator.base_quat,
            self.simulator.feet_pos[:, 2, :] - self.simulator.base_pos
        )  # RR, (N,3)

        rear_2 = quat_rotate_inverse(
            self.simulator.base_quat,
            self.simulator.feet_pos[:, 3, :] - self.simulator.base_pos
        )  # RL, (N,3)

        rear_x = torch.stack([rear_1[:, 0], rear_2[:, 0]], dim=1)  # (N,2)

        # Nominal rear-foot x location in base frame.
        # This should usually be negative, e.g. -0.20 to -0.25 m.
        rear_x_nominal = self.cfg.rewards.rear_foot_x_nominal

        # Allowed deviation around nominal rear-foot x location.
        # Example: 0.08 m allows rear_x in [nominal - 0.08, nominal + 0.08].
        rear_x_margin = self.cfg.rewards.rear_foot_x_margin

        # Penalize both too far forward and too far backward relative to nominal.
        x_error = torch.abs(rear_x - rear_x_nominal)
        overreach = torch.relu(x_error - rear_x_margin)

        # Contact sensors have their own indexing, distinct from articulation
        # body indices on Isaac Lab. Both lists are canonical FR, FL, RR, RL.
        contact = (
            self.simulator.link_contact_forces[
                :, self.simulator.feet_contact_indices[2:4], 2
            ] > 5.0
        ).float()  # (N,2)

        penalty = torch.sum(contact * overreach ** 2, dim=1)

        return penalty


    def _init_buffers(self):
        self._legacy_task_class._init_buffers(self)
        self.domain_rand_curriculum = HardPACTDomainRandCurriculum(
            self.cfg, seed=int(getattr(self.cfg, "seed", 0))
        )
        self._apply_domain_rand_curriculum_state()
        self._refresh_domain_rand_capability_report()
        self.hard_pact_features = resolve_hard_pact_features(
            getattr(self.cfg, "ablation_variant", "full")
        )
        # The legacy task builds one 288-D critic frame. Keep that deque at its
        # original width, then append a separately managed named disturbance
        # frame below. This prevents any legacy field from moving.
        self._legacy_privileged_obs_dim = (
            self.cfg.env.num_privileged_obs - DISTURBANCE_CRITIC_DIM
        )
        self.critic_obs_deque = deque(maxlen=self.cfg.env.num_priv_stack)
        self.disturbance_critic_deque = deque(maxlen=self.cfg.env.num_priv_stack)
        for _ in range(self.cfg.env.num_priv_stack):
            self.critic_obs_deque.append(torch.zeros(
                self.num_envs, self._legacy_privileged_obs_dim,
                device=self.device, dtype=torch.float32,
            ))
            self.disturbance_critic_deque.append(torch.zeros(
                self.num_envs, DISTURBANCE_CRITIC_DIM,
                device=self.device, dtype=torch.float32,
            ))

        grf_cfg = self.cfg.sim.grf
        self.grf_processor = IntervalGRFProcessor(
            self.num_envs,
            len(self.simulator.feet_indices),
            self.device,
            torch.float32,
            GRFProcessingConfig(
                vertical_deadband_n=float(grf_cfg.vertical_deadband_n),
                clip_min_n=float(grf_cfg.clip_min_n),
                clip_max_n=float(grf_cfg.clip_max_n),
                ema_alpha=float(grf_cfg.ema_alpha),
                contact_threshold_n=float(grf_cfg.contact_threshold_n),
            ),
        )
        self.simulator._hard_pact_grf_post_physics_substep = (
            self._hard_pact_grf_post_physics_substep
        )
        self._persistent_wrench_target_world = torch.zeros(
            self.num_envs, 6, device=self.device
        )
        self._current_sustained_wrench_world = torch.zeros(
            self.num_envs, 6, device=self.device
        )
        self._persistent_component_active = torch.zeros(
            self.num_envs, 2, device=self.device, dtype=torch.bool
        )
        self._current_sustained_active_mask = torch.zeros(
            self.num_envs, 1, device=self.device, dtype=torch.bool
        )
        self._persistent_start_step = torch.zeros(
            self.num_envs, 2, device=self.device, dtype=torch.long
        )
        self._persistent_end_step = torch.zeros_like(self._persistent_start_step)
        self._persistent_duration_steps = torch.zeros_like(
            self._persistent_start_step
        )
        self._persistent_next_event_step = torch.zeros_like(
            self._persistent_start_step
        )
        self._disturbance_interval_sum_sustained = torch.zeros_like(
            self._current_sustained_wrench_world
        )
        self._disturbance_interval_sum_mass_com = torch.zeros_like(
            self._current_sustained_wrench_world
        )
        self._disturbance_interval_sum_total = torch.zeros_like(
            self._current_sustained_wrench_world
        )
        self._disturbance_interval_sum_sustained_yaw_scaled = torch.zeros_like(
            self._current_sustained_wrench_world
        )
        self._disturbance_interval_sum_mass_com_yaw_scaled = torch.zeros_like(
            self._current_sustained_wrench_world
        )
        self._disturbance_interval_sum_yaw_scaled = torch.zeros_like(
            self._current_sustained_wrench_world
        )
        self._disturbance_interval_count = torch.zeros(
            self.num_envs, 1, device=self.device
        )
        self._realized_added_mass = torch.zeros(
            self.num_envs, 1, device=self.device
        )
        self._realized_com_shift_body = torch.zeros(
            self.num_envs, 3, device=self.device
        )
        self._pending_disturbance_transition = None
        self.last_transition = {}
        self._interval_executed_torque_sum = torch.zeros(
            self.num_envs, 12, device=self.device
        )
        self._interval_executed_torque_count = torch.zeros(
            self.num_envs, 1, device=self.device
        )
        self._interval_executed_torque_peak = torch.zeros(
            self.num_envs, 12, device=self.device
        )
        # The runner binds the already-created BARD/QP modules after policy
        # construction.  Until then (and when the QP is disabled), this alias
        # follows the byte-for-byte legacy torque path.
        self._hard_pact_rollout_qp_enabled = False
        self._hard_pact_policy_context_ready = False
        self._hard_pact_push_event_mask = torch.zeros(
            self.num_envs, 1, device=self.device, dtype=torch.bool
        )
        # Mirror only the validity of the legacy action queue. The actions
        # themselves remain owned by the unchanged legacy queue, so exact
        # delayed-action replay adds just one boolean per queue slot here.
        action_queue_length = (
            self.action_queue.shape[1]
            if hasattr(self, "action_queue") else 1
        )
        self._action_replay_valid_queue = torch.zeros(
            self.num_envs, action_queue_length,
            device=self.device, dtype=torch.bool,
        )
        self._pending_action_replay_transition = None
        self.simulator._hard_pact_pre_physics_substep = (
            self._hard_pact_pre_physics_substep
        )
        self._reset_persistent_wrench_state(
            torch.arange(self.num_envs, device=self.device)
        )

    def _apply_domain_rand_curriculum_state(self):
        """Publish neutral effective ranges to the backend's reset samplers."""
        ranges = self.domain_rand_curriculum.effective_ranges()
        sim = self.simulator
        sim.domain_rand_joint_dynamics_progress = self.domain_rand_curriculum.progress["joint_dynamics"]
        sim.domain_rand_mass_com_progress = self.domain_rand_curriculum.progress["mass_com"]
        sim.domain_rand_disturbance_progress = self.domain_rand_curriculum.progress["disturbance"]
        # Keep the legacy reporting attributes populated for the inherited
        # reset/logging code; progression itself remains owned by the neutral
        # controller above.
        sim.domain_rand_phase = self.domain_rand_curriculum.phase
        sim.domain_rand_reward_ema = self.domain_rand_curriculum.reward_ema
        sim.required_reward = self.domain_rand_curriculum.required_reward
        sim.mass_max_value = ranges["added_base_mass"][1]
        sim.com_delta_x_value = max(abs(v) for v in ranges["base_com_x"])
        sim.com_delta_y_value = max(abs(v) for v in ranges["base_com_y"])
        sim.com_delta_z_value = max(abs(v) for v in ranges["base_com_z"])
        sim.com_delta_z_val_bounds = list(ranges["base_com_z"])
        sim.joint_friction_bound_current = list(ranges["joint_friction"])
        sim.joint_stiffness_bound_current = list(ranges["joint_stiffness"])
        sim.joint_damping_bound_current = list(ranges["joint_damping"])
        sim.push_value = max(abs(v) for v in ranges["push_xy"])
        sim.vert_value = abs(ranges["push_z"][0])
        sim.angular_push_value = max(abs(v) for v in ranges["push_angular"])

    def _refresh_domain_rand_capability_report(self):
        """Expose requested/effective ranges without overstating a backend.

        A backend may report individual implemented APIs while keeping the
        aggregate support flag false until its real GPU curriculum smoke has
        passed.  This makes unsupported/reset-only behavior visible instead of
        silently treating a requested range as effective.
        """
        capabilities = self.simulator.hard_pact_capabilities()
        self.supports_domain_rand_curriculum = bool(
            capabilities.get("supports_domain_rand_curriculum", False)
        )
        self.domain_rand_capability_report = self.domain_rand_curriculum.report(
            capabilities.get("features", {})
        )

    def step_domain_rand_curriculum(self, iteration, mean_reward=None):
        """Advance once per PPO iteration and update backend reset ranges."""
        changed = self.domain_rand_curriculum.advance(iteration, mean_reward)
        self._apply_domain_rand_curriculum_state()
        self._refresh_domain_rand_capability_report()
        return changed

    def domain_rand_curriculum_state_dict(self):
        return self.domain_rand_curriculum.state_dict()

    def load_domain_rand_curriculum_state_dict(self, state):
        self.domain_rand_curriculum.load_state_dict(state)
        self._apply_domain_rand_curriculum_state()
        self._refresh_domain_rand_capability_report()

    def configure_hard_pact_substep_qp(self, actor_critic, bard_dynamics, qp):
        """Bind shared inference objects without duplicating model memory."""
        self._hard_pact_actor_critic = actor_critic
        self._hard_pact_bard_dynamics = bard_dynamics
        self._hard_pact_rollout_qp = qp
        self._hard_pact_rollout_qp_enabled = (
            self.hard_pact_features.execution_qp
            and bard_dynamics is not None and qp is not None
        )

    def set_hard_pact_policy_context(self, latent, explicit):
        """Hold policy-rate features and wrench prediction for one interval."""
        if not self._hard_pact_rollout_qp_enabled:
            return
        # Rollout is inference-only.  Explicitly detach and use float32 so no
        # graph or autocast activation survives across simulator substeps.
        self._hard_pact_policy_latent = latent.detach().float()
        self._hard_pact_policy_explicit = explicit.detach().float()
        with torch.no_grad():
            self._hard_pact_wrench_yaw_scaled = (
                self._hard_pact_actor_critic.physics_estimator.predict_wrench(
                    self._hard_pact_policy_latent,
                    self._hard_pact_policy_explicit,
                ).detach()
            )
        self._hard_pact_policy_context_ready = True

    @staticmethod
    def _yaw_local_to_world(values, quaternion_xyzw):
        x, y, z, w = quaternion_xyzw.unbind(-1)
        yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y.square() + z.square()))
        cosine, sine = torch.cos(yaw), torch.sin(yaw)
        result = values.clone()
        result[..., 0] = cosine[:, None] * values[..., 0] - sine[:, None] * values[..., 1]
        result[..., 1] = sine[:, None] * values[..., 0] + cosine[:, None] * values[..., 1]
        return result

    @staticmethod
    def _body_point_to_world(point_body, quaternion_xyzw):
        x, y, z, w = quaternion_xyzw.unbind(-1)
        rotation = torch.stack((
            1 - 2 * (y*y + z*z), 2 * (x*y - w*z), 2 * (x*z + w*y),
            2 * (x*y + w*z), 1 - 2 * (x*x + z*z), 2 * (y*z - w*x),
            2 * (x*z - w*y), 2 * (y*z + w*x), 1 - 2 * (x*x + y*y),
        ), dim=-1).reshape(-1, 3, 3)
        return torch.einsum("bij,bj->bi", rotation, point_body)

    def _hard_pact_grf_post_physics_substep(self):
        self.grf_processor.update_substep(
            self.simulator.hard_pact_foot_forces_world()
        )

    def _persistent_component_settings(self, component):
        cfg = self.cfg.domain_rand
        progress = float(getattr(
            self.simulator, "domain_rand_disturbance_progress", 0.0
        ))
        if component == 0:
            minimum = float(cfg.persistent_force_min_n)
            maximum = float(cfg.persistent_force_max_n)
            return (
                float(cfg.persistent_force_probability),
                cfg.persistent_force_interval_range_s,
                cfg.persistent_force_duration_range_s,
                minimum + progress * (maximum - minimum),
                slice(0, 3),
            )
        minimum = float(cfg.persistent_torque_min_nm)
        maximum = float(cfg.persistent_torque_max_nm)
        return (
            float(cfg.persistent_torque_probability),
            cfg.persistent_torque_interval_range_s,
            cfg.persistent_torque_duration_range_s,
            minimum + progress * (maximum - minimum),
            slice(3, 6),
        )

    def _sample_step_range(self, seconds_range, count):
        low = max(1, int(round(float(seconds_range[0]) / self.dt)))
        high = max(low, int(round(float(seconds_range[1]) / self.dt)))
        return torch.randint(low, high + 1, (count,), device=self.device)

    def _schedule_next_persistent_event(self, env_ids, component, control_step):
        if env_ids.numel() == 0:
            return
        interval = self._persistent_component_settings(component)[1]
        self._persistent_next_event_step[env_ids, component] = (
            control_step + self._sample_step_range(interval, env_ids.numel())
        )

    def _start_due_persistent_events(self, component, control_step):
        probability, _, duration_range, magnitude, values = (
            self._persistent_component_settings(component)
        )
        due = torch.nonzero(
            (~self._persistent_component_active[:, component])
            & (self._persistent_next_event_step[:, component] <= control_step),
            as_tuple=False,
        ).flatten()
        self._schedule_next_persistent_event(due, component, control_step)
        if due.numel() == 0 or magnitude <= 0.0:
            return
        selected = due[
            torch.rand(due.numel(), device=self.device) < probability
        ]
        if selected.numel() == 0:
            return
        duration = self._sample_step_range(duration_range, selected.numel())
        self._persistent_component_active[selected, component] = True
        self._persistent_start_step[selected, component] = control_step
        self._persistent_duration_steps[selected, component] = duration
        self._persistent_end_step[selected, component] = control_step + duration
        self._persistent_wrench_target_world[selected, values] = (
            torch.empty(selected.numel(), 3, device=self.device).uniform_(
                -magnitude, magnitude
            )
        )

    def _update_persistent_wrench(self, control_step):
        self._current_sustained_wrench_world.zero_()
        cfg = self.cfg.domain_rand
        if not bool(cfg.persistent_disturbance):
            self._persistent_component_active.zero_()
            self._current_sustained_active_mask.zero_()
            return
        for component in range(2):
            self._start_due_persistent_events(component, control_step)
            values = slice(component * 3, component * 3 + 3)
            active = self._persistent_component_active[:, component]
            elapsed = control_step - self._persistent_start_step[:, component]
            remaining = self._persistent_end_step[:, component] - control_step
            duration = self._persistent_duration_steps[:, component].clamp_min(1)
            ramp_steps = (
                duration.float() * float(cfg.persistent_ramp_fraction)
            ).clamp_min(1.0)
            amplitude = torch.minimum(
                elapsed.float() / ramp_steps,
                remaining.float() / ramp_steps,
            ).clamp_(0.0, 1.0)
            self._current_sustained_wrench_world[:, values] = torch.where(
                active.unsqueeze(-1),
                self._persistent_wrench_target_world[:, values]
                * amplitude.unsqueeze(-1),
                torch.zeros_like(self._persistent_wrench_target_world[:, values]),
            )
            finished = active & (
                control_step >= self._persistent_end_step[:, component]
            )
            if finished.any():
                self._persistent_component_active[finished, component] = False
                self._persistent_wrench_target_world[finished, values] = 0
                self._current_sustained_wrench_world[finished, values] = 0
        self._current_sustained_active_mask.copy_(
            self._persistent_component_active.any(dim=-1, keepdim=True)
        )

    def _reset_persistent_wrench_state(self, env_ids):
        if env_ids.numel() == 0:
            return
        for value in (
            self._persistent_wrench_target_world,
            self._current_sustained_wrench_world,
            self._persistent_component_active,
            self._current_sustained_active_mask,
            self._persistent_start_step,
            self._persistent_end_step,
            self._persistent_duration_steps,
        ):
            value[env_ids] = 0
        for component in range(2):
            self._schedule_next_persistent_event(
                env_ids, component, self.common_step_counter
            )

    def _current_base_quat_xyzw(self):
        accessor = getattr(self.simulator, "hard_pact_base_quat_xyzw", None)
        if accessor is not None:
            return accessor()
        robot = self.simulator._robot
        return robot.get_quat()[:, (1, 2, 3, 0)]

    def _canonical_joint_state(self):
        accessor = getattr(self.simulator, "hard_pact_joint_state", None)
        if accessor is not None:
            return accessor()
        # Retain compatibility with the lightweight legacy simulator mocks.
        if hasattr(self.simulator, "_dof_pos"):
            return self.simulator._dof_pos, self.simulator._dof_vel
        robot = self.simulator._robot
        return (robot.get_dofs_position(self.simulator._dof_indices),
                robot.get_dofs_velocity(self.simulator._dof_indices))

    def _canonical_configuration(self, quat_override=None):
        accessor = getattr(self.simulator, "hard_pact_configuration", None)
        if accessor is not None:
            configuration = accessor()
            if quat_override is None:
                return configuration
            return torch.cat((configuration[:, :3], quat_override,
                              configuration[:, 7:]), -1)
        q, _ = self._canonical_joint_state()
        quaternion = (
            self._current_base_quat_xyzw()
            if quat_override is None else quat_override
        )
        return torch.cat((self.simulator._robot.get_pos(),
                          quaternion, q), -1)

    def _canonical_velocity_world(self):
        accessor = getattr(self.simulator, "hard_pact_velocity_world", None)
        if accessor is not None:
            return accessor()
        _, qd = self._canonical_joint_state()
        robot = self.simulator._robot
        return torch.cat((robot.get_vel(), robot.get_ang(), qd), -1)

    def _canonical_randomized_parameters(self):
        accessor = getattr(self.simulator, "hard_pact_randomized_parameters", None)
        if accessor is not None:
            return accessor()
        sim = self.simulator
        return {
            "added_base_mass": self._realized_added_mass,
            "base_com_shift": self._realized_com_shift_body,
            "joint_armature": sim._joint_armature,
            "joint_friction": sim._joint_friction,
            "joint_stiffness": sim._joint_stiffness,
            "joint_damping": sim._joint_damping,
        }

    def _apply_sustained_world_wrench(self, wrench_world):
        """The only simulator external-wrench call made by HardPACT."""
        apply = getattr(self.simulator, "hard_pact_apply_base_wrench_world", None)
        if apply is not None:
            apply(wrench_world)
            return
        # Compatibility for the tiny legacy unit-test simulator mock.
        base = torch.as_tensor([self.simulator._base_link_index], device=self.device)
        solver = self.simulator._robot._solver
        solver.apply_links_external_force(
            force=wrench_world[:, :3].unsqueeze(1), links_idx=base,
            envs_idx=None, ref="link_com", local=False,
        )
        solver.apply_links_external_torque(
            torque=wrench_world[:, 3:].unsqueeze(1), links_idx=base,
            envs_idx=None, ref="link_com", local=False,
        )

    def _hard_pact_pre_physics_substep(self):
        quat = self._current_base_quat_xyzw()
        mass_com_wrench = added_mass_gravity_wrench_world(
            self._realized_added_mass,
            getattr(self.cfg.sim, "gravity", (0.0, 0.0, -9.81)),
            self._realized_com_shift_body,
            quat,
        )
        total_wrench = self._current_sustained_wrench_world + mass_com_wrench
        sustained_yaw_scaled = wrench_world_to_scaled_yaw_local(
            self._current_sustained_wrench_world,
            quat,
            self.obs_scales.base_wrench,
        )
        mass_com_yaw_scaled = wrench_world_to_scaled_yaw_local(
            mass_com_wrench, quat, self.obs_scales.base_wrench
        )
        yaw_scaled = wrench_world_to_scaled_yaw_local(
            total_wrench, quat, self.obs_scales.base_wrench
        )
        self._disturbance_interval_sum_sustained.add_(
            self._current_sustained_wrench_world
        )
        self._disturbance_interval_sum_mass_com.add_(mass_com_wrench)
        self._disturbance_interval_sum_total.add_(total_wrench)
        self._disturbance_interval_sum_sustained_yaw_scaled.add_(
            sustained_yaw_scaled
        )
        self._disturbance_interval_sum_mass_com_yaw_scaled.add_(
            mass_com_yaw_scaled
        )
        self._disturbance_interval_sum_yaw_scaled.add_(yaw_scaled)
        self._disturbance_interval_count.add_(1.0)
        if (
            getattr(self, "_hard_pact_rollout_qp_enabled", False)
            and getattr(self, "_hard_pact_policy_context_ready", False)
        ):
            self._solve_hard_pact_rollout_qp_substep(quat, mass_com_wrench)
        # `_torques` is now the exact safe command about to be sent to
        # Genesis.  The interval value remains the authoritative BARD torque.
        if hasattr(self, "_interval_executed_torque_sum"):
            getter = getattr(self.simulator, "hard_pact_executed_torque", None)
            executed = getter() if getter is not None else self.simulator._torques
            self._interval_executed_torque_sum.add_(executed)
            self._interval_executed_torque_count.add_(1.0)
            self._interval_executed_torque_peak.copy_(torch.maximum(
                self._interval_executed_torque_peak,
                executed.abs(),
            ))
        # The equivalent inertial wrench is deliberately label-only: Genesis
        # already realizes the randomized mass and CoM in its dynamics.
        self._apply_sustained_world_wrench(self._current_sustained_wrench_world)

    def _solve_hard_pact_rollout_qp_substep(self, quat, mass_com_wrench):
        r"""Refresh and solve one inference-only QP immediately before actuation.

        ``q_d`` and ``tau_ff`` are held across decimation, while PD feedback
        and the torque-rate box always use the current physics-substep state.
        In the default mode BARD/QP/head inputs are refreshed every substep.
        The optional two-anchor mode refreshes BARD/QP only at ``k={0,2}``
        and holds its control-step GRF/head features plus anchor correction.

        At substep ``k`` this path computes

        .. math::

           \tau_{nom,k}=K_p(q_d-q_k)-K_d\dot q_k+\tau_{ff},

           \hat f_k=D_F(z_t,\operatorname{sg}(e_t),\tau_{nom,k}),

        then solves ``x_k*=QP(q_k,v_k,tau_{nom,k},f_hat_k,W_hat_t,
        tau_safe,k-1)`` and sends only ``tau_safe,k`` to Genesis. The outer
        ``no_grad`` is intentional: rollout needs numeric safety decisions,
        whereas PPO later rebuilds one selected substep with autograd enabled.
        """
        # Never retain qpth/BARD/head graphs for D rollout substeps.
        with torch.no_grad():
            # Local alias shortens all live-state reads below.
            simulator = self.simulator
            # Refresh actuated q_k directly from Genesis immediately before
            # actuation; the control-rate cached state may be one substep old.
            joint_position, joint_velocity = self._canonical_joint_state()
            # Refresh qdot_k for the same reason and timestamp.
            # Simulator configuration q_sim=[p_WB(3),quat_xyzw(4),q_joints(12)].
            q_simulator = self._canonical_configuration(quat)
            # Simulator velocity v_sim=[v_WB^W(3),omega_WB^W(3),qdot(12)].
            v_world = self._canonical_velocity_world()

            # tau_nom,k = Kp(q_d-q_k)-Kd*qdot_k+tau_ff.  The legacy feedback
            # helper supplies the task's exact gains/default-pose convention.
            tau_nom = self._hard_pact_tau_ff + self._get_pinn_feedback(
                self._hard_pact_q_d, joint_position, joint_velocity
            )
            update_mode = getattr(
                self._hard_pact_rollout_qp.cfg,
                "qp_update_mode", "every_substep",
            )
            # z_t and e_t remain policy-rate values. The default retains its
            # legacy per-substep GRF evaluation; two-anchor mode evaluates the
            # torque-conditioned decoder only at k=0 and holds that prediction.
            if (
                update_mode == "two_anchor_held_correction"
                and self._qp_substep > 0
            ):
                grf_yaw_scaled = self._hard_pact_held_grf_yaw_scaled
            else:
                grf_yaw_scaled = (
                    self._hard_pact_actor_critic.physics_estimator.predict_grf(
                        self._hard_pact_policy_latent,
                        self._hard_pact_policy_explicit,
                        tau_nom,
                    )
                )
                if update_mode == "two_anchor_held_correction":
                    self._hard_pact_held_grf_yaw_scaled.copy_(grf_yaw_scaled)
            # Decoder output is observation-scaled and yaw-local. Divide by
            # obs scale to recover Newtons, reshape FR/FL/RR/RL XYZ, and rotate
            # yaw-local axes into the world axes used by J_f.
            grf_world = self._yaw_local_to_world(
                (grf_yaw_scaled / float(self.obs_scales.grf)).reshape(-1, 4, 3),
                quat,
            )
            # The wrench head is policy-rate and was evaluated once when the
            # action was chosen. Recover physical [F,T] units here.
            wrench_yaw = self._hard_pact_wrench_yaw_scaled / float(
                self.obs_scales.base_wrench
            )
            # Rotate both force and moment from yaw-local to world axes while
            # preserving ordering [Fx,Fy,Fz,Tx,Ty,Tz].
            total_wrench_world = torch.cat((
                self._yaw_local_to_world(wrench_yaw[:, :3].unsqueeze(1), quat).squeeze(1),
                self._yaw_local_to_world(wrench_yaw[:, 3:].unsqueeze(1), quat).squeeze(1),
            ), dim=-1)
            # Projection is deployment-facing. It therefore uses the total
            # predicted wrench about the nominal base reference directly and
            # never reads the privileged realized mass/CoM label.
            applied_wrench = total_wrench_world
            is_anchor = self._qp_substep in self._hard_pact_qp_anchors
            if update_mode == "two_anchor_held_correction" and not is_anchor:
                # Hold only delta_tau from the preceding anchor. PD feedback
                # above remains live at every physics substep. The final
                # analytic projection enforces both actuator magnitude and
                # the rate box relative to the actually executed substep k-1.
                torque_limit = self._hard_pact_rollout_qp.torque_limits.to(
                    device=tau_nom.device, dtype=tau_nom.dtype
                )
                rate = (
                    self._hard_pact_rollout_qp.cfg.torque_rate_limit_nm_s
                    * float(self.cfg.sim.dt)
                )
                lower = torch.maximum(
                    -torque_limit,
                    self._hard_pact_previous_substep_torque - rate,
                )
                upper = torch.minimum(
                    torque_limit,
                    self._hard_pact_previous_substep_torque + rate,
                )
                requested = tau_nom + self._hard_pact_held_correction
                safe = torch.maximum(torch.minimum(requested, upper), lower)
                setter = getattr(
                    simulator, "hard_pact_set_executed_torque", None
                )
                if setter is None:
                    simulator._torques = safe
                else:
                    setter(safe)
                correction = safe - tau_nom
                self._qp_interval_safe_sum.add_(safe)
                self._qp_interval_safe_peak.copy_(torch.maximum(
                    self._qp_interval_safe_peak, safe.abs()
                ))
                self._qp_interval_correction_sum.add_(correction)
                self._qp_interval_correction_peak.copy_(torch.maximum(
                    self._qp_interval_correction_peak, correction.abs()
                ))
                self._qp_interval_grf_sum.add_(
                    self._hard_pact_held_force_world
                )
                self._qp_interval_slack_sum.add_(self._hard_pact_held_slack)
                self._qp_interval_slack_peak.copy_(torch.maximum(
                    self._qp_interval_slack_peak,
                    self._hard_pact_held_slack,
                ))
                self._qp_interval_residual_sum.add_(
                    self._hard_pact_held_residual
                )
                self._qp_interval_residual_peak.copy_(torch.maximum(
                    self._qp_interval_residual_peak,
                    self._hard_pact_held_residual,
                ))
                self._qp_interval_stage_counts.scatter_add_(
                    1, self._hard_pact_held_stage[:, None],
                    torch.ones(self.num_envs, 1, device=self.device),
                )
                self._hard_pact_previous_substep_torque.copy_(safe)
                self._qp_substep += 1
                return
            # Empty parameters select URDF/nominal mechanics. Actual realized
            # randomization is reserved for the PINN-target mechanics cache.
            parameters = {}
            # One kinematic update builds M(q_k), b(q_k,v_k), J_f(q_k),
            # J_b(q_k), and Jdot_f(q_k,v_k)*v_k for this substep.
            context = self._hard_pact_bard_dynamics.build_context(
                q_simulator, v_world, parameters=parameters, need_qp=True
            )
            # Wall-clock measurement encloses matrix assembly inside solve and
            # qpth/fallback execution, but excludes state/head preprocessing.
            start = time.perf_counter()
            # The solver constructs min 1/2*x'Qx+p'x subject to Gx<=h, Ax=b,
            # where x=[qdd,f,tau_safe,s]. Every argument below maps directly
            # to a documented physical block in hard_pact_qp.py.
            result = self._hard_pact_rollout_qp.solve(
                differentiable=False,
                # M multiplies generalized acceleration in A[:,QDD].
                mass_matrix=context.mass_matrix,
                # b_dyn moves to the equality RHS as J_b^T*W-b_dyn.
                bias=context.bias,
                # J_f supplies -J_f^T in dynamics and J_f in contact rows.
                foot_jacobians=context.foot_jacobians,
                # J_b maps the predicted applied wrench into generalized force.
                base_jacobian=context.base_jacobian,
                # Jdot_f*v is the affine foot-acceleration contribution.
                foot_acceleration_bias=context.foot_acceleration_bias,
                # Tracking center for tau_safe and source of actor gradients in PPO.
                tau_nom=tau_nom,
                # Tracking center for the QP force decision, in world Newtons.
                force_pred_world=grf_world,
                # Fixed generalized-force RHS input, world [N,Nm] at J_b point.
                wrench_pred_world=applied_wrench,
                # e_t[3:7] gates contact acceleration continuously in [0,1].
                contact_probability=self._hard_pact_policy_explicit[:, 3:7],
                # tau_safe,k-1 centers the hard rate box for this substep.
                previous_torque=self._hard_pact_previous_substep_torque,
                # q_k/qdot_k form one-step joint position/velocity constraints.
                joint_position=joint_position,
                joint_velocity=joint_velocity,
                # Hard rate and one-step integration limits use physics dt,
                # never the D-times-larger policy/control interval.
                dt=torch.full(
                    (self.num_envs, 1), float(self.cfg.sim.dt),
                    device=self.device, dtype=tau_nom.dtype,
                ),
                previous_certified_qdd=self._hard_pact_previous_certified_qdd,
                proximal_reference=torch.cat((
                    self._hard_pact_previous_certified_qdd,
                    grf_world.flatten(1), tau_nom,
                    torch.zeros_like(grf_world.flatten(1)),
                ), dim=-1).detach(),
            )
            # Record elapsed milliseconds for interval diagnostics.
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            # This is the only torque command sent to Genesis for substep k.
            setter = getattr(simulator, "hard_pact_set_executed_torque", None)
            if setter is None:
                simulator._torques = result.tau_safe
            else:
                setter(result.tau_safe)
            # Delta_tau_k is the projection correction used by L_proj/logging.
            correction = result.tau_safe - tau_nom
            # Flatten [foot,XYZ] slack to its fixed 12-D transition ordering.
            slack = result.contact_slack.reshape(self.num_envs, -1)
            # Pack certified infinity-norm residuals as [eq,ineq,dual,comp].
            # Dual/complementarity values exist only on periodic full audits;
            # zero is the neutral interval-aggregation placeholder otherwise.
            optional_zero = torch.zeros_like(
                result.diagnostics["selected/equality_max"]
            )
            residual = torch.stack((
                result.diagnostics["selected/equality_max"],
                result.diagnostics["selected/inequality_max"],
                result.diagnostics.get(
                    "selected/stationarity_max", optional_zero
                ),
                result.diagnostics.get(
                    "selected/complementarity_max", optional_zero
                ),
            ), dim=-1).nan_to_num()
            if update_mode == "two_anchor_held_correction":
                self._hard_pact_held_correction.copy_(
                    result.tau_safe - tau_nom
                )
                self._hard_pact_held_force_world.copy_(
                    result.force_world.flatten(1)
                )
                self._hard_pact_held_qdd.copy_(result.qdd)
                self._hard_pact_held_slack.copy_(slack)
                self._hard_pact_held_residual.copy_(residual)
                self._hard_pact_held_stage.copy_(result.stage)
                self._hard_pact_held_differentiated.copy_(
                    result.differentiated_mask
                )
            # Accumulate sum/absolute peak of executed safe torque.
            self._qp_interval_safe_sum.add_(result.tau_safe)
            self._qp_interval_safe_peak.copy_(torch.maximum(
                self._qp_interval_safe_peak, result.tau_safe.abs()
            ))
            # Accumulate signed mean correction and absolute peak correction.
            self._qp_interval_correction_sum.add_(correction)
            self._qp_interval_correction_peak.copy_(torch.maximum(
                self._qp_interval_correction_peak, correction.abs()
            ))
            # QP-selected physical GRFs are averaged over the interval.
            self._qp_interval_grf_sum.add_(result.force_world.flatten(1))
            # Slack is nonnegative, so ordinary max is its physical peak.
            self._qp_interval_slack_sum.add_(slack)
            self._qp_interval_slack_peak.copy_(torch.maximum(
                self._qp_interval_slack_peak, slack
            ))
            # Residual means and peaks summarize numerical solve quality.
            self._qp_interval_residual_sum.add_(residual)
            self._qp_interval_residual_peak.copy_(torch.maximum(
                self._qp_interval_residual_peak, residual
            ))
            # stage in {0,1,2}; scatter_add forms per-env fallback counts.
            self._qp_interval_stage_counts.scatter_add_(
                1, result.stage[:, None],
                torch.ones(self.num_envs, 1, device=self.device),
            )
            # Each env shares this batched wall time; averaging later reports
            # milliseconds per batched substep solve.
            self._qp_interval_timing_ms.add_(elapsed_ms)

            # Each environment stores exactly one preselected replay point:
            # k in [0,D) for the default, or a balanced k in {0,2} here.
            selected = self._qp_sampled_substep_index.long() == self._qp_substep
            if selected.any():
                # Store compact vectors only. M/J/A/G/Q are rebuilt during PPO
                # and never occupy [rollout_length,decimation] GPU storage.
                sample = self._qp_sampled_transition
                # State required to reconstruct BARD mechanics at sampled k.
                sample["sampled_qp_q"][selected] = q_simulator[selected]
                sample["sampled_qp_v"][selected] = v_world[selected]
                # Exact rate-box center tau_safe,k-1.
                sample["sampled_qp_previous_torque"][selected] = (
                    self._hard_pact_previous_substep_torque[selected]
                )
                sample["sampled_qp_proximal_reference"][selected] = torch.cat((
                    self._hard_pact_previous_certified_qdd[selected],
                    grf_world[selected].flatten(1), tau_nom[selected],
                    torch.zeros_like(grf_world[selected].flatten(1)),
                ), dim=-1)
                # Rollout references permit frozen-policy equality diagnostics.
                sample["sampled_qp_rollout_nominal_torque"][selected] = tau_nom[selected]
                sample["sampled_qp_rollout_grf_world"][selected] = (
                    grf_world[selected].flatten(1)
                )
                # Label-only term must be subtracted once during PPO replay.
                sample["sampled_qp_mass_com_wrench_world"][selected] = (
                    mass_com_wrench[selected]
                )
                sample["sampled_qp_rollout_applied_wrench_world"][selected] = (
                    applied_wrench[selected]
                )
                sample["sampled_qp_rollout_contact_probability"][selected] = (
                    self._hard_pact_policy_explicit[selected, 3:7]
                )
                # Store all primal blocks and solver certification metadata.
                sample["sampled_qp_qdd"][selected] = result.qdd[selected]
                sample["sampled_qp_force_world"][selected] = result.force_world[selected].flatten(1)
                sample["sampled_qp_safe_torque"][selected] = result.tau_safe[selected]
                sample["sampled_qp_contact_slack"][selected] = slack[selected]
                sample["sampled_qp_stage"][selected] = (
                    result.stage[selected, None].to(torch.int16)
                )
                sample["sampled_qp_differentiated"][selected] = (
                    result.differentiated_mask[selected, None]
                )
                sample["sampled_qp_residuals"][selected] = residual[selected]
                sample["sampled_qp_timing_ms"][selected] = elapsed_ms
            # Advance the rate constraint: next substep uses this exact command.
            self._hard_pact_previous_substep_torque.copy_(result.tau_safe)
            self._hard_pact_previous_certified_qdd.copy_(torch.where(
                result.differentiated_mask[:, None], result.qdd,
                self._hard_pact_previous_certified_qdd,
            ))
            # Advance k after sampling so the first callback is k=0.
            self._qp_substep += 1

    def _begin_disturbance_interval(self):
        for value in (
            self._disturbance_interval_sum_sustained,
            self._disturbance_interval_sum_mass_com,
            self._disturbance_interval_sum_total,
            self._disturbance_interval_sum_sustained_yaw_scaled,
            self._disturbance_interval_sum_mass_com_yaw_scaled,
            self._disturbance_interval_sum_yaw_scaled,
            self._disturbance_interval_count,
        ):
            value.zero_()
        if hasattr(self, "_interval_executed_torque_sum"):
            self._interval_executed_torque_sum.zero_()
            self._interval_executed_torque_count.zero_()
            self._interval_executed_torque_peak.zero_()
        if (
            getattr(self, "_hard_pact_rollout_qp_enabled", False)
            and getattr(self, "_hard_pact_policy_context_ready", False)
        ):
            self._begin_qp_interval()

    def _begin_qp_interval(self):
        r"""Allocate interval statistics and one sampled replay row.

        For decimation ``D``, sums become ``(1/D)sum_k value_k`` at interval
        finalization and peaks become ``max_k |value_k|``. For each environment
        a single replay point is chosen with balanced strata. It is uniform
        over ``{0,...,D-1}`` in the default mode and over anchors ``{0,2}``
        in ``two_anchor_held_correction`` mode.
        """
        # All physical/diagnostic values use float32 on the simulation device.
        shape = lambda width: torch.zeros(
            self.num_envs, width, device=self.device, dtype=torch.float32
        )
        # Executed safe torque sum and absolute componentwise peak [Nm].
        self._qp_interval_safe_sum = shape(12)
        self._qp_interval_safe_peak = shape(12)
        # Signed correction sum and absolute peak, tau_safe-tau_nom [Nm].
        self._qp_interval_correction_sum = shape(12)
        self._qp_interval_correction_peak = shape(12)
        # QP primal force sum in FR/FL/RR/RL world XYZ [N].
        self._qp_interval_grf_sum = shape(12)
        # Contact-acceleration slack sum and componentwise peak [m/s^2].
        self._qp_interval_slack_sum = shape(12)
        self._qp_interval_slack_peak = shape(12)
        # [equality, inequality, stationarity, complementarity] residuals.
        self._qp_interval_residual_sum = shape(4)
        self._qp_interval_residual_peak = shape(4)
        # Counts for stage 0/full, stage 1/relaxed, stage 2/projection.
        if not hasattr(self, "_hard_pact_previous_certified_qdd"):
            self._hard_pact_previous_certified_qdd = torch.zeros(
                self.num_envs, 18, device=self.device
            )
        stage_count = 4 if bool(getattr(
            getattr(self, "_hard_pact_rollout_qp", None), "cfg", None
        ) and self._hard_pact_rollout_qp.cfg.elastic_recovery_enabled) else 3
        self._qp_interval_stage_counts = shape(stage_count)
        # Sum of batched QP wall-clock milliseconds across substeps.
        self._qp_interval_timing_ms = shape(1)
        # First callback in the interval is physics substep k=0.
        self._qp_substep = 0
        # D is the number of Genesis physics steps per policy action.
        decimation = int(self.cfg.control.decimation)
        update_mode = getattr(
            self._hard_pact_rollout_qp.cfg, "qp_update_mode", "every_substep"
        )
        if update_mode == "two_anchor_held_correction":
            if decimation != 4:
                raise ValueError(
                    "two_anchor_held_correction requires control decimation=4"
                )
            self._hard_pact_qp_anchors = (0, 2)
            self._qp_sampled_substep_index = balanced_anchor_indices(
                self.num_envs, self._hard_pact_qp_anchors, self.device
            )
            self._hard_pact_held_correction = shape(12)
            self._hard_pact_held_grf_yaw_scaled = shape(12)
            self._hard_pact_held_force_world = shape(12)
            self._hard_pact_held_qdd = shape(18)
            self._hard_pact_held_slack = shape(12)
            self._hard_pact_held_residual = shape(4)
            self._hard_pact_held_stage = torch.zeros(
                self.num_envs, device=self.device, dtype=torch.long
            )
            self._hard_pact_held_differentiated = torch.zeros(
                self.num_envs, device=self.device, dtype=torch.bool
            )
        else:
            self._hard_pact_qp_anchors = tuple(range(decimation))
            # One int16 index per env is the only decimation-dependent replay key.
            self._qp_sampled_substep_index = balanced_substep_indices(
                self.num_envs, decimation, self.device
            )
        # Preallocate exactly one compact sampled row per env. Dynamics/QP
        # matrices are intentionally absent and reconstructed during PPO.
        self._qp_sampled_transition = {
            # Which k generated every stored field below.
            "sampled_qp_substep_index": self._qp_sampled_substep_index[:, None],
            # Canonical simulator configuration and world/base twist state.
            "sampled_qp_q": shape(19), "sampled_qp_v": shape(18),
            # Hard rate-box center and rollout learned references.
            "sampled_qp_previous_torque": shape(12),
            "sampled_qp_proximal_reference": shape(54),
            "sampled_qp_rollout_nominal_torque": shape(12),
            "sampled_qp_rollout_grf_world": shape(12),
            # Label-only mass wrench and resulting applied-wrench QP input.
            "sampled_qp_mass_com_wrench_world": shape(6),
            "sampled_qp_rollout_applied_wrench_world": shape(6),
            # Held explicit-estimator contact probabilities c_i.
            "sampled_qp_rollout_contact_probability": shape(4),
            # Complete rollout primal x*=[qdd,f,tau_safe,s].
            "sampled_qp_qdd": shape(18), "sampled_qp_force_world": shape(12),
            "sampled_qp_safe_torque": shape(12),
            "sampled_qp_contact_slack": shape(12),
            # Compact fallback stage and whether qpth supplied its KKT graph.
            "sampled_qp_stage": torch.zeros(
                self.num_envs, 1, device=self.device, dtype=torch.int16
            ),
            "sampled_qp_differentiated": torch.zeros(
                self.num_envs, 1, device=self.device, dtype=torch.bool
            ),
            # Numeric certification vector and elapsed batched solve time.
            "sampled_qp_residuals": shape(4),
            "sampled_qp_timing_ms": shape(1),
        }

    def _capture_bard_pre_state(self):
        simulator = self.simulator
        self._bard_pre_q_simulator = self._canonical_configuration().clone()
        self._bard_pre_v_world_simulator = self._canonical_velocity_world().clone()

    def _capture_bard_post_state(self):
        simulator = self.simulator
        self._bard_post_v_world_simulator = self._canonical_velocity_world().clone()

    def _post_physics_step_callback(self):
        """Observe legacy push events without changing legacy push logic."""
        root_velocity = getattr(
            self.simulator, "hard_pact_root_velocity_world", None
        )
        velocity_before = (
            root_velocity() if root_velocity is not None
            else self.simulator._robot.get_dofs_velocity()[:, :6]
        ).clone()
        self._legacy_task_class._post_physics_step_callback(self)
        velocity_after = (
            root_velocity() if root_velocity is not None
            else self.simulator._robot.get_dofs_velocity()[:, :6]
        )
        self._hard_pact_push_event_mask.copy_(
            (velocity_after != velocity_before).any(dim=-1, keepdim=True)
        )

    def _end_disturbance_interval(self):
        divisor = self._disturbance_interval_count.clamp_min(1.0)
        return {
            "applied_sustained_wrench_world": (
                self._disturbance_interval_sum_sustained / divisor
            ).clone(),
            "sustained_wrench_active_mask": (
                self._current_sustained_active_mask.clone()
            ),
            "equivalent_mass_com_wrench_world": (
                self._disturbance_interval_sum_mass_com / divisor
            ).clone(),
            "applied_sustained_wrench_yaw_scaled": (
                self._disturbance_interval_sum_sustained_yaw_scaled / divisor
            ).clone(),
            "equivalent_mass_com_wrench_yaw_scaled": (
                self._disturbance_interval_sum_mass_com_yaw_scaled / divisor
            ).clone(),
            "total_external_wrench_label_world": (
                self._disturbance_interval_sum_total / divisor
            ).clone(),
            "total_external_wrench_label_yaw_scaled": (
                self._disturbance_interval_sum_yaw_scaled / divisor
            ).clone(),
            "realized_added_mass": self._realized_added_mass.clone(),
            "realized_com_shift_body": self._realized_com_shift_body.clone(),
        }

    def _capture_realized_inertial_randomization(self):
        parameters = self._canonical_randomized_parameters()
        if bool(self.cfg.domain_rand.randomize_base_mass):
            self._realized_added_mass.copy_(parameters["added_base_mass"])
        else:
            self._realized_added_mass.zero_()
        if bool(self.cfg.domain_rand.randomize_com_displacement):
            self._realized_com_shift_body.copy_(parameters["base_com_shift"])
        else:
            self._realized_com_shift_body.zero_()

    def _build_disturbance_transition(self):
        pending = self._pending_disturbance_transition
        reset = self.reset_buf.bool().reshape(-1, 1)
        timeout = self.time_out_buf.bool().reshape(-1, 1)
        non_failure_reset = getattr(
            self, "non_failure_reset_buf", torch.zeros_like(self.reset_buf)
        )
        teleport = non_failure_reset.bool().reshape(-1, 1)
        push_event = self._hard_pact_push_event_mask.clone()
        torque_divisor = self._interval_executed_torque_count.clamp_min(1.0)
        fields = {
            "applied_sustained_wrench_world": pending["applied_sustained_wrench_world"],
            "sustained_wrench_active_mask": pending["sustained_wrench_active_mask"],
            "equivalent_mass_com_wrench_world": pending["equivalent_mass_com_wrench_world"],
            "total_external_wrench_label_world": pending["total_external_wrench_label_world"],
            "total_external_wrench_label_yaw_scaled": pending["total_external_wrench_label_yaw_scaled"],
            "realized_added_mass": pending["realized_added_mass"],
            "realized_com_shift_body": pending["realized_com_shift_body"],
            "pre_q": self._bard_pre_q_simulator.clone(),
            "pre_v": self._bard_pre_v_world_simulator.clone(),
            "post_v": self._bard_post_v_world_simulator.clone(),
            "control_dt": torch.full(
                (self.num_envs, 1), float(self.dt),
                device=self.device, dtype=torch.float32,
            ),
            "physics_dt": torch.full(
                (self.num_envs, 1), float(self.cfg.sim.dt),
                device=self.device, dtype=torch.float32,
            ),
            "interval_executed_torque": (
                self._interval_executed_torque_sum / torque_divisor
            ).clone(),
            "joint_armature": self._canonical_randomized_parameters()["joint_armature"].clone(),
            "joint_friction": self._canonical_randomized_parameters()["joint_friction"].clone(),
            "joint_stiffness": self._canonical_randomized_parameters()["joint_stiffness"].clone(),
            "joint_damping": self._canonical_randomized_parameters()["joint_damping"].clone(),
            "push_event_mask": push_event,
            "reset_mask": reset,
            "timeout_mask": timeout,
            "teleport_mask": teleport,
            "physics_valid_mask": physics_transition_mask(
                reset, timeout, teleport, push_event
            ),
        }
        if self._pending_action_replay_transition is not None:
            fields.update(self._pending_action_replay_transition)
        if (
            getattr(self, "_hard_pact_rollout_qp_enabled", False)
            and getattr(self, "_hard_pact_policy_context_ready", False)
        ):
            qp_divisor = self._interval_executed_torque_count.clamp_min(1.0)
            fields.update(self._qp_sampled_transition)
            # Interval diagnostics are exposed to logging but deliberately not
            # inserted into `fields`: RolloutStorage persists every named
            # field for T steps, and PPO only needs the single sampled row.
            self.extras["hard_pact_qp_interval"] = {
                "interval_qp_safe_torque": self._qp_interval_safe_sum / qp_divisor,
                "interval_peak_executed_torque": self._interval_executed_torque_peak,
                "interval_qp_peak_safe_torque": self._qp_interval_safe_peak,
                "interval_qp_correction": self._qp_interval_correction_sum / qp_divisor,
                "interval_qp_peak_correction": self._qp_interval_correction_peak,
                "interval_qp_grf_world": self._qp_interval_grf_sum / qp_divisor,
                "interval_qp_contact_slack": self._qp_interval_slack_sum / qp_divisor,
                "interval_qp_peak_contact_slack": self._qp_interval_slack_peak,
                "interval_qp_residuals": self._qp_interval_residual_sum / qp_divisor,
                "interval_qp_peak_residuals": self._qp_interval_residual_peak,
                "interval_qp_stage_fractions": self._qp_interval_stage_counts / qp_divisor,
                "interval_qp_timing_ms": self._qp_interval_timing_ms / qp_divisor,
            }
        # These deployment-facing views are named transition values but are
        # intentionally not duplicated in the critic: its required input is
        # the scaled yaw-local total label above.
        fields["applied_sustained_wrench_yaw_scaled"] = pending[
            "applied_sustained_wrench_yaw_scaled"
        ]
        fields["equivalent_mass_com_wrench_yaw_scaled"] = pending[
            "equivalent_mass_com_wrench_yaw_scaled"
        ]
        self.last_transition = fields
        self.extras["hard_pact_transition"] = fields
        return fields

    def _update_legacy_grfs_buf_input(self):
        """Optionally replace the legacy raw GRF input with deployment EMA."""
        if bool(self.cfg.sim.grf.use_ema_grfs_buf):
            self.simulator._grfs_buf.copy_(self.grf_processor.ema.flatten(1))

    def compute_observations(self):
        # simulator.post_physics_step() refreshes _grfs_buf immediately before
        # this method, making this the precise opt-in boundary for legacy
        # critic/decoder observation construction.
        self._update_legacy_grfs_buf_input()
        result = self._legacy_task_class.compute_observations(self)
        clearance = (
            self.simulator.feet_pos[:, :, 2]
            - torch.mean(self.simulator.height_around_feet, dim=-1)
            - self.cfg.rewards.foot_height_offset
        )
        self.explicit_labels_buf = compose_explicit_estimator_target(
            self.simulator.base_lin_vel * self.obs_scales.lin_vel,
            self.grf_processor.contacts.float(),
            clearance,
        )
        if getattr(self, "_pending_disturbance_transition", None) is not None:
            disturbance_frame = pack_disturbance_fields(
                self._build_disturbance_transition()
            )
            self.disturbance_critic_deque.append(disturbance_frame)
            legacy_frames = list(self.critic_obs_deque)
            disturbance_frames = list(self.disturbance_critic_deque)
            self.privileged_obs_buf = torch.cat(
                [torch.cat((legacy, disturbance), dim=-1)
                 for legacy, disturbance in zip(
                     legacy_frames, disturbance_frames
                 )],
                dim=-1,
            )
        self._validate_hard_pact_observation_dimensions()
        return result

    def _validate_hard_pact_observation_dimensions(self):
        """Validate the shared HardPACT/HardPACTPos observation contract once.

        Both tasks intentionally differ in the meaning of the final 12 actor
        inputs (feed-forward action versus executed PD torque), but their
        actor, history, explicit-label, and stacked-critic widths must match.
        Keeping this check in the shared environment path catches backend or
        observation-builder drift at the first reset without adding per-step
        training overhead.
        """
        if getattr(self, "_hard_pact_observation_schema_validated", False):
            return
        expected = {
            "actor observation": int(self.cfg.env.num_observations),
            "observation history": int(
                self.cfg.env.num_observations * self.cfg.env.num_obs_hist
            ),
            "explicit labels": int(self.cfg.env.num_explicit_recon_obs),
            "stacked critic observation": int(
                self.cfg.env.num_privileged_obs * self.cfg.env.num_priv_stack
            ),
        }
        actual = {
            "actor observation": int(self.obs_buf.shape[-1]),
            "observation history": int(self.obs_history.shape[-1]),
            "explicit labels": int(self.explicit_labels_buf.shape[-1]),
            "stacked critic observation": int(
                self.privileged_obs_buf.shape[-1]
            ),
        }
        mismatches = {
            name: (actual[name], width)
            for name, width in expected.items()
            if actual[name] != width
        }
        if mismatches:
            details = ", ".join(
                f"{name}: got {got}, expected {wanted}"
                for name, (got, wanted) in mismatches.items()
            )
            raise RuntimeError(f"HardPACT observation schema mismatch: {details}")
        self._hard_pact_observation_schema_validated = True

    def step(self, actions):
        """Run the legacy lifecycle with a control-interval GRF target."""
        actions = self._pre_sim_step(actions)
        pre_step_base_quat = self.simulator.base_quat.clone()
        disturbances_initialized = hasattr(self, "_persistent_component_active")
        if disturbances_initialized:
            self._hard_pact_push_event_mask.zero_()
            self._capture_bard_pre_state()
            self._update_persistent_wrench(self.common_step_counter)
            self._capture_realized_inertial_randomization()
            self._begin_disturbance_interval()
        self.grf_processor.begin_interval()
        self.simulator.step(actions)
        if disturbances_initialized:
            self._capture_bard_post_state()
            self._pending_disturbance_transition = self._end_disturbance_interval()
        interval_grf = world_to_yaw_local(
            self.grf_processor.end_interval(), pre_step_base_quat
        ).clone().flatten(1)
        self.post_physics_step()

        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(
                self.privileged_obs_buf, -clip_obs, clip_obs
            )
        return (
            self.obs_buf,
            self.privileged_obs_buf,
            self.obs_history,
            self.explicit_labels_buf,
            self.rew_buf,
            self.reset_buf,
            self.extras,
            interval_grf * self.obs_scales.grf,
        )

    def _pre_sim_step(self, actions):
        """Capture the exact legacy clip/delay/nominal-torque action path."""
        # The QP rate box is relative to the torque actually present at the
        # end of the preceding control interval.  Capture it before the legacy
        # action path updates simulator control buffers.  This is one compact
        # 12-float transition field and avoids storing any QP matrices.
        previous_torque_buffer = getattr(self.simulator, "_torques", None)
        previous_executed_torque = (
            previous_torque_buffer.detach().clone()
            if previous_torque_buffer is not None else None
        )
        if previous_executed_torque is not None:
            # A reset starts a new actuator-rate trajectory. Simulator torque
            # buffers can still contain the terminal command until the first
            # post-reset physics callback, so initialize those rows at zero.
            episode_length = getattr(self, "episode_length_buf", None)
            if episode_length is not None:
                previous_executed_torque[episode_length == 0] = 0.0
        delayed_action = self._legacy_task_class._pre_sim_step(self, actions)
        # Lightweight parity-test fixtures intentionally omit HardPACT-only
        # buffers; in that case the inherited action behavior is still exact.
        if not hasattr(self, "_action_replay_valid_queue"):
            return delayed_action
        if previous_executed_torque is None:
            previous_executed_torque = torch.zeros(
                self.num_envs, 12, device=self.device, dtype=actions.dtype
            )
        if getattr(self, "_hard_pact_rollout_qp_enabled", False):
            actor = self._hard_pact_actor_critic
            # Runner construction performs one legacy zero-action reset before
            # the policy has produced a history latent.  That boundary step is
            # deliberately unprojected; it creates the initial observation and
            # is never stored as a PPO transition.  Every subsequent training
            # step has a policy context and therefore executes the substep QP.
            if actor.cenet_z is not None and actor.cenet_torso_velo is not None:
                self.set_hard_pact_policy_context(
                    actor.cenet_z, actor.cenet_torso_velo
                )

        # Shift in lockstep with the legacy action queue. A false entry means
        # the selected action is a reset/boundary zero rather than a policy
        # sample, so PPO must replay it as an exact constant zero.
        valid_queue = self._action_replay_valid_queue
        if valid_queue.shape[1] > 1:
            valid_queue[:, 1:] = valid_queue[:, :-1].clone()
        valid_queue[:, 0] = True
        if bool(self.cfg.domain_rand.randomize_ctrl_delay):
            delay = self.action_delay.clone()
        else:
            delay = torch.zeros(
                self.num_envs, device=self.device, dtype=torch.long
            )
        selected_valid = valid_queue[
            torch.arange(self.num_envs, device=self.device), delay
        ].unsqueeze(-1)

        # Match the legacy PINN torque convention exactly, evaluated at the
        # pre-step joint state on the action actually selected by the queue.
        desired_position, feedforward_torque = self._get_pinn_actions(
            delayed_action
        )
        if feedforward_torque.shape[-1] == 0:
            # The PACTPos alias has no feedforward half; retain its legacy
            # feedback-only torque convention without widening its actions.
            feedforward_torque = torch.zeros_like(desired_position)
        joint_position, joint_velocity = self._canonical_joint_state()
        nominal_torque = feedforward_torque + self._get_pinn_feedback(
            desired_position,
            joint_position,
            joint_velocity,
        )
        self._pending_action_replay_transition = {
            "sampled_action_delay": delay.to(torch.int16).unsqueeze(-1),
            "delayed_action": delayed_action.detach().clone(),
            "delayed_action_source_valid": selected_valid.clone(),
            "nominal_torque": nominal_torque.detach().clone(),
            "previous_executed_torque": previous_executed_torque,
        }
        # Hold the action-space command across decimation.  PD feedback is
        # intentionally *not* held: it is reevaluated from q_k,qdot_k in the
        # callback immediately before every physics actuation.
        self._hard_pact_q_d = desired_position.detach().clone()
        self._hard_pact_tau_ff = feedforward_torque.detach().clone()
        self._hard_pact_previous_substep_torque = previous_executed_torque.clone()
        if not hasattr(self, "_hard_pact_previous_certified_qdd"):
            self._hard_pact_previous_certified_qdd = torch.zeros(
                self.num_envs, 18, device=self.device
            )
        return delayed_action

    def reset_idx(self, env_ids):
        self._legacy_task_class.reset_idx(self, env_ids)
        rollout_qp = getattr(self, "_hard_pact_rollout_qp", None)
        if rollout_qp is not None and hasattr(rollout_qp, "clear_warm_start"):
            rollout_qp.clear_warm_start(env_ids)
        if hasattr(self, "grf_processor") and env_ids.numel() > 0:
            self.grf_processor.reset(env_ids)
        if hasattr(self, "_persistent_component_active") and env_ids.numel() > 0:
            self._reset_persistent_wrench_state(env_ids)
            for value in (
                self._persistent_wrench_target_world,
                self._current_sustained_wrench_world,
                self._persistent_component_active,
                self._current_sustained_active_mask,
                self._persistent_start_step,
                self._persistent_end_step,
                self._persistent_duration_steps,
                self._disturbance_interval_sum_sustained,
                self._disturbance_interval_sum_mass_com,
                self._disturbance_interval_sum_total,
                self._disturbance_interval_sum_sustained_yaw_scaled,
                self._disturbance_interval_sum_mass_com_yaw_scaled,
                self._disturbance_interval_sum_yaw_scaled,
                self._disturbance_interval_count,
                self._interval_executed_torque_sum,
                self._interval_executed_torque_count,
                self._hard_pact_push_event_mask,
                self._realized_added_mass,
                self._realized_com_shift_body,
            ):
                value[env_ids] = 0
            self._action_replay_valid_queue[env_ids] = False
            # These are current-interval diagnostics, not PPO targets. Clear
            # reset rows so no stale pre-reset peak/status leaks into logging.
            for name in (
                "_qp_interval_safe_sum", "_qp_interval_safe_peak",
                "_qp_interval_correction_sum", "_qp_interval_correction_peak",
                "_qp_interval_grf_sum", "_qp_interval_slack_sum",
                "_qp_interval_slack_peak", "_qp_interval_residual_sum",
                "_qp_interval_residual_peak", "_qp_interval_stage_counts",
                "_qp_interval_timing_ms",
            ):
                value = getattr(self, name, None)
                if value is not None:
                    value[env_ids] = 0
            for frame in self.disturbance_critic_deque:
                frame[env_ids] = 0
        # QP rate/proximal and held-anchor state belongs to the actuator path,
        # not the optional persistent-disturbance feature.  Always clear it at
        # an episode boundary so the first post-reset rate box is centred at
        # zero and no correction from the previous episode can be replayed.
        for name in (
            "_hard_pact_previous_substep_torque",
            "_hard_pact_previous_certified_qdd",
            "_hard_pact_q_d", "_hard_pact_tau_ff",
            "_hard_pact_held_correction",
            "_hard_pact_held_grf_yaw_scaled",
            "_hard_pact_held_force_world",
            "_hard_pact_held_qdd", "_hard_pact_held_slack",
            "_hard_pact_held_residual", "_hard_pact_held_stage",
            "_hard_pact_held_differentiated",
        ):
            value = getattr(self, name, None)
            if value is not None:
                value[env_ids] = 0


def install_hard_pact_environment_methods(task_class):
    """Install the concrete HardPACT methods on the direct PACTPos subclass."""
    for name, value in Go2HardPACT.__dict__.items():
        if name.startswith("__") or name == "_legacy_task_class":
            continue
        if callable(value):
            setattr(task_class, name, value)
    return task_class
