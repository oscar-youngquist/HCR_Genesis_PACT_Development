"""Go2 HardPACT environment with GRF and persistent-wrench extensions."""

from __future__ import annotations

from collections import deque

import torch

from rsl_rl.modules.hard_pact_physics import compose_explicit_estimator_target
from legged_gym.envs.go2.go2_pact.go2_pact import Go2PACT

from .grf import GRFProcessingConfig, IntervalGRFProcessor, world_to_yaw_local

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

    def _init_buffers(self):
        self._legacy_task_class._init_buffers(self)
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
        self._hard_pact_push_event_mask = torch.zeros(
            self.num_envs, 1, device=self.device, dtype=torch.bool
        )
        self.simulator._hard_pact_pre_physics_substep = (
            self._hard_pact_pre_physics_substep
        )
        self._reset_persistent_wrench_state(
            torch.arange(self.num_envs, device=self.device)
        )

    def _hard_pact_grf_post_physics_substep(self):
        raw = self.simulator._robot.get_links_net_contact_force()[
            :, self.simulator.feet_indices, :
        ]
        self.grf_processor.update_substep(raw)

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
        robot = self.simulator._robot
        if hasattr(robot, "get_quat"):
            quat_wxyz = robot.get_quat()
            return quat_wxyz[:, (1, 2, 3, 0)]
        return self.simulator.base_quat

    def _apply_sustained_world_wrench(self, wrench_world):
        """The only simulator external-wrench call made by HardPACT."""
        simulator = self.simulator
        base_link = torch.as_tensor(
            [simulator._base_link_index], device=self.device, dtype=torch.long
        )
        solver = simulator._robot._solver
        solver.apply_links_external_force(
            force=wrench_world[:, :3].unsqueeze(1), links_idx=base_link,
            envs_idx=None, ref="link_com", local=False,
        )
        solver.apply_links_external_torque(
            torque=wrench_world[:, 3:].unsqueeze(1), links_idx=base_link,
            envs_idx=None, ref="link_com", local=False,
        )

    def _hard_pact_pre_physics_substep(self):
        quat = self._current_base_quat_xyzw()
        mass_com_wrench = added_mass_gravity_wrench_world(
            self._realized_added_mass,
            self.cfg.sim.gravity,
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
        # _torques is the exact command sent to Genesis in this decimation
        # substep. Accumulating here produces the requested control-interval
        # executed torque without altering the legacy torque computation.
        if hasattr(self, "_interval_executed_torque_sum"):
            self._interval_executed_torque_sum.add_(self.simulator._torques)
            self._interval_executed_torque_count.add_(1.0)
        # The equivalent inertial wrench is deliberately label-only: Genesis
        # already realizes the randomized mass and CoM in its dynamics.
        self._apply_sustained_world_wrench(self._current_sustained_wrench_world)

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

    def _capture_bard_pre_state(self):
        simulator = self.simulator
        self._bard_pre_q_simulator = torch.cat((
            simulator._base_pos,
            simulator._base_quat,
            simulator._dof_pos,
        ), dim=-1).clone()
        self._bard_pre_v_world_simulator = torch.cat((
            simulator._robot.get_vel(),
            simulator._robot.get_ang(),
            simulator._dof_vel,
        ), dim=-1).clone()

    def _capture_bard_post_state(self):
        simulator = self.simulator
        self._bard_post_v_world_simulator = torch.cat((
            simulator._robot.get_vel(),
            simulator._robot.get_ang(),
            simulator._robot.get_dofs_velocity(simulator._dof_indices),
        ), dim=-1).clone()

    def _post_physics_step_callback(self):
        """Observe legacy push events without changing legacy push logic."""
        velocity_before = self.simulator._robot.get_dofs_velocity()[:, :6].clone()
        self._legacy_task_class._post_physics_step_callback(self)
        velocity_after = self.simulator._robot.get_dofs_velocity()[:, :6]
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
        if bool(self.cfg.domain_rand.randomize_base_mass):
            self._realized_added_mass.copy_(self.simulator._added_base_mass)
        else:
            self._realized_added_mass.zero_()
        if bool(self.cfg.domain_rand.randomize_com_displacement):
            self._realized_com_shift_body.copy_(self.simulator._base_com_bias)
        else:
            self._realized_com_shift_body.zero_()

    def _build_disturbance_transition(self):
        pending = self._pending_disturbance_transition
        reset = self.reset_buf.bool().reshape(-1, 1)
        timeout = self.time_out_buf.bool().reshape(-1, 1)
        teleport = self.non_failure_reset_buf.bool().reshape(-1, 1)
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
            "interval_executed_torque": (
                self._interval_executed_torque_sum / torque_divisor
            ).clone(),
            "joint_armature": self.simulator._joint_armature.clone(),
            "joint_friction": self.simulator._joint_friction.clone(),
            "joint_stiffness": self.simulator._joint_stiffness.clone(),
            "joint_damping": self.simulator._joint_damping.clone(),
            "push_event_mask": push_event,
            "reset_mask": reset,
            "timeout_mask": timeout,
            "teleport_mask": teleport,
            "physics_valid_mask": physics_transition_mask(
                reset, timeout, teleport, push_event
            ),
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
        return result

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

    def reset_idx(self, env_ids):
        self._legacy_task_class.reset_idx(self, env_ids)
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
            for frame in self.disturbance_critic_deque:
                frame[env_ids] = 0


def install_hard_pact_environment_methods(task_class):
    """Install the concrete HardPACT methods on the direct PACTPos subclass."""
    for name, value in Go2HardPACT.__dict__.items():
        if name.startswith("__") or name == "_legacy_task_class":
            continue
        if callable(value):
            setattr(task_class, name, value)
    return task_class
