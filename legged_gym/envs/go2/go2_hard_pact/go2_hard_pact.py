"""Simulator-neutral environment core for Go2 HardPACT and HardPACTPos."""

from __future__ import annotations

import os
from typing import Dict

import torch

from legged_gym import LEGGED_GYM_ROOT_DIR, SIMULATOR
from legged_gym.envs.go2.go2_pact.go2_pact import Go2PACT
from legged_gym.utils.math_utils import quat_apply, wrap_to_pi

from .backend import ADAPTERS, disable_unsupported_randomizations
from .disturbances import (
    InstantaneousPushConfig,
    InstantaneousPushes,
    SustainedBaseWrench,
    SustainedWrenchConfig,
    physics_transition_mask,
)
from .grf import GRFProcessingConfig, IntervalGRFProcessor
from .qp import (
    Go2HardPACTQP,
    HardPACTQPConfig,
    HardPACTQPInputs,
    HardPACTQPResult,
)
from .schema import (
    CANONICAL,
    RECONSTRUCTION_SCHEMA,
    fixed_gravity_normal,
    reorder_named,
    validate_transition,
    world_to_body,
    world_to_yaw_local,
    yaw_quaternion_xyzw,
)


def _backend_name(value: str) -> str:
    return "genesis" if value.startswith("genesis") else value


def _class_dict(instance) -> dict:
    return {
        key: value
        for key, value in vars(instance.__class__).items()
        if not key.startswith("_") and not callable(value)
    }


class Go2HardPACTCore(Go2PACT):
    """One task implementation shared verbatim by all simulator adapters."""

    position_pretraining = False
    expected_backend = None

    def __init__(self, cfg, sim_params, sim_device, headless):
        backend_name = _backend_name(SIMULATOR)
        if backend_name not in ADAPTERS:
            raise ValueError(f"Go2 HardPACT does not support simulator {SIMULATOR!r}")
        if self.expected_backend is not None and backend_name != self.expected_backend:
            raise ValueError(
                f"task registration requires {self.expected_backend}, but SIMULATOR={SIMULATOR}"
            )
        self.backend_name = backend_name
        self.backend_capabilities = ADAPTERS[backend_name].capabilities
        self.domain_randomization_report = disable_unsupported_randomizations(
            cfg.domain_rand, self.backend_capabilities
        )
        self._bard = None
        self._physics_reference_serial = 0
        super().__init__(cfg, sim_params, sim_device, headless)
        self.extras["backend_contract"] = self.backend.metadata()
        self.extras["domain_randomization"] = self.domain_randomization_report
        self._initialize_bard()

    @property
    def policy_action_dim(self):
        return 12 if self.position_pretraining else 24

    def _init_buffers(self):
        self.backend = ADAPTERS[self.backend_name](self.simulator, self.cfg)
        self.backend.augment_simulator_buffers()
        super()._init_buffers()
        # The legacy initializer allocates coupled actions. PACTPos still keeps
        # a 24-D observation history, but only queues a 12-D sampled action.
        max_delay = int(self.cfg.domain_rand.ctrl_delay_step_range[1])
        self.action_queue = torch.zeros(
            self.num_envs, max_delay + 1, self.policy_action_dim,
            device=self.device, dtype=torch.float,
        )
        self.action_delay = torch.randint(
            int(self.cfg.domain_rand.ctrl_delay_step_range[0]), max_delay + 1,
            (self.num_envs,), device=self.device,
        )
        self.raw_sampled_action = torch.zeros(
            self.num_envs, self.policy_action_dim, device=self.device
        )
        self.delayed_nominal_action = torch.zeros(
            self.num_envs, 24, device=self.device
        )
        self.nominal_torque = torch.zeros(self.num_envs, 12, device=self.device)
        self.safe_torque = torch.zeros_like(self.nominal_torque)
        self.executed_torque = torch.zeros_like(self.nominal_torque)
        self.previous_safe_torque = torch.zeros_like(self.nominal_torque)
        self.torque_average = torch.zeros_like(self.nominal_torque)
        self.torque_peak = torch.zeros_like(self.nominal_torque)
        self.pending_exploration_noise = torch.zeros_like(self.raw_sampled_action)
        self.teleport_mask = torch.zeros(self.num_envs, 1, device=self.device, dtype=torch.bool)
        self.reconstruction_target = torch.zeros(self.num_envs, 79, device=self.device)
        self.explicit_labels_buf = torch.zeros(self.num_envs, 11, device=self.device)

        grf_cfg = self.cfg.sim.grf
        self.grf_processor = IntervalGRFProcessor(
            self.num_envs, 4, self.device, torch.float32,
            GRFProcessingConfig(
                vertical_deadband_n=float(grf_cfg.vertical_deadband_n),
                clip_min_n=float(grf_cfg.clip_min_n),
                clip_max_n=float(grf_cfg.clip_max_n),
                ema_alpha=float(grf_cfg.ema_alpha),
                contact_threshold_n=float(grf_cfg.contact_threshold_n),
            ),
        )
        push_cfg = self.cfg.disturbances.instantaneous
        self.instantaneous_pushes = InstantaneousPushes(
            self.num_envs, self.device, torch.float32,
            InstantaneousPushConfig(
                enabled=bool(push_cfg.enabled),
                probability=float(push_cfg.probability),
                interval_steps_min=int(push_cfg.interval_steps_min),
                interval_steps_max=int(push_cfg.interval_steps_max),
                planar_delta_v=tuple(push_cfg.planar_delta_v),
                downward_delta_vz=tuple(push_cfg.downward_delta_vz),
                angular_delta_v=tuple(push_cfg.angular_delta_v),
            ),
        )
        wrench_cfg = self.cfg.disturbances.sustained_wrench
        self.sustained_wrench = SustainedBaseWrench(
            self.num_envs, self.device, torch.float32,
            SustainedWrenchConfig(
                enabled=bool(wrench_cfg.enabled),
                force_probability=float(wrench_cfg.force_probability),
                torque_probability=float(wrench_cfg.torque_probability),
                interval_steps=tuple(wrench_cfg.interval_steps),
                duration_steps=tuple(wrench_cfg.duration_steps),
                force_interval_steps=tuple(wrench_cfg.force_interval_steps),
                torque_interval_steps=tuple(wrench_cfg.torque_interval_steps),
                force_duration_steps=tuple(wrench_cfg.force_duration_steps),
                torque_duration_steps=tuple(wrench_cfg.torque_duration_steps),
                ramp_fraction=float(wrench_cfg.ramp_fraction),
                force_bounds_n=tuple(wrench_cfg.force_bounds_n),
                torque_bounds_nm=tuple(wrench_cfg.torque_bounds_nm),
                force_normalizer_n=float(wrench_cfg.force_normalizer_n),
                torque_normalizer_nm=float(wrench_cfg.torque_normalizer_nm),
            ),
        )
        qp_cfg = self.cfg.qp
        dtype_name = qp_cfg.reference_dtype
        if (
            torch.device(self.device).type == "cuda"
            and bool(qp_cfg.use_float32_gpu)
        ):
            dtype_name = qp_cfg.gpu_dtype
        dtype = torch.float64 if dtype_name == "float64" else torch.float32
        self.qp = Go2HardPACTQP(HardPACTQPConfig(
            enabled=bool(qp_cfg.enabled),
            friction=float(qp_cfg.friction),
            acceleration_weight=float(qp_cfg.acceleration_weight),
            grf_tracking_weight=float(qp_cfg.grf_tracking_weight),
            torque_tracking_weight=float(qp_cfg.torque_tracking_weight),
            contact_slack_weight=float(qp_cfg.contact_slack_weight),
            hessian_regularization=float(qp_cfg.hessian_regularization),
            solver=str(qp_cfg.solver),
            solver_dtype=dtype,
            gpu_chunk_size=int(qp_cfg.gpu_chunk_size),
        ))
        self.last_qp_result = self._projection_only_result(
            self.nominal_torque,
            fallback_code=2 if bool(self.cfg.features.use_qp) else 0,
        )
        self.last_transition: Dict[str, torch.Tensor] = {}

    def _initialize_bard(self):
        if not bool(self.cfg.bard.enabled):
            return
        try:
            from legged_gym.dynamics import BardGo2Dynamics

            urdf = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
            default = self.simulator.default_dof_pos[0] if self.simulator.default_dof_pos.ndim == 2 else self.simulator.default_dof_pos
            self._bard = BardGo2Dynamics(
                urdf,
                self.cfg.asset.dof_names,
                CANONICAL.foot_names,
                "base",
                device=self.device,
                batch_capacity=int(self.cfg.bard.batch_capacity),
                default_joint_position=default,
            )
        except (ImportError, FileNotFoundError, RuntimeError) as exc:
            if bool(self.cfg.bard.required):
                raise RuntimeError("required Go2 HardPACT BARD initialization failed") from exc
            self.extras["bard_unavailable"] = str(exc)

    def reset(self):
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        zeros = torch.zeros(self.num_envs, self.policy_action_dim, device=self.device)
        result = self.step(zeros, allow_missing_physics_references=True)
        return result[0], result[1]

    def set_exploration_noise(self, noise: torch.Tensor):
        if noise.shape != self.pending_exploration_noise.shape:
            raise ValueError("exploration-noise shape does not match sampled action")
        self.pending_exploration_noise.copy_(noise)

    def _delay_action(self, raw_action):
        clip = float(self.cfg.normalization.clip_actions)
        raw_action = raw_action.to(self.device).clamp(-clip, clip)
        if raw_action.shape != self.raw_sampled_action.shape:
            raise ValueError(
                f"expected sampled action {tuple(self.raw_sampled_action.shape)}, got {tuple(raw_action.shape)}"
            )
        self.raw_sampled_action.copy_(raw_action)
        self.action_queue[:, 1:] = self.action_queue[:, :-1].clone()
        self.action_queue[:, 0] = raw_action
        if self.cfg.domain_rand.randomize_ctrl_delay:
            delayed = self.action_queue[
                torch.arange(self.num_envs, device=self.device), self.action_delay
            ]
        else:
            delayed = raw_action
        if self.position_pretraining:
            coupled = torch.cat((delayed, torch.zeros_like(delayed)), dim=-1)
        else:
            coupled = delayed
        self.delayed_nominal_action.copy_(coupled)
        return coupled

    def _gains(self):
        p = self.backend.ordered_backend_joint_values(self.simulator._p_gains)
        d = self.backend.ordered_backend_joint_values(self.simulator._d_gains)
        if p.ndim == 1:
            p = p.unsqueeze(0).expand(self.num_envs, -1)
            d = d.unsqueeze(0).expand(self.num_envs, -1)
        p = p[:, :12]
        d = d[:, :12]
        kp_scale = self.backend.ordered_backend_joint_values(
            self.simulator._kp_scale
        )[:, :12]
        kd_scale = self.backend.ordered_backend_joint_values(
            self.simulator._kd_scale
        )[:, :12]
        return p * kp_scale, d * kd_scale

    def _default_joint_position(self):
        default = self.backend.ordered_backend_joint_values(
            self.simulator._default_dof_pos
        )
        if default.ndim == 1:
            default = default.unsqueeze(0)
        return default[:, :12]

    def _nominal_torque(self, coupled_action):
        q = self.simulator.dof_pos[:, :12]
        qd = self.simulator.dof_vel[:, :12]
        default = self._default_joint_position()
        kp, kd = self._gains()
        desired = default + float(self.cfg.control.action_scale) * coupled_action[:, :12]
        feedback = kp * (desired - q) - kd * qd
        motor = self.backend.ordered_backend_joint_values(
            self.simulator._motor_strength
        )[:, :12]
        feedforward = float(self.cfg.control.torque_scale) * coupled_action[:, 12:]
        if self.position_pretraining:
            feedforward.zero_()
        feedback_weight = self.simulator.feedback_tau_weight
        feedforward_weight = self.simulator.feedforward_tau_weight
        weighted_feedback = motor * feedback_weight * feedback
        weighted_feedforward = motor * feedforward_weight * feedforward
        nominal = weighted_feedback + weighted_feedforward
        sim = self.simulator
        sim.feedback_torques = feedback
        sim.feedforward_torques = feedforward
        sim.combined_feedback_torques = weighted_feedback
        sim.combined_feedforward_torques = weighted_feedforward
        sim.first_loop_feedback = feedback
        sim._unweighted_torques = nominal
        self.nominal_torque.copy_(nominal)
        return nominal

    def _projection_only_result(self, nominal, previous=None, fallback_code=2):
        limit = self.simulator.torque_limits[:12]
        rate = float(self.cfg.control.torque_rate_limit)
        if previous is None:
            previous = (
                self.previous_safe_torque
                if self.previous_safe_torque.shape[0] == nominal.shape[0]
                else torch.zeros_like(nominal)
            )
        safe = self.qp.actuator_rate_projection(
            nominal, previous, limit, rate, self.dt
        ) if hasattr(self, "qp") else nominal.clamp(-limit, limit)
        batch = nominal.shape[0]
        zeros_1 = nominal.new_zeros(batch, 1)
        return HardPACTQPResult(
            safe_torque=safe,
            acceleration=nominal.new_zeros(batch, 18),
            grf=nominal.new_zeros(batch, 12),
            contact_slack=nominal.new_zeros(batch, 12),
            correction=safe - nominal,
            equality_residual=zeros_1.clone(),
            inequality_violation=zeros_1.clone(),
            minimum_margin=zeros_1.clone(),
            active_constraints=zeros_1.clone(),
            fallback=torch.full(
                (batch, 1), fallback_code,
                device=nominal.device, dtype=torch.long,
            ),
            status=torch.full(
                (batch, 1), fallback_code,
                device=nominal.device, dtype=torch.long,
            ),
            forward_time_ms=zeros_1.clone(),
        )

    @staticmethod
    def _yaw_rotation(base_quat_xyzw):
        yaw_quat = yaw_quaternion_xyzw(base_quat_xyzw)
        z = yaw_quat[:, 2]
        w = yaw_quat[:, 3]
        return torch.stack((
            w.square() - z.square(), -2.0 * w * z, torch.zeros_like(w),
            2.0 * w * z, w.square() - z.square(), torch.zeros_like(w),
            torch.zeros_like(w), torch.zeros_like(w), torch.ones_like(w),
        ), dim=-1).reshape(-1, 3, 3)

    def _solve_qp(self, state, predicted_grf, predicted_wrench, contact_probability):
        if not bool(self.cfg.features.use_qp):
            return self._projection_only_result(
                self.nominal_torque, fallback_code=0
            )
        if self._bard is None:
            return self._projection_only_result(self.nominal_torque)
        q = torch.cat((
            state.base_position_world, state.base_quaternion_xyzw, state.joint_position
        ), dim=-1)
        terms = self._bard.terms(q.detach(), state.velocity_world.detach(), parameters=None)
        rotation = self._yaw_rotation(state.base_quaternion_xyzw)
        rotation_t = rotation.transpose(1, 2)
        foot_j = torch.einsum("bij,bfjn->bfin", rotation_t, terms.foot_jacobians)
        base_rotation = torch.zeros(
            self.num_envs, 6, 6, device=self.device, dtype=rotation.dtype
        )
        base_rotation[:, :3, :3] = rotation_t
        base_rotation[:, 3:, 3:] = rotation_t
        base_j = base_rotation @ terms.base_jacobian
        limits = self.simulator.dof_pos_limits[:12]
        velocity_limit = self.simulator.dof_vel_limits[:12]
        gravity_world = self.nominal_torque.new_tensor([0.0, 0.0, -9.81]).expand(self.num_envs, -1)
        normal = fixed_gravity_normal(
            gravity_world,
            yaw_quaternion_xyzw(state.base_quaternion_xyzw),
        )
        qp_inputs = HardPACTQPInputs(
            mass=terms.mass.detach(),
            bias=terms.bias.detach(),
            foot_jacobian=foot_j.detach(),
            foot_jdot_v=world_to_yaw_local(
                terms.foot_jdot_v.detach(), state.base_quaternion_xyzw
            ),
            base_jacobian=base_j.detach(),
            predicted_grf=predicted_grf,
            predicted_base_wrench=predicted_wrench,
            contact_probability=contact_probability.clamp(0.0, 1.0),
            nominal_torque=self.nominal_torque,
            previous_torque=self.previous_safe_torque,
            torque_limit=self.simulator.torque_limits[:12],
            torque_rate_limit=self.nominal_torque.new_full((12,), float(self.cfg.control.torque_rate_limit)),
            joint_position=state.joint_position,
            joint_velocity=state.velocity_world[:, 6:],
            joint_position_lower=limits[:, 0],
            joint_position_upper=limits[:, 1],
            joint_velocity_limit=velocity_limit,
            gravity_normal_force_frame=normal,
            dt=float(self.dt),
        )
        return self.qp.solve(qp_inputs)

    def recompute_training_objectives(self, batch, raw_sampled_action, actor):
        """Rebuild estimator and differentiable QP outputs from sampled actions."""
        from rsl_rl.algorithms.ppo_go2_hard_pact import (
            combine_physics_losses,
            delayed_action_for_update,
            inverse_dynamics_loss,
            rollout_loss,
            supervised_physics_head_losses,
        )
        physical = RECONSTRUCTION_SCHEMA.unpack(
            batch["reconstruction_target"], normalized=True
        )
        delay_steps = torch.round(
            physical["control_delay"] / float(self.dt)
        ).long()
        stored_delayed = batch["delayed_nominal_action"]
        if self.position_pretraining:
            replay_position = delayed_action_for_update(
                raw_sampled_action, stored_delayed[:, :12], delay_steps
            )
            coupled = torch.cat((replay_position, torch.zeros_like(replay_position)), dim=-1)
        else:
            coupled = delayed_action_for_update(
                raw_sampled_action, stored_delayed, delay_steps
            )
        q_joint = batch["pre_q"][:, 7:]
        qd_joint = batch["pre_v"][:, 6:]
        base_p = self.backend.ordered_backend_joint_values(
            self.simulator._p_gains
        )
        base_d = self.backend.ordered_backend_joint_values(
            self.simulator._d_gains
        )
        if base_p.ndim == 2:
            base_p = base_p[0]
            base_d = base_d[0]
        base_p = base_p[:12].to(q_joint)
        base_d = base_d[:12].to(q_joint)
        desired = self._default_joint_position()[0].to(q_joint) + float(
            self.cfg.control.action_scale
        ) * coupled[:, :12]
        feedback = (
            physical["kp_scale"] * base_p * (desired - q_joint)
            - physical["kd_scale"] * base_d * qd_joint
        )
        feedforward = (
            float(self.cfg.control.torque_scale) * coupled[:, 12:]
        )
        nominal = physical["motor_strength_scale"] * (
            batch["feedback_branch_weight"] * feedback
            + batch["feedforward_branch_weight"] * feedforward
        )
        encoded = actor.encode_policy_history(batch["history"])
        references = actor.physics_references(
            batch["history"], nominal, encoded=encoded
        )

        if bool(self.cfg.features.use_qp) and self._bard is not None:
            q = batch["pre_q"]
            v = batch["pre_v"]
            terms = self._bard.terms(q.detach(), v.detach(), parameters=None)
            rotation = self._yaw_rotation(q[:, 3:7])
            rotation_t = rotation.transpose(1, 2)
            foot_j = torch.einsum("bij,bfjn->bfin", rotation_t, terms.foot_jacobians)
            base_rotation = torch.zeros(
                q.shape[0], 6, 6, device=q.device, dtype=q.dtype
            )
            base_rotation[:, :3, :3] = rotation_t
            base_rotation[:, 3:, 3:] = rotation_t
            base_j = base_rotation @ terms.base_jacobian
            limits = self.simulator.dof_pos_limits[:12].to(q)
            velocity_limit = self.simulator.dof_vel_limits[:12].to(q)
            qp_inputs = HardPACTQPInputs(
                mass=terms.mass.detach(),
                bias=terms.bias.detach(),
                foot_jacobian=foot_j.detach(),
                foot_jdot_v=world_to_yaw_local(
                    terms.foot_jdot_v.detach(), q[:, 3:7]
                ),
                base_jacobian=base_j.detach(),
                predicted_grf=references.grf_yaw_n,
                predicted_base_wrench=references.base_wrench_yaw,
                contact_probability=encoded.explicit[:, 3:7],
                nominal_torque=nominal,
                previous_torque=batch["previous_safe_torque"].detach(),
                torque_limit=self.simulator.torque_limits[:12].to(q),
                torque_rate_limit=q.new_full((12,), float(self.cfg.control.torque_rate_limit)),
                joint_position=q_joint,
                joint_velocity=qd_joint,
                joint_position_lower=limits[:, 0],
                joint_position_upper=limits[:, 1],
                joint_velocity_limit=velocity_limit,
                gravity_normal_force_frame=fixed_gravity_normal(
                    q.new_tensor([0.0, 0.0, -9.81]).expand(q.shape[0], -1),
                    yaw_quaternion_xyzw(q[:, 3:7]),
                ),
                dt=float(self.dt),
            )
            qp_result = self.qp.solve(qp_inputs)
            safe = qp_result.safe_torque
            if (
                not bool(self.cfg.features.differentiate_qp)
                or bool(self.cfg.features.stop_gradient_qp)
            ):
                safe = safe.detach()
        else:
            qp_result = self._projection_only_result(
                nominal,
                batch["previous_safe_torque"].detach(),
                fallback_code=2 if bool(self.cfg.features.use_qp) else 0,
            )
            safe = qp_result.safe_torque

        supervised = supervised_physics_head_losses(
            references.grf_yaw_n,
            batch["interval_grf_yaw"],
            references.base_wrench_yaw,
            batch["interval_wrench_yaw"],
            batch["sustained_wrench_active_mask"],
        )
        reconstruction, auxiliary_encoded = actor.reconstruct_privileged(
            batch["history"], sample_for_auxiliary=True
        )
        reconstruction_loss = torch.nn.functional.smooth_l1_loss(
            reconstruction, batch["reconstruction_target"]
        )
        reconstructed_physical = RECONSTRUCTION_SCHEMA.unpack(
            reconstruction, normalized=True
        )
        reconstruction_metrics = {}
        for field in RECONSTRUCTION_SCHEMA.fields:
            field_slice = RECONSTRUCTION_SCHEMA.slices[field.name]
            reconstruction_metrics[
                f"reconstruction/{field.name}_huber_normalized"
            ] = torch.nn.functional.smooth_l1_loss(
                reconstruction[:, field_slice],
                batch["reconstruction_target"][:, field_slice],
            )
            reconstruction_metrics[
                f"reconstruction/{field.name}_mae_physical"
            ] = (
                reconstructed_physical[field.name] - physical[field.name]
            ).abs().mean()
        explicit_loss = torch.nn.functional.smooth_l1_loss(
            auxiliary_encoded.explicit, batch["explicit_estimator_target"]
        )
        kl = -0.5 * torch.mean(
            1.0 + auxiliary_encoded.latent_log_variance
            - auxiliary_encoded.latent_mean.square()
            - auxiliary_encoded.latent_log_variance.exp()
        )
        auxiliary = reconstruction_loss + explicit_loss + 1.0e-3 * kl
        if bool(self.cfg.features.supervised_physics_head_pretraining):
            auxiliary = (
                auxiliary
                + float(self.cfg.features.grf_supervision_weight) * supervised["grf"]
                + float(self.cfg.features.active_wrench_supervision_weight)
                * supervised["wrench_active"]
                + float(self.cfg.features.neutral_wrench_supervision_weight)
                * supervised["wrench_neutral"]
            )
        clone_loss = nominal.new_zeros(())
        if self.position_pretraining:
            clone_target = batch["executed_torque"].detach() / float(self.cfg.control.torque_scale)
            clone_loss = actor.clone_feedforward_loss(
                batch["observation"], batch["history"], clone_target
            )
            auxiliary = auxiliary + float(
                self.cfg.features.feedforward_clone_weight
            ) * clone_loss

        # Keep disabled physics objectives connected to shared estimator
        # parameters so baseline PCGrad remains a valid zero objective.
        zero = (
            references.grf_yaw_n.sum() + references.base_wrench_yaw.sum()
        ) * 0.0
        inverse = zero
        rollout = zero
        metrics = dict(reconstruction_metrics)
        if self._bard is not None and (
            bool(self.cfg.features.use_bard_inverse_loss)
            or bool(self.cfg.features.use_bard_rollout_loss)
        ):
            actual_parameters = {
                name: physical[name]
                for name in (
                    "added_base_mass", "base_com_shift", "joint_armature",
                    "joint_friction", "joint_stiffness", "joint_damping",
                )
            }
            # Yaw local -> world uses the transpose of the QP rotation.
            rotation = self._yaw_rotation(batch["pre_q"][:, 3:7])
            grf_world = torch.einsum(
                "bij,bfj->bfi", rotation, references.grf_yaw_n.reshape(-1, 4, 3)
            )
            wrench_world = torch.cat((
                torch.einsum("bij,bj->bi", rotation, references.base_wrench_yaw[:, :3]),
                torch.einsum("bij,bj->bi", rotation, references.base_wrench_yaw[:, 3:]),
            ), dim=-1)
            if bool(self.cfg.features.use_bard_inverse_loss):
                inverse, inverse_metrics = inverse_dynamics_loss(
                    self._bard,
                    batch["pre_q"], batch["pre_v"], batch["post_v"],
                    batch["average_torque"], grf_world.flatten(1), wrench_world,
                    batch["physics_valid_mask"], self.dt,
                    parameters=actual_parameters,
                )
                metrics.update({f"inverse/{key}": value for key, value in inverse_metrics.items()})
            if bool(self.cfg.features.use_bard_rollout_loss):
                rollout, rollout_metrics = rollout_loss(
                    self._bard,
                    batch["pre_q"], batch["pre_v"], batch["post_v"],
                    safe, grf_world.flatten(1), wrench_world,
                    batch["physics_valid_mask"], self.dt,
                    parameters=actual_parameters,
                )
                metrics.update({f"rollout/{key}": value for key, value in rollout_metrics.items()})
        if (
            not bool(self.cfg.features.differentiate_qp)
            or bool(self.cfg.features.stop_gradient_qp)
        ):
            projection_correction = qp_result.safe_torque.detach() - nominal
        else:
            projection_correction = qp_result.correction
        projection = projection_correction.square().mean()
        if not bool(self.cfg.features.use_qp) and not bool(
            self.cfg.features.use_soft_projection_penalty
        ):
            projection = zero
        physics = combine_physics_losses(
            inverse, rollout, projection,
            float(self.cfg.bard.lambda_inverse),
            float(self.cfg.bard.lambda_rollout),
            float(self.cfg.bard.lambda_projection),
            metrics,
        )
        metrics.update({
            "supervised/grf": supervised["grf"],
            "supervised/wrench_active": supervised["wrench_active"],
            "supervised/wrench_neutral": supervised["wrench_neutral"],
            "reconstruction": reconstruction_loss,
            "explicit": explicit_loss,
            "kl": kl,
            "clone": clone_loss,
            "grf_mae_n": (references.grf_yaw_n - batch["interval_grf_yaw"]).abs().mean(),
            "wrench_mae": (references.base_wrench_yaw - batch["interval_wrench_yaw"]).abs().mean(),
            "qp/correction_l2_nm": qp_result.correction.norm(dim=-1).mean(),
            "qp/correction_max_nm": qp_result.correction.abs().amax(dim=-1).mean(),
            "qp/contact_slack_l2": qp_result.contact_slack.norm(dim=-1).mean(),
            "qp/contact_slack_max": qp_result.contact_slack.abs().amax(dim=-1).mean(),
            "qp/equality_residual_max": torch.nan_to_num(
                qp_result.equality_residual, nan=0.0
            ).mean(),
            "qp/inequality_violation_max": torch.nan_to_num(
                qp_result.inequality_violation, nan=0.0
            ).mean(),
            "qp/minimum_margin": torch.nan_to_num(
                qp_result.minimum_margin, nan=0.0
            ).mean(),
            "qp/active_constraints": qp_result.active_constraints.float().mean(),
            "qp/fallback": (qp_result.fallback > 0).float().mean(),
            "qp/actuator_projection_fallback": (qp_result.fallback == 2).float().mean(),
            "qp/infeasible": (qp_result.status > 0).float().mean(),
            "qp/forward_time_ms": qp_result.forward_time_ms.mean(),
            "transition/physics_valid_fraction": batch["physics_valid_mask"].float().mean(),
            "transition/push_fraction": batch["instantaneous_push_mask"].float().mean(),
            "transition/wrench_active_fraction": batch["sustained_wrench_active_mask"].float().mean(),
            "transition/reset_fraction": batch["reset_mask"].float().mean(),
            "transition/timeout_fraction": batch["timeout_mask"].float().mean(),
            "transition/teleport_fraction": batch["teleport_mask"].float().mean(),
        })
        grf_error = (
            references.grf_yaw_n - batch["interval_grf_yaw"]
        ).abs().reshape(-1, 4, 3)
        for foot_index, foot_name in enumerate(CANONICAL.foot_names):
            for axis_index, axis in enumerate(("x", "y", "z")):
                metrics[f"grf/{foot_name}_{axis}_mae_n"] = grf_error[
                    :, foot_index, axis_index
                ].mean()
        wrench_error = (
            references.base_wrench_yaw - batch["interval_wrench_yaw"]
        ).abs()
        for axis_index, axis in enumerate(("fx", "fy", "fz", "tx", "ty", "tz")):
            units = "n" if axis_index < 3 else "nm"
            metrics[f"wrench/{axis}_mae_{units}"] = wrench_error[:, axis_index].mean()

        intervention = qp_result.correction.norm(dim=-1) > 1.0e-5
        metrics["qp/intervention_fraction"] = intervention.float().mean()
        for label, mask in (("intervened", intervention), ("not_intervened", ~intervention)):
            denominator = mask.float().sum().clamp_min(1.0)
            metrics[f"qp_conditioned/{label}_grf_mae_n"] = (
                grf_error.mean(dim=(1, 2)) * mask.float()
            ).sum() / denominator
            metrics[f"qp_conditioned/{label}_wrench_mae"] = (
                wrench_error.mean(dim=-1) * mask.float()
            ).sum() / denominator
        for status_code in range(3):
            metrics[f"qp/status_{status_code}_fraction"] = (
                qp_result.status == status_code
            ).float().mean()
        return {
            "physics": physics,
            "auxiliary": auxiliary,
            "safe_torque": safe,
            "nominal_torque": nominal,
            "metrics": metrics,
            "qp_result": qp_result,
        }

    def step(self, actions, physics_estimator=None, *, allow_missing_physics_references=False):
        self.teleport_mask.zero_()
        pre_state = self.backend.canonical_state()
        pre_q = torch.cat((
            pre_state.base_position_world, pre_state.base_quaternion_xyzw,
            pre_state.joint_position,
        ), dim=-1).clone()
        pre_v = pre_state.velocity_world.clone()
        coupled = self._delay_action(actions)
        nominal = self._nominal_torque(coupled)
        raw_action_applied = self.raw_sampled_action.clone()
        delayed_action_applied = self.delayed_nominal_action.clone()
        exploration_noise_applied = self.pending_exploration_noise.clone()
        previous_safe_torque_applied = self.previous_safe_torque.clone()
        feedback_branch_weight_applied = self.simulator.feedback_tau_weight.clone()
        feedforward_branch_weight_applied = self.simulator.feedforward_tau_weight.clone()
        if feedback_branch_weight_applied.ndim == 1:
            feedback_branch_weight_applied = feedback_branch_weight_applied.unsqueeze(-1)
            feedforward_branch_weight_applied = feedforward_branch_weight_applied.unsqueeze(-1)

        push_delta, push_mask = self.instantaneous_pushes.sample(self.common_step_counter)
        push_delta_applied = push_delta.clone()
        push_mask_applied = push_mask.clone()
        self.backend.add_root_velocity_delta_world(push_delta, push_mask)
        wrench_world, wrench_active = self.sustained_wrench.step(self.common_step_counter)
        wrench_world_applied = wrench_world.clone()
        wrench_active_applied = wrench_active.clone()

        if physics_estimator is not None:
            encoded = getattr(physics_estimator, "_last_encoder_output", None)
            if encoded is None:
                encoded = physics_estimator.encode_policy_history(self.obs_history)
            references = physics_estimator.physics_references(
                self.obs_history, nominal, encoded=encoded
            )
            predicted_grf = references.grf_yaw_n
            predicted_wrench = references.base_wrench_yaw
            contact_probability = encoded.explicit[:, 3:7]
            self._physics_reference_serial += 1
        else:
            if bool(self.cfg.features.use_qp) and self.init_done and not allow_missing_physics_references:
                raise RuntimeError(
                    "HardPACT QP requires the deployment physics estimator before env.step"
                )
            predicted_grf = nominal.new_zeros(self.num_envs, 12)
            predicted_wrench = nominal.new_zeros(self.num_envs, 6)
            contact_probability = nominal.new_zeros(self.num_envs, 4)

        qp_result = self._solve_qp(
            pre_state, predicted_grf, predicted_wrench, contact_probability
        )
        self.last_qp_result = qp_result
        self.safe_torque.copy_(qp_result.safe_torque)
        self.executed_torque.copy_(self.safe_torque)
        self.torque_average.copy_(self.safe_torque)
        self.torque_peak.copy_(self.safe_torque.abs())
        safe_torque_applied = self.safe_torque.clone()
        executed_torque_applied = self.executed_torque.clone()
        torque_average_applied = self.torque_average.clone()
        torque_peak_applied = self.torque_peak.clone()
        self.grf_processor.begin_interval()
        self.backend.step_safe_torque(
            self.safe_torque, wrench_world, self.grf_processor.update_substep
        )
        interval_grf_world = self.grf_processor.end_interval().clone()
        self.simulator._grfs_buf.copy_(self.grf_processor.ema.flatten(1))
        self.post_physics_step()
        out_of_bounds = getattr(
            self.simulator, "_base_pos_out_of_bounds_buf", None
        )
        if out_of_bounds is not None:
            self.teleport_mask |= out_of_bounds.bool().unsqueeze(-1)

        post_state = self.backend.canonical_state()
        post_q = torch.cat((
            post_state.base_position_world, post_state.base_quaternion_xyzw,
            post_state.joint_position,
        ), dim=-1).clone()
        post_v = post_state.velocity_world.clone()
        interval_grf_yaw = world_to_yaw_local(
            interval_grf_world, pre_state.base_quaternion_xyzw
        ).flatten(1)
        interval_wrench_yaw = torch.cat((
            world_to_yaw_local(wrench_world_applied[:, :3], pre_state.base_quaternion_xyzw),
            world_to_yaw_local(wrench_world_applied[:, 3:], pre_state.base_quaternion_xyzw),
        ), dim=-1)
        sustained_wrench_yaw_normalized = self.sustained_wrench.yaw_local_normalized(
            pre_state.base_quaternion_xyzw
        ).clone()
        reset_mask = self.reset_buf.bool().unsqueeze(-1)
        timeout_mask = self.time_out_buf.bool().unsqueeze(-1)
        teleport_mask = self.teleport_mask.clone()
        valid = physics_transition_mask(reset_mask, timeout_mask, teleport_mask, push_mask_applied)
        randomized = RECONSTRUCTION_SCHEMA.system_identification_vector(
            self.reconstruction_target
        ).clone()
        self.last_transition = {
            "raw_action": raw_action_applied,
            "delayed_nominal_action": delayed_action_applied,
            "nominal_torque": nominal.clone(),
            "feedback_branch_weight": feedback_branch_weight_applied,
            "feedforward_branch_weight": feedforward_branch_weight_applied,
            "previous_safe_torque": previous_safe_torque_applied,
            "safe_torque": safe_torque_applied,
            "executed_torque": executed_torque_applied,
            "exploration_noise": exploration_noise_applied,
            "pre_q": pre_q,
            "pre_v": pre_v,
            "post_q": post_q,
            "post_v": post_v,
            "average_torque": torque_average_applied,
            "peak_torque": torque_peak_applied,
            "interval_grf_yaw": interval_grf_yaw,
            "interval_wrench_yaw": interval_wrench_yaw,
            "instantaneous_push_delta_world": push_delta_applied,
            "instantaneous_push_mask": push_mask_applied.float(),
            "sustained_wrench_world": wrench_world_applied,
            "sustained_wrench_yaw_normalized": sustained_wrench_yaw_normalized,
            "sustained_wrench_active_mask": wrench_active_applied.float(),
            "reset_mask": reset_mask.float(),
            "timeout_mask": timeout_mask.float(),
            "teleport_mask": teleport_mask.float(),
            "physics_valid_mask": valid.float(),
            "explicit_estimator_target": self.explicit_labels_buf.clone(),
            "reconstruction_target": self.reconstruction_target.clone(),
            "randomized_parameters": randomized,
            "qp_correction": qp_result.correction.clone(),
            "qp_contact_slack": qp_result.contact_slack.clone(),
            "qp_residuals": torch.cat((
                qp_result.equality_residual,
                qp_result.inequality_violation,
                qp_result.minimum_margin,
            ), dim=-1),
            "qp_active_constraints": qp_result.active_constraints.clone(),
            "qp_fallback": qp_result.fallback.clone(),
            "qp_status": qp_result.status.clone(),
            "qp_forward_time_ms": qp_result.forward_time_ms.clone(),
        }
        validate_transition(
            self.last_transition, position_pretraining=self.position_pretraining
        )
        self.previous_safe_torque.copy_(self.safe_torque)
        self.pending_exploration_noise.zero_()
        clip_obs = float(self.cfg.normalization.clip_observations)
        self.obs_buf.clamp_(-clip_obs, clip_obs)
        self.privileged_obs_buf.clamp_(-clip_obs, clip_obs)
        self.extras["hard_pact_transition"] = self.last_transition
        self.extras["qp"] = {
            "correction_mean": qp_result.correction.norm(dim=-1).mean(),
            "fallback_fraction": (qp_result.fallback > 0).float().mean(),
            "active_constraints_mean": qp_result.active_constraints.float().mean(),
            "forward_time_ms": qp_result.forward_time_ms.mean(),
        }
        return (
            self.obs_buf, self.privileged_obs_buf, self.obs_history,
            self.explicit_labels_buf, self.rew_buf, self.reset_buf,
            self.extras, interval_grf_yaw * self.obs_scales.grf,
        )

    def _system_identification_fields(self):
        sim = self.simulator
        def joints(value):
            return self.backend.ordered_backend_joint_values(value)[:, :12]

        friction = sim._friction_values[:, :1]
        delay = self.action_delay.float().unsqueeze(-1) * float(self.dt)
        return {
            "added_base_mass": sim._added_base_mass[:, :1],
            "base_com_shift": sim._base_com_bias[:, :3],
            "kp_scale": joints(sim._kp_scale),
            "kd_scale": joints(sim._kd_scale),
            "motor_strength_scale": joints(sim._motor_strength),
            "joint_armature": sim._joint_armature[:, :1],
            "joint_friction": sim._joint_friction[:, :1],
            "joint_stiffness": sim._joint_stiffness[:, :1],
            "joint_damping": sim._joint_damping[:, :1],
            "ground_friction": friction,
            "control_delay": delay,
        }

    def compute_observations(self):
        self.llast_obs_buf.copy_(self.last_obs_buf)
        self.last_obs_buf.copy_(self.obs_buf)
        action_history = self.delayed_nominal_action
        if self.position_pretraining:
            # Required transfer contract: executed PD torque, not the dormant
            # feedforward-head prediction, occupies the second action block.
            action_history = torch.cat((
                self.delayed_nominal_action[:, :12],
                self.executed_torque / float(self.cfg.control.torque_scale),
            ), dim=-1)
        self.actions.copy_(action_history)
        self.obs_buf = torch.cat((
            self.commands[:, :3] * self.commands_scale,
            self.simulator.projected_gravity,
            self.simulator.base_ang_vel * self.obs_scales.ang_vel,
            (self.simulator.dof_pos[:, :12] - self._default_joint_position())
            * self.obs_scales.dof_pos,
            self.simulator.dof_vel[:, :12] * self.obs_scales.dof_vel,
            action_history,
        ), dim=-1)
        if self.obs_buf.shape[-1] != 57:
            raise RuntimeError(f"HardPACT actor observation must be 57-D, got {self.obs_buf.shape[-1]}")
        if self.add_noise:
            self.obs_buf += (2.0 * torch.rand_like(self.obs_buf) - 1.0) * self.noise_scale_vec

        clearance = torch.clamp(
            self.simulator.feet_pos[:, :, 2]
            - torch.mean(self.simulator.height_around_feet, dim=-1)
            - float(self.cfg.rewards.foot_height_offset),
            -1.0, 1.0,
        )
        self.explicit_labels_buf = torch.cat((
            self.simulator.base_lin_vel * self.obs_scales.lin_vel,
            self.grf_processor.contacts.float(),
            clearance,
        ), dim=-1)
        next_fields = {
            "next_base_linear_velocity_body": self.simulator.base_lin_vel,
            "next_projected_gravity": self.simulator.projected_gravity,
            "next_base_angular_velocity_body": self.simulator.base_ang_vel,
            "next_joint_position_error": self.simulator.dof_pos[:, :12] - self._default_joint_position(),
            "next_joint_velocity": self.simulator.dof_vel[:, :12],
        }
        next_fields.update(self._system_identification_fields())
        self.reconstruction_target = RECONSTRUCTION_SCHEMA.build(next_fields, normalized=True)

        self.llast_obs_hist.copy_(self.last_obs_hist)
        self.last_obs_hist.copy_(self.obs_history)
        self.obs_history_deque.append(self.obs_buf)
        self.obs_history = torch.cat(tuple(self.obs_history_deque), dim=-1)

        stages = self.grf_processor.flattened_stages()
        push = torch.cat((
            self.instantaneous_pushes.actual_delta_world,
            self.instantaneous_pushes.event_mask.float(),
        ), dim=-1)
        wrench_local = self.sustained_wrench.yaw_local_normalized(self.simulator.base_quat)
        wrench = torch.cat((wrench_local, self.sustained_wrench.active_mask.float()), dim=-1)
        masks = torch.cat((
            self.reset_buf.float().unsqueeze(-1),
            self.time_out_buf.float().unsqueeze(-1),
            self.teleport_mask.float(),
            (~physics_transition_mask(
                self.reset_buf.bool().unsqueeze(-1),
                self.time_out_buf.bool().unsqueeze(-1),
                self.teleport_mask,
                self.instantaneous_pushes.event_mask,
            )).float(),
        ), dim=-1)
        critic = torch.cat((
            self.obs_buf,
            self.explicit_labels_buf,
            stages["raw"] * self.obs_scales.grf,
            stages["deadbanded"] * self.obs_scales.grf,
            stages["clipped"] * self.obs_scales.grf,
            stages["ema"] * self.obs_scales.grf,
            stages["interval_average"] * self.obs_scales.grf,
            push,
            wrench,
            self.sustained_wrench.current_world,
            RECONSTRUCTION_SCHEMA.system_identification_vector(
                self.reconstruction_target
            ),
            masks,
        ), dim=-1)
        if critic.shape[-1] != self.cfg.env.num_privileged_obs:
            raise RuntimeError(
                f"HardPACT critic schema expected {self.cfg.env.num_privileged_obs}, got {critic.shape[-1]}"
            )
        self.privileged_obs_buf = critic

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        if not hasattr(self, "grf_processor") or env_ids.numel() == 0:
            return
        self.grf_processor.reset(env_ids)
        self.instantaneous_pushes.reset(env_ids, self.common_step_counter)
        self.sustained_wrench.reset(env_ids, self.common_step_counter)
        self.previous_safe_torque[env_ids] = 0.0
        self.nominal_torque[env_ids] = 0.0
        self.safe_torque[env_ids] = 0.0
        self.executed_torque[env_ids] = 0.0
        self.torque_average[env_ids] = 0.0
        self.torque_peak[env_ids] = 0.0
        self.raw_sampled_action[env_ids] = 0.0
        self.delayed_nominal_action[env_ids] = 0.0
        self.pending_exploration_noise[env_ids] = 0.0
        self.teleport_mask[env_ids] = self.non_failure_reset_buf[env_ids].unsqueeze(-1)
        self.action_queue[env_ids] = 0.0
        if self.cfg.domain_rand.randomize_ctrl_delay:
            low, high = map(int, self.cfg.domain_rand.ctrl_delay_step_range)
            self.action_delay[env_ids] = torch.randint(
                low, high + 1, (env_ids.numel(),), device=self.device
            )
        else:
            self.action_delay[env_ids] = 0

    def _post_physics_step_callback(self):
        # Command handling is preserved; the legacy simulator push is replaced
        # by the independently recorded six-dimensional disturbance above.
        env_ids = (
            self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt) == 0
        ).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.simulator.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(
                0.5 * wrap_to_pi(self.commands[:, 3] - heading),
                self.cfg.commands.ranges.ang_vel_yaw[0],
                self.cfg.commands.ranges.ang_vel_yaw[1],
            )

    def _feet_contact_mask(self):
        """Use the conditioned canonical GRF contact definition everywhere."""
        return self.grf_processor.contacts

    def _reward_edge_grid(self):
        """Reward-only terrain buffers; never exposed to policy, QP, or targets."""
        sim = self.simulator
        if self.cfg.terrain.mesh_type == "plane":
            return None, None
        height = getattr(sim, "_height_samples", None)
        edge = getattr(sim, "_edge_mask", None)
        if edge is None and hasattr(sim, "_terrain"):
            edge = getattr(sim._terrain, "edge_mask", None)
        if height is None or edge is None:
            raise RuntimeError(
                f"{self.backend_name} must expose reward terrain heights and edges"
            )
        if not torch.is_tensor(edge):
            edge = torch.as_tensor(edge, device=self.device, dtype=torch.bool)
        else:
            edge = edge.to(device=self.device, dtype=torch.bool)
        return height, edge

    def _feet_grid_indices(self, grid):
        points = (
            (self.simulator.feet_pos[:, :, :2] + self.cfg.terrain.border_size)
            / self.cfg.terrain.horizontal_scale
        ).long()
        return (
            points[:, :, 0].clamp(0, grid.shape[0] - 1),
            points[:, :, 1].clamp(0, grid.shape[1] - 1),
        )

    def _feet_near_edge_mask(self):
        if getattr(self, "_feet_edge_cache_step", -1) == self.common_step_counter:
            return self._feet_edge_cache
        _, edge = self._reward_edge_grid()
        if edge is None:
            near_edge = torch.zeros(
                self.num_envs, 4, device=self.device, dtype=torch.bool
            )
        else:
            feet_xy = self.simulator.feet_pos[:, :, :2]
            px, py = self._feet_grid_indices(edge)
            near_edge = torch.zeros_like(px, dtype=torch.bool)
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    ex = (px + dx).clamp(0, edge.shape[0] - 1)
                    ey = (py + dy).clamp(0, edge.shape[1] - 1)
                    edge_xy = torch.stack((
                        ex.float() * self.cfg.terrain.horizontal_scale
                        - self.cfg.terrain.border_size,
                        ey.float() * self.cfg.terrain.horizontal_scale
                        - self.cfg.terrain.border_size,
                    ), dim=-1)
                    near_edge |= edge[ex, ey] & (
                        torch.norm(feet_xy - edge_xy, dim=-1)
                        < self.cfg.rewards.feet_edge_threshold
                    )
        self._feet_edge_cache = near_edge
        self._feet_edge_cache_step = self.common_step_counter
        return near_edge

    def _reward_max_height_ahead_feet(self):
        height, _ = self._reward_edge_grid()
        if height is None:
            return torch.zeros(self.num_envs, 4, device=self.device)
        px, py = self._feet_grid_indices(height)
        direction_base = torch.cat((
            self.simulator.base_lin_vel[:, :2],
            torch.zeros(self.num_envs, 1, device=self.device),
        ), dim=-1)
        yaw = yaw_quaternion_xyzw(self.simulator.base_quat)
        direction_world = quat_apply(yaw, direction_base)[:, :2]
        norm = direction_world.norm(dim=-1, keepdim=True)
        heading = quat_apply(
            yaw, direction_base.new_tensor([1.0, 0.0, 0.0]).expand(self.num_envs, -1)
        )[:, :2]
        direction_world = torch.where(
            norm > 1.0e-4, direction_world / norm.clamp_min(1.0e-6), heading
        )
        forward = direction_world.unsqueeze(1).expand(-1, 4, -1)
        lateral = torch.stack((-forward[:, :, 1], forward[:, :, 0]), dim=-1)
        maximum = None
        for forward_cells in self.cfg.rewards.edge_clearance_forward_cells:
            for lateral_cells in self.cfg.rewards.edge_clearance_lateral_cells:
                offset = torch.round(
                    forward * forward_cells + lateral * lateral_cells
                ).long()
                sx = (px + offset[:, :, 0]).clamp(0, height.shape[0] - 1)
                sy = (py + offset[:, :, 1]).clamp(0, height.shape[1] - 1)
                sample = height[sx, sy]
                maximum = sample if maximum is None else torch.maximum(maximum, sample)
        return maximum * self.cfg.terrain.vertical_scale

    def _reward_edge_swing_clearance(self):
        swing = ~self._feet_contact_mask()
        target_z = (
            self._reward_max_height_ahead_feet()
            + self.cfg.rewards.edge_swing_clearance_margin
        )
        clearance_error = torch.relu(target_z - self.simulator.feet_pos[:, :, 2])
        mask = (self._feet_near_edge_mask() & swing).float()
        return torch.sum(mask * clearance_error.square(), dim=-1)

    def _reward_torque_cancellation(self):
        feedback = self.simulator.combined_feedback_torques
        feedforward = self.simulator.combined_feedforward_torques
        cancellation = (
            feedback.abs() + feedforward.abs() - (feedback + feedforward).abs()
        ).clamp_min(0.0)
        limits = self.simulator.torque_limits[:12].clamp_min(1.0e-6)
        normalized_excess = torch.relu(
            cancellation / limits
            - float(self.cfg.rewards.torque_cancellation_deadband)
        )
        return normalized_excess.square().mean(dim=-1)


class Go2HardPACT(Go2HardPACTCore):
    """Thin coupled-action task; all behavior lives in the shared core."""

    position_pretraining = False
