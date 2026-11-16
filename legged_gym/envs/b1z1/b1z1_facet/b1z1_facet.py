"""FACET references, temporal rewards and packets; shared Pos execution stays intact."""
import torch
import numpy as np

from legged_gym.envs.b1z1.b1z1_ppo_pos.b1z1_ppo_pos import B1Z1PPOPos
from legged_gym.envs.b1z1.directional_compliance import heading, rotate_yaw, sphere_basis
from .reference import FACETReference, FACETReferenceBank, applied_loads, measured_reference_state
from .tracking import COMPONENTS, LEGACY_TRACKING, tracking_values
from .observations import FACETObservations
from .schedules import FACETForceCurriculum
from .forces import FACETForces, FACETExposure


class B1Z1FACET(B1Z1PPOPos):
    """FACET packets include terminal successors for the dedicated Stage 5 runner."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.cfg.facet_tracking.enabled:
            self._staged_force_curriculum = FACETForceCurriculum(self.cfg, self.device)
        self._facet_forces = None
        if getattr(self.cfg, 'facet_forces', None) is not None and self.cfg.facet_forces.enabled:
            self._staged_force_curriculum = FACETExposure()
            self._facet_forces = FACETForces(self.num_envs, self.device, self.cfg.facet_forces)
        self._facet_exposure = torch.zeros(self.num_envs, 3, device=self.device)
        self.facet_reference = None
        self.facet_history = None
        self._facet_interval = None
        self._facet_seed = None
        self._facet_snapshot = None
        if self.cfg.facet_reference.enabled:
            self.facet_reference = FACETReference(self.num_envs, self.device, self.cfg.facet_reference)
            self._reset_facet_reference(self.all_env_ids)
            if self.cfg.facet_tracking.enabled:
                self.facet_history = FACETReferenceBank(self.facet_reference, self.cfg.facet_tracking.horizons)
        self._facet_observations = None
        self._facet_issued_action = None
        self._facet_terminal = None
        if self.cfg.facet_observations.enabled:
            self._facet_observations = FACETObservations(self)
            self._facet_observations.collect(self, self.all_env_ids, reset=True)

    def _facet_measured_state(self):
        sim = self.simulator
        # Isaac Lab's articulation buffers are updated by reset writes; the
        # simulator's policy-step EE/base cache may still describe a terminal state.
        if hasattr(getattr(sim, "_robot", None), "data"):
            data = sim._robot.data
            quat = data.root_link_quat_w[:, [1, 2, 3, 0]]  # wxyz -> xyzw
            yaw, rate = heading(quat, data.root_link_ang_vel_w)
            return (data.root_link_pos_w, data.root_link_lin_vel_w, yaw, rate,
                    data.body_link_pos_w[:, sim._gripper_index],
                    data.body_link_vel_w[:, sim._gripper_index, :3])
        yaw, rate = heading(sim.base_quat, sim._base_world_ang_vel)
        return sim.base_pos, sim._base_world_lin_vel, yaw, rate, sim.ee_pos, sim.ee_vel

    @torch.no_grad()
    def _reset_facet_reference(self, ids):
        state = self._facet_measured_state()
        self.facet_reference.reset(ids, *(value[ids] for value in state))
        if getattr(self, "facet_history", None) is not None:
            self.facet_history.clear(ids)

    def reset_idx(self, env_ids):
        diagnostics = {}
        exposure = getattr(self, '_facet_exposure', None)
        if exposure is not None and len(env_ids):
            values = exposure[env_ids]
            diagnostics['facet/ee_load_time_mean_N'] = (values[:, 0]/values[:, 2].clamp_min(self.dt)).mean()
            diagnostics['facet/base_load_time_mean_N'] = (values[:, 1]/values[:, 2].clamp_min(self.dt)).mean()
            exposure[env_ids] = 0
        if self.cfg.facet_tracking.enabled and len(env_ids):
            counts = self._facet_valid_steps[env_ids].sum(0)
            errors = self._facet_error_sums[env_ids].sum(0)/counts[:, None].clamp_min(1)
            total = self._facet_episode_steps[env_ids].sum().clamp_min(1)
            for h, age in enumerate(self.cfg.facet_tracking.horizons):
                diagnostics[f"facet/h{age}/valid_fraction"] = counts[h]/total
                diagnostics[f"facet/h{age}/valid_steps"] = counts[h]
                for c, name in enumerate(COMPONENTS):
                    diagnostics[f"facet/h{age}/{name}_mse"] = errors[h, c]
        super().reset_idx(env_ids)
        if getattr(self, "facet_reference", None) is not None and len(env_ids):
            self._reset_facet_reference(env_ids)
        if getattr(self, '_facet_forces', None) is not None:
            self._facet_forces.reset(env_ids)
        collector = getattr(self, '_facet_observations', None)
        if collector is not None and len(env_ids):
            collector.collect(self, env_ids, reset=True)
            if self._facet_issued_action is not None:
                collector.skip[env_ids] = True  # Already collected at this boundary.
        if diagnostics:
            self.extras["episode"].update(diagnostics)
            if self.cfg.facet_tracking.enabled:
                self._facet_error_sums[env_ids] = 0
                self._facet_valid_steps[env_ids] = 0
                self._facet_episode_steps[env_ids] = 0

    @torch.no_grad()
    def _pre_sim_step(self, actions):
        self._facet_terminal = None
        if getattr(self, '_facet_observations', None) is not None:
            # Capture policy-issued positions before delay or Pos torque bookkeeping.
            limit = self.cfg.normalization.clip_actions
            self._facet_issued_action = actions.detach().to(self.device).clamp(-limit, limit).clone()
        executed = super()._pre_sim_step(actions)
        if getattr(self, '_facet_forces', None) is not None:
            self._sample_facet_forces()
        if getattr(self, "facet_reference", None) is None:
            return executed
        measured = self._facet_measured_state()
        position, _, yaw, _, _, _ = measured
        if getattr(self, "facet_history", None) is not None:
            self._facet_seed = measured_reference_state(*measured).clone()
        u = (self.goal_timer / self.traj_timesteps).clamp(0, 1)
        sphere_rate = ((self.ee_goal_sphere-self.ee_start_sphere) *
            (30*u.square()*(1-u).square()/self.traj_timesteps/self.dt)[:, None])
        radius, elevation = self.curr_ee_goal_sphere[:, 0], self.curr_ee_goal_sphere[:, 1]
        scales = torch.stack((torch.ones_like(radius), radius, radius*elevation.cos()), -1)
        cart_rate = ((sphere_rate*scales)[:, :, None]*sphere_basis(self.curr_ee_goal_sphere)).sum(1)
        local = self.curr_ee_goal_cart.clone()
        local[:, :2] += self.ee_goal_center_offset[:, :2]
        height = self.simulator.measured_heights.mean(1) + self.cfg.goal_ee.sphere_center.z_invariant_offset
        velocity_command = rotate_yaw(torch.cat((self.commands[:, :2], torch.zeros_like(yaw[:, None])), -1), yaw)
        _, _, point = self.simulator.applied_torso_wrench_source()
        # Commands, geometry and nominal derivatives at t, not the resampled
        # successor commands. Terrain height and measured base anchor are held
        # during this dt; virtual base motion transports the attached EE setpoint.
        self._facet_interval = tuple(v.detach().clone() for v in
            (position, yaw, velocity_command[:, :2], self.commands[:, 2], local, cart_rate, height, point))
        return executed

    def post_physics_step(self):
        if getattr(self, "facet_reference", None) is not None and self._facet_interval is not None:
            position, yaw, command, yaw_command, local, local_rate, height, point = self._facet_interval
            # These buffers are the actual sampled/gated loads held throughout
            # the just-completed interval. Read before parent callbacks can reset
            # or resample; use the captured start lever, never successor geometry.
            ee, base, moment = applied_loads(self.cfg.commands, self.simulator._ee_force_world,
                self.simulator.applied_torso_wrench_source(), position, point)
            if hasattr(self, '_facet_exposure'):
                self._facet_exposure += torch.stack((ee.norm(dim=-1), base.norm(dim=-1), torch.ones_like(moment)), -1)*self.dt
            self.facet_reference.advance(self.dt, position[:, :2], yaw, command, yaw_command,
                                         local, local_rate, height, ee, base, moment)
            if getattr(self, "facet_history", None) is not None:
                self.facet_history.advance(self._facet_seed, self.dt,
                    (position[:, :2], yaw, command, yaw_command, local, local_rate, height), (ee, base, moment))
                self._facet_snapshot = self.facet_history.read()
                self._facet_seed = None
            self._facet_interval = None  # Reads/post-step calls cannot advance twice.
        super().post_physics_step()
        collector = getattr(self, '_facet_observations', None)
        if collector is not None and self._facet_issued_action is not None:
            collector.collect(self, self.all_env_ids[~collector.skip], self._facet_issued_action)
            collector.skip.zero_()
            self._facet_issued_action = None

    def get_facet_observations(self):
        """Cached current-episode packet; no sampling, history advance or normalization."""
        return {} if self._facet_observations is None else self._facet_observations.read()

    def _push_gripper(self, ids):
        source = getattr(self, '_facet_forces', None)
        if source is None:
            return super()._push_gripper(ids)
        self.simulator.apply_ee_force(self.ee_force_ext_world)

    def _sample_facet_forces(self):
        # Sample independently of which physical publisher is enabled. This
        # keeps torso-only operation valid when the EE publisher is disabled.
        ee, base = self._facet_forces.sample(self.episode_length_buf*self.dt,
            self.facet_reference._parameters['ee_stiffness'], self.training_iteration*24)
        c = self.cfg.commands
        self.ee_force_ext_world.copy_(ee if c.push_gripper_stators and c.apply_ee_external_forces else torch.zeros_like(ee))
        self.base_force_ext_world.copy_(base if c.push_robot_base and c.apply_base_external_forces else torch.zeros_like(base))
        self.base_torque_ext_world.zero_()

    def _push_robot_base(self, ids):
        if getattr(self, '_facet_forces', None) is None:
            return super()._push_robot_base(ids)
        self.simulator.apply_base_force(self.base_force_ext_world)
        self.simulator.apply_base_torque(self.base_torque_ext_world)

    def get_facet_observation_schema(self):
        """Named block and joint ordering for the separate Stage 4 API."""
        from copy import deepcopy
        return {} if self._facet_observations is None else deepcopy(self._facet_observations.schema)

    def get_facet_terminal_observations(self):
        """Complete pre-reset successor packets for just the terminal rows."""
        return None if self._facet_terminal is None else {
            k: v.detach().clone() for k, v in self._facet_terminal.items()}

    def get_facet_references(self):
        """Original Stage 2 single-start diagnostic API; no actor input changes."""
        return {} if self.facet_reference is None else self.facet_reference.read()

    def _update_command_curriculum(self, env_ids):
        if not self.cfg.facet_tracking.enabled:
            return super()._update_command_curriculum(env_ids)
        key = 'facet_base_velocity'
        # Episode sums include dt; compare against the same dt-scaled maximum.
        score = self.episode_sums[key][env_ids].mean() / self.max_episode_length
        if score > self.cfg.commands.curriculum_threshold * self.reward_scales[key]:
            for axis in ('lin_vel_x', 'lin_vel_y', 'ang_vel_yaw'):
                self.command_ranges[axis][0] = np.clip(self.command_ranges[axis][0]-.2, -self.cfg.commands.max_curriculum, 0.)
                self.command_ranges[axis][1] = np.clip(self.command_ranges[axis][1]+.2, 0., self.cfg.commands.max_curriculum)

    def _prepare_reward_function(self):
        settings = self.cfg.facet_tracking
        if not settings.enabled:
            return super()._prepare_reward_function()
        if not self.cfg.facet_reference.enabled:
            raise ValueError("FACET tracking requires facet_reference.enabled")
        settings.validate()
        # Save baseline units before parent applies dt. Register ONLY optimized
        # terms; legacy metrics must not contribute to rew_buf or positive clipping.
        nominal_scales = {key: self.reward_scales[key] for key in LEGACY_TRACKING}
        for key in settings.excluded_rewards:
            self.reward_scales.pop(key, None)
        self.reward_scales.update({"facet_"+key: weight for key, weight in settings.weights.items()})
        super()._prepare_reward_function()
        for key, weight in nominal_scales.items():
            self.reward_scales[key] = weight*self.dt  # Threshold readers retain their existing normalization.
            self.episode_sums[key] = torch.zeros(self.num_envs, device=self.device)
        self._facet_bandwidths = torch.tensor([settings.bandwidths[key] for key in COMPONENTS], device=self.device)
        self._facet_values = torch.zeros(self.num_envs, len(COMPONENTS), device=self.device)
        self._facet_error_sums = torch.zeros(self.num_envs, len(settings.horizons), len(COMPONENTS), device=self.device)
        self._facet_valid_steps = torch.zeros(self.num_envs, len(settings.horizons), device=self.device)
        self._facet_episode_steps = torch.zeros(self.num_envs, device=self.device)

    @torch.no_grad()
    def compute_reward(self):
        if self.cfg.facet_tracking.enabled:
            snapshot = self._facet_snapshot
            if snapshot is not None:
                measured = measured_reference_state(*self._facet_measured_state())
                self._facet_values, e2 = tracking_values(snapshot["targets"], measured, snapshot["valid"], self._facet_bandwidths)
                snapshot["errors_squared"] = e2
                snapshot["rewards_unweighted"] = self._facet_values
                self._facet_error_sums += torch.where(snapshot["valid"][..., None], e2, 0.)
                self._facet_valid_steps += snapshot["valid"]
            self._facet_episode_steps += 1
            # Preserve EXACT parent nominal formulas and post-callback timing.
            # Nominal diagnostics only; FACET schedules use temporal rewards.
            # These keys are absent from reward_names/reward_functions.
            for key in LEGACY_TRACKING:
                self.episode_sums[key] += getattr(super(), "_reward_"+key)()*self.reward_scales[key]
        super().compute_reward()
        collector = getattr(self, '_facet_observations', None)
        if collector is not None and self._facet_issued_action is not None:
            # Parent already refreshed physics/commands and checked termination.
            # Collect once HERE, before reset clears state/loads/history. The
            # ordinary post-step hook skips these already-collected alive rows.
            collector.collect(self, self.all_env_ids, self._facet_issued_action)
            collector.skip[:] = True
            ids = self.reset_buf.nonzero(as_tuple=False).flatten()
            if len(ids):
                self._facet_terminal = {k: v[ids].detach().clone() for k, v in collector.packet.items()}
                self._facet_terminal.update(ids=ids.clone(),
                    timeouts=(self.time_out_buf[ids] & ~self.fail_buf[ids]).clone(),
                    failures=self.fail_buf[ids].clone())

    def get_facet_tracking(self):
        """Last completed transition, including terminal targets after reset; copy-on-read."""
        return {} if self._facet_snapshot is None else {k: v.detach().clone() for k, v in self._facet_snapshot.items()}

    def _reward_facet_base_position(self):
        return self._facet_values[:, 0]

    def _reward_facet_base_velocity(self):
        return self._facet_values[:, 1]

    def _reward_facet_yaw(self):
        return self._facet_values[:, 2]

    def _reward_facet_yaw_rate(self):
        return self._facet_values[:, 3]

    def _reward_facet_ee_position(self):
        return self._facet_values[:, 4]

    def _reward_facet_ee_velocity(self):
        return self._facet_values[:, 5]
