"""Shared position-control randomization for Go2 DreamWaQ backends."""
import torch

from .go2_domain_rand_curriculum import Go2DomainRandCurriculum
from .reset_randomization_cadence import ResetRandomizationCadence


class DreamWaQAdapter:
    def __init__(self, cfg, sim_params, device, headless):
        interval = getattr(cfg.domain_rand, 'reset_resample_episodes', 0)
        if int(interval) != interval or interval < 0:
            raise ValueError('reset_resample_episodes must be a nonnegative integer')
        self._reset_cadence = None
        self.domain_rand_curriculum = Go2DomainRandCurriculum(cfg, getattr(cfg, 'seed', 0))
        self._cfg = cfg
        self._apply_curriculum_ranges()
        super().__init__(cfg, sim_params, device, headless)
        self._motor_strength = torch.ones(self._num_envs, self._num_actions, device=device)
        self.raw_delayed_actions = torch.zeros_like(self._motor_strength)
        self.requested_torques = torch.zeros_like(self._motor_strength)
        self._joint_stiffness = torch.zeros(self._num_envs, 1, device=device)
        self._rand_wrench_vels = torch.zeros(self._num_envs, 3, device=device)
        self._push_timers = torch.zeros(self._num_envs, 3, device=device)
        self._reset_adapter(torch.arange(self._num_envs, device=device))
        self._restart_reset_cadence()

    def _bounded_position_torques(self, actions):
        """Aligned PACT position control: raw reward request, bounded execution.

        Both paths use the live state at every physics substep. Motor strength
        multiplies the PD torque once; the default pose is added once.
        """
        def requested(command):
            target = self.default_dof_pos + command * self._cfg.control.action_scale
            return self._motor_strength * (
                self._kp_scale * self._p_gains * (target - self.dof_pos)
                - self._kd_scale * self._d_gains * self.dof_vel)

        self.requested_torques.copy_(requested(self.raw_delayed_actions))
        return requested(actions).clamp(-self.torque_limits, self.torque_limits)

    def _restart_reset_cadence(self):
        # Simulator trajectories/samples are not checkpointed. A resumed run
        # starts fresh episodes and must install every restored range once.
        interval = getattr(self._cfg.domain_rand, 'reset_resample_episodes', 0)
        self._reset_cadence = (ResetRandomizationCadence(self._num_envs, self._device, interval)
                               if interval > 1 else None)
        self._update_reset_ranges()

    def _update_reset_ranges(self):
        if self._reset_cadence is None:
            return
        ranges = self._randomization_ranges
        self._reset_cadence.update_ranges({
            'friction': ranges['ground_friction'],
            'base_mass': ranges['added_base_mass'],
            'com_displacement': tuple(value for axis in 'xyz' for value in ranges['base_com_' + axis]),
            'joint_armature': ranges['armature'],
            'joint_friction': ranges['joint_friction'],
            'joint_damping': ranges['joint_damping'],
            'joint_stiffness': ranges['joint_stiffness'],
            'pd_gain': tuple(ranges['kp_scale']) + tuple(ranges['kd_scale']),
            'motor_strength': ranges['motor_strength'],
        })

    def _reset_sample_ids(self, name, env_ids):
        return env_ids if self._reset_cadence is None else self._reset_cadence.select(name, env_ids)

    def _reset_domain_randomization(self, env_ids):
        if self._reset_cadence is None:
            return super()._reset_domain_randomization(env_ids)
        for name in ('friction', 'base_mass', 'com_displacement', 'joint_armature',
                     'joint_friction', 'joint_damping', 'pd_gain'):
            if getattr(self._cfg.domain_rand, 'randomize_' + name, False):
                ids = self._reset_sample_ids(name, env_ids)
                if len(ids):
                    getattr(self, '_randomize_' + name)(ids)

    def _apply_curriculum_ranges(self):
        ranges = self.domain_rand_curriculum.effective_ranges()
        d = self._cfg.domain_rand
        for field, name in (
            ('friction_range', 'ground_friction'), ('added_mass_range', 'added_base_mass'),
            ('com_pos_x_range', 'base_com_x'), ('com_pos_y_range', 'base_com_y'),
            ('com_pos_z_range', 'base_com_z'), ('joint_friction_range', 'joint_friction'),
            ('joint_damping_range', 'joint_damping'), ('joint_armature_range', 'armature'),
        ):
            setattr(d, field, ranges[name])
        self._randomization_ranges = ranges
        self._update_reset_ranges()

    def advance_domain_randomization(self, iteration, tracking_reward):
        if self._cfg.domain_rand.use_domainrand_curriculum:
            self.domain_rand_curriculum.advance(iteration, tracking_reward)
            self._apply_curriculum_ranges()

    def load_domain_randomization(self, state):
        self.domain_rand_curriculum.load_state_dict(state)
        self._apply_curriculum_ranges()
        self._restart_reset_cadence()

    def _sample_intervals(self, count):
        d = self._cfg.domain_rand
        # Independent per-environment event intervals, as in Go2 PACT.
        bounds = ((d.push_interval_min, d.push_interval_max),
                  (d.vert_interval_min, d.vert_interval_max),
                  (d.wrench_timeout_min, d.wrench_timeout_max))
        return torch.stack([
            torch.empty(count, device=self._device).uniform_(low, high)
            for low, high in bounds
        ], dim=-1)

    def _reset_adapter(self, env_ids):
        if len(env_ids) == 0:
            return
        d = self._cfg.domain_rand
        if d.randomize_motor_strength:
            ids = self._reset_sample_ids('motor_strength', env_ids)
            if len(ids):
                self._motor_strength[ids] = torch.empty(
                    len(ids), self._num_actions, device=self._device).uniform_(*d.motor_strength_range)
        if d.randomize_joint_stiffness:
            ids = self._reset_sample_ids('joint_stiffness', env_ids)
            if len(ids):
                values = torch.empty(len(ids), 1, device=self._device).uniform_(
                    *self._randomization_ranges['joint_stiffness'])
                self._joint_stiffness[ids] = values
                self._install_joint_stiffness(ids, values.expand(-1, self._num_actions))
        self._push_timers[env_ids] = self._sample_intervals(len(env_ids))
        self._rand_push_vels[env_ids] = 0
        self._rand_wrench_vels[env_ids] = 0

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        if self._reset_cadence is not None:
            self._reset_cadence.begin_reset(env_ids)
        super().reset_idx(env_ids)
        self._reset_adapter(env_ids)
        self.raw_delayed_actions[env_ids] = 0
        self.requested_torques[env_ids] = 0

    def push_robots(self):
        self._push_timers -= self._cfg.control.decimation * self._sim_params['dt']
        due = self._push_timers <= 0
        ranges = self._randomization_ranges
        self._rand_push_vels.zero_()
        self._rand_wrench_vels.zero_()
        self._rand_push_vels[:, :2] = torch.empty(self._num_envs, 2, device=self._device).uniform_(
            *ranges['push_xy']) * due[:, 0:1]
        self._rand_push_vels[:, 2:3] = torch.empty(self._num_envs, 1, device=self._device).uniform_(
            *ranges['push_z']) * due[:, 1:2]
        self._rand_wrench_vels[:] = torch.empty(self._num_envs, 3, device=self._device).uniform_(
            *ranges['push_angular']) * due[:, 2:3]
        ids = due.any(dim=-1).nonzero(as_tuple=False).flatten()
        if len(ids):
            self._apply_velocity_push(ids, self._rand_push_vels[ids], self._rand_wrench_vels[ids])
        self._push_timers.copy_(torch.where(due, self._sample_intervals(self._num_envs), self._push_timers))
