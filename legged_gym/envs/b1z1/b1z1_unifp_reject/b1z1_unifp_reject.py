"""Estimator-only force cancellation in the existing yaw/local command frame."""
import torch
from legged_gym.envs.b1z1.b1z1_unifp_original.b1z1_unifp_original import B1Z1UniFPOriginal


class B1Z1UniFPReject(B1Z1UniFPOriginal):
    reject_external_forces = True

    @property
    def force_command_stream_enabled(self):
        return False

    def set_impedance_force_estimates(self, prediction):
        prediction = torch.as_tensor(prediction, device=self.device,
                                     dtype=self.estimated_ee_force_local.dtype).detach()
        if prediction.shape != (self.num_envs, 12):
            raise ValueError("Expected 12-D [velocity, EE position, EE force, base force]")
        self.estimated_ee_force_local.copy_(prediction[:, 6:9] / self.obs_scales.ee_force)
        self.estimated_base_force_local.copy_(prediction[:, 9:12] / self.obs_scales.base_force)

    def _reset_impedance_force_filters(self, env_ids):
        super()._reset_impedance_force_filters(env_ids)
        self.estimated_ee_force_local[env_ids] = 0.
        self.estimated_base_force_local[env_ids] = 0.

    def step(self, actions):
        # Generic play overrides this option; rejection must also remain active there.
        self.cfg.commands.use_external_impedance_compensation = True
        return super().step(actions)
