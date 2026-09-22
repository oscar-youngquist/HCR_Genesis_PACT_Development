"""Upstream policy labels over the retained B1Z1 physics and observation buffers."""
import torch
from legged_gym.envs.b1z1.b1z1_unifp.b1z1_unifp import B1Z1UniFP


class B1Z1UniFPOriginal(B1Z1UniFP):
    reject_external_forces = False

    def original_adaptation_target(self, labels):
        # Retain the internal 20-D buffer; upstream places EE force before base force.
        return torch.cat((labels[:, :6], labels[:, 9:12], labels[:, 6:9]), dim=-1)

    def get_observations(self):
        obs, history, privileged, labels = super().get_observations()
        return obs, history, privileged, self.original_adaptation_target(labels)

    def step(self, actions):
        result = list(super().step(actions))
        result[3] = self.original_adaptation_target(result[3])
        return tuple(result)
