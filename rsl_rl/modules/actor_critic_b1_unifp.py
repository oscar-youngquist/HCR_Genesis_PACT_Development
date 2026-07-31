import torch

from .actor_critic_unifp import ActorCriticUniFP


class ActorCriticB1UniFP(ActorCriticUniFP):
    """UniFP concurrent-state estimator specialized to B1 torso dynamics."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.adaptation_labels = [
            "base_velocity_loss", "force_base_loss", "foot_contact_loss", "foot_height_loss"
        ]
        self.adaptation_dims = [3, 3, 4, 4]
        self.adaptation_weights = [0.2, 1.0, 1.0, 1.0]

    def actor_estimates(self, explicit_prediction):
        """Convert contact logits while leaving velocity, force, and height continuous."""
        return torch.cat((
            explicit_prediction[:, :6],
            torch.sigmoid(explicit_prediction[:, 6:10]),
            explicit_prediction[:, 10:14],
        ), dim=-1)
