from .actor_critic_unifp import ActorCriticUniFP


class ActorCriticB1UniFP(ActorCriticUniFP):
    """UniFP concurrent-state estimator specialized to B1 torso dynamics."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.adaptation_labels = ["base_velocity_loss", "force_base_loss"]
        self.adaptation_dims = [3, 3]
        self.adaptation_weights = [0.2, 1.0]

