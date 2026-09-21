"""Source-policy snapshots for B1Z1's control-step action-delay queue."""

import torch


class B1Z1ActionReplay:
    def __init__(self, max_delay):
        self.length = int(max_delay) + 1
        self.queue = None

    def push(self, transition, delay):
        # Store both random draws, not frozen policy outputs: replay retains gradients.
        current = torch.cat((
            transition.observations, transition.histories, transition.latent_noise,
            (transition.actions - transition.mu) / transition.sigma.clamp_min(1e-8),
            torch.ones_like(transition.mu[:, :1]),
        ), dim=-1).detach()
        if self.queue is None:
            self.queue = current.new_zeros(current.shape[0], self.length, current.shape[1])
        self.queue[:, 1:] = self.queue[:, :-1].clone()
        self.queue[:, 0] = current
        rows = torch.arange(current.shape[0], device=current.device)
        return self.queue[rows, delay.to(current.device).long()].clone()

    def reset(self, dones):
        # Initial zero-action slots have no policy source and cannot supervise PINN.
        self.queue[dones.flatten().bool()] = 0
