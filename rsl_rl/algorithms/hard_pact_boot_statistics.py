"""Valid-row sufficient statistics for HardPACT encoder bootstrapping."""
import random

import torch


class ValidBootStatistics:
    def __init__(self):
        self.count = 0
        self.target_sum = None
        self.target_square_sum = None
        self.error_square_sum = None

    @torch.no_grad()
    def add(self, target, reconstruction, valid):
        # Select before arithmetic: invalid rows may contain NaN or infinity.
        rows = valid.reshape(-1).bool()
        target = target.detach()[rows].double()
        reconstruction = reconstruction.detach()[rows].double()
        if target.shape[0] == 0:
            return
        if self.target_sum is None:
            self.target_sum = torch.zeros_like(target[0])
            self.target_square_sum = torch.zeros_like(target[0])
            self.error_square_sum = target.new_zeros(())
        self.target_sum += target.sum(0)
        self.target_square_sum += target.square().sum(0)
        self.error_square_sum += (reconstruction - target).square().sum()
        self.count += target.shape[0]

    def errors(self):
        if not self.count:
            return None
        variance = (self.target_square_sum / self.count
                    - (self.target_sum / self.count).square()).clamp_min(0).mean()
        mse = self.error_square_sum / (self.count * self.target_sum.numel())
        return variance, mse

    def update_boot(self, algorithm):
        errors = self.errors()
        if errors is None:
            return  # Preserve both boot state and Python RNG on empty updates.
        variance, mse = errors
        probability = float(torch.tanh(variance / (mse * algorithm.boot_mult + 1e-8)))
        algorithm.use_boot = random.random() < probability
