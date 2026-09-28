"""Per-environment reset sampling, independent of simulator and training RNG."""
import torch


class ResetRandomizationCadence:
    def __init__(self, num_envs, device, interval):
        if int(interval) != interval or interval < 0:
            raise ValueError("reset_resample_episodes must be a nonnegative integer")
        self.interval = int(interval)
        self.episodes = torch.zeros(num_envs, device=device, dtype=torch.long)
        self.started = torch.zeros(num_envs, device=device, dtype=torch.bool)
        self.ranges, self.generations, self.last, self.seen = {}, {}, {}, {}

    def begin_reset(self, env_ids):
        # Each environment's first reset initializes it, not a completed episode.
        self.episodes[env_ids] += self.started[env_ids].long()
        self.started[env_ids] = True

    def update_ranges(self, ranges):
        for name, bounds in ranges.items():
            bounds = tuple(bounds)
            if self.ranges.get(name) != bounds:
                self.ranges[name] = bounds
                self.generations[name] = self.generations.get(name, 0) + 1

    def select(self, name, env_ids):
        if name not in self.last:
            self.last[name] = torch.full_like(self.episodes, -self.interval)
            self.seen[name] = torch.full_like(self.episodes, -1)
        generation = self.generations.get(name, 0)
        due = ((self.episodes[env_ids]-self.last[name][env_ids] >= self.interval)
               | (self.seen[name][env_ids] != generation))
        selected = env_ids[due]
        self.last[name][selected] = self.episodes[selected]
        self.seen[name][selected] = generation
        return selected
