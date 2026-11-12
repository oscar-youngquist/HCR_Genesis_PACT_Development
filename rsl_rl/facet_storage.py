"""FACET rollout tensors: feedforward PPO rows, chronological estimator sequences."""
import torch


def gae(rewards, values, successor_values, done, gamma, lam):
    """Timeouts have terminal-successor V; failures have zero successor V.

    Both kinds of boundary stop the recursive advantage; neither bootstraps
    from a reset observation. Arrays are time-major [T,N].
    """
    advantage = torch.zeros_like(rewards)
    carry = torch.zeros_like(rewards[0])
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma*successor_values[t] - values[t]
        carry = delta + gamma*lam*(~done[t])*carry
        advantage[t] = carry
    return advantage, advantage+values


class FACETStorage:
    def __init__(self, steps, count, observation_dim, privileged_dim, device, phase):
        self.steps, self.count, self.phase, self.cursor = steps, count, phase, 0
        widths = dict(obs=observation_dim, priv=privileged_dim, wrench=9, features=160,
                      actions=17, means=17, stds=17)
        self.data = {k: torch.zeros(steps, count, d, device=device) for k, d in widths.items()}
        for key in ('values', 'log_probs', 'rewards', 'successor_values'):
            self.data[key] = torch.zeros(steps, count, device=device)
        for key in ('done', 'is_init', 'valid'):
            self.data[key] = torch.zeros(steps, count, dtype=torch.bool, device=device)
        self.initial_hidden = torch.zeros(count, 128, device=device)

    @torch.no_grad()
    def add(self, **row):
        if self.cursor >= self.steps:
            raise RuntimeError('FACET rollout is full')
        for key, destination in self.data.items():
            destination[self.cursor].copy_(row[key].reshape_as(destination[self.cursor]))
        self.cursor += 1

    def finish(self, gamma, lam):
        if self.cursor != self.steps:
            raise RuntimeError('FACET PPO requires a complete rollout')
        d = self.data
        advantages, d['returns'] = gae(d['rewards'], d['values'], d['successor_values'], d['done'], gamma, lam)
        # Match PILOT's full-rollout sample-std normalization, not per-minibatch.
        d['advantages'] = (advantages-advantages.mean())/(advantages.std()+1e-8)

    def minibatches(self, count):
        flat = {k: v.flatten(0, 1) for k, v in self.data.items()}
        permutation = torch.randperm(self.steps*self.count, device=self.initial_hidden.device)
        for ids in torch.tensor_split(permutation, min(count, len(permutation))):
            yield {k: v[ids] for k, v in flat.items()}

    def sequences(self, count):
        # Shuffle robots, never time. Each sequence keeps its collection-time
        # initial raw GRU state and all mid-sequence episode reset masks.
        permutation = torch.randperm(self.count, device=self.initial_hidden.device)
        for ids in torch.tensor_split(permutation, min(count, self.count)):
            yield {k: v[:, ids].transpose(0, 1) for k, v in self.data.items()}, self.initial_hidden[ids]
