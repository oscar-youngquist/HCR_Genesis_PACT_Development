"""Owned, synchronized dual-policy rollouts. No simulator or delayed-PPO remapping."""
import torch


class RolloutStorageMULE:
    def __init__(self, config, device='cpu'):
        self.config, self.device = config, torch.device(device)
        self.step, self.phase, self.returns_ready = 0, None, False
        n, t, s = config.num_envs, config.num_steps_per_env, config.schema
        widths = dict(observations=(45,), critic=(s.critic_dim,), history=(s.history_dim,),
            adaptive_history=(s.adaptive_history_dim,), context_velocity=(3,), context_latent=(config.latent_dim,),
            nominal_raw=(12,), adaptive_raw=(12,), nominal_mean=(12,), adaptive_mean=(12,),
            nominal_std=(12,), adaptive_std=(12,), log_probs=(2,), values=(2,), rewards=(2,),
            terminal_values=(2,), velocity_b=(3,), next_observation=(45,),
            combined_raw=(12,), combined_clipped=(12,), issued_target=(12,), executed_target=(12,),
            applied_torque=(12,))
        self.data = {k: torch.zeros(t, n, *w, device=device) for k, w in widths.items()}
        for k in ('terminated', 'timed_out', 'terminal_valid', 'auxiliary_valid'):
            self.data[k] = torch.zeros(t, n, dtype=torch.bool, device=device)
        for k in ('episode_id', 'next_episode_id', 'timestamp', 'next_timestamp', 'executed_source_id'):
            self.data[k] = torch.zeros(t, n, dtype=torch.long, device=device)
        self.returns = torch.zeros(t, n, 2, device=device)
        self.advantages = torch.zeros_like(self.returns)
        self.raw_advantages = torch.zeros_like(self.returns)

    @property
    def full(self): return self.step == self.config.num_steps_per_env

    @torch.no_grad()
    def add(self, row, phase):
        if self.full: raise RuntimeError('Rollout is full')
        if phase not in ('nominal_only', 'joint_adaptation'): raise ValueError('Unknown rollout phase')
        if self.phase is not None and phase != self.phase: raise RuntimeError('Phase is immutable within a rollout')
        if set(row) != set(self.data): raise ValueError(f'Rollout fields differ: {set(row) ^ set(self.data)}')
        for key, target in self.data.items():
            value = row[key]
            if value.shape != target[self.step].shape or value.dtype != target.dtype:
                raise ValueError(f'Invalid rollout field {key}: {value.shape}, {value.dtype}')
        self.phase = phase
        for key, target in self.data.items(): target[self.step].copy_(row[key].detach())
        self.step += 1

    @torch.no_grad()
    def compute_returns(self, last_values):
        if not self.full or self.returns_ready: raise RuntimeError('Returns require one complete, unconsumed rollout')
        if last_values.shape != (self.config.num_envs, 2): raise ValueError('Expected both final values')
        d, c = self.data, self.config
        gae = torch.zeros_like(last_values)
        for t in reversed(range(self.step)):
            failure = d['terminated'][t]
            timeout = d['timed_out'][t] & ~failure
            done = failure | timeout
            if torch.any(timeout & ~d['terminal_valid'][t]): raise ValueError('Timeout requires terminal value')
            next_v = last_values if t == self.step-1 else d['values'][t+1]
            bootstrap = torch.where(failure[:, None], 0.,
                torch.where(timeout[:, None], d['terminal_values'][t], next_v))
            delta = d['rewards'][t] + c.gamma * bootstrap - d['values'][t]
            gae = delta + c.gamma * c.lam * (~done[:, None]) * gae
            self.raw_advantages[t].copy_(gae)
        self.returns.copy_(self.raw_advantages + d['values'])
        mean = self.raw_advantages.mean((0, 1), keepdim=True)
        std = self.raw_advantages.std((0, 1), unbiased=False, keepdim=True)
        self.advantages.copy_((self.raw_advantages - mean) / (std + 1e-8))
        self.returns_ready = True

    def batches(self, epochs=1):
        if not self.returns_ready: raise RuntimeError('Compute returns before minibatching')
        fields = dict(self.data, returns=self.returns, advantages=self.advantages)
        flat = {k: v.flatten(0, 1) for k, v in fields.items()}
        size = self.step * self.config.num_envs
        for _ in range(epochs):
            for ids in torch.tensor_split(torch.randperm(size, device=self.device), self.config.num_mini_batches):
                yield dict({k: v[ids] for k, v in flat.items()}, indices=ids, phase=self.phase)

    def clear(self):
        self.step, self.phase, self.returns_ready = 0, None, False
