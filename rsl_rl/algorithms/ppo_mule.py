"""Dual PPO with frozen rollout context; native context learning runs last."""
from copy import deepcopy
import math
import torch
from legged_gym.envs.go2.go2_mule.mule_schema import MULEAuxiliaryBatch
from rsl_rl.modules.actor_critic_mule import RolloutContext
from rsl_rl.storage.rollout_storage_mule import RolloutStorageMULE
from rsl_rl.utils.adaptive_entropy import performance_gap_entropy


class PPOMULE:
    def __init__(self, model, config, device='cpu'):
        self.model, self.config = model.to(device), config
        self.device = torch.device(device)
        self.groups = model.parameter_groups()
        self.optimizers = {k: torch.optim.Adam(params, lr=config.auxiliary_learning_rate
            if k == 'auxiliary' else config.learning_rate) for k, params in self.groups.items()}
        self.schedulers = {k: dict(learning_rate=config.learning_rate, entropy_coef=config.entropy_coef,
            optimizer_steps=0, entropy_updates=0) for k in ('nominal', 'adaptive')}
        self.storage = RolloutStorageMULE(config, device)
        self.pending = None
        self.iteration = 0

    @property
    def active_owners(self):
        return ('nominal', 'adaptive') if self.model.phase == 'joint_adaptation' else ('nominal',)

    def require_boundary(self):
        if self.pending is not None or self.storage.step:
            raise RuntimeError('Operation requires a completed update; pending/unconsumed rollout exists')

    def transition_to_joint_adaptation(self):
        self.require_boundary()
        self.model.set_phase('joint_adaptation')  # Idempotent, no copy/reinitialization.

    @torch.no_grad()
    def act(self, observation, history, adaptive_history, standing_position):
        if self.pending is not None or self.storage.full: raise RuntimeError('Consume transition/update first')
        o = observation
        sample = self.model.sample_actions(o.actor, history, adaptive_history)
        dist = self.model.distributions(o.actor, adaptive_history, sample.context)
        command = self.model.compose_actions(sample.nominal_raw, sample.adaptive_raw, standing_position)
        zeros = torch.zeros_like(sample.nominal_raw)
        row = dict(observations=o.actor, critic=o.critic, history=history, adaptive_history=adaptive_history,
            context_velocity=sample.context.velocity_b, context_latent=sample.context.latent,
            nominal_raw=sample.nominal_raw, adaptive_raw=sample.adaptive_raw,
            nominal_mean=dist.nominal.mean, nominal_std=dist.nominal.stddev,
            adaptive_mean=zeros if dist.adaptive is None else dist.adaptive.mean,
            adaptive_std=zeros if dist.adaptive is None else dist.adaptive.stddev,
            log_probs=torch.stack((sample.nominal_log_prob, sample.adaptive_log_prob), -1),
            values=torch.stack(self.model.values(o.critic), -1), velocity_b=o.velocity_b,
            episode_id=o.episode_id, timestamp=torch.full_like(o.episode_id, o.timestamp),
            combined_raw=command.combined_raw, combined_clipped=command.combined_clipped,
            issued_target=command.position_target)
        self.pending = (sample.phase, {k: v.detach().clone() for k, v in row.items()})
        return command

    def record_outcome(self, **fields):
        if self.pending is None: raise RuntimeError('No issued action')
        phase, row = self.pending
        if phase != self.model.phase: raise RuntimeError('Phase changed during collection')
        if set(fields) & set(row): raise ValueError('Outcome cannot overwrite issued-action fields')
        self.storage.add(dict(row, **fields), phase)
        self.pending = None

    @torch.no_grad()
    def compute_returns(self, critic):
        if self.pending is not None: raise RuntimeError('Pending transition')
        self.storage.compute_returns(torch.stack(self.model.values(critic), -1))

    def policy_objective(self, batch, owner):
        if owner not in self.active_owners or batch['phase'] != self.model.phase:
            raise RuntimeError('Inactive/mismatched PPO phase')
        i = 0 if owner == 'nominal' else 1
        context = RolloutContext(batch['context_velocity'], batch['context_latent'])
        dist = getattr(self.model.distributions(batch['observations'], batch['adaptive_history'], context), owner)
        log_prob = dist.log_prob(batch[owner + '_raw']).sum(-1)
        log_ratio = log_prob - batch['log_probs'][:, i]
        ratio = log_ratio.exp()
        adv = batch['advantages'][:, i]
        surrogate = -torch.minimum(ratio * adv, ratio.clamp(1-self.config.clip_param, 1+self.config.clip_param) * adv).mean()
        value = getattr(self.model, owner + '_critic')(batch['critic'].detach()).squeeze(-1)
        target, old = batch['returns'][:, i], batch['values'][:, i]
        value_error = (value - target).square()
        if self.config.use_clipped_value_loss:
            clipped = old + (value-old).clamp(-self.config.clip_param, self.config.clip_param)
            value_error = torch.maximum(value_error, (clipped-target).square())
        value_loss, entropy = value_error.mean(), dist.entropy().sum(-1).mean()
        with torch.no_grad():
            old_mean, old_std = batch[owner+'_mean'], batch[owner+'_std']
            kl = (torch.log(dist.stddev/old_std) +
                (old_std.square()+(old_mean-dist.mean).square())/(2*dist.stddev.square()) - .5).sum(-1).mean()
        loss = surrogate + self.config.value_loss_coef * value_loss - self.schedulers[owner]['entropy_coef'] * entropy
        return loss, dict(surrogate=surrogate, value=value_loss, entropy=entropy, kl=kl,
                         ratio_mean=ratio.mean(), clip_fraction=((ratio-1).abs()>self.config.clip_param).float().mean())

    def _adjust_lr(self, owner, kl):
        s = self.schedulers[owner]
        if self.config.schedule == 'adaptive':
            if kl > self.config.desired_kl*2: s['learning_rate'] = max(1e-5, s['learning_rate']/1.5)
            elif 0 < kl < self.config.desired_kl/2: s['learning_rate'] = min(1e-2, s['learning_rate']*1.5)
        for group in self.optimizers[owner].param_groups: group['lr'] = s['learning_rate']

    def update_entropy(self, performance):
        keys = ('lin_vel_tracking', 'ang_vel_tracking', 'terrain_level')
        if (not self.config.use_adaptive_entropy or performance is None or
                any(k not in performance or not math.isfinite(float(performance[k])) for k in keys)): return
        c = self.config
        value = performance_gap_entropy(performance, c.adaptive_ent_bounds,
            (c.adaptive_ent_lin_threshold, c.adaptive_ent_ang_threshold, c.adaptive_ent_ter_threshold),
            c.adaptive_ent_softmax_temp)
        value = min(c.adaptive_ent_bounds[1], max(c.adaptive_ent_bounds[0], value))
        for owner in self.active_owners:
            self.schedulers[owner]['entropy_coef'] = value
            self.schedulers[owner]['entropy_updates'] += 1

    def update(self):
        if self.pending is not None or not self.storage.returns_ready: raise RuntimeError('Incomplete rollout')
        if self.storage.phase != self.model.phase: raise RuntimeError('Rollout phase changed')
        totals, counts = {}, {}
        def log(prefix, metrics, count):
            for key, value in metrics.items():
                tag = prefix + '/' + key
                totals[tag] = totals.get(tag, 0.) + float(value.detach())*count
                counts[tag] = counts.get(tag, 0) + count
        # Both separate PPO objectives finish before ANY context optimizer step.
        for batch in self.storage.batches(self.config.num_learning_epochs):
            for owner in self.active_owners:
                self.model.zero_grad(set_to_none=True)
                loss, metrics = self.policy_objective(batch, owner)
                self._adjust_lr(owner, float(metrics['kl']))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.groups[owner], self.config.max_grad_norm, error_if_nonfinite=True)
                self.optimizers[owner].step()
                self.model.project_action_std(owner)
                self.schedulers[owner]['optimizer_steps'] += 1
                log(owner, metrics, len(batch['indices']))
        aux_steps, valid_count = 0, 0
        for batch in self.storage.batches(self.config.auxiliary_epochs):
            self.model.zero_grad(set_to_none=True)
            targets = MULEAuxiliaryBatch(batch['history'], batch['velocity_b'], batch['next_observation'],
                batch['auxiliary_valid'], batch['episode_id'], batch['next_episode_id'])
            losses = self.model.context_estimator.losses(targets)
            if not losses.should_step: continue
            losses.total.backward()
            torch.nn.utils.clip_grad_norm_(self.groups['auxiliary'], self.config.max_grad_norm, error_if_nonfinite=True)
            self.optimizers['auxiliary'].step()
            aux_steps += 1; valid_count += losses.valid_count
            log('auxiliary', {k: getattr(losses, k) for k in ('total', 'velocity_mse', 'reconstruction_mse', 'latent_kl')}, losses.valid_count)
        result = {k: totals[k]/counts[k] for k in totals}
        result.update(auxiliary_steps=aux_steps, auxiliary_valid_samples=valid_count)
        if not valid_count:
            result.update({f'auxiliary/{k}': 0. for k in ('total', 'velocity_mse', 'reconstruction_mse', 'latent_kl')})
        self.iteration += 1
        self.storage.clear()
        return result

    def state_dict(self):
        self.require_boundary()
        return deepcopy(dict(version=1, model=self.model.state_dict(), config=self.config.resolved(),
            learning_contract=self.config.learning_contract(), optimizers={k: o.state_dict() for k, o in self.optimizers.items()},
            schedulers=self.schedulers, phase=self.model.phase, iteration=self.iteration))

    def validate_state_dict(self, state):
        if state.get('version') != 1 or state.get('learning_contract') != self.config.learning_contract():
            raise ValueError('Incompatible MULE learning contract (only budget/save/log overrides are supported)')
        self.model.validate_state_dict(state['model'])
        if state['phase'] != state['model']['_extra_state']['phase']: raise ValueError('Conflicting saved phases')
        if not isinstance(state['iteration'], int) or state['iteration'] < 0: raise ValueError('Invalid iteration')
        if set(state['optimizers']) != set(self.optimizers) or set(state['schedulers']) != set(self.schedulers):
            raise ValueError('Missing optimizer/scheduler state')
        for owner, optimizer in self.optimizers.items():
            # Preflight on disposable optimizer/parameter copies; live moments untouched.
            candidate = deepcopy(optimizer)
            candidate.load_state_dict(state['optimizers'][owner])
            for group in candidate.param_groups:
                if not math.isfinite(group['lr']) or group['lr'] <= 0: raise ValueError('Invalid saved LR')
                for p in group['params']:
                    for key, value in candidate.state.get(p, {}).items():
                        if isinstance(value, torch.Tensor):
                            if not torch.isfinite(value).all(): raise ValueError('Nonfinite optimizer state')
                            if key != 'step' and value.shape != p.shape: raise ValueError('Optimizer shape mismatch')
        for s in state['schedulers'].values():
            if (not math.isfinite(s['learning_rate']) or s['learning_rate'] <= 0 or
                not math.isfinite(s['entropy_coef']) or s['entropy_coef'] < 0 or
                any(not isinstance(s[k], int) or s[k] < 0 for k in ('optimizer_steps', 'entropy_updates'))):
                raise ValueError('Invalid saved scheduler')

    def load_state_dict(self, state):
        self.require_boundary()
        self.validate_state_dict(state)
        self.model.load_state_dict(state['model'])
        for key, optimizer in self.optimizers.items(): optimizer.load_state_dict(state['optimizers'][key])
        self.schedulers = deepcopy(state['schedulers'])
        self.iteration = state['iteration']
