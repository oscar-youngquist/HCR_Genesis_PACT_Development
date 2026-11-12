"""FACET-only PPO and estimator ownership; no PACT auxiliary machinery.

Estimator residual/targets/cadence/EMA follow active-adaptation c0d6f5a,
ppo_facet.py (MIT; FACET_LICENSE.txt). PPO protocol follows local PILOT.
"""
from copy import deepcopy
import torch
from torch import nn


class FACETPPO:
    def __init__(self, model, cfg):
        self.model, self.cfg = model, cfg
        self.phase, self.transitioned = 'teacher', False
        self.student_initialized = False  # Separate from phase: adaptation must never recopy it.
        self.learning_rate = cfg['learning_rate']
        self.current_entropy_coef = cfg['entropy_coef']
        self.ema = deepcopy(model.estimator).requires_grad_(False).eval()
        self.optimizer = torch.optim.AdamW([
            dict(params=[p for n, p in model.teacher.named_parameters() if n != 'std'], weight_decay=cfg['weight_decay']),
            dict(params=[model.teacher.std], weight_decay=0.),
            dict(params=list(model.critic.parameters()), weight_decay=cfg['weight_decay']),
            dict(params=list(model.priv_encoder.parameters())+list(model.force_encoder.parameters()), weight_decay=cfg['weight_decay']),
        ], lr=self.learning_rate)
        self.estimator_optimizer = torch.optim.Adam(model.estimator.parameters(), lr=cfg['estimator_learning_rate'])
        model.student.requires_grad_(False)
        if not isinstance(cfg['student_start_iteration'], int) or cfg['student_start_iteration'] < 0:
            raise ValueError('student_start_iteration must be a nonnegative integer')
        if not isinstance(cfg.get('adaptation_iterations', 0), int) or cfg.get('adaptation_iterations', 0) < 0:
            raise ValueError('adaptation_iterations must be a nonnegative integer')
        if cfg['estimator_teacher_interval'] < 1 or cfg['estimator_epochs'] < 1 or not 0 < cfg['estimator_ema_tau'] <= 1:
            raise ValueError('Invalid FACET estimator cadence/EMA')

    def distribution(self, batch):
        m = self.model
        if self.phase == 'teacher':
            return m.teacher(batch['obs'], m.priv_encoder(batch['priv']), m.force_encoder(batch['wrench']))
        # Detached collection-time EMA features: PPO cannot train the estimator.
        return m.student(batch['obs'], batch['features'][..., :128].detach(), batch['features'][..., 128:].detach())

    def value(self, batch):
        return self.model.critic(torch.cat((batch['wrench'], batch['obs'], batch['priv']), -1)).squeeze(-1)

    def ppo_loss(self, batch):
        dist, value = self.distribution(batch), self.value(batch)
        ratio = (dist.log_prob(batch['actions'])-batch['log_probs']).exp()
        clip = self.cfg['clip_param']
        policy = torch.maximum(-batch['advantages']*ratio,
            -batch['advantages']*ratio.clamp(1-clip, 1+clip)).mean()
        error = (value-batch['returns']).square()
        if self.cfg['use_clipped_value_loss']:
            clipped = batch['values']+(value-batch['values']).clamp(-clip, clip)
            error = torch.maximum(error, (clipped-batch['returns']).square())
        loss = policy+self.cfg['value_loss_coef']*error.mean()-self.current_entropy_coef*dist.entropy().mean()
        with torch.no_grad():
            kl = (torch.log(dist.stddev/batch['stds']+1e-5) +
                  (batch['stds'].square()+(batch['means']-dist.mean).square())/(2*dist.stddev.square())-.5).sum(-1).mean()
        return loss, dict(surrogate=policy.detach(), value=error.mean().detach(), kl=kl)

    def estimator_loss(self, batch, initial_hidden):
        # Preserve upstream masking of first observations; padding is excluded
        # from BOTH recurrence and denominator. Padding may only be trailing.
        features = self.model.estimator.mlp(batch['obs'])
        h, outputs = initial_hidden.detach(), []
        for t in range(features.shape[1]):
            reset_h = h*(~batch['is_init'][:, t, None])
            candidate = self.model.estimator.gru(features[:, t], reset_h)
            h = torch.where(batch['valid'][:, t, None], candidate, h)
            outputs.append(self.model.estimator.norm(h)+features[:, t])
        predicted = self.model.estimator.out(torch.stack(outputs, 1))
        with torch.no_grad():
            target = torch.cat((self.model.priv_encoder(batch['priv']), self.model.force_encoder(batch['wrench'])), -1)
        mask = (batch['valid'] & ~batch['is_init']).float()
        denominator = batch['valid'].sum().clamp_min(1)
        error = (predicted[..., :160]-target).square()
        priv = (error[..., :128].mean(-1)*mask).sum()/denominator
        force = (error[..., 128:].mean(-1)*mask).sum()/denominator
        # The independent last 9 reconstruction rows have no target/loss.
        return priv+force, dict(estimator_priv=priv.detach(), estimator_force=force.detach())

    def update(self, storage, iteration):
        if storage.phase != self.phase:
            raise RuntimeError('Cannot reuse teacher data for student PPO (or vice versa)')
        metrics, count = {}, 0
        for _ in range(0 if self.phase == 'adaptation' else self.cfg['num_learning_epochs']):
            for batch in storage.minibatches(self.cfg['num_mini_batches']):
                loss, values = self.ppo_loss(batch)
                if self.cfg['schedule'] == 'adaptive' and self.cfg['desired_kl'] is not None:
                    kl = float(values['kl'])
                    if kl > 2*self.cfg['desired_kl']:
                        self.learning_rate = max(1e-5, self.learning_rate/1.5)
                    elif 0 < kl < self.cfg['desired_kl']/2:
                        self.learning_rate = min(1e-2, self.learning_rate*1.5)
                    for group in self.optimizer.param_groups:
                        group['lr'] = self.learning_rate
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_([p for g in self.optimizer.param_groups for p in g['params']], self.cfg['max_grad_norm'])
                self.optimizer.step()
                for key, value in values.items():
                    metrics[key] = metrics.get(key, 0.)+value
                count += 1
        metrics = {k: float(v/count) for k, v in metrics.items()}
        # Upstream update index starts at zero: supervise updates 0,2,4,... .
        if self.phase != 'teacher' or iteration % self.cfg['estimator_teacher_interval'] == 0:
            estimator_metrics, batches = {}, 0
            for _ in range(self.cfg['estimator_epochs']):
                for batch, h in storage.sequences(self.cfg['num_mini_batches']):
                    self.estimator_optimizer.zero_grad(set_to_none=True)
                    loss, values = self.estimator_loss(batch, h)
                    loss.backward()
                    self.estimator_optimizer.step()
                    for key, value in values.items():
                        estimator_metrics[key] = estimator_metrics.get(key, 0.)+value
                    batches += 1
            with torch.no_grad():
                for target, source in zip(self.ema.parameters(), self.model.estimator.parameters()):
                    target.lerp_(source, self.cfg['estimator_ema_tau'])
            metrics.update({k: float(v/batches) for k, v in estimator_metrics.items()})
        return metrics

    @torch.no_grad()
    def switch_to_student(self, copy_actor=True):
        if self.transitioned:
            if self.phase == 'adaptation':
                self.phase = 'student'
                self.model.student.requires_grad_(True)
                self.model.critic.requires_grad_(True)
                return True
            return False
        m = self.model
        if copy_actor:
            m.student.load_state_dict(m.teacher.state_dict())
        # Keep the SAME optimizer, critic moments and LR. Actor parameter shapes
        # match exactly, so Adam first/second moments and step counters transfer.
        mapping = dict(zip(m.teacher.parameters(), m.student.parameters()))
        for group in self.optimizer.param_groups[:2]:
            old = group['params']
            group['params'] = [mapping[p] for p in old]
            for p in old:
                if p in self.optimizer.state:
                    state = self.optimizer.state.pop(p)
                    if copy_actor:
                        self.optimizer.state[mapping[p]] = deepcopy(state)
        for module in (m.teacher, m.priv_encoder, m.force_encoder):
            module.requires_grad_(False)
            for p in module.parameters():
                p.grad = None
        m.student.requires_grad_(True)
        self.phase, self.transitioned = 'student', True
        self.student_initialized = True
        return True

    def switch_to_adaptation(self):
        if self.phase != 'teacher':
            return False
        self.switch_to_student()
        self.phase = 'adaptation'
        self.model.student.requires_grad_(False)
        self.model.critic.requires_grad_(False)
        return True

    def update_entropy(self, linear, angular, terrain):
        c = self.cfg
        if not c['use_adaptive_entropy']:
            return
        gaps = [max(0., threshold-value)/threshold if threshold > 0 else 0.
                for value, threshold in ((linear, c['adaptive_ent_lin_threshold']),
                    (angular, c['adaptive_ent_ang_threshold']), (terrain, c['adaptive_ent_ter_threshold']))]
        gaps = next(self.model.parameters()).new_tensor(gaps)
        low, high = c['adaptive_ent_bounds']
        self.current_entropy_coef = low + float(
            (torch.softmax(gaps/c['adaptive_ent_softmax_temp'], 0)*gaps).sum())*(high-low)

    def state_dict(self):
        return dict(model=self.model.state_dict(), ema=self.ema.state_dict(), optimizer=self.optimizer.state_dict(),
            estimator_optimizer=self.estimator_optimizer.state_dict(), phase=self.phase, transitioned=self.transitioned,
            student_initialized=self.student_initialized,
            learning_rate=self.learning_rate, entropy_coef=self.current_entropy_coef)

    def load_state_dict(self, state, load_optimizer=True):
        if state['phase'] not in ('teacher', 'adaptation', 'student') or (state['phase'] != 'teacher') != state['transitioned']:
            raise ValueError('Inconsistent FACET checkpoint phase')
        if state.get('student_initialized', state['transitioned']) != state['transitioned']:
            raise ValueError('Inconsistent student initialization status')
        self.model.load_state_dict(state['model'], strict=True)
        self.ema.load_state_dict(state['ema'], strict=True)
        if state['phase'] != 'teacher':
            self.switch_to_student(copy_actor=False)  # Never overwrite a trained student.
            if state['phase'] == 'adaptation':
                self.phase = 'adaptation'
                self.model.student.requires_grad_(False)
                self.model.critic.requires_grad_(False)
        elif self.transitioned:
            raise ValueError('Load a teacher checkpoint into a fresh FACET runner')
        if load_optimizer:
            self.optimizer.load_state_dict(state['optimizer'])
            self.estimator_optimizer.load_state_dict(state['estimator_optimizer'])
        self.learning_rate = state['learning_rate']
        self.current_entropy_coef = state['entropy_coef']
        for group in self.optimizer.param_groups:
            group['lr'] = self.learning_rate
