"""Detached iteration-frozen scalar recovery budget (not a gradient-norm cap)."""
from copy import deepcopy
import torch

DEFAULTS = dict(recovery_slack_huber_enabled=False, recovery_slack_huber_delta=1.,
    recovery_loss_budget_enabled=False, recovery_loss_budget_ratio=.5,
    recovery_loss_budget_ema_alpha=.01, recovery_loss_budget_reference_floor=1e-6)


def legacy_recovery_contract(saved, current):
    """Only absent new options may adopt explicit settings on a legacy resume."""
    result = deepcopy(saved)
    if not isinstance(saved, dict) or not isinstance(current, dict):
        return result
    if 'qp_update_mode' in saved:
        for key in DEFAULTS:
            if key in current:
                result.setdefault(key, current[key])
    for key in saved.keys() & current.keys():
        if isinstance(saved[key], dict):
            result[key] = legacy_recovery_contract(saved[key], current[key])
    return result


class RecoveryLossBudget:
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.ema = torch.zeros((), device=device)
        self.initialized = torch.zeros((), device=device, dtype=torch.bool)
        self.last_updated_iteration = torch.full((), -1, device=device, dtype=torch.long)
        self.last_completed = -1

    @torch.no_grad()
    def begin(self, iteration):
        self.iteration = int(iteration)
        self.reference = self.ema.clone()
        self.ready = self.initialized.clone()
        self.total = torch.zeros_like(self.ema)
        self.count = torch.zeros_like(self.ema)

    def apply(self, other, recovery, samples):
        with torch.no_grad():
            value = other.detach()
            valid = torch.isfinite(value) & (value >= 0) & (samples > 0)
            self.reference.copy_(torch.where(~self.ready & valid, value, self.reference))
            self.ready.logical_or_(valid)
            self.total.add_(torch.where(valid, value, 0.) * samples)
            self.count.add_(valid.to(value.dtype) * samples)
            budget = self.cfg.recovery_loss_budget_ratio * self.reference.clamp_min(
                self.cfg.recovery_loss_budget_reference_floor)
            scale = (budget / (recovery.detach() + 1e-12)).clamp(max=1.)
        applied = scale * recovery
        return other + applied, dict(uncapped=recovery.detach(), applied=applied.detach(),
                                    budget=budget, scale=scale)

    @torch.no_grad()
    def finish(self):
        if self.iteration <= self.last_completed:
            return
        mean = self.total / self.count.clamp_min(1)
        valid = (self.count > 0) & torch.isfinite(mean)
        update = torch.where(self.initialized,
            self.ema.lerp(mean, self.cfg.recovery_loss_budget_ema_alpha), mean)
        self.ema.copy_(torch.where(valid, update, self.ema))
        self.initialized.logical_or_(valid)
        self.last_updated_iteration.copy_(torch.where(valid, self.iteration, self.last_updated_iteration))
        self.last_completed = self.iteration

    def state_dict(self):
        return dict(ema=self.ema.clone(), initialized=self.initialized.clone(),
            last_updated_iteration=self.last_updated_iteration.clone(), last_completed=self.last_completed)

    def load_state_dict(self, state):
        self.validate_state_dict(state)
        for key in ('ema', 'initialized', 'last_updated_iteration'):
            getattr(self, key).copy_(state[key])
        self.last_completed = int(state['last_completed'])

    @staticmethod
    def validate_state_dict(state):
        for key in ('ema', 'initialized', 'last_updated_iteration'):
            if not isinstance(state.get(key), torch.Tensor) or state[key].shape != torch.Size([]):
                raise ValueError('Invalid recovery budget state: '+key)
        if not torch.isfinite(state['ema']) or state['ema'] < 0:
            raise ValueError('Invalid recovery budget EMA')
        if not isinstance(state.get('last_completed'), int):
            raise ValueError('Invalid recovery budget iteration')
