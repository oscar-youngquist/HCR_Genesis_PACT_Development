"""Opt-in deterministic FF usage, in pre-QP physical Nm. No training RNG."""
import torch

from .hard_pact_latent_diagnostics import (
    deterministic_subsample, policy_distribution_without_side_effects,
)
from rsl_rl.modules.hard_pact_control import requested_torque_components


class ActionChannelUsage:
    """Bounded fixed rollout inputs, reused before/after the entire update.

    Latent mean (zero epsilon) removes latent exploration; stored boot masks
    remain fixed. These are issued mean commands, not delayed/executed torques.
    State-dependent RMS removes each joint's sample mean. Randomized controller
    gains are retained and can also contribute to physical-Nm variability.
    """
    @torch.no_grad()
    def __init__(self, actor, storage, action_func, action_clip, count):
        obs = storage.observations[:storage.step].flatten(0, 1)
        indices = deterministic_subsample(obs, count)
        self.observation = obs[indices].detach()
        def sample(value):
            return value[:storage.step].flatten(0, 1)[indices].detach()
        self.history = sample(storage.observation_history)
        self.boot_mask = sample(storage.latent_boot_mask)
        self.noise = torch.zeros_like(sample(storage.latent_noise))
        fields = storage.hard_pact_fields
        self.parameters = {k: sample(fields[k]) for k in (
            'control_kp', 'control_kd', 'control_motor_strength',
            'control_feedback_weight', 'control_feedforward_weight')}
        self.position = sample(fields['pre_q'])[:, 7:]
        self.velocity = sample(fields['pre_v'])[:, 6:]
        self.action_func, self.action_clip = action_func, action_clip
        self.before = self.evaluate(actor)

    @torch.no_grad()
    def evaluate(self, actor):
        mean = policy_distribution_without_side_effects(
            actor, self.observation, self.history, self.noise, self.boot_mask)[0]
        qd, ff = self.action_func(mean.clamp(-self.action_clip, self.action_clip))
        return requested_torque_components(qd, ff, self.position, self.velocity,
                                          self.parameters)[2]

    @torch.no_grad()
    def finish(self, actor):
        after = self.evaluate(actor)
        return action_channel_metrics(self.before, after)


@torch.no_grad()
def action_channel_metrics(before, after):
    """RMS over matched finite rows x 12 joints; unavailable groups are NaN."""
    valid = before.isfinite().all(-1) & after.isfinite().all(-1)
    before, after = before[valid], after[valid]
    values = {
        'mean_ff_torque_rms_nm': after.square().mean().sqrt(),
        'state_dependent_ff_rms_nm': (after-after.mean(0)).square().mean().sqrt(),
        'ff_mean_update_rms_nm': (after-before).square().mean().sqrt(),
        'sample_count': valid.sum(),
        'nonfinite_fraction': (~valid).float().mean(),
    }
    return {'Action_Channel_Usage/'+k: v.detach() for k,v in values.items()}
