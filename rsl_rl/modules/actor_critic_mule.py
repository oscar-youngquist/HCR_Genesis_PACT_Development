"""Standalone MULE learning core; no PPO loop, simulator, or phase scheduler.

Future update order: collect one detached context snapshot + both raw samples;
finish BOTH PPO updates using that snapshot; then update CENet on matched
auxiliary transitions. Never resample/re-encode context for PPO likelihoods.
Phase switches occur between complete rollout/update iterations only.
"""
from dataclasses import dataclass
from itertools import chain
import torch
from torch import nn
from torch.distributions import Normal
from legged_gym.envs.go2.go2_mule.go2_mule_config import Go2MULECfgPPO, PHASES
from legged_gym.envs.go2.go2_mule.mule_schema import MULECommand


def mlp(widths):
    layers = []
    for i, (a, b) in enumerate(zip(widths, widths[1:])):
        layers.append(nn.Linear(a, b))
        if i < len(widths)-2: layers.append(nn.ELU())
    return nn.Sequential(*layers)


def shape(x, n, width, name):
    if x.shape != (n, width): raise ValueError(f'{name} must have shape [{n},{width}]')


@dataclass(frozen=True)
class RolloutContext:
    velocity_b: torch.Tensor  # physical m/s
    latent: torch.Tensor

    def snapshot(self):
        return RolloutContext(self.velocity_b.detach().clone(), self.latent.detach().clone())

    def index(self, ids):
        return RolloutContext(self.velocity_b[ids], self.latent[ids])


@dataclass(frozen=True)
class ContextEstimate:
    velocity_b: torch.Tensor
    latent: torch.Tensor
    mu: torch.Tensor
    logvar: torch.Tensor

    def snapshot(self): return RolloutContext(self.velocity_b, self.latent).snapshot()


@dataclass(frozen=True)
class AuxiliaryLoss:
    total: torch.Tensor
    velocity_mse: torch.Tensor
    reconstruction_mse: torch.Tensor
    latent_kl: torch.Tensor
    valid_count: int
    nonfinite_count: torch.Tensor

    @property
    def should_step(self): return self.valid_count > 0


class MULEContextEstimator(nn.Module):
    """3-D deterministic velocity + 16-D Gaussian latent. Decoder uses [v,z].

    [v,z] decoder input follows the combined CENet connection in MULE Fig.2;
    the exact split is not specified. DreamWaQ Fig.2 draws a latent-only decoder
    (and 48 outputs). We explicitly choose [scaled v,z] -> MULE's 45-D next obs.
    This interpretation, beta=1, bounded log variance, and PPO detachment are
    recorded assumptions, not claims about unavailable author training code.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder = nn.Sequential(*mlp((config.schema.history_dim,128,64)), nn.ELU())
        self.velocity_head = nn.Linear(64, 3)
        self.mu_head = nn.Linear(64, config.latent_dim)
        self.logvar_head = nn.Linear(64, config.latent_dim)
        self.decoder = mlp((3+config.latent_dim,64,128,45))

    def forward(self, history, *, deterministic=False, noise=None):
        shape(history, history.shape[0], self.config.schema.history_dim, 'history')
        features = self.encoder(history.detach())
        velocity, mu = self.velocity_head(features), self.mu_head(features)
        logvar = self.logvar_head(features).clamp(*self.config.logvar_bounds)
        if deterministic:
            if noise is not None: raise ValueError('Deterministic context does not accept noise')
            z = mu
        else:
            if noise is None: noise = torch.randn_like(mu)
            shape(noise, mu.shape[0], mu.shape[1], 'latent noise')
            z = mu + torch.exp(.5*logvar) * noise.detach()
        return ContextEstimate(velocity, z, mu, logvar)

    def features(self, estimate):
        return torch.cat((estimate.velocity_b*self.config.schema.linear_velocity_scale, estimate.latent), -1)

    def losses(self, batch, *, noise=None):
        """MSE means valid rows AND features; KL sums latent dims then row mean.

        Filter before network/loss arithmetic, including invalid NaN padding.
        No target/input gradients. Empty loss is finite and graph-free: caller
        MUST skip auxiliary optimizer.step() when should_step is false, avoiding
        momentum/weight decay updates. CUDA compaction may synchronize; no claim
        of a synchronization-free training path at this simulator-free stage.
        """
        n = batch.history.shape[0]
        shape(batch.history, n, self.config.schema.history_dim, 'history')
        shape(batch.velocity_b, n, 3, 'velocity label')
        shape(batch.next_observation, n, 45, 'next observation')
        finite = (torch.isfinite(batch.history).all(-1) & torch.isfinite(batch.velocity_b).all(-1)
                  & torch.isfinite(batch.next_observation).all(-1))
        eligible = batch.valid_rows()
        ids = (eligible & finite).nonzero(as_tuple=True)[0]
        nonfinite = (eligible & ~finite).sum().detach()
        if ids.numel() == 0:
            zero = batch.history.new_zeros(())
            return AuxiliaryLoss(zero, zero, zero, zero, 0, nonfinite)
        if noise is not None: shape(noise, n, self.config.latent_dim, 'auxiliary noise')
        estimate = self(batch.history[ids], noise=None if noise is None else noise[ids])
        reconstruction = self.decoder(self.features(estimate))
        velocity_loss = (estimate.velocity_b-batch.velocity_b[ids].detach()).square().mean()
        reconstruction_loss = (reconstruction-batch.next_observation[ids].detach()).square().mean()
        kl = .5*(estimate.mu.square()+estimate.logvar.exp()-1-estimate.logvar).sum(-1).mean()
        total = velocity_loss + reconstruction_loss + self.config.beta*kl
        return AuxiliaryLoss(total, velocity_loss, reconstruction_loss, kl, ids.numel(), nonfinite)


class GaussianActor(nn.Module):
    def __init__(self, width, config):
        super().__init__()
        self.mean = mlp((width,512,256,128,12))
        self.std = nn.Parameter(torch.tensor(config.init_noise_std))
        self.register_buffer('std_lower', torch.tensor(config.min_noise_std))
        self.register_buffer('std_upper', torch.tensor(config.max_noise_std))

    def forward(self, features):
        mean = self.mean(features)
        std = self.std.clamp(min=self.std_lower, max=self.std_upper).expand_as(mean)
        return Normal(mean, std, validate_args=False)


@dataclass(frozen=True)
class MULEDistributions:
    nominal: Normal
    adaptive: Normal | None  # None means inactive, NOT a Gaussian with tiny std.


@dataclass(frozen=True)
class MULESample:
    nominal_raw: torch.Tensor
    adaptive_raw: torch.Tensor
    nominal_log_prob: torch.Tensor
    adaptive_log_prob: torch.Tensor
    context: RolloutContext
    phase: str


@dataclass(frozen=True)
class MULEEvaluation:
    nominal_log_prob: torch.Tensor
    adaptive_log_prob: torch.Tensor
    nominal_entropy: torch.Tensor
    adaptive_entropy: torch.Tensor


class ActorCriticMULE(nn.Module):
    is_recurrent = False

    def __init__(self, config: Go2MULECfgPPO):
        super().__init__()
        self.config = config
        self.context_estimator = MULEContextEstimator(config)
        self.nominal_actor = GaussianActor(45+3+config.latent_dim, config)
        self.adaptive_actor = GaussianActor(config.schema.adaptive_history_dim+3+config.latent_dim, config)
        self.nominal_critic = mlp((config.schema.critic_dim,512,256,128,1))
        self.adaptive_critic = mlp((config.schema.critic_dim,512,256,128,1))
        self.set_phase(config.optimization_phase)

    def parameter_groups(self):
        """Disjoint owners for dual PPO and auxiliary optimizers.

        Keep adaptive parameters in its optimizer across phase changes; do not
        rebuild optimizers or lose moments. Do not step inactive owner.
        """
        return dict(nominal=tuple(chain(self.nominal_actor.parameters(), self.nominal_critic.parameters())),
                    adaptive=tuple(chain(self.adaptive_actor.parameters(), self.adaptive_critic.parameters())),
                    auxiliary=tuple(self.context_estimator.parameters()))

    def set_phase(self, phase):
        if phase not in PHASES: raise ValueError(f'Unknown MULE optimization phase: {phase}')
        self.phase = phase
        for p in self.parameter_groups()['adaptive']:
            p.requires_grad_(phase == 'joint_adaptation')
            if phase == 'nominal_only': p.grad = None

    def distributions(self, observations, adaptive_history, context):
        """Pure evaluation from stored context: no caches, RNG, or re-encoding."""
        n = observations.shape[0]
        shape(observations, n, 45, 'observations')
        shape(context.velocity_b, n, 3, 'stored velocity')
        shape(context.latent, n, self.config.latent_dim, 'stored latent')
        features = self.context_estimator.features(context).detach()
        nominal = self.nominal_actor(torch.cat((observations.detach(), features), -1))
        adaptive = None
        if self.phase == 'joint_adaptation':
            if adaptive_history is None: raise ValueError('Joint adaptation requires deployable force-augmented history')
            shape(adaptive_history, n, self.config.schema.adaptive_history_dim, 'adaptive history')
            adaptive = self.adaptive_actor(torch.cat((adaptive_history.detach(), features), -1))
        return MULEDistributions(nominal, adaptive)

    @torch.no_grad()
    def sample_actions(self, observations, history, adaptive_history=None, *, deterministic=False):
        context = self.context_estimator(history, deterministic=deterministic).snapshot()
        dist = self.distributions(observations, adaptive_history, context)
        nominal = dist.nominal.mean if deterministic else dist.nominal.sample()
        adaptive = torch.zeros_like(nominal)
        adaptive_log_prob = nominal.new_zeros(nominal.shape[0])
        if dist.adaptive is not None:
            adaptive = dist.adaptive.mean if deterministic else dist.adaptive.sample()
            adaptive_log_prob = dist.adaptive.log_prob(adaptive).sum(-1)
        return MULESample(nominal, adaptive, dist.nominal.log_prob(nominal).sum(-1),
                          adaptive_log_prob, context, self.phase)

    def evaluate_actions(self, observations, adaptive_history, context, nominal_raw, adaptive_raw, *, phase):
        if phase != self.phase: raise ValueError('Cannot replay a rollout across an optimization-phase change')
        if context is None: raise ValueError('PPO requires exact stored rollout context')
        n = observations.shape[0]
        shape(nominal_raw, n, 12, 'raw nominal sample'); shape(adaptive_raw, n, 12, 'raw adaptive sample')
        dist = self.distributions(observations, adaptive_history, context)
        zero = observations.new_zeros(n)
        return MULEEvaluation(dist.nominal.log_prob(nominal_raw.detach()).sum(-1),
            zero if dist.adaptive is None else dist.adaptive.log_prob(adaptive_raw.detach()).sum(-1),
            dist.nominal.entropy().sum(-1), zero if dist.adaptive is None else dist.adaptive.entropy().sum(-1))

    def values(self, privileged):
        shape(privileged, privileged.shape[0], self.config.schema.critic_dim, 'critic inputs')
        nominal = self.nominal_critic(privileged.detach()).squeeze(-1)
        adaptive = (self.adaptive_critic(privileged.detach()).squeeze(-1)
                    if self.phase == 'joint_adaptation' else torch.zeros_like(nominal))
        return nominal, adaptive

    def compose_actions(self, nominal_raw, adaptive_raw, standing_position):
        shape(nominal_raw, nominal_raw.shape[0], 12, 'raw nominal action')
        shape(adaptive_raw, nominal_raw.shape[0], 12, 'raw adaptive action')
        if self.phase == 'nominal_only' and torch.any(adaptive_raw != 0):
            raise ValueError('nominal_only requires exactly zero adaptive correction')
        if standing_position.shape not in ((12,), nominal_raw.shape): raise ValueError('Expected canonical standing pose')
        combined = nominal_raw + adaptive_raw
        clipped = combined.clamp(-self.config.clip_actions, self.config.clip_actions)
        target = standing_position.detach() + self.config.action_scale*clipped
        return MULECommand(nominal_raw, adaptive_raw, combined, clipped, target)

    @torch.no_grad()
    def project_action_std(self, owner):
        """Project the stored parameter after each active optimizer step."""
        if owner not in ('nominal', 'adaptive'): raise ValueError('Unknown actor owner')
        if owner == 'adaptive' and self.phase == 'nominal_only':
            raise ValueError('Inactive actor must not be updated')
        actor = getattr(self, owner + '_actor')
        actor.std.copy_(actor.std.clamp(actor.std_lower, actor.std_upper))

    @torch.no_grad()
    def inference(self, observations, history, adaptive_history=None):
        """Deployable only; deterministic mean latent and both action means; no RNG."""
        return self.sample_actions(observations, history, adaptive_history, deterministic=True)

    def get_extra_state(self):
        return dict(contract=self.config.contract(), phase=self.phase)

    def set_extra_state(self, state):
        self._validate_extra(state)
        self.set_phase(state['phase'])

    def _validate_extra(self, state):
        if not isinstance(state, dict) or not self.config.accepts_model_contract(state.get('contract')) or state.get('phase') not in PHASES:
            raise ValueError('Incompatible MULE checkpoint contract/phase; explicit conversion required')

    def validate_state_dict(self, state_dict):
        # Validate before mutating weights or phase; missing metadata never silently loads.
        self._validate_extra(state_dict.get('_extra_state'))
        expected = self.state_dict()
        if set(expected) != set(state_dict): raise ValueError('MULE checkpoint keys differ')
        for key, value in expected.items():
            if isinstance(value, torch.Tensor) and (not isinstance(state_dict[key], torch.Tensor) or value.shape != state_dict[key].shape):
                raise ValueError(f'MULE checkpoint shape mismatch: {key}')
            if isinstance(value, torch.Tensor):
                if value.dtype != state_dict[key].dtype or not torch.isfinite(state_dict[key]).all():
                    raise ValueError(f'MULE checkpoint dtype/nonfinite mismatch: {key}')
                if key.endswith(('.std_lower', '.std_upper')) and not torch.equal(value, state_dict[key].to(value.device)):
                    raise ValueError(f'MULE checkpoint exploration contract mismatch: {key}')

    def load_state_dict(self, state_dict, strict=True, assign=False):
        self.validate_state_dict(state_dict)
        return super().load_state_dict(state_dict, strict=strict, assign=assign)
