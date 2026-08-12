import torch
import torch.nn as nn
from torch.distributions import Normal
from .module_utils import init_weights

class AC_Args:

    adaptation_module_encoder_hidden_dims = [512, 256, 128]

    adaptation_module_decoder_hidden_dims = [128, 64]

    adaptation_module_decoder_recon_hidden_dims = [128, 256, 512]
    
    adaptation_labels = [
        "base_velocity_loss",
        "gripper_pos_loss",
        "force_ee_loss",
        "force_base_loss",
        "foot_contact_loss",
        "foot_height_loss",
    ]
    adaptation_dims = [3, 3, 3, 3, 4, 4]
    adaptation_weights = [0.2, 0.2, 1.0, 1.0, 1.0, 1.0]


class ActorCriticUniFP(nn.Module):
    is_recurrent = False

    def __init__(self,  num_obs,
                        num_privileged_obs,
                        num_obs_pred,
                        num_single_obs,
                        num_actions,
                        num_privileged_obs_single=None,
                        actor_hidden_dims=[256, 256, 256],
                        critic_hidden_dims=[256, 256, 256],
                        activation='elu',
                        init_noise_std=1.0,
                        min_noise_std=1.0e-3,
                        max_noise_std=float("inf"),
                        enable_additional_diagnostics=True,
                        **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super().__init__()
        self.enable_additional_diagnostics = bool(enable_additional_diagnostics)
        
        self.adaptation_labels = AC_Args.adaptation_labels
        self.adaptation_dims = AC_Args.adaptation_dims
        self.adaptation_weights = AC_Args.adaptation_weights

        if len(self.adaptation_weights) < len(self.adaptation_labels):
            # pad
            self.adaptation_weights += [1.0] * (len(self.adaptation_labels) - len(self.adaptation_weights))

        self.num_obs = num_obs
        self.num_privileged_obs = num_privileged_obs
        self.num_obs_pred = num_obs_pred
        # self.num_latent_dim = int(num_obs / num_single_obs) * 2
        self.num_latent_dim = 64
        self.num_obs_now = num_single_obs
        self.num_privileged_obs_single = num_privileged_obs_single or num_privileged_obs

        activation = get_activation(activation)

        # Shared history trunk. As in PACT, the trunk feeds independent mean
        # and log-variance projections instead of one concatenated output.
        adaptation_module_encoder_layers = []
        adaptation_module_encoder_layers.append(nn.Linear(self.num_obs, AC_Args.adaptation_module_encoder_hidden_dims[0]))
        adaptation_module_encoder_layers.append(activation)
        for l in range(len(AC_Args.adaptation_module_encoder_hidden_dims) - 1):
            adaptation_module_encoder_layers.append(
                nn.Linear(AC_Args.adaptation_module_encoder_hidden_dims[l],
                          AC_Args.adaptation_module_encoder_hidden_dims[l + 1]))
            adaptation_module_encoder_layers.append(activation)
        self.adaptation_encoder_module = nn.Sequential(*adaptation_module_encoder_layers)
        encoder_output_dim = AC_Args.adaptation_module_encoder_hidden_dims[-1]
        self.adaptation_mean_module = nn.Linear(encoder_output_dim, self.num_latent_dim)
        self.adaptation_logvar_module = nn.Sequential(
            nn.Linear(encoder_output_dim, self.num_latent_dim),
            nn.Hardtanh(min_val=-5.0, max_val=5.0),
        )

        # Initialize the encoder layers
        self.adaptation_encoder_module.apply(init_weights)
        self.adaptation_logvar_module.apply(init_weights)
        torch.nn.init.xavier_uniform_(self.adaptation_mean_module.weight)
        if self.adaptation_mean_module.bias is not None:
            torch.nn.init.zeros_(self.adaptation_mean_module.bias)



        adaptation_module_decoder_layers = []
        adaptation_module_decoder_layers.append(nn.Linear(self.num_latent_dim, AC_Args.adaptation_module_decoder_hidden_dims[0]))
        adaptation_module_decoder_layers.append(activation)
        for l in range(len(AC_Args.adaptation_module_decoder_hidden_dims)):
            if l == len(AC_Args.adaptation_module_decoder_hidden_dims) - 1:
                adaptation_module_decoder_layers.append(
                    nn.Linear(AC_Args.adaptation_module_decoder_hidden_dims[l], self.num_obs_pred))
            else:
                adaptation_module_decoder_layers.append(
                    nn.Linear(AC_Args.adaptation_module_decoder_hidden_dims[l],
                              AC_Args.adaptation_module_decoder_hidden_dims[l + 1]))
                adaptation_module_decoder_layers.append(activation)
        self.adaptation_decoder_module = nn.Sequential(*adaptation_module_decoder_layers)

        # Initialize the state-estimator decoder
        self.adaptation_decoder_module.apply(init_weights)

        # The same sampled z predicts the configured next-frame reconstruction
        # target. B1Z1 omits the terrain-height tail, while the full privileged
        # frame remains available to the critic. This decoder is training-only.
        privileged_decoder_layers = []
        privileged_decoder_layers.append(nn.Linear(self.num_latent_dim, AC_Args.adaptation_module_decoder_recon_hidden_dims[0]))
        privileged_decoder_layers.append(activation)
        for l in range(len(AC_Args.adaptation_module_decoder_recon_hidden_dims)):
            if l == len(AC_Args.adaptation_module_decoder_recon_hidden_dims) - 1:
                privileged_decoder_layers.append(
                    nn.Linear(AC_Args.adaptation_module_decoder_recon_hidden_dims[l], 
                              self.num_privileged_obs_single))
            else:
                privileged_decoder_layers.append(
                    nn.Linear(AC_Args.adaptation_module_decoder_recon_hidden_dims[l], 
                              AC_Args.adaptation_module_decoder_recon_hidden_dims[l+1]))
                privileged_decoder_layers.append(activation)
        self.privileged_decoder_module = nn.Sequential(*privileged_decoder_layers)

        # Initialize the privileged reconstruction decoder layers
        self.privileged_decoder_module.apply(init_weights)


        # Policy
        actor_layers = []
        # Actor input = current observation + sampled z + explicit estimates.
        actor_input_dim = self.num_obs_now + self.num_latent_dim + self.num_obs_pred
        actor_layers.append(nn.Linear(actor_input_dim, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        self.actor_body = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(self.num_privileged_obs, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
        self.critic_body = nn.Sequential(*critic_layers)

        if self.enable_additional_diagnostics:
            print(f"Adaptation Encoder Module: {self.adaptation_encoder_module}")
            print(f"Adaptation Mean Module: {self.adaptation_mean_module}")
            print(f"Adaptation Logvar Module: {self.adaptation_logvar_module}")
            print(f"Adaptation Decoder Module: {self.adaptation_decoder_module}")
            print(f"Privileged Decoder Module: {self.privileged_decoder_module}")
            print(f"Actor MLP: {self.actor_body}")
            print(f"Critic MLP: {self.critic_body}")

        # Action noise. Each setting accepts either one scalar shared by every
        # action or a flat sequence with one value per action dimension.
        initial_std = self._std_config_tensor(init_noise_std, num_actions, "init_noise_std")
        minimum_std = self._std_config_tensor(min_noise_std, num_actions, "min_noise_std")
        maximum_std = self._std_config_tensor(max_noise_std, num_actions, "max_noise_std")
        if torch.any(minimum_std <= 0.0):
            raise ValueError("min_noise_std values must all be greater than zero")
        if torch.any(maximum_std < minimum_std):
            raise ValueError("max_noise_std must be greater than or equal to min_noise_std in every action dimension")
        self.std = nn.Parameter(initial_std)
        # Non-persistent buffers follow model device transfers without adding
        # keys that would break loading checkpoints created before std bounds.
        self.register_buffer("_std_clip_lwr", minimum_std, persistent=False)
        self.register_buffer("_std_clip_upr", maximum_std, persistent=False)
        self.distribution = None
        self._diagnostic_history = None
        self._diagnostic_sums = {}
        self._diagnostic_counts = {}
        self._last_latent_mean = None
        self._last_latent_logvar = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
        enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    @staticmethod
    def _std_config_tensor(value, num_actions, name):
        """Expand a scalar or validate a per-action standard-deviation list."""
        tensor = torch.as_tensor(value, dtype=torch.float)
        if tensor.ndim == 0:
            return tensor.repeat(num_actions)
        if tensor.ndim != 1 or tensor.numel() != num_actions:
            raise ValueError(
                f"{name} must be a scalar or a flat list of {num_actions} values; "
                f"got shape {tuple(tensor.shape)}"
            )
        return tensor.clone()

    def reset(self, dones=None):
        if dones is None or self._diagnostic_history is None:
            return
        done_mask = dones.reshape(-1).bool()
        self._diagnostic_history["length"][done_mask] = 0
        for name in ("mean", "sample", "mean_delta", "sample_delta"):
            self._diagnostic_history[name][done_mask] = 0.0

    def forward(self):
        raise NotImplementedError

    @torch.no_grad
    @torch.jit.ignore
    def _clip_std(self):
        self.std.copy_(torch.maximum(torch.minimum(self.std, self._std_clip_upr), self._std_clip_lwr))

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def encode_context(self, observations, sample=True):
        """Return VAE statistics and a reparameterized latent sample."""
        encoded = self.adaptation_encoder_module(observations)
        mean = self.adaptation_mean_module(encoded)
        # PACT bounds log-variance in-module to keep exp(0.5 * logvar)
        # numerically stable while retaining separate distribution heads.
        logvar = self.adaptation_logvar_module(encoded)
        z = mean + torch.exp(0.5 * logvar) * torch.randn_like(mean) if sample else mean
        return mean, logvar, z

    def actor_estimates(self, explicit_prediction):
        """Convert contact logits while preserving all continuous estimates."""
        return torch.cat((
            explicit_prediction[:, :12],
            torch.sigmoid(explicit_prediction[:, 12:16]),
            explicit_prediction[:, 16:20],
        ), dim=-1)

    def update_distribution(self, observations):
        latent_mean, latent_logvar, latent = self.encode_context(observations, sample=False)
        if self.enable_additional_diagnostics:
            self._last_latent_mean = latent_mean.detach()
            self._last_latent_logvar = latent_logvar.detach()
        actor_inputs = [
            observations[:, -self.num_obs_now:],
            latent,
            self.actor_estimates(self.adaptation_decoder_module(latent)),
        ]
        mean = self.actor_body(torch.cat(actor_inputs, dim=-1))
        self._clip_std()
        self.distribution = Normal(mean, mean*0. + self.std)

    def begin_rollout_diagnostics(self):
        """Reset scalar accumulators while retaining cross-rollout action history."""
        self._diagnostic_sums = {}
        self._diagnostic_counts = {}

    def _diagnostic_add(self, name, value, count):
        value = value.detach()
        count = count.detach() if isinstance(count, torch.Tensor) else value.new_tensor(count)
        self._diagnostic_sums[name] = self._diagnostic_sums.get(
            name, value.new_zeros(())
        ) + value
        self._diagnostic_counts[name] = self._diagnostic_counts.get(
            name, value.new_zeros(())
        ) + count

    @torch.no_grad()
    def record_rollout_diagnostics(self, sampled_action, action_clip):
        """Accumulate detached action/latent statistics without resampling."""
        action_mean = self.action_mean.detach()
        sampled_action = sampled_action.detach()
        if self._diagnostic_history is None or self._diagnostic_history["mean"].shape != action_mean.shape:
            self._diagnostic_history = {
                "mean": torch.zeros_like(action_mean),
                "sample": torch.zeros_like(sampled_action),
                "mean_delta": torch.zeros_like(action_mean),
                "sample_delta": torch.zeros_like(sampled_action),
                "length": torch.zeros(action_mean.shape[0], dtype=torch.long, device=action_mean.device),
            }

        history = self._diagnostic_history
        numel = action_mean.numel()
        self._diagnostic_add("action_mean_sq", action_mean.square().sum(), numel)
        self._diagnostic_add("action_sample_sq", sampled_action.square().sum(), numel)
        self._diagnostic_add(
            "action_noise_sq", (sampled_action - action_mean).square().sum(), numel
        )
        self._diagnostic_sums["action_mean_abs_max"] = torch.maximum(
            self._diagnostic_sums.get("action_mean_abs_max", action_mean.new_zeros(())),
            action_mean.abs().max(),
        )
        clipped = sampled_action.abs() > float(action_clip)
        self._diagnostic_add("action_clipped", clipped.sum(), clipped.numel())

        first_valid = history["length"] >= 1
        if first_valid.any():
            mean_delta = action_mean - history["mean"]
            sample_delta = sampled_action - history["sample"]
            mask = first_valid.unsqueeze(1)
            count = mask.sum() * action_mean.shape[1]
            self._diagnostic_add("action_delta_mean_sq", (mean_delta.square() * mask).sum(), count)
            self._diagnostic_add("action_delta_sample_sq", (sample_delta.square() * mask).sum(), count)

            second_valid = history["length"] >= 2
            if second_valid.any():
                second_mask = second_valid.unsqueeze(1)
                second_count = second_mask.sum() * action_mean.shape[1]
                self._diagnostic_add(
                    "action_second_mean_sq",
                    ((mean_delta - history["mean_delta"]).square() * second_mask).sum(),
                    second_count,
                )
                self._diagnostic_add(
                    "action_second_sample_sq",
                    ((sample_delta - history["sample_delta"]).square() * second_mask).sum(),
                    second_count,
                )
            history["mean_delta"].copy_(mean_delta)
            history["sample_delta"].copy_(sample_delta)

        history["mean"].copy_(action_mean)
        history["sample"].copy_(sampled_action)
        history["length"].add_(1).clamp_(max=2)

        if self._last_latent_mean is not None:
            latent_std = torch.exp(0.5 * self._last_latent_logvar)
            self._diagnostic_add(
                "latent_mean_sq", self._last_latent_mean.square().sum(),
                self._last_latent_mean.numel(),
            )
            self._diagnostic_add("latent_std", latent_std.sum(), latent_std.numel())
            self._diagnostic_sums["latent_std_max"] = torch.maximum(
                self._diagnostic_sums.get("latent_std_max", latent_std.new_zeros(())),
                latent_std.max(),
            )

    @torch.no_grad()
    def get_rollout_diagnostics(self):
        """Return finite policy scalars accumulated over the latest rollout."""
        def mean(name):
            denominator = self._diagnostic_counts.get(name)
            if denominator is None or denominator.item() <= 0:
                return 0.0
            return (self._diagnostic_sums[name] / denominator.clamp_min(1.0)).item()

        def rms(name):
            return mean(name) ** 0.5

        std = self.std.detach()
        lower = self._std_clip_lwr
        upper = self._std_clip_upr
        finite_upper = torch.isfinite(upper)
        upper_fraction = (
            ((std >= upper - 1.0e-6) & finite_upper).float().mean().item()
            if finite_upper.any() else 0.0
        )
        return {
            "Policy/action_mean_rms": rms("action_mean_sq"),
            "Policy/action_mean_abs_max": self._diagnostic_sums.get(
                "action_mean_abs_max", std.new_zeros(())
            ).item(),
            "Policy/action_sample_rms": rms("action_sample_sq"),
            "Policy/action_noise_rms": rms("action_noise_sq"),
            "Policy/action_delta_mean_rms": rms("action_delta_mean_sq"),
            "Policy/action_delta_sample_rms": rms("action_delta_sample_sq"),
            "Policy/action_second_difference_mean_rms": rms("action_second_mean_sq"),
            "Policy/action_second_difference_sample_rms": rms("action_second_sample_sq"),
            "Policy/action_clip_fraction": mean("action_clipped"),
            "Policy/latent_mean_rms": rms("latent_mean_sq"),
            "Policy/latent_posterior_std_mean": mean("latent_std"),
            "Policy/latent_posterior_std_max": self._diagnostic_sums.get(
                "latent_std_max", std.new_zeros(())
            ).item(),
            "Policy/std_mean": std.mean().item(),
            "Policy/std_min": std.min().item(),
            "Policy/std_max": std.max().item(),
            "Policy/std_at_lower_bound_fraction": (std <= lower + 1.0e-6).float().mean().item(),
            "Policy/std_at_upper_bound_fraction": upper_fraction,
        }

    def _actor_mean_from_latent(self, observations, latent):
        explicit = self.actor_estimates(self.adaptation_decoder_module(latent))
        actor_input = torch.cat(
            (observations[:, -self.num_obs_now:], latent, explicit), dim=-1
        )
        return self.actor_body(actor_input)

    @torch.no_grad()
    def latent_resample_diagnostics(self, observations):
        """Measure policy-mean sensitivity to latent sampling with isolated RNG."""
        distribution = self.distribution
        module_modes = [(module, module.training) for module in self.modules()]
        devices = [] if observations.device.type == "cpu" else [observations.device.index]
        try:
            with torch.random.fork_rng(devices=devices):
                _, _, latent_a = self.encode_context(observations, sample=True)
                _, _, latent_b = self.encode_context(observations, sample=True)
                action_a = self._actor_mean_from_latent(observations, latent_a)
                action_b = self._actor_mean_from_latent(observations, latent_b)
                return {
                    "Policy/latent_resample_mean_action_delta_rms": torch.sqrt(
                        torch.mean((action_a - action_b).square())
                    ).item()
                }
        finally:
            self.distribution = distribution
            for module, training in module_modes:
                module.training = training

    @torch.no_grad()
    def deterministic_diagnostics(self, observations):
        """Compare latent/action sampling paths without changing model RNG/state."""
        distribution = self.distribution
        module_modes = [(module, module.training) for module in self.modules()]
        devices = [] if observations.device.type == "cpu" else [observations.device.index]
        try:
            with torch.random.fork_rng(devices=devices):
                latent_mean, _, _ = self.encode_context(observations, sample=False)
                deterministic_mean = self._actor_mean_from_latent(observations, latent_mean)
                std = torch.maximum(torch.minimum(self.std, self._std_clip_upr), self._std_clip_lwr)
                gaussian_sample = deterministic_mean + torch.randn_like(deterministic_mean) * std
                _, _, sampled_latent_a = self.encode_context(observations, sample=True)
                _, _, sampled_latent_b = self.encode_context(observations, sample=True)
                sampled_latent_mean_a = self._actor_mean_from_latent(observations, sampled_latent_a)
                sampled_latent_mean_b = self._actor_mean_from_latent(observations, sampled_latent_b)

                def delta_rms(first, second):
                    return torch.sqrt(torch.mean((first - second).square())).item()

                return {
                    "Policy/deterministic_vs_gaussian_sample_action_delta_rms": delta_rms(
                        deterministic_mean, gaussian_sample
                    ),
                    "Policy/deterministic_vs_sampled_latent_mean_action_delta_rms": delta_rms(
                        deterministic_mean, sampled_latent_mean_a
                    ),
                    "Policy/gaussian_sample_vs_sampled_latent_mean_action_delta_rms": delta_rms(
                        gaussian_sample, sampled_latent_mean_a
                    ),
                    "Policy/latent_resample_mean_action_delta_rms": delta_rms(
                        sampled_latent_mean_a, sampled_latent_mean_b
                    ),
                }
        finally:
            self.distribution = distribution
            for module, training in module_modes:
                module.training = training

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_expert(self, ob, policy_info={}):
        return self.act_teacher(ob["obs"], ob["obs_pred"])

    def act_inference(self, ob, policy_info={}):
        return self.act_student(ob["obs"], policy_info=policy_info)

    def act_student(self, observations, policy_info={}):
        _, _, latent = self.encode_context(observations, sample=False)
        obs_pred = self.adaptation_decoder_module(latent)
        actor_inputs = [observations[:, -self.num_obs_now:], latent, self.actor_estimates(obs_pred)]
        actions_mean = self.actor_body(torch.cat(actor_inputs, dim=-1))
        policy_info["latents"] = self.actor_estimates(obs_pred).detach().cpu().numpy()
        return actions_mean

    def act_teacher(self, critic_observations, pred_obs, policy_info={}):
        actions_mean = self.actor_body(critic_observations)
        policy_info["latents"] = pred_obs
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic_body(critic_observations)
        return value

    def get_student_latent(self, observations):
        mean, logvar, latent = self.encode_context(observations, sample=True)
        explicit = self.adaptation_decoder_module(latent)
        return mean, logvar, latent, explicit, self.privileged_decoder_module(latent)

def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
