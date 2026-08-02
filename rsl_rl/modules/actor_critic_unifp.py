import torch
import torch.nn as nn
from torch.distributions import Normal
from .module_utils import init_weights

class AC_Args:

    adaptation_module_encoder_hidden_dims = [512, 256, 128]

    adaptation_module_decoder_hidden_dims = [128, 64]

    adaptation_module_decoder_recon_hidden_dims = [128, 256, 512]
    
    adaptation_labels = ["base_velocity_loss", "gripper_pos_loss", "force_ee_loss", "force_base_loss"] #, "force_loss"]
    adaptation_dims = [3, 3, 3, 3] #, 3]
    adaptation_weights = [.2, .2, 1.0, 1.0] #, 1]


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
                        **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super().__init__()
        
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
        self.num_latent_dim = 16
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

        # The same sampled z predicts the next single privileged frame. This
        # decoder is training-only; no privileged state enters the actor.
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
        pass

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
        """Map decoder outputs into the representation consumed by the actor."""
        return explicit_prediction

    def update_distribution(self, observations):
        mean, _, latent = self.encode_context(observations, sample=True)
        actor_inputs = [
            observations[:, -self.num_obs_now:],
            latent,
            self.actor_estimates(self.adaptation_decoder_module(latent)),
        ]
        mean = self.actor_body(torch.cat(actor_inputs, dim=-1))
        self._clip_std()
        self.distribution = Normal(mean, mean*0. + self.std)

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
