import torch
import torch.nn as nn
from torch.distributions import Normal


class AC_Args:

    adaptation_module_encoder_hidden_dims = [512, 256, 128]

    adaptation_module_decoder_hidden_dims = [128, 64]
    
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
                        actor_hidden_dims=[256, 256, 256],
                        critic_hidden_dims=[256, 256, 256],
                        activation='elu',
                        init_noise_std=1.0,
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
        self.num_latent_dim = int(num_obs / num_single_obs) * 2
        self.num_obs_now = num_single_obs

        activation = get_activation(activation)

        # Adaptation module
        adaptation_module_encoder_layers = []
        adaptation_module_encoder_layers.append(nn.Linear(self.num_obs, AC_Args.adaptation_module_encoder_hidden_dims[0]))
        adaptation_module_encoder_layers.append(activation)
        for l in range(len(AC_Args.adaptation_module_encoder_hidden_dims)):
            if l == len(AC_Args.adaptation_module_encoder_hidden_dims) - 1:
                adaptation_module_encoder_layers.append(
                    nn.Linear(AC_Args.adaptation_module_encoder_hidden_dims[l], self.num_latent_dim))
            else:
                adaptation_module_encoder_layers.append(
                    nn.Linear(AC_Args.adaptation_module_encoder_hidden_dims[l],
                              AC_Args.adaptation_module_encoder_hidden_dims[l + 1]))
                adaptation_module_encoder_layers.append(activation)
        self.adaptation_encoder_module = nn.Sequential(*adaptation_module_encoder_layers)


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



        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(self.num_obs_now + self.num_latent_dim, actor_hidden_dims[0]))
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
        print(f"Adaptation Decoder Module: {self.adaptation_decoder_module}")
        print(f"Actor MLP: {self.actor_body}")
        print(f"Critic MLP: {self.critic_body}")

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]


    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError
    
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        latent = self.adaptation_encoder_module(observations)
        mean = self.actor_body(torch.cat((observations[:, -self.num_obs_now:], latent), dim=-1))
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
        latent = self.adaptation_encoder_module(observations)
        actions_mean = self.actor_body(torch.cat((observations[:, -self.num_obs_now:], latent), dim=-1))
        obs_pred = self.adaptation_decoder_module(latent)
        policy_info["latents"] = obs_pred.detach().cpu().numpy()
        return actions_mean

    def act_teacher(self, critic_observations, pred_obs, policy_info={}):
        actions_mean = self.actor_body(critic_observations)
        policy_info["latents"] = pred_obs
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic_body(critic_observations)
        return value

    def get_student_latent(self, observations):
        latent = self.adaptation_encoder_module(observations)
        obs_pred = self.adaptation_decoder_module(latent)
        return obs_pred

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
