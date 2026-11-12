"""BR III-B/D MLPs. No inheritance from HardPACT, VAE, or physics heads.

o: 45 = omega_B(3), gravity_B(3), command_B(3), q(12), dq(12), previous action(12).
s: 264 = o plus linear velocity, terrain heights, torques, accelerations, foot forces.
p: 62 simulator randomization parameters. Their detailed packing belongs to Stage 2.
load: [CoM position_B(3) m, relative velocity_B(3) m/s, mass(1) kg], no friction.
The critic sees s,p,load truth and the phase's latent. Actors NEVER receive load truth.
"""
import torch
from torch import nn
from torch.distributions import Normal
from torch.nn import functional as F

from rsl_rl.beyond_robustness_config import BeyondRobustnessConfig, PHASES
from .hard_pact_action_std import initialize_action_std


def _mlp(width, hidden, output):
    layers = []
    for size in hidden:
        layers.extend((nn.Linear(width, size), nn.ELU()))
        width = size
    return nn.Sequential(*layers, nn.Linear(width, output))


class ActorCriticBeyondRobustness(nn.Module):
    is_recurrent = False
    observation_dim, state_dim, parameter_dim = 45, 264, 62
    latent_dim, load_dim, num_actions = 32, 7, 12

    def __init__(self, config=None):
        super().__init__()
        self.config = config or BeyondRobustnessConfig()
        self.history_dim = self.observation_dim * self.config.history_length
        hidden = (512, 256, 128)
        self.privileged_encoder = _mlp(self.state_dim + self.parameter_dim, hidden, 32)
        self.history_encoder = _mlp(self.history_dim, hidden, 32)
        self.load_estimator = _mlp(self.history_dim, (512, 256, 64), 7)
        self.actor = _mlp(45 + 32 + 7, hidden, 12)
        self.critic = _mlp(264 + 62 + 32 + 7, hidden, 1)
        initialize_action_std(self, 12, self.config.init_noise_std,
                              self.config.min_noise_std, self.config.max_noise_std)

    def teacher_latent(self, state, parameters):
        return F.normalize(self.privileged_encoder(torch.cat((state, parameters), -1)), dim=-1)

    def latent(self, history, phase, state=None, parameters=None):
        if phase not in PHASES:
            raise ValueError(f"Unknown training_phase {phase!r}")
        if phase == "teacher_student":
            if state is None or parameters is None:
                raise ValueError("Teacher mode requires s_t and p_t")
            return self.teacher_latent(state, parameters)
        # Only the teacher is normalized explicitly in the paper. Student learns its scale.
        return self.history_encoder(history)

    def distribution(self, observations, history, phase, state=None, parameters=None, latent=None):
        z = self.latent(history, phase, state, parameters) if latent is None else latent
        # Supervised estimator ownership: PPO may adapt to its estimates, not train it.
        with torch.no_grad():
            load = self.load_estimator(history)
        mean = self.actor(torch.cat((observations, z, load), -1))
        std = self.std.clamp(min=self._std_clip_lwr, max=self._std_clip_upr)
        return Normal(mean, std.expand_as(mean))

    def evaluate(self, state, parameters, load_labels, latent):
        return self.critic(torch.cat((state, parameters, load_labels, latent), -1))

    def evaluate_batch(self, batch, phase):
        z = self.latent(batch["history"], phase, batch["state"], batch["parameters"])
        dist = self.distribution(batch["observations"], batch["history"], phase, latent=z)
        value = self.evaluate(batch["state"], batch["parameters"], batch["load_labels"], z)
        return dist, value

    @torch.no_grad()
    def act_inference(self, observations, history):
        """Deployable deterministic student: no privileged/load-label arguments or caches."""
        return self.distribution(observations, history, "student_reinforcement").mean

    def input_contract(self):
        """Versioned semantic boundary, independent of simulator and learning phase.

        The paper does not give the scalar packing of all 264 state / 62 parameter
        entries. Stage 2 must establish it; these stay separate BR inputs, not a
        borrowed HardPACT privileged vector. History includes the current frame,
        resolving the paper's ambiguous inclusive H notation explicitly.
        """
        return dict(version=1, observation_dim=45, state_dim=264, parameter_dim=62,
                    history_length=self.config.history_length, history_order="oldest_to_current_zero_reset_padding",
                    observation_fields=("angular_velocity_B:3", "gravity_B:3", "command_B:3",
                                        "joint_position:12", "joint_velocity:12", "previous_action:12"),
                    load_fields=("relative_com_position_B_m:3", "relative_velocity_B_m_s:3", "mass_kg:1"),
                    actor_load="estimated_detached", critic_load="ground_truth",
                    latent_dim=32, teacher_latent_normalized=True, student_latent_normalized=False,
                    action_dim=12, action_type="gaussian_position", likelihood="raw_sample")
