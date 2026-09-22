"""Local PPO/storage with upstream deterministic four-block adaptation only."""
import torch
from torch import nn
from torch.nn import functional as F
from .ppo_unifp import PPO_UniFP
from rsl_rl.storage.rollout_storage_unifp import RolloutStorageUniFP


class PPO_UniFPOriginal(PPO_UniFP):
    # Reuse collection, GAE, timeout bootstrap and entropy utility methods only.
    # Do not call the extended constructor or update: neither VAE nor KL exists here.
    def __init__(self, actor_critic, device="cpu", decision_callback=None, **cfg):
        self.device, self.actor_critic = device, actor_critic.to(device)
        self.decision_callback = decision_callback
        defaults = dict(learning_rate=3e-4, clip_param=.2, gamma=.99, lam=.95,
            value_loss_coef=1., entropy_coef=.01, max_grad_norm=1., use_clipped_value_loss=True,
            desired_kl=.01, schedule="adaptive", num_learning_epochs=5, num_mini_batches=4,
            use_adaptive_entropy=False)
        for name, default in defaults.items():
            setattr(self, name, cfg.get(name, default))
        self.current_entropy_coef = self.entropy_coef
        self.entropy_coef_bounds = tuple(cfg.get("adaptive_ent_bounds", (.001, .01)))
        self.ent_linvelo_threshold = cfg.get("adaptive_ent_lin_threshold", .75)
        self.ent_angvelo_threshold = cfg.get("adaptive_ent_ang_threshold", .35)
        self.ent_terrain_threshold = cfg.get("adaptive_ent_ter_threshold", 6.)
        self.ent_softmax_temperature = cfg.get("adaptive_ent_softmax_temp", 2.)
        self.num_enc_epochs = int(cfg.get("num_encoder_epochs", 1))
        if self.num_enc_epochs < 1:
            raise ValueError("num_encoder_epochs must be positive")
        self.ppo_parameters = list(actor_critic.parameters())
        self.adaptation_module_parameters = list(actor_critic.adaptation_encoder_module.parameters()) + list(actor_critic.adaptation_decoder_module.parameters())
        self.optimizer = torch.optim.Adam(self.ppo_parameters, lr=self.learning_rate)
        self.adaptation_module_optimizer = torch.optim.Adam(self.adaptation_module_parameters, lr=1e-5)
        self.transition = RolloutStorageUniFP.Transition()
        self.storage = None

    def act(self, obs, critic_obs, obs_pred):
        actions = super().act(obs, critic_obs, obs_pred)
        if self.decision_callback is not None:
            self.decision_callback(self.actor_critic.last_prediction.detach())
        return actions

    def process_env_step(self, rewards, dones, infos, next_privileged_observations=None):
        # The existing storage accepts a zero-width successor field; no next-state loss.
        super().process_env_step(rewards, dones, infos, rewards.new_empty((len(rewards), 0)))

    def adaptation_loss(self, history, labels):
        prediction = self.actor_critic.adaptation_decoder_module(
            self.actor_critic.adaptation_encoder_module(history))
        losses = {name: F.mse_loss(prediction[:, 3*i:3*i+3], labels[:, 3*i:3*i+3].detach())
                  for i, name in enumerate(self.actor_critic.schema)}
        total = sum(weight * loss for weight, loss in zip(self.actor_critic.adaptation_weights, losses.values()))
        return total, losses

    def update(self, iteration):
        model = self.actor_critic
        totals = {name: 0. for name in ("value", "surrogate", "adaptation", *model.schema)}
        updates = 0
        for (obs, critic, labels, _, actions, old_values, advantages, returns,
             old_logprob, old_mu, old_sigma, _) in self.storage.mini_batch_generator(
                 self.num_mini_batches, self.num_learning_epochs):
            model.update_distribution(obs)
            logprob, value = model.get_actions_log_prob(actions), model.evaluate(critic)
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.no_grad():
                    kl = (torch.log(model.action_std / old_sigma + 1e-5)
                          + (old_sigma.square() + (old_mu-model.action_mean).square())
                          / (2*model.action_std.square()) - .5).sum(-1).mean()
                    if kl > 2*self.desired_kl:
                        self.learning_rate = max(1e-5, self.learning_rate/1.5)
                    elif 0 < kl < self.desired_kl/2:
                        self.learning_rate = min(1e-2, self.learning_rate*1.5)
                    for group in self.optimizer.param_groups:
                        group["lr"] = self.learning_rate
            ratio = (logprob - old_logprob.flatten()).exp()
            advantage = advantages.flatten()
            surrogate = torch.maximum(-advantage*ratio,
                -advantage*ratio.clamp(1-self.clip_param, 1+self.clip_param)).mean()
            value_error = (value-returns).square()
            if self.use_clipped_value_loss:
                clipped = old_values + (value-old_values).clamp(-self.clip_param, self.clip_param)
                value_error = torch.maximum(value_error, (clipped-returns).square())
            value_loss = value_error.mean()
            entropy_coef = self.current_entropy_coef if self.use_adaptive_entropy else self.entropy_coef
            loss = surrogate + self.value_loss_coef*value_loss - entropy_coef*model.entropy.mean()
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.ppo_parameters, self.max_grad_norm)
            self.optimizer.step()
            # Deterministic encoder is intentionally shared by both optimizers.
            for _ in range(self.num_enc_epochs):
                adaptation, blocks = self.adaptation_loss(obs, labels)
                self.adaptation_module_optimizer.zero_grad(set_to_none=True)
                adaptation.backward()
                nn.utils.clip_grad_norm_(self.adaptation_module_parameters, self.max_grad_norm)
                self.adaptation_module_optimizer.step()
                totals["adaptation"] += adaptation.item()/self.num_enc_epochs
                for name, error in blocks.items():
                    totals[name] += error.item()/self.num_enc_epochs
            totals["value"] += value_loss.item()
            totals["surrogate"] += surrogate.item()
            updates += 1
        self.storage.clear()
        mean = {key: value/max(updates, 1) for key, value in totals.items()}
        return mean["value"], mean["surrogate"], mean["adaptation"], {k: mean[k] for k in model.schema}, {}
