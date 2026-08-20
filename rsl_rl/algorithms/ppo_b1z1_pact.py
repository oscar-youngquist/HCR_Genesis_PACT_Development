"""PPO plus temporal context and one Pinocchio consistency objective for B1/Z1."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn, optim
import warnings

from rsl_rl.algorithms.pc_grad import PCGrad
from rsl_rl.algorithms.kl_rate_band import KLRateBandController, update_duals_from_mean
from rsl_rl.storage.rollout_storage_b1z1_pact import RolloutStorageB1Z1PACT


def _go2_relative_pinn_loss(residual, generalized_tau, generalized_external, valid):
    """Return Go2-PACT's relative residual over valid transitions."""
    denominator = (
        1.0e-8
        + torch.linalg.vector_norm(generalized_tau.detach(), dim=1)
        + torch.linalg.vector_norm(generalized_external, dim=1)
    )
    relative_error = torch.linalg.vector_norm(residual, dim=1) / denominator
    valid_rows = valid.squeeze(-1).to(relative_error.dtype)
    return (relative_error * valid_rows).sum() / valid_rows.sum().clamp_min(1.0)


class PPO_B1Z1PACT:
    def __init__(self, actor_critic, privileged_decoder, dynamics_backend, cfg, device):
        self.actor_critic, self.privileged_decoder = actor_critic, privileged_decoder
        self.dynamics_backend, self.cfg, self.device = dynamics_backend, cfg, device
        self.enable_additional_diagnostics = True
        self.clip_param, self.gamma, self.lam = cfg["clip_param"], cfg["gamma"], cfg["lam"]
        self.value_loss_coef, self.entropy_coef = cfg["value_loss_coef"], cfg["entropy_coef"]
        self.use_adaptive_entropy = cfg.get("use_adaptive_entropy", False)
        self.entropy_coef_bounds = tuple(float(value) for value in cfg.get("adaptive_ent_bounds", (self.entropy_coef, self.entropy_coef)))
        self.ent_linvelo_threshold = float(cfg.get("adaptive_ent_lin_threshold", 0.0))
        self.ent_angvelo_threshold = float(cfg.get("adaptive_ent_ang_threshold", 0.0))
        self.ent_terrain_threshold = float(cfg.get("adaptive_ent_ter_threshold", 0.0))
        self.ent_softmax_temperature = float(cfg.get("adaptive_ent_softmax_temp", 1.0))
        self.current_entropy_coef = float(self.entropy_coef)
        if len(self.entropy_coef_bounds) != 2:
            raise ValueError("adaptive_ent_bounds must contain [low, high]")
        if self.entropy_coef_bounds[0] < 0.0 or self.entropy_coef_bounds[1] < self.entropy_coef_bounds[0]:
            raise ValueError("adaptive_ent_bounds must satisfy 0 <= low <= high")
        if self.ent_softmax_temperature <= 0.0:
            raise ValueError("adaptive_ent_softmax_temp must be greater than zero")
        self.max_grad_norm, self.epochs, self.mini_batches = cfg["max_grad_norm"], cfg["num_learning_epochs"], cfg["num_mini_batches"]
        self.learning_rate = cfg["learning_rate"]
        self.desired_kl, self.schedule = cfg.get("desired_kl"), cfg.get("schedule", "fixed")
        self.use_clipped_value_loss = cfg.get("use_clipped_value_loss", True)
        self.film_identity_loss_weight = cfg.get("film_identity_loss_weight", 0.0)
        self.film_identity_error_scale = cfg.get("film_identity_error_scale", 1.0)
        if self.film_identity_loss_weight < 0.0:
            raise ValueError("film_identity_loss_weight must be nonnegative")
        if self.film_identity_error_scale <= 0.0:
            raise ValueError("film_identity_error_scale must be positive")

        # ``get_optim_groups`` is the actor-critic's source of truth for
        # actor/critic/context partitioning and its weight-decay conventions.
        actor_groups, context_groups = actor_critic.get_optim_groups()

        encoder_weight_decay = context_groups[0].get("weight_decay", 0.0)
        auxiliary_groups = list(context_groups) + [
            {
                "params": list(privileged_decoder.parameters()),
                "weight_decay": encoder_weight_decay,
                "name": "privileged_decoder",
            },
        ]

        # PPO updates the complete context pathway. Coupled PACT additionally
        # owns the privileged decoder here because its predicted force block is
        # part of the differentiable PINN graph.
        ppo_enc_groups = [
            {
                "params": list(group["params"]),
                "weight_decay": group.get("weight_decay", 0.0),
                "name": f"ppo_{group['name']}",
            }
            for group in auxiliary_groups
        ]
        auxiliary_enc_groups = [
            {
                "params": list(group["params"]),
                "weight_decay": group.get("weight_decay", 0.0),
                "name": f"auxiliary_{group['name']}",
            }
            for group in auxiliary_groups
        ]

        self.actor_optimizer = PCGrad(
            optim.Adam([*actor_groups, *ppo_enc_groups], lr=cfg["learning_rate"]),
            reduction="sum",
        )
        # Clip the same ownership boundary that is stepped by PPO, as UniFP
        # does, including any PACT decoder participating in the PINN graph.
        seen_ppo_parameters = set()
        self.ppo_parameters = []
        for group in self.actor_optimizer.optimizer.param_groups:
            for parameter in group["params"]:
                if id(parameter) not in seen_ppo_parameters:
                    seen_ppo_parameters.add(id(parameter))
                    self.ppo_parameters.append(parameter)

        self.auxiliary_optimizer = optim.Adam(
            [parameter for group in auxiliary_enc_groups for parameter in group["params"]],
            lr=cfg.get("adaptation_learning_rate", 1.0e-5),
        )

        seen_enc_parameters = set()
        self.enc_parameters = []
        for group in self.auxiliary_optimizer.param_groups:
            for parameter in group["params"]:
                if id(parameter) not in seen_enc_parameters:
                    seen_enc_parameters.add(id(parameter))
                    self.enc_parameters.append(parameter)

        self.kl_controller = KLRateBandController(
            warmup_iters=cfg.get("kl_warmup_iters", 500),
            warmup_beta_max=cfg.get("kl_warmup_beta_max", cfg["vae_kld_weight"]),
            band_warmup_iters=cfg.get("kl_band_warmup_iters", 500),
            rate_min=cfg.get("kl_r_min", 0.10), rate_max=cfg.get("kl_r_max", 1.00),
            dual_lr=cfg.get("kl_dual_lr", 1.0e-3),
            augmented_rho=cfg.get("kl_aug_rho", 0.1),
            ema_decay=cfg.get("kl_ema_decay", 0.99),
        )
        self.use_kl_rate_band = bool(cfg.get("use_kl_rate_band", True))
        self.use_cosine_kl_warmup = bool(
            cfg.get("use_cosine_kl_warmup", True)
        )

        self.transition = RolloutStorageB1Z1PACT.Transition()
        self.storage = None
        self.pinn_weight, self.pinn_updates = 0.0, 0
        # The first decoder measurement initializes the EMA; starting at
        # infinity would keep the reliability gate permanently closed.
        self.force_ema = None
        self.force_gate_active, self.force_gate_count = False, 0
        self.force_blend_min_alpha = float(cfg.get("force_blend_min_alpha", 0.01))
        if not 0.0 <= self.force_blend_min_alpha <= 1.0:
            raise ValueError("force_blend_min_alpha must lie in [0, 1]")
        # The first reconstruction EMA defines alpha_min. As reconstruction
        # improves toward the gate threshold, predicted-force authority ramps
        # linearly to one. Persist this reference across checkpoint resumes.
        self.force_blend_start_ema = None

    def init_storage(self, *args):
        self.storage = RolloutStorageB1Z1PACT(*args, device=self.device)

    def update_adaptive_entropy_coef(self, performance_metrics):
        """Increase exploration when tracking or terrain progress is below target."""
        def normalized_gap(name, threshold):
            if threshold <= 0.0:
                return 0.0
            value = float(performance_metrics.get(name, 0.0))
            return max(0.0, threshold - value) / threshold

        gaps = torch.tensor((
            normalized_gap("lin_vel_tracking", self.ent_linvelo_threshold),
            normalized_gap("ang_vel_tracking", self.ent_angvelo_threshold),
            normalized_gap("terrain_level", self.ent_terrain_threshold),
        ), dtype=torch.float32, device=self.device)
        weights = F.softmax(gaps / self.ent_softmax_temperature, dim=0)
        weighted_gap = torch.sum(weights * gaps).item()
        low, high = self.entropy_coef_bounds
        self.current_entropy_coef = low + weighted_gap * (high - low)
        return self.current_entropy_coef

    def act(self, obs, critic_obs, history, explicit_labels):
        actions = self.actor_critic.act(obs, history).detach()
        self.transition.actions = actions
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.log_probs = self.actor_critic.get_actions_log_prob(actions).detach().unsqueeze(-1)
        self.transition.mu, self.transition.sigma = self.actor_critic.action_mean.detach(), self.actor_critic.action_std.detach()
        # Snapshot every actor-time environment tensor before env.step(),
        # exactly as UniFP does for its observation and estimator labels.
        self.transition.observations = obs.detach().clone()
        self.transition.critic_observations = critic_obs.detach().clone()
        self.transition.histories = history.detach().clone()
        self.transition.explicit_targets = explicit_labels.detach().clone()
        return actions

    def process_env_step(self, rewards, dones, infos, next_privileged, dynamics_state):
        # ``dynamics_state`` is the post-step state collected by the runner.
        # Storing it beside this transition keeps action_t, v_t, and v_(t+1)
        # together until the shuffled PPO update.
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        self.transition.next_privileged = next_privileged.detach().clone()
        self.transition.dynamics_state = dynamics_state.detach().clone()
        if "time_outs" in infos:
            # Rewards are [N], matching UniFP. Squeeze the [N, 1] bootstrap
            # correction before in-place addition, then storage restores [N, 1].
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * infos["time_outs"].unsqueeze(1).to(self.device), 1
            )
        self.storage.add(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)

    def spectral_normalization(
        self,
        model: nn.Module,
        sigma_max: float = 1.0,
        n_power_iters: int = 1,
    ):
        """
        Spectral-norm clip all Linear layers except selected output layers.

        Args:
            model: network to normalize in-place
            sigma_max: maximum allowed spectral norm
            n_power_iters: number of power iterations for sigma estimate
        """

        whitelist = (nn.Linear,)

        # lazily create persistent power-iteration vectors
        if not hasattr(self, "_spec_u"):
            self._spec_u = {}

        for module_name, module in model.named_modules():
            if not isinstance(module, whitelist):
                continue

            # skip known output layers
            if module_name.endswith("out") or module_name.endswith("mean") or module_name.endswith("var") or "critic" in module_name:
                continue

            for param_name, param in module.named_parameters(recurse=False):
                if param_name != "weight" or param.ndim != 2:
                    continue

                full_name = f"{module_name}.{param_name}" if module_name else param_name
                W = param.data  # [out_dim, in_dim]

                # initialize persistent u vector once per parameter
                if full_name not in self._spec_u or self._spec_u[full_name].shape[0] != W.shape[0]:
                    u = torch.randn(W.shape[0], device=W.device, dtype=W.dtype)
                    u = u / (u.norm() + 1e-12)
                    self._spec_u[full_name] = u

                u = self._spec_u[full_name]

                with torch.no_grad():
                    # power iteration
                    for _ in range(n_power_iters):
                        v = W.t().mv(u)
                        v = v / (v.norm() + 1e-12)

                        u = W.mv(v)
                        u = u / (u.norm() + 1e-12)

                    # sigma ~= u^T W v
                    sigma = torch.dot(u, W.mv(v))

                    # save updated u for next call
                    self._spec_u[full_name] = u

                    # clip only if above threshold
                    if sigma > sigma_max:
                        param.data.mul_(sigma_max / (sigma + 1e-12))

    def compute_returns(self, critic_obs):
        self.storage.compute_returns(self.actor_critic.evaluate(critic_obs).detach(), self.gamma, self.lam)

    def _coupled_torque(self, actions, state):
        """Reconstruct the exact generalized joint torque used by Genesis.

        For learned joints,
          q_target = q_default + s_pos * a_pos
          tau_PD = Kp (q_target - q) - Kd qdot
          tau_FF = s_tau * motor_strength * a_tau
          tau = w_PD tau_PD + w_FF tau_FF.

        The two trailing arm/gripper DOFs have no learned action; their normal
        default-pose PD torque is nevertheless included because Pinocchio sees
        all 19 actuated coordinates. This prevents the residual from treating
        those real actuator torques as unexplained external forces.
        """
        q = state[:, 7:26]
        qd = state[:, 32:51]
        motor = state[:, 97:116]
        kp, kd = state[:, 116:135], state[:, 135:154]
        weights = state[:, 154:156]
        position, feedforward = actions[:, :17], actions[:, 17:34]
        default = state[:, 156:175]

        target = default[:, :17] + self.cfg["position_action_scale"] * position

        feedback = kp[:, :17] * (target - q[:, :17]) - kd[:, :17] * qd[:, :17]

        feedforward = self.cfg["torque_action_scale"] * feedforward

        controlled = (weights[:, :1] * feedback + weights[:, 1:2] * feedforward) * motor[:, :17]

        uncontrolled = kp[:, 17:] * (default[:, 17:] - q[:, 17:]) - kd[:, 17:] * qd[:, 17:]

        return torch.cat((controlled, uncontrolled), dim=-1)

    def _force_prediction_blend_alpha(self):
        """Return predicted-force authority before the reliability gate opens."""
        if self.force_gate_active:
            return 1.0
        if self.force_ema is None or self.force_blend_start_ema is None:
            return self.force_blend_min_alpha

        threshold = float(self.cfg["force_gate_threshold"])
        start = float(self.force_blend_start_ema)
        current = float(self.force_ema)
        if start <= threshold:
            # Reconstruction already met the target on its first measurement;
            # patience may still hold the Boolean gate closed.
            return 1.0 if current <= threshold else self.force_blend_min_alpha

        progress = (start - current) / (start - threshold)
        progress = min(max(progress, 0.0), 1.0)
        return self.force_blend_min_alpha + (1.0 - self.force_blend_min_alpha) * progress

    def _pinn_loss(self, mean_actions, context, privileged_force_prediction, state, valid):
        """Evaluate the observed-transition whole-body consistency loss.

        The residual has the 25 free-flyer/generalized coordinates of B1/Z1:

          r = M(q) vdot + h(q, v) - S^T tau
              - sum_i J_foot_i^T GRF_i
              - J_EE,linear^T F_EE - J_base^T W_base.

        ``M vdot + h`` is the generalized force needed by the observed motion.
        The remaining terms explain that demand with executed joint torques,
        ground reaction forces, the linear EE disturbance, and the base wrench.
        A small norm means the policy action and post-action simulator state
        tell a physically compatible story.

        Pinocchio runs on CPU/shared-memory workers in this initial version.
        Consequently gradients pass through reconstructed ``tau`` into the
        actor, but not through the model-evaluated state terms or force Jacobian
        products. A future differentiable backend can preserve this interface.
        """
        # State layout is documented by B1Z1PACT.get_pact_dynamics_state().
        # Unlike context["base_velocity"], these velocities are physical SI
        # values. The Pinocchio residual must not inherit observation scaling.
        if len(self.cfg["base_velocity_scale"]) != 3:
            raise RuntimeError("base_velocity_scale must describe [vx, vy, yaw_rate]")
        base_pos, base_quat, q = state[:, :3], state[:, 3:7], state[:, 7:26]

        v = state[:, 26:51]
        previous_v = state[:, 51:76]

        measured_grfs, measured_ee, measured_base = state[:, 76:88], state[:, 88:91], state[:, 91:97]

        # Decoder outputs use observation-space normalization and a yaw-aligned
        # frame. Convert predictions to world/SI before mixing them with the
        # measured forces consumed by Pinocchio's LWA Jacobians.
        predicted_grfs = privileged_force_prediction[:, :12] / self.cfg["grf_scale"]
        predicted_ee = privileged_force_prediction[:, 12:15] / self.cfg["ee_force_scale"]
        wrench_scale = privileged_force_prediction.new_tensor(self.cfg["base_wrench_scale"])
        predicted_base = privileged_force_prediction[:, 15:21] / wrench_scale
        x, y, z, w = base_quat.unbind(dim=-1)
        yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y.square() + z.square()))
        c, s = torch.cos(yaw), torch.sin(yaw)

        def yaw_to_world(vectors):
            shape = vectors.shape
            vectors = vectors.reshape(vectors.shape[0], -1, 3)
            world = torch.stack((
                c[:, None] * vectors[..., 0] - s[:, None] * vectors[..., 1],
                s[:, None] * vectors[..., 0] + c[:, None] * vectors[..., 1],
                vectors[..., 2],
            ), dim=-1)
            return world.reshape(shape)

        predicted_grfs = yaw_to_world(predicted_grfs)
        predicted_ee = yaw_to_world(predicted_ee)
        predicted_base = torch.cat((
            yaw_to_world(predicted_base[:, :3]),
            yaw_to_world(predicted_base[:, 3:6]),
        ), dim=-1)
        if self.cfg["predicted_force_detach"]:
            predicted_grfs, predicted_ee, predicted_base = (
                predicted_grfs.detach(), predicted_ee.detach(), predicted_base.detach()
            )

        # Before the reliability gate opens, gradually expose inverse dynamics
        # to decoder predictions instead of making an abrupt measured/predicted
        # switch. Once active, alpha remains exactly one.
        alpha = self._force_prediction_blend_alpha()
        grfs = torch.lerp(measured_grfs, predicted_grfs, alpha)
        ee_force = torch.lerp(measured_ee, predicted_ee, alpha)
        base_wrench = torch.lerp(measured_base, predicted_base, alpha)

        # The backend reproduces the simulator's mass/COM randomization before
        # computing M, h, and J^T F, so inertial mismatch is not mislabeled as
        # an actor error.
        terms = self.dynamics_backend.evaluate(
            base_pos, base_quat, q, v[:, :3], v[:, 3:6], v[:, 6:],
            grfs.view(-1, 4, 3), ee_force, base_wrench,
            state[:, 175:176], state[:, 176:179], state[:, 179:180],
        )

        # This backward difference is transition-aligned: v is v_(t+1) after
        # action_t, while previous_v was cached immediately before action_t.
        acceleration = (v - previous_v) / self.cfg["dt"]

        tau = self._coupled_torque(mean_actions, state)

        # S^T tau inserts zeros for the unactuated free-flyer base coordinates.
        generalized_tau = torch.cat((torch.zeros(tau.shape[0], 6, device=tau.device), tau), dim=-1)

        inertial = torch.bmm(terms.mass_matrix, acceleration.unsqueeze(-1)).squeeze(-1)
        residual = inertial + terms.bias - terms.generalized_contacts - generalized_tau

        # Match Go2-PACT's relative whole-body residual. Each transition is
        # normalized by the magnitude of the generalized actuator force and
        # the generalized external force (feet + EE + base wrench). Detaching
        # actuator torque matches Go2 and prevents its denominator from being
        # an actor-controlled route for reducing the loss.
        # Terminal/reset transitions remain excluded from the reduction.
        return _go2_relative_pinn_loss(
            residual, generalized_tau, terms.generalized_contacts, valid
        )


    def _compute_vae_loss(self, obs_hist_batch, obs_target, labels, valid, iteration):
        # Recompute the auxiliary graph after the actor update. The PPO
        # graph was consumed by PCGrad and sharing it here would either
        # fail on a second backward pass or retain an unnecessarily large
        # rollout graph.
        aux_context = self.actor_critic.decode_context(
            self.actor_critic.context_encoder(obs_hist_batch, sample=True)
        )
        # One z-only decoder reconstructs the next non-terrain privileged
        # state, including normalized GRF, EE-force, and base-wrench values.
        # The critic's terrain-height tail is intentionally absent here.
        aux_privileged_prediction = self.privileged_decoder(aux_context["z"])

        with torch.no_grad():
            base_velo_label = labels[:, :3]
            ee_pos_label = labels[:, 3:6]
            base_wrench_label = labels[:, 6:12]
            ee_force_label = labels[:, 12:15]
            foot_contact_label = labels[:, 15:19]
            foot_height_label = labels[:, 19:23]
            privileged_target = obs_target

        pred_velo_loss = F.mse_loss(aux_context["base_velocity"], base_velo_label)
        pred_ee_position_loss = F.mse_loss(aux_context["ee_position"], ee_pos_label)
        pred_base_wrench_loss = F.mse_loss(aux_context["base_wrench"], base_wrench_label)
        pred_ee_force_loss = F.mse_loss(aux_context["ee_force"], ee_force_label)
        # BCE-with-logits is the stable binary-state reconstruction loss.
        pred_foot_contact_loss = F.binary_cross_entropy_with_logits(
            aux_context["foot_contact_logits"], foot_contact_label,
        )
        pred_foot_height_loss = F.mse_loss(
            aux_context["foot_height"], foot_height_label,
        )

        # Loss for explicit current-state-estimation
        aux_explicit = (
            self.cfg["explicit_base_vel_weight"] * pred_velo_loss
            + self.cfg["explicit_ee_position_weight"] * pred_ee_position_loss
            + self.cfg["explicit_base_wrench_weight"] * pred_base_wrench_loss
            + self.cfg["explicit_ee_force_weight"] * pred_ee_force_loss
            + self.cfg["explicit_foot_contact_weight"] * pred_foot_contact_loss
            + self.cfg["explicit_foot_height_weight"] * pred_foot_height_loss
        )

        # VAE recon + KL losses
        privileged_error = (aux_privileged_prediction - privileged_target).square() * valid
        aux_privileged_loss = privileged_error.sum() / (
            valid.sum().clamp_min(1.0) * aux_privileged_prediction.shape[-1]
        )
        kl_per_sample = -0.5 * (
            1 + aux_context["logvar"] - aux_context["mean"].square() - aux_context["logvar"].exp()
        ).sum(dim=-1, keepdim=True)
        aux_kl = (kl_per_sample * valid).sum() / valid.sum().clamp_min(1.0)
        kl_reg_loss = self.kl_controller.loss(
            aux_kl, iteration, self.use_kl_rate_band,
            self.use_cosine_kl_warmup,
        )

        # Total loss
        aux = (
            aux_explicit
            + self.cfg["privileged_decoder_weight"] * aux_privileged_loss
            + kl_reg_loss
        )

        self.auxiliary_optimizer.zero_grad()

        aux.backward()

        torch.nn.utils.clip_grad_norm_(self.enc_parameters, self.max_grad_norm)

        # Match UniFP adaptation training: the auxiliary optimizer steps its
        # raw reconstruction gradient without a separate clipping pass.
        self.auxiliary_optimizer.step()

        # Return unweighted predictions/targets for the reliability gate.  The
        # gate compares raw MSEs, not the task-specific loss weights above, to
        # answer the simple question: does each decoder beat a constant mean?
        return {
            "base_velocity": pred_velo_loss,
            "ee_position": pred_ee_position_loss,
            "base_wrench": pred_base_wrench_loss,
            "ee_force": pred_ee_force_loss,
            "foot_contact": pred_foot_contact_loss,
            "foot_height": pred_foot_height_loss,
            "privileged_force": self._masked_force_slice_mse(aux_privileged_prediction, obs_target, valid),
            "privileged_decoder": aux_privileged_loss,
            "kl_raw": aux_kl,
            "kl_reg_loss": kl_reg_loss,
            # Sigmoid probabilities provide a bounded contact prediction for
            # the original PACT MSE-based bootstrap statistic; BCE above is
            # still the optimization objective for these binary labels.
            "explicit_prediction": torch.cat((
                aux_context["base_velocity"], aux_context["ee_position"],
                aux_context["base_wrench"], aux_context["ee_force"],
                torch.sigmoid(aux_context["foot_contact_logits"]),
                aux_context["foot_height"],
            ), dim=-1).detach(),
            "explicit_target": labels.detach(),
            "privileged_prediction": aux_privileged_prediction.detach(),
            "privileged_target": obs_target.detach(),
            "valid": valid.detach(),
        }

    def _masked_force_slice_mse(self, prediction, target, valid):
        """Measure the force block inside the next privileged-frame reconstruction."""
        start = self.cfg["privileged_force_start"]
        end = start + self.cfg["privileged_force_dim"]
        error = (prediction[:, start:end] - target[:, start:end]).square() * valid
        return error.sum() / (valid.sum().clamp_min(1.0) * (end - start))


    def _compute_rl_loss(self, batch):
        """Compute the PPO objective with the same organization as PACT PPO.

        The actor always uses latent z and the decoder-predicted explicit context.
        """
        self.actor_critic.update_distribution(
            batch["observations"], batch["histories"], sample_context=False
        )
        actions_log_prob = self.actor_critic.get_actions_log_prob(batch["actions"])
        values = self.actor_critic.evaluate(batch["critic_observations"])
        mu, sigma = self.actor_critic.action_mean, self.actor_critic.action_std

        kl_mean = values.new_zeros(())
        if self.desired_kl is not None and self.schedule == "adaptive":
            with torch.inference_mode():
                kl = torch.sum(
                    torch.log(sigma / batch["sigma"] + 1.0e-5)
                    + (batch["sigma"].square() + (batch["mu"] - mu).square()) / (2.0 * sigma.square()) - 0.5,
                    dim=-1,
                )
                kl_mean = kl.mean()
                if kl_mean > 2.0 * self.desired_kl:
                    self.learning_rate = max(1.0e-5, self.learning_rate / 1.5)
                elif 0.0 < kl_mean < self.desired_kl / 2.0:
                    self.learning_rate = min(1.0e-2, self.learning_rate * 1.5)
                for group in self.actor_optimizer.optimizer.param_groups:
                    group["lr"] = self.learning_rate

        ratio = torch.exp(actions_log_prob - batch["log_probs"].squeeze(-1))
        advantage = batch["advantages"].squeeze(-1)
        surrogate_loss = torch.maximum(
            -advantage * ratio,
            -advantage * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param),
        ).mean()
        if self.use_clipped_value_loss:
            value_clipped = batch["values"] + (values - batch["values"]).clamp(-self.clip_param, self.clip_param)
            value_loss = torch.maximum((values - batch["returns"]).square(), (value_clipped - batch["returns"]).square()).mean()
        else:
            value_loss = (values - batch["returns"]).square().mean()
        # Near a well-tracked command, FiLM should reduce to the identity
        # transform: gamma=0 and beta=0. The exponential gate relaxes this
        # constraint as base/EE tracking error grows, leaving FiLM free to
        # produce stronger corrective modulation. Detaching the gate prevents
        # the estimator from inflating its error merely to evade the penalty.
        film_identity_loss = self.actor_critic.last_film_identity_deviation.mean()
        ppo_loss = (
            surrogate_loss
            + self.value_loss_coef * value_loss
            - (self.current_entropy_coef if self.use_adaptive_entropy else self.entropy_coef)
            * self.actor_critic.entropy.mean()
            + self.film_identity_loss_weight * film_identity_loss
        )

        context = self.actor_critic.last_context
        privileged_prediction = self.privileged_decoder(context["z"])
        force_start = self.cfg["privileged_force_start"]
        force_end = force_start + self.cfg["privileged_force_dim"]
        return (
            ppo_loss, surrogate_loss, value_loss, film_identity_loss, kl_mean,
            self.actor_critic.action_mean, context, privileged_prediction[:, force_start:force_end],
        )

    @torch.no_grad()
    def _pre_update_diagnostics(self, batch):
        """Compare the untouched rollout policy with its stored distribution."""
        self.actor_critic.update_distribution(
            batch["observations"], batch["histories"], sample_context=False
        )
        mu, sigma = self.actor_critic.action_mean, self.actor_critic.action_std
        old_mu, old_sigma = batch["mu"], batch["sigma"]
        new_logprob = self.actor_critic.get_actions_log_prob(batch["actions"])
        old_logprob = batch["log_probs"].squeeze(-1)
        mu_error, sigma_error = mu - old_mu, sigma - old_sigma
        logprob_error = new_logprob - old_logprob
        ratio = torch.exp(logprob_error)
        eps = 1.0e-8
        old_sigma_safe, sigma_safe = old_sigma.clamp_min(eps), sigma.clamp_min(eps)
        kl = torch.sum(
            torch.log(sigma_safe / old_sigma_safe)
            + (old_sigma_safe.square() + (old_mu - mu).square()) / (2.0 * sigma_safe.square())
            - 0.5,
            dim=-1,
        ).mean()
        diagnostics = {
            "pre_update_mu_rms": mu_error.square().mean().sqrt().item(),
            "pre_update_mu_abs_max": mu_error.abs().max().item(),
            "pre_update_sigma_rms": sigma_error.square().mean().sqrt().item(),
            "pre_update_sigma_abs_max": sigma_error.abs().max().item(),
            "pre_update_logprob_rms": logprob_error.square().mean().sqrt().item(),
            "pre_update_logprob_abs_max": logprob_error.abs().max().item(),
            "pre_update_ratio_mean": ratio.mean().item(),
            "pre_update_ratio_std": ratio.std(unbiased=False).item(),
            "pre_update_kl": kl.item(),
        }
        if (diagnostics["pre_update_mu_abs_max"] > 1.0e-6
                or diagnostics["pre_update_sigma_abs_max"] > 1.0e-6
                or diagnostics["pre_update_logprob_abs_max"] > 1.0e-5):
            warnings.warn(f"PACT pre-update PPO inconsistency: {diagnostics}", RuntimeWarning)
        return diagnostics


    def update(self, iteration):
        self.actor_critic.train()
        self.privileged_decoder.train()
        # Delay and then ramp the physical constraint. PPO first learns a
        # minimally viable behavior before the residual competes with reward.
        if iteration >= self.cfg["pinn_init_steps"]:
            progress = min(1.0, self.pinn_updates / max(1, self.cfg["pinn_warmup"]))
            self.pinn_weight = progress * self.cfg["pinn_loss_weight"]
            self.pinn_updates += 1
        metrics = {name: 0.0 for name in (
            "value", "surrogate", "base_velo", "ee_position", "base_wrench", "ee_force", "foot_contact", "foot_height",
            "privileged_force", "privileged_decoder", "pinn", "film_identity",
            *KLRateBandController.METRIC_NAMES,
        )}
        updates = 0
        raw_kl_sum = 0.0
        diagnostics = {"lr_before_update": self.learning_rate}
        for batch in self.storage.mini_batches(self.mini_batches, self.epochs):
            if updates == 0 and self.enable_additional_diagnostics:
                diagnostics.update(self._pre_update_diagnostics(batch))
            ppo_loss, surrogate, value, film_identity, _, mean_actions, context, force_prediction = self._compute_rl_loss(batch)
            
            valid = (~batch["dones"].squeeze(-1)).float().unsqueeze(-1)
            # Both decoder losses use the next simulator state stored with this
            # action transition. Multiplying by valid masks reset transitions.
            force_start = self.cfg["privileged_force_start"]
            force_end = force_start + self.cfg["privileged_force_dim"]
            force_target = batch["next_privileged"][:, force_start:force_end]
            force_loss = ((force_prediction - force_target).square() * valid).sum() / (
                valid.sum().clamp_min(1.0) * self.cfg["privileged_force_dim"]
            )
            force_measurement = force_loss.detach().item()

            # Reliability is deliberately temporal: a single lucky minibatch
            # must not switch the PINN from measured to predicted forces.
            self.force_ema = force_measurement if self.force_ema is None else (
                self.cfg["force_gate_ema_alpha"] * force_measurement
                + (1 - self.cfg["force_gate_ema_alpha"]) * self.force_ema
            )
            if self.force_blend_start_ema is None:
                self.force_blend_start_ema = self.force_ema
            # Hysteresis avoids rapid gate toggling near the reconstruction
            # threshold after it has become active.
            threshold = self.cfg["force_gate_threshold"] if not self.force_gate_active else self.cfg["force_gate_hysteresis"]
            self.force_gate_count = self.force_gate_count + 1 if self.force_ema < threshold else 0
            self.force_gate_active = self.force_gate_count >= self.cfg["force_gate_patience"]

            # Calculate PINN loss
            pinn = self._pinn_loss(mean_actions, context, force_prediction, batch["dynamics_state"], valid) if self.pinn_weight > 0 else ppo_loss.new_zeros(())

            self.actor_optimizer.zero_grad()

            # PCGrad resolves conflicts between reward optimization and the
            # physical-consistency gradient once PINN warmup gives that loss a
            # nonzero weight. Its projectors require exactly two objectives.
            if self.pinn_weight <= 0.0:
                ppo_loss.backward()
            elif self.pinn_weight >= self.cfg["pinn_loss_weight"]:
                self.actor_optimizer.pc_backward_ppgrad(
                    [ppo_loss, self.pinn_weight * pinn]
                )
            else:
                self.actor_optimizer.pc_backward_pinn(
                    [ppo_loss, self.pinn_weight * pinn]
                )

            nn.utils.clip_grad_norm_(self.ppo_parameters, self.max_grad_norm)
            self.actor_optimizer.step()

            # Recompute the auxiliary graph after PCGrad consumes the PPO
            # graph. The shared optimizer updates encoder and both decoders
            # exactly once from their combined objective.
            aux = self._compute_vae_loss(
                obs_hist_batch=batch["histories"],
                obs_target=batch["next_privileged"], labels=batch["explicit_targets"], valid=valid,
                iteration=iteration,
            )

            # Log metrics
            for name, val in (("value", value), ("surrogate", surrogate), ("base_velo", aux["base_velocity"]),
                              ("ee_position", aux["ee_position"]),
                              ("base_wrench", aux["base_wrench"]), ("ee_force", aux["ee_force"]),
                              ("foot_contact", aux["foot_contact"]),
                              ("foot_height", aux["foot_height"]),
                              ("privileged_force", aux["privileged_force"]), ("privileged_decoder", aux["privileged_decoder"]),
                              ("kl_raw", aux["kl_raw"]), ("kl_reg_loss", aux["kl_reg_loss"]),
                              ("pinn", pinn), ("film_identity", film_identity)):
                metrics[name] += val.detach().item()
            raw_kl_sum += aux["kl_raw"].detach().item()
            updates += 1

            self.spectral_normalization(self.actor_critic, sigma_max=10.0)

        self.storage.clear()

        mean_metrics = {key: value / max(1, updates) for key, value in metrics.items()}
        mean_raw_kl = update_duals_from_mean(
            self.kl_controller, raw_kl_sum, updates, iteration, self.device,
            enabled=self.use_kl_rate_band,
            use_cosine_warmup=self.use_cosine_kl_warmup,
        )
        if mean_raw_kl is not None:
            controller_metrics = self.kl_controller.metrics(
                mean_raw_kl,
                torch.tensor(mean_metrics["kl_reg_loss"], device=self.device),
                iteration, self.use_kl_rate_band,
                self.use_cosine_kl_warmup,
            )
            for name in KLRateBandController.METRIC_NAMES:
                if name not in ("kl_raw", "kl_reg_loss"):
                    mean_metrics[name] = controller_metrics[name].item()
        diagnostics["lr_after_update"] = self.learning_rate
        # Isaac Gym's Python 3.8 predates the dict-union operator.
        return {
            **mean_metrics,
            **diagnostics,
            "force_gate_ema": self.force_ema or 0.0,
            "force_gate_active": float(self.force_gate_active),
            "force_prediction_alpha": self._force_prediction_blend_alpha(),
        }
