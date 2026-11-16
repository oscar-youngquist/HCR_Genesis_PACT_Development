"""Simulator-independent export of the pre-compliance 81-D coupled FiLM V02 policy.

Only tensors cross the scripted API. History is caller-owned, oldest first;
getters never implicitly advance it. No training runner or dynamics is exported.
"""
from copy import deepcopy
import torch
from torch import nn


class LegacyPACTDeployment(nn.Module):
    def __init__(self, actor, settings):
        super().__init__()
        self.encoder = deepcopy(actor.context_encoder.trunk)
        self.mean_hidden = deepcopy(actor.context_encoder.mean_hidden)
        self.mean = deepcopy(actor.context_encoder.latent_mean)
        self.explicit = deepcopy(actor.explicit_decoder.network)
        self.force = deepcopy(actor.physics_decoder.force)
        self.grf = deepcopy(actor.physics_decoder.grf)
        self.film = deepcopy(actor.film.network)
        self.actor = deepcopy(actor.actor_trunk)
        self.position = deepcopy(actor.position_head)
        self.torque = deepcopy(actor.torque_head)
        self.base_soft = float(settings['base_soft'])
        self.ee_soft = float(settings['ee_soft'])
        self.clip_obs = float(settings['clip_obs'])
        self.clip_action = float(settings['clip_action'])
        self.position_scale = float(settings['position_scale'])
        self.angular_scale = float(settings['angular_scale'])
        self.q_scale = float(settings['q_scale'])
        self.qd_scale = float(settings['qd_scale'])
        self.dt = float(settings['dt'])
        self.cycle = float(settings['cycle'])
        self.grf_torque_scale = float(actor.physics_decoder.torque_scale)
        for name in ('q_default', 'kp', 'kd', 'ff_scale', 'command_scale', 'walking_threshold'):
            self.register_buffer(name, torch.tensor(settings[name], dtype=torch.float32))

    def _context(self, history: torch.Tensor):
        features = self.encoder(history)
        z = self.mean(self.mean_hidden(features))  # Posterior mean, never a latent sample.
        raw = self.explicit(features)
        explicit = torch.cat((raw[:, :6], .01 + .98*raw[:, 6:10].sigmoid(), raw[:, 10:14]), -1)
        force = self.force(torch.cat((z, explicit), -1))
        return z, explicit, force

    def forward(self, obs: torch.Tensor, history: torch.Tensor, tracking_history: torch.Tensor):
        """81-D normalized obs, 2592-D history, previous [N,5,6] errors -> action,new errors."""
        torch._assert(obs.dim() == 2 and obs.size(1) == 81, 'Expected [N,81] legacy observation')
        torch._assert(history.dim() == 2 and history.size(1) == 2592, 'Expected [N,2592] observation history')
        torch._assert(tracking_history.dim() == 3 and tracking_history.size(1) == 5 and tracking_history.size(2) == 6, 'Expected [N,5,6] previous tracking history')
        torch._assert(obs.size(0) == history.size(0) and obs.size(0) == tracking_history.size(0), 'Batch mismatch')
        obs = obs.clamp(-self.clip_obs, self.clip_obs)
        z, explicit, forces = self._context(history)
        error = obs[:, -6:] - explicit[:, :6]
        updated = torch.cat((tracking_history[:, 1:], error.unsqueeze(1)), 1)
        base, ee = updated[:, :, :3], updated[:, :, 3:]
        bm, em = base.abs().sum(-1, keepdim=True), ee.abs().sum(-1, keepdim=True)
        condition = torch.cat((base/(bm+self.base_soft), ee/(em+self.ee_soft), bm.tanh(), em.tanh()), -1).flatten(1)
        gamma, beta = self.film(condition).chunk(2, -1)
        modulated = z*(1+gamma)+beta
        inputs = torch.cat((obs, modulated, explicit[:, :6], forces, explicit[:, 6:]), -1)
        features = self.actor(inputs)
        return torch.cat((self.position(features), self.torque(features)), -1), updated

    @torch.jit.export
    def initial_state(self, batch_size: int):
        """Move the module to the desired device before allocating state."""
        return (self.q_default.new_zeros((batch_size, 2592)),
                self.q_default.new_zeros((batch_size, 5, 6)),
                self.q_default.new_zeros((batch_size, 34)))

    @torch.jit.export
    def build_observation(self, roll_pitch: torch.Tensor, omega_body: torch.Tensor,
                          q: torch.Tensor, qd: torch.Tensor, phase: torch.Tensor,
                          previous_issued_action: torch.Tensor, commands: torch.Tensor):
        """SI inputs; roll/pitch (NOT gravity), body angular velocity, spherical EE commands."""
        torch._assert(q.size(1) == 19 and qd.size(1) == 19, 'Use all 19 joints in manifest order')
        torch._assert(commands.size(1) == 6, 'Commands: vx,vy,yaw_rate,radius,elevation,azimuth')
        angle = phase.reshape(-1, 1) * (2*torch.pi)
        return torch.cat((roll_pitch, omega_body*self.angular_scale,
            (q[:, :17]-self.q_default[:17])*self.q_scale, qd[:, :17]*self.qd_scale,
            angle.sin(), angle.cos(), previous_issued_action,
            commands*self.command_scale), -1)

    @torch.jit.export
    def step(self, observation: torch.Tensor, history: torch.Tensor,
             tracking_history: torch.Tensor, is_init: torch.Tensor):
        """Append ONE boundary observation and advance FiLM ONCE; no duplicate calls per step."""
        reset = is_init.reshape(-1, 1).to(torch.bool)
        history = torch.where(reset, torch.zeros_like(history), history)
        tracking_history = torch.where(reset.unsqueeze(-1), torch.zeros_like(tracking_history), tracking_history)
        # Native history stores pre-clipped observations. Actor input is clipped separately.
        history = torch.cat((history[:, 81:], observation), -1)
        action, tracking_history = self.forward(observation, history, tracking_history)
        issued = action.clamp(-self.clip_action, self.clip_action)
        target, ff = self.action_targets(issued)
        return action, issued, target, ff, history, tracking_history

    @torch.jit.export
    def action_targets(self, issued: torch.Tensor):
        torch._assert(issued.size(1) == 34, 'Expected 17 position + 17 feedforward actions')
        issued = issued.clamp(-self.clip_action, self.clip_action)
        zeros = issued.new_zeros((issued.size(0), 2))
        target = self.q_default + torch.cat((issued[:, :17]*self.position_scale, zeros), -1)
        ff = torch.cat((issued[:, 17:]*self.ff_scale, zeros), -1)
        return target, ff

    @torch.jit.export
    def advance_phase(self, phase: torch.Tensor, base_commands: torch.Tensor):
        walking = (base_commands.abs() > self.walking_threshold).any(-1)
        return torch.where(walking, torch.remainder(phase + self.dt/self.cycle, 1.), torch.zeros_like(phase))

    @torch.jit.export
    def apply_delay(self, issued: torch.Tensor, queue: torch.Tensor,
                    delay_steps: torch.Tensor, is_init: torch.Tensor):
        """Optional explicit action FIFO [N,L,34], newest first; never used as observation history."""
        torch._assert(queue.dim() == 3 and queue.size(2) == 34, 'Expected action FIFO [N,L,34]')
        torch._assert(bool((delay_steps >= 0).all()) and bool((delay_steps < queue.size(1)).all()), 'Delay outside queue')
        queue = torch.where(is_init.reshape(-1, 1, 1), torch.zeros_like(queue), queue)
        queue = torch.cat((issued.unsqueeze(1), queue[:, :-1]), 1)
        executed = queue.gather(1, delay_steps.reshape(-1, 1, 1).expand(-1, 1, 34)).squeeze(1)
        return executed, queue

    @torch.jit.export
    def controller(self, executed_action: torch.Tensor, q: torch.Tensor, qd: torch.Tensor,
                   kp_scale: torch.Tensor, kd_scale: torch.Tensor, motor_strength: torch.Tensor,
                   feedback_weight: torch.Tensor, feedforward_weight: torch.Tensor,
                   execution_limits: torch.Tensor):
        """Per-substep effective PD/FF [Nm], requested total, clipped total; limits supplied explicitly."""
        torch._assert(bool((execution_limits >= 0).all()) and bool(torch.isfinite(execution_limits).all()), 'Execution limits must be finite and nonnegative')
        target, ff = self.action_targets(executed_action)
        pd = (self.kp*kp_scale*(target-q) - self.kd*kd_scale*qd)*feedback_weight*motor_strength
        ff = ff*feedforward_weight*motor_strength
        requested = pd+ff
        clipped = torch.maximum(torch.minimum(requested, execution_limits), -execution_limits)
        return pd, ff, requested, clipped

    @torch.jit.export
    def estimate(self, history: torch.Tensor):
        """Unmodulated latent, 14-D explicit estimates, 9-D forces; observation-scaled units."""
        return self._context(history)

    @torch.jit.export
    def predict_grf(self, history: torch.Tensor, nominal_torque: torch.Tensor):
        """Optional diagnostic; 19 physical joint torques -> 12 normalized GRF components."""
        torch._assert(nominal_torque.size(1) == 19, 'GRF requires all 19 joint torques')
        z, explicit, _ = self._context(history)
        return self.grf(torch.cat((z, explicit, nominal_torque/self.grf_torque_scale), -1))
