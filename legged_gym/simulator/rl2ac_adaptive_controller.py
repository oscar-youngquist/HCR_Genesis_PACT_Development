import torch


class RL2ACAdaptiveCtrl:
    def __init__(self, num_envs, config, device="cuda", dtype=torch.float32):
        self.device = device
        self.dtype = dtype

        self.B = num_envs
        self.J = 12

        # Scalars (broadcasted)
        # self.alpha = 50.0
        # self.kappa = 1.2
        self.eta = 0.01
        # self.lambda_0 = 3.0
        # self.k_0 = 20.0

        self.alpha = config.rl2ac.alpha
        self.kappa = config.rl2ac.kappa
        self.lambda_0 = config.rl2ac.lambda_0
        self.k_0 = config.rl2ac.k_0

        print(f"RL2AC params: alpha={self.alpha}, kappa={self.kappa}, lambda_0={self.lambda_0}, k_0={self.k_0}")

        # State flags
        self.use_proactive_ctrl = True

        # Joint-space vectors: [B, J]
        self.phi = torch.zeros(self.B, self.J, device=device, dtype=dtype)
        self.s = torch.zeros_like(self.phi)
        self.tau = torch.zeros_like(self.phi)
        self.epsilon = torch.zeros_like(self.phi)


        self.phi_diag = torch.zeros(self.B, self.J, self.J, device=device, dtype=dtype)
        self.s_diag = torch.zeros(self.B, self.J, self.J, device=device, dtype=dtype)
        self.epsilon_diag = torch.zeros(self.B, self.J, self.J, device=device, dtype=dtype)

        self.q = torch.zeros_like(self.phi)
        self.qdot = torch.zeros_like(self.phi)

        self.q_ref = torch.zeros_like(self.phi)
        self.q_des = torch.zeros_like(self.phi)

        self.tau_des = torch.zeros_like(self.phi)
        self.tauDes_old = torch.zeros_like(self.phi)

        self.comp_old = torch.zeros_like(self.phi)
        self.comp = torch.zeros_like(self.phi)

        # Adaptive matrices: [B, J, J]
        self.Gamma = torch.eye(self.J, device=device, dtype=dtype).repeat(self.B, 1, 1)
        self.K = torch.zeros(self.B, self.J, self.J, device=device, dtype=dtype)

        # Numerical stability constants
        self.gamma_max_norm = 1e3
        self.phi_norm_max = 2.0
        self.min_lambda = 0.0
        self.dt_min = 1e-5

    def reset_adaptive_controller(self, env_ids):
        self.Gamma[env_ids] = torch.eye(self.J, device=self.device, dtype=self.dtype).unsqueeze(0)
        self.K[env_ids] = 0.0
        self.comp_old[env_ids] = 0.0
        self.comp[env_ids] = 0.0   
   
        self.phi[env_ids] = 0.0
        self.s[env_ids] = 0.0
        self.epsilon[env_ids] = 0.0
        
        self.phi_diag[env_ids] = 0.0
        self.s_diag[env_ids] = 0.0
        self.epsilon_diag[env_ids] = 0.0
    
    # ------------------------------------------------------------------
    # State update (called every sim step)
    # ------------------------------------------------------------------

    def update_state(
        self,
        qpos,           # [B, nq]
        qvel,           # [B, nv]
        qfrc_actuator,  # [B, nv]
    ):
        qj = qpos
        qdj = qvel

        if self.use_proactive_ctrl:
            self.phi = self.q_des - self.q_ref
            self.s = qdj - self.alpha * (self.q_ref - qj)
        else:
            self.phi = self.q_des - qj
            self.s = qdj - self.alpha * (self.q_des - qj)

        # Sliding variable
        # self.s = qdj - self.alpha * (self.q_des - qj)


        # Torque tracking error
        self.tau = qfrc_actuator
        self.epsilon = self.tau - (self.tau_des + self.comp)

        # Log states
        self.q.copy_(qj)
        self.qdot.copy_(qdj)

        # copy over to the diagonal matricies
        for i in range (0, self.J):
            self.phi_diag[:,i,i] = self.phi[:,i]
            self.s_diag[:,i,i] = self.s[:,i]
            self.epsilon_diag[:,i,i] = self.epsilon[:,i]

        # ---- Stability: clamp ||phi||
        # phi_norm = torch.norm(self.phi, dim=1, keepdim=True).clamp(min=1e-6)
        # scale = torch.clamp(self.phi_norm_max / phi_norm, max=1.0)
        # self.phi.mul_(scale)

    # ------------------------------------------------------------------
    # Command update
    # ------------------------------------------------------------------

    def update_cmd(self, q_ref, q_des, tau_cmd):
        self.q_ref.copy_(q_ref)
        self.q_des.copy_(q_des)

        self.tauDes_old.copy_(self.tau_des)
        self.tau_des.copy_(tau_cmd)

    # ------------------------------------------------------------------
    # Adaptive compensation update
    # ------------------------------------------------------------------

    def update_compensation(self, dt):
        self._update_forgetting_factor()
        self._update_gamma(dt)
        self._update_K(dt)

        self.comp_old.copy_(self.comp)
        self.comp = torch.einsum("bij,bj->bi", self.K, self.phi)
        # self.comp = torch.bmm(self.K, self.phi.unsqueeze(-1)).squeeze(-1)

        # print(torch.norm(self.K[0]))
        # print(torch.norm(self.phi[0]))
        # print(torch.norm(self.comp[0]))
        # print("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-")

        return self.comp

    # ------------------------------------------------------------------
    # Internal updates
    # ------------------------------------------------------------------

    def _update_forgetting_factor(self):
        gamma_norm = torch.norm(self.Gamma, dim=(1, 2))
        # print(gamma_norm.shape)
        lambda_val = self.lambda_0 * (1.0 - (gamma_norm / self.k_0))
        self.lambda_val = lambda_val

    def _update_gamma(self, dt):
        # Elementwise equivalent of: Γ φ φᵀ Γ
        phi_outer = self.phi.unsqueeze(2) * self.phi.unsqueeze(1)  # [B,J,J]

        # dGamma = self.lambda_val[:, None, None] * self.Gamma
        # dGamma -= ((self.Gamma @ self.phi_diag) @ self.phi_diag) @ self.Gamma
        
        dGamma = (
            self.lambda_val[:, None, None] * self.Gamma
            - torch.bmm(self.Gamma, torch.bmm(phi_outer, self.Gamma))
        )

        self.Gamma += dt * dGamma
        
        self.Gamma = 0.5 * (self.Gamma + self.Gamma.transpose(-1, -2))
        self.Gamma = self.Gamma + 1e-6 * torch.eye(self.Gamma.shape[-1],
                                                   device=self.Gamma.device,
                                                   dtype=self.Gamma.dtype,
                                                   ).unsqueeze(0)

        # ---- Stability: norm clamp + symmetrize
        # gamma_norm = torch.norm(self.Gamma, dim=(1, 2), keepdim=True).clamp(min=1e-6)
        # scale = torch.clamp(self.gamma_max_norm / gamma_norm, max=1.0)
        # self.Gamma.mul_(scale)

        # self.Gamma = 0.5 * (self.Gamma + self.Gamma.transpose(1, 2))

    def _update_K(self, dt):
        # Equivalent to: -Γ φ (s + κ ε)ᵀ
        rhs = self.s + self.kappa * self.epsilon
        dK = -torch.einsum("bij,bj,bk->bik", self.Gamma, self.phi, rhs)
        dK -= self.eta * self.K
        # dK = -self.Gamma @ self.phi_diag @ (self.s_diag + self.kappa * self.epsilon_diag)
        # dK -= self.eta * self.K

        self.K += dt * dK
