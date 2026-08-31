import sys
import numpy as np
import torch
import time
from legged_gym import SIMULATOR

# Base class for RL tasks
class BaseTask():

    def __init__(self, cfg, sim_params, sim_device, headless):
        
        self.render_fps = 50
        self.last_frame_time = 0

        self.device = sim_device
        self.headless = headless

        self.num_envs = cfg.env.num_envs
        self.num_obs = cfg.env.num_observations
        self.num_privileged_obs = cfg.env.num_privileged_obs
        self.num_actions = cfg.env.num_actions
        
        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        # allocate buffers
        self.obs_buf = torch.zeros(self.num_envs, self.num_obs, device=self.device, dtype=torch.float)
        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.int)
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.int)
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.int)
        if self.num_privileged_obs is not None:
            self.privileged_obs_buf = torch.zeros(self.num_envs, self.num_privileged_obs, device=self.device, dtype=torch.float)
        else: 
            self.privileged_obs_buf = None

        self.extras = {}
        
        if SIMULATOR == "genesis":
            from legged_gym.simulator.genesis_simulator import GenesisSimulator
            from legged_gym.simulator.genesis_simulator_pact import GenesisSimulator_PACT

            # HardPACT executes a QP-selected total torque directly. Reuse the
            # mature Go2 PACT Genesis asset/randomization implementation while
            # the new task core owns the controller. Legacy tasks keep the
            # original simulator selection unchanged.
            if getattr(cfg.sim, "use_hard_pact_simulator", False):
                self.simulator = GenesisSimulator_PACT(cfg, sim_params, sim_device, self.headless)
            else:
                self.simulator = GenesisSimulator(cfg, sim_params, sim_device, self.headless)
        elif SIMULATOR == "isaacgym":
            from legged_gym.simulator.isaacgym_simulator import IsaacGymSimulator

            self.simulator = IsaacGymSimulator(cfg, sim_params, sim_device, self.headless)
        elif SIMULATOR == "isaaclab":
            from legged_gym.simulator.isaaclab_simulator import IsaacLabSimulator

            self.simulator = IsaacLabSimulator(cfg, sim_params, sim_device, self.headless)
        elif SIMULATOR == "genesis_pact":
            from legged_gym.simulator.genesis_simulator_pact import GenesisSimulator_PACT

            self.simulator = GenesisSimulator_PACT(cfg, sim_params, sim_device, self.headless)
        elif SIMULATOR == "genesis_pact_pos":
            from legged_gym.simulator.genesis_simulator_pact_pos import GenesisSimulator_PACT_Pos

            self.simulator = GenesisSimulator_PACT_Pos(cfg, sim_params, sim_device, self.headless)
        elif SIMULATOR == "genesis_pact_water":
            from legged_gym.simulator.genesis_simulator_pact_water import GenesisSimulator_PACT_Water

            self.simulator = GenesisSimulator_PACT_Water(cfg, sim_params, sim_device, self.headless)
        elif SIMULATOR == "genesis_pact_nopinn":
            from legged_gym.simulator.genesis_simulator_pact_nopinn import GenesisSimulator_PACT_NoPINN

            self.simulator = GenesisSimulator_PACT_NoPINN(cfg, sim_params, sim_device, self.headless)
        elif SIMULATOR == "genesis_pact_postau":
            from legged_gym.simulator.genesis_simulator_pact_postau import GenesisSimulator_PACT_PosTau

            self.simulator = GenesisSimulator_PACT_PosTau(cfg, sim_params, sim_device, self.headless)
        elif SIMULATOR == "genesis_pact_rl2ac":
            from legged_gym.simulator.genesis_simulator_pact_rl2ac import GenesisSimulator_PACT_RL2AC

            self.simulator = GenesisSimulator_PACT_RL2AC(cfg, sim_params, sim_device, self.headless)
        
        else:
            raise ValueError(f"Unknown simulator: {SIMULATOR}")

    def get_observations(self):
        return self.obs_buf
    
    def get_privileged_observations(self):
        return self.privileged_obs_buf

    def reset_idx(self, env_ids):
        """Reset selected robots"""
        raise NotImplementedError

    def reset(self):
        """ Reset all robots"""
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        obs, privileged_obs, _, _, _ = self.step(torch.zeros(self.num_envs, self.num_actions, device=self.device, requires_grad=False))
        return obs, privileged_obs

    def step(self, actions):
        raise NotImplementedError
