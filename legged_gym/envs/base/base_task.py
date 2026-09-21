import sys
import numpy as np
import torch
import time
from legged_gym import SIMULATOR

# Base class for RL tasks
class BaseTask():

    def __init__(self, cfg, sim_params, sim_device, headless):
        # Import backend implementations only when a task is instantiated.
        # Eager imports form a cycle through simulator -> terrain -> envs ->
        # BaseTask when a simulator package is imported directly.
        if "isaacgym" in SIMULATOR:
            from legged_gym.simulator.isaacgym_simulator import IsaacGymSimulator
            from legged_gym.simulator.isaacgym_simulator_b1z1 import (
                IsaacGymSimulatorB1Z1UniFP,
                IsaacGymSimulatorB1Z1PACT,
                IsaacGymSimulatorB1Z1PACTPos,
            )
        elif "genesis" in SIMULATOR:
            from legged_gym.simulator.genesis_simulator import GenesisSimulator
            from legged_gym.simulator.genesis_simulator_pact import GenesisSimulator_PACT
            from legged_gym.simulator.genesis_simulator_pact_pos import GenesisSimulator_PACT_Pos
            from legged_gym.simulator.genesis_simulator_pact_water import GenesisSimulator_PACT_Water
            from legged_gym.simulator.genesis_simulator_pact_nopinn import GenesisSimulator_PACT_NoPINN
            from legged_gym.simulator.genesis_simulator_pact_postau import GenesisSimulator_PACT_PosTau
            from legged_gym.simulator.genesis_simulator_pact_rl2ac import GenesisSimulator_PACT_RL2AC
            from legged_gym.simulator.genesis_simulator_kite import GenesisSimulator_KITE
            from legged_gym.simulator.genesis_simulator_kite_depth import GenesisSimulator_KITE_Depth
            from legged_gym.simulator.genesis_simulator_b1z1_unifp import GenesisSimulatorB1Z1UniFP
            from legged_gym.simulator.genesis_simulator_b1z1_pact import GenesisSimulatorB1Z1PACT
            from legged_gym.simulator.genesis_simulator_b1z1_pact_pos import GenesisSimulatorB1Z1PACTPos
        elif "isaaclab" in SIMULATOR:
            from legged_gym.simulator.isaaclab_simulator import IsaacLabSimulator
            from legged_gym.simulator.isaaclab_simulator_b1z1 import (
                IsaacLabSimulatorB1Z1UniFP, IsaacLabSimulatorB1Z1PACT,
                IsaacLabSimulatorB1Z1PACTPos,
            )
        
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
            from legged_gym.simulator import GenesisSimulator
            self.simulator = GenesisSimulator(cfg, sim_params, sim_device, self.headless)
        elif SIMULATOR == "isaacgym":
            from legged_gym.simulator import IsaacGymSimulator
            self.simulator = IsaacGymSimulator(cfg, sim_params, sim_device, self.headless)
        elif SIMULATOR == "isaaclab":
            if getattr(cfg.sim, "use_pact_adapter", False):
                from legged_gym.simulator import IsaacLabSimulator_PACT
                simulator_cls = IsaacLabSimulator_PACT
            else:
                from legged_gym.simulator import IsaacLabSimulator
                simulator_cls = IsaacLabSimulator
            self.simulator = simulator_cls(
                cfg, sim_params, sim_device, self.headless
            )
        elif SIMULATOR == "isaaclab_b1z1_unifp":
            self.simulator = IsaacLabSimulatorB1Z1UniFP(cfg, sim_params, sim_device, self.headless)
        elif SIMULATOR == "isaaclab_b1z1_pact":
            self.simulator = IsaacLabSimulatorB1Z1PACT(cfg, sim_params, sim_device, self.headless)
        elif SIMULATOR == "isaaclab_b1z1_pact_pos":
            self.simulator = IsaacLabSimulatorB1Z1PACTPos(cfg, sim_params, sim_device, self.headless)
        elif SIMULATOR == "genesis_pact":
            from legged_gym.simulator import GenesisSimulator_PACT
            self.simulator = GenesisSimulator_PACT(cfg, sim_params, sim_device, self.headless)
        elif SIMULATOR == "genesis_pact_pos":
            from legged_gym.simulator import GenesisSimulator_PACT_Pos
            self.simulator = GenesisSimulator_PACT_Pos(cfg, sim_params, sim_device, self.headless)
        elif SIMULATOR == "genesis_pact_water":
            from legged_gym.simulator import GenesisSimulator_PACT_Water
            self.simulator = GenesisSimulator_PACT_Water(cfg, sim_params, sim_device, self.headless)
        elif SIMULATOR == "genesis_pact_nopinn":
            from legged_gym.simulator import GenesisSimulator_PACT_NoPINN
            self.simulator = GenesisSimulator_PACT_NoPINN(cfg, sim_params, sim_device, self.headless)
        elif SIMULATOR == "genesis_pact_postau":
            from legged_gym.simulator import GenesisSimulator_PACT_PosTau
            self.simulator = GenesisSimulator_PACT_PosTau(cfg, sim_params, sim_device, self.headless)
        elif SIMULATOR == "genesis_pact_rl2ac":
            from legged_gym.simulator import GenesisSimulator_PACT_RL2AC
            self.simulator = GenesisSimulator_PACT_RL2AC(cfg, sim_params, sim_device, self.headless)
        elif SIMULATOR == "genesis_kite":
            self.simulator = GenesisSimulator_KITE(
                cfg, sim_params, sim_device, self.headless
            )
        elif SIMULATOR == "genesis_kite_depth":
            self.simulator = GenesisSimulator_KITE_Depth(
                cfg, sim_params, sim_device, self.headless
            )
        elif SIMULATOR in ("genesis_b1z1_unifp", "genesis_b1_unifp"):
            self.simulator = GenesisSimulatorB1Z1UniFP(
                cfg, sim_params, sim_device, self.headless
            )
        elif SIMULATOR == "genesis_b1z1_pact":
            self.simulator = GenesisSimulatorB1Z1PACT(
                cfg, sim_params, sim_device, self.headless
            )
        elif SIMULATOR == "genesis_b1z1_pact_pos":
            self.simulator = GenesisSimulatorB1Z1PACTPos(
                cfg, sim_params, sim_device, self.headless
            )
        elif SIMULATOR == "isaacgym_b1z1_unifp":
            self.simulator = IsaacGymSimulatorB1Z1UniFP(
                cfg, sim_params, sim_device, self.headless
            )
        elif SIMULATOR == "isaacgym_b1z1_pact_pos":
            self.simulator = IsaacGymSimulatorB1Z1PACTPos(
                cfg, sim_params, sim_device, self.headless
            )
        elif SIMULATOR == "isaacgym_b1z1_pact":
            self.simulator = IsaacGymSimulatorB1Z1PACT(
                cfg, sim_params, sim_device, self.headless
            )
        
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
