"""Position-only PACT without PINNs, using UniFP PD gains."""
from legged_gym.envs.b1z1.b1z1_pact.ablation_configs import make_b1z1_pact_ablation_configs
from legged_gym.envs.b1z1.b1z1_unifp.b1z1_unifp_config import B1Z1UniFPCfg


_PositionCfg, _PositionCfgPPO = make_b1z1_pact_ablation_configs(6)

class B1Z1PPOPosCfg(_PositionCfg):
    class control(_PositionCfg.control):
        # Only this baseline uses UniFP gains; ablations 6/7 retain PACT gains.
        stiffness = B1Z1UniFPCfg.control.stiffness.copy()
        damping = B1Z1UniFPCfg.control.damping.copy()


class B1Z1PPOPosCfgPPO(_PositionCfgPPO):
    class runner(_PositionCfgPPO.runner):
        experiment_name = "b1z1_ppo_pos"
        run_name = "position_only"
