"""Liquid evaluation retains the legacy Go2 PACT checkpoint architecture."""
from legged_gym.envs.base.liquid_payload_config import liquid_evaluation_config
from legged_gym.envs.go2.go2_pact.go2_pact_config import GO2PACTCfg, GO2PACTCfgPPO


class GO2PACTWaterCfg(liquid_evaluation_config(GO2PACTCfg,.060)):
    # Collision top is -0.020 + 0.150/2 = .055 m; 5 mm clearance.
    pass


class GO2PACTWaterCfgPPO(GO2PACTCfgPPO):
    pass
