"""Liquid evaluation retains the legacy Go1 PACT checkpoint architecture."""
from legged_gym.envs.base.liquid_payload_config import liquid_evaluation_config
from legged_gym.envs.go1.go1_pact.go1_pact_config import GO1PACTCfg, GO1PACTCfgPPO


class GO1PACTWaterCfg(liquid_evaluation_config(GO1PACTCfg,.057)):
    # Original Go1 tank mount: half the .114 m torso height.
    pass


class GO1PACTWaterCfgPPO(GO1PACTCfgPPO):
    class runner(GO1PACTCfgPPO.runner):
        load_run = 'May17_21-20-40_pact_100hz_spec_smartcurr_original_jointrand_hightau'
