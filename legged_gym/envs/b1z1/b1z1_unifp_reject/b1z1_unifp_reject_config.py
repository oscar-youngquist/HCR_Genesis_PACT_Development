from legged_gym.envs.b1z1.b1z1_unifp_original.b1z1_unifp_original_config import (
    B1Z1UniFPOriginalCfg, B1Z1UniFPOriginalCfgPPO,
)


class B1Z1UniFPRejectCfg(B1Z1UniFPOriginalCfg):
    def __init__(self):
        super().__init__()
        self.commands.use_external_impedance_compensation = True
        self.commands.compensate_ee_external_force = True
        self.commands.compensate_base_external_force = True


class B1Z1UniFPRejectCfgPPO(B1Z1UniFPOriginalCfgPPO):
    def __init__(self):
        super().__init__()
        self.runner.experiment_name = "b1z1_unifp_reject"
        self.runner.run_name = "estimated_force_rejection"
