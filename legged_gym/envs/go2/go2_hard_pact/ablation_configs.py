"""Feature-only ablation profiles; environment behavior remains shared."""

from .go2_hard_pact_config import GO2HardPACTCfg, GO2HardPACTCfgPPO


class _NoBARD:
    enabled = False
    required = False


class GO2HardPACTBaselineCfg(GO2HardPACTCfg):
    class features(GO2HardPACTCfg.features):
        use_bard_inverse_loss = False
        use_bard_rollout_loss = False
        use_qp = False
        differentiate_qp = False
        use_soft_projection_penalty = False

    class bard(GO2HardPACTCfg.bard):
        enabled = False
        required = False

    class qp(GO2HardPACTCfg.qp):
        enabled = False


class GO2HardPACTSoftOnlyCfg(GO2HardPACTCfg):
    class features(GO2HardPACTCfg.features):
        use_qp = False
        differentiate_qp = False

    class qp(GO2HardPACTCfg.qp):
        enabled = False


class GO2HardPACTHardOnlyCfg(GO2HardPACTCfg):
    class features(GO2HardPACTCfg.features):
        use_bard_inverse_loss = False
        use_bard_rollout_loss = False
        use_qp = True


class GO2HardPACTFullCfg(GO2HardPACTCfg):
    pass


class GO2HardPACTStopGradientQPCfg(GO2HardPACTCfg):
    class features(GO2HardPACTCfg.features):
        differentiate_qp = False
        stop_gradient_qp = True


class GO2HardPACTSoftPenaltyCfg(GO2HardPACTCfg):
    class features(GO2HardPACTCfg.features):
        use_bard_inverse_loss = False
        use_bard_rollout_loss = False
        use_qp = False
        differentiate_qp = False
        use_soft_projection_penalty = True

    class bard(GO2HardPACTCfg.bard):
        enabled = False
        required = False

    class qp(GO2HardPACTCfg.qp):
        enabled = False


class GO2HardPACTInverseOnlyCfg(GO2HardPACTCfg):
    class features(GO2HardPACTCfg.features):
        use_bard_inverse_loss = True
        use_bard_rollout_loss = False
        use_qp = False
        differentiate_qp = False

    class qp(GO2HardPACTCfg.qp):
        enabled = False


class GO2HardPACTRolloutOnlyCfg(GO2HardPACTCfg):
    class features(GO2HardPACTCfg.features):
        use_bard_inverse_loss = False
        use_bard_rollout_loss = True
        use_qp = False
        differentiate_qp = False

    class qp(GO2HardPACTCfg.qp):
        enabled = False


def _ppo_profile(name, inverse, rollout, qp):
    class Profile(GO2HardPACTCfgPPO):
        class algorithm(GO2HardPACTCfgPPO.algorithm):
            use_bard_inverse_loss = inverse
            use_bard_rollout_loss = rollout
            use_qp = qp

        class runner(GO2HardPACTCfgPPO.runner):
            run_name = name

    Profile.__name__ = f"GO2HardPACT{name.title().replace('_', '')}CfgPPO"
    return Profile


GO2HardPACTBaselineCfgPPO = _ppo_profile("baseline", False, False, False)
GO2HardPACTSoftOnlyCfgPPO = _ppo_profile("soft_only", True, True, False)
GO2HardPACTHardOnlyCfgPPO = _ppo_profile("hard_only", False, False, True)
GO2HardPACTFullCfgPPO = _ppo_profile("full", True, True, True)
GO2HardPACTStopGradientQPCfgPPO = _ppo_profile("stop_gradient_qp", True, True, True)
GO2HardPACTSoftPenaltyCfgPPO = _ppo_profile("soft_penalty", False, False, False)
GO2HardPACTInverseOnlyCfgPPO = _ppo_profile("inverse_only", True, False, False)
GO2HardPACTRolloutOnlyCfgPPO = _ppo_profile("rollout_only", False, True, False)

