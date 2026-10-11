"""Matched initial force exposure and nonzero rejection hold, without simulation."""
import pytest

from legged_gym.envs.b1z1.rejection_curriculum import RejectionCurriculum
from legged_gym.envs.b1z1.b1z1_unifp.b1z1_unifp_config import B1Z1UniFPCfg
from legged_gym.envs.b1z1.b1z1_unifp_original.b1z1_unifp_original_config import B1Z1UniFPOriginalCfg
from legged_gym.envs.b1z1.b1z1_unifp_reject.b1z1_unifp_reject_config import B1Z1UniFPRejectCfg
from legged_gym.envs.b1z1.b1z1_pact.b1z1_pact_config import B1Z1PACTCfg
from legged_gym.envs.b1z1.b1z1_pact_pos.b1z1_pact_pos_config import B1Z1PACTPosCfg
from legged_gym.envs.b1z1.b1z1_ppo_pos.b1z1_ppo_pos_config import B1Z1PPOPosCfg
from legged_gym.envs.b1z1.b1z1_pact.ablation_configs import make_b1z1_pact_ablation_configs


@pytest.mark.parametrize("cfg_type", [B1Z1UniFPCfg, B1Z1UniFPOriginalCfg,
    B1Z1UniFPRejectCfg, B1Z1PACTCfg, B1Z1PACTPosCfg, B1Z1PPOPosCfg,
    *[make_b1z1_pact_ablation_configs(i)[0] for i in range(4, 12)]])
@pytest.mark.parametrize("metrics", [(None, None, None), (0., 0., 1000.)])
def test_schedule_and_resume(cfg_type, metrics):
    cfg = cfg_type().commands
    assert cfg.use_shared_rejection_curriculum
    c = RejectionCurriculum(cfg, "cpu")
    assert c.beta == .2 and c.external_scale(0) == .25
    for iteration, beta, external, stage in (
        (0, .2, .25, 1), (7999, .2, .25, 1), (8000, .2, .25, 2),
        (8500, .6, .25, 2), (9000, 1., .25, 3), (13000, 1., .625, 3),
        (17000, 1., 1., 3), (18000, 1., 1., 3),
    ):
        c.update(iteration, *metrics)
        assert c.beta == pytest.approx(beta)
        assert c.command_scale(iteration) == pytest.approx(beta)
        assert c.external_scale(iteration) == pytest.approx(external)
        assert c.metrics(iteration)["Rejection/stage"] == stage
        resumed = RejectionCurriculum(cfg, "cpu")
        resumed.load_state_dict(c.state_dict())
        assert resumed.state_dict() == c.state_dict()
        c = resumed  # Nonzero hold must not be misinterpreted as a latched ramp.
