"""Boundary and resume checks for optional stepped external disturbances."""
from copy import deepcopy

import pytest

from legged_gym.envs.b1z1.rejection_curriculum import RejectionCurriculum
from legged_gym.envs.b1z1.b1z1_unifp.b1z1_unifp_config import B1Z1UniFPCfg
from legged_gym.envs.b1z1.b1z1_unifp_reject.b1z1_unifp_reject_config import B1Z1UniFPRejectCfg
from legged_gym.envs.b1z1.b1z1_pact.b1z1_pact_config import B1Z1PACTCfg
from legged_gym.envs.b1z1.b1z1_pact_pos.b1z1_pact_pos_config import B1Z1PACTPosCfg
from legged_gym.envs.b1z1.b1z1_pact.ablation_configs import make_b1z1_pact_ablation_configs


@pytest.mark.parametrize("cls", [B1Z1UniFPCfg, B1Z1UniFPRejectCfg, B1Z1PACTCfg,
    B1Z1PACTPosCfg, *[make_b1z1_pact_ablation_configs(i)[0] for i in range(4, 12)]])
@pytest.mark.parametrize("stepped", [False, True])
def test_steps_and_continuous_schedule(cls, stepped):
    cfg = deepcopy(cls().commands)
    assert cfg.reject_external_step_enabled is False
    assert cfg.reject_external_step_hold_iterations == 3000
    cfg.reject_external_step_enabled = stepped
    cfg.reject_nominal_tracking_iterations = 0
    cfg.force_curriculum_gate_start_iteration = 2
    cfg.force_curriculum_latest_start_iteration = 2
    cfg.reject_compensation_ramp_iterations = 2
    c = RejectionCurriculum(cfg, "cpu")
    c.update(0)
    assert c.external_scale(0) == .25
    c.update(2)
    c.update(3)
    assert c.external_scale(3) == .25
    c.update(4)
    assert c.beta == 1. and c.full_iteration == 4
    for elapsed, level in [(0, .5), (2999, .5), (3000, .75), (5999, .75), (6000, 1.), (9000, 1.)]:
        iteration = 4 + elapsed
        c.update(iteration)
        expected = level if stepped else .25 + .75 * min(elapsed / cfg.reject_external_ramp_iterations, 1.)
        assert c.external_scale(iteration) == pytest.approx(expected)
        assert c.metrics(iteration)["ForceCurriculum/external_scale"] == pytest.approx(expected)
        resumed = RejectionCurriculum(cfg, "cpu")
        resumed.load_state_dict(c.state_dict())
        assert resumed.external_scale(iteration) == pytest.approx(expected)
        c = resumed


@pytest.mark.parametrize("hold", [0, -1, 1.5, True])
def test_invalid_step_duration(hold):
    cfg = deepcopy(B1Z1PACTCfg().commands)
    cfg.reject_external_step_enabled = True
    cfg.reject_external_step_hold_iterations = hold
    with pytest.raises(ValueError, match="positive integer"):
        RejectionCurriculum(cfg, "cpu")
