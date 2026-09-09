"""Production diagnostics stay inexpensive without disabling safety/logging."""
import pytest

from legged_gym.envs.go2.go2_hard_pact.go2_hard_pact_config import GO2HardPACTCfgPPO
from legged_gym.envs.go2.go2_hard_pact_pos.go2_hard_pact_pos_config import GO2HardPACTPosCfgPPO
from rsl_rl.algorithms.hard_pact_qp import HardPACTQPConfig


@pytest.mark.parametrize("cfg", [GO2HardPACTCfgPPO, GO2HardPACTPosCfgPPO])
def test_optional_expensive_diagnostics_off_basic_logging_retained(cfg):
    assert not cfg.algorithm.ppo_latent_diagnostics_enabled
    assert not cfg.algorithm.force_decoder_diagnostics_enabled
    assert cfg.runner.console_iteration
    assert cfg.runner.console_reward_terms
    assert cfg.runner.console_pinn_timing
    assert cfg.runner.console_qp_timing
    assert not cfg.runner.console_model_summary
    assert not cfg.runner.console_detailed_losses


def test_qp_minimal_retains_failure_capture_and_sampling_debugging():
    algorithm = GO2HardPACTCfgPPO.algorithm
    qp = HardPACTQPConfig(**algorithm.hard_pact_qp)
    assert qp.diagnostics_level == "minimal"
    assert qp.full_audit_period == 1000
    assert qp.full_audit_sample_size == 8
    assert qp.exception_capture_enabled and qp.exception_capture_limit == 1
    assert algorithm.ppo_qp_sampling_logging_enabled
    assert not algorithm.pcgrad_diagnostics_enabled
    assert not algorithm.profile_bard_timing
    # These safety/solver controls are not diagnostic cost-cutting switches.
    assert qp.check_q_spd and qp.check_equality_rank
    assert qp.rollout_feasibility_tolerance == 1e-3
    assert qp.ppo_feasibility_tolerance == 1e-3
    assert qp.ppo_duality_gap_policy == "require"
