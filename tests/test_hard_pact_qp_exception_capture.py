"""Real exception handling plus current-training-config cuPIQP reproduction."""
from dataclasses import replace
from unittest.mock import patch
import os

import pytest
import torch

from test_go2_hard_pact_qp import make_qp, qp_data
from rsl_rl.algorithms.hard_pact_qp import HardPACTQPConfig, HardPACTDifferentiableQP
from rsl_rl.algorithms.hard_pact_qp_capture import replay_snapshot
from legged_gym.envs.go2.go2_hard_pact.go2_hard_pact_config import GO2HardPACTCfg, GO2HardPACTCfgPPO


@pytest.mark.parametrize('error_type', [RuntimeError, OSError])
def test_exception_capture_is_bounded_and_replays(tmp_path, error_type):
    qp = make_qp(exception_capture_enabled=True, exception_capture_dir=str(tmp_path))
    with patch('rsl_rl.algorithms.hard_pact_qp.QPFunction', side_effect=error_type('forced backend failure')):
        with pytest.warns(UserWarning, match='forced backend failure'):
            first = qp.solve(**qp_data())
        qp.solve(**qp_data())
    assert not first.differentiated_mask.any()
    files = list(tmp_path.glob('*.pt'))
    assert len(files) == 1
    snapshot = torch.load(files[0], weights_only=True)
    assert 'forced backend failure' in snapshot['traceback']
    assert snapshot['tensors']['Q'].shape == (2, 24, 24)
    result = replay_snapshot(snapshot, 'cpu')
    assert torch.isfinite(result).all()


def test_disabled_capture_has_no_disk_output(tmp_path):
    qp = make_qp(exception_capture_enabled=False, exception_capture_dir=str(tmp_path))
    with patch('rsl_rl.algorithms.hard_pact_qp.QPFunction', side_effect=ValueError('forced')):
        qp.solve(**qp_data())
    assert not list(tmp_path.iterdir())


@pytest.mark.skipif(not torch.cuda.is_available(), reason='real cuPIQP requires CUDA')
@pytest.mark.parametrize('differentiable', [False, True])
def test_live_training_config_cupiqp(differentiable, tmp_path):
    # Use the current config verbatim except the user's CLI solver selection
    # and the diagnostic output directory. Size is explicit and can reproduce
    # full rollout chunks via HARDPACT_QP_TEST_BATCH=4096.
    cfg = HardPACTQPConfig(**GO2HardPACTCfgPPO.algorithm.hard_pact_qp)
    cfg = replace(cfg, qp_solver='cupiqp', exception_capture_dir=str(tmp_path))
    batch = int(os.environ.get('HARDPACT_QP_TEST_BATCH', '8'))
    qp = HardPACTDifferentiableQP(cfg, torch.full((12,), 23.5),
                                 torch.full((12,), -2.), torch.full((12,), 2.),
                                 torch.full((12,), 30.))
    data = {k: v.cuda() for k, v in qp_data(batch, torch.float32).items()}
    data['dt'].fill_(GO2HardPACTCfg.control.dt / GO2HardPACTCfg.control.decimation)
    if differentiable:
        data['tau_nom'].requires_grad_(True)
    # Repeated calls exercise cached solver state. These synthetic mechanics
    # deliberately do not claim to replace real BARD/simulator coverage.
    for _ in range(2):
        result = qp.solve(differentiable=differentiable, **data)
        assert torch.isfinite(result.tau_safe).all()
        for key, value in result.diagnostics.items():
            if key.endswith('solver_exception'):
                assert not value.any(), key
        if differentiable:
            result.tau_safe.square().sum().backward()
            assert torch.isfinite(data['tau_nom'].grad).all()
    assert not list(tmp_path.glob('*.pt'))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='real cuPIQP requires CUDA')
def test_varying_fallback_batches_do_not_accumulate_solvers(tmp_path):
    cfg = replace(HardPACTQPConfig(**GO2HardPACTCfgPPO.algorithm.hard_pact_qp),
                  qp_solver='cupiqp', exception_capture_dir=str(tmp_path))
    qp = HardPACTDifferentiableQP(cfg, torch.full((12,), 23.5),
                                 torch.full((12,), -2.), torch.full((12,), 2.),
                                 torch.full((12,), 30.))
    backend = qp._backend_instances['cupiqp']
    with torch.inference_mode():
        for batch in range(2, 18):
            for relaxed, elastic in ((False, False), (True, False), (True, True)):
                data = {k: v.cuda() for k, v in qp_data(batch, torch.float32).items()}
                _, _, diagnostic = qp._solve_stage(data, relaxed, elastic=elastic)
                assert not diagnostic['solver_exception'].any()
            assert len(backend._rollout_cache) <= 3
    assert not list(tmp_path.glob('*.pt'))
