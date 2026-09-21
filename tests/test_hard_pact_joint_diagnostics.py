"""Physical original-limit candidate metrics, independent of solver acceptance."""
import torch

from rsl_rl.algorithms.hard_pact_qp_diagnostics import QPIterationDiagnostics
from test_hard_pact_reduced_qp import inputs, solver


def record(aggregate, data, a, accepted, stage="primary"):
    aggregate.joint_candidate(stage, data, a, accepted,
        torch.full((12,), -1.), torch.ones(12), torch.ones(12),
        torch.full((12,), 2.), 1.)


def test_separate_families_recovery_original_bounds_and_nonfinite():
    d = inputs(4)
    d["dt"].fill_(0.1)
    d["joint_position"][0, 0] = 1.5
    d["joint_velocity"][1, 1] = 1.5
    a = torch.zeros(4,18, dtype=torch.float64)
    a[2, 8] = 3.
    a[3, 6] = float("nan")
    agg = QPIterationDiagnostics()
    record(agg, d, a, torch.tensor([True, True, True, False]), "recovery")
    out = agg.finalize(a)
    prefix = "model_candidate/recovery/accepted/"
    for family, magnitude in (("position_rad", .5), ("velocity_rad_s", .5), ("acceleration_rad_s2", 1.)):
        assert abs(out[prefix+family+"/max"].item() - magnitude) < 1e-12
        assert torch.isclose(out[prefix+family+"/mean"], torch.tensor(magnitude/36))
        assert torch.isclose(out[prefix+family+"/coordinate_fraction"], torch.tensor(1/36))
        assert torch.isclose(out[prefix+family+"/any_joint_fraction"], torch.tensor(1/3))
        assert out[prefix+family+"/count"] == 36
    rejected = "model_candidate/recovery/rejected/"
    assert out[rejected+"nonfinite_rows"] == 1
    assert out[rejected+"finite_rows"] == 0
    assert torch.isnan(out[rejected+"position_rad/mean"])
    assert torch.isnan(out[rejected+"position_rad/max"])


def test_envelope_conflict_pair_and_deterministic_ties():
    d = inputs(2)
    d["dt"].fill_(1.)
    d["joint_position"][1, 0] = 3.
    agg = QPIterationDiagnostics()
    agg.joint_envelope(d, -torch.ones(12), torch.ones(12), torch.ones(12), torch.ones(12), 1.)
    out = agg.finalize(d["tau_nom"])
    p = "model_candidate/joint_envelope/"
    assert out[p+"lower/acceleration"] == 1  # exact ties prefer acceleration
    assert out[p+"empty_row_fraction"] == .5
    assert out[p+"conflict_rad_s2/max"] == 1
    assert out[p+"conflict_pair/acceleration_position/rad_s2/max"] == 1
    assert torch.isclose(out[p+"conflict_pair/acceleration_position/fraction"], torch.tensor(1/24))


def test_count_weighted_partition_invariance_and_detachment():
    d = inputs(5)
    d["joint_position"][:, 0] = torch.arange(5)
    a = torch.zeros(5,18, requires_grad=True)
    whole, chunks = QPIterationDiagnostics(), QPIterationDiagnostics()
    accepted = torch.tensor([True, False, True, True, False])
    record(whole, d, a, accepted)
    for sl in (slice(0,1), slice(1,5)):
        record(chunks, {k:v[sl] for k,v in d.items()}, a[sl], accepted[sl])
    w, c = whole.finalize(a), chunks.finalize(a)
    for key in w:
        if key.startswith("model_candidate/"):
            torch.testing.assert_close(w[key], c[key], equal_nan=True)
            assert not w[key].requires_grad


def test_stage_metrics_preserved_and_unscheduled_skips():
    from unittest.mock import patch
    d = inputs(1)
    qp = solver(diagnostics_level="physical")
    qp.diagnostics_scheduled = False
    with patch.object(qp, "_joint_candidate_diagnostics", side_effect=AssertionError):
        qp.solve(**d)
    agg = QPIterationDiagnostics()
    a = torch.zeros(1,18)
    d["joint_position"].fill_(2)
    record(agg,d,a,torch.tensor([False]))
    record(agg,d,a,torch.tensor([True]),"recovery")
    out = agg.finalize(a)
    assert out["model_candidate/primary/rejected/position_rad/max"] == 1
    assert out["model_candidate/recovery/accepted/position_rad/max"] == 1


def test_real_solve_path_keeps_rejected_primary_and_accepted_recovery_metrics():
    from unittest.mock import patch
    from rsl_rl.algorithms.hard_pact_qp_backends import QPBackendResult
    d = inputs(1)
    d["bias"][:,6] = 200  # nonempty envelope, impossible with hard torque limits
    d["joint_velocity"][:,0] = -29.9
    qp = solver(diagnostics_level="physical", soft_joint_recovery_enabled=True)
    def candidate(m):
        x = torch.zeros_like(m.p)
        if x.shape[1] == 48:
            x[:,24:36] = 200  # recovery can satisfy its own softened inequalities
        return QPBackendResult(x / m.variable_scale)
    with patch.object(qp, "_backend_solve", side_effect=candidate):
        out = qp.solve(differentiable=False, **d)
    assert out.stage.item() == 1
    metrics = qp.iteration_diagnostics["rollout"].finalize(d["tau_nom"])
    for stage, status in (("primary", "rejected"), ("recovery", "accepted")):
        p = f"model_candidate/{stage}/{status}/"
        assert torch.isclose(metrics[p+"velocity_rad_s/max"], torch.tensor(1.9,dtype=torch.float64))
        assert metrics[p+"nonempty_envelope_fraction"] == 1
        assert metrics[p+"rows"] == 1
