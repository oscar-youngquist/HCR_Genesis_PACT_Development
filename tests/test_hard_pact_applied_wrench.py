"""Applied-only labels preserve substep frames and randomized-mechanics inputs."""
import inspect
from types import SimpleNamespace

import pytest
import torch

from test_go2_hard_pact_disturbances import _persistent_task
from legged_gym.envs.go2.go2_hard_pact.go2_hard_pact import Go2HardPACT
from legged_gym.envs.go2.go2_hard_pact_pos.go2_hard_pact_pos import Go2HardPACTPos
from rsl_rl.algorithms.ppo_hard_pact import PPO_HardPACT


@pytest.mark.parametrize("task_class", [Go2HardPACT, Go2HardPACTPos])
@pytest.mark.parametrize("loaded", [False, True])
def test_actual_ramped_wrench_only_with_mass_com_and_rotating_yaw(task_class, loaded):
    task = _persistent_task(task_class=task_class)
    task.obs_scales = SimpleNamespace(base_wrench=.01)
    task._realized_added_mass = torch.tensor([[3.]])
    task._realized_com_shift_body = torch.tensor([[.2, -.1, .05]])
    applied = []
    task.simulator.hard_pact_apply_base_wrench_world = lambda value: applied.append(value.clone())
    for suffix in ("sustained", "mass_com", "total", "sustained_yaw_scaled",
                   "mass_com_yaw_scaled", "yaw_scaled", "yaw_physical"):
        setattr(task, "_disturbance_interval_sum_"+suffix, torch.zeros(1, 6))
    task._disturbance_interval_count = torch.zeros(1, 1)
    world = torch.tensor([[8., -4., 2., 1., -2., 3.]]) if loaded else torch.zeros(1, 6)
    rotations = [torch.tensor([[0., 0., 0., 1.]]), torch.tensor([[0., 0., 2**-.5, 2**-.5]])]
    for factor, quat in zip((.25, .75), rotations):
        # The event sampler has already ramped/gated this buffer; do not label its unscaled target.
        task._current_sustained_wrench_world.copy_(factor*world)
        task._current_sustained_active_mask.fill_(loaded)
        task._current_base_quat_xyzw = lambda q=quat: q
        task._hard_pact_pre_physics_substep()
    result = task._end_disturbance_interval()
    yaw90 = world[:, [1, 0, 2, 4, 3, 5]] * torch.tensor([[1., -1., 1., 1., -1., 1.]])
    expected_yaw = (.25*world + .75*yaw90)/2
    torch.testing.assert_close(result["total_external_wrench_label_world"], torch.stack(applied).mean(0))
    torch.testing.assert_close(result["total_external_wrench_label_world"], .5*world)
    torch.testing.assert_close(result["total_external_wrench_label_yaw_scaled"], expected_yaw*.01)
    torch.testing.assert_close(result["total_external_wrench_label_yaw_normalized"],
        expected_yaw/torch.tensor(task.cfg.deployment_physics.wrench_scale))
    assert result["equivalent_mass_com_wrench_world"].abs().sum() > 0  # Diagnostic, not label.
    assert result["sustained_wrench_active_mask"].item() == loaded


def test_pinn_wiring_does_not_consume_mass_wrench_diagnostic():
    source = inspect.getsource(PPO_HardPACT._compute_bard_loss)
    assert "equivalent_mass_com_wrench_world" not in source
    assert "label_mass_wrench" not in source
    assert "total_wrench_world=applied_at_base[sl]" in source
    assert "applied_wrench_world=applied_at_base[sl]" in source
    # Actual randomized inertia is still supplied to BARD; the QP stays separate.
    assert '"added_base_mass": batch["realized_added_mass"].detach()' in source
    assert '"base_com_shift": batch["realized_com_shift_body"].detach()' in source
