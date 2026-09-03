"""Focused smoke coverage for the decimation-four held-correction QP mode."""

from types import SimpleNamespace

import torch

from legged_gym.envs.go2.go2_hard_pact.go2_hard_pact import Go2HardPACT
from rsl_rl.algorithms.hard_pact_qp import projection_loss


def test_two_anchor_rollout_reset_and_ppo_backward():
    batch = 8
    physics_dt = 0.01
    torque_rate = 10.0

    class Robot:
        def __init__(self):
            self.q = torch.zeros(batch, 12)
            self.v = torch.zeros(batch, 12)

        def get_dofs_position(self, _indices):
            return self.q

        def get_dofs_velocity(self, _indices):
            return self.v

        def get_pos(self):
            return torch.zeros(batch, 3)

        def get_vel(self):
            return torch.zeros(batch, 3)

        def get_ang(self):
            return torch.zeros(batch, 3)

    class Heads:
        def __init__(self):
            self.grf_calls = 0

        def predict_grf(self, _latent, _explicit, tau_nom):
            self.grf_calls += 1
            return 0.1 * tau_nom

    class Dynamics:
        def __init__(self):
            self.calls = 0

        def build_context(self, *_args, **_kwargs):
            self.calls += 1
            return SimpleNamespace(
                mass_matrix=torch.eye(18).expand(batch, -1, -1),
                bias=torch.zeros(batch, 18),
                foot_jacobians=torch.zeros(batch, 4, 3, 18),
                base_jacobian=torch.zeros(batch, 6, 18),
                foot_acceleration_bias=torch.zeros(batch, 4, 3),
            )

    class QP:
        def __init__(self):
            self.calls = 0
            self.clear_calls = 0
            self.torque_limits = torch.full((12,), 2.0)
            self.cfg = SimpleNamespace(
                qp_update_mode="two_anchor_held_correction",
                torque_rate_limit_nm_s=torque_rate,
                elastic_recovery_enabled=False,
                slack_scale_m_s2=1.0,
            )

        def clear_warm_start(self, _env_ids):
            self.clear_calls += 1

        def solve(self, *, differentiable, tau_nom, force_pred_world,
                  wrench_pred_world, contact_probability, previous_torque,
                  **_kwargs):
            self.calls += int(not differentiable)
            # A compact differentiable stand-in for the selected PPO QP row.
            correction = (
                0.05 * tau_nom
                + 0.002 * force_pred_world.reshape(-1, 12)
                + 0.001 * wrench_pred_world.repeat_interleave(2, dim=-1)
                + 0.01 * contact_probability.repeat_interleave(3, dim=-1)
            )
            rate = torque_rate * physics_dt
            lower = torch.maximum(-self.torque_limits, previous_torque - rate)
            upper = torch.minimum(self.torque_limits, previous_torque + rate)
            safe = torch.maximum(torch.minimum(tau_nom + correction, upper), lower)
            zeros = torch.zeros(tau_nom.shape[0])
            return SimpleNamespace(
                qdd=torch.zeros(tau_nom.shape[0], 18),
                force_world=force_pred_world,
                tau_safe=safe,
                contact_slack=contact_probability[:, :, None].expand(-1, -1, 3),
                stage=torch.zeros(tau_nom.shape[0], dtype=torch.long),
                differentiated_mask=torch.ones(tau_nom.shape[0], dtype=torch.bool),
                diagnostics={
                    "selected/equality_max": zeros,
                    "selected/inequality_max": zeros,
                    "selected/stationarity_max": zeros,
                    "selected/complementarity_max": zeros,
                },
            )

    class Simulator:
        def __init__(self):
            self._robot = Robot()
            self._dof_indices = torch.arange(12)
            self._torques = torch.zeros(batch, 12)
            self.history = []

        def hard_pact_set_executed_torque(self, torque):
            self._torques = torque
            self.history.append(torque.clone())

    class Legacy:
        @staticmethod
        def reset_idx(_task, _env_ids):
            return None

    task = Go2HardPACT.__new__(Go2HardPACT)
    task.num_envs = batch
    task.device = torch.device("cpu")
    task.cfg = SimpleNamespace(
        sim=SimpleNamespace(dt=physics_dt),
        control=SimpleNamespace(decimation=4),
    )
    task.obs_scales = SimpleNamespace(grf=1.0, base_wrench=1.0)
    task.simulator = Simulator()
    heads = Heads()
    dynamics = Dynamics()
    qp = QP()
    task._hard_pact_actor_critic = SimpleNamespace(physics_estimator=heads)
    task._hard_pact_bard_dynamics = dynamics
    task._hard_pact_rollout_qp = qp
    task._legacy_task_class = Legacy
    task._hard_pact_policy_latent = torch.zeros(batch, 16)
    task._hard_pact_policy_explicit = torch.full((batch, 11), 0.5)
    task._hard_pact_wrench_yaw_scaled = torch.zeros(batch, 6)
    task._hard_pact_q_d = torch.ones(batch, 12)
    task._hard_pact_tau_ff = torch.zeros(batch, 12)
    task._hard_pact_previous_substep_torque = torch.ones(batch, 12)
    task._hard_pact_previous_certified_qdd = torch.ones(batch, 18)
    task._get_pinn_feedback = (
        lambda desired, position, velocity: 3.0 * (desired - position) - velocity
    )

    # One episode reset must clear both the torque-rate center and held state.
    task._begin_qp_interval()
    task._hard_pact_held_correction.fill_(1.0)
    task.reset_idx(torch.arange(batch))
    assert qp.clear_calls == 1
    assert torch.count_nonzero(task._hard_pact_previous_substep_torque) == 0
    assert torch.count_nonzero(task._hard_pact_held_correction) == 0
    task._hard_pact_q_d.fill_(1.0)

    quaternion = torch.tensor([[0.0, 0.0, 0.0, 1.0]]).expand(batch, -1)
    mass_wrench = torch.zeros(batch, 6)
    previous = torch.zeros(batch, 12)
    control_steps = 3
    for control_step in range(control_steps):
        task._begin_qp_interval()
        sampled = task._qp_sampled_substep_index.long()
        assert set(sampled.tolist()) == {0, 2}
        assert torch.bincount(sampled // 2, minlength=2).tolist() == [4, 4]
        for substep in range(4):
            task.simulator._robot.q.fill_(0.05 * (4 * control_step + substep))
            task.simulator._robot.v.fill_(0.01 * substep)
            task._solve_hard_pact_rollout_qp_substep(quaternion, mass_wrench)
            executed = task.simulator._torques
            assert torch.isfinite(executed).all()
            assert (executed.abs() <= qp.torque_limits + 1.0e-7).all()
            assert ((executed - previous).abs() <= torque_rate * physics_dt + 1.0e-7).all()
            previous = executed.clone()

    # Exactly two mechanics/QP refreshes and one GRF prediction per interval.
    assert qp.calls == 2 * control_steps
    assert dynamics.calls == 2 * control_steps
    assert heads.grf_calls == control_steps
    assert len(task.simulator.history) == 4 * control_steps

    # One selected-QP PPO-style backward: all learned inputs and their shared
    # source receive finite, nonzero gradients through the projection loss.
    source = torch.randn(batch, 5, requires_grad=True)
    tau_head = torch.nn.Linear(5, 12, bias=False)
    grf_head = torch.nn.Linear(5, 12, bias=False)
    wrench_head = torch.nn.Linear(5, 6, bias=False)
    contact_head = torch.nn.Linear(5, 4, bias=False)
    tau_nom = tau_head(source)
    grf = grf_head(source).reshape(batch, 4, 3)
    wrench = wrench_head(source)
    contact = contact_head(source).sigmoid()
    differentiated = qp.solve(
        differentiable=True,
        tau_nom=tau_nom,
        force_pred_world=grf,
        wrench_pred_world=wrench,
        contact_probability=contact,
        previous_torque=torch.zeros_like(tau_nom),
    )
    loss = projection_loss(
        differentiated.tau_safe,
        tau_nom,
        qp.torque_limits,
        torch.ones(batch, 1, dtype=torch.bool),
        differentiated.differentiated_mask[:, None],
        contact_slack=differentiated.contact_slack,
        slack_scale=1.0,
    )
    loss.backward()
    assert torch.isfinite(loss)
    for parameter in (
        source, tau_head.weight, grf_head.weight,
        wrench_head.weight, contact_head.weight,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
