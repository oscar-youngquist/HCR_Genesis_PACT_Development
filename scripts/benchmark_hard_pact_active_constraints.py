"""Bounded CUDA benchmark: real Go2 BARD refresh + cuPIQP, not a simulator run.

Both modes see the same seeded canonical pose/velocity sequence. Their own
previous executed torque centers the next hard rate box. A separate same-data
comparison measures primal/objective differences without contaminating timing.
Profiling synchronizations are confined to this explicitly requested benchmark.
"""
import argparse
from dataclasses import asdict, replace
import importlib.metadata
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from legged_gym.dynamics import BardGo2Dynamics, SIMULATOR_JOINT_ORDER
from legged_gym.envs.go2.go2_hard_pact.go2_hard_pact_config import GO2HardPACTCfg
from rsl_rl.algorithms.hard_pact_qp import HardPACTDifferentiableQP, HardPACTQPConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--intervals", type=int, default=5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.batch_size <= 256 or not 1 <= args.intervals <= 20:
        parser.error("bounded benchmark requires batch 1..256 and intervals 1..20")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        parser.error("this benchmark requires a real CUDA GPU")
    torch.manual_seed(7301)
    cfg = HardPACTQPConfig(qp_solver="cupiqp", solver_dtype="float32", cuda_event_profiling=True)
    n = args.batch_size
    nominal = torch.tensor([GO2HardPACTCfg.init_state.default_joint_angles[name]
                            for name in SIMULATOR_JOINT_ORDER], device=device)
    mechanics = BardGo2Dynamics(str(ROOT / "resources/robots/go2/urdf/go2.urdf"),
                                device=device, batch_capacity=n, default_joint_position=nominal)
    torque_limits = torch.tensor([23.5, 23.5, 45.0] * 4, device=device)
    lower = torch.tensor([-1.047, -.663, -2.721] * 4, device=device)
    upper = torch.tensor([1.047, 2.966, -.837] * 4, device=device)
    def solver(mode):
        return HardPACTDifferentiableQP(replace(cfg, qp_update_mode=mode), torque_limits,
                                       lower, upper, torch.full((12,), 30., device=device))
    dt = .005
    q = torch.zeros(n, 19, device=device)
    q[:, 2], q[:, 6], q[:, 7:] = .44, 1., nominal
    offsets = torch.randn(n, 12, device=device) * .01
    ids = torch.arange(n, device=device)

    def packet(step, previous):
        current = q.clone()
        current[:, 7:] += offsets + .002 * torch.sin(torch.as_tensor(step * .1, device=device))
        v = torch.zeros(n, 18, device=device)
        v[:, 6:] = .04 * torch.cos(torch.as_tensor(step * .1, device=device))
        context = mechanics.build_context(current, v, parameters={}, need_qp=True)
        forces = torch.zeros(n, 4, 3, device=device)
        forces[..., 2] = 30.
        # No privileged mechanics: nominal deployment model only. Live PD is
        # bounded before QP, as in the existing backend's non-QP torque path.
        tau = (30. * (nominal - current[:, 7:]) - .8 * v[:, 6:]).clamp(-torque_limits, torque_limits)
        return dict(mass_matrix=context.mass_matrix, bias=context.bias,
                    foot_jacobians=context.foot_jacobians, base_jacobian=context.base_jacobian,
                    foot_acceleration_bias=context.foot_acceleration_bias,
                    tau_nom=tau, force_pred_world=forces, wrench_pred_world=torch.zeros(n, 6, device=device),
                    contact_probability=torch.full((n, 4), .8, device=device), previous_torque=previous,
                    joint_position=current[:, 7:], joint_velocity=v[:, 6:], dt=torch.full((n, 1), dt, device=device))

    report = {"kind": "seeded canonical-state replay with real BARD, not simulator trajectories",
              "gpu": torch.cuda.get_device_name(device), "torch": torch.__version__,
              "cuda": torch.version.cuda, "cupiqp": importlib.metadata.version("cupiqp"),
              "seed": 7301, "batch_size": n, "intervals": args.intervals, "physics_dt": dt,
              "decimation": 4, "config": asdict(cfg),
              "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}
    with torch.no_grad():
        for mode in ("every_substep", "active_constraint_update"):
            qp = solver(mode)
            previous = torch.zeros(n, 12, device=device)
            # One complete untimed policy interval initializes BARD kernels,
            # cuPIQP allocations and every active/fallback path encountered.
            for k in range(4):
                data = packet(k, previous)
                previous = qp.solve(differentiable=False, environment_ids=ids,
                                    environment_count=n, substep_index=k, **data).tau_safe
            qp.clear_warm_start()
            qp.begin_iteration_diagnostics("rollout")
            previous.zero_()
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            times, mechanical_events = [], []
            rate_violation = torch.zeros((), device=device)
            for interval in range(args.intervals):
                start = time.perf_counter()
                for k in range(4):
                    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    begin.record()
                    data = packet(4 * interval + k, previous)
                    end.record()
                    mechanical_events.append((begin, end))
                    result = qp.solve(differentiable=False, environment_ids=ids,
                                      environment_count=n, substep_index=k, **data)
                    assert torch.isfinite(result.tau_safe).all()
                    rate_violation = torch.maximum(rate_violation,
                        ((result.tau_safe - previous).abs() - cfg.torque_rate_limit_nm_s * dt).clamp_min(0).max())
                    previous = result.tau_safe
                torch.cuda.synchronize(device)
                times.append(1000 * (time.perf_counter() - start))
            metrics = {k: float(v) for k, v in qp.iteration_metrics("rollout", previous).items()}
            report[mode] = dict(control_step_ms=times, mean_ms=statistics.mean(times),
                std_ms=statistics.pstdev(times), mechanics_total_ms=sum(a.elapsed_time(b) for a, b in mechanical_events),
                peak_torch_mib=torch.cuda.max_memory_allocated(device) / 2**20,
                rate_violation_nm=float(rate_violation), metrics=metrics)
            import cupy
            report[mode]["cupy_pool_total_mib"] = cupy.get_default_memory_pool().total_bytes() / 2**20
            del qp

        # Same-data forward comparison: reference uses the active run's exact
        # previous executed command. Never confuse trajectory drift with QP parity.
        active, full = solver("active_constraint_update"), solver("every_substep")
        previous = torch.zeros(n, 12, device=device)
        differences = {}
        for k in range(4):
            data = packet(k, previous)
            a = active.solve(differentiable=False, environment_ids=ids, environment_count=n, substep_index=k, **data)
            b = full.solve(differentiable=False, **data)
            for name in ("tau_safe", "qdd", "force_world", "contact_slack"):
                differences[name] = max(differences.get(name, 0.), float((getattr(a, name) - getattr(b, name)).abs().max()))
            matrices = active._build(data)
            def objective(r):
                x = torch.cat((r.qdd, r.force_world.flatten(1), r.tau_safe, r.contact_slack.flatten(1)), -1)
                z = x / matrices.variable_scale
                return .5 * (z * (matrices.Q @ z[..., None]).squeeze(-1)).sum(-1) + (matrices.p * z).sum(-1)
            differences["objective"] = max(differences.get("objective", 0.), float((objective(a) - objective(b)).abs().max()))
            previous = a.tau_safe
        report["same_data_max_absolute_differences"] = differences
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), **{mode: {k: report[mode][k] for k in ("mean_ms", "std_ms", "peak_torch_mib")}
                       for mode in ("every_substep", "active_constraint_update")}}, indent=2))


if __name__ == "__main__":
    main()
