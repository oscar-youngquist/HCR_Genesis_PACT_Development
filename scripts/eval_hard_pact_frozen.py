#!/usr/bin/env python3
"""Frozen Isaac Lab controller comparison; no runner, optimizer, or backward.

Duration is measured AFTER the common 100-control-step pre-QP prefix. Each
seed/controller runs in a fresh process, including fresh simulator/QP state.
JSON null means unavailable, never a successful solve or a zero residual.
"""
from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import asdict, fields, replace
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import traceback
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PREFIX = 100
VARIANTS = {
    "pre_qp": None,
    "analytic": None,
    "qp_every_substep": "every_substep",
    "qp_single_anchor": "single_anchor_held_correction",
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--resolved-config", required=True, type=Path)
    p.add_argument("--solver", choices=("qpth", "cupiqp", "moreau"), default="cupiqp")
    p.add_argument("--seeds", type=int, nargs="+", default=[1])
    p.add_argument("--num-envs", type=int, default=64)
    p.add_argument("--duration", type=float, default=20.0, help="seconds after activation")
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--sample-actions", action="store_true", help="seeded latent and action sampling")
    p.add_argument("--smoke", action="store_true", help="small terrain grid and one explicit reset after activation")
    p.add_argument("--trace-window", type=int, default=10)
    p.add_argument("--packet-limit", type=int, default=10)
    p.add_argument("--variant", choices=VARIANTS, help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    if args.num_envs < 1 or args.duration <= 0 or args.trace_window < 1 or not 0 <= args.packet_limit <= 10:
        p.error("positive environment count, duration and trace window; packet limit must be 0..10")
    for name in ("checkpoint", "resolved_config"):
        value = getattr(args, name).resolve()
        if not value.is_file():
            p.error(f"missing {name}: {value}")
        setattr(args, name, value)
    args.output_dir = args.output_dir.resolve()
    return args


def sha_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_hash(values):
    digest = hashlib.sha256()
    for name, tensor in sorted(values.items()):
        digest.update(name.encode())
        digest.update(tensor.detach().contiguous().cpu().numpy().tobytes())
    return digest.hexdigest()


def write_json(path, value):
    def clean(item):
        if isinstance(item, float) and not math.isfinite(item):
            return None
        if isinstance(item, dict):
            return {str(k): clean(v) for k, v in item.items()}
        if isinstance(item, (list, tuple)):
            return [clean(v) for v in item]
        if hasattr(item, "tolist"):
            return clean(item.tolist())
        return item
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(value), indent=2, allow_nan=False, default=str) + "\n")


def dependency_versions():
    result = {}
    for name in ("torch", "isaaclab", "isaacsim", "qpth", "cupiqp", "moreau", "bard", "warp-lang", "numpy", "cupy-cuda12x"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def apply_config(target, values):
    """Apply the resolved values, not today's subclass defaults. Keep dict leaves."""
    for key, value in values.items():
        current = getattr(target, key, None)
        if isinstance(value, dict) and current is not None and not isinstance(current, dict):
            apply_config(current, value)
        else:
            setattr(target, key, copy.deepcopy(value))


def extend_evaluation_timeout(env, duration):
    """Allow prefix + complete measurement horizon, including reset's initial step.

    Only this evaluation instance changes. The extra two control steps cover
    the legacy zero-action reset step and float-rounding at the end boundary.
    """
    count = PREFIX + max(1, math.ceil(duration / env.dt))
    env.max_episode_length = max(int(env.max_episode_length), count + 2)
    env.max_episode_length_s = env.max_episode_length * env.dt
    env.cfg.env.episode_length_s = env.max_episode_length_s
    return count


class Moments:
    """Device-side sufficient statistics; nonfinite values remain explicit."""
    def __init__(self):
        self.values = {}

    def add(self, name, value, mask=None):
        import torch
        value = value.detach().double()
        if mask is not None:
            while mask.ndim < value.ndim:
                mask = mask.unsqueeze(-1)
            value = value[mask.expand_as(value)]
        value = value.reshape(-1)
        finite = torch.isfinite(value)
        safe = torch.where(finite, value, 0.)
        item = torch.stack((finite.sum(), (~finite).sum(), safe.sum(), safe.square().sum(), safe.abs().sum(),
                            safe.abs().amax() if safe.numel() else value.new_zeros(())))
        if name not in self.values:
            self.values[name] = item
        else:
            self.values[name][:5] += item[:5]
            self.values[name][5] = torch.maximum(self.values[name][5], item[5])

    def result(self):
        result = {}
        for name, item in self.values.items():
            n, bad, total, square, absolute, maximum = item.cpu().tolist()
            result[name] = {"count": int(n), "nonfinite_count": int(bad),
                            "mean": total / n if n else None,
                            "rms": math.sqrt(square / n) if n else None,
                            "mean_abs": absolute / n if n else None,
                            "abs_max": maximum if n else None}
        return result


def make_scenario(env, count, seed):
    """Replay the task's own samplers/ramp law, independent of policy/failures.

    Shadows own ONLY sampling buffers; no physics API is called while making
    the tape. A common velocity-command tape avoids heading feedback yielding
    different requested yaw rates for diverging controller trajectories.
    """
    import torch
    shadow = copy.copy(env)
    for name, value in vars(env).items():
        if name.startswith(("_persistent_", "_current_sustained_")) and torch.is_tensor(value):
            setattr(shadow, name, value.clone())
    shadow.commands = env.commands.clone()
    sim = copy.copy(env.simulator)
    for name in ("push_timeouts", "vert_timeouts", "wrench_timeouts", "_rand_push_vels", "_rand_wrench_vels"):
        setattr(sim, name, getattr(sim, name).clone())
    sim._push_call_counter = 0
    sim._robot = SimpleNamespace(
        data=SimpleNamespace(root_link_vel_w=torch.zeros(env.num_envs, 6, device=env.device)),
        write_root_link_velocity_to_sim=lambda *a, **kw: None,
    )
    commands, wrenches, active, pushes = [], [], [], []
    ids = torch.arange(env.num_envs, device=env.device)
    period = max(1, int(env.cfg.commands.resampling_time / env.dt))
    devices = [torch.device(env.device).index or 0] if torch.device(env.device).type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed + 7103)
        shadow._reset_persistent_wrench_state(ids)
        for step in range(count + 1):
            if step % period == 0:
                shadow._resample_commands(ids)
            shadow._update_persistent_wrench(step)
            if env.cfg.domain_rand.push_robots:
                sim.push_robots()
            commands.append(shadow.commands.clone())
            wrenches.append(shadow._current_sustained_wrench_world.clone())
            active.append(shadow._current_sustained_active_mask.clone())
            pushes.append(torch.cat((sim._rand_push_vels, sim._rand_wrench_vels), -1).clone())
    return {k: torch.stack(v) for k, v in zip(
        ("commands", "wrench_world", "wrench_active", "push_delta_world"),
        (commands, wrenches, active, pushes))}


class Observer:
    """Evaluation-only hooks around the unchanged simulator and certified QP."""
    def __init__(self, env, actor, qp, args):
        import torch
        from collections import deque
        self.env, self.sim, self.actor, self.qp, self.args = env, env.simulator, actor, qp, args
        self.step, self.k, self.calls, self.packet_count = -1, 0, 0, 0
        self.metrics = Moments()
        self.previous = self.sim.hard_pact_executed_torque().clone()
        self.previous_contact = torch.zeros(env.num_envs, 4, device=env.device, dtype=torch.bool)
        self.solved_force = torch.full((env.num_envs, 4, 3), float("nan"), device=env.device)
        self.first_failure = torch.full((env.num_envs,), -1, device=env.device, dtype=torch.long)
        self.first_episode_censored = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
        self.reasons = {}
        self.ring = deque(maxlen=args.trace_window + 1)
        self.traces, self.trace_until, self.traced_failures = {}, -1, set()
        self.prefix = hashlib.sha256()
        self.latest, self.substeps = {}, []
        self._pre = self.sim._hard_pact_pre_physics_substep
        self._post = self.sim._hard_pact_grf_post_physics_substep
        self._solve = qp.solve
        self._termination = env.check_termination
        self._reset = env.reset_idx
        self.sim._hard_pact_pre_physics_substep = self.pre
        self.sim._hard_pact_grf_post_physics_substep = self.post
        qp.solve = self.solve
        env.check_termination = self.termination
        env.reset_idx = self.reset

    def reset(self, ids):
        self._reset(ids)
        self.previous[ids] = 0
        self.previous_contact[ids] = False
        self.solved_force[ids] = float("nan")
        # Shared reset clears histories, delay queue, held correction and QP
        # warm states. Do not replace its reset/reference semantics here.

    def pre(self):
        import torch
        from rsl_rl.algorithms.hard_pact_qp import held_correction_torque
        e, s = self.env, self.sim
        q, v = e._canonical_joint_state()
        warm = s._torques.detach().clone()  # same-state legacy weighted/randomized controller
        raw_previous = e._hard_pact_previous_substep_torque.clone() if hasattr(e, "_hard_pact_previous_substep_torque") else self.previous.clone()
        pd = (s.feedback_torques if hasattr(e, "_hard_pact_control_parameters")
              else e._get_pinn_feedback(e._hard_pact_q_d, q, v))
        nominal = (e._hard_pact_bounded_nominal_torque
                   if hasattr(e, "_hard_pact_control_parameters")
                   else e._hard_pact_tau_ff + pd)
        is_qp = self.step >= PREFIX and VARIANTS[self.args.variant] is not None
        # Analytic baseline projects the existing pre-QP law, not a redefined
        # PD law; the QP modes retain their own existing nominal convention.
        if self.step >= PREFIX and self.args.variant == "analytic":
            s.hard_pact_set_executed_torque(held_correction_torque(
                warm, torch.zeros_like(warm), self.previous, self.qp.torque_limits,
                self.qp.cfg.torque_rate_limit_nm_s, float(e.cfg.sim.dt), sanitize=True))
        if not is_qp:
            normalized = self.actor.physics_estimator.predict_grf(
                self.actor.cenet_z, self.actor.cenet_torso_velo, nominal)
            self.pred_force = e._yaw_local_to_world(
                self.actor.physics_estimator.grf_to_physical(normalized).reshape(-1, 4, 3),
                e._current_base_quat_xyzw())
        self._pre()  # existing QP, certification, recovery, wrench application, torque labels
        if is_qp and self.args.variant != "qp_every_substep":
            # Held modes freeze the yaw-local prediction, not its world
            # rotation; use the same live quaternion as the execution path.
            self.pred_force = e._yaw_local_to_world(
                self.actor.physics_estimator.grf_to_physical(e._hard_pact_held_grf_normalized).reshape(-1, 4, 3),
                e._current_base_quat_xyzw())
        executed = s.hard_pact_executed_torque().clone()
        phase = "evaluation" if self.step >= PREFIX else "prefix"
        limit = self.qp.torque_limits
        torque_violation = (executed.abs() - limit).clamp_min(0)
        rate_violation = ((executed - self.previous).abs() - self.qp.cfg.torque_rate_limit_nm_s * float(e.cfg.sim.dt)).clamp_min(0)
        values = {
            "executed_torque_nm": executed, "warmup_law_nm": warm, "qp_nominal_nm": nominal,
            "warmup_minus_qp_nominal_nm": warm - nominal,
            "qp_pd_nm": pd, "feedforward_nm": e._hard_pact_tau_ff,
            "warmup_pd_nm": s.feedback_torques, "raw_previous_nm": raw_previous,
            "actual_previous_nm": self.previous.clone(),
            "previous_mismatch_nm": raw_previous - self.previous,
            "torque_violation_nm": torque_violation, "rate_step_violation_nm": rate_violation,
            "torque_violation": torque_violation > 1e-5, "rate_violation": rate_violation > 1e-5,
            "correction_nm": executed - (nominal if is_qp else warm),
            "execution_finite": torch.isfinite(executed),
            "kp_scale": s._kp_scale, "kd_scale": s._kd_scale, "motor_strength": s._motor_strength,
        }
        self.latest = {name: value.detach().clone() for name, value in values.items()}
        for name, value in values.items():
            self.metrics.add(f"{phase}/{name}", value)
        if self.step < PREFIX:
            self.prefix.update(executed.detach().cpu().numpy().tobytes())
        self.previous.copy_(executed)

    def solve(self, **data):
        import torch
        self.calls += 1  # primary batched anchor calls, not internal recovery solves
        result = self._solve(**data)
        self.pred_force = data["force_pred_world"].detach().clone()
        self.solved_force = result.force_world.detach().clone()
        for stage, name in ((0, "full"), (2, "analytic")):
            self.metrics.add(f"qp/final_stage/{name}", result.stage == stage)
        self.metrics.add("qp/certified_solver_row", result.differentiated_mask)
        for name, value in result.diagnostics.items():
            self.metrics.add(f"qp/{name}", value)
        # Capture only bounded rows, never all environments or autograd graphs.
        # Full nominal matrices document intended physics even if recovery was
        # selected; actual recovery-stage matrices are saved alongside them.
        bad = (result.stage != 0).nonzero().flatten()
        if self.packet_count < self.args.packet_limit and (self.packet_count == 0 or bad.numel()):
            row = int(bad[0]) if bad.numel() else 0
            inputs = {k: v[row:row + 1].detach().clone() for k, v in data.items() if torch.is_tensor(v)}
            solver_inputs = {k: v.to(self.qp._solve_dtype(data["tau_nom"])) for k, v in inputs.items()}
            build = self.qp._build(solver_inputs)
            builds = {"full": {f.name: getattr(build, f.name).detach().cpu().clone() for f in fields(build)}}
            packet = {
                "schema_version": 1, "variant": self.args.variant, "control_step": self.step,
                "substep": self.k, "environment": row, "solver_settings": asdict(self.qp.cfg),
                "inputs": {k: v.cpu() for k, v in inputs.items()}, "matrices": builds,
                "state": self.env._canonical_configuration()[row:row + 1].cpu(),
                "velocity_world": self.env._canonical_velocity_world()[row:row + 1].cpu(),
                "observation": self.obs[row:row + 1].cpu(), "history": self.history[row:row + 1].cpu(),
                "raw_action": self.actions[row:row + 1].cpu(),
                "delayed_action": self.env._pending_action_replay_transition["delayed_action"][row:row + 1].cpu(),
                "latent": self.actor.cenet_z[row:row + 1].cpu(),
                "explicit": self.actor.cenet_torso_velo[row:row + 1].cpu(),
                "outputs": {k: getattr(result, k)[row:row + 1].cpu().clone() for k in
                            ("qdd", "force_world", "tau_safe", "stage", "differentiated_mask")},
                "diagnostics": {k: v[row:row + 1].cpu() for k, v in result.diagnostics.items()},
            }
            torch.save(packet, self.args.output_dir / f"replay_{self.packet_count:02d}.pt")
            self.packet_count += 1
        return result

    def post(self):
        import torch
        self._post()  # contact sensor has already advanced this physics substep
        measured = self.env.grf_processor.clipped.clone()
        contact = self.env.grf_processor.contacts.clone()
        probability = self.actor.cenet_torso_velo[:, 3:7]
        transition = contact != self.previous_contact
        phase = "evaluation" if self.step >= PREFIX else "prefix"
        for state, mask in (("all", torch.ones_like(contact)), ("stance", contact),
                            ("swing", ~contact), ("transition", transition)):
            for name, force in (("predicted", self.pred_force), ("solved", self.solved_force)):
                self.metrics.add(f"{phase}/grf/{name}/{state}/error_n", force - measured, mask)
            self.metrics.add(f"{phase}/contact/{state}/error", probability - contact.float(), mask)
            self.metrics.add(f"{phase}/contact/{state}/accuracy", (probability >= .5) == contact, mask)
        for name, value in (("predicted_grf_world_n", self.pred_force), ("solved_grf_world_n", self.solved_force),
                            ("measured_grf_world_n", measured), ("measured_raw_grf_world_n", self.env.grf_processor.raw),
                            ("predicted_contact", probability), ("measured_contact", contact)):
            self.latest[name] = value.detach().clone()
            self.metrics.add(f"{phase}/{name}", value)
        for foot, name in enumerate(("FR", "FL", "RR", "RL")):
            self.metrics.add(f"{phase}/grf/{name}/predicted_error_n", self.pred_force[:, foot] - measured[:, foot])
        self.previous_contact.copy_(contact)
        self.substeps.append({k: v[:4].clone() for k, v in self.latest.items()})
        self.k += 1

    def termination(self):
        import torch
        self._termination()
        e, s = self.env, self.sim
        new = e.reset_buf.bool() & (self.first_failure < 0)
        self.first_failure[new] = self.step
        # Time limits and non-failure terrain resets are right censoring, not
        # falls, even if a user-specified horizon still reaches a time limit.
        self.first_episode_censored[new] = (e.time_out_buf.bool() | e.non_failure_reset_buf.bool())[new]
        conditions = {
            "contact": (s.link_contact_forces[:, s.termination_contact_indices].norm(dim=-1) > 10).any(-1),
            "timeout": e.time_out_buf.bool(), "out_of_bounds": e.non_failure_reset_buf.bool(),
        }
        height = (s.base_pos[:, 2:3] - s.measured_heights).mean(-1)
        for term in e.cfg.termination.termination_terms:
            if term in ("roll", "pitch"):
                angle = s._base_euler[:, 0 if term == "roll" else 1]
                conditions[term] = ((angle + math.pi) % (2 * math.pi) - math.pi).abs() > getattr(e.cfg.termination, f"{term}_threshold")
            elif term.startswith("height_"):
                bound = getattr(e.cfg.termination, term)
                conditions[term] = height < bound if term == "height_min" else height > bound
        for name, flag in conditions.items():
            self.reasons[name] = self.reasons.get(name, torch.zeros_like(new, dtype=torch.long)) + (new & flag)
        self.metrics.add("episode/ended_duration_seconds", e.episode_length_buf.float() * e.dt, e.reset_buf.bool())
        phase = "evaluation" if self.step >= PREFIX else "prefix"
        self.metrics.add(f"{phase}/tracking/velocity_error_m_s", s.base_lin_vel[:, :2] - e.commands[:, :2])
        self.metrics.add(f"{phase}/tracking/yaw_error_rad_s", s.base_ang_vel[:, 2] - e.commands[:, 2])

    def finish_control(self):
        import torch
        # Keep the bounded trace ring on-device; transfer selected windows
        # only at export, not dozens of CUDA-to-host copies per substep.
        record = {key: torch.stack([row[key] for row in self.substeps]) for key in self.substeps[0]}
        record.update(observation=self.obs[:4].clone(), history=self.history[:4].clone(),
                      action=self.actions[:4].clone(), command=self.env.commands[:4].clone())
        self.ring.append((self.step, record))
        new_failures = (self.first_failure[:4] == self.step).nonzero().flatten().cpu().tolist()
        if new_failures or self.step == PREFIX:
            self.traces.update(self.ring)
            self.trace_until = max(self.trace_until, self.step + self.args.trace_window)
        if self.step <= self.trace_until:
            self.traces[self.step] = record
        self.k, self.substeps = 0, []


def build_actor(env, training, checkpoint):
    from legged_gym.envs.go2.go2_hard_pact.deployment import calculate_physics_head_gains
    from rsl_rl.modules.actor_critic_hard_pact import ActorCritic_HardPACT
    p = training["policy"]
    gains = calculate_physics_head_gains(env.cfg)
    actor = ActorCritic_HardPACT(
        env.num_obs, env.num_privileged_obs * env.num_crit_obs_stack, env.num_actions,
        p["actor_layers"], p["critic_layers"], env.num_obs * env.num_obs_hist,
        p["cenet_enc_latent_dim"], p["cenet_velo_dim"], p["cenet_enc_layers"],
        p["activation"], p["init_noise_std"],
        cenet_explicit_layers=p["cenet_explicit_layers"], grf_decoder_layers=p["grf_decoder_layers"],
        wrench_decoder_layers=p["wrench_decoder_layers"], grf_scale_n=gains.grf_scale_n,
        wrench_scale=gains.wrench_scale_n_nm, wrench_qp_clip=gains.wrench_qp_clip_n_nm,
        contact_epsilon=p["contact_epsilon"],
    ).to(env.device)
    actor.load_state_dict(checkpoint["model_state_dict"], strict=True)
    actor.eval().requires_grad_(False)
    return actor


def frozen_policy_actions(actor, obs, history, sample=False):
    """Avoid act()'s in-place standard-deviation clamp on frozen checkpoints."""
    import torch
    if not sample:
        return actor.act_inference(obs, history)
    mean, logvar, z, explicit = actor.cenet_enc_forward(history)
    actor.cenet_mean, actor.cenet_logvar = mean, logvar
    actor.cenet_z, actor.cenet_torso_velo = z, explicit
    action_mean = torch.cat(actor.actor_forward(torch.cat((obs, z, explicit), -1)), -1)
    return action_mean + torch.randn_like(action_mean) * actor.std.clamp(actor._std_clip_lwr, 5.)


def worker(args):
    import faulthandler
    import signal
    faulthandler.register(signal.SIGUSR1)
    source_hash = sha_file(Path(__file__))
    os.environ["SIMULATOR"] = "isaaclab"
    from legged_gym.scripts.train_hard_pact import prepare_solver_runtime
    prepare_solver_runtime(["--qp_solver", args.solver])
    import numpy as np
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Real Isaac Lab evaluation requires an available CUDA GPU")
    torch.cuda.set_device(args.device)
    import legged_gym.envs  # registers tasks, does not launch a runner
    from legged_gym.utils import task_registry
    from legged_gym.utils.helpers import class_to_dict, set_seed
    from legged_gym.dynamics import create_go2_dynamics
    from rsl_rl.algorithms.hard_pact_qp import HardPACTDifferentiableQP, HardPACTQPConfig
    document = json.loads(args.resolved_config.read_text())
    resolved = document["environment"]["resolved"]
    training = document["training"]["resolved"]
    task = "go2_hard_pact_full_isaaclab"
    cfg = copy.deepcopy(task_registry.env_cfgs[task])
    apply_config(cfg, resolved)
    cfg.env.num_envs, cfg.seed, cfg.ablation_variant = args.num_envs, args.seeds[0], "full"
    cfg.terrain.curriculum = cfg.commands.curriculum = False
    cfg.commands.heading_command = False  # common velocity-command tape, including yaw rate
    cfg.rewards.use_reward_curriculum = False
    cfg.domain_rand.use_domainrand_curriculum = False
    cfg.domain_rand.use_tradeoff_curriculum = False
    if args.smoke:
        cfg.terrain.num_rows = cfg.terrain.num_cols = 2
        cfg.terrain.max_init_terrain_level = min(cfg.terrain.max_init_terrain_level, 1)
    set_seed(cfg.seed)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    env = task_registry.get_task_class(task)(cfg, class_to_dict(cfg.sim), args.device, True)
    app = env.simulator._app_launcher.app
    try:
        steps = extend_evaluation_timeout(env, args.duration)
        if "hard_pact_domain_rand_curriculum" in checkpoint:
            env.load_domain_rand_curriculum_state_dict(checkpoint["hard_pact_domain_rand_curriculum"])
        # No curriculum advancement or optimizer objects are created. Freeze
        # the restored stage, retaining randomization and disturbance magnitudes.
        env.use_tradeoff = False
        frozen_curriculum = copy.deepcopy(env.domain_rand_curriculum_state_dict())
        actor = build_actor(env, training, checkpoint)
        weights_before = tensor_hash(actor.state_dict())
        mode = VARIANTS[args.variant]
        qp_cfg = replace(HardPACTQPConfig.from_dict(training["algorithm"]["hard_pact_qp"]),
                         enabled=True, warmup_iterations=0, qp_solver=args.solver,
                         rollout_qp_solver=None, ppo_qp_solver=None,
                         qp_update_mode=mode or "every_substep",
                         exception_capture_dir=str(args.output_dir / "exceptions"))
        s = env.simulator
        qp = HardPACTDifferentiableQP(qp_cfg, s.torque_limits, s.dof_pos_limits_hard[:, 0],
                                     s.dof_pos_limits_hard[:, 1], s.dof_vel_limits)
        dynamics = create_go2_dynamics("bard", cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=ROOT),
                                      device=args.device, batch_capacity=args.num_envs,
                                      default_joint_position=s.default_dof_pos.reshape(-1, 12)[0])
        env.configure_hard_pact_substep_qp(actor, dynamics, qp)
        env.set_hard_pact_qp_enabled(False)
        env.reset()
        scenario = make_scenario(env, steps, cfg.seed)
        scenario_hash = tensor_hash(scenario)
        initial_hash = tensor_hash({"q": env._canonical_configuration(), "v": env._canonical_velocity_world(),
                                    **env._canonical_randomized_parameters()})
        observer = Observer(env, actor, qp, args)
        compute_observations = env.compute_observations
        def matched_observations():
            # Anchor-index sampling consumes RNG in the existing QP path.
            # Isolate observation noise so that it cannot perturb this trial's
            # sensor-noise tape relative to the no-QP controller.
            with torch.random.fork_rng(devices=[torch.device(env.device).index or 0]):
                torch.manual_seed(cfg.seed * 1000003 + 90001 + observer.step)
                compute_observations()
        env.compute_observations = matched_observations
        # Tape playback is independent of asynchronous episode resets. The
        # original environment still performs all physics/filter/history resets.
        def set_wrench(_step):
            t = max(observer.step, 0)
            env._current_sustained_wrench_world.copy_(scenario["wrench_world"][t])
            env._current_sustained_active_mask.copy_(scenario["wrench_active"][t])
        def push():
            delta = scenario["push_delta_world"][max(observer.step, 0)]
            ids = delta.any(-1).nonzero().flatten()
            s._rand_push_vels.copy_(delta[:, :3])
            s._rand_wrench_vels.copy_(delta[:, 3:])
            if ids.numel():
                s._robot.write_root_link_velocity_to_sim(s._robot.data.root_link_vel_w[ids, :6] + delta[ids], env_ids=ids)
        env._update_persistent_wrench, s.push_robots = set_wrench, push
        env._resample_commands = lambda ids: env.commands.__setitem__(ids, scenario["commands"][max(observer.step + 1, 0), ids])
        callback = env._post_physics_step_callback
        def post_control():
            callback()
            env.commands.copy_(scenario["commands"][min(observer.step + 1, steps)])
        env._post_physics_step_callback = post_control
        env.commands.copy_(scenario["commands"][0])
        # Rebuild initial observations once with the shared initial commands.
        env.compute_observations()
        qp.clear_warm_start()
        reset_checked = False
        with torch.no_grad():
            for step in range(steps):
                observer.step = step
                env.set_hard_pact_qp_enabled(step >= PREFIX and mode is not None)
                obs, history, _, _ = env.get_observations()
                observer.obs, observer.history = obs.clone(), history.clone()
                # Separate deterministic RNG stream: QP sampling/recovery can
                # never change action/latent or observation-noise draws.
                torch.manual_seed(cfg.seed * 1000003 + step)
                actions = frozen_policy_actions(actor, obs, history, args.sample_actions)
                observer.actions = actions.clone()
                result = env.step(actions)
                for name, value in zip(("observation", "critic", "history", "explicit", "reward"), result[:5]):
                    observer.metrics.add(f"finite/{name}", torch.isfinite(value))
                heads = actor.physics_estimator
                wrench_normalized = (env._hard_pact_wrench_raw_normalized if step >= PREFIX and mode is not None
                                     else heads.predict_wrench(actor.cenet_z, actor.cenet_torso_velo))
                raw = heads.wrench_to_physical(wrench_normalized)
                clipped = heads.wrench_to_qp_physical(wrench_normalized)
                target = env._pending_disturbance_transition["total_external_wrench_label_yaw_normalized"] * heads.wrench_scale
                for name, value in (("raw", raw), ("clipped", clipped), ("target", target), ("raw_error", raw - target)):
                    observer.metrics.add(f"wrench/{name}_yaw_n_nm", value)
                    for axis, component in enumerate(("Fx_n", "Fy_n", "Fz_n", "Tx_nm", "Ty_nm", "Tz_nm")):
                        observer.metrics.add(f"wrench/{name}/{component}", value[:, axis])
                    # Control-rate wrench head and interval-average target.
                    for substep in observer.substeps:
                        substep[f"wrench_{name}_yaw_n_nm"] = value[:4].clone()
                observer.finish_control()
                if step == PREFIX or (step + 1) % 200 == 0:
                    print(f"{args.variant}: control step {step + 1}/{steps}; QP calls={observer.calls}", flush=True)
                if args.smoke and step == PREFIX + 1:
                    env.reset_idx(torch.tensor([0], device=env.device))
                    reset_checked = bool(env._action_replay_valid_queue[0].eq(0).all())
                    env.compute_observations()
        metrics = observer.metrics.result()
        actual_steps = steps - PREFIX
        expected_calls = actual_steps * (cfg.control.decimation if mode == "every_substep" else int(mode is not None))
        unchanged = tensor_hash(actor.state_dict()) == weights_before
        current_curriculum = env.domain_rand_curriculum_state_dict()
        curriculum_unchanged = all(current_curriculum[k] == frozen_curriculum[k] for k in ("progress", "last_iteration"))
        first = observer.first_failure.cpu().numpy()
        summary = {
            "variant": args.variant, "seed": cfg.seed, "num_envs": args.num_envs,
            "prefix_control_steps": PREFIX, "evaluation_control_steps": actual_steps,
            "control_dt": env.dt, "physics_dt": cfg.sim.dt,
            "survival_fraction": float(observer.first_episode_censored.float().mean()),
            "first_episode_unended_fraction": float((first < 0).mean()),
            "survival_is_right_censored": True,
            "survived_to_activation_fraction": float(((first < 0) | (first >= PREFIX)).mean()),
            "first_episode_duration_seconds": (np.where(first < 0, steps, first + 1) * env.dt).tolist(),
            "first_episode_censored": observer.first_episode_censored.cpu().tolist(),
            "failure_reasons": {k: int(v.sum()) for k, v in observer.reasons.items()},
            "qp_calls": observer.calls, "expected_qp_calls": expected_calls,
            "held_substep_commands": actual_steps * (cfg.control.decimation - 1) * args.num_envs if mode == "single_anchor_held_correction" else 0,
            "weights_and_buffers_unchanged": unchanged, "smoke_reset_checked": reset_checked,
            "curriculum_unchanged": curriculum_unchanged,
            "initial_state_hash": initial_hash, "scenario_hash": scenario_hash,
            "prefix_executed_torque_hash": observer.prefix.hexdigest(), "metrics": metrics,
            "packets": observer.packet_count,
            "violations": {name: metrics[f"evaluation/{name}"] for name in
                           ("torque_violation_nm", "rate_step_violation_nm", "torque_violation", "rate_violation")},
        }
        metadata = {
            "schema_version": 1, "checkpoint": str(args.checkpoint), "checkpoint_sha256": sha_file(args.checkpoint),
            "source_config": document, "effective_environment": class_to_dict(cfg), "solver_settings": asdict(qp_cfg),
            "evaluation_script_sha256": source_hash,
            "evaluation_overrides": {"command_mode": "shared sampled velocity commands (heading feedback disabled)",
                                     "curricula": "frozen at restored checkpoint domain-randomization stage",
                                     "episode_timeout_seconds": env.max_episode_length_s,
                                     "episode_timeout_reason": "cover prefix plus full evaluation without artificial termination",
                                     "controller": args.variant, "smoke_terrain_2x2": args.smoke},
            "checkpoint_iteration_ignored_for_qp_activation": checkpoint.get("iter"),
            "frozen_curriculum": frozen_curriculum, "curriculum_after": env.domain_rand_curriculum_state_dict(),
            "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "git_diff": subprocess.check_output(["git", "diff"], cwd=ROOT, text=True),
            "versions": dependency_versions(),
            "device": args.device, "gpu": torch.cuda.get_device_name(), "dtype": str(next(actor.parameters()).dtype),
            "cuda_runtime": torch.version.cuda,
            "sample_actions": args.sample_actions, "optimizers_created": False,
            "metric_notes": {"null": "unavailable (including unsolved GRFs/gaps), not zero",
                             "forces": "world XYZ Newtons; canonical FR FL RR RL",
                             "wrench": "yaw-local Fx Fy Fz Tx Ty Tz, N/Nm",
                             "rate_violation": "abs(tau_k-tau_actual_previous)-rate_limit*physics_dt; Nm per substep",
                             "counts": "real rows only; primary batched QP calls exclude recovery attempts",
                             "survival": "fraction without a first-episode physical failure; timeouts/out-of-bounds/right edge are censored, not falls; not a Kaplan-Meier estimate",
                             "errors": "mean_abs is MAE and rms is RMSE for error metrics; GRF/contact errors are per physics substep; all rows including later episodes",
                             "traces": "at most four envs, windows around activation and each first failure",
                             "reset_comparison": "common initial state; subsequent resets use normal task lifecycle"},
        }
        write_json(args.output_dir / "metadata.json", metadata)
        write_json(args.output_dir / "summary.json", summary)
        if observer.traces:
            indices = sorted(observer.traces)
            np.savez_compressed(args.output_dir / "traces.npz", control_step=np.array(indices),
                                **{k: torch.stack([observer.traces[i][k] for i in indices]).cpu().numpy()
                                   for k in observer.traces[indices[0]]})
        finite = all(v["mean"] == 1.0 for k, v in metrics.items() if k.startswith("finite/") or k.endswith("execution_finite"))
        if not unchanged or not curriculum_unchanged or observer.calls != expected_calls or not finite or (args.smoke and not reset_checked):
            raise RuntimeError("Frozen evaluation smoke failed; see summary.json (failures were not discarded)")
        print(json.dumps({k: summary[k] for k in ("variant", "survival_fraction", "qp_calls", "weights_and_buffers_unchanged")}), flush=True)
    except BaseException as error:
        # Persist failures before simulator shutdown, which can itself stall.
        write_json(args.output_dir / "failure.json", {"exception": repr(error), "traceback": traceback.format_exc()})
        raise
    finally:
        # Exercise the shared headless adapter's STOP fix before clearing the
        # singleton: training uses the same adapter, not an evaluation workaround.
        env.simulator._sim.stop()
        env.simulator._sim.clear_instance()
        app.close(wait_for_replicator=False)


def main(argv=None):
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.variant:
        try:
            worker(args)
        except BaseException as error:
            write_json(args.output_dir / "failure.json", {"exception": repr(error), "traceback": traceback.format_exc()})
            raise
        return
    summaries, statuses = [], []
    remaining = args.packet_limit
    for seed in args.seeds:
        for variant in VARIANTS:
            destination = args.output_dir / f"seed_{seed}" / variant
            destination.mkdir(parents=True, exist_ok=True)
            budget = min(2, remaining)
            command = [sys.executable, str(Path(__file__).resolve()), "--checkpoint", str(args.checkpoint),
                       "--resolved-config", str(args.resolved_config), "--solver", args.solver,
                       "--seeds", str(seed), "--num-envs", str(args.num_envs), "--duration", str(args.duration),
                       "--output-dir", str(destination), "--device", args.device, "--variant", variant,
                       "--packet-limit", str(budget), "--trace-window", str(args.trace_window)]
            command += [flag for flag, enabled in (("--smoke", args.smoke), ("--sample-actions", args.sample_actions)) if enabled]
            with (destination / "console.log").open("w") as stream:
                result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            statuses.append({"seed": seed, "variant": variant, "exit_code": result.returncode, "command": command})
            if (destination / "summary.json").exists():
                summary = json.loads((destination / "summary.json").read_text())
                summaries.append(summary)
                remaining -= summary["packets"]
            print(f"{variant} seed={seed}: exit={result.returncode}; {destination}", flush=True)
    alignment = {}
    for seed in args.seeds:
        group = [x for x in summaries if x["seed"] == seed]
        alignment[str(seed)] = {key: len({x[key] for x in group}) == 1 and len(group) == len(VARIANTS)
                               for key in ("initial_state_hash", "scenario_hash", "prefix_executed_torque_hash")}
    write_json(args.output_dir / "aggregate.json", {"trials": summaries, "statuses": statuses, "alignment": alignment})
    with (args.output_dir / "aggregate.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("seed", "variant", "metric", "count", "nonfinite_count", "mean", "rms", "mean_abs", "abs_max"))
        for trial in summaries:
            for name in ("survival_fraction", "survived_to_activation_fraction", "qp_calls", "expected_qp_calls", "held_substep_commands"):
                writer.writerow((trial["seed"], trial["variant"], name, 1, 0, trial[name], "", "", ""))
            for name, metric in trial["metrics"].items():
                writer.writerow((trial["seed"], trial["variant"], name, *(metric[k] for k in
                                ("count", "nonfinite_count", "mean", "rms", "mean_abs", "abs_max"))))
    if any(s["exit_code"] for s in statuses) or not all(all(v.values()) for v in alignment.values()):
        raise SystemExit("Evaluation failures/alignment differences recorded in aggregate.json; inspect console.log")


if __name__ == "__main__":
    main()
