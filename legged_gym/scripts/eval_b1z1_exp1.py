"""Frozen Experiment 1 evaluation. No runner, optimizer, or training storage is built."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
# Import the pure utility without importing utils/__init__ (which selects a backend).
spec = importlib.util.spec_from_file_location("b1z1_experiment", ROOT / "legged_gym/utils/b1z1_experiment.py")
util = importlib.util.module_from_spec(spec)
spec.loader.exec_module(util)

# EDIT HERE. Timing/threshold choices are provisional, not publication criteria.
# All times are seconds; Cartesian quantities use SI units. Spherical EE entries
# are [radius in m, elevation in rad, azimuth in rad], in the native yaw frame.
EVAL_CONFIG = {
    "event_times": [3., 7., 11.],  # onset, changed load, removal (overlap on both bodies)
    "transition_seconds": 1.,
    "velocity_ranges": [[0.2, 0.6], [-0.4, 0.4], [-0.6, 0.6]],
    "ee_center_sphere": [0.5, 0.1, 0.],
    "ee_amplitude_sphere": [0.04, 0.1, 0.15], "ee_period": 6.,
    "ee_tolerance_m": 0.10, "velocity_tolerance_m_s": 0.20, "yaw_tolerance_rad_s": 0.30,
    "recovery_window_seconds": 2., 
    "recovery_dwell_seconds": 0.5,
    "height_min_m": 0.3, 
    "height_max_m": 1.1, 
    "tilt_limit_rad": 0.9,
    "torso_contact_threshold_n": 10., 
    "torso_body_names": ["base", "trunk"],  # Resolve articulation/contact order by name.
    "reject_beta": 1.,  # Frozen estimator-only compensation; never restore curriculum.
    "chunk_rows": 2048,
    "initial_joint_jitter_rad": 0.02, 
    "initial_yaw_range_rad": [-0.2, 0.2],
    # Strict nested overrides: unknown fields fail instead of silently being ignored.
    # Enabled physical DR uses configured final ranges, sampled once per trial with
    # separate feature-keyed RNG streams. Equal endpoints give a fixed value.
    "overrides": {
        "terrain": {"mesh_type": "plane", "curriculum": False},
        "noise": {"add_noise": True, "noise_level": 1.0},
        "domain_rand": {
            "use_domainrand_curriculum": False, 
            "push_robots": False,
            "randomize_friction": True, 
            "friction_range": [0.80, 1.20],
            "randomize_base_mass": False, 
            "added_mass_min": -2., "max_added_mass_max": 10.,
            "randomize_gripper_mass": False, 
            "gripper_mass_min": -0.01, "max_gripper_added_mass_max": 0.1,
            "randomize_com_displacement": False,
            "com_displacement_x_max": 0.15, 
            "com_displacement_y_max": 0.15,
            "com_displacement_z_max": 0.15,
            "randomize_ctrl_delay": False, "ctrl_delay_step_range": [0, 1],
            "randomize_pd_gain": False, "kp_range": [0.8, 1.2], "kd_range": [0.8, 1.2],
            "randomize_motor_strength": False, "motor_strength_range": [0.85, 1.15],
            "randomize_joint_armature": True, "joint_armature_range": [0.015, 0.015],
            "randomize_joint_friction": True, "joint_friction_range_end": [0.1, 0.1],
            "randomize_joint_stiffness": False, "joint_stiffness_range_end": [0., 0.],
            "randomize_joint_damping": True, "joint_damping_range_end": [0.40, 0.40],
        },
    },
    "randomize_branch_weights": False,  # PACT-only; native configured ranges apply.
    "branch_weight_overrides": {},  # E.g. pact_weight_bias_min/max, pact_balanced_prob.
}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", default="b1z1_pact")
    p.add_argument("--run", required=True, help="Existing run directory or registered experiment run name")
    p.add_argument("--ckpt", default="-1", help="Checkpoint number, model file, or -1 for latest")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--condition", choices=("nominal", "ee-disturbance", "combined-disturbance"), default="nominal")
    p.add_argument("--load-range", choices=("in", "out"), default="in")
    p.add_argument("--num-envs", type=int, default=2)
    p.add_argument("--episodes", type=int, default=4)
    p.add_argument("--horizon-seconds", type=float, default=15.)
    p.add_argument("--motion-mode", choices=("stationary", "locomotion", "mixed"), default="mixed")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--gpu", default="cuda:0")
    return p


def simulator_family(task):
    if not task.startswith("b1z1_"):
        raise ValueError("Experiment 1 requires a registered B1Z1 task (PILOT is b1z1_pact)")
    if "unifp" in task:
        return "isaaclab_b1z1_unifp"
    return "isaaclab_b1z1_pact_pos" if task == "b1z1_pact_pos" else "isaaclab_b1z1_pact"


def apply_overrides(cfg, overrides, prefix=""):
    for key, value in overrides.items():
        if not hasattr(cfg, key):
            raise ValueError(f"Unknown evaluation override: {prefix}{key}")
        if isinstance(value, dict):
            apply_overrides(getattr(cfg, key), value, prefix + key + ".")
        else:
            setattr(cfg, key, value)


def configure(cfg, args, config=None):
    config = EVAL_CONFIG if config is None else config
    cfg = util.materialize(cfg)
    apply_overrides(cfg, config["overrides"])
    if hasattr(cfg.control, "randomize_pact_weights"):
        apply_overrides(cfg.control, config["branch_weight_overrides"], "control.")
    cfg.env.num_envs = args.num_envs
    cfg.seed = args.seed
    cfg.commands.curriculum = False
    cfg.use_force_shifted_target = False
    cfg.terrain.curriculum = False
    cfg.domain_rand.use_domainrand_curriculum = False
    cfg.domain_rand.push_robots = False
    for key in ("use_tradeoff_curriculum", "use_tradeoff", "randomize_pact_weights"):
        if hasattr(cfg.control, key):
            setattr(cfg.control, key, False)
    if hasattr(cfg.control, "randomize_pact_weights"):
        cfg.control.randomize_pact_weights = config["randomize_branch_weights"]
    for key in ("use_reward_curriculum", "use_force_shifted_target"):
        if hasattr(cfg.rewards, key):
            setattr(cfg.rewards, key, False)
    for key in ("heading_command",
                "randomize_base_force_gains", "randomize_gripper_force_gains",
                "use_external_impedance_compensation"):
        if hasattr(cfg.commands, key):
            setattr(cfg.commands, key, False)
    # Do not disable the application path: only the adapter replaces samplers.
    for key in ("push_robot_base", "push_gripper_stators", "apply_ee_external_forces",
                "apply_base_external_forces", "apply_base_external_torques"):
        setattr(cfg.commands, key, True)
    # Keep terrain dimensions/height observation schema, not training curricula.
    cfg.env.episode_length_s = args.horizon_seconds + 10.
    for requested, name in zip(config["velocity_ranges"], ("lin_vel_x", "lin_vel_y", "ang_vel_yaw")):
        lo, hi = getattr(cfg.commands.ranges, name)
        if requested[0] < lo or requested[1] > hi:
            raise ValueError(f"Evaluation {name} range {requested} exceeds configured training range {[lo, hi]}")
    return cfg


def validate_experiment(config):
    import numpy as np
    times = config["event_times"]
    if len(times) != 3 or not (0 <= times[0] < times[1] < times[2]):
        raise ValueError("event_times must be increasing onset/change/removal seconds")
    if not 0 <= config["reject_beta"] <= 1:
        raise ValueError("reject_beta must lie in [0,1]")
    for key in ("ee_period", "ee_tolerance_m", "velocity_tolerance_m_s", "yaw_tolerance_rad_s",
                "recovery_window_seconds", "recovery_dwell_seconds", "chunk_rows"):
        if not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    for ranges in (config["velocity_ranges"], [config["initial_yaw_range_rad"]]):
        for lo, hi in ranges:
            if not np.isfinite([lo, hi]).all() or lo > hi:
                raise ValueError("Invalid evaluation range")


def resolve_checkpoint(run, ckpt, experiment):
    from legged_gym.utils.helpers import get_load_path
    root = ROOT / "logs/b1z1" / experiment
    run_path = Path(run).expanduser()
    if not run_path.is_dir():
        run_path = root / run
    if not run_path.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_path}")
    if str(ckpt).lstrip("-").isdigit():
        path = Path(get_load_path(str(run_path.parent), run_path.name, int(ckpt)))
    else:
        path = Path(ckpt).expanduser()
        if not path.is_absolute():
            path = run_path / path
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path.resolve()


def build_actor(env, training, state=None):
    """Mirror registered runner constructors without invoking their training setup."""
    from rsl_rl.utils.runner_registry import runner_registry
    from rsl_rl import modules
    runner = runner_registry.get_runner_class(training.runner_class_name)
    p = vars(training.policy)
    critic = env.num_privileged_obs * env.num_crit_obs_stack
    history = env.num_obs * env.num_obs_hist
    if runner.__name__ in ("B1Z1PACTRunner", "B1Z1PACTPosRunner"):
        name = "ActorCriticB1Z1PACT" if runner.__name__ == "B1Z1PACTRunner" else "ActorCriticB1Z1PACTPos"
        module = __import__("rsl_rl.modules.actor_critic_b1z1_pact" + ("_pos" if name.endswith("Pos") else ""), fromlist=[name])
        cls = getattr(module, name)
        aliases = dict(latent_dim=p["cenet_latent_dim"], context_layers=p["cenet_enc_layers"], explicit_dim=env.num_exp_labels)
        keywords = {key: aliases.get(key, p.get(key, param.default))
                    for key, param in inspect.signature(cls).parameters.items()
                    if key in aliases or key in p}
        actor = cls(env.num_obs, critic, env.num_actions, history, **keywords)
    elif runner.__name__ == "OnPolicyRunnerUniFPOriginal":
        from rsl_rl.modules.actor_critic_unifp_original import ActorCriticUniFPOriginal
        actor = ActorCriticUniFPOriginal(history, critic, env.num_pred_obs, env.num_obs, env.num_actions,
                                        min_noise_std=p.get("min_noise_std"))
    elif runner.__name__ == "OnPolicyRunnerUniFP":
        cls = getattr(modules, training.runner.policy_class_name)
        actor = cls(history, critic, env.num_pred_obs, env.num_obs, env.num_actions,
                    num_privileged_obs_single=env.cfg.env.num_privileged_recon_obs,
                    enable_additional_diagnostics=False, **p)
    else:
        raise ValueError(f"Unsupported registered runner: {runner.__name__}; add an explicit inference adapter")
    if state is not None:
        if "architecture" in state and hasattr(actor, "architecture") and state["architecture"] != actor.architecture:
            raise ValueError("Checkpoint architecture tag does not match the selected task")
        actor.load_state_dict(state["model_state_dict"], strict=True)
    return actor.to(env.device).eval().requires_grad_(False)


def inference(actor, obs, history):
    """Original's act_inference clamps std in-place; compute its exact mean without it."""
    if hasattr(actor, "context_encoder"):
        return actor.act_inference(obs, history)
    if hasattr(actor, "architecture"):
        import torch
        z = actor.adaptation_encoder_module(history)
        actor.last_prediction = actor.adaptation_decoder_module(z)
        return actor.actor_body(torch.cat((history[:, -obs.shape[-1]:], z), -1))
    return actor.act_inference({"obs": history}, {})


def recorded_inference(actor, obs, history):
    """Capture native FiLM activations from action_t's pass, not a second forward."""
    film = getattr(actor, "film", None)
    if film is None or getattr(actor, "conditioning_mode", "film") != "film":
        return inference(actor, obs, history), {}
    records = {}

    def modulation(module, inputs, output):
        gamma, beta = output.chunk(2, dim=-1)
        records.update(film_gamma_t=gamma.detach().clone(), film_beta_t=beta.detach().clone())

    def features(module, inputs, output):
        # Native dimensionless features; gamma is residual scale, NOT 1+gamma.
        records.update(film_features_pre_t=inputs[0].detach().clone(),
                       film_features_post_t=output[0].detach().clone(),
                       film_condition_t=inputs[1].detach().clone())

    handles = []
    try:
        handles.append(film.network.register_forward_hook(modulation))
        handles.append(film.register_forward_hook(features))
        action = inference(actor, obs, history)
        if not records:
            raise RuntimeError("FiLM-enabled actor did not execute FiLM during inference")
        return action, records
    finally:
        # No hooks persist into other decoder calls, playback, or training.
        for handle in handles:
            handle.remove()


@contextmanager
def isolated_rng(seed, device):
    """Native reset/property setters may draw randomness; restore every RNG afterward."""
    import numpy as np
    import torch
    py_state, np_state = random.getstate(), np.random.get_state()
    devices = [torch.device(device).index or 0] if str(device).startswith("cuda") else []
    try:
        with torch.random.fork_rng(devices=devices):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)


def model_hash(actor):
    import hashlib
    h = hashlib.sha256()
    for name, value in actor.state_dict().items():
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def grf_conditioning_before_step(env, training, actor, actions):
    """Match the runner's nominal Nm input, without building a learner/optimizer."""
    if (training.runner_class_name != "B1Z1PACTRunner" or
            getattr(training.algorithm, "dynamics_backend", "pinocchio").lower() != "bard"):
        return None
    from types import SimpleNamespace
    from legged_gym.torque_action_scaling import simulator_torque_action_scale
    from rsl_rl.algorithms.ppo_b1z1_pact import PPO_B1Z1PACT
    # Reuse the exact runner helper, including its passive-DOF convention.
    # Training deliberately uses action_t, NOT the delay-buffer action; x_t is
    # captured before integration. This is distinct from executed interval torque.
    owner = SimpleNamespace(actor_critic=actor, cfg={
        "position_action_scale": env.cfg.control.action_scale,
        "torque_action_scale": simulator_torque_action_scale(env.simulator),
    })
    clip = env.cfg.normalization.clip_actions
    nominal = PPO_B1Z1PACT._coupled_torque(
        owner, actions.clamp(-clip, clip), env.get_pact_dynamics_state())
    limit = 1.1 * env.simulator.torque_limits
    return nominal.clamp(-limit, limit).detach().clone()


class EvaluationAdapter:
    """Instance-only hooks: no training class is modified and no automatic reset occurs."""
    def __init__(self, env, config, seed):
        import torch
        self.env, self.sim, self.config, self.seed = env, env.simulator, config, seed
        if not hasattr(env, "base_torque_ext_world"):
            # UniFP predicts base force, not moments. This evaluation-only applied
            # moment buffer adds no observation channel or privileged estimate.
            env.base_torque_ext_world = torch.zeros_like(env.base_force_ext_world)
        self.scenarios, self.step_index = [], 0
        self.saved_reset = env.reset_idx
        self.saved_observations = env.compute_observations
        self.noise = bool(env.add_noise)
        env.add_noise = False  # Inject keyed noise into the newest history slot below.
        env.use_tradeoff = False
        env.reset_idx = lambda ids: None  # Retain physical terminal successor until recorded.
        env.check_termination = lambda: env.reset_buf.zero_()
        env.compute_reward = lambda: env.rew_buf.zero_()  # Scores are common, not training rewards.
        # This training reward captures BARD state before step(), even when reward
        # computation is bypassed. Evaluation needs neither its teacher nor a runner.
        env.reward_scales.pop("tracking_arm_feedforward", None)
        env._resample_commands = lambda ids: None
        env._post_physics_step_callback = self.post_control
        env.compute_observations = self.observations
        env._push_gripper = lambda ids: None
        env._push_robot_base = lambda ids: None
        # A teleport is not a physical successor. Prohibit terrain wrapping during eval.
        self.teleported = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        if hasattr(self.sim, "_check_base_pos_out_of_bound"):
            self.sim._check_base_pos_out_of_bound = lambda: None
        self.sim.collect_b1z1_bard_interval = False
        self.substeps = []
        self.compute_torques = self.sim._compute_torques
        self.sensor_update = self.sim._contact_sensors.update
        self.sim._compute_torques = self.torque
        self.sim._contact_sensors.update = self.contact_update
        self.pending = None
        self.body_masses = None
        self.torso_indices = [self.sim._body_names.index(name) for name in config["torso_body_names"]
                              if name in self.sim._body_names]
        if not self.torso_indices:
            raise ValueError("No configured torso bodies exist in the articulation")

    def torque(self, actions):
        import torch
        s = self.sim
        requested = self.compute_torques(actions)
        target = s.default_dof_pos.expand_as(s.dof_pos).clone()
        target[:, :s._num_learned_actions] += actions[:, :s._num_learned_actions] * s._cfg.control.action_scale
        self.pending = {"delayed_action": actions.detach().clone(), "position_target": target,
                        "torque_request_motor_scaled": requested.detach().clone(),
                        "torque_request_pre_motor": s.unclipped_torques.detach().clone(),
                        "pd_torque": s.feedback_torques.detach().clone(),
                        "ff_torque": getattr(s, "feedforward_torques", torch.zeros_like(requested)).detach().clone()}
        # Position controllers (including UniFP) have no learned FF contribution.
        # UniFP allocates combined_* buffers but never fills them; use its actual PD.
        position_only = actions.shape[-1] == s._num_learned_actions
        pd = s.feedback_torques if position_only else s.combined_feedback_torques
        ff = torch.zeros_like(requested) if position_only else s.combined_feedforward_torques
        self.pending["pd_weighted_motor"] = (pd * s._motor_strength).clone()
        self.pending["ff_weighted_motor"] = (ff * s._motor_strength).clone()
        return requested

    def contact_update(self, *args, **kwargs):
        import torch
        result = self.sensor_update(*args, **kwargs)
        if self.pending is not None:
            s = self.sim
            row = dict(self.pending)
            row["executed_torque"] = s.executed_torques.clone()
            row["grf_raw_world"] = s._sensor_foot_forces_world().clone()
            masses = self.body_masses
            com = (s._robot.data.body_com_pos_w * masses[..., None]).sum(1) / masses.sum(1, keepdim=True)
            feet = s._robot.data.body_link_pos_w[:, s.feet_indices]
            row["support_moment_world"] = torch.linalg.cross(feet - com[:, None], row["grf_raw_world"], dim=-1).sum(1)
            # The wrench composer applies these world vectors at each body's CoM.
            row["ee_load_point_world"] = s._robot.data.body_com_pos_w[:, s._gripper_index].clone()
            row["base_load_point_world"] = s._robot.data.body_com_pos_w[:, s._base_link_index].clone()
            row["joint_position"] = s.dof_pos.clone()
            row["joint_velocity"] = s.dof_vel.clone()
            limits = s.dof_pos_limits_hard if hasattr(s, "dof_pos_limits_hard") else s.dof_pos_limits
            row["position_violation"] = (row["joint_position"] < limits[..., 0]) | (row["joint_position"] > limits[..., 1])
            row["velocity_violation"] = row["joint_velocity"].abs() > s.dof_vel_limits
            row["saturation"] = row["torque_request_motor_scaled"].abs() >= 1.1 * s.torque_limits
            self.substeps.append({key: value.detach().clone() for key, value in row.items()})
        return result

    def publish_targets(self, time):
        import torch
        from legged_gym.envs.b1z1.b1z1_pact.b1z1_pact import sphere2cart
        e = self.env
        for slot, scenario in enumerate(self.scenarios):
            command, sphere, _, _ = scenario.at(time)
            e.commands[slot, :3] = torch.as_tensor(command, device=e.device)
            e.curr_ee_goal_sphere[slot] = torch.as_tensor(sphere, device=e.device)
        e.curr_ee_goal_cart.copy_(sphere2cart(e.curr_ee_goal_sphere))
        e._refresh_curr_ee_goal_world()
        e.commands[:, 3:6] = e.curr_ee_goal_sphere

    def post_control(self):
        self.env._step_contact_targets()
        self.publish_targets((self.step_index + 1) * self.env.dt)

    def observations(self):
        import numpy as np
        import torch
        e = self.env
        self.saved_observations()
        if self.noise:
            for slot, scenario in enumerate(self.scenarios):
                rng = np.random.default_rng(util.seed_for(self.seed, scenario.metadata["trial_id"], "noise", self.step_index + 1))
                noise = torch.as_tensor(rng.uniform(-1., 1., e.num_obs), device=e.device, dtype=e.obs_buf.dtype)
                e.obs_buf[slot] += noise * e.noise_scale_vec
            # Existing history is oldest->newest; preserve reset-filled earlier slots.
            if hasattr(e, "obs_history_slots"):
                e.obs_history_slots[e._obs_history_slot].copy_(e.obs_buf)
            e.obs_history[:, -e.num_obs:] = e.obs_buf

    def reset(self, scenarios):
        import torch
        e, s = self.env, self.sim
        self.scenarios, self.step_index = scenarios, -1
        for slot, scenario in enumerate(scenarios):
            ids = torch.tensor([slot], device=e.device, dtype=torch.long)
            trial = scenario.metadata["trial_id"]
            if e.cfg.terrain.mesh_type != "plane":
                import numpy as np
                rng = np.random.default_rng(util.seed_for(self.seed, trial, "terrain"))
                level = int(rng.integers(s._terrain_origins.shape[0]))
                kind = int(rng.integers(s._terrain_origins.shape[1]))
                s._terrain_levels[slot], s._terrain_types[slot] = level, kind
                s._env_origins[slot] = s._terrain_origins[level, kind]
            with isolated_rng(util.seed_for(self.seed, trial, "initial"), e.device):
                self.saved_reset(ids)
                # Native reset streams differ across methods. Overwrite physical initial
                # conditions with a dedicated common stream after native buffer clearing.
            with isolated_rng(util.seed_for(self.seed, trial, "physical_initial"), e.device):
                q = s.default_dof_pos.expand(1, -1).clone()
                q += (2 * torch.rand_like(q) - 1) * self.config["initial_joint_jitter_rad"]
                limits = s.dof_pos_limits_hard if hasattr(s, "dof_pos_limits_hard") else s.dof_pos_limits
                q = q.clamp(limits[..., 0], limits[..., 1])
                s.reset_dofs(ids, q, torch.zeros_like(q))
                low, high = self.config["initial_yaw_range_rad"]
                yaw = low + (high - low) * torch.rand(1, device=e.device)
                quat = torch.zeros((1, 4), device=e.device)
                quat[:, 2], quat[:, 3] = torch.sin(yaw / 2), torch.cos(yaw / 2)
                pos = s.base_init_pos.reshape(1, -1) + s._env_origins[ids]
                s.reset_root_states(ids, pos, quat, torch.zeros((1, 3), device=e.device), torch.zeros((1, 3), device=e.device))
            # Feature-keyed randomization does not depend on batch size or reset ordering.
            dr = e.cfg.domain_rand
            for flag, method in (("randomize_friction", "_randomize_friction"),
                                 ("randomize_base_mass", "_randomize_mass"),
                                 ("randomize_com_displacement", "_randomize_com_displacement"),
                                 ("randomize_joint_armature", "_randomize_joint_armature"),
                                 ("randomize_joint_friction", "_randomize_joint_friction"),
                                 ("randomize_joint_stiffness", "_randomize_joint_stiffness"),
                                 ("randomize_joint_damping", "_randomize_joint_damping"),
                                 ("randomize_pd_gain", "_randomize_pd_gain"),
                                 ("randomize_motor_strength", "_randomize_motor_strength")):
                active = getattr(dr, flag, False) or (method == "_randomize_mass" and dr.randomize_gripper_mass)
                if active:
                    with isolated_rng(util.seed_for(self.seed, trial, method), e.device):
                        getattr(s, method)(ids)
            if getattr(e.cfg.control, "randomize_pact_weights", False):
                with isolated_rng(util.seed_for(self.seed, trial, "branch_weights"), e.device):
                    e._randomize_pact_torque_weights(ids)
            if dr.randomize_ctrl_delay:
                with isolated_rng(util.seed_for(self.seed, trial, "delay"), e.device):
                    lo, hi = dr.ctrl_delay_step_range
                    e.action_delay[ids] = torch.randint(lo, hi + 1, (1,), device=e.device)
        self.pending = None
        self.body_masses = s._robot.root_physx_view.get_masses().to(e.device)
        s.post_physics_step()
        e.commands.zero_()
        self.publish_targets(0.)
        self.observations()
        e.reset_buf.zero_()
        self.substeps.clear()

    def loads(self, time):
        import torch
        e, s = self.env, self.sim
        for name in ("ee_force_ext_world", "base_force_ext_world", "base_torque_ext_world"):
            getattr(e, name).zero_()
        for slot, scenario in enumerate(self.scenarios):
            load = torch.as_tensor(scenario.at(time)[2], device=e.device, dtype=e.ee_force_ext_world.dtype)
            e.ee_force_ext_world[slot] = load[:3]
            e.base_force_ext_world[slot] = load[3:6]
            e.base_torque_ext_world[slot] = load[6:]
        s.apply_ee_force(e.ee_force_ext_world)
        s.apply_base_force(e.base_force_ext_world)
        s.apply_base_torque(e.base_torque_ext_world)

    def failures(self):
        import torch
        s, c = self.sim, self.config
        if self.env.cfg.terrain.mesh_type != "plane":
            p = s.base_pos
            outside = ((p[:, 0] <= s._terrain_x_range[0]) | (p[:, 0] >= s._terrain_x_range[1]) |
                       (p[:, 1] <= s._terrain_y_range[0]) | (p[:, 1] >= s._terrain_y_range[1]))
            if outside.any():
                raise RuntimeError("Trial left the terrain domain; enlarge the evaluation terrain. Teleporting is forbidden.")
        height = s.base_pos[:, 2] - s.measured_heights.mean(-1)
        tilt = torch.atan2(torch.sin(s.base_euler[:, :2]), torch.cos(s.base_euler[:, :2])).abs()
        contact = (s.link_contact_forces[:, self.torso_indices].norm(dim=-1) > c["torso_contact_threshold_n"]).any(-1)
        bad = torch.stack((contact, (height < c["height_min_m"]) | (height > c["height_max_m"]),
                           tilt[:, 0] > c["tilt_limit_rad"], tilt[:, 1] > c["tilt_limit_rad"]), -1)
        # Common physical failures terminate immediately at this control endpoint.
        triggered = bad
        finite = torch.isfinite(s.dof_pos).all(-1) & torch.isfinite(s.base_pos).all(-1)
        reasons = []
        for row, ok in zip(triggered.cpu().tolist(), finite.cpu().tolist()):
            reasons.append("nonfinite_state" if not ok else "+".join(name for name, value in zip(("torso_contact", "height", "roll", "pitch"), row) if value))
        return reasons


def numpy_value(value):
    return value.detach().cpu().numpy()


def main(argv=None, experiment=None):
    # Opt-in evaluation protocol hooks; Experiment 1 retains its original defaults.
    args = (parser() if experiment is None else experiment.parser()).parse_args(argv)
    config = EVAL_CONFIG if experiment is None else experiment.config
    if experiment is None:
        validate_experiment(config)
    else:
        experiment.validate(config, args)
    if args.horizon_seconds <= 0 or args.seed < 0:
        raise ValueError("horizon must be positive and seed nonnegative")
    trial_batches = list(util.batches(args.episodes, args.num_envs) if experiment is None else experiment.batches(args))
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Use an empty output directory; evaluation never overwrites records")
    os.environ["SIMULATOR"] = simulator_family(args.task)
    # Backend selection precedes imports of environments, runners and simulator classes.
    import numpy as np
    import torch
    import legged_gym.envs
    from legged_gym.utils import task_registry
    from legged_gym.utils.helpers import class_to_dict
    if args.task not in task_registry.task_classes:
        raise ValueError(f"Unregistered task {args.task}; choose a registered b1z1 task name")
    training = util.materialize(task_registry.train_cfgs[args.task])
    cfg = configure(task_registry.env_cfgs[args.task], args, config)
    env_class = task_registry.get_task_class(args.task)
    reject = args.task in ("b1z1_unifp_original", "b1z1_unifp_reject")
    if reject:
        # Same Original architecture/checkpoint; only the evaluation environment adds
        # estimator-based rejection. Preserve its existing LPF/order/reset semantics.
        env_class = task_registry.get_task_class("b1z1_unifp_reject")
        reject_commands = util.materialize(task_registry.env_cfgs["b1z1_unifp_reject"].commands)
        for key, value in vars(reject_commands).items():
            if not hasattr(cfg.commands, key):
                setattr(cfg.commands, key, value)
        cfg.commands.use_external_impedance_compensation = True
        cfg.commands.compensate_ee_external_force = cfg.commands.compensate_base_external_force = True
    path = resolve_checkpoint(args.run, args.ckpt, training.runner.experiment_name)
    state = torch.load(path, map_location="cpu", weights_only=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    env = None
    writer = util.NumericWriter(args.output_dir, config["chunk_rows"])
    trials = []
    progress = None
    try:
        with isolated_rng(args.seed, args.gpu):
            env = env_class(cfg, class_to_dict(cfg.sim), args.gpu, args.headless)
            actor = build_actor(env, training, state)
        before = model_hash(actor)
        privileged_decoder = None
        if "privileged_decoder_state_dict" in state and hasattr(actor, "explicit_decoder"):
            from rsl_rl.modules.actor_critic_b1z1_pact import B1Z1PACTDecoder
            p = training.policy
            explicit_width = actor.explicit_decoder.network[-1].out_features
            privileged_decoder = B1Z1PACTDecoder(
                p.cenet_latent_dim + explicit_width,
                cfg.env.num_privileged_recon_obs - cfg.env.privileged_force_start - cfg.env.num_privileged_force_obs,
                hidden=p.privileged_decoder_layers, activation=p.activation).to(env.device)
            privileged_decoder.load_state_dict(state["privileged_decoder_state_dict"], strict=True)
            privileged_decoder.eval().requires_grad_(False)
        if reject:
            env._staged_force_curriculum.beta = config["reject_beta"]
        adapter_class = EvaluationAdapter if experiment is None else experiment.Adapter
        adapter = adapter_class(env, config, args.seed)
        s = env.simulator
        steps = math.ceil(args.horizon_seconds / env.dt)
        metadata = dict(schema_version=1, task=args.task, runner=training.runner_class_name,
                        checkpoint=str(path), checkpoint_sha256=util.file_hash(path),
                        checkpoint_iteration=state.get("iter", state.get("iteration")), checkpoint_training_seed=None,
                        evaluation_seed=args.seed, backend=os.environ["SIMULATOR"],
                        resolved_environment=class_to_dict(cfg), resolved_training=class_to_dict(training),
                        evaluation=config, arguments={**vars(args), "output_dir": str(args.output_dir)},
                        dt=env.dt, physics_dt=env.dt / cfg.control.decimation, prescribed_steps=steps,
                        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                        git_dirty=bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT)),
                        evaluator_sha256=util.file_hash(__file__),
                        utilities_sha256=util.file_hash(ROOT / "legged_gym/utils/b1z1_experiment.py"),
                        conventions={"quaternion": "xyzw", "world": "z-up", "torque": "Nm",
                                     "forces": "world N", "joint": "configured order, rad and rad/s",
                                     "grf_raw_world": "interval mean of raw sensor samples, not filtered",
                                     "prediction": "native normalized decoder units; unavailable fields omitted",
                                     "film": "action_t inference pass: post=pre*(1+gamma)+beta; dimensionless actor features; condition in native normalized units; fields omitted without FiLM",
                                     "torque_request_pre_motor": "before motor strength; actual clip=1.1*rated",
                                     "prediction_grf_torque_input_nm": (
                                         "clipped pre-step nominal action torque; no action delay"
                                         if training.runner_class_name == "B1Z1PACTRunner" and
                                         getattr(training.algorithm, "dynamics_backend", "pinocchio").lower() == "bard"
                                         else "final executed substep torque, including action delay; field omitted without GRF decoder"),
                                     "EE_reference": "nominal yaw-aligned workspace target evaluated at successor"},
                        dof_names=list(cfg.asset.dof_names), foot_indices=s.feet_indices.tolist(),
                        body_names=list(s._body_names), sensor_body_names=list(s._contact_sensors.body_names),
                        feet_sensor_indices=s._feet_contact_indices.tolist(),
                        gripper_index=int(s._gripper_index), base_index=int(s._base_link_index),
                        failure_reason_bits={"torso_contact": 1, "height": 2, "roll": 4, "pitch": 8,
                                             "nonfinite_state": 16, "nonfinite_record": 32},
                        unavailable=["checkpoint_training_seed (not stored in checkpoint)"])
        urdf = Path(getattr(cfg.asset, "isaacgym_file", cfg.asset.file).replace("{LEGGED_GYM_ROOT_DIR}", str(ROOT)))
        metadata["urdf_source"] = str(urdf)
        metadata["urdf_sha256"] = util.file_hash(urdf)
        converted = Path(s._robot.cfg.spawn.asset_path)
        metadata["converted_urdf"] = str(converted)
        metadata["converted_urdf_sha256"] = util.file_hash(converted)
        if experiment is not None:
            metadata.update(experiment.metadata(env, args, config, state))
            if metadata["checkpoint_training_seed"] is not None:
                metadata["unavailable"].remove("checkpoint_training_seed (not stored in checkpoint)")
        def save_json(name, data):
            (args.output_dir / name).write_text(json.dumps(data, indent=2, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x), allow_nan=False))
        save_json("metadata.json", metadata)
        with torch.no_grad():
            from tqdm import tqdm
            progress = tqdm(total=args.episodes, unit="episode", dynamic_ncols=True,
                            desc=f"{args.task} | {args.condition} | seed {args.seed}")
            for ids in trial_batches:
                scenarios = [(util.Scenario(args.seed, trial, args.condition, args.load_range, args.motion_mode, config)
                              if experiment is None else experiment.scenario(args, trial)) for trial in ids]
                adapter.reset(scenarios)
                # Mechanics are fixed within a trial: store full body properties once,
                # avoiding repeated host PhysX queries in the control loop.
                masses = s._robot.root_physx_view.get_masses().to(env.device)
                inertias = s._robot.root_physx_view.get_inertias()
                coms = s._robot.root_physx_view.get_coms()
                with (args.output_dir / "initial_states.jsonl").open("a") as stream:
                    for slot, scenario in enumerate(scenarios):
                        initial = dict(trial_id=scenario.metadata["trial_id"],
                                       joint_position=numpy_value(s.dof_pos[slot]).tolist(),
                                       base_position_world=numpy_value(s.base_pos[slot]).tolist(),
                                       base_quaternion_xyzw=numpy_value(s.base_quat[slot]).tolist(),
                                       origin_world=numpy_value(s._env_origins[slot]).tolist(),
                                       body_masses_kg=numpy_value(masses[slot]).tolist(),
                                       body_inertias_kg_m2=numpy_value(inertias[slot]).tolist(),
                                       body_local_com_pose=numpy_value(coms[slot]).tolist())
                        # Environment-grid translations are not scenario changes.
                        canonical = {k: v for k, v in initial.items() if k not in ("base_position_world", "origin_world")}
                        canonical["base_position_relative_origin"] = numpy_value(s.base_pos[slot] - s._env_origins[slot]).tolist()
                        initial["initial_state_hash"] = util.digest(canonical)
                        stream.write(json.dumps(initial, allow_nan=False) + "\n")
                metric_class = util.TrialMetrics if experiment is None else experiment.TrialMetrics
                metrics = [metric_class(steps, env.dt, scenario, config) for scenario in scenarios]
                alive = [True] * len(ids)
                reasons = [""] * len(ids)
                peak_loads = np.zeros((len(ids), 9))
                training_load_bounds = np.array([50.] * 3 + [35.] * 3 + [10.] * 3)
                for step in range(steps):
                    adapter.step_index = step
                    obs, history, privileged, _ = env.get_observations()
                    obs, history, privileged = obs.clone(), history.clone(), privileged.clone()
                    action, film_records = recorded_inference(actor, obs, history)
                    if reject:
                        env.set_impedance_force_estimates(actor.last_prediction.detach())
                    # Prediction recording is deterministic and never used as privileged
                    # policy input. No posterior sampling or training forward occurs.
                    predictions = {}
                    if hasattr(actor, "decode_context"):
                        context = actor.decode_context(actor.context_encoder.forward_inf(history))
                        predictions = {"prediction_" + k: v.detach().clone() for k, v in context.items() if torch.is_tensor(v)}
                    elif hasattr(actor, "last_prediction"):
                        predictions["prediction_explicit"] = actor.last_prediction.detach().clone()
                        predictions["prediction_z"] = actor.adaptation_encoder_module(history).detach().clone()
                    elif hasattr(actor, "encode_context"):
                        _, _, z = actor.encode_context(history, sample=False)
                        predictions["prediction_z"] = z.clone()
                        predictions["prediction_explicit"] = actor.adaptation_decoder_module(z).clone()
                        if hasattr(actor, "privileged_decoder_module"):
                            predictions["prediction_privileged"] = actor.privileged_decoder_module(z).clone()
                    if privileged_decoder is not None:
                        predictions["prediction_privileged"] = privileged_decoder(torch.cat((context["z"], context["explicit_condition"]), -1)).clone()
                    current = {"obs_t": obs, "history_t": history, "privileged_t": privileged,
                               "action_raw": action.clone(), "joint_position_t": s.dof_pos.clone(),
                               "joint_velocity_t": s.dof_vel.clone(), "base_position_t": s.base_pos.clone(),
                               "base_quaternion_t": s.base_quat.clone(),
                               "base_velocity_world_t": s._base_world_lin_vel.clone(),
                               "base_angular_velocity_world_t": s._base_world_ang_vel.clone(),
                               "ee_position_t_world": s.ee_pos.clone(), "ee_velocity_t_world": s._ee_vel.clone(),
                               "ee_reference_t_world": env.curr_ee_goal_cart_world.clone(),
                               "commands_t": env.commands.clone()}
                    grf_nominal = grf_conditioning_before_step(env, training, actor, action) if hasattr(actor, "predict_grf") else None
                    adapter.loads(step * env.dt)
                    adapter.substeps.clear()
                    env.step(action)
                    failures = adapter.failures()
                    stacks = {key: torch.stack([r[key] for r in adapter.substeps]) for key in adapter.substeps[0]}
                    data = {**current, **predictions, **film_records,
                            **{key + "_interval_mean": value.float().mean(0) for key, value in stacks.items() if value.dtype != torch.bool},
                            "action_clipped": action.clamp(-cfg.normalization.clip_actions, cfg.normalization.clip_actions),
                            "commands_effective": env.commands.clone(), "ee_position_world": s.ee_pos.clone(),
                            "ee_velocity_world": s._ee_vel.clone(), "ee_reference_world": env.curr_ee_goal_cart_world.clone(),
                            "base_position_world": s.base_pos.clone(), "base_quaternion_xyzw": s.base_quat.clone(),
                            "base_velocity_body": s.base_lin_vel.clone(), "base_angular_velocity_body": s.base_ang_vel.clone(),
                            "base_velocity_world": s._base_world_lin_vel.clone(), "base_angular_velocity_world": s._base_world_ang_vel.clone(),
                            "joint_position": s.dof_pos.clone(), "joint_velocity": s.dof_vel.clone(),
                            "ee_force_world": s._external_force_world[:, s._gripper_index].clone(),
                            "base_force_world": s._external_force_world[:, s._base_link_index].clone(),
                            "base_moment_world": s._external_torque_world[:, s._base_link_index].clone(), "feet_position_world": s.feet_pos.clone(),
                            "contacts": s.foot_contacts.clone(), "grf_filtered_world": s._grfs_buf.reshape(env.num_envs, -1, 3).clone(),
                            "obs_next": env.obs_buf.clone(), "history_next": env.obs_history.clone()}
                    if predictions and hasattr(actor, "predict_grf"):
                        # Legacy/non-BARD runners condition on the final executed
                        # substep, including delay; BARD uses the pre-step nominal.
                        grf_input = stacks["executed_torque"][-1] if grf_nominal is None else grf_nominal
                        data["prediction_grf"] = actor.predict_grf(context, grf_input).clone()
                        data["prediction_grf_torque_input_nm"] = grf_input.clone()
                    body_com = s._robot.data.body_com_pos_w
                    mass = masses
                    data["robot_com_world"] = (body_com * mass[..., None]).sum(1) / mass.sum(1, keepdim=True)
                    data["support_moment_world"] = stacks["support_moment_world"].mean(0)
                    for key in ("_motor_strength", "_kp_scale", "_kd_scale", "_added_base_mass", "_base_com_bias", "_added_gripper_mass", "_friction_values", "_joint_armature", "_joint_friction", "_joint_damping", "_joint_stiffness", "feedback_tau_weight", "feedforward_tau_weight"):
                        if hasattr(s, key):
                            data["mechanics_" + key.lstrip("_")] = getattr(s, key).clone()
                    for key in ("position_violation", "velocity_violation", "saturation"):
                        data[key + "_substep_count"] = stacks[key].sum(0)
                    data["torque_request_abs_peak"] = stacks["torque_request_motor_scaled"].abs().amax(0)
                    data["joint_velocity_abs_peak"] = stacks["joint_velocity"].abs().amax(0)
                    data["joint_position_min"] = stacks["joint_position"].amin(0)
                    data["joint_position_max"] = stacks["joint_position"].amax(0)
                    data["ee_reference_velocity_world"] = (data["ee_reference_world"] - current["ee_reference_t_world"]) / env.dt
                    if experiment is not None:
                        experiment.augment_data(adapter, stacks, data)
                    for slot, trial in enumerate(ids):
                        if not alive[slot]:
                            continue
                        row = {key: numpy_value(value[slot]) for key, value in data.items()}
                        delay = int(env.action_delay[slot]) if cfg.domain_rand.randomize_ctrl_delay else 0
                        command, _, _, phase = scenarios[slot].at(step * env.dt)
                        finite = all(np.isfinite(v).all() for v in row.values())
                        if not finite:
                            failures[slot] = failures[slot] or "nonfinite_record"
                        failure = bool(failures[slot])
                        actual_load = np.concatenate([row[k] for k in ("ee_force_world", "base_force_world", "base_moment_world")])
                        peak_loads[slot] = np.maximum(peak_loads[slot], np.abs(actual_load))
                        row["applied_load_exceeds_training_bounds"] = np.abs(actual_load) > training_load_bounds
                        row.update(trial_id=np.int64(trial), environment_id=np.int64(slot), step=np.int64(step),
                                   source_action_id=np.int64(step - delay), time_seconds=np.float64(step * env.dt),
                                   successor_time_seconds=np.float64((step + 1) * env.dt),
                                   dt=np.float64(env.dt), phase=np.int64(phase), command_nominal=command,
                                   valid=np.bool_(finite), physical_failure=np.bool_(failure),
                                   alive_for_metrics=np.bool_(finite and not failure),
                                   failure_reason_code=np.int64(sum(bit for name, bit in metadata["failure_reason_bits"].items()
                                                                   if name in failures[slot].split("+"))),
                                   horizon_complete=np.bool_(step == steps - 1), artificial_reset=np.bool_(False),
                                   rated_torque_limits=numpy_value(s._torque_limits),
                                   actual_clipping_limits=numpy_value(1.1 * s._torque_limits),
                                   joint_position_limits=numpy_value(s.dof_pos_limits_hard if hasattr(s, "dof_pos_limits_hard") else s.dof_pos_limits),
                                   joint_velocity_limits=numpy_value(s.dof_vel_limits),
                                   nominal_kp=numpy_value(s._p_gains), nominal_kd=numpy_value(s._d_gains),
                                   environment_origin_world=numpy_value(s._env_origins[slot]))
                        if experiment is not None:
                            experiment.augment_row(row, scenarios[slot], step * env.dt, (step + 1) * env.dt)
                            # Exp2 scores changing commands at the successor timestamp;
                            # commands_t and interval-start references remain untouched.
                            command, phase = row["command_nominal"], int(row["task_stage_successor"])
                        # Physical terminal transitions are retained, but not counted as
                        # alive tolerance passes or finite pre-failure error statistics.
                        metrics[slot].add((step + 1) * env.dt, row["ee_position_world"] - row["ee_reference_world"],
                                          row["base_velocity_body"][:2] - command[:2], row["base_angular_velocity_body"][2] - command[2], phase,
                                          valid=finite and not failure,
                                          violations=tuple(int(row[k + "_substep_count"].sum()) for k in ("position_violation", "velocity_violation", "saturation")),
                                          denominators=(cfg.control.decimation * s.dof_pos.shape[1],) * 3)
                        writer.append(row)
                        # Count each trial once, at failure or horizon, not per robot step.
                        if failure or step == steps - 1:
                            progress.update(1)
                        if failure:
                            alive[slot], reasons[slot] = False, failures[slot]
                    if not any(alive):
                        break
                for metric, reason, peaks in zip(metrics, reasons, peak_loads):
                    result = metric.finish(reason)
                    result["actual_load_abs_peak"] = peaks.tolist()
                    result["actual_exceeded_training_bounds"] = (peaks > training_load_bounds).tolist()
                    result["loaded_event_observed"] = bool(peaks.any())
                    trials.append(result)
                save_json("trials.json", trials)
        metadata["weights_unchanged"] = before == model_hash(actor)
        metadata["prediction_fields_available"] = [k for k in writer.schema if k.startswith("prediction_")]
        metadata["film_fields_available"] = [k for k in writer.schema if k.startswith("film_")]
        if not metadata["film_fields_available"]:
            metadata["unavailable"].append("FiLM features/modulation (model does not use FiLM conditioning)")
        for key in ("prediction_z", "prediction_explicit", "prediction_grf", "prediction_privileged"):
            if key not in writer.schema:
                metadata["unavailable"].append(key + " (not exposed by this model/checkpoint)")
        if not metadata["weights_unchanged"]:
            raise RuntimeError("Frozen evaluation changed policy parameters/buffers")
        save_json("metadata.json", metadata)
        save_json("seed_summary.json", (util.aggregate_seed(trials) if experiment is None else experiment.aggregate_seed(trials)))
        scalar_keys = [key for key, value in trials[0].items() if not isinstance(value, (list, dict))]
        with (args.output_dir / "trials.csv").open("w") as stream:
            table = csv.DictWriter(stream, fieldnames=scalar_keys, extrasaction="ignore")
            table.writeheader()
            table.writerows(trials)
        writer.close()
        (args.output_dir / "COMPLETE").write_text(f"{len(trials)} trials; frozen weights verified\n")
    except Exception:
        # Kit can terminate the process during close(); report before shutdown.
        import traceback
        report = traceback.format_exc()
        print(report, file=sys.stderr, flush=True)
        (args.output_dir / "ERROR.txt").write_text(report)
        raise
    finally:
        if progress is not None:
            progress.close()
        writer.close()
        if env is not None:
            env.simulator._app_launcher.app.close()


if __name__ == "__main__":
    if os.environ.get("B1Z1_EXP1_WORKER") == "1":
        main()
    else:
        # Kit may exit(0) during shutdown even with a pending Python exception.
        # A tiny parent verifies completion so failed evaluations cannot look valid.
        args = parser().parse_args()
        if args.output_dir.exists() and any(args.output_dir.iterdir()):
            raise SystemExit("Use an empty output directory")
        result = subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()), *sys.argv[1:]],
                                env={**os.environ, "B1Z1_EXP1_WORKER": "1"})
        raise SystemExit(result.returncode or (0 if (args.output_dir / "COMPLETE").exists() else 1))
