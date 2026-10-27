"""Frozen task-inspired loco-manipulation evaluation; reuses Experiment 1 execution."""
import argparse
import copy
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


exp1 = load_module("exp2_execution", HERE / "eval_b1z1_exp1.py")
tasks = load_module("exp2_profiles", HERE.parent / "utils/b1z1_experiment2.py")

# EDIT HERE. Independent deep copy preserves Exp1 settings and registered configs.
# Initialization/noise/physical randomization and immediate failure thresholds match
# Exp1. Units are SI; native spherical coordinates are radius/elevation/azimuth.
EVAL_CONFIG = {
    **copy.deepcopy(exp1.EVAL_CONFIG),
    "move_start_s": 3., "velocity_ramp_s": .5,
    "load_times_s": [7., 11., 15., 23.], "loads_n": [25., 45., 65., 0.],
    "stop_s": 19., "finish_s": 28.,
    "elevation_rad": .10, "azimuth_rad": 0.,
    "velocity_ranges": [[-.4, .4], [0., 0.], [0., 0.]],
    "ee_tolerance_m": .10, "velocity_tolerance_m_s": .20, "yaw_tolerance_rad_s": .30,
    "recovery_window_seconds": 2., "recovery_dwell_seconds": .5,
    # Common validation envelope, NOT a clipping operation or proof of reachability.
    "workspace_sphere_bounds": [[.3, .75], [-.5, .8], [-.8, .8]],
    "profiles": {
        "pushing": dict(velocity_m_s=.4, radius_start=.50, radius_end=.60, force_direction_task=[-1., 0., 0.]),
        "pulling": dict(velocity_m_s=-.4, radius_start=.60, radius_end=.45, force_direction_task=[1., 0., 0.]),
        "carrying": dict(velocity_m_s=.4, radius_start=.50, cartesian_offset_m=[.08, .05, 0.], force_direction_task=[0., 0., -1.]),
    },
}
# These Exp1-only trajectory keys are deliberately not used by the task profiles.
for _key in ("event_times", "ee_center_sphere", "ee_amplitude_sphere", "ee_period"):
    EVAL_CONFIG.pop(_key, None)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", default="b1z1_pact")
    p.add_argument("--run", required=True)
    p.add_argument("--ckpt", default="-1")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-envs", type=int, default=10)
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--profile", choices=("pushing", "pulling", "carrying"), default="pushing")
    p.add_argument("--loading", choices=("loaded", "unloaded"), default="loaded")
    p.add_argument("--trial-ids", help="Comma-separated stable IDs; length must equal --episodes")
    p.add_argument("--horizon-seconds", type=float, default=28.)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--gpu", default="cuda:0")
    p.add_argument("--headless", action="store_true")
    return p


class Adapter(exp1.EvaluationAdapter):
    def reset(self, scenarios):
        super().reset(scenarios)
        import numpy as np
        for slot, scenario in enumerate(scenarios):
            x, y, z, w = exp1.numpy_value(self.sim.base_quat[slot])
            scenario.set_heading(np.arctan2(2*(w*z+x*y), 1.-2*(y*y+z*z)))

    def contact_update(self, *args, **kwargs):
        result = super().contact_update(*args, **kwargs)
        if self.pending is None:
            return result
        import torch
        s, row = self.sim, self.substeps[-1]
        body = s._robot.data
        mass = self.body_masses[..., None]
        com = (body.body_com_pos_w*mass).sum(1)/mass.sum(1)
        com_velocity = (body.body_com_vel_w[..., :3]*mass).sum(1)/mass.sum(1)
        forces = s._contact_sensors.data.net_forces_w[:, s._contact_indices]
        feet = body.body_link_pos_w[:, s.feet_indices]
        ee_force = s._external_force_world[:, s._gripper_index]
        # Net contact forces lack contact-point/free-moment information. Moments
        # approximate point forces at link origins; imposed EE loads act at body CoM.
        row.update(robot_com_world=com.clone(), robot_com_velocity_world=com_velocity.clone(),
                   body_link_positions_world=body.body_link_pos_w.clone(),
                   body_com_positions_world=body.body_com_pos_w.clone(),
                   body_com_velocities_world=body.body_com_vel_w.clone(),
                   body_contact_forces_raw_world=forces.clone(),
                   body_contacts=(forces.norm(dim=-1) > s._foot_contact_force_threshold).float(),
                   foot_positions_raw_world=feet.clone(),
                   ee_load_moment_world=torch.linalg.cross(row["ee_load_point_world"]-com, ee_force, dim=-1),
                   all_contact_moment_world=torch.linalg.cross(body.body_link_pos_w-com[:, None], forces, dim=-1).sum(1),
                   ee_force_applied_world=ee_force.clone())
        return result


class Experiment2:
    config = EVAL_CONFIG
    parser = staticmethod(parser)
    Adapter = Adapter
    TrialMetrics = tasks.TrialMetrics
    aggregate_seed = staticmethod(tasks.aggregate_seed)

    @staticmethod
    def validate(config, args):
        tasks.validate(config, args.horizon_seconds)
        # Only labels consumed by the shared progress bar/metadata, not Exp1 sampling.
        args.condition, args.load_range, args.motion_mode = args.profile, args.loading, "loco-manipulation"

    @staticmethod
    def batches(args):
        return tasks.selected_batches(args.episodes, args.num_envs, args.trial_ids)

    def scenario(self, args, trial):
        return tasks.Scenario(args.seed, trial, args.profile, args.loading, self.config)

    @staticmethod
    def metadata(env, args, config, state):
        import math
        # Never silently quantize task boundaries differently for different control dt.
        boundaries = [config["move_start_s"], config["move_start_s"]+config["velocity_ramp_s"],
                      *config["load_times_s"], config["stop_s"], config["stop_s"]+config["velocity_ramp_s"], config["finish_s"]]
        if any(not math.isclose(t/env.dt, round(t/env.dt), abs_tol=1e-6) for t in boundaries):
            raise ValueError("Experiment 2 task boundaries must align with the control timestep")
        s = env.simulator
        # Simulator index mappings may be lists or tensors; serialize native ints.
        foot_indices = {int(i) for i in s.feet_indices}
        return dict(experiment=2, task_stages=dict(enumerate(tasks.STAGES)), attribution_tasks=tasks.ATTRIBUTION,
                    checkpoint_training_seed=state.get("training_seed", state.get("seed")),
                    exp2_evaluator_sha256=exp1.util.file_hash(__file__),
                    exp2_utilities_sha256=exp1.util.file_hash(HERE.parent / "utils/b1z1_experiment2.py"),
                    shortened_sequence=args.horizon_seconds < config["finish_s"],
                    nonfoot_body_indices=[i for i in range(len(s._body_names)) if i not in foot_indices],
                    articulation_to_contact_sensor_indices=[int(i) for i in s._contact_indices],
                    configured_to_articulation_dof_indices=[int(i) for i in s._dof_indices],
                    experiment2_conventions={
                        "task_force_frame": "initial reset heading, world z up; constant for each trial",
                        "EE_trajectory": "native yaw-aligned workspace; position reference follows current base yaw/translation",
                        "velocity_command": "native body-frame forward m/s, lateral m/s, yaw rad/s",
                        "reference_timing": "commands_t/ee_reference_t_world consumed by actor; command_nominal/ee_reference_world at successor",
                        "task_stage": "task_stage_interval labels applied interval; task_stage_successor labels scored endpoint",
                        "body_contact_forces_raw_world_interval_mean": "N; articulation order resolved through sensor indices, includes non-foot contacts",
                        "body_com_quaternion_wxyz": "successor body inertial-frame rotations, wxyz; other simulator quaternions remain xyzw",
                        "body_com_velocities_world_interval_mean": "world linear m/s then angular rad/s, articulation order",
                        "robot_com_velocity_world_interval_mean": "mass-weighted body CoM linear velocity, m/s",
                        "support_moment_world": "mean of per-substep cross(foot link origin - whole CoM, raw foot force), N*m; point-force approximation",
                        "all_contact_moment_world_interval_mean": "same approximation using every articulation body's net sensor force; not contact wrenches",
                        "ee_load_moment_world_interval_mean": "per-substep cross(gripper CoM - whole CoM, actual applied EE force), N*m",
                        "torque_mean_square": "per-joint mean of substep squared torques, (N*m)^2, after branch/motor scaling",
                        "opposing_channel_cancellation_mean": "mean((abs(PD)+abs(FF)-abs(PD+FF))/2), N*m, before saturation",
                        "body_properties": "initial_states.jsonl: masses kg, body CoM inertias kg*m^2 (PhysX row-major), raw PhysX local CoM poses xyz+xyzw",
                        "unavailable": "individual contact-point wrenches, tangential contact moments; net body forces only"})

    @staticmethod
    def augment_data(adapter, stacks, data):
        import torch
        data.update(tasks.torque_statistics(stacks["pd_weighted_motor"], stacks["ff_weighted_motor"], stacks["executed_torque"], xp=torch))
        # Physical successor body frames permit offline momentum/FK calculations.
        body = adapter.sim._robot.data
        data["body_com_quaternion_wxyz"] = body.body_com_quat_w.clone()
        data["body_com_positions_world"] = body.body_com_pos_w.clone()
        data["body_com_velocities_world"] = body.body_com_vel_w.clone()
        data["robot_com_velocity_world"] = stacks["robot_com_velocity_world"][-1].clone()
        data["ee_moment_world"] = adapter.sim._external_torque_world[:, adapter.sim._gripper_index].clone()

    @staticmethod
    def augment_row(row, scenario, start, end):
        import numpy as np
        c0, s0, _, stage0 = scenario.at(start)
        c1, s1, _, stage1 = scenario.at(end)
        row.update(command_nominal_t=c0, command_nominal=c1, ee_reference_sphere_t=s0,
                   ee_reference_sphere=s1, task_stage_interval=np.int64(stage0),
                   task_stage_successor=np.int64(stage1),
                   load_event_index=np.int64(np.searchsorted(scenario.config["load_times_s"], start, side="right")-1),
                   base_command_event_index=np.int64(np.searchsorted([scenario.config["move_start_s"], scenario.config["stop_s"]], start, side="right")-1))


def main(argv=None):
    exp1.main(argv, experiment=Experiment2())


if __name__ == "__main__":
    if os.environ.get("B1Z1_EXP2_WORKER") == "1":
        main()
    else:
        args = parser().parse_args()
        Experiment2.validate(EVAL_CONFIG, args)
        Experiment2.batches(args)
        if args.output_dir.exists() and any(args.output_dir.iterdir()):
            raise SystemExit("Use an empty output directory")
        result = subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()), *sys.argv[1:]],
                                env={**os.environ, "B1Z1_EXP2_WORKER": "1"})
        # Isaac Kit shutdown can mask exceptions: require a flushed completion marker.
        raise SystemExit(result.returncode or (0 if (args.output_dir / "COMPLETE").exists() else 1))
