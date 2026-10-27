"""Pure NumPy task profiles and bounded online metrics for Experiment 2."""
import copy
import importlib.util
import math
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location("exp2_common", Path(__file__).with_name("b1z1_experiment.py"))
common = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(common)

STAGES = ("stationary_unloaded", "start_ramp", "moving_unloaded", "load25_ee_motion",
          "load45", "load65_moving", "loaded_stop_ramp", "loaded_stationary", "released")
ATTRIBUTION = {"A1": "b1z1_pact_ab6_position_film", "A3": "b1z1_pact_ab8_coupled_film",
               "A6": "b1z1_pact_ab9_representation"}


def smooth(x):
    """Quintic interpolation: zero first/second derivatives at both endpoints."""
    x = np.clip(x, 0., 1.)
    return x**3 * (10. + x * (-15. + 6.*x))


def sphere_to_cart(sphere):
    r, elevation, azimuth = sphere
    return r * np.array([np.cos(elevation)*np.cos(azimuth), np.cos(elevation)*np.sin(azimuth), np.sin(elevation)])


def cart_to_sphere(cart):
    return np.array([np.linalg.norm(cart), np.arctan2(cart[2], np.linalg.norm(cart[:2])), np.arctan2(cart[1], cart[0])])


def selected_batches(episodes, num_envs, trial_ids=None):
    if episodes < 1 or num_envs < 1:
        raise ValueError("episodes and num-envs must be positive")
    ids = list(range(episodes)) if trial_ids is None else [int(x) for x in trial_ids.split(",")]
    if len(ids) != episodes or len(set(ids)) != len(ids) or min(ids) < 0:
        raise ValueError("trial-ids must contain exactly episodes unique nonnegative IDs")
    return [ids[i:i+num_envs] for i in range(0, len(ids), num_envs)]


class Scenario:
    def __init__(self, seed, trial, profile, loading, config):
        self.config = copy.deepcopy(config)
        self.profile, self.loading, self.heading = profile, loading, 0.
        c, p = self.config, self.config["profiles"][profile]
        self.start_sphere = np.array([p["radius_start"], p["elevation_rad"], c["azimuth_rad"]])
        self.end_cart = sphere_to_cart(self.start_sphere) + np.array(p.get("cartesian_offset_m", [0., 0., 0.]))
        motion = dict(profile=profile, parameters={k: v for k, v in p.items() if k != "force_direction_task"},
                      elevation=p["elevation_rad"], azimuth=c["azimuth_rad"],
                      start=c["move_start_s"], ramp=c["velocity_ramp_s"], trajectory=c["load_times_s"][:2], stop=c["stop_s"])
        if c.get("heading_command", False):
            motion["heading_control"] = dict(reference="initial_heading", yaw_rate_bounds=c["velocity_ranges"][2])
        pair = dict(seed=seed, trial_id=trial, motion=motion, overrides=c["overrides"],
                    joint_jitter=c["initial_joint_jitter_rad"], yaw=c["initial_yaw_range_rad"],
                    branch_weights=c["branch_weight_overrides"], randomize_branch_weights=c["randomize_branch_weights"])
        self.metadata = dict(seed=seed, trial_id=trial, profile=profile, loading=loading,
                             condition=profile, load_range=loading, motion_mode="loco-manipulation",
                             config=c, pair_id=common.digest(pair), motion_hash=common.digest(motion),
                             profile_hash=common.digest(dict(motion=motion, load_times=c["load_times_s"], loads=c["loads_n"],
                                                            force_direction_task=p["force_direction_task"])),
                             initial_seed=common.seed_for(seed, trial, "initial"),
                             base_command_events=[dict(time=c["move_start_s"], kind="start", ramp_seconds=c["velocity_ramp_s"]),
                                                  dict(time=c["stop_s"], kind="stop", ramp_seconds=c["velocity_ramp_s"])])
        self.set_heading(0.)

    def set_heading(self, yaw):
        # Set from the measured RESET pose, never from moving torso yaw.
        self.heading = float(yaw)
        self.metadata["initial_task_heading_rad"] = self.heading
        events = []
        for i, time in enumerate(self.config["load_times_s"]):
            load = self.at(time)[2]
            previous = np.array(events[-1]["load"]) if events else np.zeros(9)
            events.append(dict(time=time, kind="removal" if i == 3 else "onset" if i == 0 else "change",
                               load=load.tolist(), changed=bool(np.any(load != previous)),
                               exceeds_training_bounds=(np.abs(load) > np.array([50.]*3+[35.]*3+[10.]*3)).tolist()))
        self.metadata["events"] = events
        payload = {k: v for k, v in self.metadata.items() if k != "scenario_hash"}
        self.metadata["scenario_hash"] = common.digest(payload)

    def stage(self, time):
        c = self.config
        boundaries = [c["move_start_s"], c["move_start_s"]+c["velocity_ramp_s"],
                      *c["load_times_s"][:3], c["stop_s"], c["stop_s"]+c["velocity_ramp_s"], c["load_times_s"][3]]
        return int(np.searchsorted(boundaries, time, side="right"))

    def at(self, time):
        c, p = self.config, self.config["profiles"][self.profile]
        command = np.array([p["velocity_m_s"] * smooth((time-c["move_start_s"])/c["velocity_ramp_s"])
                            * (1.-smooth((time-c["stop_s"])/c["velocity_ramp_s"])), 0., 0.])
        blend = smooth((time-c["load_times_s"][0])/(c["load_times_s"][2]-c["load_times_s"][0]))
        sphere = self.start_sphere.copy()
        if "radius_end" in p:
            sphere[0] += blend * (p["radius_end"]-sphere[0])
        else:
            start = sphere_to_cart(sphere)
            sphere = cart_to_sphere(start + blend * (self.end_cart-start))
        index = int(np.searchsorted(c["load_times_s"], time, side="right"))-1
        magnitude = c["loads_n"][index] if index >= 0 and self.loading == "loaded" else 0.
        direction = np.array(p["force_direction_task"], dtype=float)
        cos, sin = math.cos(self.heading), math.sin(self.heading)
        world = np.array([cos*direction[0]-sin*direction[1], sin*direction[0]+cos*direction[1], direction[2]])
        return command, sphere, np.r_[magnitude*world, np.zeros(6)], self.stage(time)


def quaternion_yaw(quaternion, xp=np):
    """Heading of body x in world XY; simulator quaternion ordering is xyzw."""
    x, y, z, w = (quaternion[..., i] for i in range(4))
    return xp.arctan2(2*(w*z+x*y), 1.-2*(y*y+z*z))


def heading_yaw_rate(yaw, initial_heading, config, xp=np):
    """Native PACT/UniFP heading mapping: radians -> bounded rad/s, no deadband."""
    error = ((initial_heading - yaw + math.pi) % (2*math.pi) - math.pi) / math.pi
    low, high = config["velocity_ranges"][2]
    return xp.where(error >= 0., error * high, -error * low)


def validate(config, horizon):
    c = config
    if c.get("heading_command", False):
        low, high = c["velocity_ranges"][2]
        if not np.isfinite([low, high]).all() or not low < 0 < high:
            raise ValueError("Heading control requires finite yaw-rate bounds spanning zero")
    times = [0., c["move_start_s"], c["move_start_s"]+c["velocity_ramp_s"], *c["load_times_s"][:3],
             c["stop_s"], c["stop_s"]+c["velocity_ramp_s"], c["load_times_s"][3], c["finish_s"]]
    if len(c["load_times_s"]) != 4 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Expected ordered start/ramp/load25/load45/load65/stop/ramp/release/finish times")
    if len(c["loads_n"]) != 4 or not np.isfinite(c["loads_n"]).all() or min(c["loads_n"]) < 0 or c["loads_n"][-1] != 0:
        raise ValueError("Four nonnegative loads required, ending in zero")
    if not np.isfinite(horizon) or horizon <= 0 or not 0 <= c["reject_beta"] <= 1:
        raise ValueError("Invalid horizon or rejection strength")
    for key in ("ee_tolerance_m", "velocity_tolerance_m_s", "yaw_tolerance_rad_s", "recovery_window_seconds",
                "recovery_dwell_seconds", "chunk_rows"):
        if not np.isfinite(c[key]) or c[key] <= 0:
            raise ValueError(f"Invalid {key}")
    if c["recovery_dwell_seconds"] > c["recovery_window_seconds"] or any(
            b-a < c["recovery_window_seconds"] for a, b in zip(c["load_times_s"], [*c["load_times_s"][1:], c["finish_s"]])):
        raise ValueError("Load response windows must be complete in the configured sequence")
    bounds = np.asarray(c["workspace_sphere_bounds"])
    if bounds.shape != (3, 2) or not np.isfinite(bounds).all() or np.any(bounds[:, 0] >= bounds[:, 1]):
        raise ValueError("Invalid common workspace bounds")
    for profile, p in c["profiles"].items():
        if not c["velocity_ranges"][0][0] <= p["velocity_m_s"] <= c["velocity_ranges"][0][1]:
            raise ValueError("Profile velocity exceeds declared evaluation command range")
        if not np.isclose(np.linalg.norm(p["force_direction_task"]), 1.):
            raise ValueError("Force direction must be a unit vector")
        s = Scenario(0, 0, profile, "loaded", c)
        for time in np.linspace(0, c["finish_s"], 257):
            cmd, sphere, load, _ = s.at(time)
            if not np.isfinite(np.r_[cmd, sphere, load]).all() or np.any(sphere < bounds[:, 0]) or np.any(sphere > bounds[:, 1]):
                raise ValueError(f"Invalid {profile} waypoint; no method-specific clipping is permitted")


def torque_statistics(pd, ff, executed, xp=np):
    """Arrays shaped (physics_substeps, ..., joints); compute BEFORE averaging."""
    out = {}
    for name, value in (("pd_weighted_motor", pd), ("ff_weighted_motor", ff), ("executed_torque", executed)):
        out[name+"_mean_square"] = (value*value).mean(axis=0)
        out[name+"_abs_peak"] = xp.amax(abs(value), axis=0)
    out["opposing_channel_cancellation_mean"] = ((abs(pd)+abs(ff)-abs(pd+ff))/2).mean(axis=0)
    return out


def point_force_moment(points, forces, com):
    """World N*m about simultaneous whole-body CoM, not torso origin."""
    return np.cross(points-com[..., None, :], forces).sum(axis=-2)


class TrialMetrics(common.TrialMetrics):
    def __init__(self, horizon, dt, scenario, config):
        super().__init__(horizon, dt, scenario, config)
        self.phases = {str(i): dict(count=0, ee_sum=0., velocity_sum=0.) for i in range(len(STAGES))}
        self.stage_passes = np.zeros(len(STAGES), dtype=int)
        self.stage_denominators = np.bincount([scenario.stage((i+1)*dt) for i in range(horizon)], minlength=len(STAGES))

    def add(self, time, ee_error, velocity_error, yaw_error, phase, valid=True, **kwargs):
        before = self.passed
        super().add(time, ee_error, velocity_error, yaw_error, phase, valid=valid, **kwargs)
        self.stage_passes[phase] += self.passed-before

    def finish(self, failure_reason):
        result = super().finish(failure_reason)
        for event, descriptor in zip(result["event_metrics"], self.scenario.metadata["events"]):
            event["kind"] = descriptor["kind"]
        result["stage_statistics"] = {
            name: dict(valid_steps=self.phases[str(i)]["count"], prescribed_steps=int(self.stage_denominators[i]),
                       tracking_pass_steps=int(self.stage_passes[i]),
                       S_track=float(self.stage_passes[i]/self.stage_denominators[i]) if self.stage_denominators[i] else None,
                       ee_error_m=self.phases[str(i)]["ee_sum"]/self.phases[str(i)]["count"] if self.phases[str(i)]["count"] else None,
                       planar_velocity_error_m_s=self.phases[str(i)]["velocity_sum"]/self.phases[str(i)]["count"] if self.phases[str(i)]["count"] else None)
            for i, name in enumerate(STAGES)}
        return result


def aggregate_seed(trials):
    # Shared helper pools physical-limit counts/exposure within each seed;
    # tracking keeps equal trial weights. Only stage summaries are Exp2-specific.
    summary = common.aggregate_seed(trials)
    for key, result in summary.items():
        rows = [t for t in trials if "/".join(str(t[k]) for k in ("seed", "condition", "load_range", "motion_mode")) == key]
        result.pop("phases")  # Exp1's three phase labels are not Exp2's task stages.
        result["stages"] = {}
        for stage in STAGES:
            result["stages"][stage] = {}
            for metric in ("ee_error_m", "planar_velocity_error_m_s", "S_track"):
                values = [t["stage_statistics"][stage][metric] for t in rows if t["stage_statistics"][stage][metric] is not None]
                result["stages"][stage][metric] = float(np.mean(values)) if values else None
                result["stages"][stage][metric+"_trial_count"] = len(values)
    return summary
