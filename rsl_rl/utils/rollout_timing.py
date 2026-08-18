"""Low-overhead rollout phase timing shared by B1Z1 training runners."""

from __future__ import annotations

import os
import time

import torch


ROLLOUT_TIMING_TAGS = {
    "policy": "Perf/Rollout/policy_action_inference_ms",
    "force_events": "Perf/Rollout/force_event_updates_ms",
    "simulator": "Perf/Rollout/simulator_step_ms",
    "post_physics": "Perf/Rollout/post_physics_refresh_callback_ms",
    "rewards": "Perf/Rollout/rewards_ms",
    "resets": "Perf/Rollout/resets_ms",
    "observations": "Perf/Rollout/observation_construction_ms",
    "transition_storage": "Perf/Rollout/transition_storage_copies_ms",
    "total": "Perf/Rollout/total_collection_ms",
}


class RolloutPhaseTimer:
    """Accumulate CUDA-event timings and synchronize exactly once per rollout."""

    def __init__(self, device):
        self.device = torch.device(device)
        self.use_cuda = self.device.type == "cuda" and torch.cuda.is_available()
        self._samples = {name: [] for name in ROLLOUT_TIMING_TAGS if name != "total"}
        self._total_start = time.perf_counter()

    def start(self, phase):
        if self.use_cuda:
            event = torch.cuda.Event(enable_timing=True)
            event.record(torch.cuda.current_stream(self.device))
            return event
        return time.perf_counter()

    def stop(self, phase, start):
        if self.use_cuda:
            end = torch.cuda.Event(enable_timing=True)
            end.record(torch.cuda.current_stream(self.device))
            self._samples[phase].append((start, end))
        else:
            self._samples[phase].append((time.perf_counter() - start) * 1000.0)

    def finish(self):
        if self.use_cuda:
            # This is the only synchronization introduced by rollout timing.
            torch.cuda.synchronize(self.device)
        metrics = {}
        for phase, samples in self._samples.items():
            if self.use_cuda:
                elapsed_ms = sum(start.elapsed_time(end) for start, end in samples)
            else:
                elapsed_ms = sum(samples)
            metrics[ROLLOUT_TIMING_TAGS[phase]] = elapsed_ms
        metrics[ROLLOUT_TIMING_TAGS["total"]] = (
            time.perf_counter() - self._total_start
        ) * 1000.0
        return metrics


def startup_metadata(env, runner_device, observations, policy):
    """Resolve consistent device and robot-asset metadata across simulator backends."""
    simulator = env.simulator
    runner_device = torch.device(runner_device)
    is_isaac = "isaacgym" in simulator.__class__.__module__
    asset_template = (
        getattr(env.cfg.asset, "isaacgym_file", env.cfg.asset.file)
        if is_isaac else env.cfg.asset.file
    )
    urdf_path = os.path.abspath(
        asset_template.replace(
            "{LEGGED_GYM_ROOT_DIR}",
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        )
    )
    body_count = int(getattr(simulator, "_num_bodies", getattr(getattr(simulator, "_robot", None), "n_links", 0)))
    dof_count = int(getattr(simulator, "_num_dof", getattr(getattr(simulator, "_robot", None), "n_dofs", 0)))
    if is_isaac and getattr(simulator, "_envs", None):
        shape_count = len(simulator._gym.get_actor_rigid_shape_properties(
            simulator._envs[0], simulator._actor_handles[0]
        ))
    else:
        robot = getattr(simulator, "_robot", None)
        shape_count = int(getattr(robot, "n_geoms", 0))
        if shape_count == 0 and robot is not None:
            shape_count = sum(len(getattr(link, "geoms", ())) for link in robot.links)
    obs_devices = sorted({str(value.device) for value in observations if value is not None})
    policy_device = next(policy.parameters()).device
    gpu_name = (
        torch.cuda.get_device_name(runner_device)
        if runner_device.type == "cuda" and torch.cuda.is_available() else "CPU"
    )
    return {
        "env_device": str(env.device),
        "runner_device": str(runner_device),
        "observation_devices": ",".join(obs_devices),
        "policy_device": str(policy_device),
        "gpu_name": gpu_name,
        "urdf_path": urdf_path,
        "asset_body_count": body_count,
        "asset_dof_count": dof_count,
        "asset_rigid_shape_count": shape_count,
    }


def log_startup_metadata(metadata, writer=None):
    """Print startup placement once and mirror it into TensorBoard text."""
    message = "B1Z1 startup diagnostics:\n" + "\n".join(
        f"  {name}: {value}" for name, value in metadata.items()
    )
    print(message)
    if writer is not None:
        writer.add_text(
            "Startup/B1Z1",
            "  \n".join(f"**{name}**: {value}" for name, value in metadata.items()),
            0,
        )
