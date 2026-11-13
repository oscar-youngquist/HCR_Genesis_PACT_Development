"""FACET reward configuration and pure horizon reductions (no dt scaling here)."""
from dataclasses import dataclass, field
import math

import torch


COMPONENTS = ("base_position", "base_velocity", "yaw", "yaw_rate", "ee_position", "ee_velocity")
LEGACY_TRACKING = ("tracking_lin_vel_force_world", "tracking_ang_vel", "tracking_ee_force_world")
CONFLICTING_SHAPING = ("no_physical_progress", "stand_still", "stand_still_contact",
    "ref_dof_leg", "feet_contact_number", "walking_ref_dof", "walking_ref_swing_dof",
    "arm_progress_before_torso", "early_torso_tilt")
FF_ONLY = ('tracking_arm_feedforward', 'torque_cancellation',
    'leg_feedforward_action_rate', 'leg_feedforward_action_smoothness',
    'arm_feedforward_action_rate', 'arm_feedforward_action_smoothness',
    'leg_feedforward_torques', 'arm_feedforward_torques')


@dataclass
class FACETTrackingCfg:
    enabled: bool = True  # False retains Stage 2's exact nominal rewards and diagnostic-only reference.
    horizons: tuple = (8, 16, 32)  # Ages in control intervals, NOT future prediction offsets.
    weights: dict = field(default_factory=lambda: dict(zip(COMPONENTS, (.75, 1.5, .75, 1., 1., 1.))))
    bandwidths: dict = field(default_factory=lambda: dict(zip(COMPONENTS, (.25, .25, .25, .25, .10, .25))))
    # Fraction of the maximum EE-position reward, not a distance threshold.
    force_curriculum_reward_fraction: float = .75
    # Legacy scales remain intact in rewards.scales for diagnostic units and the
    # disabled switch. These terms are excluded when registering the objective.
    excluded_rewards: tuple = LEGACY_TRACKING + CONFLICTING_SHAPING + FF_ONLY

    def validate(self):
        if not 0 < self.force_curriculum_reward_fraction < 1:
            raise ValueError("FACET force curriculum reward fraction must lie in (0, 1)")
        if not set(LEGACY_TRACKING + CONFLICTING_SHAPING).issubset(self.excluded_rewards):
            raise ValueError("FACET tracking must exclude legacy tracking and competing nominal shaping")
        for name, values in (("weights", self.weights), ("bandwidths", self.bandwidths)):
            if set(values) != set(COMPONENTS):
                raise ValueError(f"FACET {name} must specify exactly {COMPONENTS}")
            if any(not math.isfinite(v) or v < 0 or (name == "bandwidths" and v == 0) for v in values.values()):
                raise ValueError(f"Invalid FACET {name}")


@torch.no_grad()
def tracking_values(targets, measured, valid, bandwidths):
    """Return N x 6 rewards and N x H x 6 squared SI errors.

    Measured state is the refreshed successor; all targets end at this same
    timestamp. Average each nonlinear reward, never the target. Invalid starts
    contribute neither reward nor denominator; an empty bank contributes zero.
    """
    difference = targets-measured[:, None]
    yaw = torch.atan2(difference[..., 4].sin(), difference[..., 4].cos())
    e2 = torch.stack((difference[..., :2].square().sum(-1), difference[..., 2:4].square().sum(-1),
        yaw.square(), difference[..., 5].square(), difference[..., 6:9].square().sum(-1),
        difference[..., 9:12].square().sum(-1)), -1)
    rewards = torch.exp(-e2/bandwidths)
    # Released FACET velocity shaping applies to base XY and yaw rate only.
    penalty = e2.new_tensor((0., .25, 0., .25, 0., 0.))
    rewards = rewards-penalty*e2
    count = valid.sum(1).clamp_min(1)
    mean = torch.where(valid[..., None], rewards, 0.).sum(1)/count[:, None]
    return mean, e2
