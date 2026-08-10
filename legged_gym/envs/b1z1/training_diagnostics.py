"""Shared controls for optional B1Z1 training diagnostics."""


CORE_EPISODE_REWARDS = {
    "tracking_lin_vel_force_world",
    "tracking_ang_vel",
}


def additional_diagnostics_enabled(env):
    """Default to legacy full logging when no runner has supplied a flag."""
    return bool(getattr(env, "enable_additional_diagnostics", True))


def should_log_episode_reward(env, reward_name):
    """Retain reward signals consumed by training curricula when details are off."""
    return additional_diagnostics_enabled(env) or reward_name in CORE_EPISODE_REWARDS
